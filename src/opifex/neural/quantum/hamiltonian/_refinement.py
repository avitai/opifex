r"""QHNet self / pair interaction refinement layers for Fock-block features.

These two layers are the expressivity core a NequIP-style trunk lacks for
Hamiltonian prediction (Yu et al. 2023, "QHNet", arXiv:2306.04922; reference
``divelab/AIRS`` ``OpenDFT/QHBench/QH9/models/QHNet.py`` ``SelfNetLayer`` /
``PairNetLayer``). The trunk produces per-atom equivariant features; the **Fock
blocks** are rank-2 tensors that need *products* of those features:

* :class:`SelfInteractionLayer` forms the diagonal-block feature from a channel-wise
  **self** tensor product ``tp(W_l x, W_r x)`` of an atom's own feature -- the
  products ``D^{l_i} (x) D^{l_j}`` the on-site block ``H_ii`` transforms as.
* :class:`PairInteractionLayer` forms the off-diagonal-block feature from a
  channel-wise **pair** tensor product ``tp(x[src], x[dst])`` of the two endpoints,
  with the per-edge weights modulated by the radial embedding and the inner product
  of the endpoints (QHNet's ``fc_node_pair(edge_attr) * fc(s0)``) -- the genuine
  bilinear coupling the off-diagonal block ``H_ij`` needs.

Both reuse the opifex equivariant primitives
(:class:`~opifex.neural.equivariant.ChannelwiseTensorProduct` for the ``O(mul)``
``"uuu"`` coupling, :class:`~opifex.neural.equivariant.NormGate` nonlinearity,
:func:`~opifex.neural.equivariant.inner_product`,
:class:`~opifex.neural.equivariant.EquivariantLinear`) and accumulate residually
across the layer stack (QHNet's ``fii`` / ``fij`` running sums). They operate in
an **all-even** irrep space (the trunk's parities are relabelled to even at the
refinement boundary, matching QHNet's ``hidden_irrep_base``); they are therefore
SO(3)-equivariant, which is all the matrix head requires.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int  # noqa: TC002

from opifex.neural.equivariant import (
    ChannelwiseTensorProduct,
    EquivariantLinear,
    inner_product,
    Irreps,
    IrrepsArray,
    NormGate,
    rms_normalize,
)


def _scalar_part(features: IrrepsArray) -> Float[Array, "... num_scalars"]:
    """Concatenate the ``l = 0`` block chunks of ``features`` into a flat array."""
    leading = features.array.shape[:-1]
    scalar_chunks = [
        chunk.reshape(*leading, mul)
        for (mul, irrep), chunk in zip(features.irreps.blocks, features.chunks, strict=True)
        if irrep.l == 0
    ]
    return jnp.concatenate(scalar_chunks, axis=-1)


class SelfInteractionLayer(nnx.Module):
    r"""Refine a per-atom feature by a channel-wise self tensor product.

    Realises QHNet's ``SelfNetLayer``: two norm-gated linear projections of the
    input are coupled by a channel-wise (``"uuu"``) tensor product, a residual is
    added, and a final norm-gated linear projection produces the refined feature,
    which is accumulated onto the running diagonal-block feature.

    Args:
        irreps: The (all-even) per-atom feature layout; input, coupling and output
            all share it.
        rngs: Random number generators (keyword-only) seeding the weights.
    """

    def __init__(self, irreps: Irreps | str, *, rngs: nnx.Rngs) -> None:
        """Build the projections, the self tensor product and the norm gates."""
        super().__init__()
        self.irreps = Irreps(irreps)
        self.gate_left = NormGate(self.irreps, rngs=rngs)
        self.gate_right = NormGate(self.irreps, rngs=rngs)
        self.linear_left = EquivariantLinear(self.irreps, self.irreps, rngs=rngs)
        self.linear_right = EquivariantLinear(self.irreps, self.irreps, rngs=rngs)
        self.tensor_product = ChannelwiseTensorProduct(
            self.irreps, self.irreps, self.irreps, rngs=rngs
        )
        self.gate_out = NormGate(self.irreps, rngs=rngs)
        self.linear_out = EquivariantLinear(self.irreps, self.irreps, rngs=rngs)

    def __call__(
        self, features: IrrepsArray, accumulated: IrrepsArray | None = None
    ) -> IrrepsArray:
        """Return the refined per-atom feature, optionally added to ``accumulated``.

        Args:
            features: Per-atom feature ``(n_atoms, irreps.dim)``.
            accumulated: Running diagonal-block feature from earlier layers (added
                residually), or ``None`` for the first refinement layer.

        Returns:
            The refined feature with the same layout as ``features``.
        """
        # Bound the magnitude before the squaring self tensor product (an
        # unnormalised trunk's growing activations otherwise blow up).
        features = rms_normalize(features)
        left = self.linear_left(self.gate_left(features))
        right = self.linear_right(self.gate_right(features))
        refined = self.tensor_product(left, right)
        refined = IrrepsArray(self.irreps, refined.array + features.array)
        refined = self.linear_out(self.gate_out(refined))
        if accumulated is not None:
            refined = IrrepsArray(self.irreps, refined.array + accumulated.array)
        return refined


class PairInteractionLayer(nnx.Module):
    r"""Refine a per-edge feature by a channel-wise pair tensor product.

    Realises QHNet's ``PairNetLayer``: the two endpoint features are coupled by a
    channel-wise (``"uuu"``) tensor product whose per-edge weights are the
    elementwise product of a radial-embedding MLP and an MLP of the pair scalar
    features (the endpoint scalars and the inner product of the endpoints). A final
    norm-gated linear projection produces the refined per-edge feature, accumulated
    onto the running off-diagonal-block feature.

    Args:
        irreps: The (all-even) feature layout; node inputs, coupling and per-edge
            output all share it.
        edge_radial_dim: Width of the per-edge radial embedding.
        weight_hidden_dim: Hidden width of the per-edge weight MLPs.
        rngs: Random number generators (keyword-only) seeding the weights.
    """

    def __init__(
        self,
        irreps: Irreps | str,
        *,
        edge_radial_dim: int,
        weight_hidden_dim: int = 64,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the projections, the external-weight pair TP and the weight MLPs."""
        super().__init__()
        self.irreps = Irreps(irreps)
        self._num_scalars = sum(mul for mul, irrep in self.irreps.blocks if irrep.l == 0)
        self._num_vectors = sum(mul for mul, irrep in self.irreps.blocks if irrep.l > 0)
        self.linear_inner = EquivariantLinear(self.irreps, self.irreps, rngs=rngs)
        self.gate_pre = NormGate(self.irreps, rngs=rngs)
        self.linear_node = EquivariantLinear(self.irreps, self.irreps, rngs=rngs)
        self.tensor_product = ChannelwiseTensorProduct(
            self.irreps, self.irreps, self.irreps, internal_weights=False, rngs=rngs
        )
        weight_numel = self.tensor_product.weight_numel
        # QHNet pair scalar features: endpoint scalars (src + dst) + endpoint inner
        # product on the non-scalar irreps.
        scalar_dim = 2 * self._num_scalars + self._num_vectors
        self.radial_weight_hidden = nnx.Linear(edge_radial_dim, weight_hidden_dim, rngs=rngs)
        self.radial_weight_out = nnx.Linear(weight_hidden_dim, weight_numel, rngs=rngs)
        self.scalar_weight_hidden = nnx.Linear(scalar_dim, weight_hidden_dim, rngs=rngs)
        self.scalar_weight_out = nnx.Linear(weight_hidden_dim, weight_numel, rngs=rngs)
        self.gate_out = NormGate(self.irreps, rngs=rngs)
        self.linear_out = EquivariantLinear(self.irreps, self.irreps, rngs=rngs)

    def _pair_scalars(
        self,
        node_features: IrrepsArray,
        senders: Int[Array, " n_edges"],
        receivers: Int[Array, " n_edges"],
    ) -> Float[Array, "n_edges scalar_dim"]:
        """Build the invariant per-edge scalar features driving the weight MLP."""
        node_scalars = _scalar_part(node_features)
        sender_array = node_features.array[senders]
        receiver_array = node_features.array[receivers]
        inner = inner_product(
            IrrepsArray(self.irreps, receiver_array),
            IrrepsArray(self.irreps, sender_array),
        ).array
        inner_non_scalar = inner[..., self._num_scalars :]
        return jnp.concatenate(
            [node_scalars[receivers], node_scalars[senders], inner_non_scalar], axis=-1
        )

    def __call__(
        self,
        node_features: IrrepsArray,
        senders: Int[Array, " n_edges"],
        receivers: Int[Array, " n_edges"],
        edge_radial: Float[Array, "n_edges edge_radial_dim"],
        accumulated: IrrepsArray | None = None,
    ) -> IrrepsArray:
        """Return the refined per-edge feature, optionally added to ``accumulated``.

        Args:
            node_features: Per-atom feature ``(n_atoms, irreps.dim)``.
            senders: ``(n_edges,)`` source atom index per directed edge.
            receivers: ``(n_edges,)`` destination atom index per directed edge.
            edge_radial: ``(n_edges, edge_radial_dim)`` per-edge radial embedding.
            accumulated: Running off-diagonal-block feature from earlier layers
                (added residually), or ``None`` for the first refinement layer.

        Returns:
            The refined per-edge feature ``(n_edges, irreps.dim)``.
        """
        # Bound the magnitude before the squaring pair tensor product.
        node_features = rms_normalize(node_features)
        inner_nodes = self.linear_inner(node_features)
        pair_scalars = self._pair_scalars(inner_nodes, senders, receivers)
        projected = self.linear_node(self.gate_pre(node_features))
        radial_hidden = jax.nn.silu(self.radial_weight_hidden(edge_radial))
        scalar_hidden = jax.nn.silu(self.scalar_weight_hidden(pair_scalars))
        weights = self.radial_weight_out(radial_hidden) * self.scalar_weight_out(scalar_hidden)
        sender_features = IrrepsArray(self.irreps, projected.array[senders])
        receiver_features = IrrepsArray(self.irreps, projected.array[receivers])
        refined = self.tensor_product(sender_features, receiver_features, weights=weights)
        refined = self.linear_out(self.gate_out(refined))
        if accumulated is not None:
            refined = IrrepsArray(self.irreps, refined.array + accumulated.array)
        return refined


__all__ = ["PairInteractionLayer", "SelfInteractionLayer"]
