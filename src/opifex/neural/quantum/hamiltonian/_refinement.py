r"""QHNet self-interaction refinement layer for the diagonal Fock block feature.

The expressivity core a NequIP-style trunk lacks for Hamiltonian prediction (Yu
et al. 2023, "QHNet", arXiv:2306.04922; reference ``divelab/AIRS``
``OpenDFT/QHBench/QH9/models/QHNet.py`` ``SelfNetLayer``). The trunk produces
per-atom equivariant features; the **Fock blocks** are rank-2 tensors that need
*products* of those features:

* :class:`SelfInteractionLayer` forms the diagonal-block feature from a channel-wise
  **self** tensor product ``tp(W_l x, W_r x)`` of an atom's own feature -- the
  products ``D^{l_i} (x) D^{l_j}`` the on-site block ``H_ii`` transforms as.

The off-diagonal counterpart -- QHNet's ``PairNetLayer`` -- is realised by the
SO(2)-frame
:class:`~opifex.neural.quantum.hamiltonian.so2_convolution.SO2PairInteractionLayer`,
which replaces the dense ``O(L^3)`` ``tp(x[src], x[dst])`` (the dominant cost) with
the cheap eSCN order-diagonal operations (QHNetV2, arXiv:2506.09398).

It reuses the opifex equivariant primitives
(:class:`~opifex.neural.equivariant.ChannelwiseTensorProduct` for the ``O(mul)``
``"uuu"`` coupling, :class:`~opifex.neural.equivariant.NormGate` nonlinearity,
:class:`~opifex.neural.equivariant.EquivariantLinear`) and accumulates residually
across the layer stack (QHNet's ``fii`` running sum). It operates in an
**all-even** irrep space (the trunk's parities are relabelled to even at the
refinement boundary, matching QHNet's ``hidden_irrep_base``); it is therefore
SO(3)-equivariant, which is all the matrix head requires.
"""

from __future__ import annotations

from flax import nnx

from opifex.neural.equivariant import (
    ChannelwiseTensorProduct,
    EquivariantLinear,
    Irreps,
    IrrepsArray,
    NormGate,
    rms_normalize,
)


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


__all__ = ["SelfInteractionLayer"]
