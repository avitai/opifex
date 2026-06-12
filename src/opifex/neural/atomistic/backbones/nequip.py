r"""NequIP backbone: E(3)-equivariant tensor-product message passing.

NequIP (Batzner et al. 2022, "E(3)-equivariant graph neural networks for
data-efficient and accurate interatomic potentials", Nat. Commun. 13, 2453;
arXiv:2101.03164) is the canonical high-:math:`l` E(3)-equivariant potential.
Node features are *steerable* :class:`~opifex.neural.equivariant.IrrepsArray`\ s
(a direct sum of ``mul x l`` irreps). Each interaction is a Clebsch-Gordan
**tensor-product convolution**: the sender's features are tensored with the
spherical harmonics ``Y_l(\hat r_{ij})`` of the edge direction, with the path
contributions modulated per edge by a radial network ``R(r_{ij})`` (an MLP on a
Bessel expansion, with no bias so ``R(0) = 0`` and smoothly cut off at ``r_c``):

.. math::
   m_i = \frac{1}{\sqrt{\bar n}} \sum_{j \in \mathcal N(i)}
       R(r_{ij}) \odot \bigl(h_j \otimes_{\text{CG}} Y(\hat r_{ij})\bigr).

A self-interaction :class:`~opifex.neural.equivariant.EquivariantLinear`, an
equivariant :func:`~opifex.neural.equivariant.gate` nonlinearity and a residual
self-connection complete each layer (the convolution outline of the reference
``../jax-md/jax_md/_nn/nequip.py``). The final ``0e`` (scalar) channel is read
out per atom; because the whole network is E(3)-equivariant and the readout
selects an invariant scalar, the per-atom ``node_features`` are E(3)- and
permutation-invariant -- the contract
:class:`opifex.neural.atomistic.heads.EnergyHead` consumes.

**Reuse of opifex's Q0 kit (no primitive reimplemented):**
:func:`~opifex.neural.equivariant.spherical_harmonics` (edge embedding),
:class:`~opifex.neural.equivariant.FullyConnectedTensorProduct` (the CG path
weights -- modulated per edge by the radial network here, the equivariant
analogue of NequIP's per-path radial weights),
:class:`~opifex.neural.equivariant.EquivariantLinear` (self-interaction),
:func:`~opifex.neural.equivariant.gate` (equivariant nonlinearity),
:class:`~opifex.neural.equivariant.BesselBasis` /
:func:`~opifex.neural.equivariant.cosine_cutoff` (radial network + envelope), and
the graph + scatter primitives via the shared
:mod:`opifex.neural.atomistic.backbones._message_passing` helper. It plugs into
:class:`opifex.neural.atomistic.AtomisticModel` through the
:class:`opifex.core.quantum.protocols.Backbone` protocol (self-registered
``"nequip"``).

Deferred: a MACE-style higher-body-order **symmetric contraction** upgrade
(Batatia et al. 2022, "MACE", arXiv:2206.07697; reference
``../mace/mace/modules/symmetric_contraction.py``) would replace the two-body
edge tensor product with a many-body product basis. It is intentionally *not*
built here; the :class:`NequIPConfig` ``correlation`` field is the documented
hook for it (validated to ``1`` for now).

References:
    * Batzner et al. 2022, arXiv:2101.03164 -- the architecture.
    * ``../jax-md/jax_md/_nn/nequip.py`` (Flax + e3nn-jax) -- the convolution
      outline (linear, TP + aggregate, neighbour normalisation, self-connection,
      gate) validated against numerically.
    * ``../e3nn-jax`` -- the irreps / tensor-product / gate semantics reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.core.quantum.registry import register_backbone
from opifex.neural.atomistic.backbones._message_passing import edge_geometry, EdgeGeometry
from opifex.neural.equivariant import (
    apply_scalar_weights,
    BesselBasis,
    cosine_cutoff,
    EquivariantLinear,
    FullyConnectedTensorProduct,
    gate,
    Irreps,
    IrrepsArray,
    scatter_sum,
    spherical_harmonics,
)


if TYPE_CHECKING:
    from opifex.core.quantum.molecular_system import MolecularSystem


_MAX_ATOMIC_NUMBER = 118
"""Highest supported nuclear charge; the embedding table has one row per Z=0..118."""

_DEFAULT_CORRELATION = 1
"""Body-order correlation. Only ``1`` (two-body edge TP) is implemented; the
MACE-style symmetric-contraction upgrade (>1) is a documented deferral."""


def _gate_input_irreps(hidden_irreps: Irreps) -> Irreps:
    r"""Return the pre-gate irreps that yield ``hidden_irreps`` after gating.

    The :func:`opifex.neural.equivariant.gate` consumes one extra ``0e`` gate
    scalar per non-scalar multiplicity (placed after the regular scalars) and
    drops them on output. To obtain ``hidden_irreps`` after the gate, the layer
    before it must therefore emit ``hidden_irreps`` plus ``n_gate x 0e``, where
    ``n_gate`` is the number of non-scalar multiplicities.

    Args:
        hidden_irreps: The desired post-gate layout.

    Returns:
        The pre-gate layout (regular scalars, then gate scalars, then vectors).
    """
    scalar_blocks = [(mul, ir) for mul, ir in hidden_irreps.blocks if ir.l == 0]
    vector_blocks = [(mul, ir) for mul, ir in hidden_irreps.blocks if ir.l > 0]
    num_gates = sum(mul for mul, _ in vector_blocks)
    gate_block = [(num_gates, Irreps("0e").blocks[0][1])] if num_gates > 0 else []
    return Irreps(tuple(scalar_blocks) + tuple(gate_block) + tuple(vector_blocks))


@dataclass(frozen=True, slots=True, kw_only=True)
class NequIPConfig:
    """Hyper-parameters of a :class:`NequIP` backbone.

    Attributes:
        hidden_irreps: Steerable layout of the per-atom hidden features, e.g.
            ``"16x0e + 8x1o + 4x2e"``.
        sh_lmax: Maximum spherical-harmonic degree of the edge embedding.
        num_interactions: Number of tensor-product convolution layers ``T``.
        num_radial_basis: Number of Bessel radial-basis functions.
        radial_hidden_dim: Hidden width of the radial network MLP.
        cutoff: Connection / cutoff radius ``r_c`` (in the system's length units).
        average_num_neighbors: Constant ``sqrt`` normaliser for the aggregated
            message (NequIP's ``n_neighbors`` internal normalisation).
        correlation: Body-order correlation. Only ``1`` is implemented; ``>1``
            (MACE symmetric contraction) is a documented deferral.
    """

    hidden_irreps: str = "16x0e + 8x1o + 4x2e"
    sh_lmax: int = 2
    num_interactions: int = 3
    num_radial_basis: int = 8
    radial_hidden_dim: int = 64
    cutoff: float = 5.0
    average_num_neighbors: float = 1.0
    correlation: int = _DEFAULT_CORRELATION


class _RadialNetwork(nnx.Module):
    """Bias-free MLP mapping a radial-basis expansion to per-multiplicity weights.

    No bias guarantees ``R(0) = 0`` (NequIP / MACE convention); the output is one
    scalar weight per output multiplicity, broadcast over each irrep's ``2l+1``
    components -- an equivariant (``0e`` x irrep) modulation of the tensor-product
    paths.
    """

    def __init__(self, config: NequIPConfig, num_weights: int, *, rngs: nnx.Rngs) -> None:
        """Build the two-layer bias-free radial MLP."""
        super().__init__()
        self.hidden = nnx.Linear(
            config.num_radial_basis, config.radial_hidden_dim, use_bias=False, rngs=rngs
        )
        self.out = nnx.Linear(config.radial_hidden_dim, num_weights, use_bias=False, rngs=rngs)

    def __call__(
        self, radial: Float[Array, "max_edges num_radial_basis"]
    ) -> Float[Array, "max_edges num_weights"]:
        """Return the per-edge, per-multiplicity radial weights."""
        return self.out(nnx.silu(self.hidden(radial)))


class _ConvolutionLayer(nnx.Module):
    """One NequIP tensor-product convolution + self-interaction + gate layer."""

    def __init__(
        self,
        node_irreps: Irreps,
        sh_irreps: Irreps,
        hidden_irreps: Irreps,
        config: NequIPConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the edge TP, radial network, self-interaction and gate-output linear."""
        super().__init__()
        self.node_irreps = node_irreps
        self.hidden_irreps = hidden_irreps
        self._gate_irreps = _gate_input_irreps(hidden_irreps)
        # Edge tensor product: h_j (x) Y(r_hat) -> message irreps.
        self.tensor_product = FullyConnectedTensorProduct(
            node_irreps, sh_irreps, self._gate_irreps, rngs=rngs
        )
        self.radial_network = _RadialNetwork(config, self._gate_irreps.num_irreps, rngs=rngs)
        # Self-interaction (residual self-connection) onto the post-gate layout.
        self.self_interaction = EquivariantLinear(node_irreps, hidden_irreps, rngs=rngs)
        self.average_num_neighbors = config.average_num_neighbors

    def __call__(
        self,
        node_features: IrrepsArray,
        edge_sh: IrrepsArray,
        geometry: EdgeGeometry,
        radial: Float[Array, "max_edges num_radial_basis"],
        envelope: Float[Array, "max_edges 1"],
        num_atoms: int,
    ) -> IrrepsArray:
        """Return the updated node features after one convolution layer.

        Args:
            node_features: Current per-atom steerable features.
            edge_sh: Spherical-harmonic embedding ``Y(r_hat)`` per edge.
            geometry: Per-edge geometry (clamped indices + validity mask).
            radial: Radial-basis expansion of the edge lengths.
            envelope: Smooth cutoff envelope per edge (zero on padded slots).
            num_atoms: Number of atoms (static segment count for the scatter).

        Returns:
            The post-gate node features with ``self.hidden_irreps``.
        """
        sender_features = IrrepsArray(node_features.irreps, node_features.array[geometry.senders])
        message = self.tensor_product(sender_features, edge_sh)
        weights = self.radial_network(radial) * envelope
        message = apply_scalar_weights(message, weights)
        aggregated = scatter_sum(message.array, geometry.receivers, num_segments=num_atoms)
        aggregated = aggregated / jnp.sqrt(self.average_num_neighbors)
        gated = gate(IrrepsArray(self._gate_irreps, aggregated))
        self_connection = self.self_interaction(node_features)
        return IrrepsArray(self.hidden_irreps, gated.array + self_connection.array)


@register_backbone("nequip")
class NequIP(nnx.Module):
    """E(3)-equivariant tensor-product backbone (Batzner et al. 2022).

    Satisfies :class:`opifex.core.quantum.protocols.Backbone`: maps a
    :class:`~opifex.core.quantum.molecular_system.MolecularSystem` and its padded
    edge index to ``{"node_features": (n_atoms, num_scalar_features)}`` invariant
    scalars (the ``0e`` channels of the final steerable features).

    Args:
        config: Backbone hyper-parameters. Defaults to :class:`NequIPConfig`.
        rngs: Random number generators (keyword-only) seeding all weights.

    Raises:
        ValueError: If ``config.correlation`` is not ``1`` (the MACE-style
            higher-correlation upgrade is a documented deferral) or the hidden
            irreps carry no ``0e`` scalar channel to read out.
    """

    def __init__(self, *, config: NequIPConfig | None = None, rngs: nnx.Rngs) -> None:
        """Build the embedding, edge SH layout, radial basis and convolution layers."""
        super().__init__()
        self.config = config if config is not None else NequIPConfig()
        if self.config.correlation != _DEFAULT_CORRELATION:
            raise ValueError(
                f"NequIP supports correlation={_DEFAULT_CORRELATION} (two-body edge "
                f"tensor product) only; correlation={self.config.correlation} requires "
                "the MACE-style symmetric contraction, which is a documented deferral."
            )
        self.hidden_irreps = Irreps(self.config.hidden_irreps)
        self._num_scalars = sum(mul for mul, ir in self.hidden_irreps.blocks if ir.l == 0)
        if self._num_scalars == 0:
            raise ValueError(
                f"hidden_irreps must contain a 0e scalar channel to read out, got "
                f"{self.hidden_irreps!r}"
            )
        self.sh_irreps = spherical_harmonics(self.config.sh_lmax, jnp.zeros((1, 3))).irreps
        # Scalar atomic-number embedding lifted into the hidden irreps layout.
        self.embedding = nnx.Embed(
            num_embeddings=_MAX_ATOMIC_NUMBER + 1,
            features=self._num_scalars,
            rngs=rngs,
        )
        self._embedding_irreps = Irreps(f"{self._num_scalars}x0e")
        self.embedding_linear = EquivariantLinear(
            self._embedding_irreps, self.hidden_irreps, rngs=rngs
        )
        self.radial_basis = BesselBasis(self.config.num_radial_basis, self.config.cutoff)
        self.layers = nnx.List(
            [
                _ConvolutionLayer(
                    self.hidden_irreps,
                    self.sh_irreps,
                    self.hidden_irreps,
                    self.config,
                    rngs=rngs,
                )
                for _ in range(self.config.num_interactions)
            ]
        )

    def __call__(self, system: MolecularSystem, graph: tuple[Array, Array]) -> dict[str, Array]:
        """Run the NequIP message passing and return per-atom scalar features.

        Args:
            system: The molecular system (atomic numbers + positions).
            graph: The ``(senders, receivers)`` padded edge index.

        Returns:
            ``{"node_features": Array}`` of shape ``(n_atoms, num_scalars)`` --
            the invariant ``0e`` channels of the final steerable features.
        """
        geometry = edge_geometry(system.positions, graph)
        lengths = geometry.lengths[:, 0]
        radial = self.radial_basis(lengths)
        envelope = (cosine_cutoff(lengths, self.config.cutoff) * geometry.mask[:, 0])[:, None]
        edge_sh = spherical_harmonics(self.sh_irreps, geometry.vectors)
        scalar_embedding = IrrepsArray(
            self._embedding_irreps, self.embedding(system.atomic_numbers)
        )
        node_features = self.embedding_linear(scalar_embedding)
        for layer in self.layers:
            node_features = layer(
                node_features, edge_sh, geometry, radial, envelope, system.n_atoms
            )
        # Read out the invariant 0e scalar channels (first block by SH ordering).
        scalar_features = node_features.chunks[0].reshape(system.n_atoms, self._num_scalars)
        return {"node_features": scalar_features}


__all__ = ["NequIP", "NequIPConfig"]
