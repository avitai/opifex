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

Equivariant linear mixings before and after the tensor product, an equivariant
:func:`~opifex.neural.equivariant.gate` nonlinearity and a residual
self-connection complete each layer. The final ``0e`` (scalar) channel is read
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

Body order is set by :class:`NequIPConfig` ``correlation``: ``1`` is the two-body
edge tensor product with a gate nonlinearity, while ``> 1`` adds a MACE-style
higher-body-order **symmetric contraction**
(:class:`~opifex.neural.equivariant.SymmetricContraction`) on the aggregated
message -- a per-element product basis that replaces the gate as the body-order
nonlinearity (Batatia et al. 2022, "MACE", arXiv:2206.07697). ``correlation > 1``
requires uniform-multiplicity hidden irreps and ``species`` (the per-element
weights).

References:
    * Batzner et al. 2022, arXiv:2101.03164 -- the two-body architecture.
    * Batatia et al. 2022, arXiv:2206.07697 -- the higher-body-order contraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.core.quantum.registry import register_backbone
from opifex.neural.atomistic.backbones._message_passing import edge_geometry, EdgeGeometry
from opifex.neural.dtypes import default_float_dtype
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
    SymmetricContraction,
)
from opifex.neural.equivariant._assembly import from_chunks


_MAX_ATOMIC_NUMBER = 118
"""Highest supported nuclear charge; the embedding table has one row per Z=0..118."""

_DEFAULT_CORRELATION = 1
"""Default body-order correlation: ``1`` is the two-body edge tensor product;
``> 1`` enables the MACE-style symmetric contraction."""


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
        correlation: Body-order correlation. ``1`` is the two-body edge tensor
            product with a gate nonlinearity; ``> 1`` adds the MACE-style symmetric
            contraction (requires uniform-multiplicity ``hidden_irreps`` and
            ``species``).
        sh_normalization: Normalisation convention for the edge spherical-harmonic
            embedding, one of ``"component"`` (default; unit per-component variance,
            the NequIP convention that keeps the embedding at the unit scale the
            tensor-product weight init assumes), ``"integral"`` or ``"norm"``.
        normalize_gate_act: If ``True`` (default), the gate rescales each activation
            to unit second moment under a standard-normal input, so feature
            magnitudes do not drift across stacked gated interaction layers.
        species: Sorted distinct atomic numbers in the dataset (e.g. ``(1, 6, 8)``
            for an H/C/O system). When non-empty, each interaction's self-connection
            is **species-indexed** (a per-element residual, the NequIP convention:
            the skip is a tensor product of the node features with the one-hot atom
            type) instead of a single shared linear -- giving every element its own
            self-interaction. Empty (default) uses the shared linear self-connection.
    """

    hidden_irreps: str = "16x0e + 8x1o + 4x2e"
    sh_lmax: int = 2
    num_interactions: int = 3
    num_radial_basis: int = 8
    radial_hidden_dim: int = 64
    cutoff: float = 5.0
    average_num_neighbors: float = 1.0
    correlation: int = _DEFAULT_CORRELATION
    sh_normalization: str = "component"
    normalize_gate_act: bool = True
    species: tuple[int, ...] = ()


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
        dtype = default_float_dtype()
        self.hidden = nnx.Linear(
            config.num_radial_basis,
            config.radial_hidden_dim,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.out = nnx.Linear(
            config.radial_hidden_dim, num_weights, use_bias=False, param_dtype=dtype, rngs=rngs
        )

    def __call__(
        self, radial: Float[Array, "max_edges num_radial_basis"]
    ) -> Float[Array, "max_edges num_weights"]:
        """Return the per-edge, per-multiplicity radial weights."""
        return self.out(nnx.silu(self.hidden(radial)))


def _to_channel_basis(features: IrrepsArray) -> Array:
    """Uniform-multiplicity features -> ``(..., channels, single_particle_dim)``.

    Concatenates the per-irrep chunks (each ``(..., channels, ir.dim)``) along the
    last axis, giving the single-particle basis the symmetric contraction consumes.
    Requires every irrep block to share one multiplicity (the channel count).
    """
    return jnp.concatenate(list(features.chunks), axis=-1)


def _from_channel_basis(channel_array: Array, hidden_irreps: Irreps) -> IrrepsArray:
    """Inverse of :func:`_to_channel_basis`: channel basis -> ``hidden_irreps``."""
    chunks: list[Array | None] = []
    cursor = 0
    for _, irrep in hidden_irreps:
        chunks.append(channel_array[..., cursor : cursor + irrep.dim])
        cursor += irrep.dim
    return from_chunks(hidden_irreps, chunks, channel_array.shape[:-2], channel_array.dtype)


class _ConvolutionLayer(nnx.Module):
    """One NequIP interaction block (Batzner et al. 2022).

    1. ``linear_up`` -- an equivariant linear self-mixing of the node features
       *before* the edge tensor product (the ``linear_1`` step),
    2. the edge tensor product ``h_j (x) Y(r_hat)`` modulated by the radial
       network, scatter-summed onto receivers and divided by
       ``sqrt(avg_num_neighbors)``,
    3. ``linear_down`` -- an equivariant linear mixing of the aggregated message
       (the ``linear_2`` step),
    4. a self-connection (residual) added *before* the gate, and
    5. the gate nonlinearity (``silu`` even / ``tanh`` odd, ``silu`` gates).

    The two equivariant linear mixings (``linear_up`` / ``linear_down``) are the
    learned-capacity core of the NequIP interaction; omitting them materially
    under-fits forces (energy gradients) on hard molecular benchmarks.
    """

    def __init__(
        self,
        node_irreps: Irreps,
        sh_irreps: Irreps,
        hidden_irreps: Irreps,
        config: NequIPConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the interaction submodules (gated, or product basis if correlation>1)."""
        super().__init__()
        self.node_irreps = node_irreps
        self.hidden_irreps = hidden_irreps
        self._num_species = len(config.species)
        self._correlation = config.correlation
        # linear_up: equivariant self-mixing of node features before the edge TP.
        self.linear_up = EquivariantLinear(node_irreps, node_irreps, rngs=rngs)
        # Two-body (correlation=1) gate output vs higher-body-order (correlation>1)
        # symmetric contraction. The contraction is the body-order nonlinearity, so it
        # replaces the gate; it needs species conditioning and uniform-multiplicity
        # hidden irreps (the channel-wise product basis).
        if self._correlation > 1:
            self._message_irreps = hidden_irreps
            multiplicities = {mul for mul, _ in hidden_irreps}
            if len(multiplicities) != 1:
                raise ValueError(
                    "correlation > 1 requires uniform-multiplicity hidden_irreps "
                    f"(one channel count), got {hidden_irreps!r}."
                )
            if self._num_species == 0:
                raise ValueError("correlation > 1 requires config.species (per-element weights).")
            channels = multiplicities.pop()
            single = Irreps(tuple((1, ir) for _, ir in hidden_irreps))
            self._channels = channels
            self.linear_down = EquivariantLinear(hidden_irreps, hidden_irreps, rngs=rngs)
            self.symmetric_contraction = SymmetricContraction(
                single,
                single,
                correlation=self._correlation,
                num_species=self._num_species,
                num_channels=channels,
                rngs=rngs,
            )
            self.product_linear = EquivariantLinear(hidden_irreps, hidden_irreps, rngs=rngs)
            self.self_interaction: EquivariantLinear | FullyConnectedTensorProduct = (
                FullyConnectedTensorProduct(
                    node_irreps, Irreps(f"{self._num_species}x0e"), hidden_irreps, rngs=rngs
                )
            )
        else:
            self._message_irreps = _gate_input_irreps(hidden_irreps)
            self._normalize_gate_act = config.normalize_gate_act
            self.linear_down = EquivariantLinear(
                self._message_irreps, self._message_irreps, rngs=rngs
            )
            if self._num_species > 0:
                self.self_interaction = FullyConnectedTensorProduct(
                    node_irreps, Irreps(f"{self._num_species}x0e"), self._message_irreps, rngs=rngs
                )
            else:
                self.self_interaction = EquivariantLinear(
                    node_irreps, self._message_irreps, rngs=rngs
                )
        # Edge tensor product: h_j (x) Y(r_hat) -> message irreps.
        self.tensor_product = FullyConnectedTensorProduct(
            node_irreps, sh_irreps, self._message_irreps, rngs=rngs
        )
        self.radial_network = _RadialNetwork(config, self._message_irreps.num_irreps, rngs=rngs)
        self.average_num_neighbors = config.average_num_neighbors

    def _aggregate_message(
        self,
        node_features: IrrepsArray,
        edge_sh: IrrepsArray,
        geometry: EdgeGeometry,
        radial: Float[Array, "max_edges num_radial_basis"],
        envelope: Float[Array, "max_edges 1"],
        num_atoms: int,
    ) -> Array:
        """Form the radial-weighted edge tensor product, aggregated onto receivers."""
        mixed_nodes = self.linear_up(node_features)
        sender_features = IrrepsArray(mixed_nodes.irreps, mixed_nodes.array[geometry.senders])
        message = self.tensor_product(sender_features, edge_sh)
        message = apply_scalar_weights(message, self.radial_network(radial) * envelope)
        aggregated = scatter_sum(message.array, geometry.receivers, num_segments=num_atoms)
        return aggregated / jnp.sqrt(self.average_num_neighbors)

    def _self_connection(self, node_features: IrrepsArray, node_attrs: IrrepsArray | None) -> Array:
        """Per-element (or shared) residual self-connection."""
        if isinstance(self.self_interaction, FullyConnectedTensorProduct):
            if node_attrs is None:
                raise ValueError("species-indexed self-connection requires node_attrs.")
            return self.self_interaction(node_features, node_attrs).array
        return self.self_interaction(node_features).array

    def __call__(
        self,
        node_features: IrrepsArray,
        edge_sh: IrrepsArray,
        geometry: EdgeGeometry,
        radial: Float[Array, "max_edges num_radial_basis"],
        envelope: Float[Array, "max_edges 1"],
        num_atoms: int,
        node_attrs: IrrepsArray | None = None,
    ) -> IrrepsArray:
        """Return the updated node features after one interaction block.

        Args:
            node_features: Current per-atom steerable features.
            edge_sh: Spherical-harmonic embedding ``Y(r_hat)`` per edge.
            geometry: Per-edge geometry (clamped indices + validity mask).
            radial: Radial-basis expansion of the edge lengths.
            envelope: Smooth cutoff envelope per edge (zero on padded slots).
            num_atoms: Number of atoms (static segment count for the scatter).
            node_attrs: One-hot atom-type attributes; required for the
                species-indexed self-connection / symmetric contraction.

        Returns:
            The updated node features with ``self.hidden_irreps``.
        """
        aggregated = self._aggregate_message(
            node_features, edge_sh, geometry, radial, envelope, num_atoms
        )
        mixed_message = self.linear_down(IrrepsArray(self._message_irreps, aggregated))
        self_connection = self._self_connection(node_features, node_attrs)
        if self._correlation > 1:
            if node_attrs is None:
                raise ValueError("correlation > 1 requires node_attrs (species one-hot).")
            # Symmetric contraction (body-order nonlinearity) on the channel basis.
            channel_basis = _to_channel_basis(mixed_message)  # (n, channels, single.dim)
            single_irreps = self.symmetric_contraction.irreps_in
            contracted = self.symmetric_contraction(
                IrrepsArray(single_irreps, channel_basis), node_attrs.array
            )
            product = _from_channel_basis(contracted.array, self.hidden_irreps)
            return IrrepsArray(
                self.hidden_irreps, self.product_linear(product).array + self_connection
            )
        combined = IrrepsArray(self._message_irreps, mixed_message.array + self_connection)
        # Gate with the NequIP activations (silu even / tanh odd, silu gates),
        # normalised to unit second moment when configured.
        return gate(
            combined,
            even_act=jax.nn.silu,
            odd_act=jax.nn.tanh,
            gate_act=jax.nn.silu,
            normalize_act=self._normalize_gate_act,
        )


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
        ValueError: If ``config.correlation < 1`` or the hidden irreps carry no
            ``0e`` scalar channel to read out.
    """

    def __init__(self, *, config: NequIPConfig | None = None, rngs: nnx.Rngs) -> None:
        """Build the embedding, edge SH layout, radial basis and convolution layers."""
        super().__init__()
        self.config = config if config is not None else NequIPConfig()
        if self.config.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.config.correlation}.")
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
            param_dtype=default_float_dtype(),
            rngs=rngs,
        )
        self._embedding_irreps = Irreps(f"{self._num_scalars}x0e")
        self.embedding_linear = EquivariantLinear(
            self._embedding_irreps, self.hidden_irreps, rngs=rngs
        )
        # Optional species conditioning: a static atomic-number -> compact type-index
        # lookup feeding the per-element one-hot self-connection (see NequIPConfig).
        self._num_species = len(self.config.species)
        if self._num_species > 0:
            lookup = [0] * (_MAX_ATOMIC_NUMBER + 1)
            for type_index, atomic_number in enumerate(self.config.species):
                lookup[atomic_number] = type_index
            self._species_lookup = jnp.asarray(lookup, dtype=jnp.int32)
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
        # The edge embedding uses the configured SH normalization ("component" by
        # default): unit per-component variance keeps every message at the scale the
        # tensor-product weight init assumes. "integral" is ~sqrt(4*pi) larger and
        # silently mis-conditions the network.
        edge_sh = spherical_harmonics(
            self.sh_irreps, geometry.vectors, normalization=self.config.sh_normalization
        )
        scalar_embedding = IrrepsArray(
            self._embedding_irreps, self.embedding(system.atomic_numbers)
        )
        node_features = self.embedding_linear(scalar_embedding)
        # One-hot atom-type attributes for the species-indexed self-connection.
        node_attrs: IrrepsArray | None = None
        if self._num_species > 0:
            type_index = self._species_lookup[system.atomic_numbers]
            one_hot = jax.nn.one_hot(type_index, self._num_species, dtype=node_features.array.dtype)
            node_attrs = IrrepsArray(Irreps(f"{self._num_species}x0e"), one_hot)
        for layer in self.layers:
            node_features = layer(
                node_features, edge_sh, geometry, radial, envelope, system.n_atoms, node_attrs
            )
        # Read out the invariant 0e scalar channels (first block by SH ordering).
        scalar_features = node_features.chunks[0].reshape(system.n_atoms, self._num_scalars)
        return {"node_features": scalar_features}


__all__ = ["NequIP", "NequIPConfig"]
