r"""PaiNN backbone: polarizable scalar/vector equivariant message passing.

PaiNN (Schuett, Unke & Gastegger 2021, "Equivariant message passing for the
prediction of tensorial properties and molecular spectra", ICML; arXiv:2102.03150)
augments SchNet's invariant scalar features ``s_i`` with **equivariant vector
features** ``v_i`` (Cartesian ``l = 1`` channels, shape ``(3, F)``) and updates
them without Clebsch-Gordan tensor products -- using only rotationally covariant
combinations of the scalars, the vectors, the edge unit-direction ``\hat r_{ij}``
and scalar (dot-product) invariants.

Message block (paper eq. 7-8), per edge ``j -> i``:

.. math::
   \phi_{ij} = W_s\,\mathrm{act}(W\,s_j),\qquad
   \mathcal W_{ij} = \mathrm{MLP}(\mathrm{RBF}(r_{ij}))\,f_{\text{cut}}(r_{ij}),

split channelwise into ``(\phi^s, \phi^{vv}, \phi^{vs})``; the scalar message is
``\sum_j \phi^s_{ij}`` and the vector message is
``\sum_j \phi^{vv}_{ij}\, v_j + \phi^{vs}_{ij}\, \hat r_{ij}``.

Update block (paper eq. 9-10) mixes each atom's own vectors through two linear
maps ``U, V`` (no bias, equivariant), forms the scalar invariant
``\langle U v, V v\rangle`` and ``\lVert V v\rVert``, runs a gated MLP, and adds
the gated vector update ``a_{vv}\, U v``.

Because every operation is either a scalar function of invariants or a scalar
times an equivariant vector, the scalars stay invariant and the vectors stay
equivariant under rotation; the emitted per-atom ``node_features`` are the
invariant scalars (the contract :class:`opifex.neural.atomistic.heads.EnergyHead`
consumes).

This implementation **composes opifex's Q0 kit** (graph + scatter via the shared
:mod:`opifex.neural.atomistic.backbones._message_passing` helper,
:class:`~opifex.neural.equivariant.BesselBasis`,
:func:`~opifex.neural.equivariant.cosine_cutoff`) rather than reimplementing
primitives, and plugs into :class:`opifex.neural.atomistic.AtomisticModel`
through the :class:`opifex.core.quantum.protocols.Backbone` protocol
(self-registered ``"painn"``).

References:
    * Schuett, Unke & Gastegger 2021, arXiv:2102.03150 -- the message/update
      equations (eq. 7-10) and the scalar/vector channel split.
    * Satorras et al. 2021, "E(n) Equivariant Graph Neural Networks" (ICML),
      ``../artifex .../layers/egnn.py`` -- the E(n) "scalar-gates-vector"
      update rule used as the equivariance reference (dense ``[B, N, N]``
      layout there; reformulated here for the sparse ``(senders, receivers)``
      atomistic contract, so it is an *algorithm reference*, not a wrapped block).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.core.quantum.registry import register_backbone
from opifex.neural.atomistic.backbones._message_passing import edge_geometry, EdgeGeometry
from opifex.neural.dtypes import default_float_dtype
from opifex.neural.equivariant import BesselBasis, cosine_cutoff, scatter_sum


_MAX_ATOMIC_NUMBER = 118
"""Highest supported nuclear charge; the embedding table has one row per Z=0..118."""

_VECTOR_NORM_EPSILON = 1e-12
"""Additive guard so ``grad(norm)`` of the vector-feature norm stays finite at 0."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PaiNNConfig:
    """Hyper-parameters of a :class:`PaiNN` backbone.

    Attributes:
        feature_dim: Width ``F`` of the scalar (and vector) feature channels.
        num_interactions: Number of message + update interaction blocks ``T``.
        num_radial_basis: Number of Bessel radial-basis functions.
        cutoff: Connection / cutoff radius ``r_c`` (in the system's length units).
    """

    feature_dim: int = 64
    num_interactions: int = 3
    num_radial_basis: int = 16
    cutoff: float = 5.0


class _MessageBlock(nnx.Module):
    """PaiNN message block (paper eq. 7-8): scalar/vector message from neighbours."""

    def __init__(self, config: PaiNNConfig, *, rngs: nnx.Rngs) -> None:
        """Build the scalar feature MLP and the radial filter-generating linear."""
        super().__init__()
        feature_dim = config.feature_dim
        dtype = default_float_dtype()
        self.scalar_hidden = nnx.Linear(feature_dim, feature_dim, param_dtype=dtype, rngs=rngs)
        self.scalar_out = nnx.Linear(feature_dim, 3 * feature_dim, param_dtype=dtype, rngs=rngs)
        self.filter = nnx.Linear(
            config.num_radial_basis, 3 * feature_dim, param_dtype=dtype, rngs=rngs
        )

    def __call__(
        self,
        scalars: Float[Array, "n_atoms feature_dim"],
        vectors: Float[Array, "n_atoms 3 feature_dim"],
        geometry: EdgeGeometry,
        radial: Float[Array, "max_edges num_radial_basis"],
        envelope: Float[Array, "max_edges 1"],
        num_atoms: int,
    ) -> tuple[Float[Array, "n_atoms feature_dim"], Float[Array, "n_atoms 3 feature_dim"]]:
        """Return the scalar/vector message residuals aggregated onto receivers.

        Args:
            scalars: Current per-atom scalar features.
            vectors: Current per-atom equivariant vector features ``(n, 3, F)``.
            geometry: Per-edge geometry (clamped indices, unit vectors, mask).
            radial: Radial-basis expansion of the edge lengths.
            envelope: Smooth cutoff envelope per edge (zero on padded slots).
            num_atoms: Number of atoms (static segment count for the scatter).

        Returns:
            ``(delta_scalars, delta_vectors)`` residual updates.
        """
        feature_dim = scalars.shape[-1]
        # Scalar transform of neighbour features, gated by the radial filter.
        phi = self.scalar_out(nnx.silu(self.scalar_hidden(scalars)))[geometry.senders]
        weights = self.filter(radial) * envelope
        gated = phi * weights
        phi_scalar = gated[:, :feature_dim]
        phi_vv = gated[:, feature_dim : 2 * feature_dim]
        phi_vs = gated[:, 2 * feature_dim :]
        # Vector message: scale neighbour vectors + emit along the edge direction.
        neighbour_vectors = vectors[geometry.senders]
        edge_vector_message = (
            phi_vv[:, None, :] * neighbour_vectors
            + phi_vs[:, None, :] * geometry.unit_vectors[:, :, None]
        )
        delta_scalars = scatter_sum(phi_scalar, geometry.receivers, num_segments=num_atoms)
        delta_vectors = scatter_sum(edge_vector_message, geometry.receivers, num_segments=num_atoms)
        return delta_scalars, delta_vectors


class _UpdateBlock(nnx.Module):
    """PaiNN update block (paper eq. 9-10): per-atom scalar/vector gated mixing."""

    def __init__(self, config: PaiNNConfig, *, rngs: nnx.Rngs) -> None:
        """Build the equivariant vector linears ``U, V`` and the gated scalar MLP."""
        super().__init__()
        feature_dim = config.feature_dim
        dtype = default_float_dtype()
        # Bias-free linear maps over the channel axis keep equivariance (no l-mix).
        self.vector_u = nnx.Linear(
            feature_dim, feature_dim, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.vector_v = nnx.Linear(
            feature_dim, feature_dim, use_bias=False, param_dtype=dtype, rngs=rngs
        )
        self.update_hidden = nnx.Linear(2 * feature_dim, feature_dim, param_dtype=dtype, rngs=rngs)
        self.update_out = nnx.Linear(feature_dim, 3 * feature_dim, param_dtype=dtype, rngs=rngs)

    def __call__(
        self,
        scalars: Float[Array, "n_atoms feature_dim"],
        vectors: Float[Array, "n_atoms 3 feature_dim"],
    ) -> tuple[Float[Array, "n_atoms feature_dim"], Float[Array, "n_atoms 3 feature_dim"]]:
        """Return the scalar/vector update residuals for one atom-wise block.

        Args:
            scalars: Current per-atom scalar features.
            vectors: Current per-atom equivariant vector features ``(n, 3, F)``.

        Returns:
            ``(delta_scalars, delta_vectors)`` residual updates.
        """
        feature_dim = scalars.shape[-1]
        u_vectors = self.vector_u(vectors)
        v_vectors = self.vector_v(vectors)
        # Invariant scalars from the vectors: ||V v|| and <U v, V v>.
        v_norm = jnp.sqrt(jnp.sum(v_vectors**2, axis=1) + _VECTOR_NORM_EPSILON)
        inner = jnp.sum(u_vectors * v_vectors, axis=1)
        combined = jnp.concatenate([scalars, v_norm], axis=-1)
        gated = self.update_out(nnx.silu(self.update_hidden(combined)))
        a_vv = gated[:, :feature_dim]
        a_sv = gated[:, feature_dim : 2 * feature_dim]
        a_ss = gated[:, 2 * feature_dim :]
        delta_vectors = a_vv[:, None, :] * u_vectors
        delta_scalars = a_sv * inner + a_ss
        return delta_scalars, delta_vectors


@register_backbone("painn")
class PaiNN(nnx.Module):
    """Equivariant scalar/vector message-passing backbone (Schuett et al. 2021).

    Satisfies :class:`opifex.core.quantum.protocols.Backbone`: maps a
    :class:`~opifex.core.quantum.molecular_system.MolecularSystem` and its padded
    edge index to ``{"node_features": (n_atoms, feature_dim)}`` invariant scalars
    (the equivariant vector channels are internal state).

    Args:
        config: Backbone hyper-parameters. Defaults to :class:`PaiNNConfig`.
        rngs: Random number generators (keyword-only) seeding all weights.
    """

    def __init__(self, *, config: PaiNNConfig | None = None, rngs: nnx.Rngs) -> None:
        """Build the atomic-number embedding, radial basis and interaction blocks."""
        super().__init__()
        self.config = config if config is not None else PaiNNConfig()
        self.embedding = nnx.Embed(
            num_embeddings=_MAX_ATOMIC_NUMBER + 1,
            features=self.config.feature_dim,
            param_dtype=default_float_dtype(),
            rngs=rngs,
        )
        self.radial_basis = BesselBasis(self.config.num_radial_basis, self.config.cutoff)
        self.messages = nnx.List(
            [_MessageBlock(self.config, rngs=rngs) for _ in range(self.config.num_interactions)]
        )
        self.updates = nnx.List(
            [_UpdateBlock(self.config, rngs=rngs) for _ in range(self.config.num_interactions)]
        )

    def __call__(self, system: MolecularSystem, graph: tuple[Array, Array]) -> dict[str, Array]:
        """Run the PaiNN message passing and return per-atom scalar features.

        Args:
            system: The molecular system (atomic numbers + positions).
            graph: The ``(senders, receivers)`` padded edge index.

        Returns:
            ``{"node_features": Array}`` of shape ``(n_atoms, feature_dim)``.
        """
        geometry = edge_geometry(system.positions, graph)
        lengths = geometry.lengths[:, 0]
        radial = self.radial_basis(lengths)
        envelope = (cosine_cutoff(lengths, self.config.cutoff) * geometry.mask[:, 0])[:, None]
        scalars = self.embedding(system.atomic_numbers)
        vectors = jnp.zeros((system.n_atoms, 3, self.config.feature_dim), dtype=scalars.dtype)
        for message, update in zip(self.messages, self.updates, strict=True):
            delta_scalars, delta_vectors = message(
                scalars, vectors, geometry, radial, envelope, system.n_atoms
            )
            scalars = scalars + delta_scalars
            vectors = vectors + delta_vectors
            delta_scalars, delta_vectors = update(scalars, vectors)
            scalars = scalars + delta_scalars
            vectors = vectors + delta_vectors
        return {"node_features": scalars}


__all__ = ["PaiNN", "PaiNNConfig"]
