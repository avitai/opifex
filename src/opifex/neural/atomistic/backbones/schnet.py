r"""SchNet backbone: continuous-filter convolutional interatomic potential.

SchNet (Schuett et al. 2018, "SchNet -- A deep learning architecture for
molecules and materials", J. Chem. Phys. 148, 241722; arXiv:1706.08566) is the
canonical *invariant* message-passing potential. Atoms carry scalar features
``x_i`` initialised from an atomic-number embedding; interaction blocks update
them with **continuous-filter convolutions**

.. math::
   x_i^{(t+1)} = x_i^{(t)} + \sum_{j \in \mathcal N(i)}
       \bigl(W_{\text{cf}}\, x_j^{(t)}\bigr) \odot W_{\text{filter}}(r_{ij}),

where the filter ``W_{\text{filter}}(r_{ij})`` is an MLP on a radial-basis
expansion of the interatomic distance, smoothly damped by a cutoff envelope.
Because the messages depend on positions only through the (invariant) distance
``r_{ij} = |r_i - r_j|`` and are summed (permutation invariant), the per-atom
output features are E(3)- and permutation-invariant scalars -- the contract the
:class:`opifex.neural.atomistic.heads.EnergyHead` consumes.

This implementation **composes opifex's Q0 kit** rather than reimplementing
primitives: :func:`opifex.neural.equivariant.radius_graph` /
:func:`~opifex.neural.equivariant.scatter_sum` (graph + aggregation, via the
shared :mod:`opifex.neural.atomistic.backbones._message_passing` helper),
:class:`~opifex.neural.equivariant.BesselBasis` (radial basis) and
:func:`~opifex.neural.equivariant.cosine_cutoff` (smooth cutoff). It plugs into
:class:`opifex.neural.atomistic.AtomisticModel` through the
:class:`opifex.core.quantum.protocols.Backbone` protocol and self-registers as
``"schnet"``.

References:
    * Schuett et al. 2018, arXiv:1706.08566 -- the continuous-filter
      convolution, shifted-softplus activation and interaction-block residual.
    * ``../e3nn-jax`` radius-graph / scatter primitives (reused via Q0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.core.quantum.registry import register_backbone
from opifex.neural.atomistic.backbones._message_passing import edge_geometry, EdgeGeometry
from opifex.neural.equivariant import BesselBasis, cosine_cutoff, scatter_sum


if TYPE_CHECKING:
    from opifex.core.quantum.molecular_system import MolecularSystem


_MAX_ATOMIC_NUMBER = 118
"""Highest supported nuclear charge; the embedding table has one row per Z=0..118."""


def shifted_softplus(x: Float[Array, ...]) -> Float[Array, ...]:
    r"""Compute the shifted-softplus activation ``ln(1 + e^x) - ln 2``.

    The smooth, ``f(0) = 0`` nonlinearity used throughout SchNet
    (arXiv:1706.08566); shifting by ``ln 2`` centres it at the origin.

    Args:
        x: Input array.

    Returns:
        The shifted-softplus of ``x`` (same shape).
    """
    return jax.nn.softplus(x) - jnp.log(2.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class SchNetConfig:
    """Hyper-parameters of a :class:`SchNet` backbone.

    Attributes:
        feature_dim: Width ``F`` of the per-atom scalar feature vector.
        num_interactions: Number of continuous-filter interaction blocks ``T``.
        num_radial_basis: Number of Bessel radial-basis functions.
        cutoff: Connection / cutoff radius ``r_c`` (in the system's length units).
        filter_hidden_dim: Hidden width of the radial filter-generating MLP.
    """

    feature_dim: int = 64
    num_interactions: int = 3
    num_radial_basis: int = 16
    cutoff: float = 5.0
    filter_hidden_dim: int = 64


class _InteractionBlock(nnx.Module):
    """A single SchNet continuous-filter convolution + atomwise update block."""

    def __init__(self, config: SchNetConfig, *, rngs: nnx.Rngs) -> None:
        """Build the filter-generating MLP and the atomwise update MLP."""
        super().__init__()
        feature_dim = config.feature_dim
        self.atomwise_in = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        self.filter_in = nnx.Linear(config.num_radial_basis, config.filter_hidden_dim, rngs=rngs)
        self.filter_out = nnx.Linear(config.filter_hidden_dim, feature_dim, rngs=rngs)
        self.update_hidden = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        self.update_out = nnx.Linear(feature_dim, feature_dim, rngs=rngs)

    def __call__(
        self,
        features: Float[Array, "n_atoms feature_dim"],
        geometry: EdgeGeometry,
        radial: Float[Array, "max_edges num_radial_basis"],
        envelope: Float[Array, "max_edges 1"],
        num_atoms: int,
    ) -> Float[Array, "n_atoms feature_dim"]:
        """Return the residual feature update for one interaction block.

        Args:
            features: Current per-atom scalar features.
            geometry: Per-edge geometry (clamped indices + validity mask).
            radial: Radial-basis expansion of the edge lengths.
            envelope: Smooth cutoff envelope per edge (zero on padded slots).
            num_atoms: Number of atoms (static segment count for the scatter).

        Returns:
            The residual update ``Delta x`` of shape ``(n_atoms, feature_dim)``.
        """
        # Continuous filter: MLP(radial-basis) damped smoothly to zero at r_c.
        filter_values = self.filter_out(shifted_softplus(self.filter_in(radial)))
        filter_values = filter_values * envelope
        # Message: filtered neighbour features (the "continuous-filter conv").
        neighbour = self.atomwise_in(features)[geometry.senders]
        messages = neighbour * filter_values
        aggregated = scatter_sum(messages, geometry.receivers, num_segments=num_atoms)
        # Atomwise update MLP.
        hidden = shifted_softplus(self.update_hidden(aggregated))
        return self.update_out(hidden)


@register_backbone("schnet")
class SchNet(nnx.Module):
    """Invariant continuous-filter convolutional backbone (Schuett et al. 2018).

    Satisfies :class:`opifex.core.quantum.protocols.Backbone`: maps a
    :class:`~opifex.core.quantum.molecular_system.MolecularSystem` and its padded
    edge index to ``{"node_features": (n_atoms, feature_dim)}`` invariant scalars.

    Args:
        config: Backbone hyper-parameters. Defaults to :class:`SchNetConfig`.
        rngs: Random number generators (keyword-only) seeding all weights.
    """

    def __init__(self, *, config: SchNetConfig | None = None, rngs: nnx.Rngs) -> None:
        """Build the atomic-number embedding, radial basis and interaction blocks."""
        super().__init__()
        self.config = config if config is not None else SchNetConfig()
        self.embedding = nnx.Embed(
            num_embeddings=_MAX_ATOMIC_NUMBER + 1,
            features=self.config.feature_dim,
            rngs=rngs,
        )
        self.radial_basis = BesselBasis(self.config.num_radial_basis, self.config.cutoff)
        self.interactions = nnx.List(
            [_InteractionBlock(self.config, rngs=rngs) for _ in range(self.config.num_interactions)]
        )

    def __call__(self, system: MolecularSystem, graph: tuple[Array, Array]) -> dict[str, Array]:
        """Run the SchNet message passing and return per-atom scalar features.

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
        features = self.embedding(system.atomic_numbers)
        for interaction in self.interactions:
            features = features + interaction(features, geometry, radial, envelope, system.n_atoms)
        return {"node_features": features}


__all__ = ["SchNet", "SchNetConfig", "shifted_softplus"]
