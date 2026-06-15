r"""Extension-seam protocols for the atomistic-model subsystem.

These ``@runtime_checkable`` protocols are the swappable seams of the
backbone-to-named-heads architecture shared by SchNetPack-2, fairchem and MACE
(see ``00-research-landscape.md`` "library architecture"). Domain code depends on
these abstractions, never on concrete implementations or infrastructure
(dependency inversion / Open-Closed).

The seams:

* :class:`Space` -- injects boundary conditions as a ``displacement``/``shift``
  pair (the ``../jax-md/jax_md/space.py`` pattern; Schoenholz & Cubuk 2020).
* :class:`NeighborList` -- turns a :class:`MolecularSystem` into the
  ``(senders, receivers)`` edge index within a cutoff. The default
  :class:`RadiusNeighborList` adapter delegates to
  :func:`opifex.neural.equivariant.radius_graph`.
* :class:`Backbone` -- computes per-atom (and optionally per-edge) embeddings.
* :class:`PropertyHead` -- maps embeddings to one or more named properties.
* :class:`AtomisticModel` -- the assembled ``system -> dict[str, Array]`` model
  declaring its :attr:`AtomisticModel.implemented_properties`.

Concrete backbones (SchNet / PaiNN / NequIP / MACE) plug into this surface via the
:class:`Backbone` protocol and the family registries in
``opifex.core.quantum.registry``; they are deliberately *not* implemented here --
this module defines only the contracts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.neural.equivariant import radius_graph


@runtime_checkable
class Space(Protocol):
    """Boundary-condition seam: a ``displacement``/``shift`` function pair.

    Mirrors the ``(displacement_fn, shift_fn)`` contract of
    ``../jax-md/jax_md/space.py``: :meth:`displacement` computes the (possibly
    minimum-image) separation between two points and :meth:`shift` advances a
    position by a displacement (wrapping back into the cell when periodic).
    """

    def displacement(self, ra: Array, rb: Array) -> Array:
        """Return the separation ``ra - rb`` under these boundary conditions."""
        ...

    def shift(self, position: Array, delta: Array) -> Array:
        """Return ``position`` advanced by ``delta`` under these boundary conditions."""
        ...


@runtime_checkable
class NeighborList(Protocol):
    """Edge-construction seam: build ``(senders, receivers)`` within a cutoff.

    Implementations turn a :class:`MolecularSystem` into a statically-shaped edge
    index (``max_edges`` long, padded with ``-1``) suitable for message passing.
    """

    def __call__(self, system: MolecularSystem, *, max_edges: int) -> tuple[Array, Array]:
        """Return ``(senders, receivers)`` integer arrays of length ``max_edges``."""
        ...


@runtime_checkable
class Backbone(Protocol):
    """Embedding seam: ``(system, graph) -> per-atom/edge embeddings``.

    The returned dict maps embedding names (e.g. ``"node_features"``) to arrays.
    Concrete backbones (SchNet / PaiNN / NequIP / MACE) implement this protocol.
    """

    def __call__(self, system: MolecularSystem, graph: tuple[Array, Array]) -> dict[str, Array]:
        """Return named embedding arrays for the given system and edge index."""
        ...


@runtime_checkable
class PropertyHead(Protocol):
    """Readout seam: ``(system, graph, embeddings) -> named properties``.

    Each head owns exactly one property family (energy, forces, stress, ...) and
    declares it via :attr:`implemented_properties` (single responsibility).
    """

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """Names of the properties this head emits."""
        ...

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        """Return the named property outputs for this head."""
        ...


@runtime_checkable
class AtomisticModel(Protocol):
    """Assembled-model seam: ``system -> dict[str, Array]``.

    The top-level contract a calculator / trainer depends on. It declares the set
    of properties it can produce via :attr:`implemented_properties`.
    """

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """Names of every property the model produces."""
        ...

    def __call__(self, system: MolecularSystem) -> dict[str, Array]:
        """Return the predicted properties for ``system``."""
        ...


class RadiusNeighborList:
    """Default :class:`NeighborList` adapter delegating to ``radius_graph``.

    A thin boundary adapter over
    :func:`opifex.neural.equivariant.radius_graph`: it reuses the dense
    pairwise-distance edge builder rather than duplicating the cutoff logic
    (DRY). Periodic minimum-image neighbour search is a future variant that will
    accept a :class:`Space`.

    Args:
        cutoff: Connection radius ``r_c`` (positive), in the system's length
            units (Bohr for :class:`MolecularSystem`).
        self_loops: Whether to include ``(i, i)`` self edges. Default ``False``.
    """

    def __init__(self, cutoff: float, *, self_loops: bool = False) -> None:
        """Store the cutoff radius and self-loop policy."""
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.cutoff = cutoff
        self.self_loops = self_loops

    def __call__(self, system: MolecularSystem, *, max_edges: int) -> tuple[Array, Array]:
        """Build the radius-graph edge index for ``system``.

        Args:
            system: The molecular system whose positions define the point cloud.
            max_edges: Static upper bound on the number of edges (output length).

        Returns:
            A ``(senders, receivers)`` pair of integer arrays of shape
            ``(max_edges,)``, padded with ``-1``.
        """
        return radius_graph(
            system.positions,
            self.cutoff,
            max_edges=max_edges,
            self_loops=self.self_loops,
        )


__all__ = [
    "AtomisticModel",
    "Backbone",
    "NeighborList",
    "PropertyHead",
    "RadiusNeighborList",
    "Space",
]
