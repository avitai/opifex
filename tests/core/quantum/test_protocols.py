r"""Tests for the atomistic-model extension-seam protocols.

Each protocol is ``@runtime_checkable`` so a structurally-conforming dummy class
satisfies ``isinstance`` without nominal inheritance (dependency inversion). The
checks confirm the contract surface (method names / properties) is what callers
depend on, and that the default ``NeighborList`` adapter delegates to
``radius_graph``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.protocols import (
    AtomisticModel,
    Backbone,
    NeighborList,
    PropertyHead,
    Space,
)


def _make_system() -> MolecularSystem:
    return MolecularSystem(
        atomic_numbers=jnp.asarray([1, 1]),
        positions=jnp.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )


class _DummyNeighborList:
    def __call__(self, system: MolecularSystem, *, max_edges: int) -> tuple[Array, Array]:
        return jnp.asarray([0]), jnp.asarray([1])


class _DummyBackbone:
    def __call__(self, system: MolecularSystem, graph: tuple[Array, Array]) -> dict[str, Array]:
        return {"node_features": jnp.zeros((system.n_atoms, 4))}


class _DummyHead:
    @property
    def implemented_properties(self) -> tuple[str, ...]:
        return ("energy",)

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        return {"energy": jnp.asarray(0.0)}


class _DummyModel:
    @property
    def implemented_properties(self) -> tuple[str, ...]:
        return ("energy",)

    def __call__(self, system: MolecularSystem) -> dict[str, Array]:
        return {"energy": jnp.asarray(0.0)}


class _DummySpace:
    def displacement(self, ra: Array, rb: Array) -> Array:
        return ra - rb

    def shift(self, position: Array, delta: Array) -> Array:
        return position + delta


class TestRuntimeCheckable:
    def test_neighbor_list_protocol(self) -> None:
        assert isinstance(_DummyNeighborList(), NeighborList)

    def test_backbone_protocol(self) -> None:
        assert isinstance(_DummyBackbone(), Backbone)

    def test_property_head_protocol(self) -> None:
        assert isinstance(_DummyHead(), PropertyHead)

    def test_atomistic_model_protocol(self) -> None:
        assert isinstance(_DummyModel(), AtomisticModel)

    def test_space_protocol(self) -> None:
        assert isinstance(_DummySpace(), Space)

    def test_non_conforming_is_rejected(self) -> None:
        assert not isinstance(object(), Backbone)


class TestRadiusNeighborList:
    def test_builds_edges_within_cutoff(self) -> None:
        """The default neighbour list delegates to ``radius_graph``."""
        from opifex.core.quantum.protocols import RadiusNeighborList

        neighbor_list = RadiusNeighborList(cutoff=2.0)
        assert isinstance(neighbor_list, NeighborList)
        senders, receivers = neighbor_list(_make_system(), max_edges=8)
        pairs = {
            (int(s), int(r))
            for s, r in zip(senders.tolist(), receivers.tolist(), strict=True)
            if s >= 0
        }
        assert pairs == {(0, 1), (1, 0)}

    def test_cutoff_excludes_far_atoms(self) -> None:
        from opifex.core.quantum.protocols import RadiusNeighborList

        neighbor_list = RadiusNeighborList(cutoff=1.0)
        senders, _ = neighbor_list(_make_system(), max_edges=8)
        assert int(jnp.sum(senders >= 0)) == 0
