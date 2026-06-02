r"""Assembled atomistic model: a backbone plus named property heads.

:class:`AtomisticModel` is the concrete shared-logic class behind every
interatomic potential in opifex. It implements the convergent
**backbone -> named heads** architecture of SchNetPack-2 / fairchem / MACE
(``00-research-landscape.md`` "library architecture"):

#. build the neighbour graph with the injected
   :class:`opifex.core.quantum.protocols.NeighborList`;
#. run the :class:`opifex.core.quantum.protocols.Backbone` to get per-atom
   embeddings;
#. run each :class:`opifex.core.quantum.protocols.PropertyHead` and merge the
   outputs into a single ``dict[str, Array]``.

The energy is computed once from the backbone embeddings. Conservative
force/stress heads need the energy *as a function of geometry*, so the model
injects two closures into the per-head ``embeddings`` dict: a positions-to-energy
closure (key :data:`~opifex.neural.atomistic.heads.forces.ENERGY_FN_KEY`) and a
symmetric-strain-to-energy closure (key
:data:`~opifex.neural.atomistic.heads.stress.STRAIN_ENERGY_FN_KEY`), following the
strain-displacement recipe of ``../mace`` (``mace/modules/utils.py``). Heads that
do not need them simply ignore those keys.

Concrete backbones (SchNet, PaiNN, NequIP, MACE) are **not** defined here -- they
plug into this base via the ``Backbone`` protocol and the family registries in
``opifex.core.quantum.registry``. This module is the reusable assembly only
(single responsibility: orchestration + the conservative-readout closures).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from dataclasses import replace
from typing import TYPE_CHECKING

from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.neural.atomistic.heads.energy import EnergyHead
from opifex.neural.atomistic.heads.forces import ENERGY_FN_KEY
from opifex.neural.atomistic.heads.stress import STRAIN_ENERGY_FN_KEY


if TYPE_CHECKING:
    from opifex.core.quantum.molecular_system import MolecularSystem
    from opifex.core.quantum.protocols import Backbone, NeighborList, PropertyHead


_ENERGY_HEAD_NAME = "energy"


class AtomisticModel(nnx.Module):
    """A backbone plus a dict of named property heads (the MLIP assembly).

    Args:
        backbone: Embedding producer satisfying the ``Backbone`` protocol.
        heads: Mapping of head name to ``PropertyHead``. Must include an
            ``"energy"`` head (:class:`EnergyHead`) -- the conservative
            force/stress heads differentiate its output.
        neighbor_list: Edge builder satisfying the ``NeighborList`` protocol.
        max_edges: Static upper bound on the number of edges (output size of the
            neighbour list under ``jit``).

    Raises:
        ValueError: If no ``"energy"`` head is supplied.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        heads: dict[str, PropertyHead],
        neighbor_list: NeighborList,
        max_edges: int,
    ) -> None:
        """Store the backbone, heads, neighbour-list builder and edge bound."""
        super().__init__()
        if _ENERGY_HEAD_NAME not in heads:
            raise ValueError(
                "AtomisticModel requires an 'energy' head; conservative "
                "force/stress heads differentiate its output."
            )
        self.backbone = backbone
        self.heads = nnx.Dict(heads)
        self.neighbor_list = neighbor_list
        self.max_edges = max_edges

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """The union of every property emitted by the configured heads."""
        properties: list[str] = []
        for head in self.heads.values():
            properties.extend(head.implemented_properties)
        return tuple(dict.fromkeys(properties))

    def _graph(self, system: MolecularSystem) -> tuple[Array, Array]:
        return self.neighbor_list(system, max_edges=self.max_edges)

    def _energy_at(self, system: MolecularSystem, graph: tuple[Array, Array]) -> Array:
        """Run the backbone and energy head, returning the scalar total energy."""
        embeddings = self.backbone(system, graph)
        energy_head = self.heads[_ENERGY_HEAD_NAME]
        return energy_head(system, graph, embeddings)["energy"]

    def _energy_of_positions(
        self, system: MolecularSystem, graph: tuple[Array, Array]
    ) -> Callable[[Array], Array]:
        """Build a ``positions -> energy`` closure for the conservative force head."""

        def energy_fn(positions: Array) -> Array:
            moved = replace(system, positions=positions)
            return self._energy_at(moved, graph)

        return energy_fn

    def _energy_of_strain(
        self, system: MolecularSystem, graph: tuple[Array, Array]
    ) -> Callable[[Array], Array]:
        """Build a ``strain -> energy`` closure (strain-displacement virial).

        Applies a symmetrised infinitesimal strain to both positions and cell --
        the ``../mace`` ``get_symmetric_displacement`` recipe -- so that the
        energy's strain-gradient at zero strain is the (symmetric) virial.
        """

        def strain_energy_fn(strain: Array) -> Array:
            symmetric = 0.5 * (strain + strain.T)
            strained_positions = system.positions + system.positions @ symmetric
            strained_cell = None if system.cell is None else system.cell + system.cell @ symmetric
            strained = replace(system, positions=strained_positions, cell=strained_cell)
            return self._energy_at(strained, graph)

        return strain_energy_fn

    def __call__(self, system: MolecularSystem) -> dict[str, Array]:
        """Run the model and return every configured property.

        Args:
            system: The molecular system to evaluate.

        Returns:
            A ``dict`` merging every head's named outputs.
        """
        graph = self._graph(system)
        embeddings = dict(self.backbone(system, graph))
        # Inject the geometry-to-energy closures the conservative heads need.
        embeddings[ENERGY_FN_KEY] = self._energy_of_positions(system, graph)  # type: ignore[assignment]
        embeddings[STRAIN_ENERGY_FN_KEY] = self._energy_of_strain(system, graph)  # type: ignore[assignment]
        outputs: dict[str, Array] = {}
        for head in self.heads.values():
            outputs.update(head(system, graph, embeddings))
        return outputs


__all__ = ["AtomisticModel", "EnergyHead"]
