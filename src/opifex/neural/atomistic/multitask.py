r"""Multi-task / multi-fidelity energy head keyed by ``task_name`` (the UMA design).

A single shared backbone feeds many task-specific energy readouts; the active
readout is chosen at call time by a *task label* -- the design of the Meta-FAIR
Universal Models for Atoms (UMA; Wood et al. 2025, arXiv:2506.23971,
``../fairchem`` ``fairchem/core/models/base.py``: a ``ModuleDict`` of output heads
indexed by name, with the dataset/``task_name`` selecting the head and its own
reference-energy normalisation). It is the multi-head fine-tuning recipe of MACE
(``../mace`` ``mace/tools/multihead_tools.py`` and the per-head
``ScaleShiftBlock.forward(x, head)`` in ``mace/modules/blocks.py``), where each
head owns an independent affine ``scale``/``shift`` (per-dataset ``E0`` +
normaliser).

:class:`MultiTaskEnergyHead` holds one
:class:`~opifex.neural.atomistic.heads.energy.EnergyHead` per task -- each with its
*own* MLP parameters and its *own*
:class:`~opifex.neural.atomistic.scale_shift.AtomicScaleShift` (per-task ``E0`` +
normaliser). It satisfies the
:class:`~opifex.core.quantum.protocols.PropertyHead` contract
(``implemented_properties == ("energy",)``), so it drops directly into
:class:`~opifex.neural.atomistic.base.AtomisticModel` as the required ``"energy"``
head.

Task-selection mechanism (``jit``-safe by construction)
-------------------------------------------------------
The active task is a **Python-level string** passed through the standard
``PropertyHead.__call__`` ``embeddings`` mapping under the key
:data:`TASK_NAME_KEY` -- the protocol signature is unchanged. Because the label is
an ordinary Python value (not a JAX tracer), the dict lookup that selects *which*
per-task :class:`EnergyHead` runs is resolved at **trace time**: the chosen task
is effectively a static argument and the traced graph contains only that one
readout. Selection therefore never branches on a tracer, so the head is
``jit``/``grad``/``vmap`` clean for any *fixed* task (treat ``task_name`` as a
``static_argnum`` when jitting a function that varies it). When the key is absent
the :attr:`~MultiTaskEnergyHead.default_task` (the first registered task) is used.
"""

from __future__ import annotations

import logging
from typing import Final

from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.core.quantum.registry import register_property_head
from opifex.neural.atomistic.heads.energy import EnergyHead

# Eager (not TYPE_CHECKING): used as a runtime default-argument value below.
from opifex.neural.atomistic.scale_shift import AtomicScaleShift


_logger = logging.getLogger(__name__)

#: ``embeddings`` key carrying the active task label (a Python ``str``). The label
#: is static under ``jit`` -- it selects the per-task readout at trace time.
TASK_NAME_KEY: Final = "task_name"


@register_property_head("multitask_energy")
class MultiTaskEnergyHead(nnx.Module):
    """Task-conditioned energy readout: one :class:`EnergyHead` per task.

    Args:
        feature_dim: Width of the backbone's ``"node_features"`` embedding (shared
            by every per-task readout).
        task_names: Ordered task labels. The first is the
            :attr:`default_task` used when no label is supplied at call time. Must
            be non-empty with no duplicates.
        hidden_dim: Hidden width of each per-task MLP. Defaults to ``feature_dim``.
        scale_shifts: Optional mapping of task label to its
            :class:`AtomicScaleShift` (per-task ``E0`` + normaliser). Tasks absent
            from the mapping get :meth:`AtomicScaleShift.identity`.
        rngs: Random number generators (keyword-only). Each per-task MLP is seeded
            from a distinct fold of ``rngs`` so the readouts are independent.

    Raises:
        ValueError: If ``task_names`` is empty or contains duplicates.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        task_names: tuple[str, ...],
        hidden_dim: int | None = None,
        scale_shifts: dict[str, AtomicScaleShift] | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Build one independent :class:`EnergyHead` per task label."""
        super().__init__()
        if not task_names:
            raise ValueError("MultiTaskEnergyHead requires at least one task name.")
        if len(set(task_names)) != len(task_names):
            raise ValueError(f"task_names must be unique, got {task_names!r}.")
        resolved_shifts = scale_shifts or {}
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.default_task = task_names[0]
        heads: dict[str, EnergyHead] = {
            name: EnergyHead(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                scale_shift=resolved_shifts.get(name, AtomicScaleShift.identity()),
                rngs=rngs,
            )
            for name in task_names
        }
        self.task_heads = nnx.Dict(heads)
        _logger.debug("MultiTaskEnergyHead built with tasks %s", task_names)

    @property
    def implemented_properties(self) -> tuple[str, ...]:
        """This head emits the (task-conditioned) total ``"energy"``."""
        return ("energy",)

    @property
    def task_names(self) -> tuple[str, ...]:
        """The registered task labels (insertion order preserved)."""
        return tuple(self.task_heads.keys())

    def task_head(self, task_name: str) -> EnergyHead:
        """Return the per-task :class:`EnergyHead` registered under ``task_name``.

        Raises:
            KeyError: If ``task_name`` is not registered; the message lists the
                available task labels.
        """
        if task_name not in self.task_names:
            available = sorted(self.task_names)
            raise KeyError(f"Unknown task {task_name!r}. Available tasks: {available!r}.")
        return self.task_heads[task_name]

    def __call__(
        self,
        system: MolecularSystem,
        graph: tuple[Array, Array],
        embeddings: dict[str, Array],
    ) -> dict[str, Array]:
        """Route to the active task's readout and return its total ``"energy"``.

        Args:
            system: The molecular system (passed through to the per-task readout
                for its atom-count / scale-shift contract).
            graph: The ``(senders, receivers)`` edge index (passed through).
            embeddings: Must contain ``"node_features"``; may contain
                :data:`TASK_NAME_KEY` (a Python ``str``) selecting the active task.
                When absent, :attr:`default_task` is used.

        Returns:
            ``{"energy": scalar}`` from the selected task's readout (with that
            task's own ``E0`` + normaliser applied).
        """
        task_name = embeddings.get(TASK_NAME_KEY, self.default_task)
        if not isinstance(task_name, str):
            raise TypeError(
                f"{TASK_NAME_KEY!r} must be a Python str (static task label), "
                f"got {type(task_name).__name__}."
            )
        return self.task_head(task_name)(system, graph, embeddings)

    def with_task(
        self,
        task_name: str,
        *,
        feature_dim: int | None = None,
        hidden_dim: int | None = None,
        scale_shift: AtomicScaleShift | None = None,
        rngs: nnx.Rngs,
    ) -> MultiTaskEnergyHead:
        """Return this head with a new per-task readout added (UMA ``add_tasks``).

        Mirrors the fairchem ``HydraModel.add_tasks`` flow of attaching an
        inference/fine-tune head to an existing backbone (``../fairchem``
        ``fairchem/core/models/base.py``). Mutation is in place (NNX modules are
        mutable); the same instance is returned for chaining.

        Args:
            task_name: Label for the new task; must not already exist.
            feature_dim: Embedding width for the new readout. Defaults to the
                width the head was built with.
            hidden_dim: Hidden width for the new readout. Defaults to the head's.
            scale_shift: Per-task ``E0`` + normaliser. Defaults to identity.
            rngs: Random number generators seeding the new readout (keyword-only).

        Raises:
            ValueError: If ``task_name`` is already registered.
        """
        if task_name in self.task_names:
            raise ValueError(
                f"Task {task_name!r} already registered; tasks are {self.task_names!r}."
            )
        self.task_heads[task_name] = EnergyHead(
            feature_dim=feature_dim if feature_dim is not None else self.feature_dim,
            hidden_dim=hidden_dim if hidden_dim is not None else self.hidden_dim,
            scale_shift=scale_shift if scale_shift is not None else AtomicScaleShift.identity(),
            rngs=rngs,
        )
        _logger.debug("Added task %r; tasks now %s", task_name, self.task_names)
        return self

    def without_task(self, task_name: str) -> MultiTaskEnergyHead:
        """Return this head with ``task_name``'s readout removed.

        Args:
            task_name: Label of the task to drop.

        Raises:
            KeyError: If ``task_name`` is not registered.
            ValueError: If removing it would leave the head with no tasks.
        """
        if task_name not in self.task_names:
            available = sorted(self.task_names)
            raise KeyError(f"Unknown task {task_name!r}. Available tasks: {available!r}.")
        if len(self.task_names) == 1:
            raise ValueError("Cannot remove the last task; a head needs at least one task.")
        del self.task_heads[task_name]
        if self.default_task == task_name:
            self.default_task = self.task_names[0]
        _logger.debug("Removed task %r; tasks now %s", task_name, self.task_names)
        return self


__all__ = ["TASK_NAME_KEY", "MultiTaskEnergyHead"]
