"""High-level learn-to-optimize engine: meta-train, apply, benchmark, persist.

A thin orchestration over the L2O building blocks — it meta-trains a :class:`LearnedOptimizer`
on a :class:`TaskFamily` with PES, applies the trained optimiser to new tasks, benchmarks it
honestly against a tuned classical baseline on held-out tasks, and serialises the meta-learned
parameters ``theta``. There is no hidden objective or fabricated speedup: every task carries its
own loss and every reported number is measured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax  # noqa: TC002 — kept eager per project convention
import orbax.checkpoint as ocp

from opifex.optimization.l2o.baselines import loss_curve
from opifex.optimization.l2o.benchmark import benchmark_on_held_out_tasks
from opifex.optimization.l2o.meta_train import meta_train


if TYPE_CHECKING:
    from pathlib import Path

    from jaxtyping import PyTree

    from opifex.optimization.l2o.core import Task, TaskFamily
    from opifex.optimization.l2o.learned import LearnedOptimizer


class L2OEngine:
    """Meta-train a learned optimiser on a task family, then apply/benchmark/persist it."""

    def __init__(self, learned_optimizer: LearnedOptimizer, task_family: TaskFamily) -> None:
        """Bind the learned optimiser and the task family it is meta-trained on."""
        self.learned_optimizer = learned_optimizer
        self.task_family = task_family
        self.theta: PyTree | None = None

    def meta_train(self, key: jax.Array, **kwargs: object) -> jax.Array:
        """Meta-train the optimiser (PES); store ``theta`` and return the loss curve."""
        self.theta, losses = meta_train(
            self.learned_optimizer,
            self.task_family,
            key,
            **kwargs,  # type: ignore[arg-type]
        )
        return losses

    def _require_theta(self) -> PyTree:
        """Return the trained ``theta`` or raise if meta-training has not been run."""
        if self.theta is None:
            raise RuntimeError("Call meta_train() (or load_theta()) before using the optimiser.")
        return self.theta

    def optimize(self, task: Task, start: PyTree, *, num_steps: int, key: jax.Array) -> jax.Array:
        """Apply the trained optimiser to ``task`` from ``start``; return its loss curve."""
        optimizer = self.learned_optimizer.opt_fn(self._require_theta())
        return loss_curve(optimizer, task, start, num_steps=num_steps, key=key)

    def benchmark(self, key: jax.Array, **kwargs: object) -> dict[str, jax.Array]:
        """Benchmark the trained optimiser against a tuned baseline on held-out tasks."""
        return benchmark_on_held_out_tasks(
            self.learned_optimizer,
            self._require_theta(),
            self.task_family,
            key,
            **kwargs,  # type: ignore[arg-type]
        )

    def save_theta(self, directory: Path) -> None:
        """Serialise the trained ``theta`` to ``directory`` with Orbax."""
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(directory.resolve(), self._require_theta())

    def load_theta(self, directory: Path, template: PyTree) -> PyTree:
        """Restore ``theta`` from ``directory`` (``template`` gives the target structure)."""
        with ocp.StandardCheckpointer() as checkpointer:
            self.theta = checkpointer.restore(directory.resolve(), template)
        return self.theta


__all__ = ["L2OEngine"]
