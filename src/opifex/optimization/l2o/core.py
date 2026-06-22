"""Core learn-to-optimize abstractions: tasks and the optimizer interface.

A learned optimizer is meta-trained to minimise a *distribution* of objectives. The
objective is carried by a :class:`Task` (an ``init`` for the optimisee parameters plus a
``loss``), and a :class:`TaskFamily` samples tasks for meta-generalisation. This mirrors the
canonical design in Google's ``learned_optimization`` library
(``learned_optimization/tasks/base.py``); see Andrychowicz et al. 2016
(``arXiv:1606.04474``) for the original learning-to-learn formulation.

The key contrast with the previous opifex L2O code is that the objective lives *on the
task*: optimisers and meta-trainers close over ``task.loss`` rather than guessing a
placeholder. ``Task.normalizer`` maps a raw loss onto a comparable scale so a meta-loss
aggregated across differently-conditioned tasks is not dominated by the worst-scaled task
(``learned_optimization/tasks/base.py`` ``normalizer``).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import jax


if TYPE_CHECKING:
    from jaxtyping import Array, Float, PyTree


class Task(abc.ABC):
    """A single optimisation objective: ``init`` the params, evaluate ``loss``.

    Subclasses implement :meth:`init` (sample the initial optimisee parameters) and
    :meth:`loss` (a scalar objective). :meth:`normalizer` defaults to the identity and
    should be overridden when meta-training mixes tasks of very different loss scales.
    """

    @abc.abstractmethod
    def init(self, key: Array) -> PyTree:
        """Sample the initial optimisee parameter pytree."""

    @abc.abstractmethod
    def loss(self, params: PyTree, key: Array) -> Float[Array, ""]:
        """Return the scalar objective at ``params`` (``key`` for stochastic tasks)."""

    def normalizer(self, loss: Float[Array, ""]) -> Float[Array, ""]:
        """Map a raw loss onto a comparable scale (identity by default)."""
        return loss

    def loss_and_grad(self, params: PyTree, key: Array) -> tuple[Float[Array, ""], PyTree]:
        """Convenience: value-and-gradient of :meth:`loss` w.r.t. ``params``."""
        return jax.value_and_grad(self.loss)(params, key)


class TaskFamily(abc.ABC):
    """A distribution over :class:`Task` instances for meta-generalisation."""

    @abc.abstractmethod
    def sample(self, key: Array) -> Task:
        """Draw a task from the family."""


class _SingleTaskFamily(TaskFamily):
    """A degenerate family that always returns the same task."""

    def __init__(self, task: Task) -> None:
        """Wrap a fixed ``task`` as a one-element family."""
        self._task = task

    def sample(self, key: Array) -> Task:  # noqa: ARG002 - fixed task ignores the key
        """Return the wrapped task regardless of ``key``."""
        return self._task


def single_task_to_family(task: Task) -> TaskFamily:
    """Lift a fixed :class:`Task` into a :class:`TaskFamily` (mirrors the reference).

    Meta-training always consumes a family; this adapts a single task for the case where
    no task distribution is needed (e.g. overfitting a learned optimiser to one problem).
    """
    return _SingleTaskFamily(task)


__all__ = ["Task", "TaskFamily", "single_task_to_family"]
