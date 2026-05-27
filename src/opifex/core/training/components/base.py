"""Base protocols for training callbacks.

Defines the ``TrainingCallback`` epoch/batch lifecycle-hook contract used
by ``Trainer.train`` and ``opifex.core.solver.strategies``. The
:class:`opifex.core.training.components.lifecycle.TrainingComponent` is a
separate ``setup``/``step``/``cleanup`` abstraction used by
``CheckpointComponent``, ``RecoveryComponent``, etc. — keeping the
callback / component names distinct removes the previous
duplicate-class collision.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrainingCallback(Protocol):
    """Protocol for epoch / batch training callbacks.

    Implementations hook into the four loop phases below to drive features
    like adaptive sampling, loss balancing, or curriculum learning. See
    :class:`components.lifecycle.TrainingComponent` for the orthogonal
    setup/step/cleanup contract used by stateful side-effect components.
    """

    def on_epoch_begin(self, epoch: int, state: Any) -> None:
        """Called at the beginning of each epoch."""
        ...

    def on_epoch_end(self, epoch: int, state: Any, metrics: dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        ...

    def on_batch_begin(self, batch: int, state: Any) -> None:
        """Called at the beginning of each batch."""
        ...

    def on_batch_end(self, batch: int, state: Any, loss: float, metrics: dict[str, Any]) -> None:
        """Called at the end of each batch."""
        ...


class BaseCallback:
    """Base class for callbacks with no-op default implementations."""

    def on_epoch_begin(self, epoch: int, state: Any) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, epoch: int, state: Any, metrics: dict[str, Any]) -> None:
        """Called at the end of each epoch."""

    def on_batch_begin(self, batch: int, state: Any) -> None:
        """Called at the beginning of each batch."""

    def on_batch_end(self, batch: int, state: Any, loss: float, metrics: dict[str, Any]) -> None:
        """Called at the end of each batch."""
