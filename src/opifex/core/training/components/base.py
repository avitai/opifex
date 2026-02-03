"""Base protocols for training components.

This module defines the fundamental TrainingComponent protocol that allows
modular extension of the Trainer.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrainingComponent(Protocol):
    """Protocol for modular training components.

    Components can hook into various stages of the training loop to implement
    features like adaptive sampling, loss balancing, or curriculum learning.
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

    def on_batch_end(
        self, batch: int, state: Any, loss: float, metrics: dict[str, Any]
    ) -> None:
        """Called at the end of each batch."""
        ...


class BaseComponent:
    """Base class for components with no-op default implementations."""

    def on_epoch_begin(self, epoch: int, state: Any) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, epoch: int, state: Any, metrics: dict[str, Any]) -> None:
        """Called at the end of each epoch."""

    def on_batch_begin(self, batch: int, state: Any) -> None:
        """Called at the beginning of each batch."""

    def on_batch_end(
        self, batch: int, state: Any, loss: float, metrics: dict[str, Any]
    ) -> None:
        """Called at the end of each batch."""
