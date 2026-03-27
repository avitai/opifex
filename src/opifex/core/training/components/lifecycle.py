"""Base lifecycle component for training pipeline composition.

Provides the setup/step/cleanup interface used by CheckpointComponent,
MixedPrecisionComponent, RecoveryComponent, and FlexibleOptimizerFactory.

This module exists as a separate file to avoid circular imports between
``components/__init__.py`` and ``components/recovery.py``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from flax import nnx


class TrainingComponent:
    """Base class for all training components.

    Provides lifecycle methods (setup, step, cleanup) for component composition.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the training component.

        Args:
            config: Component-specific configuration dictionary
        """
        self.config = config if config is not None else {}
        self.name = self.__class__.__name__

    def setup(self, model: nnx.Module, training_state: Any) -> None:
        """Setup the component with model and training state.

        Args:
            model: The neural network model
            training_state: Current training state
        """
        # Base implementation does nothing

    def step(self, model: nnx.Module, training_state: Any) -> Any | None:
        """Execute component logic for current training step.

        Args:
            model: The neural network model
            training_state: Current training state

        Returns:
            Optional dict with step information, or None
        """
        # Base implementation returns None
        return None

    def cleanup(self) -> None:
        """Cleanup resources when component is no longer needed."""
        # Base implementation does nothing
