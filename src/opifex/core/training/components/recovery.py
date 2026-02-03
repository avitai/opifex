"""Error recovery and stability management for training.

This module provides production-grade error handling and recovery mechanisms
for scientific machine learning training.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.core.training.components import TrainingComponent


if TYPE_CHECKING:
    from flax import nnx

    from opifex.core.training.monitoring.metrics import TrainingState


class ErrorRecoveryManager(TrainingComponent):
    """Production-grade error handling and recovery mechanisms."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize error recovery manager.

        Args:
            config: Configuration including recovery strategies and thresholds
        """
        super().__init__(config)
        self.max_retries = self.config.get("max_retries", 3)
        self.checkpoint_on_error = self.config.get("checkpoint_on_error", True)
        self.gradient_clip_threshold = self.config.get("gradient_clip_threshold", 10.0)
        self.loss_explosion_threshold = self.config.get("loss_explosion_threshold", 1e6)
        self.recovery_attempts = 0
        self.last_stable_state: TrainingState | None = None

    def setup(self, model: nnx.Module, training_state: TrainingState) -> None:
        """Setup error recovery with initial stable state."""
        self.last_stable_state = training_state
        self.recovery_attempts = 0

    def check_training_stability(
        self, loss: float, grads: Any, training_state: TrainingState
    ) -> tuple[bool, str | None]:
        """Check if training is stable and suggest recovery if needed.

        Args:
            loss: Current training loss
            grads: Current gradients
            training_state: Current training state

        Returns:
            Tuple of (is_stable, recovery_suggestion)
        """
        # Check for loss explosion
        if loss > self.loss_explosion_threshold:
            return False, "loss_explosion"

        # Check for gradient explosion
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))
        )
        if grad_norm > self.gradient_clip_threshold:
            return False, "gradient_explosion"

        # Check for NaN values
        if jnp.isnan(loss):
            return False, "nan_loss"

        if any(jnp.any(jnp.isnan(g)) for g in jax.tree_util.tree_leaves(grads)):
            return False, "nan_gradients"

        # Training appears stable
        return True, None

    def apply_gradient_clipping(self, grads: Any) -> Any:
        """Apply gradient clipping for stability.

        Args:
            grads: Raw gradients

        Returns:
            Clipped gradients
        """
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))
        )

        if grad_norm > self.gradient_clip_threshold:
            clip_factor = self.gradient_clip_threshold / grad_norm
            return jax.tree_util.tree_map(lambda g: g * clip_factor, grads)

        return grads

    def recover_from_instability(
        self, issue_type: str, training_state: TrainingState
    ) -> TrainingState:
        """Attempt recovery from training instability.

        Args:
            issue_type: Type of instability detected
            training_state: Current (potentially unstable) training state

        Returns:
            Recovered training state
        """
        self.recovery_attempts += 1

        if self.recovery_attempts > self.max_retries:
            raise RuntimeError(
                f"Maximum recovery attempts ({self.max_retries}) exceeded. "
                f"Training failed due to persistent {issue_type}."
            )

        # Recovery strategies based on issue type
        if issue_type in ["loss_explosion", "gradient_explosion"]:
            # Reduce learning rate and restore last stable state
            if self.last_stable_state is not None:
                training_state = self.last_stable_state
                # Reduce learning rate by factor of 2
                current_lr = self.config.get("learning_rate", 1e-3)
                new_lr = current_lr * 0.5
                training_state.recovery_state["reduced_lr"] = new_lr

        elif (
            issue_type in ["nan_loss", "nan_gradients"]
            and self.last_stable_state is not None
        ):
            # Reinitialize problematic parameters
            training_state = self.last_stable_state
            training_state.recovery_state["reinitialized"] = True

        return training_state

    def update_stable_state(self, training_state: TrainingState) -> None:
        """Update the last known stable training state."""
        self.last_stable_state = training_state
        self.recovery_attempts = 0  # Reset recovery attempts on stable training
