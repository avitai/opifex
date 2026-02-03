"""IncrementalTrainer - Simple incremental training for neural operators.

This module provides a focused implementation of incremental training
with gradient-based mode expansion, designed to pass the comprehensive
test suite and enable neuraloperator examples reproduction.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx


class IncrementalTrainer:
    """Simple incremental training for neural operators.

    Focused solely on incremental mode expansion with gradient analysis.
    Designed following test-driven development principles.
    """

    def __init__(self, model: nnx.Module, rngs: nnx.Rngs):
        """Initialize incremental trainer.

        Args:
            model: Neural operator model (must be FourierNeuralOperator or similar)
            rngs: Random number generators for JAX

        Raises:
            TypeError: If model is not an nnx.Module or rngs is not nnx.Rngs
        """
        if not isinstance(model, nnx.Module):
            raise TypeError("Model must be an nnx.Module")
        if not isinstance(rngs, nnx.Rngs):
            raise TypeError("rngs must be nnx.Rngs")

        self.model = model
        self.rngs = rngs
        self.current_modes = self._get_initial_modes()

        # Configuration for gradient analysis
        self.variance_threshold = 1.0

        # Initialize optimizer for training steps
        # Initialize optimizer for training steps
        self.optimizer = nnx.Optimizer(
            model, optax.adam(learning_rate=1e-3), wrt=nnx.Param
        )

    def _get_initial_modes(self) -> tuple[int, ...]:
        """Get initial modes from the model."""
        if hasattr(self.model, "modes"):
            # FNO stores modes as a single integer, convert to tuple for consistency
            modes = getattr(self.model, "modes", (4, 4))  # type: ignore[attr-defined]
            if isinstance(modes, int):
                return (modes, modes)
            return modes
        # Default fallback
        return (4, 4)

    def should_expand_modes(self, gradients: Any) -> bool:
        """Simple gradient variance analysis for mode expansion.

        Args:
            gradients: Gradient dictionary from training step

        Returns:
            bool: True if modes should be expanded, False otherwise
        """
        if not gradients:
            return False

        variance = self._compute_gradient_variance(gradients)
        return self._should_expand_based_on_variance(variance)

    def _compute_gradient_variance(self, gradients: Any) -> float:
        """Compute gradient variance for expansion decision.

        Args:
            gradients: Gradient dictionary

        Returns:
            float: Computed variance value
        """
        total_variance = 0.0
        count = 0

        def compute_variance_recursive(grad_dict):
            nonlocal total_variance, count

            if isinstance(grad_dict, dict):
                for value in grad_dict.values():
                    compute_variance_recursive(value)
            elif isinstance(grad_dict, jnp.ndarray):
                variance = jnp.var(grad_dict)
                total_variance += float(variance)
                count += 1

        compute_variance_recursive(gradients)

        if count == 0:
            return 0.0
        return total_variance / count

    def _should_expand_based_on_variance(self, variance: float) -> bool:
        """Check if variance exceeds threshold for expansion.

        Args:
            variance: Computed gradient variance

        Returns:
            bool: True if should expand, False otherwise
        """
        return variance > self.variance_threshold

    def expand_modes(self, new_modes: tuple[int, ...]) -> None:
        """Expand model modes to new configuration.

        Args:
            new_modes: New mode configuration

        Raises:
            ValueError: If new_modes are invalid (smaller than current or negative)
        """
        # Validate new modes
        if any(new_mode < 0 for new_mode in new_modes):
            raise ValueError("Modes cannot be negative")

        if len(new_modes) != len(self.current_modes):
            raise ValueError("New modes must have same dimensionality as current modes")

        if any(
            new_mode < current_mode
            for new_mode, current_mode in zip(
                new_modes, self.current_modes, strict=False
            )
        ):
            raise ValueError("New modes must be greater than or equal to current modes")

        # If modes are the same, no change needed
        if new_modes == self.current_modes:
            return

        # Update current modes
        self.current_modes = new_modes

        # Update model if it has modes attribute
        if hasattr(self.model, "modes"):
            # FNO expects a single integer for modes, use the first value
            self.model.modes = new_modes[0]  # type: ignore[attr-defined]

    def train_step(self, x: jax.Array, y: jax.Array) -> float:
        """Single training step with incremental capabilities.

        Args:
            x: Input data
            y: Target data

        Returns:
            float: Computed loss value

        Raises:
            ValueError: If input shapes are invalid
            TypeError: If inputs are not JAX arrays
        """
        # Validate inputs
        if not isinstance(x, jax.Array):
            raise TypeError("x must be a JAX array")
        if not isinstance(y, jax.Array):
            raise TypeError("y must be a JAX array")

        # Check shape compatibility (basic validation)
        if x.shape[0] != y.shape[0]:
            raise ValueError("Batch sizes of x and y must match")

        # Define loss function
        @nnx.value_and_grad
        def loss_fn(model):
            predictions = model(x)  # type: ignore[arg-type]
            return jnp.mean((predictions - y) ** 2)

        # Compute loss and gradients
        loss, grads = loss_fn(self.model)

        # Update model parameters
        self.optimizer.update(self.model, grads)

        return loss
