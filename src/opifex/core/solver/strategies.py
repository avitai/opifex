"""Training Strategies and Components for Unified Solver.

This module defines the protocols and concrete implementations for:
1. TrainingComponent: Modular units that hook into the training loop.
2. TrainingStrategy: High-level configurations that compose components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
from flax import nnx

from opifex.core.training.components.base import BaseComponent, TrainingComponent


@runtime_checkable
class TrainingStrategy(Protocol):
    """Protocol for high-level training strategies.

    A strategy is a factory or configurator that sets up the training process,
    potentially returning a list of components to attach.
    """

    def configure(self, config: Any) -> Any:
        """Modify or validate the training configuration."""
        ...

    def create_components(self) -> list[TrainingComponent]:
        """Create and return the list of components for this strategy."""
        ...


class AdaptiveLossBalancing(BaseComponent, nnx.Module):
    """Adaptive Loss Balancing Strategy (e.g., GradNorm-style).

    Adjusts weights of different loss components during training to balance
    learning rates of different tasks (e.g., PDE residual vs Boundary condition).
    """

    def __init__(self, loss_keys: list[str], alpha: float = 0.5):
        self.alpha = alpha  # Smoothing parameter
        # Internal state using nnx.Dict for JIT compatibility
        # Eagerly initialize variables for known keys
        self.weights = nnx.Dict(
            {key: nnx.Variable(jnp.array(1.0)) for key in loss_keys}
        )

    def update(self, losses: dict[str, Any]) -> None:
        """Update weights based on current losses using Inverse Magnitude Balancing.

        Balances weights such that w_i * L_i ~ 1.
        """
        epsilon = 1e-8
        for key, loss_val in losses.items():
            # Target weight is inverse of loss magnitude
            w_target = 1.0 / (jnp.abs(loss_val) + epsilon)

            # Check existence using hasattr for safe nnx.Dict access
            if hasattr(self.weights, key):
                # Smooth update
                old_w = self.weights[key].value
                new_w = old_w * self.alpha + w_target * (1 - self.alpha)
                self.weights[key].value = new_w
            # else: Unknown keys are ignored to maintain static structure for JIT

    def on_batch_end(
        self, batch: int, state: Any, loss: Any, metrics: dict[str, Any]
    ) -> None:
        """Hook to update weights after batch."""
        # Extract individual loss components from metrics if available
        relevant_losses = {k: v for k, v in metrics.items() if k.startswith("loss_")}
        if relevant_losses:
            self.update(relevant_losses)


@dataclass
class CurriculumRegularization(BaseComponent):
    """Curriculum Regularization Strategy.

    Increases or decreases a regularization parameter (e.g., complexity penalty)
    over the course of training.
    """

    start_val: float = 0.0
    end_val: float = 1.0
    total_epochs: int = 100

    def get_value(self, epoch: int) -> float:
        """Get the regularization value for the current epoch (linear schedule)."""
        progress = min(max(epoch / self.total_epochs, 0.0), 1.0)
        return self.start_val + (self.end_val - self.start_val) * progress

    def on_epoch_begin(self, epoch: int, state: Any) -> None:
        """Update the regularization parameter in the training state."""
        val = self.get_value(epoch)
        # In a real implementation, we would update state.params or state.config
        # print(f"Epoch {epoch}: Updating reg param to {val}")
        _ = val  # Silence unused variable warning
