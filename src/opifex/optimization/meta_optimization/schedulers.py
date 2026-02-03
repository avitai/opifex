"""Adaptive learning rate scheduling for meta-optimization.

This module implements various adaptive learning rate scheduling strategies
including cosine annealing, performance-based adaptation, and quantum-aware
scheduling for scientific applications.

Author: Opifex Framework Team
Date: December 2024
License: MIT
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduling for meta-optimization.

    This class implements various adaptive learning rate scheduling strategies
    including cosine annealing, performance-based adaptation, and quantum-aware
    scheduling for scientific applications.

    Attributes:
        schedule_type: Type of scheduling algorithm
        initial_lr: Initial learning rate
        final_lr: Final learning rate (for annealing schedules)
        adaptation_period: Period for adaptation cycles
        warmup_steps: Number of warmup steps
        patience: Patience for performance-based adaptation
        factor: Factor for learning rate reduction
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        schedule_type: str = "cosine_annealing",
        initial_lr: float = 1e-3,
        final_lr: float = 1e-6,
        adaptation_period: int = 100,
        warmup_steps: int = 0,
        patience: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-8,
        **kwargs: Any,
    ):
        """Initialize adaptive learning rate scheduler.

        Args:
            schedule_type: Type of scheduling
                ('cosine_annealing', 'performance_based', 'quantum_aware')
            initial_lr: Initial learning rate
            final_lr: Final learning rate
            adaptation_period: Period for complete adaptation cycle
            warmup_steps: Number of warmup steps
            patience: Patience for performance-based adaptation
            factor: Reduction factor for learning rate
            min_lr: Minimum allowed learning rate
            **kwargs: Additional scheduler-specific parameters
        """
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.adaptation_period = adaptation_period
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr

        # Additional parameters for specific schedulers
        self.scf_threshold = kwargs.get("scf_threshold", 1e-6)
        self.energy_stability_factor = kwargs.get("energy_stability_factor", 0.95)

        # Performance tracking for adaptive schedules
        self._performance_history = []
        self._steps_since_improvement = 0
        self._best_performance = float("inf")

    def get_learning_rate(self, step: int) -> jax.Array:
        """Get learning rate for current step.

        Args:
            step: Current optimization step

        Returns:
            Learning rate for current step
        """
        step_array = jnp.asarray(step)

        if self.schedule_type == "cosine_annealing":
            return self._cosine_annealing_schedule(step_array)
        if self.schedule_type == "linear":
            return self._linear_schedule(step_array)
        if self.schedule_type == "exponential":
            return self._exponential_schedule(step_array)
        # Default to constant learning rate
        return jnp.array(self.initial_lr)

    def _cosine_annealing_schedule(self, step: jax.Array) -> jax.Array:
        """Compute cosine annealing learning rate schedule."""
        # Handle warmup
        if self.warmup_steps > 0:
            warmup_lr = self.initial_lr * step / self.warmup_steps
            step = jnp.maximum(0, step - self.warmup_steps)
            warmup_mask = step < self.warmup_steps
        else:
            warmup_lr = self.initial_lr
            warmup_mask = False

        # Cosine annealing
        progress = jnp.minimum(step / self.adaptation_period, 1.0)
        cosine_factor = 0.5 * (1 + jnp.cos(jnp.pi * progress))
        lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine_factor

        # Apply warmup if necessary
        lr = jnp.where(warmup_mask, warmup_lr, lr)

        return jnp.maximum(lr, self.min_lr)

    def _linear_schedule(self, step: jax.Array) -> jax.Array:
        """Compute linear learning rate schedule."""
        progress = jnp.minimum(step / self.adaptation_period, 1.0)
        lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress
        return jnp.maximum(lr, self.min_lr)

    def _exponential_schedule(self, step: jax.Array) -> jax.Array:
        """Compute exponential learning rate schedule."""
        progress = jnp.minimum(step / self.adaptation_period, 1.0)
        decay_rate = jnp.log(self.final_lr / self.initial_lr)
        lr = self.initial_lr * jnp.exp(decay_rate * progress)
        return jnp.maximum(lr, self.min_lr)

    def adapt_from_performance(self, loss_history: list[float]) -> float:
        """Adapt learning rate based on performance history.

        Args:
            loss_history: Recent loss values

        Returns:
            Adapted learning rate
        """
        if len(loss_history) < self.patience:
            return self.initial_lr

        # Check for improvement
        current_loss = loss_history[-1]

        if current_loss < self._best_performance:
            self._best_performance = current_loss
            self._steps_since_improvement = 0
        else:
            self._steps_since_improvement += 1

        # Reduce learning rate if no improvement
        if self._steps_since_improvement >= self.patience:
            return max(
                self.initial_lr * self.factor,
                self.min_lr,
            )

        return self.initial_lr

    def adapt_from_quantum_metrics(
        self, scf_errors: list[float], energy_changes: list[float]
    ) -> float:
        """Adapt learning rate based on quantum mechanical metrics.

        Args:
            scf_errors: SCF convergence errors
            energy_changes: Energy change magnitudes

        Returns:
            Quantum-adapted learning rate
        """
        if not scf_errors or not energy_changes:
            return self.initial_lr

        # Current SCF error and energy stability
        current_scf_error = scf_errors[-1]
        avg_energy_change = jnp.mean(jnp.array(energy_changes[-5:]))

        # Adaptive strategy based on SCF convergence
        if current_scf_error < self.scf_threshold:
            # SCF converged, can use higher learning rate
            lr_factor = 1.2
        elif current_scf_error > 10 * self.scf_threshold:
            # SCF struggling, reduce learning rate
            lr_factor = 0.5
        else:
            # Intermediate regime
            lr_factor = 1.0

        # Further adaptation based on energy stability
        if avg_energy_change < 1e-6:
            # Very stable, can increase learning rate
            lr_factor *= self.energy_stability_factor

        adapted_lr = self.initial_lr * lr_factor
        return float(jnp.clip(adapted_lr, self.min_lr, 10 * self.initial_lr))


__all__ = ["AdaptiveLearningRateScheduler"]
