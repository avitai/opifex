"""Integrated meta-optimization system.

This module provides the main MetaOptimizer class that integrates
learn-to-optimize algorithms, adaptive learning rate scheduling,
warm-starting strategies, and performance monitoring.

Author: Opifex Framework Team
Date: December 2024
License: MIT
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from flax import nnx

    from opifex.core.training.config import MetaOptimizerConfig

import jax
import jax.numpy as jnp
import optax

from opifex.optimization.meta_optimization.monitoring import PerformanceMonitor
from opifex.optimization.meta_optimization.neural_learner import LearnToOptimize
from opifex.optimization.meta_optimization.schedulers import (
    AdaptiveLearningRateScheduler,
)
from opifex.optimization.meta_optimization.warm_starting import WarmStartingStrategy


class MetaOptimizer:
    """Integrated meta-optimization system.

    This class provides a complete meta-optimization system that integrates
    learn-to-optimize algorithms, adaptive learning rate scheduling,
    warm-starting strategies, and performance monitoring.

    Attributes:
        config: Meta-optimizer configuration
        l2o_engine: Learn-to-optimize engine
        learning_rate_scheduler: Adaptive learning rate scheduler
        warm_start_strategy: Warm-starting strategy
        performance_monitor: Performance monitoring system
        current_step: Current optimization step
    """

    def __init__(
        self,
        config: MetaOptimizerConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize integrated meta-optimizer system.

        Args:
            config: Meta-optimizer configuration
            rngs: Random number generators
        """
        self.config = config
        self.current_step = 0

        # Initialize components based on configuration
        if config.meta_algorithm == "l2o":
            self.l2o_engine = LearnToOptimize(
                meta_network_layers=[128, 64, 32],
                base_optimizer=config.base_optimizer,
                meta_learning_rate=config.meta_learning_rate,
                quantum_aware=config.quantum_aware,
                scf_integration=config.scf_adaptation,
                rngs=rngs,
            )
        else:
            self.l2o_engine = None

        # Adaptive learning rate scheduler
        self.learning_rate_scheduler = AdaptiveLearningRateScheduler(
            schedule_type="cosine_annealing",
            initial_lr=config.meta_learning_rate,
        )

        # Warm-starting strategy
        self.warm_start_strategy = WarmStartingStrategy(
            strategy_type=config.warm_start_strategy,
        )

        # Performance monitoring
        if config.performance_tracking:
            self.performance_monitor = PerformanceMonitor(
                metrics=["loss", "gradient_norm", "learning_rate"],
                quantum_aware=config.quantum_aware,
                analytics_enabled=True,
            )
        else:
            self.performance_monitor = None

        # Base optimizer
        self._create_base_optimizer()

    def _create_base_optimizer(self):
        """Create base optimizer based on configuration."""
        lr = self.config.meta_learning_rate

        if self.config.base_optimizer == "adam":
            self.base_optimizer = optax.adam(lr)
        elif self.config.base_optimizer == "sgd":
            self.base_optimizer = optax.sgd(lr)
        elif self.config.base_optimizer == "rmsprop":
            self.base_optimizer = optax.rmsprop(lr)
        elif self.config.base_optimizer == "adamw":
            self.base_optimizer = optax.adamw(lr)
        else:
            self.base_optimizer = optax.adam(lr)  # Default

    def init_optimizer_state(self, params: jax.Array) -> Any:
        """Initialize optimizer state.

        Args:
            params: Initial parameters

        Returns:
            Initial optimizer state
        """
        return self.base_optimizer.init(params)

    def step(
        self,
        loss_fn: Callable[[jax.Array], jax.Array],
        params: jax.Array,
        opt_state: Any,
        step: int,
    ) -> tuple[jax.Array, Any, dict[str, Any]]:
        """Perform single meta-optimization step.

        Args:
            loss_fn: Loss function to optimize
            params: Current parameters
            opt_state: Current optimizer state
            step: Current step number

        Returns:
            Tuple of (new_params, new_opt_state, meta_info)
        """
        self.current_step = step

        # Compute gradient
        loss_value = loss_fn(params)
        gradient = jax.grad(loss_fn)(params)
        gradient_norm = jnp.linalg.norm(gradient)

        # Get current learning rate
        current_lr = self.learning_rate_scheduler.get_learning_rate(step)

        # Update base optimizer with current learning rate
        updated_optimizer = optax.scale(-float(current_lr))

        if self.config.meta_algorithm == "l2o" and self.l2o_engine is not None:
            # Use L2O for parameter update
            previous_updates = jnp.zeros((0, params.size))  # Simplified
            meta_update = self.l2o_engine.compute_update(gradient, previous_updates)

            # Ensure meta_update matches parameter shape
            if meta_update.size > params.size:
                meta_update = meta_update[: params.size]
            elif meta_update.size < params.size:
                padding = jnp.zeros(params.size - meta_update.size)
                meta_update = jnp.concatenate([meta_update, padding])
            meta_update = meta_update.reshape(params.shape)

            # Apply meta-update using optax pattern
            new_params = optax.apply_updates(
                params, -meta_update
            )  # Negative for gradient descent
            new_opt_state = opt_state  # L2O manages its own state

            meta_gradient_norm = jnp.linalg.norm(meta_update)
        else:
            # Use standard optimizer
            updates, new_opt_state = updated_optimizer.update(
                gradient, opt_state, params
            )
            new_params = optax.apply_updates(params, updates)
            meta_gradient_norm = gradient_norm

        # Performance monitoring
        if self.performance_monitor is not None:
            self.performance_monitor.update_metrics(
                step=step,
                loss=float(loss_value),
                gradient_norm=float(gradient_norm),
                learning_rate=float(current_lr),
            )

        # Prepare meta-information
        meta_info = {
            "learning_rate": float(current_lr),
            "gradient_norm": float(gradient_norm),
            "meta_gradient_norm": float(meta_gradient_norm),
            "loss": float(loss_value),
        }

        # Add performance analytics if available
        if self.performance_monitor is not None and step % 10 == 0:
            analytics = self.performance_monitor.get_performance_analytics()
            meta_info.update(analytics)

        # Convert to proper types for return signature
        return_params = jax.tree.map(jnp.asarray, new_params)
        return_info = {
            k: jnp.asarray(v) if isinstance(v, (int, float)) else v
            for k, v in meta_info.items()
        }

        return return_params, new_opt_state, return_info

    def store_optimization_result(
        self, params: jax.Array, problem_features: jax.Array
    ) -> None:
        """Store optimization result for future warm-starting.

        Args:
            params: Final optimized parameters
            problem_features: Features characterizing the problem
        """
        # Add to warm-start strategy memory
        if hasattr(self.warm_start_strategy, "_parameter_memory"):
            self.warm_start_strategy._parameter_memory.append(params)
            self.warm_start_strategy._problem_features_memory.append(problem_features)

            # Keep memory size bounded
            if (
                len(self.warm_start_strategy._parameter_memory)
                > self.warm_start_strategy.memory_size
            ):
                self.warm_start_strategy._parameter_memory.pop(0)
                self.warm_start_strategy._problem_features_memory.pop(0)

    def get_warm_start_params(
        self,
        current_problem_features: jax.Array,
        target_shape: tuple[int, ...],
    ) -> jax.Array:
        """Get warm-start parameters for new problem.

        Args:
            current_problem_features: Features of current problem
            target_shape: Target shape for parameters

        Returns:
            Warm-start parameters
        """
        if (
            hasattr(self.warm_start_strategy, "_parameter_memory")
            and self.warm_start_strategy._parameter_memory
        ):
            # Find most similar previous problem
            similarities = []
            for prev_features in self.warm_start_strategy._problem_features_memory:
                similarity = jnp.dot(current_problem_features, prev_features) / (
                    jnp.linalg.norm(current_problem_features)
                    * jnp.linalg.norm(prev_features)
                )
                similarities.append(float(similarity))

            best_idx = jnp.argmax(jnp.array(similarities))
            best_params = self.warm_start_strategy._parameter_memory[best_idx]

            # Adapt to target shape
            if best_params.shape == target_shape:
                return self.warm_start_strategy.get_warm_start_params(
                    best_params, current_problem_features
                )
            # Shape mismatch, return random initialization
            return jax.random.normal(jax.random.PRNGKey(42), target_shape)
        # No previous results, return random initialization
        return jax.random.normal(jax.random.PRNGKey(42), target_shape)

    def quantum_step(
        self,
        energy_fn: Callable[[jax.Array], jax.Array],
        orbital_coeffs: jax.Array,
        opt_state: Any,
        scf_context: dict[str, Any],
        step: int,
    ) -> tuple[jax.Array, Any, dict[str, Any]]:
        """Perform quantum-aware meta-optimization step.

        Args:
            energy_fn: Energy function to minimize
            orbital_coeffs: Current orbital coefficients
            opt_state: Current optimizer state
            scf_context: SCF iteration context
            step: Current step number

        Returns:
            Tuple of (new_coeffs, new_opt_state, quantum_info)
        """
        if not self.config.quantum_aware:
            # Fall back to standard step
            return self.step(energy_fn, orbital_coeffs, opt_state, step)

        # Standard optimization step
        new_coeffs, new_opt_state, meta_info = self.step(
            energy_fn, orbital_coeffs, opt_state, step
        )

        # Quantum-specific adaptations
        if self.l2o_engine is not None and self.l2o_engine.quantum_aware:
            # Use quantum-aware L2O adaptations
            scf_history = scf_context.get("energy_history", jnp.array([]))
            quantum_adaptation = self.l2o_engine.compute_quantum_update(
                new_coeffs, scf_history
            )

            # Apply quantum adaptation
            new_coeffs = new_coeffs + 0.1 * quantum_adaptation

        # Add quantum-specific information
        quantum_info = meta_info.copy()
        quantum_info.update(
            {
                "scf_iteration": scf_context.get("iteration", 0),
                "scf_acceleration": True,  # Placeholder
                "energy_prediction": float(energy_fn(new_coeffs)),
            }
        )

        # Update quantum performance metrics
        if (
            self.performance_monitor is not None
            and self.performance_monitor.quantum_aware
        ):
            energy_error = abs(float(energy_fn(new_coeffs) - energy_fn(orbital_coeffs)))
            self.performance_monitor.update_metrics(
                step=step,
                energy_error=energy_error,
                scf_iterations=scf_context.get("iteration", 0),
            )

        return new_coeffs, new_opt_state, quantum_info


__all__ = ["MetaOptimizer"]
