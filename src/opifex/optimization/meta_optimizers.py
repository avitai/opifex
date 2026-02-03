"""Advanced meta-optimization algorithms for scientific machine learning.

This module implements meta-learning approaches to optimization including
learn-to-optimize (L2O) algorithms, adaptive learning rate scheduling,
warm-starting strategies, and performance monitoring. All implementations
follow FLAX NNX patterns and are designed for scientific computing applications.

Key Features:
    - Learn-to-optimize (L2O) meta-learning algorithms
    - Adaptive learning rate scheduling with multiple strategies
    - Warm-starting based on problem similarity
    - Performance monitoring and analytics
    - Quantum-aware optimization adaptations
    - Integration with existing training infrastructure

Classes:
    MetaOptimizerConfig: Configuration for meta-optimization algorithms
    AdaptiveLearningRateScheduler: Dynamic learning rate adaptation
    WarmStartingStrategy: Initialization from previous optimizations
    LearnToOptimize: Learn-to-optimize meta-learning engine
    PerformanceMonitor: Performance tracking and analytics
    MetaOptimizer: Integrated meta-optimization system

Author: Opifex Framework Team
Date: December 2024
License: MIT
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx

# Import from centralized configuration module (single source of truth)
from opifex.core.training.config import MetaOptimizerConfig
from opifex.neural.base import StandardMLP


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
        **kwargs,
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


class WarmStartingStrategy:
    """Warm-starting strategies for optimization acceleration.

    This class implements various warm-starting strategies to accelerate
    optimization by leveraging information from previous optimizations
    or similar problems.

    Attributes:
        strategy_type: Type of warm-starting strategy
        similarity_threshold: Threshold for problem similarity
        adaptation_steps: Steps for parameter adaptation
        memory_size: Size of optimization memory
        adaptation_ratio: Ratio for optimizer state adaptation
    """

    def __init__(
        self,
        strategy_type: str = "parameter_transfer",
        similarity_threshold: float = 0.8,
        adaptation_steps: int = 5,
        memory_size: int = 10,
        adaptation_ratio: float = 0.9,
        similarity_metric: str = "cosine",
        min_similarity: float = 0.7,
    ):
        """Initialize warm-starting strategy.

        Args:
            strategy_type: Strategy type
                ('parameter_transfer', 'optimizer_state_transfer',
                 'molecular_similarity')
            similarity_threshold: Threshold for considering problems similar
            adaptation_steps: Number of adaptation steps
            memory_size: Maximum number of previous optimizations to remember
            adaptation_ratio: Ratio for adapting previous states
            similarity_metric: Metric for similarity computation
            min_similarity: Minimum similarity for warm-starting
        """
        self.strategy_type = strategy_type
        self.similarity_threshold = similarity_threshold
        self.adaptation_steps = adaptation_steps
        self.memory_size = memory_size
        self.adaptation_ratio = adaptation_ratio
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity

        # Memory for previous optimizations
        self._parameter_memory = []
        self._problem_features_memory = []
        self._optimizer_state_memory = []

    def get_warm_start_params(
        self, previous_params: jax.Array, current_problem_features: jax.Array
    ) -> jax.Array:
        """Get warm-start parameters based on parameter transfer.

        Args:
            previous_params: Parameters from previous optimization
            current_problem_features: Features of current problem

        Returns:
            Warm-start parameters for current problem
        """
        if self.strategy_type == "parameter_transfer":
            # Simple parameter transfer with optional adaptation
            adapted_params = previous_params * self.adaptation_ratio

            # Add small random perturbation for exploration
            noise = 0.1 * jax.random.normal(
                jax.random.PRNGKey(42), previous_params.shape
            )
            return adapted_params + noise

        # Default: return parameters as-is
        return previous_params

    def adapt_optimizer_state(
        self, previous_opt_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Adapt optimizer state for warm-starting.

        Args:
            previous_opt_state: Previous optimizer state

        Returns:
            Adapted optimizer state
        """
        adapted_state = {}

        for key, value in previous_opt_state.items():
            if key == "step":
                # Reset step count but keep some history
                adapted_state[key] = jnp.array(max(0, int(value * 0.1)))
            elif isinstance(value, jax.Array):
                # Scale momentum/variance terms
                adapted_state[key] = value * self.adaptation_ratio
            else:
                # Keep other state elements as-is
                adapted_state[key] = value

        return adapted_state

    def get_molecular_warm_start(
        self,
        previous_fingerprints: jax.Array,
        previous_params: jax.Array,
        current_fingerprint: jax.Array,
    ) -> jax.Array:
        """Get warm-start parameters based on molecular similarity.

        Args:
            previous_fingerprints: Fingerprints of previous molecules
            previous_params: Parameters for previous molecules
            current_fingerprint: Fingerprint of current molecule

        Returns:
            Warm-start parameters based on most similar molecule
        """
        # Compute similarities
        if self.similarity_metric == "cosine":
            similarities = jnp.dot(previous_fingerprints, current_fingerprint) / (
                jnp.linalg.norm(previous_fingerprints, axis=1)
                * jnp.linalg.norm(current_fingerprint)
            )
        elif self.similarity_metric == "euclidean":
            distances = jnp.linalg.norm(
                previous_fingerprints - current_fingerprint, axis=1
            )
            similarities = 1.0 / (1.0 + distances)
        else:
            # Default to uniform similarity
            similarities = jnp.ones(len(previous_fingerprints))

        # Find most similar molecule
        most_similar_idx = jnp.argmax(similarities)
        max_similarity = similarities[most_similar_idx]

        if max_similarity > self.min_similarity:
            # Use parameters from most similar molecule
            return previous_params[most_similar_idx]
        # No similar molecule found, return average parameters
        return jnp.mean(previous_params, axis=0)


class LearnToOptimize(nnx.Module):
    """Learn-to-optimize (L2O) meta-learning system.

    This class implements learn-to-optimize algorithms that use neural networks
    to learn optimization strategies from data. The meta-network learns to
    predict good parameter updates based on gradient information and
    optimization history.

    Attributes:
        meta_network: Neural network for learning optimization rules
        base_optimizer: Base optimization algorithm
        meta_learning_rate: Learning rate for meta-network training
        unroll_steps: Number of unrolling steps for meta-gradient computation
        adaptive_step_size: Enable adaptive step size learning
        quantum_aware: Enable quantum-specific adaptations
        scf_integration: Enable SCF convergence acceleration
    """

    def __init__(
        self,
        meta_network_layers: list[int] | None = None,
        base_optimizer: str = "adam",
        meta_learning_rate: float = 1e-4,
        unroll_steps: int = 20,
        adaptive_step_size: bool = False,
        quantum_aware: bool = False,
        scf_integration: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Learn-to-Optimize meta-optimizer.

        Args:
            meta_network_layers: Architecture of meta-network
            base_optimizer: Base optimizer to enhance
            meta_learning_rate: Learning rate for meta-network training
            unroll_steps: Number of unroll steps for meta-gradients
            adaptive_step_size: Enable adaptive step size learning
            quantum_aware: Enable quantum-specific optimizations
            scf_integration: Enable SCF convergence acceleration
            rngs: Random number generators for initialization
        """
        if meta_network_layers is None:
            meta_network_layers = [128, 64, 32]

        super().__init__()

        self.meta_network_layers = meta_network_layers
        self.base_optimizer = base_optimizer
        self.meta_learning_rate = meta_learning_rate
        self.unroll_steps = unroll_steps
        self.adaptive_step_size = adaptive_step_size
        self.quantum_aware = quantum_aware
        self.scf_integration = scf_integration

        # Meta-network for learning optimization rules
        # Input: [gradient, previous_updates, loss_history]
        # Output: [parameter_update] or [parameter_update, step_size]
        output_dim = (
            meta_network_layers[0]
            if not adaptive_step_size
            else meta_network_layers[0] + 1
        )

        layers = [*meta_network_layers, output_dim]
        self.meta_network = StandardMLP(layers, rngs=rngs)

        # Meta-optimizer for training the meta-network
        self.meta_optimizer = optax.adam(meta_learning_rate)
        # Convert GraphState to parameters for optax compatibility
        meta_params = nnx.state(self.meta_network, nnx.Param)
        self.meta_opt_state = self.meta_optimizer.init(
            jax.tree.map(lambda x: x, meta_params)
        )

    def compute_update(
        self,
        gradient: jax.Array,
        previous_updates: jax.Array,
        loss_history: jax.Array | None = None,
    ) -> jax.Array:
        """Compute parameter update using meta-network.

        Args:
            gradient: Current gradient
            previous_updates: History of previous updates
            loss_history: History of loss values

        Returns:
            Predicted parameter update
        """
        # Prepare input features for meta-network
        input_features = self._prepare_meta_input(
            gradient, previous_updates, loss_history
        )

        # Get meta-network prediction
        meta_output = self.meta_network(input_features)

        if self.adaptive_step_size:
            # Split output into update direction and step size
            update_direction = meta_output[:-1]
            step_size = jnp.abs(meta_output[-1])  # Ensure positive step size
            parameter_update = step_size * update_direction
        else:
            parameter_update = meta_output

        return parameter_update

    def _prepare_meta_input(
        self,
        gradient: jax.Array,
        previous_updates: jax.Array,
        loss_history: jax.Array | None = None,
    ) -> jax.Array:
        """Prepare input features for meta-network."""
        # Normalize gradient
        grad_norm = jnp.linalg.norm(gradient)
        normalized_grad = gradient / (grad_norm + 1e-8)

        # Features from previous updates
        if previous_updates.size > 0:
            avg_update = jnp.mean(previous_updates, axis=0)
            update_variance = jnp.var(previous_updates, axis=0)
        else:
            avg_update = jnp.zeros_like(gradient)
            update_variance = jnp.zeros_like(gradient)

        # Combine features
        features = jnp.concatenate(
            [
                normalized_grad,
                avg_update,
                update_variance,
                jnp.array([grad_norm]),  # Include gradient norm as scalar feature
            ]
        )

        # Pad or truncate to match meta-network input size
        target_size = self.meta_network_layers[0]  # Use actual network input size
        if len(features) > target_size:
            features = features[:target_size]
        elif len(features) < target_size:
            padding = jnp.zeros(target_size - len(features))
            features = jnp.concatenate([features, padding])

        return features

    def compute_meta_gradients(
        self,
        loss_fn: Callable[[jax.Array], jax.Array],
        initial_params: jax.Array,
    ) -> dict[str, jax.Array]:
        """Compute meta-gradients for meta-network training.

        Args:
            loss_fn: Loss function for optimization problem
            initial_params: Initial parameters for optimization

        Returns:
            Meta-gradients for meta-network parameters
        """

        def meta_loss_fn(meta_params):
            # Create temporary meta-network with correct architecture
            output_size = initial_params.size
            layers = [*self.meta_network_layers, output_size]
            temp_meta_network = StandardMLP(layers, rngs=nnx.Rngs(42))
            nnx.update(temp_meta_network, meta_params)

            # Simulate optimization trajectory using current meta-network
            params = initial_params
            total_loss = 0.0
            previous_updates = jnp.zeros((0, output_size))

            for _step in range(self.unroll_steps):
                # Compute gradient
                gradient = jax.grad(loss_fn)(params)

                # Get meta-network update
                input_features = self._prepare_meta_input(gradient, previous_updates)
                update = temp_meta_network(input_features)

                # Ensure update has correct shape - truncate or pad to match params
                if update.size > initial_params.size:
                    update = update[: initial_params.size]
                elif update.size < initial_params.size:
                    padding = jnp.zeros(initial_params.size - update.size)
                    update = jnp.concatenate([update, padding])
                update = update.reshape(initial_params.shape)

                # Apply update
                params = params - update

                # Accumulate loss
                step_loss = loss_fn(params)
                total_loss += step_loss

                # Update history
                previous_updates = jnp.concatenate(
                    [previous_updates, update.flatten().reshape(1, -1)], axis=0
                )
                if previous_updates.shape[0] > 5:  # Keep only recent history
                    previous_updates = previous_updates[-5:]

            return total_loss / self.unroll_steps

        # Compute meta-gradients
        meta_params = nnx.state(self.meta_network, nnx.Param)
        meta_grads = jax.grad(meta_loss_fn)(meta_params)

        # Convert State to dictionary for compatibility - properly handle JAX types
        return jax.tree.map(lambda x: x, dict(meta_grads))

    def compute_adaptive_update(
        self,
        gradient: jax.Array,
        previous_updates: jax.Array,
    ) -> jax.Array:
        """Compute adaptive parameter update."""
        # Prepare meta-network input
        input_features = self._prepare_meta_input(gradient, previous_updates)

        # Get adaptive update from meta-network
        return self.meta_network(input_features)

    def compute_quantum_update(
        self,
        orbital_params: jax.Array,
        scf_history: jax.Array,
    ) -> jax.Array:
        """Compute quantum-aware parameter update.

        Args:
            orbital_params: Orbital coefficient parameters
            scf_history: SCF convergence history

        Returns:
            Quantum-adapted parameter update
        """
        if not self.quantum_aware:
            return jnp.zeros_like(orbital_params)

        # Simplified quantum adaptation
        # In practice, this would use sophisticated quantum mechanical insights

        # SCF convergence-based adaptation
        scf_trend = (
            jnp.diff(scf_history[-5:]) if len(scf_history) > 1 else jnp.array([0.0])
        )
        scf_acceleration = jnp.mean(scf_trend)

        # Orbital-based features
        orbital_norm = jnp.linalg.norm(orbital_params)
        orbital_features = jnp.array([orbital_norm, scf_acceleration])

        # Pad features to match meta-network input
        padded_features = jnp.zeros(128)
        padded_features = padded_features.at[: len(orbital_features)].set(
            orbital_features
        )

        # Get quantum adaptation from meta-network
        quantum_update = self.meta_network(padded_features)

        # Reshape to match orbital parameters
        return quantum_update[: orbital_params.size].reshape(orbital_params.shape)


class PerformanceMonitor:
    """Performance monitoring and analytics for meta-optimization.

    This class provides comprehensive performance monitoring capabilities
    including metric tracking, convergence detection, and performance
    analytics for optimization algorithms.

    Attributes:
        metrics: List of metrics to track
        window_size: Size of rolling window for metrics
        tracking_frequency: Frequency of metric updates
        convergence_tolerance: Tolerance for convergence detection
        convergence_patience: Patience for convergence detection
        analytics_enabled: Enable detailed analytics
        quantum_aware: Enable quantum-specific metrics
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        window_size: int = 100,
        tracking_frequency: int = 1,
        convergence_tolerance: float = 1e-6,
        convergence_patience: int = 10,
        analytics_enabled: bool = False,
        quantum_aware: bool = False,
    ):
        """Initialize performance monitor.

        Args:
            metrics: List of metrics to track
            window_size: Rolling window size
            tracking_frequency: How often to update metrics
            convergence_tolerance: Tolerance for convergence
            convergence_patience: Patience for convergence detection
            analytics_enabled: Enable detailed analytics
            quantum_aware: Enable quantum metrics
        """
        if metrics is None:
            metrics = ["loss", "gradient_norm"]

        self.metrics = metrics
        self.window_size = window_size
        self.tracking_frequency = tracking_frequency
        self.convergence_tolerance = convergence_tolerance
        self.convergence_patience = convergence_patience
        self.analytics_enabled = analytics_enabled
        self.quantum_aware = quantum_aware

        # Metric storage
        self._metric_history = {metric: [] for metric in metrics}
        self._step_history = []

        # Convergence tracking
        self._convergence_state = dict.fromkeys(metrics, False)
        self._steps_since_improvement = dict.fromkeys(metrics, 0)
        self._best_values = {metric: float("inf") for metric in metrics}

    def update_metrics(self, step: int, **metric_values) -> None:
        """Update tracked metrics.

        Args:
            step: Current optimization step
            **metric_values: Metric values to update
        """
        if step % self.tracking_frequency != 0:
            return

        self._step_history.append(step)

        for metric, value in metric_values.items():
            if metric in self.metrics:
                self._metric_history[metric].append(float(value))

                # Keep only recent history
                if len(self._metric_history[metric]) > self.window_size:
                    self._metric_history[metric] = self._metric_history[metric][
                        -self.window_size :
                    ]

                # Update convergence tracking
                self._update_convergence_tracking(metric, value)

    def _update_convergence_tracking(self, metric: str, value: float) -> None:
        """Update convergence tracking for a metric."""
        if value < self._best_values[metric] - self.convergence_tolerance:
            self._best_values[metric] = value
            self._steps_since_improvement[metric] = 0
        else:
            self._steps_since_improvement[metric] += 1

    def get_metric_history(self, metric: str) -> list[float]:
        """Get history of a specific metric.

        Args:
            metric: Metric name

        Returns:
            List of metric values
        """
        return self._metric_history.get(metric, [])

    def check_convergence(self, metric: str) -> bool:
        """Check if a metric has converged.

        Args:
            metric: Metric name

        Returns:
            True if metric has converged
        """
        if metric not in self._metric_history:
            return False

        history = self._metric_history[metric]
        if len(history) < self.convergence_patience:
            return False

        # Check if recent values are stable
        recent_values = history[-self.convergence_patience :]
        value_range = max(recent_values) - min(recent_values)

        return value_range < self.convergence_tolerance

    def get_performance_analytics(self) -> dict[str, Any]:
        """Get comprehensive performance analytics.

        Returns:
            Dictionary containing performance analytics
        """
        if not self.analytics_enabled:
            return {}

        analytics = {}

        for metric in self.metrics:
            history = self._metric_history[metric]
            if not history:
                continue

            # Basic statistics
            analytics[f"{metric}_mean"] = jnp.mean(jnp.array(history))
            analytics[f"{metric}_std"] = jnp.std(jnp.array(history))
            analytics[f"{metric}_min"] = min(history)
            analytics[f"{metric}_max"] = max(history)

            # Convergence analysis
            if len(history) > 10:
                # Simple convergence rate estimation
                recent_slope = (history[-1] - history[-10]) / 10
                analytics[f"{metric}_convergence_rate"] = recent_slope

                # Stability analysis
                recent_history = history[-min(20, len(history)) :]
                stability = 1.0 / (1.0 + jnp.var(jnp.array(recent_history)))
                analytics[f"{metric}_stability"] = float(stability)

        # Overall optimization efficiency
        if self._metric_history.get("loss"):
            loss_history = self._metric_history["loss"]
            if len(loss_history) > 1:
                total_improvement = loss_history[0] - loss_history[-1]
                steps_taken = len(loss_history)
                efficiency = total_improvement / steps_taken if steps_taken > 0 else 0.0
                analytics["optimization_efficiency"] = efficiency

        analytics["convergence_rate"] = sum(self._convergence_state.values()) / len(
            self.metrics
        )
        analytics["stability_metrics"] = {
            metric: self._steps_since_improvement[metric] < self.convergence_patience
            for metric in self.metrics
        }

        return analytics

    def get_quantum_analytics(self) -> dict[str, Any]:
        """Get quantum-specific performance analytics.

        Returns:
            Dictionary containing quantum analytics
        """
        if not self.quantum_aware:
            return {}

        quantum_analytics = {}

        # SCF efficiency analysis
        if "scf_iterations" in self._metric_history:
            scf_history = self._metric_history["scf_iterations"]
            if scf_history:
                avg_scf_iters = jnp.mean(jnp.array(scf_history))
                scf_trend = (
                    jnp.diff(jnp.array(scf_history[-10:]))
                    if len(scf_history) > 1
                    else [0]
                )
                quantum_analytics["scf_efficiency"] = {
                    "average_iterations": float(avg_scf_iters),
                    "recent_trend": float(jnp.mean(jnp.array(scf_trend))),
                    "acceleration": float(
                        jnp.mean(jnp.array(scf_trend)) < 0
                    ),  # Decreasing is good
                }

        # Energy convergence analysis
        if "energy_error" in self._metric_history:
            energy_history = self._metric_history["energy_error"]
            if energy_history:
                convergence_rate = (
                    (energy_history[0] - energy_history[-1]) / len(energy_history)
                    if len(energy_history) > 1
                    else 0
                )
                quantum_analytics["energy_convergence_rate"] = float(convergence_rate)

        # Chemical accuracy tracking
        if "chemical_accuracy" in self._metric_history:
            accuracy_history = self._metric_history["chemical_accuracy"]
            if accuracy_history:
                target_achieved = [
                    acc < 1e-3 for acc in accuracy_history
                ]  # 1 kcal/mol target
                quantum_analytics["chemical_accuracy_progress"] = {
                    "target_achieved_ratio": sum(target_achieved)
                    / len(target_achieved),
                    "best_accuracy": min(accuracy_history),
                    "current_accuracy": accuracy_history[-1],
                }

        return quantum_analytics


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


# Export all public classes and functions
__all__ = [
    "AdaptiveLearningRateScheduler",
    "LearnToOptimize",
    "MetaOptimizer",
    "MetaOptimizerConfig",
    "PerformanceMonitor",
    "WarmStartingStrategy",
]
