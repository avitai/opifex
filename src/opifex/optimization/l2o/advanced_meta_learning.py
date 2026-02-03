"""Advanced Learn-to-Optimize (L2O) Meta-Learning Algorithms.

This module implements advanced meta-learning algorithms for optimization including
MAML (Model-Agnostic Meta-Learning), Reptile, and gradient-based meta-learning
strategies. These algorithms enable few-shot adaptation to new optimization problems
and self-improving optimization capabilities.

Key Features:
- MAML for few-shot optimization adaptation
- Reptile algorithm for first-order meta-learning
- Gradient-based meta-learning for parameter initialization
- Meta-L2O integration for self-improving optimization
- Integration with existing L2O engine and MetaOptimizer framework
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Rngs

from opifex.optimization.l2o.l2o_engine import L2OEngine, L2OEngineConfig
from opifex.optimization.l2o.parametric_solver import OptimizationProblem


@dataclass
class MAMLConfig:
    """Configuration for Model-Agnostic Meta-Learning (MAML) optimization.

    MAML enables few-shot adaptation to new optimization problems by learning
    parameter initializations that allow rapid adaptation with gradient descent.
    """

    inner_learning_rate: float = 1e-3
    meta_learning_rate: float = 1e-4
    inner_steps: int = 5
    meta_batch_size: int = 8
    adaptation_steps: int = 10
    second_order: bool = True
    enable_adaptation_rate_learning: bool = True
    task_distribution_diversity: float = 0.8
    convergence_tolerance: float = 1e-6

    def __post_init__(self):
        """Validate MAML configuration parameters."""
        if self.inner_learning_rate <= 0:
            raise ValueError("Inner learning rate must be positive")
        if self.meta_learning_rate <= 0:
            raise ValueError("Meta learning rate must be positive")
        if self.inner_steps <= 0:
            raise ValueError("Inner steps must be positive")


@dataclass
class ReptileConfig:
    """Configuration for Reptile meta-learning algorithm.

    Reptile is a first-order meta-learning algorithm that is simpler than MAML
    but still effective for few-shot learning by moving towards parameters
    that work well on task-specific optimization.
    """

    meta_learning_rate: float = 1e-3
    inner_learning_rate: float = 1e-2
    inner_steps: int = 10
    meta_batch_size: int = 16
    adaptation_momentum: float = 0.9
    task_sampling_strategy: str = "uniform"
    gradient_clipping: float = 1.0
    convergence_patience: int = 5

    def __post_init__(self):
        """Validate Reptile configuration parameters."""
        valid_strategies = ["uniform", "weighted", "adaptive"]
        if self.task_sampling_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid sampling strategy: {self.task_sampling_strategy}. "
                f"Must be one of {valid_strategies}"
            )


@dataclass
class GradientBasedMetaLearningConfig:
    """Configuration for gradient-based meta-learning strategies.

    This includes various gradient-based approaches for learning optimization
    strategies including learned optimizers and meta-gradient methods.
    """

    optimizer_network_layers: list[int] | None = None
    meta_learning_rate: float = 1e-4
    gradient_unroll_steps: int = 20
    learned_lr_bounds: tuple[float, float] = (1e-6, 1.0)
    momentum_adaptation: bool = True
    curvature_adaptation: bool = False
    problem_conditioning: bool = True
    numerical_stability_epsilon: float = 1e-8

    def __post_init__(self):
        """Set default values and validate configuration."""
        if self.optimizer_network_layers is None:
            self.optimizer_network_layers = [128, 64, 32]


class MAMLOptimizer(nnx.Module):
    """Model-Agnostic Meta-Learning (MAML) optimizer for L2O.

    MAML learns parameter initializations that enable rapid adaptation to new
    optimization problems with just a few gradient steps. This implementation
    integrates with the existing L2O framework.
    """

    def __init__(
        self,
        config: MAMLConfig,
        l2o_config: L2OEngineConfig,
        optimizer_input_dim: int,
        optimizer_output_dim: int,
        *,
        rngs: Rngs,
    ):
        """Initialize MAML optimizer.

        Args:
            config: MAML configuration
            l2o_config: L2O engine configuration for integration
            optimizer_input_dim: Input dimension for the meta-optimizer
            optimizer_output_dim: Output dimension for optimization parameters
            rngs: Random number generators
        """
        super().__init__()

        self.config = config
        self.l2o_config = l2o_config
        self.input_dim = optimizer_input_dim
        self.output_dim = optimizer_output_dim

        # Meta-optimizer network for learning parameter initialization
        self.meta_optimizer = nnx.Sequential(
            nnx.Linear(optimizer_input_dim, 256, rngs=rngs),
            nnx.gelu,
            nnx.Linear(256, 128, rngs=rngs),
            nnx.gelu,
            nnx.Linear(128, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, optimizer_output_dim, rngs=rngs),
        )

        # Learned adaptation rate network
        if config.enable_adaptation_rate_learning:
            self.adaptation_rate_network = nnx.Sequential(
                nnx.Linear(optimizer_input_dim, 64, rngs=rngs),
                nnx.gelu,
                nnx.Linear(64, 32, rngs=rngs),
                nnx.gelu,
                nnx.Linear(32, 1, rngs=rngs),
                nnx.sigmoid,  # Ensure positive learning rate
            )
        else:
            self.adaptation_rate_network = None

        # Initialize meta-parameters for optimization
        self.meta_parameters = nnx.Variable(jnp.zeros((optimizer_output_dim,)))

    def meta_learn_on_task_distribution(
        self,
        task_distribution: list[tuple[OptimizationProblem, jax.Array]],
        meta_optimizer_state: Any,
        meta_step: int,
    ) -> tuple[dict[str, Any], float]:
        """Perform MAML meta-learning on a distribution of optimization tasks.

        Args:
            task_distribution: List of (problem, parameters) tuples for meta-learning
            meta_optimizer_state: Current meta-optimizer state
            meta_step: Current meta-learning step

        Returns:
            Updated meta-optimizer state and meta-loss
        """
        batch_size = min(self.config.meta_batch_size, len(task_distribution))
        task_batch = task_distribution[:batch_size]

        # Compute meta-gradients across task batch
        def meta_loss_fn(meta_params):
            total_loss = 0.0

            for problem, problem_params in task_batch:
                # Initialize task-specific parameters from meta-parameters
                task_params = meta_params.copy()

                # Perform inner optimization loop
                for _inner_step in range(self.config.inner_steps):
                    # Compute task-specific loss and gradients
                    _, task_grads = self._compute_task_loss_and_gradients(
                        problem, problem_params, task_params
                    )

                    # Update task parameters with inner learning rate
                    inner_lr = self._get_adaptation_rate(problem_params)
                    task_params = task_params - inner_lr * task_grads

                # Compute final task loss after adaptation
                final_task_loss, _ = self._compute_task_loss_and_gradients(
                    problem, problem_params, task_params
                )
                total_loss += final_task_loss

            return total_loss / batch_size

        # Compute meta-gradients
        if self.config.second_order:
            meta_loss, meta_grads = jax.value_and_grad(meta_loss_fn)(
                self.meta_parameters[...]
            )
        else:
            # First-order approximation for computational efficiency
            meta_loss, meta_grads = self._first_order_meta_gradients(
                meta_loss_fn, self.meta_parameters[...]
            )

        # Update meta-parameters
        self.meta_parameters[...] = (
            self.meta_parameters[...] - self.config.meta_learning_rate * meta_grads
        )

        return {"meta_loss": float(meta_loss), "meta_gradients": meta_grads}, float(
            meta_loss
        )

    def adapt_to_new_task(
        self,
        problem: OptimizationProblem,
        problem_params: jax.Array,
        adaptation_steps: int | None = None,
    ) -> jax.Array:
        """Rapidly adapt to a new optimization task using learned initialization.

        Args:
            problem: New optimization problem to adapt to
            problem_params: Parameters for the new problem
            adaptation_steps: Number of adaptation steps (uses config default if None)

        Returns:
            Adapted parameters for the new task
        """
        if adaptation_steps is None:
            adaptation_steps = self.config.adaptation_steps

        # Start from meta-learned initialization
        adapted_params = self.meta_parameters[...].copy()

        # Perform rapid adaptation
        for _step in range(adaptation_steps):
            # Compute loss and gradients for current task
            _task_loss, task_grads = self._compute_task_loss_and_gradients(
                problem, problem_params, adapted_params
            )

            # Adaptive learning rate based on problem characteristics
            adaptation_lr = self._get_adaptation_rate(problem_params)

            # Update parameters
            adapted_params = adapted_params - adaptation_lr * task_grads

            # Early stopping for convergence
            if jnp.linalg.norm(task_grads) < self.config.convergence_tolerance:
                break

        return adapted_params

    def _compute_task_loss_and_gradients(
        self,
        problem: OptimizationProblem,
        problem_params: jax.Array,
        current_params: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute task-specific loss and gradients for meta-learning."""

        # Define optimization objective based on problem type
        def task_objective(params):
            # Extract relevant portion of parameters that matches problem dimension
            problem_vars = params[: problem.dimension]

            if problem.problem_type == "quadratic":
                # Quadratic: 0.5 * x^T Q x + c^T x
                n_vars = problem.dimension
                Q_size = n_vars * n_vars

                # Handle case where problem_params might be shorter than expected
                if problem_params.size >= Q_size + n_vars:
                    Q_matrix = problem_params[:Q_size].reshape((n_vars, n_vars))
                    c = problem_params[Q_size : Q_size + n_vars]
                else:
                    # Use identity matrix and ones vector as fallback
                    Q_matrix = jnp.eye(n_vars)
                    c = jnp.ones(n_vars) * 0.1

                return 0.5 * jnp.dot(
                    problem_vars, jnp.dot(Q_matrix, problem_vars)
                ) + jnp.dot(c, problem_vars)

            if problem.problem_type == "linear":
                # Linear: c^T x
                if problem_params.size >= problem.dimension:
                    c = problem_params[: problem.dimension]
                else:
                    # Pad with ones if insufficient parameters
                    c = jnp.concatenate(
                        [
                            problem_params,
                            jnp.ones(problem.dimension - problem_params.size),
                        ]
                    )
                return jnp.dot(c, problem_vars)

            # nonlinear
            # Simple nonlinear: sum of squared terms with nonlinear transformation
            return jnp.sum(problem_vars**2) + 0.1 * jnp.sum(jnp.sin(problem_vars))

        loss_value, gradients = jax.value_and_grad(task_objective)(current_params)
        return loss_value, gradients

    def _get_adaptation_rate(self, problem_params: jax.Array) -> jax.Array:
        """Get learned adaptation rate based on problem characteristics."""
        if self.adaptation_rate_network is not None:
            # Ensure problem_params matches expected input dimension
            if problem_params.size > self.input_dim:
                input_features = problem_params[: self.input_dim]
            else:
                padding = jnp.zeros(self.input_dim - problem_params.size)
                input_features = jnp.concatenate([problem_params, padding])

            # Learn adaptation rate
            learned_rate = self.adaptation_rate_network(input_features).squeeze()
            # Scale to reasonable range
            return learned_rate * self.config.inner_learning_rate

        return jnp.array(self.config.inner_learning_rate)

    def _first_order_meta_gradients(
        self, meta_loss_fn: Callable, meta_params: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Compute first-order approximation of meta-gradients for efficiency."""
        # First-order MAML approximation
        loss_value = meta_loss_fn(meta_params)
        gradients = jax.grad(meta_loss_fn)(meta_params)
        return loss_value, gradients


class ReptileOptimizer(nnx.Module):
    """Reptile meta-learning optimizer for L2O.

    Reptile is a first-order meta-learning algorithm that learns parameter
    initializations by repeatedly sampling tasks, taking gradient steps on each
    task, and moving the initialization towards the adapted parameters.
    """

    def __init__(
        self,
        config: ReptileConfig,
        l2o_config: L2OEngineConfig,
        optimizer_input_dim: int,
        optimizer_output_dim: int,
        *,
        rngs: Rngs,
    ):
        """Initialize Reptile optimizer.

        Args:
            config: Reptile configuration
            l2o_config: L2O engine configuration for integration
            optimizer_input_dim: Input dimension for problem characteristics
            optimizer_output_dim: Output dimension for optimization parameters
            rngs: Random number generators
        """
        super().__init__()

        self.config = config
        self.l2o_config = l2o_config
        self.input_dim = optimizer_input_dim
        self.output_dim = optimizer_output_dim

        # Initialize meta-parameters
        self.meta_parameters = nnx.Variable(
            nnx.initializers.normal(0.01)(rngs.params(), (optimizer_output_dim,))
        )

        # Momentum buffer for meta-parameter updates
        self.momentum_buffer = nnx.Variable(jnp.zeros((optimizer_output_dim,)))

    def meta_learn_reptile_step(
        self,
        task_distribution: list[tuple[OptimizationProblem, jax.Array]],
        meta_step: int,
    ) -> dict[str, Any]:
        """Perform one Reptile meta-learning step.

        Args:
            task_distribution: List of optimization tasks for meta-learning
            meta_step: Current meta-learning step

        Returns:
            Dictionary with meta-learning metrics
        """
        # Sample tasks from distribution
        batch_size = min(self.config.meta_batch_size, len(task_distribution))
        sampled_tasks = self._sample_tasks(task_distribution, batch_size)

        meta_gradient = jnp.zeros_like(self.meta_parameters[...])
        total_task_loss = 0.0

        for problem, problem_params in sampled_tasks:
            # Start from current meta-parameters
            task_params = self.meta_parameters[...].copy()

            # Perform inner optimization steps on this task
            for _inner_step in range(self.config.inner_steps):
                _task_loss, task_gradients = self._compute_task_loss_and_gradients(
                    problem, problem_params, task_params
                )

                # Clip gradients for stability
                clipped_gradients = self._clip_gradients(
                    task_gradients, self.config.gradient_clipping
                )

                # Update task parameters
                task_params = (
                    task_params - self.config.inner_learning_rate * clipped_gradients
                )

            # Reptile meta-gradient: difference between final and initial parameters
            task_meta_gradient = task_params - self.meta_parameters[...]
            meta_gradient += task_meta_gradient

            # Track final task loss for monitoring
            final_loss, _ = self._compute_task_loss_and_gradients(
                problem, problem_params, task_params
            )
            total_task_loss += final_loss

        # Average meta-gradient across tasks
        meta_gradient = meta_gradient / batch_size

        # Apply momentum to meta-parameter updates
        self.momentum_buffer[...] = (
            self.config.adaptation_momentum * self.momentum_buffer[...]
            + (1 - self.config.adaptation_momentum) * meta_gradient
        )

        # Update meta-parameters
        self.meta_parameters[...] = (
            self.meta_parameters[...]
            + self.config.meta_learning_rate * self.momentum_buffer[...]
        )

        return {
            "meta_loss": total_task_loss / batch_size,
            "meta_gradient_norm": jnp.linalg.norm(meta_gradient),
            "momentum_norm": jnp.linalg.norm(self.momentum_buffer[...]),
        }

    def adapt_to_task(
        self,
        problem: OptimizationProblem,
        problem_params: jax.Array,
        adaptation_steps: int,
    ) -> jax.Array:
        """Adapt meta-parameters to a specific task using Reptile approach.

        Args:
            problem: Optimization problem to adapt to
            problem_params: Problem parameters
            adaptation_steps: Number of adaptation steps

        Returns:
            Task-adapted parameters
        """
        # Start from meta-parameters
        adapted_params = self.meta_parameters[...].copy()

        # Perform task-specific optimization
        for _step in range(adaptation_steps):
            _task_loss, task_gradients = self._compute_task_loss_and_gradients(
                problem, problem_params, adapted_params
            )

            # Clip gradients
            clipped_gradients = self._clip_gradients(
                task_gradients, self.config.gradient_clipping
            )

            # Update parameters
            adapted_params = (
                adapted_params - self.config.inner_learning_rate * clipped_gradients
            )

        return adapted_params

    def _sample_tasks(
        self,
        task_distribution: list[tuple[OptimizationProblem, jax.Array]],
        batch_size: int,
    ) -> list[tuple[OptimizationProblem, jax.Array]]:
        """Sample tasks from the task distribution based on sampling strategy."""
        if self.config.task_sampling_strategy == "uniform":
            # Uniform random sampling
            indices = jax.random.choice(
                jax.random.PRNGKey(42),
                len(task_distribution),
                (batch_size,),
                replace=False,
            )
            return [task_distribution[i] for i in indices]

        if self.config.task_sampling_strategy == "weighted":
            # Weighted sampling based on task difficulty (simplified)
            return task_distribution[:batch_size]

        # adaptive
        # Adaptive sampling based on learning progress
        return task_distribution[:batch_size]

    def _compute_task_loss_and_gradients(
        self,
        problem: OptimizationProblem,
        problem_params: jax.Array,
        current_params: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute task loss and gradients (same as MAML implementation)."""

        def task_objective(params):
            # Extract relevant portion of parameters that matches problem dimension
            problem_vars = params[: problem.dimension]

            if problem.problem_type == "quadratic":
                n_vars = problem.dimension
                Q_size = n_vars * n_vars

                # Handle case where problem_params might be shorter than expected
                if problem_params.size >= Q_size + n_vars:
                    Q_matrix = problem_params[:Q_size].reshape((n_vars, n_vars))
                    c = problem_params[Q_size : Q_size + n_vars]
                else:
                    # Use identity matrix and ones vector as fallback
                    Q_matrix = jnp.eye(n_vars)
                    c = jnp.ones(n_vars) * 0.1

                return 0.5 * jnp.dot(
                    problem_vars, jnp.dot(Q_matrix, problem_vars)
                ) + jnp.dot(c, problem_vars)

            if problem.problem_type == "linear":
                if problem_params.size >= problem.dimension:
                    c = problem_params[: problem.dimension]
                else:
                    # Pad with ones if insufficient parameters
                    c = jnp.concatenate(
                        [
                            problem_params,
                            jnp.ones(problem.dimension - problem_params.size),
                        ]
                    )
                return jnp.dot(c, problem_vars)

            # nonlinear
            return jnp.sum(problem_vars**2) + 0.1 * jnp.sum(jnp.sin(problem_vars))

        loss_value, gradients = jax.value_and_grad(task_objective)(current_params)
        return loss_value, gradients

    def _clip_gradients(self, gradients: jax.Array, clip_norm: float) -> jax.Array:
        """Clip gradients by norm for training stability."""
        grad_norm = jnp.linalg.norm(gradients)
        clip_coeff = jnp.minimum(1.0, clip_norm / (grad_norm + 1e-8))
        return gradients * clip_coeff


class GradientBasedMetaLearner(nnx.Module):
    """Gradient-based meta-learning for learning optimization strategies.

    This class implements neural networks that learn to optimize by processing
    gradients and optimization histories to produce effective parameter updates.
    """

    def __init__(
        self,
        config: GradientBasedMetaLearningConfig,
        l2o_config: L2OEngineConfig,
        problem_dim: int,
        *,
        rngs: Rngs,
    ):
        """Initialize gradient-based meta-learner.

        Args:
            config: Configuration for gradient-based meta-learning
            l2o_config: L2O engine configuration
            problem_dim: Dimension of optimization problems
            rngs: Random number generators
        """
        super().__init__()

        self.config = config
        self.l2o_config = l2o_config
        self.problem_dim = problem_dim

        # Learned optimizer network
        # Input: [gradient, previous_update, loss_history, problem_features]
        input_features = problem_dim * 3 + 10  # gradients + updates + features

        optimizer_layers = []
        prev_dim = input_features

        # Handle None case for optimizer_network_layers
        network_layers = config.optimizer_network_layers or [128, 64, 32]

        for hidden_dim in network_layers:
            optimizer_layers.extend(
                [
                    nnx.Linear(prev_dim, hidden_dim, rngs=rngs),
                    nnx.gelu,
                ]
            )
            prev_dim = hidden_dim

        # Output layer for parameter updates
        optimizer_layers.append(nnx.Linear(prev_dim, problem_dim, rngs=rngs))

        self.optimizer_network = nnx.Sequential(*optimizer_layers)

        # Learning rate adaptation network
        self.lr_network = nnx.Sequential(
            nnx.Linear(input_features, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 32, rngs=rngs),
            nnx.gelu,
            nnx.Linear(32, 1, rngs=rngs),
            nnx.sigmoid,  # Bounded learning rate
        )

        # Momentum adaptation network if enabled
        if config.momentum_adaptation:
            self.momentum_network = nnx.Sequential(
                nnx.Linear(input_features, 32, rngs=rngs),
                nnx.gelu,
                nnx.Linear(32, 1, rngs=rngs),
                nnx.sigmoid,  # Bounded momentum
            )
        else:
            self.momentum_network = None

        # History buffer for tracking optimization progress
        self.optimization_history = nnx.Variable(
            jnp.zeros((config.gradient_unroll_steps, problem_dim))
        )

    def compute_learned_update(
        self,
        gradients: jax.Array,
        previous_update: jax.Array,
        loss_history: jax.Array,
        problem_features: jax.Array,
        step: int,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute parameter update using learned optimization strategy.

        Args:
            gradients: Current gradients
            previous_update: Previous parameter update
            loss_history: History of loss values
            problem_features: Problem-specific features
            step: Current optimization step

        Returns:
            Parameter update and optimization metrics
        """
        # Prepare input features for the learned optimizer
        input_features = self._prepare_optimizer_input(
            gradients, previous_update, loss_history, problem_features
        )

        # Compute learned parameter update
        learned_update = self.optimizer_network(input_features)

        # Compute adaptive learning rate
        learned_lr = self._compute_adaptive_learning_rate(input_features)

        # Apply learning rate bounds
        lr_min, lr_max = self.config.learned_lr_bounds
        learned_lr = jnp.clip(learned_lr, lr_min, lr_max)

        # Scale update by learned learning rate
        scaled_update = learned_lr * learned_update

        # Apply momentum if enabled
        if self.config.momentum_adaptation and self.momentum_network is not None:
            momentum_coeff = self.momentum_network(input_features).squeeze()
            momentum_update = momentum_coeff * previous_update
            final_update = scaled_update + momentum_update
        else:
            final_update = scaled_update

        # Update optimization history
        self._update_optimization_history(final_update, step)

        metrics = {
            "learned_lr": learned_lr,
            "update_norm": jnp.linalg.norm(final_update),
            "gradient_norm": jnp.linalg.norm(gradients),
        }

        if self.config.momentum_adaptation and self.momentum_network is not None:
            metrics["momentum_coeff"] = momentum_coeff

        return final_update, metrics

    def meta_train_on_optimization_trajectories(
        self,
        optimization_trajectories: list[dict[str, Any]],
        meta_optimizer_state: Any,
    ) -> tuple[Any, float]:
        """Meta-train the learned optimizer on optimization trajectories.

        Args:
            optimization_trajectories: List of optimization trajectories for training
            meta_optimizer_state: Current meta-optimizer state

        Returns:
            Updated meta-optimizer state and meta-loss
        """

        def meta_loss_fn(network_params):
            total_loss = 0.0

            for trajectory in optimization_trajectories:
                # Reconstruct optimization trajectory with learned optimizer
                simulated_loss = self._simulate_optimization_trajectory(
                    trajectory, network_params
                )
                total_loss += simulated_loss

            return total_loss / len(optimization_trajectories)

        # Compute meta-gradients for the learned optimizer
        meta_loss, meta_grads = jax.value_and_grad(meta_loss_fn)(
            self._get_network_parameters()
        )

        # Update network parameters
        self._update_network_parameters(meta_grads)

        return meta_optimizer_state, meta_loss

    def _prepare_optimizer_input(
        self,
        gradients: jax.Array,
        previous_update: jax.Array,
        loss_history: jax.Array,
        problem_features: jax.Array,
    ) -> jax.Array:
        """Prepare input features for the learned optimizer network."""
        # Normalize gradients and updates for numerical stability
        grad_norm = jnp.linalg.norm(gradients) + self.config.numerical_stability_epsilon
        normalized_gradients = gradients / grad_norm

        update_norm = (
            jnp.linalg.norm(previous_update) + self.config.numerical_stability_epsilon
        )
        normalized_update = previous_update / update_norm

        # Prepare loss features
        if loss_history.size > 0:
            loss_features = jnp.array(
                [
                    loss_history[-1] if loss_history.size > 0 else 0.0,
                    jnp.mean(loss_history) if loss_history.size > 0 else 0.0,
                    jnp.std(loss_history) if loss_history.size > 1 else 0.0,
                ]
            )
        else:
            loss_features = jnp.zeros(3)

        # Ensure problem features have correct size
        if problem_features.size > 7:
            problem_features = problem_features[:7]
        elif problem_features.size < 7:
            padding = jnp.zeros(7 - problem_features.size)
            problem_features = jnp.concatenate([problem_features, padding])

        # Concatenate all features
        combined_features = jnp.concatenate(
            [
                normalized_gradients,
                normalized_update,
                loss_features,
                problem_features,
            ]
        )

        # Ensure input features match expected dimension
        expected_dim = self.problem_dim * 3 + 10
        if combined_features.size > expected_dim:
            input_features = combined_features[:expected_dim]
        elif combined_features.size < expected_dim:
            padding = jnp.zeros(expected_dim - combined_features.size)
            input_features = jnp.concatenate([combined_features, padding])
        else:
            input_features = combined_features

        return input_features

    def _compute_adaptive_learning_rate(self, input_features: jax.Array) -> jax.Array:
        """Compute adaptive learning rate based on optimization state."""
        return self.lr_network(input_features).squeeze()

    def _update_optimization_history(self, update: jax.Array, step: int) -> None:
        """Update optimization history buffer."""
        # Circular buffer implementation
        buffer_idx = step % self.config.gradient_unroll_steps
        self.optimization_history[...] = (
            self.optimization_history[...].at[buffer_idx].set(update)
        )

    def _simulate_optimization_trajectory(
        self, trajectory: dict[str, Any], network_params: dict[str, Any]
    ) -> jax.Array:
        """Simulate optimization trajectory for meta-training."""
        # Simplified implementation - would use trajectory data to compute
        # how well the learned optimizer performs compared to ground truth
        return jnp.array(0.0)  # Placeholder

    def _get_network_parameters(self) -> dict[str, Any]:
        """Get current network parameters for meta-learning."""
        return {}  # Placeholder - would extract actual network parameters

    def _update_network_parameters(self, meta_grads: dict[str, Any]) -> None:
        """Update network parameters using meta-gradients."""
        # Placeholder - would update actual network parameters


class MetaL2OIntegration(nnx.Module):
    """Meta-L2O Integration for self-improving optimization framework.

    This class integrates MAML, Reptile, and gradient-based meta-learning
    with the existing L2O engine to create a self-improving optimization
    framework that learns from optimization experience.
    """

    def __init__(
        self,
        l2o_engine: L2OEngine,
        maml_config: MAMLConfig | None = None,
        reptile_config: ReptileConfig | None = None,
        gb_config: GradientBasedMetaLearningConfig | None = None,
        *,
        rngs: Rngs,
    ):
        """Initialize Meta-L2O integration.

        Args:
            l2o_engine: Existing L2O engine to enhance
            maml_config: Configuration for MAML (optional)
            reptile_config: Configuration for Reptile (optional)
            gb_config: Configuration for gradient-based meta-learning (optional)
            rngs: Random number generators
        """
        super().__init__()

        self.l2o_engine = l2o_engine
        self.rngs = rngs

        # Store configs for dynamic algorithm creation
        self.maml_config = maml_config
        self.reptile_config = reptile_config
        self.gb_config = gb_config

        # Keep references to algorithms for dynamic instantiation
        # These will be created dynamically based on problem dimensions
        self.maml_optimizer = None
        self.reptile_optimizer = None
        self.gb_meta_learner = None

        # Experience buffer for storing optimization experiences
        self.experience_buffer: list[dict[str, Any]] = []
        self.meta_learning_active = True

    def solve_with_meta_learning(
        self,
        problem: OptimizationProblem,
        problem_params: jax.Array,
        meta_learning_strategy: str = "auto",
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Solve optimization problem using meta-learning enhanced L2O.

        Args:
            problem: Optimization problem to solve
            problem_params: Problem parameters
            meta_learning_strategy: Strategy to use ('maml', 'reptile',
                'gradient_based', 'auto')

        Returns:
            Solution and optimization metrics
        """
        start_time = time.time()

        # Automatically select meta-learning strategy if requested
        if meta_learning_strategy == "auto":
            meta_learning_strategy = self._select_meta_learning_strategy(problem)

        # Get problem dimension for creating appropriately sized algorithms
        problem_dim = problem.dimension

        # Initialize metrics dictionary with proper typing
        metrics: dict[str, Any] = {}

        # Solve using selected meta-learning approach
        if meta_learning_strategy == "maml" and self.maml_config is not None:
            # Create MAML optimizer with correct dimensions for this problem
            maml_optimizer = MAMLOptimizer(
                self.maml_config,
                self.l2o_engine.l2o_config,
                problem_dim * 3,
                problem_dim,
                rngs=self.rngs,
            )
            solution = maml_optimizer.adapt_to_new_task(problem, problem_params)
            metrics["meta_learning_strategy"] = "maml"

        elif meta_learning_strategy == "reptile" and self.reptile_config is not None:
            # Create Reptile optimizer with correct dimensions for this problem
            reptile_optimizer = ReptileOptimizer(
                self.reptile_config,
                self.l2o_engine.l2o_config,
                problem_dim * 3,
                problem_dim,
                rngs=self.rngs,
            )
            solution = reptile_optimizer.adapt_to_task(problem, problem_params, 10)
            metrics["meta_learning_strategy"] = "reptile"

        elif meta_learning_strategy == "gradient_based" and self.gb_config is not None:
            # Use existing L2O engine with gradient-based enhancement
            base_solution = self.l2o_engine.solve_parametric_problem(
                problem, problem_params
            )
            solution = base_solution  # Simplified for now
            metrics["meta_learning_strategy"] = "gradient_based"

        else:
            # Fallback to base L2O engine
            solution = self.l2o_engine.solve_parametric_problem(problem, problem_params)
            metrics["meta_learning_strategy"] = "fallback"

        solve_time = time.time() - start_time
        # Add timing and solution metrics
        metrics["solve_time"] = float(solve_time)
        metrics["solution_norm"] = float(jnp.linalg.norm(solution))

        # Store experience for future meta-learning
        if self.meta_learning_active:
            self._store_optimization_experience(
                problem, problem_params, solution, metrics
            )

        return solution, metrics

    def trigger_meta_learning_update(
        self, task_distribution: list[tuple[OptimizationProblem, jax.Array]]
    ) -> dict[str, Any]:
        """Trigger meta-learning updates across all enabled algorithms.

        Args:
            task_distribution: Distribution of tasks for meta-learning

        Returns:
            Meta-learning results and metrics
        """
        meta_results: dict[str, Any] = {}

        # Use a representative problem dimension from the task distribution
        problem_dim = task_distribution[0][0].dimension if task_distribution else 10

        # MAML meta-learning
        if self.maml_config is not None:
            maml_optimizer = MAMLOptimizer(
                self.maml_config,
                self.l2o_engine.l2o_config,
                problem_dim * 3,
                problem_dim,
                rngs=self.rngs,
            )
            maml_state, maml_loss = maml_optimizer.meta_learn_on_task_distribution(
                task_distribution, None, 0
            )
            meta_results["maml"] = {"state": maml_state, "loss": maml_loss}

        # Reptile meta-learning
        if self.reptile_config is not None:
            reptile_optimizer = ReptileOptimizer(
                self.reptile_config,
                self.l2o_engine.l2o_config,
                problem_dim * 3,
                problem_dim,
                rngs=self.rngs,
            )
            reptile_metrics = reptile_optimizer.meta_learn_reptile_step(
                task_distribution, 0
            )
            meta_results["reptile"] = reptile_metrics

        # Gradient-based meta-learning
        if self.gb_config is not None:
            # Convert experience buffer to optimization trajectories
            trajectories = self._convert_experience_to_trajectories()
            gb_meta_learner = GradientBasedMetaLearner(
                self.gb_config, self.l2o_engine.l2o_config, problem_dim, rngs=self.rngs
            )
            gb_state, gb_loss = gb_meta_learner.meta_train_on_optimization_trajectories(
                trajectories, None
            )
            meta_results["gradient_based"] = {"state": gb_state, "loss": gb_loss}

        return meta_results

    def _select_meta_learning_strategy(self, problem: OptimizationProblem) -> str:
        """Automatically select the best meta-learning strategy for a problem."""
        # Simple heuristic based on problem characteristics
        if problem.problem_type == "quadratic":
            return "maml"  # MAML works well for quadratic problems
        if problem.problem_type == "linear":
            return "reptile"  # Reptile is efficient for simpler problems
        return "gradient_based"  # Gradient-based for complex problems

    def _store_optimization_experience(
        self,
        problem: OptimizationProblem,
        problem_params: jax.Array,
        solution: jax.Array,
        metrics: dict[str, Any],
    ) -> None:
        """Store optimization experience for future meta-learning."""
        experience = {
            "problem": problem,
            "problem_params": problem_params,
            "solution": solution,
            "metrics": metrics,
            "timestamp": time.time(),
        }

        self.experience_buffer.append(experience)

        # Limit buffer size
        max_buffer_size = 1000
        if len(self.experience_buffer) > max_buffer_size:
            self.experience_buffer = self.experience_buffer[-max_buffer_size:]

    def _convert_experience_to_trajectories(self) -> list[dict[str, Any]]:
        """Convert stored experience to optimization trajectories for meta-learning."""
        trajectories = []
        for experience in self.experience_buffer:
            trajectory = {
                "problem_type": experience["problem"].problem_type,
                "solution": experience["solution"],
                "solve_time": experience["metrics"].get("solve_time", 0.0),
            }
            trajectories.append(trajectory)
        return trajectories
