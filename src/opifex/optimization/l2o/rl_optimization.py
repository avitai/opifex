"""Reinforcement Learning-Based Optimization Strategy for L2O Framework.

This module implements a Deep Q-Network (DQN) agent that learns optimization
strategies dynamically, making meta-decisions about when to apply different
optimization algorithms.

The RL agent observes optimization problem features, convergence history, and
resource constraints to select optimal algorithms, adjust hyperparameters,
and determine stopping criteria.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from opifex.optimization.l2o.adaptive_schedulers import MetaSchedulerConfig
from opifex.optimization.meta_optimization import MetaOptimizer, MetaOptimizerConfig


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opifex.optimization.l2o.parametric_solver import OptimizationProblem


@dataclass
class RLOptimizationConfig:
    """Configuration for reinforcement learning-based optimization.

    Attributes:
        state_dim: Dimension of state representation
        action_dim: Number of discrete actions available
        hidden_dims: Hidden layer dimensions for DQN
        learning_rate: Learning rate for DQN training
        discount_factor: Reward discount factor (gamma)
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Exploration decay rate
        replay_buffer_size: Size of experience replay buffer
        batch_size: Batch size for DQN training
        target_update_freq: Frequency to update target network
        reward_convergence_weight: Weight for convergence speed in reward
        reward_quality_weight: Weight for solution quality in reward
        reward_efficiency_weight: Weight for computational efficiency in reward
        max_episode_length: Maximum optimization steps per episode
    """

    state_dim: int = 64
    action_dim: int = 12  # Algorithm selection + hyperparameter adjustments
    hidden_dims: Sequence[int] = (256, 256, 128)
    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    replay_buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    reward_convergence_weight: float = 0.4
    reward_quality_weight: float = 0.4
    reward_efficiency_weight: float = 0.2
    max_episode_length: int = 1000

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not (0 < self.learning_rate < 1):
            raise ValueError("learning_rate must be in (0, 1)")
        if not (0 < self.discount_factor <= 1):
            raise ValueError("discount_factor must be in (0, 1]")
        if not (0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.replay_buffer_size <= 0:
            raise ValueError("replay_buffer_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total_weight = (
            self.reward_convergence_weight
            + self.reward_quality_weight
            + self.reward_efficiency_weight
        )
        if not jnp.isclose(total_weight, 1.0):
            raise ValueError("Reward weights must sum to 1.0")


class DQNNetwork(nnx.Module):
    """Deep Q-Network for optimization strategy selection.

    The DQN takes optimization state as input and outputs Q-values for each
    possible action (algorithm selection, hyperparameter adjustment, etc.).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 128),
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize DQN network.

        Args:
            state_dim: Dimension of state representation
            action_dim: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            rngs: Random number generators
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []

        # Input layer
        layers.append(nnx.Linear(state_dim, hidden_dims[0], rngs=rngs))
        layers.append(nnx.relu)

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nnx.Linear(hidden_dims[i], hidden_dims[i + 1], rngs=rngs))
            layers.append(nnx.relu)

        # Output layer
        layers.append(nnx.Linear(hidden_dims[-1], action_dim, rngs=rngs))

        self.network = nnx.Sequential(*layers)

    def __call__(self, state: jax.Array) -> jax.Array:
        """Forward pass through DQN.

        Args:
            state: Optimization state representation

        Returns:
            Q-values for each action
        """
        return self.network(state)


class StateEncoder(nnx.Module):
    """Encoder for optimization problem state representation.

    Converts optimization problem features, convergence history, and resource
    constraints into a fixed-size state vector for the DQN.
    """

    def __init__(
        self,
        output_dim: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize state encoder.

        Args:
            output_dim: Dimension of encoded state
            rngs: Random number generators
        """
        super().__init__()

        self.output_dim = output_dim

        # Problem feature encoder
        self.problem_encoder = nnx.Sequential(
            nnx.Linear(32, 64, rngs=rngs),  # Problem parameters, constraints, etc.
            nnx.relu,
            nnx.Linear(64, 32, rngs=rngs),
        )

        # Convergence history encoder
        self.history_encoder = nnx.Sequential(
            nnx.Linear(16, 32, rngs=rngs),  # Recent objective values, gradients
            nnx.relu,
            nnx.Linear(32, 16, rngs=rngs),
        )

        # Resource encoder
        self.resource_encoder = nnx.Sequential(
            nnx.Linear(8, 16, rngs=rngs),  # Time, memory, computation remaining
            nnx.relu,
            nnx.Linear(16, 8, rngs=rngs),
        )

        # Final fusion layer
        self.fusion = nnx.Sequential(
            nnx.Linear(32 + 16 + 8, output_dim, rngs=rngs),
            nnx.tanh,
        )

    def __call__(
        self,
        problem_features: jax.Array,
        convergence_history: jax.Array,
        resource_constraints: jax.Array,
    ) -> jax.Array:
        """Encode optimization state.

        Args:
            problem_features: Problem-specific features (dimension, constraints, etc.)
            convergence_history: Recent optimization progress
            resource_constraints: Available computational resources

        Returns:
            Encoded state representation
        """
        # Encode individual components
        problem_encoded = self.problem_encoder(problem_features)
        history_encoded = self.history_encoder(convergence_history)
        resource_encoded = self.resource_encoder(resource_constraints)

        # Concatenate and fuse
        combined = jnp.concatenate([problem_encoded, history_encoded, resource_encoded])
        return self.fusion(combined)


@dataclass
class Experience:
    """Single experience for replay buffer."""

    state: jax.Array
    action: int
    reward: float
    next_state: jax.Array
    done: bool


class ExperienceReplayBuffer:
    """Experience replay buffer for DQN training.

    Stores optimization experiences and provides efficient sampling for training.
    """

    def __init__(self, capacity: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.position = 0

    def push(self, experience: Experience):
        """Add experience to buffer.

        Args:
            experience: Optimization experience to store
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Batch of sampled experiences
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {len(self.buffer)}"
            )

        # Use deterministic sampling for reproducibility in tests
        # In production, this could use JAX random for better randomness
        indices = jnp.arange(len(self.buffer))
        selected_indices = indices[-batch_size:]  # Take most recent experiences
        return [self.buffer[int(i)] for i in selected_indices]

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class RewardFunction:
    """Reward function for RL-based optimization.

    Computes rewards based on convergence speed, solution quality, and
    computational efficiency.
    """

    def __init__(self, config: RLOptimizationConfig):
        """Initialize reward function.

        Args:
            config: RL optimization configuration
        """
        self.config = config

    def compute_reward(
        self,
        objective_improvement: float,
        convergence_speed: float,
        computational_cost: float,
        constraint_violation: float = 0.0,
    ) -> float:
        """Compute reward for optimization step.

        Args:
            objective_improvement: Improvement in objective function value
            convergence_speed: Rate of convergence (higher is better)
            computational_cost: Computational resources used (lower is better)
            constraint_violation: Degree of constraint violation (lower is better)

        Returns:
            Computed reward value
        """
        # Convergence reward (based on objective improvement)
        convergence_reward = (
            jnp.tanh(objective_improvement) * self.config.reward_convergence_weight
        )

        # Quality reward (based on convergence speed)
        quality_reward = jnp.tanh(convergence_speed) * self.config.reward_quality_weight

        # Efficiency reward (based on computational cost)
        efficiency_reward = (
            jnp.tanh(1.0 / (computational_cost + 1e-8))
            * self.config.reward_efficiency_weight
        )

        # Penalty for constraint violations
        constraint_penalty = -10.0 * constraint_violation

        total_reward = (
            convergence_reward + quality_reward + efficiency_reward + constraint_penalty
        )

        return float(total_reward)


class RLOptimizationAgent(nnx.Module):
    """Reinforcement learning agent for optimization strategy selection.

    Uses Deep Q-Network to learn optimal optimization strategies based on
    problem characteristics and optimization progress.
    """

    def __init__(
        self,
        config: RLOptimizationConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize RL optimization agent.

        Args:
            config: RL optimization configuration
            rngs: Random number generators
        """
        super().__init__()

        self.config = config

        # Create networks
        self.state_encoder = StateEncoder(config.state_dim, rngs=rngs)
        self.dqn = DQNNetwork(
            config.state_dim, config.action_dim, config.hidden_dims, rngs=rngs
        )
        self.target_dqn = DQNNetwork(
            config.state_dim, config.action_dim, config.hidden_dims, rngs=rngs
        )

        # Initialize target network with same weights
        self._update_target_network()

        # Create optimizer
        self.optimizer = nnx.Optimizer(
            self.dqn, optax.adam(config.learning_rate), wrt=nnx.Param
        )

        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(config.replay_buffer_size)

        # Reward function
        self.reward_function = RewardFunction(config)

        # Training state
        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.episode_count = 0
        self.rng_key = rngs.params()

    def _update_target_network(self):
        """Update target network with current DQN weights."""
        # Copy parameters from main DQN to target DQN using nnx.update
        main_state = nnx.state(self.dqn)
        nnx.update(self.target_dqn, main_state)

    def encode_state(
        self,
        problem: OptimizationProblem,
        convergence_history: jax.Array,
        resource_usage: dict[str, float],
    ) -> jax.Array:
        """Encode current optimization state.

        Args:
            problem: Current optimization problem
            convergence_history: Recent optimization progress
            resource_usage: Current resource consumption

        Returns:
            Encoded state representation
        """
        # Extract problem features
        problem_features = jnp.array(
            [
                float(problem.dimension),
                float(len(problem.constraints) if problem.constraints else 0),
                *([0.0] * 30),  # Fixed size feature vector (30 features total)
            ]
        )

        # Pad convergence history to fixed size
        history_padded = jnp.pad(
            convergence_history, (0, max(0, 16 - convergence_history.shape[0]))
        )[:16]

        # Extract resource constraints
        resource_features = jnp.array(
            [
                resource_usage.get("time_remaining", 1.0),
                resource_usage.get("memory_usage", 0.0),
                resource_usage.get("computational_cost", 0.0),
                resource_usage.get("max_iterations_remaining", 1.0),
                0.0,
                0.0,
                0.0,
                0.0,  # Padding to reach 8 dimensions
            ]
        )

        return self.state_encoder(problem_features, history_padded, resource_features)

    def select_action(
        self,
        state: jax.Array,
        training: bool = True,
    ) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current optimization state
            training: Whether in training mode

        Returns:
            Selected action index
        """
        if training:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            if jax.random.uniform(subkey) < self.epsilon:
                # Random exploration
                self.rng_key, subkey = jax.random.split(self.rng_key)
                return int(jax.random.randint(subkey, (), 0, self.config.action_dim))
        # Greedy action selection
        q_values = self.dqn(state)
        return int(jnp.argmax(q_values))

    def store_experience(
        self,
        state: jax.Array,
        action: int,
        reward: float,
        next_state: jax.Array,
        done: bool,
    ):
        """Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is complete
        """
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)

    def train_step(self) -> dict[str, float]:
        """Perform single training step.

        Returns:
            Training metrics
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        # Sample batch from replay buffer
        experiences = self.replay_buffer.sample(self.config.batch_size)

        # Prepare batch data
        states = jnp.stack([exp.state for exp in experiences])
        actions = jnp.array([exp.action for exp in experiences])
        rewards = jnp.array([exp.reward for exp in experiences])
        next_states = jnp.stack([exp.next_state for exp in experiences])
        dones = jnp.array([exp.done for exp in experiences])

        # Compute target Q-values
        next_q_values = self.target_dqn(next_states)
        max_next_q_values = jnp.max(next_q_values, axis=1)
        targets = rewards + self.config.discount_factor * max_next_q_values * (
            1 - dones
        )

        # Define loss function
        def loss_fn(params):
            dqn_with_params = nnx.merge(nnx.graphdef(self.dqn), params)
            q_values = dqn_with_params(states)
            action_q_values = q_values[jnp.arange(len(actions)), actions]
            return jnp.mean(optax.l2_loss(action_q_values, targets))

        # Compute gradients and update
        loss, grads = nnx.value_and_grad(loss_fn)(nnx.state(self.dqn, nnx.Param))
        self.optimizer.update(self.dqn, grads)

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.config.target_update_freq == 0:
            self._update_target_network()

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end, self.epsilon * self.config.epsilon_decay
        )

        return {
            "loss": float(loss),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }


class ActionInterpreter:
    """Interprets DQN actions into optimization strategy modifications.

    Maps discrete action indices to specific optimization algorithms,
    hyperparameter adjustments, and stopping criteria.
    """

    def __init__(self):
        """Initialize action interpreter."""
        # Define action mappings
        self.action_mappings = {
            0: "switch_to_adam",
            1: "switch_to_sgd",
            2: "switch_to_lbfgs",
            3: "increase_learning_rate",
            4: "decrease_learning_rate",
            5: "increase_momentum",
            6: "decrease_momentum",
            7: "switch_to_adaptive_scheduler",
            8: "switch_to_constant_scheduler",
            9: "early_stop",
            10: "continue_optimization",
            11: "reset_to_traditional_solver",
        }

    def interpret_action(
        self,
        action: int,
        current_config: MetaSchedulerConfig,
    ) -> tuple[str, dict[str, Any]]:
        """Interpret action into optimization strategy modification.

        Args:
            action: Action index from DQN
            current_config: Current optimization configuration

        Returns:
            Tuple of (action_type, parameters)
        """
        action_type = self.action_mappings.get(action, "continue_optimization")

        # Define parameter modifications based on action
        if action_type == "increase_learning_rate":
            new_lr = min(current_config.base_learning_rate * 1.5, 1e-2)
            return action_type, {"learning_rate": new_lr}
        if action_type == "decrease_learning_rate":
            new_lr = max(current_config.base_learning_rate * 0.5, 1e-6)
            return action_type, {"learning_rate": new_lr}
        if action_type == "increase_momentum":
            new_momentum = min(
                0.99, current_config.patience * 0.1
            )  # Use patience as base for momentum
            return action_type, {"momentum": new_momentum}
        if action_type == "decrease_momentum":
            new_momentum = max(
                0.1, current_config.patience * 0.05
            )  # Use patience as base for momentum
            return action_type, {"momentum": new_momentum}
        return action_type, {}


class RLOptimizationEngine(nnx.Module):
    """Main RL-based optimization engine.

    Integrates the RL agent with existing L2O framework to provide
    intelligent optimization strategy selection and adaptation.
    """

    def __init__(
        self,
        config: RLOptimizationConfig,
        meta_optimizer: MetaOptimizer | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize RL optimization engine.

        Args:
            config: RL optimization configuration
            meta_optimizer: Existing meta-optimizer for integration
            rngs: Random number generators
        """
        super().__init__()

        self.config = config
        self.meta_optimizer = meta_optimizer or MetaOptimizer(
            config=MetaOptimizerConfig(), rngs=rngs
        )

        # RL components
        self.rl_agent = RLOptimizationAgent(config, rngs=rngs)
        self.action_interpreter = ActionInterpreter()

        # Performance tracking
        self.episode_rewards: list[float] = []
        self.optimization_metrics: list[dict[str, Any]] = []

    def solve_with_rl(
        self,
        problem: OptimizationProblem,
        max_iterations: int = 1000,
        training: bool = True,
    ) -> dict[str, Any]:
        """Solve optimization problem using RL-guided strategy selection.

        Args:
            problem: Optimization problem to solve
            max_iterations: Maximum optimization iterations
            training: Whether to train the RL agent

        Returns:
            Optimization results with RL metrics
        """
        # Initialize optimization state
        convergence_history = jnp.array([])
        resource_usage = {
            "time_remaining": 1.0,
            "memory_usage": 0.0,
            "computational_cost": 0.0,
            "max_iterations_remaining": 1.0,
        }

        # Initial state encoding
        state = self.rl_agent.encode_state(problem, convergence_history, resource_usage)

        episode_reward = 0.0
        optimization_results = []

        for iteration in range(min(max_iterations, self.config.max_episode_length)):
            # Select action
            action = self.rl_agent.select_action(state, training=training)

            # Interpret action
            action_type, params = self.action_interpreter.interpret_action(
                action, MetaSchedulerConfig()
            )

            # Execute optimization step with selected strategy
            step_result = self._execute_optimization_step(
                problem, action_type, params, iteration
            )

            # Compute reward
            reward = self.rl_agent.reward_function.compute_reward(
                step_result["objective_improvement"],
                step_result["convergence_speed"],
                step_result["computational_cost"],
                step_result.get("constraint_violation", 0.0),
            )

            episode_reward += reward

            # Update resource usage
            resource_usage["time_remaining"] = max(
                0.0, resource_usage["time_remaining"] - 0.001
            )
            resource_usage["computational_cost"] += step_result["computational_cost"]
            resource_usage["max_iterations_remaining"] = (
                max_iterations - iteration
            ) / max_iterations

            # Update convergence history
            convergence_history = jnp.append(
                convergence_history[-15:], step_result["objective_value"]
            )

            # Encode next state
            next_state = self.rl_agent.encode_state(
                problem, convergence_history, resource_usage
            )

            # Check termination
            done = (
                step_result["converged"]
                or iteration >= max_iterations - 1
                or action_type == "early_stop"
            )

            # Store experience for training
            if training:
                self.rl_agent.store_experience(state, action, reward, next_state, done)

                # Train agent
                if len(self.rl_agent.replay_buffer) >= self.config.batch_size:
                    train_metrics = self.rl_agent.train_step()
                    optimization_results.append(train_metrics)

            # Update state
            state = next_state

            if done:
                break

        # Store episode metrics
        if training:
            self.episode_rewards.append(episode_reward)
            self.rl_agent.episode_count += 1

        return {
            "solution": step_result.get("solution", jnp.zeros(problem.dimension)),
            "objective_value": step_result.get("objective_value", float("inf")),
            "converged": step_result.get("converged", False),
            "iterations": iteration + 1,
            "episode_reward": episode_reward,
            "rl_metrics": optimization_results,
            "action_sequence": [action_type],  # Could track full sequence
        }

    def _execute_optimization_step(
        self,
        problem: OptimizationProblem,
        action_type: str,
        params: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        """Execute single optimization step based on RL action.

        Args:
            problem: Optimization problem
            action_type: Type of optimization action
            params: Action parameters
            iteration: Current iteration number

        Returns:
            Step results including metrics
        """
        # For now, use simplified step execution
        # In practice, this would integrate with actual optimization algorithms

        # Simulate optimization step
        if hasattr(self, "_previous_objective"):
            objective_improvement = max(
                0.0, self._previous_objective - (iteration * 0.1)
            )
        else:
            objective_improvement = 1.0
            self._previous_objective: float = 10.0

        current_objective = max(0.1, 10.0 - iteration * 0.1)
        self._previous_objective = current_objective

        return {
            "solution": jnp.ones(problem.dimension) * (1.0 / (iteration + 1)),
            "objective_value": current_objective,
            "objective_improvement": objective_improvement,
            "convergence_speed": 1.0 / (iteration + 1),
            "computational_cost": 0.01,
            "converged": current_objective < 0.5,
            "constraint_violation": 0.0,
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get RL agent performance metrics.

        Returns:
            Performance metrics dictionary
        """
        if not self.episode_rewards:
            return {}

        return {
            "average_episode_reward": float(jnp.mean(jnp.array(self.episode_rewards))),
            "total_episodes": len(self.episode_rewards),
            "exploration_rate": self.rl_agent.epsilon,
            "training_steps": self.rl_agent.step_count,
            "replay_buffer_size": len(self.rl_agent.replay_buffer),
        }
