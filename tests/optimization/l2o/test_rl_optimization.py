"""Comprehensive test suite for reinforcement learning-based optimization.

Tests cover DQN implementation, state encoding, experience replay, reward function,
action interpretation, and integration with existing L2O framework.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.optimization.l2o.adaptive_schedulers import MetaSchedulerConfig
from opifex.optimization.l2o.parametric_solver import OptimizationProblem
from opifex.optimization.l2o.rl_optimization import (
    ActionInterpreter,
    DQNNetwork,
    Experience,
    ExperienceReplayBuffer,
    RewardFunction,
    RLOptimizationAgent,
    RLOptimizationConfig,
    RLOptimizationEngine,
    StateEncoder,
)


class TestRLOptimizationConfig:
    """Test suite for RL optimization configuration."""

    def test_rl_optimization_config_initialization(self):
        """Test basic configuration initialization."""
        config = RLOptimizationConfig()

        assert config.state_dim == 64
        assert config.action_dim == 12
        assert config.learning_rate == 1e-4
        assert config.discount_factor == 0.99
        assert config.epsilon_start == 1.0
        assert config.epsilon_end == 0.01
        assert config.replay_buffer_size == 10000
        assert config.batch_size == 32

        # Check reward weights sum to 1.0
        total_weight = (
            config.reward_convergence_weight
            + config.reward_quality_weight
            + config.reward_efficiency_weight
        )
        assert abs(total_weight - 1.0) < 1e-10

    def test_rl_optimization_config_custom_initialization(self):
        """Test configuration with custom parameters."""
        config = RLOptimizationConfig(
            state_dim=128,
            action_dim=16,
            learning_rate=5e-4,
            discount_factor=0.95,
            epsilon_start=0.8,
            epsilon_end=0.05,
            replay_buffer_size=5000,
            batch_size=64,
            reward_convergence_weight=0.5,
            reward_quality_weight=0.3,
            reward_efficiency_weight=0.2,
        )

        assert config.state_dim == 128
        assert config.action_dim == 16
        assert config.learning_rate == 5e-4
        assert config.discount_factor == 0.95
        assert config.epsilon_start == 0.8
        assert config.epsilon_end == 0.05
        assert config.replay_buffer_size == 5000
        assert config.batch_size == 64

        # Check custom reward weights
        total_weight = (
            config.reward_convergence_weight
            + config.reward_quality_weight
            + config.reward_efficiency_weight
        )
        assert abs(total_weight - 1.0) < 1e-10

    def test_rl_optimization_config_validation(self):
        """Test configuration parameter validation."""
        # Test invalid state_dim
        with pytest.raises(ValueError, match="state_dim must be positive"):
            RLOptimizationConfig(state_dim=0)

        # Test invalid action_dim
        with pytest.raises(ValueError, match="action_dim must be positive"):
            RLOptimizationConfig(action_dim=-1)

        # Test invalid learning_rate
        with pytest.raises(ValueError, match="learning_rate must be in \\(0, 1\\)"):
            RLOptimizationConfig(learning_rate=0.0)

        with pytest.raises(ValueError, match="learning_rate must be in \\(0, 1\\)"):
            RLOptimizationConfig(learning_rate=1.5)

        # Test invalid discount_factor
        with pytest.raises(ValueError, match="discount_factor must be in \\(0, 1\\]"):
            RLOptimizationConfig(discount_factor=0.0)

        with pytest.raises(ValueError, match="discount_factor must be in \\(0, 1\\]"):
            RLOptimizationConfig(discount_factor=1.5)

        # Test invalid epsilon values
        with pytest.raises(ValueError, match="epsilon values must satisfy"):
            RLOptimizationConfig(epsilon_start=0.5, epsilon_end=0.8)

        # Test invalid replay_buffer_size
        with pytest.raises(ValueError, match="replay_buffer_size must be positive"):
            RLOptimizationConfig(replay_buffer_size=0)

        # Test invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            RLOptimizationConfig(batch_size=-1)

        # Test invalid reward weights
        with pytest.raises(ValueError, match=r"Reward weights must sum to 1\.0"):
            RLOptimizationConfig(
                reward_convergence_weight=0.5,
                reward_quality_weight=0.3,
                reward_efficiency_weight=0.1,  # Sum = 0.9, not 1.0
            )


class TestDQNNetwork:
    """Test suite for Deep Q-Network implementation."""

    def test_dqn_network_initialization(self):
        """Test DQN network initialization."""
        rngs = nnx.Rngs(42)
        dqn = DQNNetwork(
            state_dim=64,
            action_dim=12,
            hidden_dims=(128, 128, 64),
            rngs=rngs,
        )

        assert dqn.state_dim == 64
        assert dqn.action_dim == 12
        assert isinstance(dqn.network, nnx.Sequential)

    def test_dqn_network_forward_pass(self):
        """Test DQN forward pass."""
        rngs = nnx.Rngs(42)
        dqn = DQNNetwork(state_dim=64, action_dim=12, rngs=rngs)

        # Test single state
        state = jnp.ones(64)
        q_values = dqn(state)

        assert q_values.shape == (12,)
        assert jnp.all(jnp.isfinite(q_values))

    def test_dqn_network_batch_forward_pass(self):
        """Test DQN batch forward pass."""
        rngs = nnx.Rngs(42)
        dqn = DQNNetwork(state_dim=64, action_dim=12, rngs=rngs)

        # Test batch of states
        batch_states = jnp.ones((32, 64))
        q_values = jax.vmap(dqn)(batch_states)

        assert q_values.shape == (32, 12)
        assert jnp.all(jnp.isfinite(q_values))

    def test_dqn_network_gradient_flow(self):
        """Test gradient flow through DQN."""
        rngs = nnx.Rngs(42)
        dqn = DQNNetwork(state_dim=64, action_dim=12, rngs=rngs)

        def loss_fn(params):
            dqn_with_params = nnx.merge(nnx.graphdef(dqn), params)
            state = jnp.ones(64)
            q_values = dqn_with_params(state)
            return jnp.sum(q_values**2)

        params = nnx.state(dqn, nnx.Param)
        loss, grads = nnx.value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads))


class TestStateEncoder:
    """Test suite for state encoder."""

    def test_state_encoder_initialization(self):
        """Test state encoder initialization."""
        rngs = nnx.Rngs(42)
        encoder = StateEncoder(output_dim=64, rngs=rngs)

        assert encoder.output_dim == 64
        assert isinstance(encoder.problem_encoder, nnx.Sequential)
        assert isinstance(encoder.history_encoder, nnx.Sequential)
        assert isinstance(encoder.resource_encoder, nnx.Sequential)
        assert isinstance(encoder.fusion, nnx.Sequential)

    def test_state_encoder_encoding(self):
        """Test state encoding functionality."""
        rngs = nnx.Rngs(42)
        encoder = StateEncoder(output_dim=64, rngs=rngs)

        # Create test inputs
        problem_features = jnp.ones(32)
        convergence_history = jnp.ones(16)
        resource_constraints = jnp.ones(8)

        # Encode state
        encoded_state = encoder(
            problem_features, convergence_history, resource_constraints
        )

        assert encoded_state.shape == (64,)
        assert jnp.all(jnp.isfinite(encoded_state))
        assert jnp.all(jnp.abs(encoded_state) <= 1.0)  # tanh activation bounds

    def test_state_encoder_deterministic(self):
        """Test that state encoder is deterministic."""
        rngs = nnx.Rngs(42)
        encoder = StateEncoder(output_dim=64, rngs=rngs)

        problem_features = jnp.ones(32)
        convergence_history = jnp.ones(16)
        resource_constraints = jnp.ones(8)

        # Encode same state twice
        state1 = encoder(problem_features, convergence_history, resource_constraints)
        state2 = encoder(problem_features, convergence_history, resource_constraints)

        assert jnp.allclose(state1, state2)

    def test_state_encoder_different_inputs(self):
        """Test state encoder with different inputs."""
        rngs = nnx.Rngs(42)
        encoder = StateEncoder(output_dim=64, rngs=rngs)

        # Test with different problem features
        problem_features1 = jnp.ones(32)
        problem_features2 = jnp.ones(32) * 2.0

        convergence_history = jnp.ones(16)
        resource_constraints = jnp.ones(8)

        state1 = encoder(problem_features1, convergence_history, resource_constraints)
        state2 = encoder(problem_features2, convergence_history, resource_constraints)

        # Different inputs should produce different states
        assert not jnp.allclose(state1, state2)


class TestExperienceReplayBuffer:
    """Test suite for experience replay buffer."""

    def test_experience_replay_buffer_initialization(self):
        """Test replay buffer initialization."""
        buffer = ExperienceReplayBuffer(capacity=1000)

        assert buffer.capacity == 1000
        assert len(buffer) == 0

    def test_experience_replay_buffer_push_and_sample(self):
        """Test pushing and sampling experiences."""
        buffer = ExperienceReplayBuffer(capacity=100)

        # Create and push experiences
        for i in range(50):
            experience = Experience(
                state=jnp.ones(64) * i,
                action=i % 12,
                reward=float(i),
                next_state=jnp.ones(64) * (i + 1),
                done=i == 49,
            )
            buffer.push(experience)

        assert len(buffer) == 50

        # Sample experiences
        sampled = buffer.sample(batch_size=10)
        assert len(sampled) == 10

        # Check that all sampled experiences are valid
        for exp in sampled:
            assert isinstance(exp, Experience)
            assert exp.state.shape == (64,)
            assert isinstance(exp.action, int)
            assert isinstance(exp.reward, float)
            assert exp.next_state.shape == (64,)
            assert isinstance(exp.done, bool)

    def test_experience_replay_buffer_capacity_limit(self):
        """Test replay buffer capacity limitation."""
        buffer = ExperienceReplayBuffer(capacity=10)

        # Push more experiences than capacity
        for i in range(20):
            experience = Experience(
                state=jnp.ones(64) * i,
                action=i % 12,
                reward=float(i),
                next_state=jnp.ones(64) * (i + 1),
                done=False,
            )
            buffer.push(experience)

        # Buffer should not exceed capacity
        assert len(buffer) == 10

        # Sample should work
        sampled = buffer.sample(batch_size=5)
        assert len(sampled) == 5

    def test_experience_replay_buffer_sample_insufficient_data(self):
        """Test sampling when insufficient data available."""
        buffer = ExperienceReplayBuffer(capacity=100)

        # Push only a few experiences
        for i in range(3):
            experience = Experience(
                state=jnp.ones(64),
                action=i,
                reward=1.0,
                next_state=jnp.ones(64),
                done=False,
            )
            buffer.push(experience)

        # Trying to sample more than available should raise error
        with pytest.raises(ValueError, match="Cannot sample"):
            buffer.sample(batch_size=10)


class TestRewardFunction:
    """Test suite for reward function."""

    def test_reward_function_initialization(self):
        """Test reward function initialization."""
        config = RLOptimizationConfig()
        reward_fn = RewardFunction(config)

        assert reward_fn.config == config

    def test_reward_function_compute_reward(self):
        """Test reward computation."""
        config = RLOptimizationConfig()
        reward_fn = RewardFunction(config)

        # Test positive reward
        reward = reward_fn.compute_reward(
            objective_improvement=1.0,
            convergence_speed=1.0,
            computational_cost=0.1,
            constraint_violation=0.0,
        )

        assert isinstance(reward, float)
        assert reward > 0  # Should be positive for good performance

    def test_reward_function_constraint_penalty(self):
        """Test constraint violation penalty."""
        config = RLOptimizationConfig()
        reward_fn = RewardFunction(config)

        # Test with constraint violation
        reward_with_violation = reward_fn.compute_reward(
            objective_improvement=1.0,
            convergence_speed=1.0,
            computational_cost=0.1,
            constraint_violation=0.5,
        )

        reward_without_violation = reward_fn.compute_reward(
            objective_improvement=1.0,
            convergence_speed=1.0,
            computational_cost=0.1,
            constraint_violation=0.0,
        )

        # Reward with violation should be lower
        assert reward_with_violation < reward_without_violation

    def test_reward_function_efficiency_component(self):
        """Test computational efficiency component."""
        config = RLOptimizationConfig()
        reward_fn = RewardFunction(config)

        # Test with low computational cost
        reward_efficient = reward_fn.compute_reward(
            objective_improvement=1.0,
            convergence_speed=1.0,
            computational_cost=0.01,
        )

        # Test with high computational cost
        reward_inefficient = reward_fn.compute_reward(
            objective_improvement=1.0,
            convergence_speed=1.0,
            computational_cost=10.0,
        )

        # More efficient should have higher reward
        assert reward_efficient > reward_inefficient


class TestRLOptimizationAgent:
    """Test suite for RL optimization agent."""

    def test_rl_optimization_agent_initialization(self):
        """Test RL agent initialization."""
        config = RLOptimizationConfig()
        rngs = nnx.Rngs(42)
        agent = RLOptimizationAgent(config, rngs=rngs)

        assert agent.config == config
        assert isinstance(agent.state_encoder, StateEncoder)
        assert isinstance(agent.dqn, DQNNetwork)
        assert isinstance(agent.target_dqn, DQNNetwork)
        assert isinstance(agent.replay_buffer, ExperienceReplayBuffer)
        assert isinstance(agent.reward_function, RewardFunction)
        assert agent.epsilon == config.epsilon_start
        assert agent.step_count == 0
        assert agent.episode_count == 0

    def test_rl_optimization_agent_state_encoding(self):
        """Test optimization state encoding."""
        config = RLOptimizationConfig()
        rngs = nnx.Rngs(42)
        agent = RLOptimizationAgent(config, rngs=rngs)

        # Create test problem
        problem = OptimizationProblem(
            problem_type="quadratic",
            dimension=5,
            constraints=None,
        )

        convergence_history = jnp.array([10.0, 8.0, 6.0, 4.0])
        resource_usage = {
            "time_remaining": 0.8,
            "memory_usage": 0.2,
            "computational_cost": 0.1,
            "max_iterations_remaining": 0.9,
        }

        # Encode state
        state = agent.encode_state(problem, convergence_history, resource_usage)

        assert state.shape == (config.state_dim,)
        assert jnp.all(jnp.isfinite(state))

    def test_rl_optimization_agent_action_selection(self):
        """Test action selection with epsilon-greedy policy."""
        config = RLOptimizationConfig(
            epsilon_start=0.01, epsilon_end=0.0
        )  # Greedy selection
        rngs = nnx.Rngs(42)
        agent = RLOptimizationAgent(config, rngs=rngs)

        state = jnp.ones(config.state_dim)

        # Test greedy action selection
        action = agent.select_action(state, training=False)
        assert isinstance(action, int)
        assert 0 <= action < config.action_dim

        # Same state should give same action in greedy mode
        action2 = agent.select_action(state, training=False)
        assert action == action2

    def test_rl_optimization_agent_experience_storage(self):
        """Test experience storage."""
        config = RLOptimizationConfig()
        rngs = nnx.Rngs(42)
        agent = RLOptimizationAgent(config, rngs=rngs)

        # Store experiences
        for i in range(10):
            agent.store_experience(
                state=jnp.ones(config.state_dim) * i,
                action=i % config.action_dim,
                reward=float(i),
                next_state=jnp.ones(config.state_dim) * (i + 1),
                done=i == 9,
            )

        assert len(agent.replay_buffer) == 10

    def test_rl_optimization_agent_training_step(self):
        """Test training step execution."""
        config = RLOptimizationConfig(batch_size=4)
        rngs = nnx.Rngs(42)
        agent = RLOptimizationAgent(config, rngs=rngs)

        # Store enough experiences for training
        for i in range(10):
            agent.store_experience(
                state=jnp.ones(config.state_dim) * i,
                action=i % config.action_dim,
                reward=float(i),
                next_state=jnp.ones(config.state_dim) * (i + 1),
                done=i == 9,
            )

        # Perform training step
        metrics = agent.train_step()

        assert "loss" in metrics
        assert "epsilon" in metrics
        assert "step_count" in metrics
        assert jnp.isfinite(metrics["loss"])
        assert agent.step_count == 1


class TestActionInterpreter:
    """Test suite for action interpreter."""

    def test_action_interpreter_initialization(self):
        """Test action interpreter initialization."""
        interpreter = ActionInterpreter()

        assert hasattr(interpreter, "action_mappings")
        assert len(interpreter.action_mappings) == 12
        assert 0 in interpreter.action_mappings
        assert 11 in interpreter.action_mappings

    def test_action_interpreter_interpret_action(self):
        """Test action interpretation."""
        interpreter = ActionInterpreter()
        config = MetaSchedulerConfig()

        # Test learning rate increase action
        action_type, params = interpreter.interpret_action(
            3, config
        )  # increase_learning_rate
        assert action_type == "increase_learning_rate"
        assert "learning_rate" in params
        assert params["learning_rate"] > config.base_learning_rate

        # Test learning rate decrease action
        action_type, params = interpreter.interpret_action(
            4, config
        )  # decrease_learning_rate
        assert action_type == "decrease_learning_rate"
        assert "learning_rate" in params
        assert params["learning_rate"] < config.base_learning_rate

    def test_action_interpreter_momentum_actions(self):
        """Test momentum adjustment actions."""
        interpreter = ActionInterpreter()
        config = MetaSchedulerConfig()

        # Test momentum increase
        action_type, params = interpreter.interpret_action(
            5, config
        )  # increase_momentum
        assert action_type == "increase_momentum"
        assert "momentum" in params

        # Test momentum decrease
        action_type, params = interpreter.interpret_action(
            6, config
        )  # decrease_momentum
        assert action_type == "decrease_momentum"
        assert "momentum" in params

    def test_action_interpreter_algorithm_switching(self):
        """Test algorithm switching actions."""
        interpreter = ActionInterpreter()
        config = MetaSchedulerConfig()

        # Test switch to Adam
        action_type, _ = interpreter.interpret_action(0, config)
        assert action_type == "switch_to_adam"

        # Test switch to SGD
        action_type, _ = interpreter.interpret_action(1, config)
        assert action_type == "switch_to_sgd"

        # Test switch to L-BFGS
        action_type, _ = interpreter.interpret_action(2, config)
        assert action_type == "switch_to_lbfgs"

    def test_action_interpreter_invalid_action(self):
        """Test handling of invalid action indices."""
        interpreter = ActionInterpreter()
        config = MetaSchedulerConfig()

        # Test invalid action index
        action_type, params = interpreter.interpret_action(999, config)
        assert action_type == "continue_optimization"
        assert params == {}


class TestRLOptimizationEngine:
    """Test suite for RL optimization engine."""

    def test_rl_optimization_engine_initialization(self):
        """Test RL optimization engine initialization."""
        config = RLOptimizationConfig()
        rngs = nnx.Rngs(42)
        engine = RLOptimizationEngine(config, rngs=rngs)

        assert engine.config == config
        assert isinstance(engine.rl_agent, RLOptimizationAgent)
        assert isinstance(engine.action_interpreter, ActionInterpreter)
        assert engine.episode_rewards == []
        assert engine.optimization_metrics == []

    def test_rl_optimization_engine_solve_with_rl(self):
        """Test RL-based optimization solving."""
        config = RLOptimizationConfig(max_episode_length=10)
        rngs = nnx.Rngs(42)
        engine = RLOptimizationEngine(config, rngs=rngs)

        # Create test problem
        problem = OptimizationProblem(
            problem_type="quadratic",
            dimension=3,
            constraints=None,
        )

        # Solve with RL
        result = engine.solve_with_rl(problem, max_iterations=10, training=False)

        assert "solution" in result
        assert "objective_value" in result
        assert "converged" in result
        assert "iterations" in result
        assert "episode_reward" in result
        assert "rl_metrics" in result

        assert result["solution"].shape == (problem.dimension,)
        assert isinstance(result["objective_value"], float)
        assert isinstance(result["converged"], bool)
        assert isinstance(result["iterations"], int)
        assert result["iterations"] <= 10

    def test_rl_optimization_engine_training_mode(self):
        """Test RL engine in training mode."""
        config = RLOptimizationConfig(max_episode_length=5, batch_size=2)
        rngs = nnx.Rngs(42)
        engine = RLOptimizationEngine(config, rngs=rngs)

        problem = OptimizationProblem(
            problem_type="quadratic",
            dimension=2,
            constraints=None,
        )

        # Solve multiple episodes to accumulate training data
        for _ in range(3):
            result = engine.solve_with_rl(problem, max_iterations=5, training=True)
            assert "episode_reward" in result

        # Check that episodes were recorded
        assert len(engine.episode_rewards) == 3
        assert engine.rl_agent.episode_count == 3

    def test_rl_optimization_engine_performance_metrics(self):
        """Test performance metrics collection."""
        config = RLOptimizationConfig(max_episode_length=5)
        rngs = nnx.Rngs(42)
        engine = RLOptimizationEngine(config, rngs=rngs)

        # Initially no metrics
        metrics = engine.get_performance_metrics()
        assert metrics == {}

        # Run some episodes
        problem = OptimizationProblem(
            problem_type="quadratic",
            dimension=2,
            constraints=None,
        )

        for _ in range(3):
            engine.solve_with_rl(problem, max_iterations=5, training=True)

        # Check metrics
        metrics = engine.get_performance_metrics()
        assert "average_episode_reward" in metrics
        assert "total_episodes" in metrics
        assert "exploration_rate" in metrics
        assert "training_steps" in metrics
        assert "replay_buffer_size" in metrics

        assert metrics["total_episodes"] == 3
        assert isinstance(metrics["average_episode_reward"], float)


class TestIntegrationWithExistingFramework:
    """Test integration with existing L2O framework."""

    def test_rl_optimization_config_compatibility(self):
        """Test RL config compatibility with existing framework."""
        rl_config = RLOptimizationConfig()
        scheduler_config = MetaSchedulerConfig()

        # Both configs should be usable together
        assert isinstance(rl_config.learning_rate, float)
        assert isinstance(scheduler_config.base_learning_rate, float)

        # RL config should not interfere with scheduler config
        assert hasattr(rl_config, "discount_factor")
        assert hasattr(scheduler_config, "patience")

    def test_rl_optimization_with_existing_problem(self):
        """Test RL optimization with existing optimization problems."""
        config = RLOptimizationConfig(max_episode_length=5)
        rngs = nnx.Rngs(42)
        engine = RLOptimizationEngine(config, rngs=rngs)

        # Use existing OptimizationProblem class
        problem = OptimizationProblem(
            problem_type="quadratic",
            dimension=4,
            constraints=None,
        )

        result = engine.solve_with_rl(problem, max_iterations=5, training=False)

        # Verify result format matches expectations
        assert result["solution"].shape == (4,)
        assert "objective_value" in result
        assert "converged" in result

    def test_rl_optimization_l2o_integration(self):
        """Test seamless integration with L2O framework."""
        config = RLOptimizationConfig()
        rngs = nnx.Rngs(42)

        # RL engine should work without breaking existing interfaces
        engine = RLOptimizationEngine(config, rngs=rngs)

        # Should have standard optimization methods
        assert hasattr(engine, "solve_with_rl")
        assert hasattr(engine, "get_performance_metrics")

        # Should work with standard problem format
        problem = OptimizationProblem(
            problem_type="quadratic",
            dimension=2,
            constraints=None,
        )

        result = engine.solve_with_rl(problem, training=False)
        assert isinstance(result, dict)
        assert "solution" in result


if __name__ == "__main__":
    pytest.main([__file__])
