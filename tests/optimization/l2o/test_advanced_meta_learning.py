"""Test-driven development tests for advanced L2O meta-learning algorithms.

These tests define the expected behavior of MAML, Reptile, gradient-based meta-learning,
and Meta-L2O integration for Phase 5.1.3 implementation.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

# Import the new advanced meta-learning components
from opifex.optimization.l2o.advanced_meta_learning import (
    GradientBasedMetaLearner,
    GradientBasedMetaLearningConfig,
    MAMLConfig,
    MAMLOptimizer,
    MetaL2OIntegration,
    ReptileConfig,
    ReptileOptimizer,
)

# Import existing L2O components for integration testing
from opifex.optimization.l2o.l2o_engine import L2OEngine, L2OEngineConfig
from opifex.optimization.l2o.parametric_solver import OptimizationProblem
from opifex.optimization.meta_optimization import MetaOptimizerConfig


class TestMAMLConfig:
    """Test cases for MAML configuration validation."""

    def test_maml_config_initialization(self):
        """Test MAML configuration with default values."""
        config = MAMLConfig()

        assert config.inner_learning_rate == 1e-3
        assert config.meta_learning_rate == 1e-4
        assert config.inner_steps == 5
        assert config.meta_batch_size == 8
        assert config.adaptation_steps == 10
        assert config.second_order is True
        assert config.enable_adaptation_rate_learning is True

    def test_maml_config_custom_initialization(self):
        """Test custom MAML configuration."""
        config = MAMLConfig(
            inner_learning_rate=5e-3,
            meta_learning_rate=2e-4,
            inner_steps=3,
            meta_batch_size=4,
            second_order=False,
        )

        assert config.inner_learning_rate == 5e-3
        assert config.meta_learning_rate == 2e-4
        assert config.inner_steps == 3
        assert config.meta_batch_size == 4
        assert config.second_order is False

    def test_maml_config_validation(self):
        """Test MAML configuration validation."""
        with pytest.raises(ValueError, match="Inner learning rate must be positive"):
            MAMLConfig(inner_learning_rate=-1e-3)

        with pytest.raises(ValueError, match="Meta learning rate must be positive"):
            MAMLConfig(meta_learning_rate=-1e-4)

        with pytest.raises(ValueError, match="Inner steps must be positive"):
            MAMLConfig(inner_steps=0)


class TestReptileConfig:
    """Test cases for Reptile configuration validation."""

    def test_reptile_config_initialization(self):
        """Test Reptile configuration with default values."""
        config = ReptileConfig()

        assert config.meta_learning_rate == 1e-3
        assert config.inner_learning_rate == 1e-2
        assert config.inner_steps == 10
        assert config.meta_batch_size == 16
        assert config.adaptation_momentum == 0.9
        assert config.task_sampling_strategy == "uniform"

    def test_reptile_config_custom_initialization(self):
        """Test custom Reptile configuration."""
        config = ReptileConfig(
            meta_learning_rate=5e-3,
            inner_learning_rate=2e-2,
            task_sampling_strategy="weighted",
            gradient_clipping=0.5,
        )

        assert config.meta_learning_rate == 5e-3
        assert config.inner_learning_rate == 2e-2
        assert config.task_sampling_strategy == "weighted"
        assert config.gradient_clipping == 0.5

    def test_reptile_config_validation(self):
        """Test Reptile configuration validation."""
        with pytest.raises(ValueError, match="Invalid sampling strategy"):
            ReptileConfig(task_sampling_strategy="invalid_strategy")


class TestGradientBasedMetaLearningConfig:
    """Test cases for gradient-based meta-learning configuration."""

    def test_gb_config_initialization(self):
        """Test gradient-based meta-learning configuration defaults."""
        config = GradientBasedMetaLearningConfig()

        assert config.optimizer_network_layers == [128, 64, 32]
        assert config.meta_learning_rate == 1e-4
        assert config.gradient_unroll_steps == 20
        assert config.learned_lr_bounds == (1e-6, 1.0)
        assert config.momentum_adaptation is True

    def test_gb_config_custom_initialization(self):
        """Test custom gradient-based meta-learning configuration."""
        config = GradientBasedMetaLearningConfig(
            optimizer_network_layers=[64, 32],
            gradient_unroll_steps=10,
            momentum_adaptation=False,
        )

        assert config.optimizer_network_layers == [64, 32]
        assert config.gradient_unroll_steps == 10
        assert config.momentum_adaptation is False


class TestMAMLOptimizer:
    """Test cases for MAML optimizer implementation."""

    @pytest.fixture
    def maml_config(self):
        """Fixture providing MAML configuration."""
        return MAMLConfig(
            inner_learning_rate=1e-3,
            meta_learning_rate=1e-4,
            inner_steps=3,
            meta_batch_size=4,
        )

    @pytest.fixture
    def l2o_config(self):
        """Fixture providing L2O engine configuration."""
        return L2OEngineConfig(solver_type="parametric")

    @pytest.fixture
    def maml_optimizer(self, maml_config, l2o_config):
        """Fixture providing MAML optimizer instance."""
        rngs = nnx.Rngs(42)
        return MAMLOptimizer(
            config=maml_config,
            l2o_config=l2o_config,
            optimizer_input_dim=20,
            optimizer_output_dim=10,
            rngs=rngs,
        )

    def test_maml_optimizer_initialization(self, maml_config, l2o_config):
        """Test MAML optimizer initialization."""
        rngs = nnx.Rngs(42)
        optimizer = MAMLOptimizer(
            config=maml_config,
            l2o_config=l2o_config,
            optimizer_input_dim=20,
            optimizer_output_dim=10,
            rngs=rngs,
        )

        assert optimizer.config == maml_config
        assert optimizer.l2o_config == l2o_config
        assert optimizer.input_dim == 20
        assert optimizer.output_dim == 10
        assert hasattr(optimizer, "meta_optimizer")
        assert optimizer.meta_parameters.value.shape == (10,)

    def test_maml_adaptation_to_new_task(self, maml_optimizer):
        """Test MAML adaptation to new optimization tasks."""
        # Create a simple quadratic optimization problem
        problem = OptimizationProblem(problem_type="quadratic", dimension=5)

        # Problem parameters: Q matrix + c vector
        Q_matrix = jnp.eye(5) * 2.0  # Positive definite
        c_vector = jnp.ones(5) * 0.5
        problem_params = jnp.concatenate([Q_matrix.flatten(), c_vector])

        # Adapt to new task
        adapted_params = maml_optimizer.adapt_to_new_task(
            problem, problem_params, adaptation_steps=5
        )

        assert adapted_params.shape == (maml_optimizer.output_dim,)
        assert jnp.isfinite(adapted_params).all()

    def test_maml_meta_learning_on_task_distribution(self, maml_optimizer):
        """Test MAML meta-learning across task distribution."""
        # Create a small task distribution
        task_distribution = []
        for i in range(4):
            problem = OptimizationProblem(problem_type="quadratic", dimension=5)
            Q_matrix = jnp.eye(5) * (1.0 + i * 0.5)
            c_vector = jnp.ones(5) * (0.1 + i * 0.1)
            problem_params = jnp.concatenate([Q_matrix.flatten(), c_vector])
            task_distribution.append((problem, problem_params))

        # Perform meta-learning step
        meta_state, meta_loss = maml_optimizer.meta_learn_on_task_distribution(
            task_distribution, None, 0
        )

        assert isinstance(meta_state, dict)
        assert "meta_loss" in meta_state
        assert "meta_gradients" in meta_state
        assert jnp.isfinite(meta_loss)

    def test_maml_adaptation_rate_learning(self, maml_config, l2o_config):
        """Test MAML with learned adaptation rates."""
        maml_config.enable_adaptation_rate_learning = True
        rngs = nnx.Rngs(42)
        optimizer = MAMLOptimizer(
            config=maml_config,
            l2o_config=l2o_config,
            optimizer_input_dim=20,
            optimizer_output_dim=10,
            rngs=rngs,
        )

        assert optimizer.adaptation_rate_network is not None

        # Test adaptation rate computation
        problem_params = jnp.ones(20)
        adaptation_rate = optimizer._get_adaptation_rate(problem_params)

        assert jnp.isfinite(adaptation_rate)
        assert adaptation_rate > 0


class TestReptileOptimizer:
    """Test cases for Reptile optimizer implementation."""

    @pytest.fixture
    def reptile_config(self):
        """Fixture providing Reptile configuration."""
        return ReptileConfig(
            meta_learning_rate=1e-3,
            inner_learning_rate=1e-2,
            inner_steps=5,
            meta_batch_size=8,
        )

    @pytest.fixture
    def l2o_config(self):
        """Fixture providing L2O engine configuration."""
        return L2OEngineConfig(solver_type="parametric")

    @pytest.fixture
    def reptile_optimizer(self, reptile_config, l2o_config):
        """Fixture providing Reptile optimizer instance."""
        rngs = nnx.Rngs(42)
        return ReptileOptimizer(
            config=reptile_config,
            l2o_config=l2o_config,
            optimizer_input_dim=15,
            optimizer_output_dim=8,
            rngs=rngs,
        )

    def test_reptile_optimizer_initialization(self, reptile_config, l2o_config):
        """Test Reptile optimizer initialization."""
        rngs = nnx.Rngs(42)
        optimizer = ReptileOptimizer(
            config=reptile_config,
            l2o_config=l2o_config,
            optimizer_input_dim=15,
            optimizer_output_dim=8,
            rngs=rngs,
        )

        assert optimizer.config == reptile_config
        assert optimizer.l2o_config == l2o_config
        assert optimizer.input_dim == 15
        assert optimizer.output_dim == 8
        assert optimizer.meta_parameters.value.shape == (8,)
        assert optimizer.momentum_buffer.value.shape == (8,)

    def test_reptile_meta_learning_step(self, reptile_optimizer):
        """Test Reptile meta-learning step."""
        # Create task distribution for meta-learning
        task_distribution = []
        for i in range(6):
            problem = OptimizationProblem(problem_type="linear", dimension=4)
            c_vector = jnp.ones(4) * (0.5 + i * 0.2)
            task_distribution.append((problem, c_vector))

        # Perform Reptile meta-learning step
        metrics = reptile_optimizer.meta_learn_reptile_step(task_distribution, 0)

        assert isinstance(metrics, dict)
        assert "meta_loss" in metrics
        assert "meta_gradient_norm" in metrics
        assert "momentum_norm" in metrics
        assert jnp.isfinite(metrics["meta_loss"])
        assert jnp.isfinite(metrics["meta_gradient_norm"])

    def test_reptile_task_adaptation(self, reptile_optimizer):
        """Test Reptile adaptation to individual tasks."""
        problem = OptimizationProblem(problem_type="linear", dimension=4)
        c_vector = jnp.array([1.0, 2.0, 0.5, 1.5])

        adapted_params = reptile_optimizer.adapt_to_task(
            problem, c_vector, adaptation_steps=5
        )

        assert adapted_params.shape == (reptile_optimizer.output_dim,)
        assert jnp.isfinite(adapted_params).all()

    def test_reptile_gradient_clipping(self, reptile_optimizer):
        """Test gradient clipping functionality."""
        gradients = jnp.array([10.0, -15.0, 5.0, -8.0])
        clip_norm = 1.0

        clipped_gradients = reptile_optimizer._clip_gradients(gradients, clip_norm)

        grad_norm = jnp.linalg.norm(clipped_gradients)
        assert grad_norm <= clip_norm + 1e-6  # Allow small numerical tolerance


class TestGradientBasedMetaLearner:
    """Test cases for gradient-based meta-learning implementation."""

    @pytest.fixture
    def gb_config(self):
        """Fixture providing gradient-based meta-learning configuration."""
        return GradientBasedMetaLearningConfig(
            optimizer_network_layers=[64, 32],
            gradient_unroll_steps=10,
        )

    @pytest.fixture
    def l2o_config(self):
        """Fixture providing L2O engine configuration."""
        return L2OEngineConfig(solver_type="parametric")

    @pytest.fixture
    def gb_meta_learner(self, gb_config, l2o_config):
        """Fixture providing gradient-based meta-learner instance."""
        rngs = nnx.Rngs(42)
        return GradientBasedMetaLearner(
            config=gb_config, l2o_config=l2o_config, problem_dim=6, rngs=rngs
        )

    def test_gb_meta_learner_initialization(self, gb_config, l2o_config):
        """Test gradient-based meta-learner initialization."""
        rngs = nnx.Rngs(42)
        learner = GradientBasedMetaLearner(
            config=gb_config, l2o_config=l2o_config, problem_dim=6, rngs=rngs
        )

        assert learner.config == gb_config
        assert learner.l2o_config == l2o_config
        assert learner.problem_dim == 6
        assert hasattr(learner, "optimizer_network")
        assert hasattr(learner, "lr_network")

    def test_gb_learned_update_computation(self, gb_meta_learner):
        """Test computation of learned parameter updates."""
        gradients = jnp.array([0.1, -0.2, 0.05, -0.1, 0.15, -0.05])
        previous_update = jnp.array([0.01, -0.02, 0.005, -0.01, 0.015, -0.005])
        loss_history = jnp.array([1.0, 0.8, 0.6])
        problem_features = jnp.ones(7)

        update, metrics = gb_meta_learner.compute_learned_update(
            gradients, previous_update, loss_history, problem_features, step=1
        )

        assert update.shape == (gb_meta_learner.problem_dim,)
        assert jnp.isfinite(update).all()
        assert isinstance(metrics, dict)
        assert "learned_lr" in metrics
        assert "update_norm" in metrics
        assert "gradient_norm" in metrics

    def test_gb_adaptive_learning_rate(self, gb_meta_learner):
        """Test adaptive learning rate computation."""
        # Prepare input features
        input_features = jnp.ones(28)  # Based on _prepare_optimizer_input logic

        learned_lr = gb_meta_learner._compute_adaptive_learning_rate(input_features)

        assert jnp.isfinite(learned_lr)
        assert learned_lr >= gb_meta_learner.config.learned_lr_bounds[0]
        assert learned_lr <= gb_meta_learner.config.learned_lr_bounds[1]

    def test_gb_optimization_history_update(self, gb_meta_learner):
        """Test optimization history buffer updates."""
        update = jnp.array([0.1, -0.05, 0.02, -0.08, 0.04, -0.01])
        step = 3

        # Update history
        gb_meta_learner._update_optimization_history(update, step)

        # Check that history was updated
        history_shape = gb_meta_learner.optimization_history.value.shape
        expected_shape = (
            gb_meta_learner.config.gradient_unroll_steps,
            gb_meta_learner.problem_dim,
        )
        assert history_shape == expected_shape


class TestMetaL2OIntegration:
    """Test cases for Meta-L2O integration framework."""

    @pytest.fixture
    def l2o_engine(self):
        """Fixture providing L2O engine for integration."""
        rngs = nnx.Rngs(42)
        l2o_config = L2OEngineConfig(solver_type="parametric")
        meta_config = MetaOptimizerConfig(meta_algorithm="l2o")

        return L2OEngine(l2o_config=l2o_config, meta_config=meta_config, rngs=rngs)

    @pytest.fixture
    def maml_config(self):
        """Fixture providing MAML configuration."""
        return MAMLConfig(inner_steps=3, meta_batch_size=4)

    @pytest.fixture
    def reptile_config(self):
        """Fixture providing Reptile configuration."""
        return ReptileConfig(inner_steps=5, meta_batch_size=6)

    @pytest.fixture
    def gb_config(self):
        """Fixture providing gradient-based meta-learning configuration."""
        return GradientBasedMetaLearningConfig(gradient_unroll_steps=8)

    def test_meta_l2o_integration_initialization(
        self, l2o_engine, maml_config, reptile_config, gb_config
    ):
        """Test Meta-L2O integration initialization."""
        rngs = nnx.Rngs(42)
        integration = MetaL2OIntegration(
            l2o_engine=l2o_engine,
            maml_config=maml_config,
            reptile_config=reptile_config,
            gb_config=gb_config,
            rngs=rngs,
        )

        assert integration.l2o_engine == l2o_engine
        assert integration.maml_config is not None
        assert integration.reptile_config is not None
        assert integration.gb_config is not None
        assert integration.meta_learning_active is True

    def test_meta_l2o_integration_maml_only(self, l2o_engine, maml_config):
        """Test Meta-L2O integration with MAML only."""
        rngs = nnx.Rngs(42)
        integration = MetaL2OIntegration(
            l2o_engine=l2o_engine, maml_config=maml_config, rngs=rngs
        )

        assert integration.maml_config is not None
        assert integration.reptile_config is None
        assert integration.gb_config is None

    def test_meta_l2o_solve_with_meta_learning(
        self, l2o_engine, maml_config, reptile_config
    ):
        """Test solving problems with meta-learning strategies."""
        rngs = nnx.Rngs(42)
        integration = MetaL2OIntegration(
            l2o_engine=l2o_engine,
            maml_config=maml_config,
            reptile_config=reptile_config,
            rngs=rngs,
        )

        # Test problem
        problem = OptimizationProblem(problem_type="quadratic", dimension=5)
        Q_matrix = jnp.eye(5) * 1.5
        c_vector = jnp.ones(5) * 0.3
        problem_params = jnp.concatenate([Q_matrix.flatten(), c_vector])

        # Test with different meta-learning strategies
        for strategy in ["maml", "reptile", "auto", "fallback"]:
            solution, metrics = integration.solve_with_meta_learning(
                problem, problem_params, meta_learning_strategy=strategy
            )

            assert solution.shape == (5,)  # Based on problem dimension
            assert jnp.isfinite(solution).all()
            assert isinstance(metrics, dict)
            assert "meta_learning_strategy" in metrics
            assert "solve_time" in metrics

    def test_meta_l2o_strategy_selection(self, l2o_engine):
        """Test automatic meta-learning strategy selection."""
        rngs = nnx.Rngs(42)
        integration = MetaL2OIntegration(l2o_engine=l2o_engine, rngs=rngs)

        # Test strategy selection for different problem types
        quadratic_problem = OptimizationProblem(problem_type="quadratic", dimension=5)
        linear_problem = OptimizationProblem(problem_type="linear", dimension=5)
        nonlinear_problem = OptimizationProblem(problem_type="nonlinear", dimension=5)

        quad_strategy = integration._select_meta_learning_strategy(quadratic_problem)
        linear_strategy = integration._select_meta_learning_strategy(linear_problem)
        nonlinear_strategy = integration._select_meta_learning_strategy(
            nonlinear_problem
        )

        assert quad_strategy in ["maml", "reptile", "gradient_based", "fallback"]
        assert linear_strategy in ["maml", "reptile", "gradient_based", "fallback"]
        assert nonlinear_strategy in ["maml", "reptile", "gradient_based", "fallback"]

    def test_meta_l2o_experience_storage(self, l2o_engine):
        """Test optimization experience storage and management."""
        rngs = nnx.Rngs(42)
        integration = MetaL2OIntegration(l2o_engine=l2o_engine, rngs=rngs)

        # Store some optimization experiences
        for i in range(5):
            problem = OptimizationProblem(problem_type="linear", dimension=3)
            problem_params = jnp.ones(3) * (i + 1)
            solution = jnp.ones(3) * (i + 1)  # Mock solution
            metrics = {"solve_time": 0.1 * (i + 1)}

            integration._store_optimization_experience(
                problem, problem_params, solution, metrics
            )

        assert len(integration.experience_buffer) == 5

        # Test trajectory conversion
        trajectories = integration._convert_experience_to_trajectories()
        assert len(trajectories) == 5
        assert all("problem_type" in traj for traj in trajectories)

    def test_meta_l2o_trigger_meta_learning_update(
        self, l2o_engine, maml_config, reptile_config
    ):
        """Test triggering meta-learning updates across algorithms."""
        rngs = nnx.Rngs(42)
        integration = MetaL2OIntegration(
            l2o_engine=l2o_engine,
            maml_config=maml_config,
            reptile_config=reptile_config,
            rngs=rngs,
        )

        # Create task distribution for meta-learning
        task_distribution = []
        for i in range(4):
            problem = OptimizationProblem(problem_type="quadratic", dimension=3)
            Q_matrix = jnp.eye(3) * (1.0 + i * 0.3)
            c_vector = jnp.ones(3) * (0.2 + i * 0.1)
            problem_params = jnp.concatenate([Q_matrix.flatten(), c_vector])
            task_distribution.append((problem, problem_params))

        # Trigger meta-learning updates
        meta_results = integration.trigger_meta_learning_update(task_distribution)

        assert isinstance(meta_results, dict)
        assert "maml" in meta_results or "reptile" in meta_results
        # MAML and Reptile should have results if they were initialized


class TestIntegrationWithExistingFramework:
    """Test integration with existing L2O and meta-optimization frameworks."""

    def test_l2o_engine_integration_compatibility(self):
        """Test that advanced meta-learning integrates seamlessly with L2O engine."""
        rngs = nnx.Rngs(42)

        # Create base L2O engine
        l2o_config = L2OEngineConfig(solver_type="parametric")
        meta_config = MetaOptimizerConfig(meta_algorithm="l2o")
        l2o_engine = L2OEngine(
            l2o_config=l2o_config, meta_config=meta_config, rngs=rngs
        )

        # Test that engine works without meta-learning
        problem = OptimizationProblem(problem_type="linear", dimension=3)
        problem_params = jnp.array([1.0, 2.0, 0.5])

        solution = l2o_engine.solve_parametric_problem(problem, problem_params)
        assert solution.shape == (3,)  # Based on problem dimension
        assert jnp.isfinite(solution).all()

    def test_enhanced_l2o_with_meta_learning(self):
        """Test enhanced L2O capabilities with meta-learning."""
        rngs = nnx.Rngs(42)

        # Create enhanced L2O with meta-learning
        l2o_config = L2OEngineConfig(solver_type="parametric")
        meta_config = MetaOptimizerConfig(meta_algorithm="l2o")
        l2o_engine = L2OEngine(
            l2o_config=l2o_config, meta_config=meta_config, rngs=rngs
        )

        maml_config = MAMLConfig(inner_steps=2, meta_batch_size=2)
        integration = MetaL2OIntegration(
            l2o_engine=l2o_engine, maml_config=maml_config, rngs=rngs
        )

        # Test enhanced solving capabilities
        problem = OptimizationProblem(problem_type="quadratic", dimension=4)
        Q_matrix = jnp.eye(4) * 2.0
        c_vector = jnp.ones(4) * 0.4
        problem_params = jnp.concatenate([Q_matrix.flatten(), c_vector])

        solution, metrics = integration.solve_with_meta_learning(
            problem, problem_params, meta_learning_strategy="auto"
        )

        assert solution.shape == (4,)
        assert jnp.isfinite(solution).all()
        assert "meta_learning_strategy" in metrics
        assert "solve_time" in metrics

    def test_performance_comparison_framework(self):
        """Test performance comparison between different approaches."""
        rngs = nnx.Rngs(42)

        # Setup components
        l2o_config = L2OEngineConfig(solver_type="parametric")
        meta_config = MetaOptimizerConfig(meta_algorithm="l2o")
        l2o_engine = L2OEngine(
            l2o_config=l2o_config, meta_config=meta_config, rngs=rngs
        )

        # Test problem
        problem = OptimizationProblem(problem_type="linear", dimension=3)
        problem_params = jnp.array([1.0, 0.5, 2.0])

        # Base L2O solution
        base_solution = l2o_engine.solve_parametric_problem(problem, problem_params)

        # Enhanced meta-learning solution
        maml_config = MAMLConfig(inner_steps=2)
        integration = MetaL2OIntegration(
            l2o_engine=l2o_engine, maml_config=maml_config, rngs=rngs
        )
        enhanced_solution, enhanced_metrics = integration.solve_with_meta_learning(
            problem, problem_params, meta_learning_strategy="maml"
        )

        # Both should produce valid solutions
        assert base_solution.shape == enhanced_solution.shape
        assert jnp.isfinite(base_solution).all()
        assert jnp.isfinite(enhanced_solution).all()
        assert isinstance(enhanced_metrics, dict)
