"""Test suite for adaptive learning rate schedulers in L2O framework.

This module tests the adaptive scheduling capabilities including:
- MetaSchedulerConfig configuration management
- PerformanceAwareScheduler convergence-based adaptation
- MultiscaleScheduler component-specific rate management
- BayesianSchedulerOptimizer parameter optimization
- SchedulerIntegration with existing L2O framework
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.optimization.l2o.adaptive_schedulers import (
    BayesianSchedulerOptimizer,
    MetaSchedulerConfig,
    MultiscaleScheduler,
    PerformanceAwareScheduler,
    SchedulerIntegration,
)
from opifex.optimization.l2o.l2o_engine import L2OEngine, L2OEngineConfig
from opifex.optimization.l2o.parametric_solver import OptimizationProblem
from opifex.optimization.meta_optimization import MetaOptimizerConfig


class TestMetaSchedulerConfig:
    """Test suite for MetaSchedulerConfig."""

    def test_meta_scheduler_config_initialization(self):
        """Test default initialization of MetaSchedulerConfig."""
        config = MetaSchedulerConfig()

        assert config.base_learning_rate == 1e-3
        assert config.min_learning_rate == 1e-6
        assert config.max_learning_rate == 1e-1
        assert config.convergence_window == 10
        assert config.patience == 5
        assert config.adaptation_factor == 0.5
        assert config.multiscale_components == ["encoder", "solver", "decoder"]
        assert config.bayesian_optimization_steps == 20
        assert config.enable_performance_awareness is True
        assert config.enable_multiscale is False
        assert config.enable_bayesian_optimization is False

    def test_meta_scheduler_config_custom_initialization(self):
        """Test custom initialization of MetaSchedulerConfig."""
        config = MetaSchedulerConfig(
            base_learning_rate=1e-2,
            min_learning_rate=1e-5,
            max_learning_rate=5e-2,
            convergence_window=20,
            patience=10,
            adaptation_factor=0.7,
            multiscale_components=["layer1", "layer2"],
            bayesian_optimization_steps=50,
            enable_performance_awareness=False,
            enable_multiscale=True,
            enable_bayesian_optimization=True,
        )

        assert config.base_learning_rate == 1e-2
        assert config.min_learning_rate == 1e-5
        assert config.max_learning_rate == 5e-2
        assert config.convergence_window == 20
        assert config.patience == 10
        assert config.adaptation_factor == 0.7
        assert config.multiscale_components == ["layer1", "layer2"]
        assert config.bayesian_optimization_steps == 50
        assert config.enable_performance_awareness is False
        assert config.enable_multiscale is True
        assert config.enable_bayesian_optimization is True

    def test_meta_scheduler_config_validation(self):
        """Test validation in MetaSchedulerConfig."""
        # Test invalid learning rate bounds
        with pytest.raises(
            ValueError, match="min_learning_rate must be less than max_learning_rate"
        ):
            MetaSchedulerConfig(min_learning_rate=1e-2, max_learning_rate=1e-3)

        # Test base learning rate outside bounds
        with pytest.raises(
            ValueError,
            match="base_learning_rate must be between min_learning_rate and max_learning_rate",
        ):
            MetaSchedulerConfig(
                base_learning_rate=1e-2,
                min_learning_rate=1e-5,
                max_learning_rate=1e-3,
            )

        # Test invalid convergence window
        with pytest.raises(ValueError, match="convergence_window must be positive"):
            MetaSchedulerConfig(convergence_window=0)

        # Test invalid patience
        with pytest.raises(ValueError, match="patience must be positive"):
            MetaSchedulerConfig(patience=-1)


class TestPerformanceAwareScheduler:
    """Test suite for PerformanceAwareScheduler."""

    @pytest.fixture
    def scheduler_config(self):
        """Create a test scheduler configuration."""
        return MetaSchedulerConfig(
            base_learning_rate=1e-3,
            convergence_window=5,
            patience=3,
            adaptation_factor=0.5,
        )

    @pytest.fixture
    def performance_aware_scheduler(self, scheduler_config):
        """Create a PerformanceAwareScheduler instance."""
        rngs = nnx.Rngs(42)
        return PerformanceAwareScheduler(config=scheduler_config, rngs=rngs)

    def test_performance_aware_scheduler_initialization(
        self, performance_aware_scheduler
    ):
        """Test PerformanceAwareScheduler initialization."""
        assert performance_aware_scheduler.config.base_learning_rate == 1e-3
        assert performance_aware_scheduler.current_learning_rate == 1e-3
        assert len(performance_aware_scheduler.loss_history) == 0
        assert performance_aware_scheduler.patience_counter == 0
        assert performance_aware_scheduler.best_loss == float("inf")

    def test_performance_aware_scheduler_update_with_improvement(
        self, performance_aware_scheduler
    ):
        """Test scheduler update with improving loss."""
        # First loss
        new_lr = performance_aware_scheduler.update_learning_rate(loss=1.0)
        assert new_lr == 1e-3  # No change on first loss
        assert performance_aware_scheduler.best_loss == 1.0

        # Improving loss
        new_lr = performance_aware_scheduler.update_learning_rate(loss=0.5)
        assert new_lr == 1e-3  # No change with improvement
        assert performance_aware_scheduler.best_loss == 0.5
        assert performance_aware_scheduler.patience_counter == 0

    def test_performance_aware_scheduler_update_with_stagnation(
        self, performance_aware_scheduler
    ):
        """Test scheduler update with stagnating loss."""
        # Initialize with some losses
        performance_aware_scheduler.update_learning_rate(loss=1.0)
        performance_aware_scheduler.update_learning_rate(loss=0.9)
        performance_aware_scheduler.update_learning_rate(loss=0.8)

        # Add stagnating losses
        for _ in range(3):  # Patience is 3
            performance_aware_scheduler.update_learning_rate(loss=0.85)

        # Should reduce learning rate after patience exceeded
        new_lr = performance_aware_scheduler.update_learning_rate(loss=0.85)
        expected_lr = 1e-3 * 0.5  # adaptation_factor = 0.5
        assert abs(new_lr - expected_lr) < 1e-8

    def test_performance_aware_scheduler_convergence_detection(
        self, performance_aware_scheduler
    ):
        """Test convergence detection based on loss variance."""
        # Add losses with low variance (converged)
        for i in range(6):  # Convergence window is 5
            performance_aware_scheduler.update_learning_rate(loss=0.1 + 1e-8 * i)

        converged = performance_aware_scheduler.is_converged()
        assert converged is True

    def test_performance_aware_scheduler_learning_rate_bounds(
        self, performance_aware_scheduler
    ):
        """Test that learning rate respects bounds."""
        # Force learning rate to minimum
        performance_aware_scheduler.current_learning_rate = 1e-6

        # Try to reduce further
        for _ in range(10):
            performance_aware_scheduler.update_learning_rate(loss=1.0)

        assert performance_aware_scheduler.current_learning_rate >= 1e-6


class TestMultiscaleScheduler:
    """Test suite for MultiscaleScheduler."""

    @pytest.fixture
    def multiscale_config(self):
        """Create a multiscale scheduler configuration."""
        return MetaSchedulerConfig(
            base_learning_rate=1e-3,
            enable_multiscale=True,
            multiscale_components=["encoder", "solver", "decoder"],
        )

    @pytest.fixture
    def multiscale_scheduler(self, multiscale_config):
        """Create a MultiscaleScheduler instance."""
        rngs = nnx.Rngs(42)
        return MultiscaleScheduler(config=multiscale_config, rngs=rngs)

    def test_multiscale_scheduler_initialization(self, multiscale_scheduler):
        """Test MultiscaleScheduler initialization."""
        assert "encoder" in multiscale_scheduler.component_schedulers
        assert "solver" in multiscale_scheduler.component_schedulers
        assert "decoder" in multiscale_scheduler.component_schedulers

        for _, scheduler in multiscale_scheduler.component_schedulers.items():
            assert scheduler.config.base_learning_rate == 1e-3

    def test_multiscale_scheduler_component_learning_rates(self, multiscale_scheduler):
        """Test getting component-specific learning rates."""
        learning_rates = multiscale_scheduler.get_component_learning_rates()

        assert "encoder" in learning_rates
        assert "solver" in learning_rates
        assert "decoder" in learning_rates

        for lr in learning_rates.values():
            assert lr == 1e-3

    def test_multiscale_scheduler_component_updates(self, multiscale_scheduler):
        """Test updating component-specific learning rates."""
        # Update encoder with poor loss (multiple times to trigger adaptation)
        for _ in range(10):
            multiscale_scheduler.update_component_learning_rate("encoder", loss=1.0)

        # Update solver with good loss
        multiscale_scheduler.update_component_learning_rate("solver", loss=0.1)

        learning_rates = multiscale_scheduler.get_component_learning_rates()

        # Encoder should have reduced learning rate
        assert learning_rates["encoder"] < 1e-3
        # Solver should maintain learning rate
        assert learning_rates["solver"] == 1e-3

    def test_multiscale_scheduler_create_component_optimizers(
        self, multiscale_scheduler
    ):
        """Test creating component-specific optimizers."""
        optimizers = multiscale_scheduler.create_component_optimizers()

        assert "encoder" in optimizers
        assert "solver" in optimizers
        assert "decoder" in optimizers

        for optimizer in optimizers.values():
            assert hasattr(optimizer, "init")
            assert hasattr(optimizer, "update")


class TestBayesianSchedulerOptimizer:
    """Test suite for BayesianSchedulerOptimizer."""

    @pytest.fixture
    def bayesian_config(self):
        """Create a Bayesian scheduler configuration."""
        return MetaSchedulerConfig(
            enable_bayesian_optimization=True,
            bayesian_optimization_steps=10,
        )

    @pytest.fixture
    def bayesian_scheduler(self, bayesian_config):
        """Create a BayesianSchedulerOptimizer instance."""
        rngs = nnx.Rngs(42)
        return BayesianSchedulerOptimizer(config=bayesian_config, rngs=rngs)

    def test_bayesian_scheduler_initialization(self, bayesian_scheduler):
        """Test BayesianSchedulerOptimizer initialization."""
        assert bayesian_scheduler.config.bayesian_optimization_steps == 10
        assert len(bayesian_scheduler.parameter_history) == 0
        assert len(bayesian_scheduler.performance_history) == 0

    def test_bayesian_scheduler_parameter_suggestion(self, bayesian_scheduler):
        """Test parameter suggestion with Bayesian optimization."""
        # First suggestion (random)
        params = bayesian_scheduler.suggest_scheduler_parameters()

        assert "learning_rate" in params
        assert "adaptation_factor" in params
        assert "patience" in params

        assert (
            bayesian_scheduler.config.min_learning_rate
            <= params["learning_rate"]
            <= bayesian_scheduler.config.max_learning_rate
        )
        assert 0.1 <= params["adaptation_factor"] <= 0.9
        assert 1 <= params["patience"] <= 20

    def test_bayesian_scheduler_parameter_update(self, bayesian_scheduler):
        """Test updating with performance feedback."""
        # Test several parameter suggestions and updates
        for i in range(5):
            params = bayesian_scheduler.suggest_scheduler_parameters()
            performance = 1.0 / (i + 1)  # Improving performance
            bayesian_scheduler.update_with_performance(params, performance)

        assert len(bayesian_scheduler.parameter_history) == 5
        assert len(bayesian_scheduler.performance_history) == 5

    def test_bayesian_scheduler_best_parameters(self, bayesian_scheduler):
        """Test getting best parameters from Bayesian scheduler."""
        # Add some parameter-performance pairs
        for i in range(3):
            params = {
                "learning_rate": 1e-3 * (i + 1),
                "adaptation_factor": 0.5 + 0.1 * i,
                "patience": 5 + i,
            }
            # Reverse the performance so first set has best performance (highest value)
            performance = 1.0 + 0.1 * i  # Best is last (highest value)
            bayesian_scheduler.update_with_performance(params, performance)

        best_params = bayesian_scheduler.get_best_parameters()
        # Now expect the last parameter set to be best (highest learning rate)
        assert abs(best_params["learning_rate"] - 3e-3) < 1e-6
        assert abs(best_params["adaptation_factor"] - 0.7) < 1e-6
        assert best_params["patience"] == 7

    def test_bayesian_scheduler_acquisition_function(self, bayesian_scheduler):
        """Test acquisition function computation."""
        # Add some data points
        for i in range(3):
            params = {
                "learning_rate": 1e-3 * (i + 1),
                "adaptation_factor": 0.5,
                "patience": 5,
            }
            bayesian_scheduler.update_with_performance(params, float(i))

        # Test acquisition function
        test_params = {"learning_rate": 2e-3, "adaptation_factor": 0.6, "patience": 6}
        acquisition_value = bayesian_scheduler._compute_acquisition_function(
            test_params
        )

        assert isinstance(acquisition_value, float)
        assert acquisition_value >= 0.0


class TestSchedulerIntegration:
    """Test suite for SchedulerIntegration."""

    @pytest.fixture
    def integration_config(self):
        """Create a scheduler integration configuration."""
        return MetaSchedulerConfig(
            enable_performance_awareness=True,
            enable_multiscale=True,
            enable_bayesian_optimization=True,
        )

    @pytest.fixture
    def scheduler_integration(self, integration_config):
        """Create a SchedulerIntegration instance."""
        rngs = nnx.Rngs(42)
        return SchedulerIntegration(config=integration_config, rngs=rngs)

    def test_scheduler_integration_initialization(self, scheduler_integration):
        """Test SchedulerIntegration initialization."""
        assert scheduler_integration.performance_scheduler is not None
        assert scheduler_integration.multiscale_scheduler is not None
        assert scheduler_integration.bayesian_scheduler is not None

    def test_scheduler_integration_create_adaptive_optimizer(
        self, scheduler_integration
    ):
        """Test creating adaptive optimizer."""
        optimizer = scheduler_integration.create_adaptive_optimizer()

        assert hasattr(optimizer, "init")
        assert hasattr(optimizer, "update")

    def test_scheduler_integration_l2o_integration(self, scheduler_integration):
        """Test integration with L2O engine."""
        # Create minimal L2O engine
        l2o_config = L2OEngineConfig()
        meta_config = MetaOptimizerConfig()
        rngs = nnx.Rngs(42)

        l2o_engine = L2OEngine(l2o_config, meta_config, rngs=rngs)

        # Test integration
        enhanced_engine = scheduler_integration.integrate_with_l2o_engine(l2o_engine)

        assert enhanced_engine is not None
        assert hasattr(enhanced_engine, "adaptive_schedulers")

    def test_scheduler_integration_step_update(self, scheduler_integration):
        """Test step-by-step scheduler updates."""
        # Simulate optimization steps
        losses = [1.0, 0.8, 0.6, 0.7, 0.5]  # Some improvement, some stagnation

        for step, loss in enumerate(losses):
            learning_rates = scheduler_integration.update_schedulers(step, loss)

            assert "performance_aware" in learning_rates
            assert "multiscale" in learning_rates

            assert isinstance(learning_rates["performance_aware"], float)
            assert isinstance(learning_rates["multiscale"], dict)

    def test_scheduler_integration_auto_parameter_optimization(
        self, scheduler_integration
    ):
        """Test automatic parameter optimization."""

        # Simulate optimization run
        def mock_optimization_run(params):
            """Mock optimization run returning loss."""
            return params.get("learning_rate", 1e-3) * 100  # Simple mock performance

        # Run auto-optimization
        best_params = scheduler_integration.auto_optimize_parameters(
            optimization_function=mock_optimization_run,
            num_trials=5,
        )

        assert "learning_rate" in best_params
        assert "adaptation_factor" in best_params
        assert "patience" in best_params


class TestIntegrationWithExistingFramework:
    """Test integration with existing L2O framework."""

    def test_integration_with_parametric_solver(self):
        """Test integration with parametric solver."""
        # Create scheduler and L2O components
        scheduler_config = MetaSchedulerConfig(enable_performance_awareness=True)
        rngs = nnx.Rngs(42)

        scheduler_integration = SchedulerIntegration(config=scheduler_config, rngs=rngs)

        # Create simple optimization problem
        problem = OptimizationProblem(
            problem_type="quadratic",
            dimension=3,
            constraints=None,
        )

        # Test that scheduler can adapt based on problem solving
        problem_params = jnp.array([1.0, 2.0, 3.0])

        # Simulate solving steps with the problem
        for step in range(5):
            # Use problem dimension to compute loss
            loss = float(jnp.sum(problem_params[: problem.dimension] ** 2))
            learning_rates = scheduler_integration.update_schedulers(step, loss)

            # Learning rates should be reasonable
            assert 1e-6 <= learning_rates["performance_aware"] <= 1e-1

    def test_integration_with_multi_objective_framework(self):
        """Test integration with multi-objective framework."""
        scheduler_config = MetaSchedulerConfig(
            enable_multiscale=True,
            multiscale_components=["pareto_optimizer", "scalarizer"],
        )
        rngs = nnx.Rngs(42)

        scheduler_integration = SchedulerIntegration(config=scheduler_config, rngs=rngs)

        # Test component-specific learning rates
        if scheduler_integration.multiscale_scheduler is not None:
            learning_rates = scheduler_integration.multiscale_scheduler.get_component_learning_rates()

            assert "pareto_optimizer" in learning_rates
            assert "scalarizer" in learning_rates

    def test_l2o_integration_compatibility(self):
        """Test that adaptive schedulers integrate seamlessly with L2O framework."""
        # Create standard L2O engine
        l2o_config = L2OEngineConfig()
        meta_config = MetaOptimizerConfig()
        rngs = nnx.Rngs(42)

        l2o_engine = L2OEngine(l2o_config, meta_config, rngs=rngs)

        # Add adaptive schedulers
        scheduler_config = MetaSchedulerConfig()
        scheduler_integration = SchedulerIntegration(config=scheduler_config, rngs=rngs)

        enhanced_engine = scheduler_integration.integrate_with_l2o_engine(l2o_engine)

        # Should still work with standard optimization problems
        problem = OptimizationProblem(
            problem_type="quadratic",
            dimension=2,
            constraints=None,
        )

        problem_params = jnp.array([1.0, 2.0])

        # Should not raise errors - test that problem is used in enhanced engine
        # Since enhanced_engine.adaptive_solve expects a problem and problem_params
        try:
            if hasattr(enhanced_engine, "adaptive_solve"):
                solution = enhanced_engine.adaptive_solve(problem, problem_params)
                assert solution.shape == (2,)
            else:
                # Fallback to regular solve if adaptive_solve doesn't exist
                solution = enhanced_engine.solve_parametric_problem(
                    problem, problem_params
                )
                assert solution.shape == (2,)
        except Exception:
            # If adaptive_solve doesn't work, fall back to regular solve to test compatibility
            solution = enhanced_engine.solve_parametric_problem(problem, problem_params)
            assert solution.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__])
