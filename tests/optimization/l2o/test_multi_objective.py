"""Tests for Multi-Objective L2O Framework.

This module tests the multi-objective optimization capabilities including
Pareto frontier approximation, learned scalarization, and performance indicators.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.optimization.l2o.l2o_engine import L2OEngine, L2OEngineConfig
from opifex.optimization.l2o.multi_objective import (
    MultiObjectiveConfig,
    MultiObjectiveL2OEngine,
    ObjectiveScalarizer,
    ParetoFrontierOptimizer,
    PerformanceIndicators,
)
from opifex.optimization.meta_optimization import MetaOptimizerConfig


class TestMultiObjectiveConfig:
    """Test multi-objective configuration validation and initialization."""

    def test_multi_objective_config_initialization(self):
        """Test default multi-objective configuration initialization."""
        config = MultiObjectiveConfig()
        assert config.num_objectives == 2
        assert config.pareto_points_target == 100
        assert config.scalarization_strategy == "learned"
        assert config.diversity_pressure == 0.1
        assert config.convergence_tolerance == 1e-6
        assert config.max_pareto_iterations == 500
        assert config.hypervolume_reference_point is None
        assert config.adaptive_weights is True
        assert config.dominated_solution_filtering is True

    def test_multi_objective_config_custom_initialization(self):
        """Test custom multi-objective configuration."""
        config = MultiObjectiveConfig(
            num_objectives=3,
            pareto_points_target=50,
            scalarization_strategy="weighted_sum",
            diversity_pressure=0.2,
            convergence_tolerance=1e-5,
            max_pareto_iterations=200,
            hypervolume_reference_point=[1.0, 1.0, 1.0],
            adaptive_weights=False,
            dominated_solution_filtering=False,
        )
        assert config.num_objectives == 3
        assert config.pareto_points_target == 50
        assert config.scalarization_strategy == "weighted_sum"
        assert config.diversity_pressure == 0.2
        assert config.convergence_tolerance == 1e-5
        assert config.max_pareto_iterations == 200
        assert config.hypervolume_reference_point == [1.0, 1.0, 1.0]
        assert config.adaptive_weights is False
        assert config.dominated_solution_filtering is False

    def test_multi_objective_config_validation(self):
        """Test multi-objective configuration validation."""
        # Test invalid number of objectives
        with pytest.raises(ValueError, match="at least 2 objectives"):
            MultiObjectiveConfig(num_objectives=1)

        # Test invalid pareto points target
        with pytest.raises(ValueError, match="must be positive"):
            MultiObjectiveConfig(pareto_points_target=0)

        # Test invalid scalarization strategy
        with pytest.raises(ValueError, match="Invalid scalarization strategy"):
            MultiObjectiveConfig(scalarization_strategy="invalid")

        # Test invalid hypervolume reference point dimension
        with pytest.raises(ValueError, match="same dimension as objectives"):
            MultiObjectiveConfig(
                num_objectives=3, hypervolume_reference_point=[1.0, 1.0]
            )


class TestParetoFrontierOptimizer:
    """Test Pareto frontier optimization neural network."""

    def test_pareto_frontier_optimizer_initialization(self):
        """Test Pareto frontier optimizer initialization."""
        config = MultiObjectiveConfig(num_objectives=2)
        rngs = nnx.Rngs(42)

        optimizer = ParetoFrontierOptimizer(config, problem_dimension=5, rngs=rngs)

        assert optimizer.config == config
        assert optimizer.problem_dimension == 5
        assert optimizer.frontier_network is not None
        assert optimizer.reference_point.shape == (2,)
        assert jnp.allclose(optimizer.reference_point, jnp.ones(2) * 10.0)

    def test_pareto_frontier_optimizer_with_reference_point(self):
        """Test Pareto frontier optimizer with custom reference point."""
        config = MultiObjectiveConfig(
            num_objectives=3, hypervolume_reference_point=[2.0, 3.0, 1.0]
        )
        rngs = nnx.Rngs(42)

        optimizer = ParetoFrontierOptimizer(config, problem_dimension=4, rngs=rngs)

        expected_ref = jnp.array([2.0, 3.0, 1.0])
        assert jnp.allclose(optimizer.reference_point, expected_ref)

    def test_generate_pareto_solutions(self):
        """Test Pareto solution generation."""
        config = MultiObjectiveConfig(num_objectives=2, pareto_points_target=10)
        rngs = nnx.Rngs(42)

        optimizer = ParetoFrontierOptimizer(config, problem_dimension=3, rngs=rngs)

        # Define simple objective functions
        def obj1(x):
            return jnp.sum(x**2)

        def obj2(x):
            return jnp.sum((x - 1) ** 2)

        objective_functions = [obj1, obj2]

        solutions, objective_values = optimizer.generate_pareto_solutions(
            objective_functions
        )

        # Check output shapes
        assert solutions.shape[1] == 3  # problem dimension
        assert objective_values.shape[1] == 2  # num objectives
        assert len(solutions) <= config.pareto_points_target
        assert len(objective_values) == len(solutions)

    def test_uniform_preferences_generation(self):
        """Test uniform preference vector generation."""
        config = MultiObjectiveConfig(num_objectives=2, pareto_points_target=5)
        rngs = nnx.Rngs(42)

        optimizer = ParetoFrontierOptimizer(config, problem_dimension=2, rngs=rngs)
        preferences = optimizer._generate_uniform_preferences()

        assert preferences.shape == (5, 2)
        # Check that preferences sum to 1 (for 2 objectives)
        assert jnp.allclose(jnp.sum(preferences, axis=1), 1.0)

    def test_uniform_preferences_generation_multi_objective(self):
        """Test uniform preference generation for >2 objectives."""
        config = MultiObjectiveConfig(num_objectives=3, pareto_points_target=8)
        rngs = nnx.Rngs(42)

        optimizer = ParetoFrontierOptimizer(config, problem_dimension=2, rngs=rngs)
        preferences = optimizer._generate_uniform_preferences()

        assert preferences.shape == (8, 3)
        # Check that preferences sum to 1
        assert jnp.allclose(jnp.sum(preferences, axis=1), 1.0, atol=1e-6)

    def test_non_dominated_solution_identification(self):
        """Test identification of non-dominated solutions."""
        config = MultiObjectiveConfig(num_objectives=2)
        rngs = nnx.Rngs(42)

        optimizer = ParetoFrontierOptimizer(config, problem_dimension=2, rngs=rngs)

        # Create test objective values with known dominated/non-dominated solutions
        objective_values = jnp.array(
            [
                [1.0, 5.0],  # Non-dominated
                [2.0, 4.0],  # Non-dominated
                [3.0, 3.0],  # Non-dominated
                [4.0, 2.0],  # Non-dominated
                [5.0, 1.0],  # Non-dominated
                [2.5, 4.5],  # Dominated by (2.0, 4.0)
                [3.5, 3.5],  # Dominated by (3.0, 3.0)
            ]
        )

        non_dominated_mask = optimizer._identify_non_dominated_solutions(
            objective_values
        )

        # Check that dominated solutions are correctly identified
        expected_non_dominated = jnp.array([True, True, True, True, True, False, False])
        assert jnp.array_equal(non_dominated_mask, expected_non_dominated)

    def test_scalarization_methods(self):
        """Test different scalarization strategies."""
        config_weighted = MultiObjectiveConfig(scalarization_strategy="weighted_sum")
        config_chebyshev = MultiObjectiveConfig(scalarization_strategy="chebyshev")
        config_achievement = MultiObjectiveConfig(scalarization_strategy="achievement")

        rngs = nnx.Rngs(42)

        optimizer_weighted = ParetoFrontierOptimizer(config_weighted, 2, rngs=rngs)
        optimizer_chebyshev = ParetoFrontierOptimizer(config_chebyshev, 2, rngs=rngs)
        optimizer_achievement = ParetoFrontierOptimizer(
            config_achievement, 2, rngs=rngs
        )

        objectives = jnp.array([2.0, 3.0])
        preference = jnp.array([0.6, 0.4])

        # Test weighted sum
        weighted_result = optimizer_weighted._scalarize_objectives(
            objectives, preference
        )
        expected_weighted = 0.6 * 2.0 + 0.4 * 3.0
        assert jnp.isclose(weighted_result, expected_weighted)

        # Test Chebyshev
        chebyshev_result = optimizer_chebyshev._scalarize_objectives(
            objectives, preference
        )
        expected_chebyshev = jnp.max(preference * objectives)
        assert jnp.isclose(chebyshev_result, expected_chebyshev)

        # Test achievement
        achievement_result = optimizer_achievement._scalarize_objectives(
            objectives, preference
        )
        expected_achievement = jnp.max(preference * objectives) + 0.01 * jnp.sum(
            preference * objectives
        )
        assert jnp.isclose(achievement_result, expected_achievement)


class TestObjectiveScalarizer:
    """Test learned objective scalarization strategies."""

    def test_objective_scalarizer_initialization(self):
        """Test objective scalarizer initialization."""
        config = MultiObjectiveConfig(num_objectives=3)
        rngs = nnx.Rngs(42)

        scalarizer = ObjectiveScalarizer(config, problem_features_dim=10, rngs=rngs)

        assert scalarizer.config == config
        assert scalarizer.features_dim == 10
        assert scalarizer.weight_network is not None

    def test_learn_scalarization_weights(self):
        """Test learning of scalarization weights."""
        config = MultiObjectiveConfig(num_objectives=2, adaptive_weights=False)
        rngs = nnx.Rngs(42)

        scalarizer = ObjectiveScalarizer(config, problem_features_dim=5, rngs=rngs)

        problem_features = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        objective_values_history = jnp.array([[1.0, 2.0], [1.5, 1.8]])
        performance_feedback = jnp.array([0.8, 0.9])

        weights = scalarizer.learn_scalarization_weights(
            problem_features, objective_values_history, performance_feedback
        )

        assert weights.shape == (2,)
        assert jnp.isclose(jnp.sum(weights), 1.0)  # Should sum to 1 due to softmax
        assert jnp.all(weights >= 0)  # Should be non-negative

    def test_adaptive_weight_learning(self):
        """Test adaptive weight adjustment based on performance feedback."""
        config = MultiObjectiveConfig(num_objectives=2, adaptive_weights=True)
        rngs = nnx.Rngs(42)

        scalarizer = ObjectiveScalarizer(config, problem_features_dim=3, rngs=rngs)

        problem_features = jnp.array([1.0, 2.0, 3.0])
        objective_values_history = jnp.array([[1.0, 2.0]])
        performance_feedback = jnp.array(
            [0.5, 0.9]
        )  # First objective performing poorly

        weights = scalarizer.learn_scalarization_weights(
            problem_features, objective_values_history, performance_feedback
        )

        # Should still sum to 1 and be non-negative
        assert jnp.isclose(jnp.sum(weights), 1.0)
        assert jnp.all(weights >= 0)

    def test_scalarize_objectives_strategies(self):
        """Test different objective scalarization strategies."""
        config = MultiObjectiveConfig(num_objectives=2)
        rngs = nnx.Rngs(42)

        scalarizer = ObjectiveScalarizer(config, problem_features_dim=3, rngs=rngs)

        objectives = jnp.array([2.0, 3.0])
        weights = jnp.array([0.7, 0.3])

        # Test weighted sum
        weighted_result = scalarizer.scalarize_objectives(
            objectives, weights, strategy="weighted_sum"
        )
        expected_weighted = 0.7 * 2.0 + 0.3 * 3.0
        assert jnp.isclose(weighted_result, expected_weighted)

        # Test Chebyshev
        chebyshev_result = scalarizer.scalarize_objectives(
            objectives, weights, strategy="chebyshev"
        )
        expected_chebyshev = jnp.max(weights * objectives)
        assert jnp.isclose(chebyshev_result, expected_chebyshev)

        # Test achievement
        achievement_result = scalarizer.scalarize_objectives(
            objectives, weights, strategy="achievement"
        )
        expected_achievement = jnp.max(weights * objectives) + 0.01 * jnp.sum(
            weights * objectives
        )
        assert jnp.isclose(achievement_result, expected_achievement)


class TestPerformanceIndicators:
    """Test performance indicators for multi-objective optimization quality."""

    def test_hypervolume_2d(self):
        """Test hypervolume calculation for 2D Pareto front."""
        pareto_front = jnp.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
            ]
        )
        reference_point = jnp.array([5.0, 5.0])

        hypervolume = PerformanceIndicators.compute_hypervolume(
            pareto_front, reference_point
        )

        # Should be positive for a valid Pareto front
        assert hypervolume > 0
        assert jnp.isfinite(hypervolume)

    def test_hypervolume_higher_dimensions(self):
        """Test hypervolume calculation for higher dimensional Pareto front."""
        pareto_front = jnp.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 2.5],
                [1.5, 1.5, 2.0],
            ]
        )
        reference_point = jnp.array([3.0, 3.0, 4.0])

        hypervolume = PerformanceIndicators.compute_hypervolume(
            pareto_front, reference_point
        )

        assert hypervolume >= 0
        assert jnp.isfinite(hypervolume)

    def test_spread_indicator(self):
        """Test spread (diversity) indicator calculation."""
        # Test with well-distributed points
        pareto_front_good = jnp.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
            ]
        )

        spread_good = PerformanceIndicators.compute_spread_indicator(pareto_front_good)

        # Test with clustered points
        pareto_front_bad = jnp.array(
            [
                [1.0, 1.0],
                [1.1, 1.1],
                [1.0, 1.1],
                [1.1, 1.0],
            ]
        )

        spread_bad = PerformanceIndicators.compute_spread_indicator(pareto_front_bad)

        # Well-distributed points should have higher spread
        assert spread_good > spread_bad
        assert spread_good >= 0
        assert spread_bad >= 0

    def test_spread_indicator_edge_cases(self):
        """Test spread indicator edge cases."""
        # Single point
        single_point = jnp.array([[1.0, 2.0]])
        spread_single = PerformanceIndicators.compute_spread_indicator(single_point)
        assert spread_single == 0.0

        # Empty array
        empty_front = jnp.array([]).reshape(0, 2)
        spread_empty = PerformanceIndicators.compute_spread_indicator(empty_front)
        assert spread_empty == 0.0

    def test_convergence_indicator_without_true_front(self):
        """Test convergence indicator without true Pareto front."""
        pareto_front = jnp.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
            ]
        )

        convergence = PerformanceIndicators.compute_convergence_indicator(pareto_front)

        assert convergence >= 0
        assert jnp.isfinite(convergence)

    def test_convergence_indicator_with_true_front(self):
        """Test convergence indicator with known true Pareto front."""
        # Approximate front
        pareto_front = jnp.array(
            [
                [1.1, 3.9],
                [2.1, 2.9],
                [3.1, 1.9],
            ]
        )

        # True front
        true_front = jnp.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
            ]
        )

        convergence = PerformanceIndicators.compute_convergence_indicator(
            pareto_front, true_front
        )

        # Should be small positive value indicating good convergence
        assert convergence > 0
        assert convergence < 1.0  # Should be reasonably close
        assert jnp.isfinite(convergence)

    def test_convergence_indicator_edge_cases(self):
        """Test convergence indicator edge cases."""
        # Single point
        single_point = jnp.array([[1.0, 2.0]])
        convergence_single = PerformanceIndicators.compute_convergence_indicator(
            single_point
        )
        assert convergence_single == 0.0


class TestMultiObjectiveL2OEngine:
    """Test the complete multi-objective L2O engine."""

    def test_multi_objective_l2o_engine_initialization(self):
        """Test multi-objective L2O engine initialization."""
        mo_config = MultiObjectiveConfig(num_objectives=2)
        l2o_config = L2OEngineConfig()
        meta_config = MetaOptimizerConfig(meta_algorithm="l2o")

        # Create mock L2O engine
        rngs = nnx.Rngs(42)
        l2o_engine = L2OEngine(l2o_config, meta_config, rngs=rngs)

        mo_engine = MultiObjectiveL2OEngine(
            mo_config, l2o_engine, problem_dimension=3, rngs=rngs
        )

        assert mo_engine.config == mo_config
        assert mo_engine.l2o_engine == l2o_engine
        assert mo_engine.problem_dimension == 3
        assert mo_engine.pareto_optimizer is not None
        assert mo_engine.scalarizer is not None
        assert mo_engine.performance_indicators is not None

    def test_solve_multi_objective_problem(self):
        """Test solving a multi-objective optimization problem."""
        mo_config = MultiObjectiveConfig(
            num_objectives=2, pareto_points_target=5, max_pareto_iterations=10
        )
        l2o_config = L2OEngineConfig()
        meta_config = MetaOptimizerConfig(meta_algorithm="l2o")

        rngs = nnx.Rngs(42)
        l2o_engine = L2OEngine(l2o_config, meta_config, rngs=rngs)

        mo_engine = MultiObjectiveL2OEngine(
            mo_config, l2o_engine, problem_dimension=2, rngs=rngs
        )

        # Define simple test objectives
        def objective1(x):
            return jnp.sum(x**2)

        def objective2(x):
            return jnp.sum((x - 1) ** 2)

        objective_functions = [objective1, objective2]
        problem_features = jnp.array([1.0, 2.0, 3.0, 4.0])  # 2 * problem_dimension

        results = mo_engine.solve_multi_objective_problem(
            objective_functions, problem_features
        )

        # Check that all expected keys are present
        expected_keys = [
            "pareto_solutions",
            "pareto_objectives",
            "learned_weights",
            "hypervolume",
            "spread",
            "convergence",
            "num_pareto_points",
            "solve_time",
            "pareto_training_converged",
            "pareto_training_iterations",
        ]

        for key in expected_keys:
            assert key in results

        # Check output shapes and types
        assert results["pareto_solutions"].shape[1] == 2  # problem dimension
        assert results["pareto_objectives"].shape[1] == 2  # num objectives
        assert results["learned_weights"].shape == (2,)
        assert isinstance(results["hypervolume"], float)
        assert isinstance(results["spread"], float)
        assert isinstance(results["convergence"], float)
        assert isinstance(results["num_pareto_points"], int)
        assert isinstance(results["solve_time"], float)
        assert isinstance(results["pareto_training_converged"], bool)
        assert isinstance(results["pareto_training_iterations"], int)

    def test_solve_with_preference(self):
        """Test solving with user preference vector."""
        mo_config = MultiObjectiveConfig(num_objectives=2)
        l2o_config = L2OEngineConfig()
        meta_config = MetaOptimizerConfig(meta_algorithm="l2o")

        rngs = nnx.Rngs(42)
        l2o_engine = L2OEngine(l2o_config, meta_config, rngs=rngs)

        mo_engine = MultiObjectiveL2OEngine(
            mo_config, l2o_engine, problem_dimension=2, rngs=rngs
        )

        # Define test objectives
        def objective1(x):
            return jnp.sum(x**2)

        def objective2(x):
            return jnp.sum((x - 1) ** 2)

        objective_functions = [objective1, objective2]
        preference_vector = jnp.array([0.7, 0.3])
        problem_features = jnp.array([1.0, 2.0, 3.0, 4.0])

        solution, metrics = mo_engine.solve_with_preference(
            objective_functions, preference_vector, problem_features
        )

        # Check output shapes and types
        assert solution.shape == (2,)  # problem dimension
        assert "objectives" in metrics
        assert "scalarized_value" in metrics
        assert "preference_vector" in metrics

        assert metrics["objectives"].shape == (2,)  # num objectives
        assert isinstance(metrics["scalarized_value"], float)
        assert jnp.array_equal(metrics["preference_vector"], preference_vector)


class TestIntegrationWithL2OFramework:
    """Test integration with existing L2O framework."""

    def test_multi_objective_with_constraints(self):
        """Test multi-objective optimization with constraints."""
        mo_config = MultiObjectiveConfig(
            num_objectives=2, pareto_points_target=5, max_pareto_iterations=5
        )
        l2o_config = L2OEngineConfig()
        meta_config = MetaOptimizerConfig(meta_algorithm="l2o")

        rngs = nnx.Rngs(42)
        l2o_engine = L2OEngine(l2o_config, meta_config, rngs=rngs)

        mo_engine = MultiObjectiveL2OEngine(
            mo_config, l2o_engine, problem_dimension=2, rngs=rngs
        )

        # Define objectives and constraints
        def objective1(x):
            return jnp.sum(x**2)

        def objective2(x):
            return jnp.sum((x - 1) ** 2)

        def constraint(x):
            # Simple box constraint: -1 <= x <= 1
            return jnp.array([jnp.max(jnp.abs(x)) - 1])

        objective_functions = [objective1, objective2]
        problem_features = jnp.array([1.0, 2.0, 3.0, 4.0])

        results = mo_engine.solve_multi_objective_problem(
            objective_functions, problem_features, constraint_function=constraint
        )

        # Should complete without errors
        assert "pareto_solutions" in results
        assert "pareto_objectives" in results

    def test_performance_comparison_metrics(self):
        """Test performance comparison between different strategies."""
        mo_config = MultiObjectiveConfig(
            num_objectives=2, pareto_points_target=3, max_pareto_iterations=5
        )
        l2o_config = L2OEngineConfig()
        meta_config = MetaOptimizerConfig(meta_algorithm="l2o")

        rngs = nnx.Rngs(42)
        l2o_engine = L2OEngine(l2o_config, meta_config, rngs=rngs)

        mo_engine = MultiObjectiveL2OEngine(
            mo_config, l2o_engine, problem_dimension=2, rngs=rngs
        )

        # Test with different scalarization strategies
        strategies = ["weighted_sum", "chebyshev", "achievement"]
        results_by_strategy = {}

        def objective1(x):
            return jnp.sum(x**2)

        def objective2(x):
            return jnp.sum((x - 1) ** 2)

        objective_functions = [objective1, objective2]
        problem_features = jnp.array([1.0, 2.0, 3.0, 4.0])

        for strategy in strategies:
            # Update scalarization strategy
            mo_engine.config.scalarization_strategy = strategy

            results = mo_engine.solve_multi_objective_problem(
                objective_functions, problem_features
            )

            results_by_strategy[strategy] = results

        # All strategies should produce valid results
        for strategy in strategies:
            assert "hypervolume" in results_by_strategy[strategy]
            assert "spread" in results_by_strategy[strategy]
            assert "convergence" in results_by_strategy[strategy]
            assert results_by_strategy[strategy]["hypervolume"] >= 0
            assert results_by_strategy[strategy]["spread"] >= 0
            assert results_by_strategy[strategy]["convergence"] >= 0


class TestJITCompatibility:
    """Test JIT compilation compatibility for multi-objective optimization components."""

    def test_pareto_frontier_optimizer_jit_compilation(self):
        """Test that ParetoFrontierOptimizer components can be JIT compiled."""
        import time

        config = MultiObjectiveConfig(num_objectives=2, pareto_points_target=10)
        optimizer = ParetoFrontierOptimizer(config, 4, rngs=nnx.Rngs(42))

        # Test JIT compilation of the frontier network directly
        jitted_frontier_network = nnx.jit(optimizer.frontier_network)

        # Test data
        preference_vector = jnp.array([0.6, 0.4])

        # Test that JIT compilation works
        solution = jitted_frontier_network(preference_vector)

        assert solution.shape == (4,)
        assert jnp.isfinite(solution).all()

        # Test performance improvement with JIT
        # Warmup
        _ = jitted_frontier_network(preference_vector)

        # Time JIT version
        start_time = time.time()
        for _ in range(100):
            _ = jitted_frontier_network(preference_vector)
        jit_time = time.time() - start_time

        # Time non-JIT version
        start_time = time.time()
        for _ in range(100):
            _ = optimizer.frontier_network(preference_vector)
        non_jit_time = time.time() - start_time

        # JIT should be faster or at least not significantly slower
        assert jit_time <= non_jit_time * 2.0  # Allow some overhead for small problems

        # Test batch processing with JIT
        batch_preferences = jax.random.uniform(jax.random.PRNGKey(0), (20, 2))
        # Normalize preferences to sum to 1
        batch_preferences = batch_preferences / jnp.sum(
            batch_preferences, axis=-1, keepdims=True
        )

        # JIT compile batch processing
        jitted_batch_frontier = nnx.jit(jax.vmap(optimizer.frontier_network))
        batch_solutions = jitted_batch_frontier(batch_preferences)

        assert batch_solutions.shape == (20, 4)
        assert jnp.isfinite(batch_solutions).all()

    def test_performance_indicators_jit_compilation(self):
        """Test that PerformanceIndicators methods can be JIT compiled."""
        # Create test Pareto front
        pareto_front = jnp.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        reference_point = jnp.array([5.0, 5.0])

        # JIT compile performance indicator methods
        jitted_hypervolume = nnx.jit(PerformanceIndicators.compute_hypervolume)
        jitted_spread = nnx.jit(PerformanceIndicators.compute_spread_indicator)
        jitted_convergence = nnx.jit(
            PerformanceIndicators.compute_convergence_indicator
        )

        # Test JIT compilation works
        hypervolume = jitted_hypervolume(pareto_front, reference_point)
        spread = jitted_spread(pareto_front)
        convergence = jitted_convergence(pareto_front)

        assert jnp.isfinite(hypervolume)
        assert jnp.isfinite(spread)
        assert jnp.isfinite(convergence)
        assert hypervolume >= 0
        assert spread >= 0
        assert convergence >= 0

    def test_objective_scalarizer_jit_compilation(self):
        """Test that ObjectiveScalarizer can be JIT compiled."""
        config = MultiObjectiveConfig(num_objectives=3)
        scalarizer = ObjectiveScalarizer(config, 6, rngs=nnx.Rngs(42))

        # Test data
        problem_features = jax.random.normal(jax.random.PRNGKey(0), (6,))
        objective_values = jax.random.uniform(jax.random.PRNGKey(1), (10, 3))
        performance_feedback = jax.random.uniform(jax.random.PRNGKey(2), (3,))
        objectives = jax.random.uniform(jax.random.PRNGKey(3), (3,))
        weights = jnp.array([0.4, 0.3, 0.3])

        # JIT compile methods
        jitted_learn_weights = nnx.jit(scalarizer.learn_scalarization_weights)
        jitted_scalarize = nnx.jit(scalarizer.scalarize_objectives)

        # Test JIT compilation works
        learned_weights = jitted_learn_weights(
            problem_features, objective_values, performance_feedback
        )
        scalarized_value = jitted_scalarize(objectives, weights)

        assert learned_weights.shape == (3,)
        assert jnp.isfinite(learned_weights).all()
        assert jnp.abs(jnp.sum(learned_weights) - 1.0) < 1e-6  # Weights should sum to 1
        assert jnp.isfinite(scalarized_value)

    def test_multi_objective_engine_jit_compilation(self):
        """Test that MultiObjectiveL2OEngine components can be JIT compiled."""

        # Create mock L2O engine (simplified)
        from typing import cast

        class MockL2OEngine:
            def __init__(self):
                pass

            def solve(self, *args, **kwargs):
                # Return a simple mock solution
                return jnp.array([1.0, 2.0, 3.0])

            def batch_solve(self, *args, **kwargs):
                # Return batch mock solutions
                return jnp.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]])

            def solve_parametric_problem(self, *args, **kwargs):
                return jnp.array([1.0, 2.0, 3.0])

            def solve_gradient_problem(self, *args, **kwargs):
                return jnp.array([1.0, 2.0, 3.0])

            def solve_automatically(self, *args, **kwargs):
                return jnp.array([1.0, 2.0, 3.0])

        mock_engine = cast("L2OEngine", MockL2OEngine())

        config = MultiObjectiveConfig(num_objectives=2, pareto_points_target=5)
        engine = MultiObjectiveL2OEngine(config, mock_engine, 3, rngs=nnx.Rngs(42))

        # Test JIT compilation of individual components
        preference_vector = jnp.array([0.6, 0.4])

        # JIT compile the pareto optimizer's frontier network
        jitted_frontier = nnx.jit(engine.pareto_optimizer.frontier_network)
        solution = jitted_frontier(preference_vector)

        assert solution.shape == (3,)
        assert jnp.isfinite(solution).all()

        # JIT compile the scalarizer components
        objectives = jax.random.uniform(jax.random.PRNGKey(3), (2,))
        weights = jnp.array([0.6, 0.4])

        jitted_scalarize = nnx.jit(engine.scalarizer.scalarize_objectives)
        scalarized_value = jitted_scalarize(objectives, weights)

        assert jnp.isfinite(scalarized_value)

    def test_batch_jit_compatibility(self):
        """Test JIT compilation with batch processing."""
        config = MultiObjectiveConfig(num_objectives=2, pareto_points_target=8)
        optimizer = ParetoFrontierOptimizer(config, 3, rngs=nnx.Rngs(42))

        # Create batch processing function
        def batch_process_solutions(preference_vectors):
            def single_solution(pref):
                return optimizer.frontier_network(pref)

            return jax.vmap(single_solution)(preference_vectors)

        # JIT compile batch processing
        jitted_batch_process = nnx.jit(batch_process_solutions)

        # Test batch processing
        batch_preferences = jax.random.uniform(jax.random.PRNGKey(0), (5, 2))
        # Normalize preferences to sum to 1
        batch_preferences = batch_preferences / jnp.sum(
            batch_preferences, axis=-1, keepdims=True
        )

        batch_solutions = jitted_batch_process(batch_preferences)

        assert batch_solutions.shape == (5, 3)
        assert jnp.isfinite(batch_solutions).all()
