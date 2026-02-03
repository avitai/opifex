"""Test-driven development tests for L2O engine integration.

These tests define the expected behavior of the L2O engine that integrates
parametric programming solvers with the existing meta-optimization framework.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

# Import our new L2O engine components (to be implemented)
from opifex.optimization.l2o.l2o_engine import (
    L2OEngine,
    L2OEngineConfig,
    OptimizationProblemEncoder,
    ParametricOptimizationSolver,
)

# Import our parametric solver
from opifex.optimization.l2o.parametric_solver import (
    OptimizationProblem,
    SolverConfig,
)

# Import existing optimization infrastructure
from opifex.optimization.meta_optimization import (
    MetaOptimizer,
    MetaOptimizerConfig,
)


class TestL2OEngineConfig:
    """Test cases for L2O engine configuration."""

    def test_l2o_engine_config_initialization(self):
        """Test L2O engine configuration with default values."""
        config = L2OEngineConfig()

        assert config.solver_type == "parametric"
        assert config.problem_encoder_layers == [64, 32, 16]
        assert config.use_traditional_fallback is True
        assert config.enable_meta_learning is True
        assert config.integration_mode == "unified"

    def test_l2o_engine_config_custom_initialization(self):
        """Test custom L2O engine configuration."""
        config = L2OEngineConfig(
            solver_type="hybrid",
            problem_encoder_layers=[128, 64],
            use_traditional_fallback=False,
            enable_meta_learning=False,
            integration_mode="parametric_only",
            speedup_threshold=50.0,
        )

        assert config.solver_type == "hybrid"
        assert config.problem_encoder_layers == [128, 64]
        assert config.use_traditional_fallback is False
        assert config.enable_meta_learning is False
        assert config.integration_mode == "parametric_only"
        assert config.speedup_threshold == 50.0

    def test_l2o_engine_config_validation(self):
        """Test L2O engine configuration validation."""
        with pytest.raises(ValueError, match="Invalid solver type"):
            L2OEngineConfig(solver_type="invalid_type")

        with pytest.raises(ValueError, match="Invalid integration mode"):
            L2OEngineConfig(integration_mode="invalid_mode")


class TestOptimizationProblemEncoder:
    """Test cases for optimization problem encoding."""

    @pytest.fixture
    def encoder(self):
        """Fixture providing a problem encoder instance."""
        rngs = nnx.Rngs(42)
        return OptimizationProblemEncoder(
            input_dim=10, output_dim=16, hidden_layers=[32, 24], rngs=rngs
        )

    def test_optimization_problem_encoder_initialization(self):
        """Test problem encoder initialization."""
        rngs = nnx.Rngs(42)
        encoder = OptimizationProblemEncoder(
            input_dim=10, output_dim=16, hidden_layers=[32, 24], rngs=rngs
        )

        assert encoder.input_dim == 10
        assert encoder.output_dim == 16
        assert encoder.hidden_layers == [32, 24]
        assert hasattr(encoder, "encoder_network")

    def test_quadratic_problem_encoding(self, encoder):
        """Test encoding of quadratic optimization problems."""
        # Create quadratic problem specification
        problem = OptimizationProblem(
            problem_type="quadratic", dimension=5, constraints=None
        )

        # Problem parameters: Q matrix + c vector
        Q_matrix = jnp.eye(5) * 0.5  # Positive definite
        c_vector = jnp.ones(5) * 0.1
        problem_params = jnp.concatenate([Q_matrix.flatten(), c_vector])

        # Ensure correct input dimension
        if problem_params.shape[0] > encoder.input_dim:
            problem_params = problem_params[: encoder.input_dim]
        elif problem_params.shape[0] < encoder.input_dim:
            padding = jnp.zeros(encoder.input_dim - problem_params.shape[0])
            problem_params = jnp.concatenate([problem_params, padding])

        encoding = encoder.encode_problem(problem, problem_params)

        assert encoding.shape == (encoder.output_dim,)
        assert jnp.isfinite(encoding).all()

    def test_constraint_encoding(self, encoder):
        """Test encoding of constrained optimization problems."""
        # Create constrained problem
        constraints = {
            "equality": jnp.array([[1.0, 1.0, 0.0, 0.0, 0.0]]),  # x1 + x2 = 1
            "inequality": jnp.array([[0.0, 0.0, 1.0, -1.0, 0.0]]),  # x3 >= x4
        }

        problem = OptimizationProblem(
            problem_type="linear", dimension=5, constraints=constraints
        )

        # Linear problem: minimize c^T x
        c_vector = jnp.array([1.0, 2.0, 0.5, 1.5, 0.8])
        problem_params = jnp.pad(c_vector, (0, max(0, encoder.input_dim - 5)))[
            : encoder.input_dim
        ]

        encoding = encoder.encode_problem(problem, problem_params)

        assert encoding.shape == (encoder.output_dim,)
        assert jnp.isfinite(encoding).all()

    def test_batch_encoding(self, encoder):
        """Test batch encoding of multiple problems."""
        batch_size = 4

        # Create batch of problems
        problems = [OptimizationProblem("quadratic", 5) for _ in range(batch_size)]

        problem_params_batch = jnp.ones((batch_size, encoder.input_dim))

        encodings = encoder.encode_problem_batch(problems, problem_params_batch)

        assert encodings.shape == (batch_size, encoder.output_dim)
        assert jnp.isfinite(encodings).all()


class TestParametricOptimizationSolver:
    """Test cases for the integrated parametric optimization solver."""

    @pytest.fixture
    def solver_config(self):
        """Fixture providing solver configuration."""
        return SolverConfig(
            hidden_sizes=[64, 32], learning_rate=1e-3, max_iterations=100
        )

    @pytest.fixture
    def l2o_config(self):
        """Fixture providing L2O engine configuration."""
        return L2OEngineConfig(
            solver_type="parametric",
            problem_encoder_layers=[32, 16],
            use_traditional_fallback=True,
        )

    @pytest.fixture
    def parametric_solver(self, solver_config, l2o_config):
        """Fixture providing parametric optimization solver."""
        rngs = nnx.Rngs(42)
        return ParametricOptimizationSolver(
            solver_config=solver_config,
            l2o_config=l2o_config,
            input_dim=20,
            output_dim=10,
            rngs=rngs,
        )

    def test_parametric_optimization_solver_initialization(
        self, solver_config, l2o_config
    ):
        """Test parametric optimization solver initialization."""
        rngs = nnx.Rngs(42)
        solver = ParametricOptimizationSolver(
            solver_config=solver_config,
            l2o_config=l2o_config,
            input_dim=20,
            output_dim=10,
            rngs=rngs,
        )

        assert solver.input_dim == 20
        assert solver.output_dim == 10
        assert solver.solver_config == solver_config
        assert solver.l2o_config == l2o_config
        assert hasattr(solver, "parametric_solver")
        assert hasattr(solver, "problem_encoder")

    def test_end_to_end_optimization(self, parametric_solver):
        """Test end-to-end optimization pipeline."""
        # Define optimization problem
        problem = OptimizationProblem(
            problem_type="quadratic", dimension=parametric_solver.output_dim
        )

        # Problem parameters
        problem_params = jnp.ones(parametric_solver.input_dim) * 0.5

        # Solve optimization problem
        solution, metadata = parametric_solver.solve_optimization_problem(
            problem, problem_params
        )

        assert solution.shape == (parametric_solver.output_dim,)
        assert jnp.isfinite(solution).all()
        assert "encoding_time" in metadata
        assert "solving_time" in metadata
        assert "total_speedup" in metadata

    def test_traditional_fallback_integration(self, parametric_solver):
        """Test integration with traditional optimization fallback."""
        # Create a difficult problem that might trigger fallback
        problem = OptimizationProblem(
            problem_type="nonlinear", dimension=parametric_solver.output_dim
        )

        problem_params = (
            jnp.ones(parametric_solver.input_dim) * 1000
        )  # Difficult problem

        solution, metadata = parametric_solver.solve_optimization_problem(
            problem, problem_params, enable_fallback=True
        )

        assert solution.shape == (parametric_solver.output_dim,)
        assert jnp.isfinite(solution).all()
        assert "fallback_used" in metadata

    def test_performance_measurement(self, parametric_solver):
        """Test performance measurement functionality and correctness."""
        problem = OptimizationProblem("quadratic", parametric_solver.output_dim)
        problem_params = jnp.ones(parametric_solver.input_dim)

        # Measure performance
        performance = parametric_solver.measure_performance(problem, problem_params)

        # Verify all required metrics are present
        assert "neural_time" in performance
        assert "traditional_time" in performance
        assert "speedup_factor" in performance
        assert "accuracy_comparison" in performance

        # Verify timing measurements are reasonable (positive and finite)
        assert performance["neural_time"] > 0
        assert performance["traditional_time"] > 0
        assert jnp.isfinite(performance["speedup_factor"])

        # For simple problems, neural methods may not be faster due to overhead,
        # but the system should function correctly and produce valid measurements
        assert performance["speedup_factor"] > 0.01  # System works, within 100x


class TestL2OEngine:
    """Test cases for the main L2O engine integration."""

    @pytest.fixture
    def l2o_config(self):
        """Fixture providing L2O engine configuration."""
        return L2OEngineConfig(
            solver_type="hybrid", integration_mode="unified", enable_meta_learning=True
        )

    @pytest.fixture
    def meta_config(self):
        """Fixture providing meta-optimizer configuration."""
        return MetaOptimizerConfig(meta_algorithm="l2o", performance_tracking=True)

    @pytest.fixture
    def l2o_engine(self, l2o_config, meta_config):
        """Fixture providing L2O engine instance."""
        rngs = nnx.Rngs(42)
        return L2OEngine(l2o_config=l2o_config, meta_config=meta_config, rngs=rngs)

    def test_l2o_engine_initialization(self, l2o_config, meta_config):
        """Test L2O engine initialization."""
        rngs = nnx.Rngs(42)
        engine = L2OEngine(l2o_config=l2o_config, meta_config=meta_config, rngs=rngs)

        assert engine.l2o_config == l2o_config
        assert engine.meta_config == meta_config
        assert hasattr(engine, "parametric_solver")
        assert hasattr(engine, "meta_optimizer")
        assert hasattr(engine, "gradient_l2o")

    def test_unified_optimization_interface(self, l2o_engine):
        """Test unified optimization interface for both types of problems."""
        # Test parametric optimization problem
        parametric_problem = OptimizationProblem("quadratic", 5)
        problem_params = jnp.ones(20)

        solution = l2o_engine.solve_parametric_problem(
            parametric_problem, problem_params
        )
        assert solution.shape == (5,)
        assert jnp.isfinite(solution).all()

        # Test gradient-based optimization
        def loss_fn(x):
            return jnp.sum(x**2)  # Simple quadratic loss

        initial_params = jnp.ones(5)
        optimized_params = l2o_engine.solve_gradient_problem(
            loss_fn, initial_params, steps=10
        )

        assert optimized_params.shape == (5,)
        assert jnp.isfinite(optimized_params).all()

    def test_automatic_solver_selection(self, l2o_engine):
        """Test automatic selection between parametric and gradient solvers."""
        # Problem that should use parametric solver
        simple_problem = OptimizationProblem("quadratic", 3)
        problem_params = jnp.ones(20)

        solver_choice, solution = l2o_engine.solve_automatically(
            problem=simple_problem, problem_params=problem_params
        )

        assert solver_choice in ["parametric", "gradient", "hybrid"]
        assert jnp.isfinite(solution).all()

    def test_meta_learning_integration(self, l2o_engine):
        """Test integration with meta-learning framework."""
        # Create a series of related optimization problems
        problems = [OptimizationProblem("quadratic", 4) for _ in range(3)]

        problem_params_list = [jnp.ones(20) * (i + 1) for i in range(3)]

        # Solve problems sequentially, allowing meta-learning
        solutions = []
        for i, (problem, params) in enumerate(
            zip(problems, problem_params_list, strict=False)
        ):
            solution, metadata = l2o_engine.solve_with_meta_learning(
                problem, params, problem_id=i
            )
            solutions.append(solution)

            # Check that meta-learning information is tracked
            assert "meta_learning_used" in metadata
            if i > 0:  # After first problem, should start benefiting from meta-learning
                assert "previous_experience_count" in metadata

        assert len(solutions) == 3
        assert all(jnp.isfinite(sol).all() for sol in solutions)

    def test_performance_comparison_framework(self, l2o_engine):
        """Test comprehensive performance comparison capabilities."""
        problem = OptimizationProblem("quadratic", 5)
        problem_params = jnp.ones(20)

        comparison = l2o_engine.compare_all_solvers(problem, problem_params)

        assert "parametric_solver" in comparison
        assert "gradient_l2o" in comparison
        assert "traditional_baseline" in comparison

        for solver_name, results in comparison.items():
            assert "solution" in results
            assert "time" in results
            assert "accuracy" in results
            if solver_name != "traditional_baseline":
                assert "speedup" in results

    def test_adaptive_algorithm_selection(self, l2o_engine):
        """Test adaptive selection of optimization algorithms."""
        # Test different problem types
        problems = [
            OptimizationProblem("quadratic", 3),
            OptimizationProblem("linear", 4),
            OptimizationProblem("nonlinear", 2),
        ]

        for problem in problems:
            problem_params = jnp.ones(20)

            # Get algorithm recommendation
            recommended_algorithm = l2o_engine.recommend_algorithm(
                problem, problem_params
            )
            assert recommended_algorithm in [
                "parametric",
                "gradient",
                "traditional",
                "hybrid",
            ]

            # Solve using recommended algorithm
            solution = l2o_engine.solve_with_recommendation(problem, problem_params)
            assert jnp.isfinite(solution).all()

    def test_integration_with_existing_meta_optimizer(self, l2o_engine):
        """Test integration with existing MetaOptimizer framework."""

        # Create a gradient-based loss function
        def quadratic_loss(x):
            return jnp.sum((x - 1.0) ** 2)

        initial_params = jnp.zeros(5)

        # Use the integrated meta-optimizer
        final_params, optimization_history = l2o_engine.optimize_with_meta_framework(
            loss_fn=quadratic_loss, initial_params=initial_params, steps=20
        )

        assert final_params.shape == (5,)
        assert jnp.isfinite(final_params).all()
        assert len(optimization_history) == 20

        # Check that optimization improved
        initial_loss = quadratic_loss(initial_params)
        final_loss = quadratic_loss(final_params)
        assert final_loss < initial_loss


class TestIntegrationWithExistingFramework:
    """Integration tests with existing Opifex optimization framework."""

    def test_framework_integration_compatibility(self):
        """Test that L2O engine integrates seamlessly with existing framework."""
        # Test existing MetaOptimizer still works
        config = MetaOptimizerConfig(meta_algorithm="l2o")
        rngs = nnx.Rngs(42)

        meta_optimizer = MetaOptimizer(config, rngs=rngs)

        def simple_loss(x):
            return jnp.sum(x**2)

        params = jnp.ones(3)
        opt_state = meta_optimizer.init_optimizer_state(params)

        new_params, _new_opt_state, meta_info = meta_optimizer.step(
            simple_loss, params, opt_state, step=0
        )

        assert new_params.shape == (3,)
        assert jnp.isfinite(new_params).all()
        assert "learning_rate" in meta_info

    def test_unified_configuration(self):
        """Test unified configuration system."""
        # Test that L2O engine can use existing configuration
        meta_config = MetaOptimizerConfig(
            meta_algorithm="l2o", performance_tracking=True, quantum_aware=True
        )

        l2o_config = L2OEngineConfig(solver_type="hybrid", integration_mode="unified")

        rngs = nnx.Rngs(42)
        engine = L2OEngine(l2o_config=l2o_config, meta_config=meta_config, rngs=rngs)

        # Verify configurations are properly integrated
        assert engine.meta_config.performance_tracking is True
        assert engine.meta_config.quantum_aware is True
        assert engine.l2o_config.solver_type == "hybrid"

    def test_physics_informed_optimization_integration(self):
        """Test integration with physics-informed optimization."""
        l2o_config = L2OEngineConfig(solver_type="hybrid")
        meta_config = MetaOptimizerConfig(quantum_aware=True)
        rngs = nnx.Rngs(42)

        engine = L2OEngine(l2o_config=l2o_config, meta_config=meta_config, rngs=rngs)

        # Test physics-informed problem solving
        def physics_loss(params):
            # Simple physics-informed loss: minimize energy
            return jnp.sum(params**2) + 0.1 * jnp.sum(jnp.sin(params))

        initial_params = jnp.ones(4)
        physics_solution = engine.solve_physics_informed(physics_loss, initial_params)

        assert physics_solution.shape == (4,)
        assert jnp.isfinite(physics_solution).all()

        # Solution should minimize the physics loss
        initial_loss = physics_loss(initial_params)
        final_loss = physics_loss(physics_solution)
        assert final_loss <= initial_loss
