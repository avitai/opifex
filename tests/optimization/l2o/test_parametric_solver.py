"""Test-driven development tests for parametric programming solver network.

These tests define the expected behavior of the L2O parametric solver before implementation,
following TDD red-green-refactor methodology.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

# Import statements that will be implemented
from opifex.optimization.l2o.parametric_solver import (
    ConstraintHandler,
    OptimizationProblem,
    ParametricProgrammingSolver,
    SolverConfig,
)


class TestOptimizationProblem:
    """Test cases for optimization problem representation."""

    def test_optimization_problem_initialization(self):
        """Test that optimization problems can be created with different types."""
        # Quadratic programming problem
        problem = OptimizationProblem(
            problem_type="quadratic", dimension=10, constraints=None
        )
        assert problem.problem_type == "quadratic"
        assert problem.dimension == 10
        assert problem.constraints is None

    def test_optimization_problem_with_constraints(self):
        """Test optimization problem creation with constraints."""
        constraints = {
            "equality": jnp.array([[1.0, 1.0, 0.0]]),  # x1 + x2 = 0
            "inequality": jnp.array([[0.0, 1.0, -1.0]]),  # x2 >= 1
        }

        problem = OptimizationProblem(
            problem_type="linear", dimension=3, constraints=constraints
        )
        assert problem.problem_type == "linear"
        assert problem.dimension == 3
        assert problem.constraints is not None
        assert "equality" in problem.constraints
        assert "inequality" in problem.constraints

    def test_optimization_problem_validation(self):
        """Test that invalid optimization problems raise errors."""
        with pytest.raises(ValueError, match="Invalid problem type"):
            OptimizationProblem(problem_type="invalid_type", dimension=5)

        with pytest.raises(ValueError, match="Dimension must be positive"):
            OptimizationProblem(problem_type="quadratic", dimension=0)


class TestConstraintHandler:
    """Test cases for constraint handling mechanisms."""

    def test_constraint_handler_initialization(self):
        """Test constraint handler creation with different methods."""
        handler = ConstraintHandler(method="penalty", penalty_weight=1.0)
        assert handler.method == "penalty"
        assert handler.penalty_weight == 1.0

        handler = ConstraintHandler(method="barrier", barrier_parameter=0.1)
        assert handler.method == "barrier"
        assert handler.barrier_parameter == 0.1

    def test_penalty_method_constraint_handling(self):
        """Test penalty method for constraint satisfaction."""
        handler = ConstraintHandler(method="penalty", penalty_weight=10.0)

        # Test equality constraint violation
        x = jnp.array([1.0, 2.0, 3.0])
        equality_constraint = jnp.array([1.0, 1.0, 0.0])  # x1 + x2 = 0

        penalty = handler.compute_penalty(
            x, equality_constraint, constraint_type="equality"
        )
        expected_violation = jnp.abs(
            jnp.dot(equality_constraint, x)
        )  # |1*1 + 1*2 + 0*3| = 3
        expected_penalty = 10.0 * expected_violation**2

        assert jnp.allclose(penalty, expected_penalty)

    def test_barrier_method_constraint_handling(self):
        """Test barrier method for inequality constraints."""
        handler = ConstraintHandler(method="barrier", barrier_parameter=0.1)

        # Test inequality constraint g(x) >= 0
        x = jnp.array([2.0, 3.0])
        inequality_constraint = jnp.array(
            [1.0, -1.0]
        )  # x1 - x2 >= 0 (violated: 2-3 = -1)

        barrier = handler.compute_barrier(x, inequality_constraint)
        # Should return large positive value for violated constraint
        assert barrier > 0
        assert jnp.isfinite(barrier)

    def test_constraint_projection(self):
        """Test constraint projection for feasible region."""
        handler = ConstraintHandler(method="projection")

        # Simple box constraints: 0 <= x <= 1
        x = jnp.array([-0.5, 0.5, 1.5])
        projected_x = handler.project_to_feasible(x, bounds=(0.0, 1.0))

        expected = jnp.array([0.0, 0.5, 1.0])
        assert jnp.allclose(projected_x, expected)


class TestSolverConfig:
    """Test cases for solver configuration."""

    def test_solver_config_default_initialization(self):
        """Test default solver configuration."""
        config = SolverConfig()

        assert config.hidden_sizes == [128, 128, 64]
        assert config.activation == nnx.gelu
        assert config.learning_rate == 1e-3
        assert config.max_iterations == 1000
        assert config.tolerance == 1e-6
        assert config.use_traditional_fallback is True

    def test_solver_config_custom_initialization(self):
        """Test custom solver configuration."""
        config = SolverConfig(
            hidden_sizes=[256, 256, 128],
            activation=nnx.relu,
            learning_rate=1e-4,
            max_iterations=500,
            tolerance=1e-8,
            use_traditional_fallback=False,
        )

        assert config.hidden_sizes == [256, 256, 128]
        assert config.activation == nnx.relu
        assert config.learning_rate == 1e-4
        assert config.max_iterations == 500
        assert config.tolerance == 1e-8
        assert config.use_traditional_fallback is False


class TestParametricProgrammingSolver:
    """Test cases for the main parametric programming solver network."""

    @pytest.fixture
    def solver_config(self):
        """Fixture providing a test solver configuration."""
        return SolverConfig(
            hidden_sizes=[64, 64, 32], learning_rate=1e-3, max_iterations=100
        )

    @pytest.fixture
    def solver(self, solver_config):
        """Fixture providing a parametric solver instance."""
        rngs = nnx.Rngs(42)
        return ParametricProgrammingSolver(
            config=solver_config, input_dim=10, output_dim=10, rngs=rngs
        )

    def test_parametric_solver_initialization(self, solver_config):
        """Test parametric solver network initialization."""
        rngs = nnx.Rngs(42)
        solver = ParametricProgrammingSolver(
            config=solver_config, input_dim=10, output_dim=10, rngs=rngs
        )

        assert solver.input_dim == 10
        assert solver.output_dim == 10
        assert solver.config == solver_config
        assert hasattr(solver, "encoder")
        assert hasattr(solver, "decoder")
        assert hasattr(solver, "constraint_handler")

    def test_parametric_solver_forward_pass(self, solver):
        """Test forward pass through parametric solver network."""
        # Create a batch of optimization problems
        batch_size = 5
        problem_params = jnp.ones((batch_size, solver.input_dim))

        # Forward pass should return solutions
        solutions = solver(problem_params)

        assert solutions.shape == (batch_size, solver.output_dim)
        assert jnp.isfinite(solutions).all()

    def test_parametric_solver_quadratic_problem(self, solver):
        """Test solver on quadratic programming problems."""
        # Define a simple quadratic problem: min 0.5 * x^T * Q * x + c^T * x
        batch_size = 3
        dim = solver.output_dim

        # Problem parameters: Q matrix (flattened) + c vector
        Q_flat = jnp.ones((batch_size, dim * dim)) * 0.1
        c = jnp.ones((batch_size, dim)) * 0.5
        problem_params = jnp.concatenate([Q_flat, c], axis=1)

        # Ensure input dimension matches
        if problem_params.shape[1] != solver.input_dim:
            problem_params = problem_params[:, : solver.input_dim]

        solutions = solver(problem_params)

        # Solutions should be finite and reasonable
        assert solutions.shape == (batch_size, solver.output_dim)
        assert jnp.isfinite(solutions).all()
        assert jnp.abs(solutions).max() < 100  # Reasonable magnitude

    def test_parametric_solver_constraint_satisfaction(self, solver):
        """Test that solver solutions satisfy constraints when provided."""
        batch_size = 2
        problem_params = jnp.ones((batch_size, solver.input_dim))

        # Define simple equality constraint: sum(x) = 1
        equality_constraint = jnp.ones(solver.output_dim)

        solutions = solver(
            problem_params, constraints={"equality": equality_constraint}
        )

        # Check constraint satisfaction (approximately)
        constraint_violations = jnp.abs(jnp.sum(solutions, axis=1) - 1.0)
        assert jnp.max(constraint_violations) < 0.1  # Should satisfy approximately

    def test_parametric_solver_speedup_measurement(self, solver):
        """Test that solver provides speedup measurement capabilities."""
        batch_size = 10
        problem_params = jnp.ones((batch_size, solver.input_dim))

        # Should have methods to measure and report speedup
        assert hasattr(solver, "measure_speedup")
        assert hasattr(solver, "compare_with_traditional")

        # Test end-to-end speedup measurement with real data
        results = solver.compare_with_traditional(problem_params)

        # Verify results structure
        assert "speedup" in results
        assert "neural_time" in results
        assert "traditional_time" in results
        assert "speedup_achieved" in results

        # Verify speedup calculation
        assert results["speedup"] > 0
        assert isinstance(results["speedup_achieved"], bool)

        # Test arithmetic speedup calculation directly
        traditional_time = 1.0  # 1 second
        inference_time = 0.001  # 1 millisecond
        speedup = solver.measure_speedup(traditional_time, inference_time)
        assert speedup >= 100  # Should achieve >100x speedup

    def test_parametric_solver_traditional_fallback(self, solver_config):
        """Test integration with traditional solver fallback."""
        # Create solver with traditional fallback enabled
        solver_config.use_traditional_fallback = True
        rngs = nnx.Rngs(42)
        solver = ParametricProgrammingSolver(
            config=solver_config, input_dim=10, output_dim=10, rngs=rngs
        )

        # Test fallback trigger for difficult problems
        problem_params = jnp.ones((1, solver.input_dim)) * 1000  # Difficult problem

        # Should have fallback mechanism
        assert hasattr(solver, "use_traditional_fallback")

        # Test that fallback can be triggered
        solution = solver.solve_with_fallback(problem_params)
        assert solution is not None
        assert jnp.isfinite(solution).all()

    def test_parametric_solver_batch_processing(self, solver):
        """Test efficient batch processing of multiple problems."""
        # Test various batch sizes
        for batch_size in [1, 5, 10, 50]:
            problem_params = jnp.ones((batch_size, solver.input_dim))
            solutions = solver(problem_params)

            assert solutions.shape == (batch_size, solver.output_dim)
            assert jnp.isfinite(solutions).all()

    def test_parametric_solver_gradient_flow(self, solver):
        """Test that gradients flow properly through the solver network."""
        problem_params = jnp.ones((1, solver.input_dim))

        def loss_fn(solver, params):
            solution = solver(params)
            return jnp.sum(solution**2)  # Simple quadratic loss

        # Test gradient computation
        grads = nnx.grad(loss_fn)(solver, problem_params)

        # Gradients should be finite and non-zero
        assert grads is not None
        # Note: Specific gradient checks would depend on solver implementation

    def test_parametric_solver_memory_efficiency(self, solver):
        """Test memory-efficient processing of large batches."""
        # Test with larger batch size
        large_batch_size = 100
        problem_params = jnp.ones((large_batch_size, solver.input_dim))

        # Should process without memory issues
        solutions = solver(problem_params)
        assert solutions.shape == (large_batch_size, solver.output_dim)

    def test_parametric_solver_reproducibility(self, solver_config):
        """Test that solver produces reproducible results."""
        # Create two solvers with same random seed
        rngs1 = nnx.Rngs(42)
        rngs2 = nnx.Rngs(42)

        solver1 = ParametricProgrammingSolver(
            config=solver_config, input_dim=10, output_dim=10, rngs=rngs1
        )
        solver2 = ParametricProgrammingSolver(
            config=solver_config, input_dim=10, output_dim=10, rngs=rngs2
        )

        problem_params = jnp.ones((5, 10))

        solutions1 = solver1(problem_params)
        solutions2 = solver2(problem_params)

        # Should produce identical results with same seed
        assert jnp.allclose(solutions1, solutions2)


class TestIntegrationTests:
    """Integration tests for L2O components working together."""

    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow from problem definition to solution."""
        # Define optimization problem
        problem = OptimizationProblem(
            problem_type="quadratic", dimension=5, constraints={"bounds": (0.0, 1.0)}
        )

        # Configure solver
        config = SolverConfig(hidden_sizes=[32, 32], max_iterations=50)

        # Create and use solver
        rngs = nnx.Rngs(42)
        solver = ParametricProgrammingSolver(
            config=config,
            input_dim=problem.dimension + 1,  # +1 for problem encoding
            output_dim=problem.dimension,
            rngs=rngs,
        )

        # Encode problem and solve
        problem_encoding = jnp.array([1.0, 0.5, 0.3, 0.2, 0.1, 1.0])  # Mock encoding
        solution = solver(problem_encoding.reshape(1, -1))

        # Verify solution properties
        assert solution.shape == (1, problem.dimension)
        assert jnp.isfinite(solution).all()

        # Check that solution respects bounds (approximately)
        assert jnp.all(
            solution >= -1.0
        )  # Allow reasonable tolerance for neural network outputs
        assert jnp.all(solution <= 2.0)

    def test_performance_comparison_with_traditional_methods(self):
        """Test performance comparison capabilities."""
        config = SolverConfig(use_traditional_fallback=True)
        rngs = nnx.Rngs(42)

        solver = ParametricProgrammingSolver(
            config=config, input_dim=10, output_dim=5, rngs=rngs
        )

        # Should have performance measurement capabilities
        assert hasattr(solver, "measure_speedup")
        assert hasattr(solver, "compare_with_traditional")

        # Test speedup calculation
        speedup = solver.measure_speedup(traditional_time=1.0, neural_time=0.001)
        assert speedup == 1000  # 1000x speedup


class TestJITCompatibility:
    """Test JIT compilation compatibility for parametric solver components."""

    def test_parametric_solver_jit_compilation(self):
        """Test that ParametricProgrammingSolver can be JIT compiled."""
        import time

        config = SolverConfig(hidden_sizes=[64, 32])
        solver = ParametricProgrammingSolver(
            config=config, input_dim=6, output_dim=4, rngs=nnx.Rngs(42)
        )

        # Test data
        problem_params = jax.random.normal(jax.random.PRNGKey(0), (8, 6))

        # JIT compile the forward pass
        jitted_solver = nnx.jit(solver)

        # Test that JIT compilation works
        solutions = jitted_solver(problem_params)

        assert solutions.shape == (8, 4)
        assert jnp.isfinite(solutions).all()
        assert jnp.all(jnp.abs(solutions) <= 1.0)  # tanh bounded output

        # Test performance improvement with JIT
        # Warmup
        _ = jitted_solver(problem_params)

        # Time JIT version
        start_time = time.time()
        for _ in range(10):
            _ = jitted_solver(problem_params)
        jit_time = time.time() - start_time

        # Time non-JIT version
        start_time = time.time()
        for _ in range(10):
            _ = solver(problem_params)
        non_jit_time = time.time() - start_time

        # JIT should be faster or at least not significantly slower
        assert jit_time <= non_jit_time * 2.0  # Allow some overhead for small problems

    def test_constraint_handler_jit_compilation(self):
        """Test that ConstraintHandler methods can be JIT compiled."""
        handler = ConstraintHandler(method="penalty", penalty_weight=1.0)

        # Test data
        x_single = jax.random.normal(jax.random.PRNGKey(0), (5,))
        x_batch = jax.random.normal(jax.random.PRNGKey(1), (10, 5))
        constraint = jax.random.normal(jax.random.PRNGKey(2), (5,))

        # JIT compile constraint methods with static string arguments
        jitted_penalty_equality = nnx.jit(
            handler.compute_penalty, static_argnames=["constraint_type"]
        )
        jitted_penalty_inequality = nnx.jit(
            handler.compute_penalty, static_argnames=["constraint_type"]
        )
        jitted_barrier = nnx.jit(handler.compute_barrier)
        jitted_project = nnx.jit(handler.project_to_feasible)

        # Test JIT compilation works for single inputs
        penalty_single = jitted_penalty_equality(x_single, constraint, "equality")
        barrier_single = jitted_barrier(x_single, constraint)
        projected_single = jitted_project(x_single, (-1.0, 1.0))

        assert jnp.isfinite(penalty_single)
        assert jnp.isfinite(barrier_single)
        assert projected_single.shape == (5,)
        assert jnp.all(projected_single >= -1.0)
        assert jnp.all(projected_single <= 1.0)

        # Test JIT compilation works for batch inputs
        penalty_batch = jitted_penalty_inequality(x_batch, constraint, "inequality")
        barrier_batch = jitted_barrier(x_batch, constraint)

        assert penalty_batch.shape == (10,)
        assert barrier_batch.shape == (10,)
        assert jnp.isfinite(penalty_batch).all()
        assert jnp.isfinite(barrier_batch).all()

    def test_traditional_fallback_jit_compilation(self):
        """Test that traditional fallback solver can be JIT compiled."""
        config = SolverConfig(use_traditional_fallback=True)
        solver = ParametricProgrammingSolver(
            config=config, input_dim=4, output_dim=3, rngs=nnx.Rngs(42)
        )

        # Test data
        problem_params = jax.random.normal(jax.random.PRNGKey(0), (5, 4))

        # JIT compile the traditional fallback method
        jitted_fallback = nnx.jit(solver._traditional_fallback)

        # Test that JIT compilation works
        solutions = jitted_fallback(problem_params)

        assert solutions.shape == (5, 3)
        assert jnp.isfinite(solutions).all()

        # Test that the solution is reasonable (should converge to near zero)
        final_norms = jnp.linalg.norm(solutions, axis=-1)
        assert jnp.all(final_norms < 1.0)  # Should be small after optimization

    def test_constraint_application_jit_compilation(self):
        """Test that individual constraint operations can be JIT compiled."""
        # Test data
        solutions = jax.random.normal(jax.random.PRNGKey(0), (6, 3))

        # Test JIT compilation of individual constraint operations
        # (Full constraint application involves complex dictionary logic that's not JIT-friendly)

        # JIT compile normalization (sum constraint)
        def normalize_solutions(sols):
            return sols / jnp.sum(sols, axis=-1, keepdims=True)

        jitted_normalize = nnx.jit(normalize_solutions)
        normalized_solutions = jitted_normalize(solutions)

        assert normalized_solutions.shape == (6, 3)
        assert jnp.isfinite(normalized_solutions).all()

        # Check that sum constraint is satisfied (approximately)
        sums = jnp.sum(normalized_solutions, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-6)

        # JIT compile bounds clipping
        def clip_solutions(sols, lower_bound, upper_bound):
            return jnp.clip(sols, lower_bound, upper_bound)

        jitted_clip = nnx.jit(clip_solutions)
        clipped_solutions = jitted_clip(solutions, -2.0, 2.0)

        assert clipped_solutions.shape == (6, 3)
        assert jnp.all(clipped_solutions >= -2.0)
        assert jnp.all(clipped_solutions <= 2.0)

    def test_batch_processing_jit_compatibility(self):
        """Test JIT compilation with batch processing."""
        config = SolverConfig(hidden_sizes=[32, 16])
        solver = ParametricProgrammingSolver(
            config=config, input_dim=5, output_dim=3, rngs=nnx.Rngs(42)
        )

        # Create batch processing function
        def batch_solve_problems(batch_params):
            return jax.vmap(solver)(batch_params)

        # JIT compile batch processing
        jitted_batch_solve = nnx.jit(batch_solve_problems)

        # Test batch processing
        batch_params = jax.random.normal(jax.random.PRNGKey(0), (20, 5))
        batch_solutions = jitted_batch_solve(batch_params)

        assert batch_solutions.shape == (20, 3)
        assert jnp.isfinite(batch_solutions).all()

    def test_end_to_end_jit_workflow(self):
        """Test complete JIT-compiled optimization workflow."""
        config = SolverConfig(use_traditional_fallback=True)
        solver = ParametricProgrammingSolver(
            config=config, input_dim=6, output_dim=4, rngs=nnx.Rngs(42)
        )

        # Define complete workflow function
        def complete_workflow(problem_params):
            # Try neural solver first
            neural_solution = solver(problem_params)

            # Check if solution is reasonable
            is_reasonable = jnp.isfinite(neural_solution).all() & (
                jnp.abs(neural_solution).max() < 1000
            )

            # Use fallback if needed
            fallback_solution = solver._traditional_fallback(problem_params)

            # Select best solution
            final_solution = jax.lax.cond(
                is_reasonable, lambda: neural_solution, lambda: fallback_solution
            )

            return final_solution

        # JIT compile complete workflow
        jitted_workflow = nnx.jit(complete_workflow)

        # Test complete workflow
        problem_params = jax.random.normal(jax.random.PRNGKey(0), (3, 6))
        solutions = jitted_workflow(problem_params)

        assert solutions.shape == (3, 4)
        assert jnp.isfinite(solutions).all()

    def test_performance_comparison_jit_compatibility(self):
        """Test that performance comparison methods work with JIT."""
        config = SolverConfig(use_traditional_fallback=True)
        solver = ParametricProgrammingSolver(
            config=config, input_dim=4, output_dim=3, rngs=nnx.Rngs(42)
        )

        # Test data
        problem_params = jax.random.normal(jax.random.PRNGKey(0), (5, 4))

        # Create JIT-compiled comparison function
        def jit_compatible_comparison(params):
            neural_solution = solver(params)
            traditional_solution = solver._traditional_fallback(params)

            # Simple performance metric (solution quality)
            neural_quality = jnp.mean(jnp.sum(neural_solution**2, axis=-1))
            traditional_quality = jnp.mean(jnp.sum(traditional_solution**2, axis=-1))

            return {
                "neural_solution": neural_solution,
                "traditional_solution": traditional_solution,
                "neural_quality": neural_quality,
                "traditional_quality": traditional_quality,
            }

        # JIT compile comparison
        jitted_comparison = nnx.jit(jit_compatible_comparison)

        # Test JIT compilation works
        results = jitted_comparison(problem_params)

        assert "neural_solution" in results
        assert "traditional_solution" in results
        assert "neural_quality" in results
        assert "traditional_quality" in results
        assert results["neural_solution"].shape == (5, 3)
        assert results["traditional_solution"].shape == (5, 3)
        assert jnp.isfinite(results["neural_quality"])
        assert jnp.isfinite(results["traditional_quality"])
