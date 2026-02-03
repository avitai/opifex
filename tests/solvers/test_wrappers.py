"""Tests for Solver Wrappers (Bayesian, L2O).

Adheres to SciMLSolver protocol.
 Wrappers should wrap a base solver and enhance its behavior.
"""

import jax.numpy as jnp

from opifex.core.problems import create_optimization_problem
from opifex.core.solver.interface import (
    Solution,
)

# Import wrappers (will fail initially)
from opifex.solvers.wrappers import BayesianWrapper, ConformalWrapper, L2OWrapper


class MockSolver:
    """Simple mock solver."""

    def solve(self, problem, initial_state=None, config=None):
        return Solution(
            fields={"u": jnp.array([1.0])},
            metrics={"loss": 0.5},
            execution_time=0.1,
            converged=True,
        )


def test_bayesian_wrapper_uq():
    """Test BayesianWrapper adds uncertainty quantification (UQ)."""
    base_solver = MockSolver()
    # Monte Carlo Dropout or Ensemble style wrapper
    wrapper = BayesianWrapper(base_solver, num_samples=5)

    problem = create_optimization_problem(1, lambda x: x)

    # When solved, it should return mean solution + uncertainty metrics
    solution = wrapper.solve(problem)

    assert "u_std" in solution.fields or "uncertainty" in solution.metrics
    assert solution.metrics["num_samples"] == 5


def test_l2o_wrapper_optimization():
    """Test L2OWrapper modifies solver config/state for Learning-to-Optimize."""
    base_solver = MockSolver()
    wrapper = L2OWrapper(base_solver, l2o_model_path="dummy_path")

    problem = create_optimization_problem(1, lambda x: x)

    # L2O should potentially adjust learning rate or params
    solution = wrapper.solve(problem)

    assert solution.metrics.get("l2o_active") is True


def test_conformal_wrapper_coverage():
    """Test ConformalWrapper provides prediction intervals."""
    base_solver = MockSolver()
    # Conformal prediction wrapper
    wrapper = ConformalWrapper(base_solver, alpha=0.1)  # 90% confidence

    problem = create_optimization_problem(1, lambda x: x)

    # Solve
    solution = wrapper.solve(problem)

    # Should have intervals
    assert "u_lower" in solution.fields
    assert "u_upper" in solution.fields
    assert solution.metrics["alpha"] == 0.1
