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
from opifex.solvers.wrappers import BayesianWrapper, ConformalWrapper, EnsembleWrapper


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


def test_ensemble_wrapper():
    """Test EnsembleWrapper aggregates results from multiple solvers."""
    solvers = [MockSolver(), MockSolver(), MockSolver()]
    wrapper = EnsembleWrapper(solvers)

    problem = create_optimization_problem(1, lambda x: x)
    solution = wrapper.solve(problem)

    assert "u" in solution.fields
    assert "u_std" in solution.fields
    assert solution.metrics["ensemble_size"] == 3


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
