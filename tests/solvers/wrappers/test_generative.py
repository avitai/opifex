"""Tests for GenerativeWrapper (Diffusion/Flow UQ).

Adheres to SciMLSolver protocol.
"""

import jax.numpy as jnp

from opifex.core.problems import create_optimization_problem
from opifex.core.solver.interface import (
    SciMLSolver,
    Solution,
)
from opifex.solvers.wrappers import GenerativeWrapper


# Mock Generative Solver that produces samples
class MockGenerativeSolver(SciMLSolver):
    """Mock solver behaving like a Generative Model (returning samples)."""

    def solve(self, problem, initial_state=None, config=None):
        # Return a "solution" that represents samples from a distribution
        # In a real scenario, this might return parameters of a distribution or raw samples
        # For this test, we assume the solver returns a batch of samples in 'u'

        # Simulate batch of 10 samples
        # Mean 0, Std 2
        batch_size = 10
        samples = jnp.linspace(-2, 2, batch_size)

        return Solution(
            fields={"u": samples},
            metrics={"log_likelihood": -10.5},
            execution_time=0.2,
            converged=True,
        )


def test_generative_wrapper_stats():
    """Test that GenerativeWrapper computes correct method-of-moments stats from samples."""
    base_solver = MockGenerativeSolver()
    wrapper = GenerativeWrapper(base_solver)

    problem = create_optimization_problem(1, lambda x: x)

    solution = wrapper.solve(problem)

    # Check fields
    assert "u_mean" in solution.fields
    assert "u_std" in solution.fields

    # Validation
    # Input was linspace(-2, 2, 10)
    # Mean should be close to 0
    # Variance of uniform-ish linspace is approx (b-a)^2/12 for uniform,
    # but here just checking it exists and is computed

    assert jnp.abs(solution.fields["u_mean"]) < 1e-5
    assert solution.fields["u_std"] > 0

    # Check that original raw samples might be preserved or summarization occurred
    # Design choice: Wrapper typically summarizes.

    assert solution.metrics["uq_method"] == "generative_sampling"


def test_generative_wrapper_config_passing():
    """Test that wrapper passes config correctly."""
    # This ensures the wrapper is transparent
