"""Tests for EnsembleWrapper (Multi-Model UQ).

Adheres to SciMLSolver protocol and TDD principles.
"""

import jax.numpy as jnp

from opifex.core.problems import create_optimization_problem
from opifex.core.solver.interface import (
    SciMLSolver,
    Solution,
)
from opifex.solvers.wrappers import EnsembleWrapper


class MockProtoSolver(SciMLSolver):
    """Mock solver adhering to protocol."""

    def __init__(self, id_val=0):
        self.id_val = id_val

    def solve(self, problem, initial_state=None, config=None):
        # Return a solution with a field that identifies this solver instance
        return Solution(
            fields={"u": jnp.array([float(self.id_val)])},
            metrics={"final_loss": 0.1},
            execution_time=0.1,
            converged=True,
        )


def test_ensemble_initialization():
    """Test that EnsembleWrapper initializes N independent solvers."""
    num_models = 5
    solvers = [MockProtoSolver(i) for i in range(num_models)]
    wrapper = EnsembleWrapper(solvers=solvers)

    assert len(wrapper.solvers) == num_models
    assert isinstance(wrapper.solvers[0], MockProtoSolver)


def test_ensemble_solve_aggregation():
    """Test that solve runs all models and aggregates results."""
    # Create wrapper with 3 solvers, each returning [0.0], [1.0], [2.0] respectively
    solvers = [MockProtoSolver(id_val=i) for i in range(3)]

    wrapper = EnsembleWrapper(solvers=solvers)
    problem = create_optimization_problem(1, lambda x: x)

    solution = wrapper.solve(problem)

    # Check aggregation
    # Mean of [0, 1, 2] is 1.0
    u_mean = solution.fields["u"]
    u_std = solution.fields["u_std"]

    assert jnp.allclose(u_mean, jnp.array([1.0]))
    assert jnp.allclose(
        u_std, jnp.array([0.81649658])
    )  # Population std of 0,1,2 is sqrt(2/3) ~= 0.816

    # Check auxiliary metrics
    assert solution.metrics["ensemble_size"] == 3


def test_ensemble_parallel_execution_mock():
    """Test structure for parallel execution (mocking verify it calls all)."""
    solvers = [MockProtoSolver(1), MockProtoSolver(2)]
    wrapper = EnsembleWrapper(solvers=solvers)
    problem = create_optimization_problem(1, lambda x: x)

    solution = wrapper.solve(problem)
    assert solution.converged is True
