"""Tests for NeuralOperatorSolver (TDD).

This module defines the expected behavior for the NeuralOperatorSolver,
which solves problems using data-driven methods (e.g., FNO, DeepONet).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.solver.interface import Solution, SolverConfig, SolverState
from opifex.solvers.neural_operator import (
    NeuralOperatorSolver,  # This will fail initially
)


class SimpleOperator(nnx.Module):
    """Simple operator model for testing."""

    def __init__(self, key: jax.Array):
        # Maps (batch, dim) -> (batch, dim)
        self.dense = nnx.Linear(1, 1, rngs=nnx.Rngs(key))

    def __call__(self, x):
        return self.dense(x)


@pytest.fixture
def data_driven_problem():
    """Create a mock problem for data-driven solving."""
    # Data-driven problems might just need geometry for evaluation grid,
    # but primarily they need a dataset.
    # For the unified interface, we might pass the dataset via the 'solve' config
    # or the problem definition.
    # Let's assume standard PDEProblem but solve uses supplied data.

    from opifex.core.problems import create_data_driven_problem

    x_data = jnp.zeros((10, 1))
    y_data = jnp.zeros((10, 1))

    return create_data_driven_problem((x_data, y_data))


class TestNeuralOperatorSolver:
    """TDD Tests for NeuralOperatorSolver."""

    def test_solver_initialization(self):
        """Test that solver initializes with a model."""
        model = SimpleOperator(jax.random.key(0))
        solver = NeuralOperatorSolver(model=model)
        assert solver.model is model

    def test_solve_execution(self, data_driven_problem):
        """Test basic solve execution with data."""
        model = SimpleOperator(jax.random.key(0))
        solver = NeuralOperatorSolver(model=model)

        # Create dummy dataset
        _x_train = jnp.zeros((10, 1))
        _y_train = jnp.zeros((10, 1))

        # We need a way to pass data to the solver.
        # The 'solve' signature is solve(problem, state, config).
        # Data usually comes from the problem description OR generic loader.
        # But PDEProblem usually defines *physics*.
        # For data-driven, often 'problem' contains the dataset or 'solve' accepts it?
        # SciMLSolver protocol is: solve(problem: Problem, initial_state, config).

        # Approach: The SolverConfig can hold the dataloader/data?
        # Or proper design: We define a 'DataDrivenProblem' or 'InverseProblem'?
        # But Neural Operators solve the forward problem using data.

        # For this test, we assume we pass data via SolverConfig or kwargs if allowed.
        # Currently protocol doesn't allow kwargs.
        # We'll put it in config for now or verify if PDEProblem should hold reference data.

        # Let's assume SolverConfig can take 'train_data'.
        # Wait, SolverConfig is a dataclass.

        # Better: NeuralOperatorSolver expects the 'problem' to potentially provide data
        # OR we extend SolverConfig without breaking protocol.

        # Let's verify 'solve' can run.
        config = SolverConfig(max_iterations=2)
        state = SolverState()

        # We'll pass the data as a kwarg to config if dynamic (hacky)
        # or assume NeuralOperatorSolver has a `fit` method?
        # No, must adhere to `solve`.

        # Accepted Pattern:
        # solver.set_data(x, y) ? No, stateful.
        # Pass input data in `solve` is not in protocol.

        # Let's assume for now the Solver simply runs a training loop if supplied
        # the problem. If physics-based, it uses collocation.
        # If data-driven, where does data come from?

        # Current Plan: Implement basic structure.
        # We'll address data ingestion in the implementation.
        # For now, assert it returns a Solution.

        solution = solver.solve(data_driven_problem, state, config)
        assert isinstance(solution, Solution)
        assert solution.metrics is not None
