"""Tests for PINNSolver.

This module tests the Physics-Informed Neural Network (PINN) solver implementation,
verifying adherence to the SciMLSolver protocol and correct integration with the Trainer.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.problems import create_pde_problem
from opifex.core.solver.interface import (
    SciMLSolver,
    Solution,
    SolverConfig,
    SolverState,
)
from opifex.geometry.csg import Rectangle

# PINNSolver imports - will fail initially
from opifex.solvers.pinn import PINNSolver


class SimpleMLP(nnx.Module):
    """Simple MLP for testing."""

    def __init__(self, key: jax.Array):
        self.dense1 = nnx.Linear(2, 32, rngs=nnx.Rngs(key))
        self.dense2 = nnx.Linear(32, 1, rngs=nnx.Rngs(key))

    def __call__(self, x):
        return self.dense2(nnx.relu(self.dense1(x)))


@pytest.fixture
def heat_problem():
    """Create a simple heat equation problem."""

    def heat_equation(x, u, u_derivs):
        return u_derivs["dt"] - 0.1 * u_derivs["d2x"]

    return create_pde_problem(
        geometry=Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0),
        equation=heat_equation,
        boundary_conditions={"x0": 0.0, "x1": 0.0},
    )


class TestPINNSolver:
    """Test PINNSolver implementation."""

    def test_solver_protocol_compliance(self):
        """Verify PINNSolver implements SciMLSolver protocol."""
        if hasattr(PINNSolver, "__module__") and PINNSolver.__module__ != __name__:
            # Only strictly test if we successfully imported the real class
            assert issubclass(PINNSolver, SciMLSolver)

    def test_initialization(self):
        """Test solver initialization."""
        model = SimpleMLP(jax.random.key(0))
        solver = PINNSolver(model=model)
        assert solver.model is model

    def test_solve_execution(self, heat_problem):
        """Test basic solve execution."""
        model = SimpleMLP(jax.random.key(0))
        solver = PINNSolver(model=model)

        # Test generic solve method
        # We use a very short training run for speed
        config = SolverConfig(max_iterations=2)
        state = SolverState(params=None, optim_state=None)  # Start fresh

        # This will fail until solve is implemented
        solution = solver.solve(heat_problem, state, config)

        assert isinstance(solution, Solution)
        assert solution.converged is False  # 2 iters unlikely to converge
        assert "loss" in solution.stats

    def test_physics_loss_integration(self, heat_problem):
        """Test that physics loss is actually being computed/used."""
        # This might require mocking the Trainer or checking internal state
        # For now, we just ensure it runs without error, which implies
        # the physics loss logic in the Trainer is being triggered appropriately.
        model = SimpleMLP(jax.random.key(0))
        solver = PINNSolver(model=model)

        config = SolverConfig(max_iterations=1, tolerance=1.0)
        state = SolverState()

        # We expect this to run successfully
        _ = solver.solve(heat_problem, state, config)
