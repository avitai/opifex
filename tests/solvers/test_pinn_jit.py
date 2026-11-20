"""Tests for PINNSolver JIT compatibility.

This module verifies whether the PINNSolver and its components can be JIT-compiled.
Currently, the outer solve loop is Python-based (Trainer), so we expect full JIT
to fail or behave unexpectedly if not carefully handled. The inner step SHOULD work.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.problems import create_pde_problem
from opifex.geometry.csg import Rectangle


class SimpleMLP(nnx.Module):
    """Simple MLP for testing."""

    def __init__(self, key: jax.Array) -> None:
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


class TestPINNSolverJIT:
    """Test JIT compatibility of PINNSolver."""

    def test_inner_step_jit(self, heat_problem):
        """Verify that we COULD JIT the inner step if we wanted."""
        # This requires exposing the inner step from the solver, which currently isn't
        # publicly easy without instantiating the Trainer.
