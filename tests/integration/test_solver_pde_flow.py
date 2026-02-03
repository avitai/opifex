"""Integration tests for the unified Solver-PDE-Geometry flow.

This module tests the end-to-end interaction between:
1. Geometries (Rectangle, etc.)
2. PDEProblems (defined with geometries)
3. Solvers (PINNSolver, etc.)
4. The Solution object
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.core.problems import create_pde_problem
from opifex.core.solver.interface import SolverConfig, SolverState
from opifex.geometry.csg import Rectangle
from opifex.solvers.pinn import PINNSolver


class SimpleMLP(nnx.Module):
    """Simple MLP for testing."""

    def __init__(self, key: jax.Array):
        self.dense1 = nnx.Linear(2, 32, rngs=nnx.Rngs(key))
        self.dense2 = nnx.Linear(32, 1, rngs=nnx.Rngs(key))

    def __call__(self, x):
        return self.dense2(nnx.relu(self.dense1(x)))


def poisson_equation(x, u, u_derivs):
    """Poisson equation: Laplacian(u) = -1."""
    # This assumes we have a way to calculate derivatives which PINNSolver currently mocks
    # But checking the flow only requires the function to exist.
    return u_derivs["d2x"] + u_derivs["d2y"] + 1.0


def test_full_pinn_flow():
    """Test the complete flow from geometry creation to solution."""

    # 1. Define Geometry
    rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=2.0)

    # 2. Define Problem
    problem = create_pde_problem(
        geometry=rect,
        equation=poisson_equation,
        boundary_conditions={"boundary": 0.0},  # Dirichlet 0 on all boundaries
    )

    # 3. Initialize Solver
    rng = jax.random.key(42)
    model = SimpleMLP(rng)
    solver = PINNSolver(model=model)

    # 4. Configure Solve
    config = SolverConfig(
        max_iterations=5,  # Short run for integration check
        tolerance=1e-3,
        verbose=False,
    )
    state = SolverState(rng_key=rng, step=0)

    # 5. Execute Solve
    solution = solver.solve(problem, initial_state=state, config=config)

    # 6. Verify Solution
    assert solution is not None
    assert isinstance(solution.metrics, dict)
    assert "final_train_loss" in solution.metrics
    assert solution.execution_time > 0
    # Convergence unlikely in 5 steps, but we check the flag exists
    assert hasattr(solution, "converged")

    # 7. Check that we can access stats
    assert "loss" in solution.stats
