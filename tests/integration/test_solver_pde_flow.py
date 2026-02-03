"""Integration tests for the unified Solver-PDE-Geometry flow.

This module tests the end-to-end interaction between:
1. Geometries (Rectangle, etc.)
2. PDE residual functions (explicit factory approach)
3. Solvers (PINNSolver)
4. The PINNResult object
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.geometry.csg import Rectangle
from opifex.solvers.pinn import PINNConfig, PINNSolver, poisson_residual


class SimpleMLP(nnx.Module):
    """Simple MLP for testing."""

    def __init__(self, key: jax.Array):
        self.dense1 = nnx.Linear(2, 32, rngs=nnx.Rngs(key))
        self.dense2 = nnx.Linear(32, 1, rngs=nnx.Rngs(key))

    def __call__(self, x):
        return self.dense2(nnx.relu(self.dense1(x)))


def test_full_pinn_flow():
    """Test the complete flow from geometry creation to solution."""

    # 1. Define Geometry
    rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=2.0)

    # 2. Define PDE residual using factory
    # Poisson equation: -Laplacian(u) = 1
    def source_fn(x):
        """Constant source term f(x) = 1."""
        return jnp.ones(x.shape[:-1])

    residual_fn = poisson_residual(source_fn)

    # 3. Define boundary condition
    def bc_fn(x):
        """Dirichlet BC: u = 0 on boundary."""
        return jnp.zeros_like(x[..., 0])

    # 4. Initialize Solver
    rng = jax.random.key(42)
    model = SimpleMLP(rng)
    solver = PINNSolver(model=model)

    # 5. Configure Solve
    config = PINNConfig(
        n_interior=50,
        n_boundary=20,
        num_iterations=5,  # Short run for integration check
        learning_rate=1e-3,
        print_every=0,  # Suppress output
        seed=42,
    )

    # 6. Execute Solve
    result = solver.solve(rect, residual_fn, bc_fn, config)

    # 7. Verify Solution
    assert result is not None
    assert isinstance(result.metrics, dict)
    assert "final_loss" in result.metrics
    assert result.training_time > 0
    assert len(result.losses) == 5

    # 8. Check that we can access stats
    assert result.metrics["n_iterations"] == 5
    assert result.metrics["n_interior"] == 50
    assert result.metrics["n_boundary"] == 20
