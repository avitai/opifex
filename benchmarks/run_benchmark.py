"""Comparative Benchmark: PINN vs Neural Operator.

Compares performance on a standard problem (1D Wave Equation or Poisson).
"""

import time

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.core.problems import create_data_driven_problem, create_pde_problem
from opifex.core.solver.interface import SolverConfig
from opifex.geometry.csg import Rectangle
from opifex.solvers import NeuralOperatorSolver, PINNSolver


def benchmark_poisson_2d():
    """Run a comparative benchmark between PINN and Neural Operator on 2D Poisson."""
    print("=== Benchmarking 2D Poisson Equation ===")  # noqa: T201

    # 1. Problem Definition
    geometry = Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0)

    # u_xx + u_yy = -2*pi^2 * sin(pi*x)*sin(pi*y)
    # Exact: u = sin(pi*x)*sin(pi*y)
    def exact_u(x):
        return jnp.sin(jnp.pi * x[..., 0]) * jnp.sin(jnp.pi * x[..., 1])

    def pde_residual(u, x, params):
        u_xx = jax.grad(lambda x: jax.grad(u)(x)[0])(x)[0]
        u_yy = jax.grad(lambda x: jax.grad(u)(x)[1])(x)[1]
        source = -2 * (jnp.pi**2) * exact_u(x)
        return u_xx + u_yy - source

    pinn_problem = create_pde_problem(
        geometry=geometry,
        equation=pde_residual,
        boundary_conditions={"type": "dirichlet", "value": 0.0},
        parameters={},
    )

    # Define Simple MLP Model
    class SimpleMLP(nnx.Module):
        def __init__(self, in_features, hidden, out_features, rngs):
            self.linear1 = nnx.Linear(in_features, hidden, rngs=rngs)
            self.linear2 = nnx.Linear(hidden, hidden, rngs=rngs)
            self.linear3 = nnx.Linear(hidden, out_features, rngs=rngs)
            self.relu = nnx.relu

        def __call__(self, x):
            x = self.relu(self.linear1(x))
            x = self.relu(self.linear2(x))
            return self.linear3(x)

    # 2. PINN Benchmark
    print("\n[PINN] Training...")  # noqa: T201
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)

    # Model
    pinn_model = SimpleMLP(in_features=2, hidden=32, out_features=1, rngs=rngs)

    pinn_config = SolverConfig(
        max_iterations=200
    )  # FIXED: max_epochs -> max_iterations
    pinn_solver = PINNSolver(model=pinn_model)

    start_time = time.time()
    _pinn_solution = pinn_solver.solve(pinn_problem, config=pinn_config)
    pinn_time = time.time() - start_time

    # Evaluate Error
    test_points = geometry.sample_interior(100, key)  # (100, 2)
    # PINN solution.fields currently empty in prototype solver!
    # We must use model manually for now as solver.solve returns empty fields
    pinn_pred = pinn_model(test_points)
    pinn_exact = exact_u(test_points)
    pinn_error = jnp.mean((pinn_pred.flatten() - pinn_exact.flatten()) ** 2)

    print(f"[PINN] Time: {pinn_time:.2f}s | MSE: {pinn_error:.2e}")  # noqa: T201

    # 3. Neural Operator Benchmark (requires data)
    print("\n[Neural Operator] Generating Data...")  # noqa: T201

    # Sample training data
    x_train = geometry.sample_interior(256, key)
    y_train = exact_u(x_train).reshape(-1, 1)

    no_problem = create_data_driven_problem((x_train, y_train))

    print("[Neural Operator] Training...")  # noqa: T201
    no_model = SimpleMLP(in_features=2, hidden=32, out_features=1, rngs=rngs)

    no_solver = NeuralOperatorSolver(model=no_model)
    no_config = SolverConfig(max_iterations=200)

    start_time = time.time()
    _no_solution = no_solver.solve(no_problem, config=no_config)
    no_time = time.time() - start_time

    no_pred = no_model(test_points)
    no_error = jnp.mean((no_pred.flatten() - pinn_exact.flatten()) ** 2)

    print(f"[NO]   Time: {no_time:.2f}s | MSE: {no_error:.2e}")  # noqa: T201

    # 4. Comparative Summary
    print("\n=== Summary ===")  # noqa: T201
    print(f"PINN Error: {pinn_error:.2e}")  # noqa: T201
    print(f"NO Error:   {no_error:.2e}")  # noqa: T201
    print(f"Speedup:    {pinn_time / no_time:.2f}x (NO vs PINN)")  # noqa: T201


if __name__ == "__main__":
    benchmark_poisson_2d()
