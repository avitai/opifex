# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# Your First PINN: Solving the Poisson Equation

| Property      | Value                                |
|---------------|--------------------------------------|
| Level         | Beginner                             |
| Runtime       | ~30 seconds                          |
| Memory        | ~500 MB                              |
| Prerequisites | `source activate.sh`                 |

## Overview

Solve the 1D Poisson equation with a Physics-Informed Neural Network (PINN)
using Opifex's high-level APIs.

This example demonstrates:
- **Interval**: 1D geometry for computational domain
- **create_poisson_pinn**: Factory function for Poisson PINN architecture
- **PINNSolver**: High-level solver with generic solve() API
- **poisson_residual**: Factory function to create PDE residual
- **PINNConfig**: Configuration for solver parameters

**Problem:** Find u(x) satisfying:
- PDE: -u''(x) = pi^2 * sin(pi*x)  on [-1, 1]
- BCs: u(-1) = u(1) = 0

**Exact Solution:** u(x) = sin(pi*x)

We'll achieve <0.5% L2 relative error using Opifex's built-in APIs.
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
from flax import nnx

from opifex.geometry import Interval
from opifex.neural.pinns import create_poisson_pinn
from opifex.solvers import PINNConfig, PINNSolver, poisson_residual


mpl.use("Agg")
import matplotlib.pyplot as plt


print("=" * 60)
print("Your First PINN: 1D Poisson Equation (Opifex APIs)")
print("=" * 60)
print(f"JAX backend: {jax.default_backend()}")

# %% [markdown]
"""
## The Problem

The 1D Poisson equation is a fundamental elliptic PDE:

    -d²u/dx² = f(x)

With source term f(x) = pi² sin(pi x) and boundary conditions u(-1) = u(1) = 0,
the exact solution is u(x) = sin(pi x).

This is the perfect first PINN example because:
1. Exact solution is known (we can measure error precisely)
2. Simple 1D domain (easy to visualize)
3. Demonstrates core PINN workflow with Opifex APIs
"""


# %%
def exact_solution(x):
    """Analytical solution: u(x) = sin(pi*x)."""
    return jnp.sin(jnp.pi * x)


def source_term(x):
    """Source term: f(x) = pi^2 * sin(pi*x)."""
    return jnp.pi**2 * jnp.sin(jnp.pi * x)


def boundary_condition(x):
    """Boundary condition: u = 0 at boundaries."""
    return jnp.zeros_like(x[..., 0])


# %% [markdown]
"""
## Step 1: Define the Geometry

Use Opifex's `Interval` class for 1D domains.
"""

# %%
print()
print("Defining geometry using Interval...")
geometry = Interval(-1.0, 1.0)
print(f"  Domain: [{geometry.a}, {geometry.b}]")
print(f"  Length: {geometry.length}")

# %% [markdown]
"""
## Step 2: Create the PINN Model

Use `create_poisson_pinn` factory to create an appropriate architecture.
"""

# %%
print()
print("Creating PINN model using create_poisson_pinn()...")
pinn = create_poisson_pinn(
    spatial_dim=1,
    hidden_dims=[50, 50, 50],
    rngs=nnx.Rngs(42),
)

n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
print("  Architecture: 1 -> 50 -> 50 -> 50 -> 1")
print(f"  Parameters: {n_params:,}")
print("  Activation: tanh")

# %% [markdown]
"""
## Step 3: Create PDE Residual and Solve

Use `poisson_residual()` factory to create the residual function.
The standard Poisson equation is: -∇²u = f(x).

The solver uses:
- `AutoDiffEngine` for computing Laplacians via autodiff
- `PhysicsLossComposer` for loss composition (via PINNConfig.loss_config)
"""

# %%
print()
print("Creating PDE residual using poisson_residual() factory...")
residual_fn = poisson_residual(source_term)
print("  PDE: -∇²u = π² sin(πx)")
print("  Residual: -∇²u - f(x) = 0")

# %%
print()
print("Configuring solver with PINNConfig...")
config = PINNConfig(
    n_interior=100,
    n_boundary=2,
    num_iterations=2000,
    learning_rate=1e-3,
    print_every=500,
    seed=42,
)
print(f"  Interior points: {config.n_interior}")
print(f"  Boundary points: {config.n_boundary}")
print(f"  Iterations: {config.num_iterations}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Physics loss weight: {config.loss_config.physics_loss_weight}")
print(f"  Boundary loss weight: {config.loss_config.boundary_loss_weight}")

print()
print("Solving with PINNSolver.solve()...")
print("-" * 50)

solver = PINNSolver(pinn)
result = solver.solve(
    geometry=geometry,
    residual_fn=residual_fn,
    bc_fn=boundary_condition,
    config=config,
)

print("-" * 50)
print(f"Training completed in {result.training_time:.1f}s")
print(f"Final loss: {result.final_loss:.6e}")

# %% [markdown]
"""
## Step 4: Evaluate Solution

Compare PINN prediction against the exact solution.
"""

# %%
print()
print("Evaluating solution...")

# Dense evaluation grid
x_eval = jnp.linspace(-1, 1, 200).reshape(-1, 1)
u_pred = result.model(x_eval).squeeze()
u_exact = exact_solution(x_eval.squeeze())

# Compute errors
abs_error = jnp.abs(u_pred - u_exact)
l2_error = jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2))
l2_relative = l2_error / jnp.sqrt(jnp.mean(u_exact**2))
max_error = jnp.max(abs_error)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  L2 Absolute Error:  {l2_error:.6f}")
print(f"  L2 Relative Error:  {l2_relative:.4%}")
print(f"  Maximum Error:      {max_error:.6f}")
print("=" * 60)

# %% [markdown]
"""
## Visualization

Side-by-side comparison of PINN solution vs exact solution.
"""

# %%
OUTPUT_DIR = Path("docs/assets/examples/first_pinn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Solution comparison
ax = axes[0]
ax.plot(x_eval, u_exact, "b-", linewidth=2, label="Exact: sin(pi*x)")
ax.plot(x_eval, u_pred, "r--", linewidth=2, label="PINN prediction")
ax.set_xlabel("x")
ax.set_ylabel("u(x)")
ax.set_title("Solution Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# Pointwise error
ax = axes[1]
ax.semilogy(x_eval, abs_error, "k-", linewidth=1.5)
ax.set_xlabel("x")
ax.set_ylabel("|u_pred - u_exact|")
ax.set_title(f"Pointwise Error (Max: {max_error:.2e})")
ax.grid(True, alpha=0.3)

# Training loss
ax = axes[2]
ax.semilogy(result.losses, "b-", linewidth=1)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
ax.grid(True, alpha=0.3)

plt.suptitle(
    f"1D Poisson PINN: L2 Relative Error = {l2_relative:.4%}",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "solution.png", dpi=150, bbox_inches="tight")
plt.close()

print()
print(f"Saved: {OUTPUT_DIR / 'solution.png'}")

# %% [markdown]
"""
## Summary

In this example, we used Opifex's high-level PINN APIs:

1. **Interval**: 1D geometry for domain [-1, 1]
2. **create_poisson_pinn**: Factory for creating Poisson PINN architecture
3. **poisson_residual**: Factory to create PDE residual function
4. **PINNSolver.solve**: Generic solver that accepts any residual function
5. **PINNConfig**: Configuration composing with PhysicsLossConfig for loss weights

**Key Takeaway:** Opifex uses factory functions (not string keys) for PDEs,
making the API explicit, type-safe, and infinitely extensible!

## Next Steps

- [Poisson 2D](../pinns/poisson.md) - Same problem in 2D
- [Burgers Equation](../pinns/burgers.md) - Nonlinear PDE
- [Heat Equation](../pinns/heat-equation.md) - Time-dependent problem
"""

# %%
print()
print("PINN example completed successfully!")
print(f"Achieved {l2_relative:.4%} L2 relative error with {n_params:,} parameters")
print()
print("Opifex APIs demonstrated:")
print("  - Interval (1D geometry)")
print("  - create_poisson_pinn (PINN factory)")
print("  - poisson_residual (PDE residual factory)")
print("  - PINNSolver.solve (generic solver)")
print("  - PINNConfig (solver configuration)")
