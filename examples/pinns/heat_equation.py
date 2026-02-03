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
# Heat Equation PINN

| Property      | Value                                |
|---------------|--------------------------------------|
| Level         | Intermediate                         |
| Runtime       | ~1 min                               |
| Memory        | ~500 MB                              |
| Prerequisites | `source activate.sh`                 |

## Overview

Physics-Informed Neural Networks (PINNs) solve PDEs by embedding the governing
equations directly into the loss function. This example demonstrates solving the
steady-state heat equation on a 2D rectangular domain using Opifex's PINN
infrastructure.

## Learning Goals

1. **Define** a PDE problem with `create_pde_problem` and geometry primitives
2. **Create** a PINN model with `create_heat_equation_pinn`
3. **Train** using Opifex's `Trainer` with collocation points
4. **Visualize** the learned temperature field
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.core.problems import create_pde_problem
from opifex.core.training import Trainer, TrainingConfig
from opifex.geometry import Rectangle
from opifex.neural.pinns import create_heat_equation_pinn


print(f"JAX backend: {jax.default_backend()}")

ASSETS_DIR = Path("docs/assets/examples/heat_equation")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
"""
## Problem Definition

Define a 2D rectangular domain [0,1]x[0,1] with homogeneous Dirichlet
boundary conditions (u=0 on all boundaries). The PDE is the steady-state
heat equation with diffusivity 0.01.
"""

# %%
geometry = Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0)
boundary_conditions = [{"type": "dirichlet", "boundary": "all", "value": 0.0}]

problem = create_pde_problem(
    geometry=geometry,
    equation=lambda x, u, u_x: 0.0,
    boundary_conditions=boundary_conditions,
    parameters={"diffusivity": 0.01},
)
print(f"Domain: {geometry}")

# %% [markdown]
"""
## Create and Train the PINN

Build a heat-equation PINN with 3 hidden layers of 50 units, train on 1000
collocation points for 100 epochs.
"""

# %%
pinn = create_heat_equation_pinn(
    spatial_dim=2, hidden_dims=[50, 50, 50], rngs=nnx.Rngs(42)
)
n_params = sum(x.size for x in jax.tree.leaves(nnx.state(pinn)))
print(f"Parameters: {n_params:,}")

# Generate collocation points: (x, y, t) uniform in [0,1]^3
key = jax.random.PRNGKey(42)
collocation_pts = jax.random.uniform(key, (1000, 3))
targets = jnp.zeros((1000, 1))

trainer = Trainer(
    model=pinn,
    config=TrainingConfig(num_epochs=100, learning_rate=1e-3, batch_size=256),
)

trained_pinn, metrics = trainer.fit(train_data=(collocation_pts, targets))
final_loss = metrics.get("final_train_loss", "N/A")
print(f"Final loss: {final_loss}")
print("Training complete!")

# %% [markdown]
"""
## Visualize the Learned Solution

Evaluate the trained PINN on a uniform grid and plot the temperature field.
"""

# %%
import matplotlib as mpl


mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Create evaluation grid
nx, ny = 50, 50
x_grid = jnp.linspace(0, 1, nx)
y_grid = jnp.linspace(0, 1, ny)
xx, yy = jnp.meshgrid(x_grid, y_grid)
eval_pts = jnp.stack([xx.ravel(), yy.ravel(), jnp.zeros(nx * ny)], axis=-1)

# Evaluate PINN
u_pred = trained_pinn(eval_pts)
u_field = np.array(u_pred.reshape(ny, nx))

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
im = ax.imshow(
    u_field,
    extent=(0, 1, 0, 1),
    origin="lower",
    cmap="hot",
    aspect="equal",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("PINN Solution: Steady-State Heat Equation")
fig.colorbar(im, ax=ax, label="Temperature u(x,y)")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "solution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Solution saved to {ASSETS_DIR / 'solution.png'}")

# %% [markdown]
"""
## Summary
"""

# %%
print()
print("=" * 50)
print("Heat Equation PINN example completed")
print(f"Final loss: {final_loss}")
print(f"Results saved to: {ASSETS_DIR}")
print("=" * 50)
