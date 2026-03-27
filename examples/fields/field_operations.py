# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Field Operations: Differential Operators and Fluid Simulation
#
# This example demonstrates Opifex's JAX-native field abstractions for
# scientific computing on structured grids, inspired by PhiFlow.

# %% Imports
import jax.numpy as jnp
import matplotlib as mpl


mpl.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

from opifex.fields import (
    Box,
    CenteredGrid,
    curl_2d,
    divergence,
    Extrapolation,
    gradient,
    laplacian,
    pressure_solve_spectral,
    semi_lagrangian,
)


OUTPUT_DIR = Path("docs/assets/examples/field_operations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Creating Fields on Physical Domains

# %% Create a scalar field
n = 128
box = Box(lower=(0.0, 0.0), upper=(2 * jnp.pi, 2 * jnp.pi))
coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()

# Scalar field: u(x, y) = sin(x) * cos(y)
u_values = jnp.sin(coords[..., 0]) * jnp.cos(coords[..., 1])
u = CenteredGrid(u_values, box, Extrapolation.PERIODIC)
print(f"Scalar field: {u}")
print(f"Resolution: {u.resolution}, Cell size: dx={u.dx}")

# %% [markdown]
# ## 2. Differential Operators

# %% Gradient
grad_u = gradient(u)
print(f"Gradient shape: {grad_u.values.shape}")  # (128, 128, 2)

# Analytical: ∇u = (cos(x)cos(y), -sin(x)sin(y))
grad_exact_x = jnp.cos(coords[..., 0]) * jnp.cos(coords[..., 1])
grad_exact_y = -jnp.sin(coords[..., 0]) * jnp.sin(coords[..., 1])
grad_error = jnp.max(jnp.abs(grad_u.values[..., 0] - grad_exact_x))
print(f"Gradient x-component max error: {grad_error:.6f}")

# %% Laplacian
lap_u = laplacian(u)
# Analytical: ∇²u = -2 sin(x) cos(y)
lap_exact = -2.0 * jnp.sin(coords[..., 0]) * jnp.cos(coords[..., 1])
lap_error = jnp.max(jnp.abs(lap_u.values - lap_exact))
print(f"Laplacian max error: {lap_error:.6f}")

# %% Visualize operators
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

im0 = axes[0].imshow(u.values.T, origin="lower", cmap="RdBu_r")
axes[0].set_title("u = sin(x)cos(y)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(grad_u.values[..., 0].T, origin="lower", cmap="RdBu_r")
axes[1].set_title("∂u/∂x")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

im2 = axes[2].imshow(grad_u.values[..., 1].T, origin="lower", cmap="RdBu_r")
axes[2].set_title("∂u/∂y")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

im3 = axes[3].imshow(lap_u.values.T, origin="lower", cmap="RdBu_r")
axes[3].set_title("∇²u")
plt.colorbar(im3, ax=axes[3], shrink=0.8)

plt.suptitle("Differential Operators on sin(x)cos(y)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "differential_operators.png", dpi=150)
print(f"Saved: {OUTPUT_DIR / 'differential_operators.png'}")
plt.close()

# %% [markdown]
# ## 3. Vorticity (2D Curl)

# %% Create a velocity field with rotation
# v = (-sin(y), sin(x)) — has nonzero vorticity
vx = -jnp.sin(coords[..., 1])
vy = jnp.sin(coords[..., 0])
velocity = CenteredGrid(jnp.stack([vx, vy], axis=-1), box, Extrapolation.PERIODIC)

vorticity = curl_2d(velocity)
print(f"Max vorticity: {jnp.max(vorticity.values):.4f}")

# Divergence of this field
div_v = divergence(velocity)
print(f"Max divergence: {jnp.max(jnp.abs(div_v.values)):.6f} (should be ~0)")

# %% [markdown]
# ## 4. Semi-Lagrangian Advection

# %% Advect a Gaussian pulse
cx, cy = jnp.pi, jnp.pi
pulse = jnp.exp(-2 * ((coords[..., 0] - cx) ** 2 + (coords[..., 1] - cy) ** 2))
field = CenteredGrid(pulse, box, Extrapolation.PERIODIC)

# Uniform velocity to the right
vel_uniform = CenteredGrid(
    jnp.stack([jnp.ones((n, n)), jnp.zeros((n, n))], axis=-1),
    box,
    Extrapolation.PERIODIC,
)

# Advect for several steps
advected = field
for _step in range(10):
    advected = semi_lagrangian(advected, vel_uniform, dt=0.1)

print(f"Peak before advection: {jnp.max(field.values):.4f}")
print(f"Peak after advection:  {jnp.max(advected.values):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(field.values.T, origin="lower", cmap="hot")
axes[0].set_title("Initial Gaussian pulse")
axes[1].imshow(advected.values.T, origin="lower", cmap="hot")
axes[1].set_title("After 10 semi-Lagrangian steps")
plt.suptitle("Semi-Lagrangian Advection")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "advection.png", dpi=150)
print(f"Saved: {OUTPUT_DIR / 'advection.png'}")
plt.close()

# %% [markdown]
# ## 5. Incompressible Pressure Projection

# %% Create a divergent velocity field and project it
vx_div = jnp.sin(coords[..., 0])
vy_div = jnp.sin(coords[..., 1])
vel_div = CenteredGrid(jnp.stack([vx_div, vy_div], axis=-1), box, Extrapolation.PERIODIC)

div_before = divergence(vel_div)
print(f"Divergence before projection: max={jnp.max(jnp.abs(div_before.values)):.4f}")

projected, pressure = pressure_solve_spectral(vel_div)
div_after = divergence(projected)
print(f"Divergence after projection:  max={jnp.max(jnp.abs(div_after.values)):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
im0 = axes[0].imshow(div_before.values.T, origin="lower", cmap="RdBu_r")
axes[0].set_title("Divergence (before)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(pressure.values.T, origin="lower", cmap="viridis")
axes[1].set_title("Pressure field")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

im2 = axes[2].imshow(div_after.values.T, origin="lower", cmap="RdBu_r")
axes[2].set_title("Divergence (after)")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

plt.suptitle("Pressure Projection (Helmholtz-Hodge Decomposition)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pressure_projection.png", dpi=150)
print(f"Saved: {OUTPUT_DIR / 'pressure_projection.png'}")
plt.close()

# %% [markdown]
# ## Summary
#
# | Operation | Input | Output | Method |
# |-----------|-------|--------|--------|
# | `gradient(u)` | Scalar | Vector | Central FD, O(h²) |
# | `laplacian(u)` | Scalar | Scalar | Central FD, O(h²) |
# | `divergence(v)` | Vector | Scalar | Central FD, O(h²) |
# | `curl_2d(v)` | Vector | Scalar | Central FD, O(h²) |
# | `semi_lagrangian` | Scalar + Velocity | Scalar | Backward trace + bilinear interp |
# | `pressure_solve_spectral` | Vector | Vector + Scalar | FFT Poisson solver |
