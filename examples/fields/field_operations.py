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
# Field Operations: Differential Operators and Fluid Simulation

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~20 s (GPU) / ~40 s (CPU) |
| **Prerequisites** | JAX, finite differences, vector calculus |
| **Format** | Python + Jupyter |
| **Memory** | ~0.5 GB |

## Overview

Opifex provides JAX-native field abstractions (a `CenteredGrid` on a physical `Box` with an
`Extrapolation` boundary rule), inspired by [PhiFlow](https://github.com/tum-pbs/PhiFlow). This
example does not just call the operators — it **validates** them against closed-form analytical
solutions and demonstrates an incompressible projection, reporting the errors as metrics.

## What You'll Learn

1. Build fields on a physical domain and apply `gradient` / `laplacian` / `divergence` / `curl_2d`
2. Verify the finite-difference operators against analytical truth (O(h²) accuracy)
3. Advect a pulse with `semi_lagrangian` and project a field divergence-free with
   `pressure_solve_spectral` (Helmholtz-Hodge)
"""

# %%
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

# %% [markdown]
"""
## 1. Creating fields on physical domains

A `CenteredGrid` wraps values with a physical `Box` domain and a boundary `Extrapolation`, so cell
size `dx` is derived from the domain — operators work in physical units, not pixels. We use the
smooth test field `u(x, y) = sin(x) cos(y)` on a periodic `[0, 2π]²` grid.
"""

# %% [markdown]
"""
## 2. Differential operators (validated against analytical truth)

`gradient` and `laplacian` use second-order central finite differences. Against the closed forms
`∇u = (cos x cos y, -sin x sin y)` and `∇²u = -2 sin x cos y`, the max errors are ~`4e-4` and
`5.5e-4` at 128² — sub-0.1%, consistent with O(h²).
"""

# %% [markdown]
"""
## 3. Vorticity (2D curl)

The field `v = (-sin y, sin x)` has max vorticity `≈ 2.0` and is analytically divergence-free;
`divergence(v)` returns ~`0` numerically, confirming the discrete operators are consistent.
"""

# %% [markdown]
"""
## 4. Semi-Lagrangian advection

A Gaussian pulse advected 10 steps under a uniform velocity is transported with little numerical
diffusion — the peak is conserved (`0.998 → 0.997`).
"""

# %% [markdown]
"""
## 5. Incompressible pressure projection

`pressure_solve_spectral` performs the Helmholtz-Hodge decomposition via an FFT Poisson solve,
removing the divergent component. Here it drives the max divergence from `~2.0` to `~1.7e-3`
(a ~1200x reduction) — the projection step at the heart of incompressible fluid solvers.
"""

# %% [markdown]
"""
## Summary

| Operation | Input | Output | Method |
|-----------|-------|--------|--------|
| `gradient(u)` | Scalar | Vector | Central FD, O(h²) |
| `laplacian(u)` | Scalar | Scalar | Central FD, O(h²) |
| `divergence(v)` | Vector | Scalar | Central FD, O(h²) |
| `curl_2d(v)` | Vector | Scalar | Central FD, O(h²) |
| `semi_lagrangian` | Scalar + Velocity | Scalar | Backward trace + bilinear interp |
| `pressure_solve_spectral` | Vector | Vector + Scalar | FFT Poisson solver |
"""


# %%
def main() -> dict[str, float | int]:
    """Run field operator demos and return finite scalar metrics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Creating Fields on Physical Domains
    n = 128
    box = Box(lower=(0.0, 0.0), upper=(2 * jnp.pi, 2 * jnp.pi))
    coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()

    # Scalar field: u(x, y) = sin(x) * cos(y)
    u_values = jnp.sin(coords[..., 0]) * jnp.cos(coords[..., 1])
    u = CenteredGrid(u_values, box, Extrapolation.PERIODIC)
    print(f"Scalar field: {u}")
    print(f"Resolution: {u.resolution}, Cell size: dx={u.dx}")

    # 2. Differential Operators
    grad_u = gradient(u)
    print(f"Gradient shape: {grad_u.values.shape}")  # (128, 128, 2)

    # Analytical: ∇u = (cos(x)cos(y), -sin(x)sin(y)) — validate BOTH components.
    grad_exact_x = jnp.cos(coords[..., 0]) * jnp.cos(coords[..., 1])
    grad_exact_y = -jnp.sin(coords[..., 0]) * jnp.sin(coords[..., 1])
    grad_exact = jnp.stack([grad_exact_x, grad_exact_y], axis=-1)
    grad_error = jnp.max(jnp.abs(grad_u.values - grad_exact))
    print(f"Gradient max error (both components): {grad_error:.6f}")

    lap_u = laplacian(u)
    # Analytical: ∇²u = -2 sin(x) cos(y)
    lap_exact = -2.0 * jnp.sin(coords[..., 0]) * jnp.cos(coords[..., 1])
    lap_error = jnp.max(jnp.abs(lap_u.values - lap_exact))
    print(f"Laplacian max error: {lap_error:.6f}")

    _fig, axes = plt.subplots(1, 4, figsize=(16, 4))

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

    # 3. Vorticity (2D Curl)
    # v = (-sin(y), sin(x)) — has nonzero vorticity
    vx = -jnp.sin(coords[..., 1])
    vy = jnp.sin(coords[..., 0])
    velocity = CenteredGrid(jnp.stack([vx, vy], axis=-1), box, Extrapolation.PERIODIC)

    vorticity = curl_2d(velocity)
    print(f"Max vorticity: {jnp.max(vorticity.values):.4f}")

    # Divergence of this field
    div_v = divergence(velocity)
    print(f"Max divergence: {jnp.max(jnp.abs(div_v.values)):.6f} (should be ~0)")

    # 4. Semi-Lagrangian Advection
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

    _fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(field.values.T, origin="lower", cmap="hot")
    axes[0].set_title("Initial Gaussian pulse")
    axes[1].imshow(advected.values.T, origin="lower", cmap="hot")
    axes[1].set_title("After 10 semi-Lagrangian steps")
    plt.suptitle("Semi-Lagrangian Advection")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "advection.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'advection.png'}")
    plt.close()

    # 5. Incompressible Pressure Projection
    vx_div = jnp.sin(coords[..., 0])
    vy_div = jnp.sin(coords[..., 1])
    vel_div = CenteredGrid(jnp.stack([vx_div, vy_div], axis=-1), box, Extrapolation.PERIODIC)

    div_before = divergence(vel_div)
    max_divergence_before = jnp.max(jnp.abs(div_before.values))
    print(f"Divergence before projection: max={max_divergence_before:.4f}")

    projected, pressure = pressure_solve_spectral(vel_div)
    div_after = divergence(projected)
    max_divergence_after = jnp.max(jnp.abs(div_after.values))
    print(f"Divergence after projection:  max={max_divergence_after:.4f}")

    _fig, axes = plt.subplots(1, 3, figsize=(12, 4))
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

    return {
        "gradient_max_error": float(grad_error),
        "laplacian_max_error": float(lap_error),
        "max_divergence_before": float(max_divergence_before),
        "max_divergence_after": float(max_divergence_after),
        "resolution": int(n),
    }


# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
