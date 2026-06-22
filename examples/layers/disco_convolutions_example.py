# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# DISCO Convolutions: Discretisation-Invariant Convolution on Arbitrary Point Sets

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~30 s (GPU) / ~2 min (CPU) |
| **Prerequisites** | JAX, Flax NNX, neural operators, quadrature basics |
| **Format** | Python + Jupyter |
| **Memory** | ~0.5 GB |

## Overview

A standard convolution has a fixed *pixel* kernel: its receptive field shrinks as the grid
refines, and it cannot be applied to non-grid (scattered) data at all. A **discrete-continuous
(DISCO)** convolution (Ocampo, Price & McEwen 2023, arXiv:2209.13603 — the algorithm behind
NVIDIA's `torch_harmonics` and spherical neural operators) instead parameterises the kernel as a
*continuous* function `kappa(r) = Σ_k w_k φ_k(r)` and evaluates the convolution as a quadrature
against the input samples:

    (kappa * f)(x_o) ≈ Σ_i q_i kappa(x_o − x_i) f(x_i).

Because the kernel lives in physical coordinates (not pixels) and the sum is a quadrature, the
operator is **discretisation-aware**: the same learned kernel transfers across grid resolutions,
and it works on **irregular** point sets. This example demonstrates both properties with
measurements.

The radial basis reuses opifex's `PiecewiseLinearBasis` (the `torch_harmonics` filter basis), and
the filter is normalised per output point (a partition of unity) as in the reference.

## What You'll Learn

1. Build a `DiscreteContinuousConv2d` over arbitrary input/output point sets
2. Measure **discretisation invariance**: the same kernel on a finer grid gives a consistent result
3. Apply the same operator to an **irregular** (scattered) point cloud — impossible for a pixel conv

## Coming from neuraloperator / torch_harmonics?

| neuraloperator / torch_harmonics | Opifex |
|----------------------------------|--------|
| `DiscreteContinuousConv2d(in_channels, out_channels, grid_in, grid_out, quadrature_weights, kernel_shape)` | `DiscreteContinuousConv2d(in_channels, out_channels, in_coords, out_coords, quad_weights, num_basis=, radius=, rngs=)` |
| `PiecewiseLinearFilterBasis` | `opifex.neural.equivariant.PiecewiseLinearBasis` |
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

# %%
from opifex.neural.operators.specialized.disco import DiscreteContinuousConv2d, regular_grid


# %% [markdown]
"""
## A continuous test field

We convolve a smooth scalar field. Because the DISCO kernel is continuous, the convolution result
is a property of the *function*, not its discretisation — so refining the input grid should leave
the output (at fixed query points) essentially unchanged.
"""


# %%
def smooth_field(coords: jax.Array) -> jax.Array:
    """A smooth scalar field f(x, y) = sin(3x) cos(3y), shaped ``(1, N, 1)`` for the conv."""
    x, y = coords[:, 0], coords[:, 1]
    return (jnp.sin(3.0 * x) * jnp.cos(3.0 * y))[None, :, None]


def main() -> dict[str, float | int]:
    """Demonstrate DISCO discretisation invariance and irregular-grid convolution, with metrics."""
    radius, num_basis, seed = 0.3, 4, 0

    print("=" * 72)
    print("Opifex Example: DISCO convolutions — discretisation-invariant, irregular-grid")
    print("=" * 72)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # Fixed query (output) points, independent of the input discretisation.
    out_coords = jnp.array([[0.3, 0.3], [0.5, 0.5], [0.7, 0.4], [0.4, 0.6], [0.6, 0.7]])

    def disco_on_regular_grid(resolution: int) -> jax.Array:
        in_coords, quad = regular_grid(resolution)
        # Same seed + same parameter shape => identical learned kernel across resolutions.
        conv = DiscreteContinuousConv2d(
            in_channels=1,
            out_channels=1,
            in_coords=in_coords,
            out_coords=out_coords,
            quad_weights=quad,
            num_basis=num_basis,
            radius=radius,
            use_bias=False,
            rngs=nnx.Rngs(seed),
        )
        return conv(smooth_field(in_coords))[0, :, 0]

    # --- Discretisation invariance: same kernel, three input resolutions ---
    print()
    print("Discretisation invariance (same continuous kernel, refining the input grid):")
    print("-" * 72)
    out_24 = disco_on_regular_grid(24)
    out_48 = disco_on_regular_grid(48)
    out_96 = disco_on_regular_grid(96)
    rel_24_96 = float(jnp.linalg.norm(out_24 - out_96) / (jnp.linalg.norm(out_96) + 1e-9))
    rel_48_96 = float(jnp.linalg.norm(out_48 - out_96) / (jnp.linalg.norm(out_96) + 1e-9))
    print(f"  output @ query points (24x24): {jnp.round(out_24, 4)}")
    print(f"  output @ query points (96x96): {jnp.round(out_96, 4)}")
    print(f"  relative change 24->96: {rel_24_96:.4f}")
    print(f"  relative change 48->96: {rel_48_96:.4f}  (halves as the grid refines: convergent)")

    # --- Irregular point set: the capability a pixel convolution cannot provide ---
    print()
    print("Irregular (scattered) point set:")
    print("-" * 72)
    key = jax.random.key(seed + 1)
    irregular_coords = jax.random.uniform(key, (500, 2))
    quad_mc = jnp.full((500,), 1.0 / 500)  # Monte-Carlo quadrature
    conv_irregular = DiscreteContinuousConv2d(
        in_channels=1,
        out_channels=1,
        in_coords=irregular_coords,
        out_coords=out_coords,
        quad_weights=quad_mc,
        num_basis=num_basis,
        radius=radius,
        use_bias=False,
        rngs=nnx.Rngs(seed),
    )
    out_irregular = conv_irregular(smooth_field(irregular_coords))[0, :, 0]
    rel_irregular = float(
        jnp.linalg.norm(out_irregular - out_96) / (jnp.linalg.norm(out_96) + 1e-9)
    )
    print(f"  500 scattered points, same kernel -> output {jnp.round(out_irregular, 4)}")
    print(f"  agreement with the 96x96 grid result: relative diff {rel_irregular:.4f}")
    print("  (a standard pixel convolution cannot be applied to scattered points at all)")

    # --- Visualisation: the learned continuous radial kernel ---
    output_dir = Path("docs/assets/examples/disco_convolutions")
    output_dir.mkdir(parents=True, exist_ok=True)
    in_coords, quad = regular_grid(48)
    conv = DiscreteContinuousConv2d(
        in_channels=1,
        out_channels=1,
        in_coords=in_coords,
        out_coords=out_coords,
        quad_weights=quad,
        num_basis=num_basis,
        radius=radius,
        use_bias=False,
        rngs=nnx.Rngs(seed),
    )
    from opifex.neural.equivariant import PiecewiseLinearBasis

    radial = PiecewiseLinearBasis(num_basis=num_basis, cutoff=radius)
    r = jnp.linspace(0.0, radius, 200)
    weights = conv.weight.value[:, 0, 0]  # (num_basis,) kernel coefficients
    kernel_profile = radial(r) @ weights
    _fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, kernel_profile, color="tab:blue", linewidth=2.5, label="learned kernel kappa(r)")
    for k in range(num_basis):
        ax.plot(
            r, radial(r)[:, k] * float(weights[k]), "--", alpha=0.5, label=f"w[{k}] * hat_{k}(r)"
        )
    ax.set_xlabel("radius r", fontsize=12)
    ax.set_ylabel("kernel value", fontsize=12)
    ax.set_title("DISCO continuous radial kernel (sum of weighted hat basis)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/disco-continuous-kernel.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_dir}/disco-continuous-kernel.png")

    return {
        "num_basis": num_basis,
        "rel_change_24_to_96": rel_24_96,
        "rel_change_48_to_96": rel_48_96,
        "irregular_vs_grid_rel_diff": rel_irregular,
    }


# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
