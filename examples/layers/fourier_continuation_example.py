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
# Fourier Continuation: Accurate Spectral Derivatives for Non-Periodic Functions

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~30 s (CPU/GPU) |
| **Prerequisites** | JAX, FFTs, basic spectral methods |
| **Format** | Python + Jupyter |
| **Memory** | ~0.3 GB |

## Overview

Spectral (FFT-based) differentiation is exact for periodic functions, but a *non-periodic*
function has an implicit jump at the wrap-around point — the FFT sees `f(L) != f(0)` as a
discontinuity, which produces Gibbs ringing that pollutes the derivative across the whole domain.
This is precisely why FNO-style spectral models pad/continue non-periodic inputs.

`FourierContinuationExtender` extends a signal beyond its boundaries with a smooth continuation,
so the FFT sees a (near-)periodic signal. This example quantifies the effect on a textbook case:
the spectral derivative of `exp(x)` on `[0, 1]` (strongly non-periodic — `exp(1) ≈ 2.72` jumps to
`exp(0) = 1`), measured against the exact derivative `exp(x)`.

## What You'll Learn

1. Why naive FFT differentiation fails on non-periodic functions (Gibbs ringing)
2. Use `FourierContinuationExtender` to extend a signal before spectral operations
3. Quantify the accuracy gain (relative L2 error vs the analytical derivative)

## Coming from spectral-methods tooling?

Fourier continuation (a.k.a. FC-Gram; Bruno & Lyon 2009) is the standard fix for applying
periodic spectral operators to non-periodic data. Here it is a composable `nnx.Module` you can
drop in front of any FFT-based layer.
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# %%
from opifex.neural.operators.specialized.fourier_continuation import FourierContinuationExtender


# %% [markdown]
"""
## Spectral differentiation

The spectral derivative multiplies each Fourier mode by `i k`. On a grid of `n` points spanning a
domain of length `domain_length`, the wavenumbers are `k = 2π * fftfreq(n, d=dx)`.
"""


# %%
def spectral_derivative(values: jax.Array, domain_length: float) -> jax.Array:
    """FFT-based first derivative of a 1-D signal sampled over ``domain_length``."""
    n = values.shape[0]
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=domain_length / n)
    return jnp.real(jnp.fft.ifft(1j * k * jnp.fft.fft(values)))


def continued_spectral_derivative(
    values: jax.Array, dx: float, extender: FourierContinuationExtender
) -> jax.Array:
    """Spectral derivative computed on a Fourier-continued signal, cropped back to the interior.

    The signal is smoothly extended by ``extension_length`` points on each side so the FFT sees a
    near-periodic signal; the derivative is taken on the extended domain and then restricted to the
    original samples.
    """
    extended = extender.extend_1d(values)
    ext = extender.extension_length
    deriv_extended = spectral_derivative(extended, dx * extended.shape[0])
    return deriv_extended[ext : ext + values.shape[0]]


# %% [markdown]
"""
## Run the comparison

`main()` differentiates `exp(x)` on `[0, 1]` naively and with Fourier continuation, and reports
the relative L2 error of each against the exact derivative.
"""


# %%
def main() -> dict[str, float | int]:
    """Compare naive vs Fourier-continued spectral differentiation of a non-periodic function."""
    n = 256
    domain_length = 1.0
    dx = domain_length / n

    print("=" * 72)
    print("Opifex Example: Fourier Continuation — spectral derivative of a non-periodic function")
    print("=" * 72)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # Non-periodic test function on [0, 1]: f = exp(x), f' = exp(x) (exact).
    x = jnp.linspace(0.0, domain_length, n, endpoint=False)
    f = jnp.exp(x)
    exact = jnp.exp(x)

    naive = spectral_derivative(f, domain_length)
    extender = FourierContinuationExtender(extension_type="smooth", extension_length=64)
    continued = continued_spectral_derivative(f, dx, extender)

    def rel_l2(estimate: jax.Array) -> float:
        return float(jnp.linalg.norm(estimate - exact) / jnp.linalg.norm(exact))

    naive_err, continued_err = rel_l2(naive), rel_l2(continued)

    print()
    print(f"Test function: f(x) = exp(x) on [0, 1]  (periodic-extension jump = {f[0] - f[-1]:.2f})")
    print("=" * 72)
    print("RESULTS — relative L2 error of the spectral derivative vs exact exp(x)")
    print("=" * 72)
    print(f"  Naive FFT derivative:            {naive_err:.4f}")
    print(f"  Fourier-continued derivative:    {continued_err:.4f}")
    print(f"  Continuation reduces the error by {naive_err / continued_err:.0f}x.")

    # --- Visualisation: naive ringing vs continued vs exact ---
    output_dir = Path("docs/assets/examples/fourier_continuation")
    output_dir.mkdir(parents=True, exist_ok=True)
    _fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, exact, "k-", linewidth=2.5, label="exact  exp(x)")
    ax.plot(x, naive, color="tab:red", linewidth=1.5, label=f"naive FFT (rel L2 {naive_err:.2f})")
    ax.plot(
        x,
        continued,
        color="tab:blue",
        linewidth=1.5,
        label=f"Fourier-continued (rel L2 {continued_err:.3f})",
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("df/dx", fontsize=12)
    ax.set_title("Spectral derivative of a non-periodic function", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fourier-continuation-derivative.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_dir}/fourier-continuation-derivative.png")

    return {
        "grid_points": n,
        "naive_relative_l2": naive_err,
        "continued_relative_l2": continued_err,
        "error_reduction_factor": naive_err / continued_err,
    }


# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
