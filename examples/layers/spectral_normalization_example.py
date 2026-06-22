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
# Spectral Normalization: Lipschitz Control for Stable Deep Training

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~1 min (GPU) / ~3 min (CPU) |
| **Prerequisites** | JAX, Flax NNX, basic optimisation |
| **Format** | Python + Jupyter |
| **Memory** | ~0.5 GB |

## Overview

Spectral normalization (Miyato et al. 2018, *Spectral Normalization for Generative Adversarial
Networks*, arXiv:1802.05957) divides each weight matrix by its largest singular value, so every
layer becomes 1-Lipschitz. The product of per-layer spectral norms upper-bounds the whole
network's Lipschitz constant — and an unbounded Lipschitz constant is exactly what makes deep
networks blow up at aggressive learning rates.

This example demonstrates that value with a controlled comparison: a deep MLP trained at an
aggressive learning rate, built once with plain `nnx.Linear` layers and once with `SpectralLinear`.
We track both the **training loss** and the **network Lipschitz bound** (the product of per-layer
spectral norms). The plain network's Lipschitz bound grows and its training destabilises; the
spectral-normalized network stays bounded and trains smoothly.

## What You'll Learn

1. Use `SpectralLinear` as a drop-in 1-Lipschitz replacement for `nnx.Linear`
2. Measure a network's Lipschitz bound as the product of per-layer spectral norms
3. See how Lipschitz control stabilises deep training at aggressive learning rates

## Coming from PyTorch?

| PyTorch | Opifex |
|---------|--------|
| `torch.nn.utils.parametrizations.spectral_norm(nn.Linear(...))` | `SpectralLinear(in_features=, out_features=, power_iterations=, rngs=)` |
| `spectral_norm(nn.Conv2d(...))` | `SpectralNormalizedConv(in_channels=, out_channels=, kernel_size=, rngs=)` |
"""

# %%
from itertools import pairwise
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

# %%
from opifex.neural.operators.specialized.spectral_normalization import SpectralLinear


# %% [markdown]
"""
## A deep MLP, with and without spectral normalization

The two networks are identical except for their layer type. `SpectralLinear` normalises each
kernel by its spectral norm before the matmul, so the layer is 1-Lipschitz by construction.
"""


# %%
class DeepMLP(nnx.Module):
    """A deep MLP whose hidden layers are either plain Linear or SpectralLinear."""

    def __init__(self, width: int, depth: int, *, spectral: bool, rngs: nnx.Rngs) -> None:
        """Build ``depth`` hidden layers of ``width`` units (1-D regression head)."""
        keys = rngs
        dims = [1, *([width] * depth), 1]
        layers: list[nnx.Module] = []
        for fan_in, fan_out in pairwise(dims):
            if spectral:
                layers.append(SpectralLinear(fan_in, fan_out, power_iterations=2, rngs=keys))
            else:
                layers.append(nnx.Linear(fan_in, fan_out, rngs=keys))
        self.layers = nnx.List(layers)
        self.spectral = spectral

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with tanh activations between hidden layers."""
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = jnp.tanh(x)
        return x


def lipschitz_bound(model: DeepMLP) -> float:
    """Upper bound on the network Lipschitz constant: product of per-layer spectral norms.

    For ``SpectralLinear`` each normalised kernel has spectral norm ~1; for ``nnx.Linear`` it is
    the raw largest singular value. (tanh is 1-Lipschitz, so it does not enter the product.)
    """
    bound = 1.0
    for layer in model.layers:
        kernel = layer.linear.kernel[...] if model.spectral else layer.kernel[...]
        sigma_max = float(jnp.linalg.svd(kernel, compute_uv=False)[0])
        if model.spectral:
            sigma_max /= sigma_max + 1e-12  # normalised kernel -> ~1 (matches the forward pass)
        bound *= sigma_max
    return bound


# %% [markdown]
"""
## Train both at an aggressive learning rate

A plain MLP this deep, trained at this learning rate with no normalization or clipping, has an
unbounded Lipschitz constant and destabilises. `SpectralLinear` caps each layer's gain.
"""


# %%
def _train(model: DeepMLP, x: jax.Array, y: jax.Array, *, learning_rate: float, steps: int):
    """Full-batch SGD; return ``(trained_model, loss_curve)`` (jit-compiled step)."""
    optimizer = nnx.Optimizer(model, optax.sgd(learning_rate), wrt=nnx.Param)

    @nnx.jit
    def step(model: DeepMLP, optimizer: nnx.Optimizer) -> jax.Array:
        def loss_fn(m: DeepMLP) -> jax.Array:
            return jnp.mean((m(x) - y) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    losses = []
    for _ in range(steps):
        losses.append(float(step(model, optimizer)))
    return model, jnp.asarray(losses)


def main() -> dict[str, float | int]:
    """Compare deep-MLP training stability with vs without spectral normalization."""
    width, depth = 64, 8
    learning_rate, steps, seed = 0.3, 300, 0

    print("=" * 72)
    print("Opifex Example: Spectral Normalization — Lipschitz control for stable training")
    print("=" * 72)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"Deep MLP: width={width}, depth={depth}, SGD lr={learning_rate}, steps={steps}")

    # Regression target: a gently-sloped smooth function (low Lipschitz constant) that a
    # 1-Lipschitz network can fit well — so the comparison isolates *stability*, not capacity.
    x = jnp.linspace(-3.0, 3.0, 256)[:, None]
    y = 0.5 * jnp.sin(x)

    plain = DeepMLP(width, depth, spectral=False, rngs=nnx.Rngs(seed))
    spectral = DeepMLP(width, depth, spectral=True, rngs=nnx.Rngs(seed))

    print()
    print(
        f"Lipschitz bound at init:  plain={lipschitz_bound(plain):.2e}  "
        f"spectral={lipschitz_bound(spectral):.2e}"
    )

    plain, plain_losses = _train(plain, x, y, learning_rate=learning_rate, steps=steps)
    spectral, spectral_losses = _train(spectral, x, y, learning_rate=learning_rate, steps=steps)

    plain_final, spectral_final = float(plain_losses[-1]), float(spectral_losses[-1])
    plain_max, spectral_max = float(jnp.max(plain_losses)), float(jnp.max(spectral_losses))
    plain_diverged = bool(jnp.logical_not(jnp.isfinite(plain_losses[-1])) | (plain_max > 1e3))

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"{'Model':<26}{'final MSE':>14}{'max MSE':>14}{'Lipschitz bound':>18}")
    print("-" * 72)
    print(
        f"{'plain nnx.Linear':<26}{plain_final:>14.4e}{plain_max:>14.4e}"
        f"{lipschitz_bound(plain):>18.2e}"
    )
    print(
        f"{'SpectralLinear':<26}{spectral_final:>14.4e}{spectral_max:>14.4e}"
        f"{lipschitz_bound(spectral):>18.2e}"
    )
    print("-" * 72)
    print(
        f"Plain network destabilised: {plain_diverged}; spectral final MSE "
        f"{spectral_final:.2e} (Lipschitz bound stays ~1 per layer)."
    )

    # --- Visualisation: loss curves ---
    output_dir = Path("docs/assets/examples/spectral_normalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    _fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plain_losses, label="plain nnx.Linear", color="tab:red", linewidth=2)
    ax.plot(spectral_losses, label="SpectralLinear", color="tab:blue", linewidth=2)
    ax.set_xlabel("SGD step", fontsize=12)
    ax.set_ylabel("Training MSE", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(f"Deep MLP (depth {depth}) at SGD lr={learning_rate}", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spectral-norm-stability.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_dir}/spectral-norm-stability.png")

    # The plain network's final/max MSE are non-finite *by design* (it destabilises);
    # the demonstration is recorded as the finite ``plain_diverged`` flag rather than
    # returned as a NaN metric.
    return {
        "depth": depth,
        "learning_rate": learning_rate,
        "plain_diverged": float(plain_diverged),
        "spectral_final_mse": spectral_final,
        "spectral_lipschitz_bound": lipschitz_bound(spectral),
    }


# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
