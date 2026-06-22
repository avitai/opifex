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
# Grid Embeddings: Why Positional Coordinates Matter for Neural Operators

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~3 min (GPU) / ~12 min (CPU) |
| **Prerequisites** | JAX, Flax NNX, [FNO on Darcy](../neural-operators/fno-darcy.md) |
| **Format** | Python + Jupyter |
| **Memory** | ~1 GB |

## Overview

A Fourier Neural Operator sees only channel values at grid points — it has no
intrinsic notion of *where* each point sits in the domain. `GridEmbedding2D` injects
the spatial coordinates as extra input channels, giving the operator positional
awareness. This is standard practice in neural-operator libraries (it is on by default
in `neuraloperator`), but *how much does it actually help?*

This example answers that with a controlled **ablation**: we train two otherwise-identical
FNOs on Darcy flow — one with `GridEmbedding2D`, one without — and measure the difference
in test accuracy (relative L2). Everything else (modes, width, depth, optimiser, data,
seed) is held fixed, so the gap is attributable to the positional encoding alone.

## What You'll Learn

1. Compose `GridEmbedding2D` with a `FourierNeuralOperator`
2. Quantify grid embedding's effect on test error (relative L2) with a clean ablation
3. Visualise the coordinate channels the embedding appends to the input
4. Understand why a boundary-value problem (fixed zero boundary) rewards positional awareness

## Coming from neuraloperator (PyTorch)?

| neuraloperator | Opifex |
|----------------|--------|
| `GridEmbeddingND(in_channels, dim, grid_boundaries)` | `GridEmbedding2D(in_channels=, grid_boundaries=)` / `GridEmbeddingND(...)` |
| `FNO(..., positional_embedding='grid')` (default on) | compose `GridEmbedding2D` then `FourierNeuralOperator` explicitly |
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

# %%
from opifex.core.evaluation import predict_in_batches
from opifex.core.metrics import relative_l2_error
from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.config import LossConfig, OptimizationConfig
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.base import FourierNeuralOperator


# %% [markdown]
"""
## The two models

Both models are the same FNO; the only difference is whether a `GridEmbedding2D`
prepends the two normalised spatial coordinates to the single permeability channel
(1 -> 3 input channels). Everything else — modes, width, depth, domain padding — is
identical, so any accuracy gap is attributable to the positional encoding alone.
"""


# %%
class FNO(nnx.Module):
    """Plain FNO (no positional encoding): maps the 1-channel permeability directly."""

    def __init__(
        self, modes: int, hidden_channels: int, num_layers: int, *, rngs: nnx.Rngs
    ) -> None:
        """Build a single-input-channel FNO with domain padding."""
        super().__init__()
        self.fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            domain_padding=0.25,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass for ``(batch, 1, H, W)`` permeability fields."""
        return self.fno(x)


class FNOWithGridEmbedding(nnx.Module):
    """FNO with `GridEmbedding2D`: appends the (x, y) coordinates as input channels."""

    def __init__(
        self, modes: int, hidden_channels: int, num_layers: int, *, rngs: nnx.Rngs
    ) -> None:
        """Build a grid-embedded FNO (1 physical channel + 2 coordinate channels)."""
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=1, grid_boundaries=[[0.0, 1.0], [0.0, 1.0]]
        )
        self.fno = FourierNeuralOperator(
            in_channels=self.grid_embedding.out_channels,  # 3
            out_channels=1,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            domain_padding=0.25,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Embed grid coordinates (channels-last) then apply the FNO (channels-first)."""
        x_hwc = jnp.moveaxis(x, 1, -1)
        x_embedded = self.grid_embedding(x_hwc)
        return self.fno(jnp.moveaxis(x_embedded, -1, 1))


# %% [markdown]
"""
## Run the ablation

`main()` generates Darcy data, trains both models identically, evaluates each on the
held-out test set, and returns the comparison metrics.
"""


# %%
def _collect(pipeline) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a datarax pipeline into ``(inputs, outputs)`` arrays (channels-first)."""
    inputs, outputs = [], []
    for batch in pipeline:
        inputs.append(np.asarray(batch["input"]))
        outputs.append(np.asarray(batch["output"]))
    return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)


def _train_one(
    model: nnx.Module, x_train_n: np.ndarray, y_train_n: np.ndarray, *, num_epochs: int, seed: int
) -> nnx.Module:
    """Train a model with the standard relative-L2 / AdamW operator-learning recipe."""
    config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=32,
        validation_frequency=num_epochs,  # no separate val split here
        verbose=False,
        loss_config=LossConfig(loss_type="relative_l2"),
        optimization_config=OptimizationConfig(
            optimizer="adamw", learning_rate=5e-3, weight_decay=1e-4
        ),
    )
    trainer = Trainer(model=model, config=config, rngs=nnx.Rngs(seed))
    trained, _ = trainer.fit(train_data=(x_train_n, y_train_n))
    return trained


def _eval_relative_l2(
    model: nnx.Module, x_n: np.ndarray, y: np.ndarray, *, y_mean: float, y_std: float
) -> float:
    """Mean relative-L2 error of un-normalised predictions against physical targets."""
    pred_n = predict_in_batches(model, jnp.asarray(x_n), batch_size=32)
    pred = np.asarray(pred_n) * y_std + y_mean
    return float(relative_l2_error(jnp.asarray(pred), jnp.asarray(y)))


def main() -> dict[str, float | int]:
    """Train FNO with vs without GridEmbedding2D and report the accuracy comparison."""
    resolution = 32
    n_train, n_test = 1000, 100
    num_epochs, seed = 120, 42

    print("=" * 72)
    print("Opifex Example: Grid Embeddings — FNO ablation on Darcy flow")
    print("=" * 72)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    # --- Data ---
    print()
    print(f"Generating Darcy data at {resolution}x{resolution}...")
    loaders = create_darcy_loader(
        n_samples=n_train + n_test,
        batch_size=32,
        resolution=resolution,
        val_fraction=n_test / (n_train + n_test),
        seed=seed,
    )
    x_train, y_train = _collect(loaders.train)
    x_test, y_test = _collect(loaders.val)

    # Normalisation fit on train, applied to both splits.
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()
    x_train_n = (x_train - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    x_test_n = (x_test - x_mean) / x_std

    fno_kwargs = {"modes": 12, "hidden_channels": 32, "num_layers": 4}

    # --- Train both models identically (same data, seed, and FNO hyperparameters) ---
    print()
    print(f"Training FNO WITHOUT grid embedding ({num_epochs} epochs)...")
    plain = _train_one(
        FNO(**fno_kwargs, rngs=nnx.Rngs(seed)),
        x_train_n,
        y_train_n,
        num_epochs=num_epochs,
        seed=seed,
    )
    print(f"Training FNO WITH GridEmbedding2D ({num_epochs} epochs)...")
    embedded = _train_one(
        FNOWithGridEmbedding(**fno_kwargs, rngs=nnx.Rngs(seed)),
        x_train_n,
        y_train_n,
        num_epochs=num_epochs,
        seed=seed,
    )

    plain_l2 = _eval_relative_l2(plain, x_test_n, y_test, y_mean=y_mean, y_std=y_std)
    embedded_l2 = _eval_relative_l2(embedded, x_test_n, y_test, y_mean=y_mean, y_std=y_std)

    print()
    print("=" * 72)
    print("RESULTS — test relative L2 error (lower is better)")
    print("=" * 72)
    print(f"  FNO (no grid embedding):  {plain_l2:.4f}")
    print(f"  FNO + GridEmbedding2D:    {embedded_l2:.4f}")
    print(
        f"  Grid embedding reduces the relative-L2 error by "
        f"{(1 - embedded_l2 / plain_l2) * 100:.0f}% on this boundary-value problem."
    )

    # --- Visualisation: the two grid-coordinate channels GridEmbedding2D adds ---
    output_dir = Path("docs/assets/examples/grid_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding = GridEmbedding2D(in_channels=1, grid_boundaries=[[0.0, 1.0], [0.0, 1.0]])
    sample = jnp.asarray(x_test[:1]).transpose(0, 2, 3, 1)  # (1, H, W, 1)
    embedded_sample = np.asarray(embedding(sample))[0]  # (H, W, 3)
    _fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["Permeability (input)", "Grid coord x", "Grid coord y"]
    for ax, title, channel in zip(axes, titles, range(3), strict=True):
        im = ax.imshow(embedded_sample[..., channel], cmap="viridis")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        _fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/grid-embedding-channels.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_dir}/grid-embedding-channels.png")

    return {
        "resolution": resolution,
        "plain_relative_l2": plain_l2,
        "embedded_relative_l2": embedded_l2,
        "error_reduction_percent": (1 - embedded_l2 / plain_l2) * 100,
    }


# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
