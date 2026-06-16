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
# UNO on Darcy Flow

| Property      | Value                                          |
|---------------|------------------------------------------------|
| Level         | Intermediate                                   |
| Runtime       | ~5 min (CPU) / ~1 min (GPU)                    |
| Memory        | ~2 GB                                          |
| Prerequisites | JAX, Flax NNX, Neural Operators basics         |

## Overview

Train a U-Net Neural Operator (UNO) on the Darcy flow equation. UNO
combines U-Net's multi-scale encoder-decoder architecture with Fourier
spectral convolutions, enabling operator learning with **zero-shot
super-resolution** capabilities.

This example uses the standard operator-learning recipe — grid positional
embedding, Gaussian input/output normalization, and the relative-L2 loss —
to reach a low relative L2 error on Darcy flow.

This example demonstrates:

- **create_uno** factory for quick model construction
- **GridEmbedding2D** positional encoding fed as extra input channels
- **Gaussian normalization** of inputs and outputs
- **relative-L2 loss** via `LossConfig`, the standard operator-learning objective
- **datarax DataLoader** for efficient streaming data
- **Trainer.fit()** for end-to-end training with validation
- **Zero-shot super-resolution** inference at higher resolutions

## Learning Goals

1. Create a UNO with `create_uno` factory
2. Load Darcy flow data with `create_darcy_loader` (datarax-based)
3. Apply grid embedding, normalization, and the relative-L2 loss
4. Train with Opifex's `Trainer.fit()` API
5. Evaluate predictions and demonstrate zero-shot super-resolution
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import time
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from flax import nnx


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.core.evaluation import predict_in_batches
from opifex.core.metrics import per_sample_relative_l2
from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.config import LossConfig
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.specialized import create_uno


# %% [markdown]
"""
## Configuration

We follow the standard operator-learning recipe: ~1000 training samples,
Gaussian normalization, the relative-L2 loss, and enough epochs for the
spectral weights to converge.
"""

# %% [markdown]
"""
## Data Loading with datarax

Opifex provides `create_darcy_loader` which generates Darcy flow equation data
(permeability-to-pressure mapping) and wraps it in a datarax DataLoader
for efficient streaming and batching.
"""

# %% [markdown]
"""
## Normalization

Neural operators train best on standardized fields. We fit Gaussian statistics
on the training set, normalize all splits, and un-normalize predictions before
computing physical-space errors.
"""

# %% [markdown]
"""
## Model Creation

The `create_uno` factory creates a U-Net Neural Operator with spectral
convolutions. We wrap it with `GridEmbedding2D`, which appends normalized
``(x, y)`` coordinate channels to the permeability input — the standard
positional encoding that lets spectral operators resolve boundary-value
problems. The grid embedding works directly on UNO's channels-last input.
"""


# %%
class UNOWithGrid(nnx.Module):
    """UNO with a 2D grid positional embedding on the (channels-last) input."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
        modes: int,
        n_layers: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the grid embedding and the underlying UNO.

        Args:
            input_channels: Number of physical input channels (before the grid).
            output_channels: Number of output channels.
            hidden_channels: Base number of UNO hidden channels.
            modes: Number of Fourier modes for the spectral layers.
            n_layers: Number of U-Net encoder/decoder stages.
            rngs: Random number generators.
        """
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=input_channels,
            grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        )
        self.uno = create_uno(
            input_channels=self.grid_embedding.out_channels,
            output_channels=output_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            n_layers=n_layers,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        """Append grid coordinates, then apply the UNO.

        Args:
            x: Input of shape (batch, height, width, input_channels).
            deterministic: Whether to run the UNO in deterministic mode.

        Returns:
            Output of shape (batch, height, width, output_channels).
        """
        x_embedded = self.grid_embedding(x)
        return self.uno(x_embedded, deterministic=deterministic)


# %% [markdown]
"""
## Training with Opifex Trainer

We use Opifex's `Trainer` with the relative-L2 loss (`loss_type="relative_l2"`),
the standard operator-learning objective. The `Trainer.fit()` method handles
batched training with JIT compilation, validation, and progress logging.
"""

# %% [markdown]
"""
## Evaluation

Predictions are un-normalized back to physical pressure before measuring the
relative L2 error. The test set is run through the model in batches to bound
memory use at higher resolutions.
"""


# %% [markdown]
"""
## Run the example

All run logic — configuration, data loading, normalization, model creation,
training, evaluation, zero-shot super-resolution, and visualization — lives in
`main()`. It returns a small dict of finite scalar metrics and saves the
prediction/super-resolution plots to `docs/assets/examples/uno_darcy/`.
"""


# %%
def main() -> dict[str, float | int]:
    """Train a UNO on Darcy flow and report relative-L2 error metrics.

    Returns:
        Finite scalar metrics: parameter count, final train loss, test MSE,
        mean test relative L2 error, and the zero-shot super-resolution error.
    """
    # --- Configuration ---
    resolution = 32
    n_train = 1000
    n_test = 100
    batch_size = 32
    num_epochs = 120
    learning_rate = 1e-3
    hidden_channels = 32
    modes = 12
    n_layers = 3
    seed = 42

    output_dir = Path("docs/assets/examples/uno_darcy")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Opifex Example: UNO on Darcy Flow")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Training samples: {n_train}, Test samples: {n_test}")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"UNO config: hidden={hidden_channels}, modes={modes}, layers={n_layers}")

    # --- Data loading via datarax ---
    print()
    print("Loading Darcy flow data via datarax...")
    n_samples = n_train + n_test
    loaders = create_darcy_loader(
        n_samples=n_samples,
        batch_size=batch_size,
        resolution=resolution,
        val_fraction=n_test / n_samples,
        seed=seed,
    )

    # datarax yields channels-first {"input": (b, 1, H, W), "output": (b, 1, H, W)};
    # UNO (and its grid embedding, eval, and visualization) work channels-last, so
    # we move the channel axis to the end once here, at the data boundary.
    def _collect(pipeline) -> tuple[np.ndarray, np.ndarray]:
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.moveaxis(np.asarray(batch["input"]), 1, -1))
            outputs.append(np.moveaxis(np.asarray(batch["output"]), 1, -1))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    x_train, y_train = _collect(loaders.train)
    x_test, y_test = _collect(loaders.val)

    print(f"Training data: X={x_train.shape}, Y={y_train.shape}")
    print(f"Test data:     X={x_test.shape}, Y={y_test.shape}")

    # --- Normalization (fit on train, applied to all splits) ---
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    x_train_n = (x_train - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    x_test_n = (x_test - x_mean) / x_std

    print(f"Input mean/std:  {x_mean:.4f} / {x_std:.4f}")
    print(f"Output mean/std: {y_mean:.6f} / {y_std:.6f}")

    # --- Model creation ---
    print()
    print("Creating UNO model with grid embedding...")
    in_channels = x_train.shape[-1]
    out_channels = y_train.shape[-1]

    model = UNOWithGrid(
        input_channels=in_channels,
        output_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        n_layers=n_layers,
        rngs=nnx.Rngs(seed),
    )

    params = nnx.state(model, nnx.Param)
    param_count = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print(f"Model: UNO + GridEmbedding2D (hidden={hidden_channels}, modes={modes}, layers={n_layers})")
    print(f"Input channels: {in_channels} (+ 2 grid coords = {in_channels + 2} after embedding)")
    print(f"Output channels: {out_channels}")
    print(f"Total parameters: {param_count:,}")

    # --- Training ---
    print()
    print("Setting up Trainer...")
    config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_frequency=5,
        verbose=True,
        loss_config=LossConfig(loss_type="relative_l2"),
    )
    trainer = Trainer(
        model=model,
        config=config,
        rngs=nnx.Rngs(seed),
    )

    print(f"Optimizer: Adam (lr={learning_rate}), loss: relative L2")
    print()
    print("Starting training...")
    start_time = time.time()

    trained_model, metrics = trainer.fit(
        train_data=(jnp.array(x_train_n), jnp.array(y_train_n)),
        val_data=(jnp.array(x_test_n), jnp.array((y_test - y_mean) / y_std)),
    )

    training_time = time.time() - start_time
    final_train_loss = float(metrics["final_train_loss"])
    final_val_loss = float(metrics["final_val_loss"])
    print(f"Training completed in {training_time:.1f}s")
    print(f"Final train loss: {final_train_loss}")
    print(f"Final val loss:   {final_val_loss}")

    # --- Evaluation (un-normalized to physical pressure units) ---
    print()
    print("Evaluating on test set...")
    x_test_jnp = jnp.array(x_test_n)
    y_test_jnp = jnp.array(y_test)

    predictions = (
        predict_in_batches(lambda b: trained_model(b, deterministic=True), x_test_jnp) * y_std
        + y_mean
    )

    test_mse = float(jnp.mean((predictions - y_test_jnp) ** 2))

    per_sample_rel_l2 = per_sample_relative_l2(predictions, y_test_jnp)
    mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))

    print(f"Test MSE:         {test_mse:.6e}")
    print(f"Test Relative L2: {mean_rel_l2:.6f}")
    print(f"Min Relative L2:  {float(jnp.min(per_sample_rel_l2)):.6f}")
    print(f"Max Relative L2:  {float(jnp.max(per_sample_rel_l2)):.6f}")

    # --- Zero-shot super-resolution ---
    print()
    target_resolution = resolution * 2
    print(f"Testing zero-shot super-resolution: {resolution} -> {target_resolution}")

    x_sample = x_test_jnp[0:1]
    x_high_res = jax.image.resize(
        x_sample,
        (1, target_resolution, target_resolution, in_channels),
        method="bilinear",
    )
    y_pred_high = trained_model(x_high_res, deterministic=True) * y_std + y_mean
    y_true_high = jax.image.resize(
        y_test_jnp[0:1],
        (1, target_resolution, target_resolution, out_channels),
        method="bilinear",
    )
    sr_error = float(
        jnp.sqrt(jnp.sum((y_pred_high - y_true_high) ** 2)) / jnp.sqrt(jnp.sum(y_true_high**2))
    )
    print(f"Super-resolution L2 error: {sr_error:.6f}")

    # --- Visualization: sample predictions ---
    print()
    print("Generating visualizations...")
    n_vis = 3
    indices = np.linspace(0, len(x_test) - 1, n_vis, dtype=int)

    fig, axes = plt.subplots(n_vis, 4, figsize=(16, 4 * n_vis))
    fig.suptitle("UNO Darcy Flow Predictions (Opifex)", fontsize=14, fontweight="bold")

    for row, idx in enumerate(indices):
        x_field = x_test[idx, :, :, 0]
        y_true = y_test[idx, :, :, 0]
        y_pred = np.array(predictions[idx, :, :, 0])
        error = np.abs(y_pred - y_true)

        im0 = axes[row, 0].imshow(x_field, cmap="viridis")
        axes[row, 0].set_title(f"Input {row + 1}: Permeability")
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.8)

        im1 = axes[row, 1].imshow(y_true, cmap="RdBu_r")
        axes[row, 1].set_title(f"Ground Truth {row + 1}")
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.8)

        im2 = axes[row, 2].imshow(y_pred, cmap="RdBu_r")
        axes[row, 2].set_title(f"UNO Prediction {row + 1}")
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.8)

        im3 = axes[row, 3].imshow(error, cmap="Reds")
        axes[row, 3].set_title(f"Absolute Error {row + 1}")
        axes[row, 3].axis("off")
        plt.colorbar(im3, ax=axes[row, 3], shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "uno_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions saved to {output_dir / 'uno_predictions.png'}")

    # --- Visualization: super-resolution ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"UNO Zero-Shot Super-Resolution ({resolution} -> {target_resolution})",
        fontsize=14,
        fontweight="bold",
    )

    im0 = axes[0].imshow(np.array(x_high_res[0, :, :, 0]), cmap="viridis", aspect="equal")
    axes[0].set_title("Input (High Res)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(np.array(y_pred_high[0, :, :, 0]), cmap="RdBu_r", aspect="equal")
    axes[1].set_title("UNO Prediction")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(np.array(y_true_high[0, :, :, 0]), cmap="RdBu_r", aspect="equal")
    axes[2].set_title("Ground Truth (Upsampled)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "uno_superresolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Super-resolution saved to {output_dir / 'uno_superresolution.png'}")

    print()
    print("=" * 70)
    print(f"UNO Darcy Flow example completed in {training_time:.1f}s")
    print(f"Test MSE: {test_mse:.6e}, Relative L2: {mean_rel_l2:.6f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return {
        "param_count": param_count,
        "final_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "test_mse": test_mse,
        "l2_relative_error": mean_rel_l2,
        "superresolution_l2_error": sr_error,
    }


# %% [markdown]
"""
## Results Summary

After running this example you should observe:
- A decreasing relative-L2 training loss over epochs on Darcy flow data
- Accurate predictions mapping permeability to pressure fields
- Zero-shot super-resolution capability at higher resolutions

## Next Steps

- Increase `hidden_channels` and `modes` for higher capacity
- Experiment with more training samples and epochs
- Compare UNO vs FNO on this problem (see `fno_darcy.py`)
- Try `UNeuralOperator` directly with `use_spectral=True` for spectral convolutions
- Explore the SFNO architecture for climate/spherical data
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
