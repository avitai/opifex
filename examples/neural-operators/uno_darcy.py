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
- **Grain DataLoader** for efficient streaming data
- **Trainer.fit()** for end-to-end training with validation
- **Zero-shot super-resolution** inference at higher resolutions

## Learning Goals

1. Create a UNO with `create_uno` factory
2. Load Darcy flow data with `create_darcy_loader` (Grain-based)
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

from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.config import LossConfig
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.specialized import create_uno


print("=" * 70)
print("Opifex Example: UNO on Darcy Flow")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Configuration

We follow the standard operator-learning recipe: ~1000 training samples,
Gaussian normalization, the relative-L2 loss, and enough epochs for the
spectral weights to converge.
"""

# %%
RESOLUTION = 32
N_TRAIN = 1000
N_TEST = 100
BATCH_SIZE = 32
NUM_EPOCHS = 120
LEARNING_RATE = 1e-3
HIDDEN_CHANNELS = 32
MODES = 12
N_LAYERS = 3
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/uno_darcy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
print(f"UNO config: hidden={HIDDEN_CHANNELS}, modes={MODES}, layers={N_LAYERS}")

# %% [markdown]
"""
## Data Loading with Grain

Opifex provides `create_darcy_loader` which generates Darcy flow equation data
(permeability-to-pressure mapping) and wraps it in a Google Grain DataLoader
for efficient streaming and batching.
"""

# %%
print()
print("Loading Darcy flow data via Grain...")
train_loader = create_darcy_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=True,
    seed=SEED,
    worker_count=0,
)

test_loader = create_darcy_loader(
    n_samples=N_TEST,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=False,
    seed=SEED + 1000,
    worker_count=0,
)

# Collect data from loaders into arrays for Trainer.fit()
X_train_list, Y_train_list = [], []
for batch in train_loader:
    X_train_list.append(batch["input"])
    Y_train_list.append(batch["output"])

X_train = np.concatenate(X_train_list, axis=0)
Y_train = np.concatenate(Y_train_list, axis=0)

X_test_list, Y_test_list = [], []
for batch in test_loader:
    X_test_list.append(batch["input"])
    Y_test_list.append(batch["output"])

X_test = np.concatenate(X_test_list, axis=0)
Y_test = np.concatenate(Y_test_list, axis=0)

# UNO expects channels-last: (batch, height, width, channels)
if X_train.ndim == 3:
    X_train = X_train[..., np.newaxis]
    Y_train = Y_train[..., np.newaxis]
if X_test.ndim == 3:
    X_test = X_test[..., np.newaxis]
    Y_test = Y_test[..., np.newaxis]

print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test data:     X={X_test.shape}, Y={Y_test.shape}")

# %% [markdown]
"""
## Normalization

Neural operators train best on standardized fields. We fit Gaussian statistics
on the training set, normalize all splits, and un-normalize predictions before
computing physical-space errors.
"""

# %%
x_mean, x_std = X_train.mean(), X_train.std()
y_mean, y_std = Y_train.mean(), Y_train.std()

X_train_n = (X_train - x_mean) / x_std
Y_train_n = (Y_train - y_mean) / y_std
X_test_n = (X_test - x_mean) / x_std
Y_test_n = (Y_test - y_mean) / y_std

print(f"Input mean/std:  {x_mean:.4f} / {x_std:.4f}")
print(f"Output mean/std: {y_mean:.6f} / {y_std:.6f}")

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


print()
print("Creating UNO model with grid embedding...")
in_channels = X_train.shape[-1]
out_channels = Y_train.shape[-1]

model = UNOWithGrid(
    input_channels=in_channels,
    output_channels=out_channels,
    hidden_channels=HIDDEN_CHANNELS,
    modes=MODES,
    n_layers=N_LAYERS,
    rngs=nnx.Rngs(SEED),
)

params = nnx.state(model, nnx.Param)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Model: UNO + GridEmbedding2D (hidden={HIDDEN_CHANNELS}, modes={MODES}, layers={N_LAYERS})")
print(f"Input channels: {in_channels} (+ 2 grid coords = {in_channels + 2} after embedding)")
print(f"Output channels: {out_channels}")
print(f"Total parameters: {param_count:,}")

# %% [markdown]
"""
## Training with Opifex Trainer

We use Opifex's `Trainer` with the relative-L2 loss (`loss_type="relative_l2"`),
the standard operator-learning objective. The `Trainer.fit()` method handles
batched training with JIT compilation, validation, and progress logging.
"""

# %%
print()
print("Setting up Trainer...")
config = TrainingConfig(
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    validation_frequency=5,
    verbose=True,
    loss_config=LossConfig(loss_type="relative_l2"),
)

trainer = Trainer(
    model=model,
    config=config,
    rngs=nnx.Rngs(SEED),
)

print(f"Optimizer: Adam (lr={LEARNING_RATE}), loss: relative L2")
print()
print("Starting training...")
start_time = time.time()

trained_model, metrics = trainer.fit(
    train_data=(jnp.array(X_train_n), jnp.array(Y_train_n)),
    val_data=(jnp.array(X_test_n), jnp.array(Y_test_n)),
)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.1f}s")
print(f"Final train loss: {metrics.get('final_train_loss', 'N/A')}")
print(f"Final val loss:   {metrics.get('final_val_loss', 'N/A')}")

# %% [markdown]
"""
## Evaluation

Predictions are un-normalized back to physical pressure before measuring the
relative L2 error. The test set is run through the model in batches to bound
memory use at higher resolutions.
"""

# %%
print()
print("Evaluating on test set...")
X_test_jnp = jnp.array(X_test_n)
Y_test_jnp = jnp.array(Y_test)


def predict_in_batches(
    forward: nnx.Module,
    inputs: jax.Array,
    batch_size: int = 128,
) -> jax.Array:
    """Run the model over the inputs in batches to bound memory use."""
    outputs = [
        forward(inputs[i : i + batch_size], deterministic=True)
        for i in range(0, inputs.shape[0], batch_size)
    ]
    return jnp.concatenate(outputs, axis=0)


predictions = predict_in_batches(trained_model, X_test_jnp) * y_std + y_mean

test_mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))

# Relative L2 error per sample
pred_diff = (predictions - Y_test_jnp).reshape(predictions.shape[0], -1)
Y_flat = Y_test_jnp.reshape(Y_test_jnp.shape[0], -1)
per_sample_rel_l2 = jnp.linalg.norm(pred_diff, axis=1) / jnp.linalg.norm(Y_flat, axis=1)
mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))

print(f"Test MSE:         {test_mse:.6e}")
print(f"Test Relative L2: {mean_rel_l2:.6f}")
print(f"Min Relative L2:  {float(jnp.min(per_sample_rel_l2)):.6f}")
print(f"Max Relative L2:  {float(jnp.max(per_sample_rel_l2)):.6f}")

# %% [markdown]
"""
## Zero-Shot Super-Resolution

UNO can generalize to different resolutions without retraining. Here we test
inference at 2x the training resolution. The input is normalized with the
training statistics and the prediction is un-normalized for comparison.
"""

# %%
print()
target_resolution = RESOLUTION * 2
print(f"Testing zero-shot super-resolution: {RESOLUTION} -> {target_resolution}")

# Take one test sample and upsample the (normalized) input
x_sample = X_test_jnp[0:1]
x_high_res = jax.image.resize(
    x_sample,
    (1, target_resolution, target_resolution, in_channels),
    method="bilinear",
)

# Predict at high resolution, then un-normalize
y_pred_high = trained_model(x_high_res, deterministic=True) * y_std + y_mean

# Upsample ground truth for comparison
y_true_high = jax.image.resize(
    Y_test_jnp[0:1],
    (1, target_resolution, target_resolution, out_channels),
    method="bilinear",
)

sr_error = float(
    jnp.sqrt(jnp.sum((y_pred_high - y_true_high) ** 2)) / jnp.sqrt(jnp.sum(y_true_high**2))
)
print(f"Super-resolution L2 error: {sr_error:.6f}")

# %% [markdown]
"""
## Visualization

Plot the input field, ground truth, UNO prediction, and absolute error
for selected test samples, plus the super-resolution result.
"""

# %%
print()
print("Generating visualizations...")

# --- Sample predictions ---
n_vis = 3
indices = np.linspace(0, len(X_test) - 1, n_vis, dtype=int)

fig, axes = plt.subplots(n_vis, 4, figsize=(16, 4 * n_vis))
fig.suptitle("UNO Darcy Flow Predictions (Opifex)", fontsize=14, fontweight="bold")

for row, idx in enumerate(indices):
    x_field = X_test[idx, :, :, 0]
    y_true = Y_test[idx, :, :, 0]
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
plt.savefig(OUTPUT_DIR / "uno_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Predictions saved to {OUTPUT_DIR / 'uno_predictions.png'}")

# --- Super-resolution visualization ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(
    f"UNO Zero-Shot Super-Resolution ({RESOLUTION} -> {target_resolution})",
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
plt.savefig(OUTPUT_DIR / "uno_superresolution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Super-resolution saved to {OUTPUT_DIR / 'uno_superresolution.png'}")

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
print()
print("=" * 70)
print(f"UNO Darcy Flow example completed in {training_time:.1f}s")
print(f"Test MSE: {test_mse:.6e}, Relative L2: {mean_rel_l2:.6f}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)
