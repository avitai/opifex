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
| Runtime       | ~5 min (CPU/GPU)                               |
| Memory        | ~2 GB                                          |
| Prerequisites | JAX, Flax NNX, Neural Operators basics         |

## Overview

Train a U-Net Neural Operator (UNO) on the Darcy flow equation. UNO
combines U-Net's multi-scale encoder-decoder architecture with Fourier
spectral convolutions, enabling operator learning with **zero-shot
super-resolution** capabilities.

This example demonstrates:

- **create_uno** factory for quick model construction
- **Grain DataLoader** for efficient streaming data
- **Trainer.fit()** for end-to-end training with validation
- **Zero-shot super-resolution** inference at higher resolutions

## Learning Goals

1. Create a UNO with `create_uno` factory
2. Load Darcy flow data with `create_darcy_loader` (Grain-based)
3. Train with Opifex's `Trainer.fit()` API
4. Evaluate predictions and visualize results
5. Demonstrate zero-shot super-resolution
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
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from opifex.core.training import Trainer, TrainingConfig
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.specialized import create_uno


print("=" * 70)
print("Opifex Example: UNO on Darcy Flow")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Configuration
"""

# %%
RESOLUTION = 32
N_TRAIN = 200
N_TEST = 50
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 5e-4
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/uno_darcy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

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
## Model Creation

The `create_uno` factory creates a U-Net Neural Operator with spectral
convolutions. The UNO architecture uses a U-Net encoder-decoder structure
combined with Fourier layers for multi-scale operator learning.

You can also use `UNeuralOperator` directly for more control:

```python
from opifex.neural.operators.specialized.uno import UNeuralOperator

model = UNeuralOperator(
    input_channels=1, output_channels=1,
    hidden_channels=32, modes=12, n_layers=3,
    use_spectral=True, activation=nnx.gelu,
    rngs=nnx.Rngs(42),
)
```
"""

# %%
print()
print("Creating UNO model...")
in_channels = X_train.shape[-1]
out_channels = Y_train.shape[-1]

model = create_uno(
    input_channels=in_channels,
    output_channels=out_channels,
    hidden_channels=32,
    modes=12,
    n_layers=3,
    rngs=nnx.Rngs(SEED),
)

params = nnx.state(model, nnx.Param)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print("Model: UNO (hidden=32, modes=12, layers=3)")
print(f"Input channels: {in_channels}, Output channels: {out_channels}")
print(f"Total parameters: {param_count:,}")

# %% [markdown]
"""
## Training with Opifex Trainer

We use Opifex's `Trainer` with `TrainingConfig`. The `Trainer.fit()` method
handles batched training with JIT compilation, validation, and progress logging.
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
)

trainer = Trainer(
    model=model,
    config=config,
    rngs=nnx.Rngs(SEED),
)

print(f"Optimizer: Adam (lr={LEARNING_RATE})")
print()
print("Starting training...")
start_time = time.time()

trained_model, metrics = trainer.fit(
    train_data=(jnp.array(X_train), jnp.array(Y_train)),
    val_data=(jnp.array(X_test), jnp.array(Y_test)),
)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.1f}s")
print(f"Final train loss: {metrics.get('final_train_loss', 'N/A')}")
print(f"Final val loss:   {metrics.get('final_val_loss', 'N/A')}")

# %% [markdown]
"""
## Evaluation

Evaluate the trained UNO on the test set by computing MSE and relative L2 error.
"""

# %%
print()
print("Evaluating on test set...")
X_test_jnp = jnp.array(X_test)
Y_test_jnp = jnp.array(Y_test)

predictions = trained_model(X_test_jnp, deterministic=True)

test_mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))

# Relative L2 error per sample
pred_diff = (predictions - Y_test_jnp).reshape(predictions.shape[0], -1)
Y_flat = Y_test_jnp.reshape(Y_test_jnp.shape[0], -1)
per_sample_rel_l2 = jnp.linalg.norm(pred_diff, axis=1) / jnp.linalg.norm(Y_flat, axis=1)
mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))

print(f"Test MSE:         {test_mse:.6f}")
print(f"Test Relative L2: {mean_rel_l2:.6f}")
print(f"Min Relative L2:  {float(jnp.min(per_sample_rel_l2)):.6f}")
print(f"Max Relative L2:  {float(jnp.max(per_sample_rel_l2)):.6f}")

# %% [markdown]
"""
## Zero-Shot Super-Resolution

UNO can generalize to different resolutions without retraining. Here we test
inference at 2x the training resolution.
"""

# %%
print()
target_resolution = RESOLUTION * 2
print(f"Testing zero-shot super-resolution: {RESOLUTION} -> {target_resolution}")

# Take one test sample and upsample the input
x_sample = X_test_jnp[0:1]
x_high_res = jax.image.resize(
    x_sample,
    (1, target_resolution, target_resolution, in_channels),
    method="bilinear",
)

# Predict at high resolution
y_pred_high = trained_model(x_high_res, deterministic=True)

# Upsample ground truth for comparison
y_true_high = jax.image.resize(
    Y_test_jnp[0:1],
    (1, target_resolution, target_resolution, out_channels),
    method="bilinear",
)

sr_error = float(
    jnp.sqrt(jnp.sum((y_pred_high - y_true_high) ** 2))
    / jnp.sqrt(jnp.sum(y_true_high**2))
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
    x_sample = X_test[idx, :, :, 0]
    y_true = Y_test[idx, :, :, 0]
    y_pred = np.array(predictions[idx, :, :, 0])
    error = np.abs(y_pred - y_true)

    im0 = axes[row, 0].imshow(x_sample, cmap="viridis")
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
- Decreasing training loss over epochs on Darcy flow data
- Reasonable predictions mapping permeability to pressure fields
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
print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)
