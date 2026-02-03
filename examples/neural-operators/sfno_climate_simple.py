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
# Simple SFNO for Climate Modeling

| Property      | Value                                        |
|---------------|----------------------------------------------|
| Level         | Intermediate                                 |
| Runtime       | ~3 min (CPU/GPU)                             |
| Prerequisites | JAX, Flax NNX, Spherical Harmonics basics    |

## Overview
This example demonstrates the Spherical Fourier Neural Operator (SFNO) for climate
modeling using the Opifex framework. The SFNO operates on spherical domains using
spherical harmonic transforms, making it well-suited for global climate and weather
prediction tasks.

We use Opifex's `create_climate_sfno` factory to build the model, the
`create_shallow_water_loader` for streaming data via Google Grain, and the
`Trainer` with `TrainingConfig` for the training loop.

## Learning Goals
1. Create an SFNO with `create_climate_sfno` factory
2. Load climate data with `create_shallow_water_loader` (Grain-based)
3. Train with Opifex's `Trainer.fit()` API
4. Evaluate and visualize climate predictions on a spherical domain
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

# Opifex framework imports
from opifex.core.training import Trainer, TrainingConfig
from opifex.data.loaders import create_shallow_water_loader
from opifex.neural.operators.fno.spherical import create_climate_sfno


print("=" * 70)
print("Opifex Example: Simple Spherical FNO for Climate Modeling")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Configuration

We define experiment parameters as simple variables. In production, you might
use `argparse` or configuration files.
"""

# %%
RESOLUTION = 32
N_TRAIN = 50
N_TEST = 10
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/sfno_climate_simple")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
print(f"Output directory: {OUTPUT_DIR}")

# %% [markdown]
"""
## Data Loading with Grain

Opifex provides `create_shallow_water_loader` which generates synthetic
shallow water equation data and wraps it in a Google Grain DataLoader
for efficient streaming and batching.
"""

# %%
print()
print("Loading shallow water equation data via Grain...")
train_loader = create_shallow_water_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=True,
    seed=SEED,
    worker_count=0,
)

test_loader = create_shallow_water_loader(
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

# Ensure 4D tensors: (batch, channels, height, width)
if X_train.ndim == 3:
    X_train = X_train[:, None, :, :]
    Y_train = Y_train[:, None, :, :]
if X_test.ndim == 3:
    X_test = X_test[:, None, :, :]
    Y_test = Y_test[:, None, :, :]

print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test data:     X={X_test.shape}, Y={Y_test.shape}")

# %% [markdown]
"""
## Model Creation

The `create_climate_sfno` factory creates a Spherical FNO pre-configured for
climate modeling. It sets up spherical harmonic convolution layers with the
specified maximum degree `lmax`.
"""

# %%
print()
print("Creating Spherical FNO model...")
in_channels = X_train.shape[1]
out_channels = Y_train.shape[1]

model = create_climate_sfno(
    in_channels=in_channels,
    out_channels=out_channels,
    lmax=8,
    rngs=nnx.Rngs(SEED),
)

print("Model: Spherical FNO (lmax=8)")
print(f"Input channels: {in_channels}, Output channels: {out_channels}")

# %% [markdown]
"""
## Training with Opifex Trainer

Instead of writing a manual training loop, we use Opifex's `Trainer` with
`TrainingConfig`. The `Trainer.fit()` method handles:
- Batched training with JIT compilation
- Validation at configurable intervals
- Progress logging
- Checkpointing (optional)
"""

# %%
print()
print("Setting up Trainer...")
config = TrainingConfig(
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
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

Evaluate the trained model on the test set by computing MSE and relative L2 error.
"""

# %%
print()
print("Evaluating on test set...")
X_test_jnp = jnp.array(X_test)
Y_test_jnp = jnp.array(Y_test)

predictions = trained_model(X_test_jnp)

test_mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))

# Relative L2 error per sample
pred_diff = (predictions - Y_test_jnp).reshape(predictions.shape[0], -1)
Y_flat = Y_test_jnp.reshape(Y_test_jnp.shape[0], -1)
rel_l2 = float(
    jnp.mean(jnp.linalg.norm(pred_diff, axis=1) / jnp.linalg.norm(Y_flat, axis=1))
)

print(f"Test MSE:         {test_mse:.6f}")
print(f"Test Relative L2: {rel_l2:.6f}")

# %% [markdown]
"""
## Visualization

Plot the input field, ground truth, SFNO prediction, and absolute error
for a sample from the test set.
"""

# %%
print()
print("Generating visualization...")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(
    "Spherical FNO Climate Prediction (Opifex)", fontsize=14, fontweight="bold"
)

sample_idx = 0

# Input
im0 = axes[0].imshow(X_test[sample_idx, 0], cmap="RdBu_r", aspect="equal")
axes[0].set_title("Input")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Ground truth
im1 = axes[1].imshow(Y_test[sample_idx, 0], cmap="RdBu_r", aspect="equal")
axes[1].set_title("Ground Truth")
axes[1].set_xlabel("Longitude")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

# Prediction
pred_np = np.array(predictions[sample_idx, 0])
im2 = axes[2].imshow(pred_np, cmap="RdBu_r", aspect="equal")
axes[2].set_title("SFNO Prediction")
axes[2].set_xlabel("Longitude")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

# Error
error = np.abs(pred_np - Y_test[sample_idx, 0])
im3 = axes[3].imshow(error, cmap="plasma", aspect="equal")
axes[3].set_title("Absolute Error")
axes[3].set_xlabel("Longitude")
plt.colorbar(im3, ax=axes[3], shrink=0.8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sfno_results.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Visualization saved to {OUTPUT_DIR / 'sfno_results.png'}")

# %% [markdown]
"""
## Results Summary

After running this example you should observe:
- Decreasing training loss over epochs on spherical domain data
- Reasonable predictions for the shallow water equations proxy
- Visualization comparing input, ground truth, SFNO prediction, and error

## Next Steps
- Try the comprehensive SFNO example with conservation-aware loss
- Increase `lmax` for higher spectral resolution
- Experiment with more training samples and epochs
- Explore energy and mass conservation analysis
"""

# %%
print()
print("=" * 70)
print(f"Spherical FNO Climate example completed in {training_time:.1f}s")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)
