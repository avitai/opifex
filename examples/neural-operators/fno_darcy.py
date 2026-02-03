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
# FNO on Darcy Flow

| Property      | Value                                    |
|---------------|------------------------------------------|
| Level         | Intermediate                             |
| Runtime       | ~5 min (CPU/GPU)                         |
| Memory        | ~2 GB                                    |
| Prerequisites | JAX, Flax NNX, Neural Operators basics   |

## Overview

Train a Fourier Neural Operator (FNO) on the Darcy flow equation, a 2D
elliptic PDE that maps a permeability coefficient field to the pressure
solution. This example demonstrates:

- **GridEmbedding2D** for spatial positional encoding
- **FourierNeuralOperator** for spectral operator learning
- **Grain DataLoader** for efficient streaming data
- **Trainer.fit()** for end-to-end training with validation

Equivalent to `neuraloperator/examples/models/plot_FNO_darcy.py`,
reimplemented using Opifex APIs.

## Learning Goals

1. Compose `GridEmbedding2D` with `FourierNeuralOperator`
2. Load Darcy flow data with `create_darcy_loader` (Grain-based)
3. Train with Opifex's `Trainer.fit()` API
4. Evaluate with L2 relative error and comprehensive visualization
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
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.base import FourierNeuralOperator


print("=" * 70)
print("Opifex Example: FNO on Darcy Flow")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Configuration
"""

# %%
RESOLUTION = 64
N_TRAIN = 200
N_TEST = 50
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
MODES = 12
HIDDEN_WIDTH = 32
NUM_LAYERS = 4
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/fno_darcy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
print(f"FNO config: modes={MODES}, width={HIDDEN_WIDTH}, layers={NUM_LAYERS}")

# %% [markdown]
"""
## Data Loading with Grain

Opifex provides `create_darcy_loader` which generates Darcy flow equation data
and wraps it in a Google Grain DataLoader. Each sample maps a permeability
coefficient field to the pressure solution.
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

# Ensure 4D: FNO expects (batch, channels, height, width)
if X_train.ndim == 3:
    X_train = X_train[:, np.newaxis, :, :]
    Y_train = Y_train[:, np.newaxis, :, :]
if X_test.ndim == 3:
    X_test = X_test[:, np.newaxis, :, :]
    Y_test = Y_test[:, np.newaxis, :, :]

print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test data:     X={X_test.shape}, Y={Y_test.shape}")

# %% [markdown]
"""
## Model Creation

We compose `GridEmbedding2D` with `FourierNeuralOperator` to inject spatial
coordinates as additional input channels. This positional encoding helps the
FNO learn spatially varying operators.
"""


# %%
class FNOWithEmbedding(nnx.Module):
    """FNO model with built-in grid embedding for positional encoding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        hidden_channels: int,
        num_layers: int,
        grid_boundaries: list[list[float]],
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=in_channels,
            grid_boundaries=grid_boundaries,
        )
        self.fno = FourierNeuralOperator(
            in_channels=self.grid_embedding.out_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: grid embedding -> FNO."""
        # (batch, channels, H, W) -> (batch, H, W, channels) for embedding
        x_hwc = jnp.moveaxis(x, 1, -1)
        x_embedded = self.grid_embedding(x_hwc)
        # (batch, H, W, channels) -> (batch, channels, H, W) for FNO
        x_chw = jnp.moveaxis(x_embedded, -1, 1)
        return self.fno(x_chw)


print()
print("Creating FNO model with grid embedding...")
model = FNOWithEmbedding(
    in_channels=1,
    out_channels=1,
    modes=MODES,
    hidden_channels=HIDDEN_WIDTH,
    num_layers=NUM_LAYERS,
    grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
    rngs=nnx.Rngs(SEED),
)

params = nnx.state(model, nnx.Param)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print("Model: FNO + GridEmbedding2D")
print("  Input channels: 1 (+ 2 grid coords = 3 after embedding)")
print(f"  Fourier modes: {MODES}, Hidden width: {HIDDEN_WIDTH}, Layers: {NUM_LAYERS}")
print(f"  Total parameters: {param_count:,}")

# %% [markdown]
"""
## Training with Opifex Trainer

The `Trainer.fit()` method handles the full training loop with JIT compilation,
batching, validation, and progress logging.
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

Evaluate the trained FNO on the test set with detailed per-sample metrics.
"""

# %%
print()
print("Running evaluation...")
X_test_jnp = jnp.array(X_test)
Y_test_jnp = jnp.array(Y_test)

predictions = trained_model(X_test_jnp)

# Overall metrics
test_mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))

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
## Visualization

Create visualizations of sample predictions and error analysis.
"""

# %%
print()
print("Generating visualizations...")

# --- Sample predictions ---
n_vis = min(4, len(X_test))
fig, axes = plt.subplots(n_vis, 4, figsize=(16, 4 * n_vis))
fig.suptitle("FNO Darcy Flow Predictions (Opifex)", fontsize=14, fontweight="bold")

if n_vis == 1:
    axes = axes[np.newaxis, :]

for i in range(n_vis):
    im0 = axes[i, 0].imshow(X_test[i, 0], cmap="viridis")
    axes[i, 0].set_title("Input (Permeability)" if i == 0 else "")
    axes[i, 0].axis("off")
    if i == 0:
        plt.colorbar(im0, ax=axes[i, 0], shrink=0.8)

    im1 = axes[i, 1].imshow(Y_test[i, 0], cmap="RdBu_r")
    axes[i, 1].set_title("Ground Truth" if i == 0 else "")
    axes[i, 1].axis("off")
    if i == 0:
        plt.colorbar(im1, ax=axes[i, 1], shrink=0.8)

    pred_np = np.array(predictions[i, 0])
    im2 = axes[i, 2].imshow(pred_np, cmap="RdBu_r")
    axes[i, 2].set_title("FNO Prediction" if i == 0 else "")
    axes[i, 2].axis("off")
    if i == 0:
        plt.colorbar(im2, ax=axes[i, 2], shrink=0.8)

    error = np.abs(pred_np - Y_test[i, 0])
    im3 = axes[i, 3].imshow(error, cmap="Reds")
    axes[i, 3].set_title("Absolute Error" if i == 0 else "")
    axes[i, 3].axis("off")
    if i == 0:
        plt.colorbar(im3, ax=axes[i, 3], shrink=0.8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Sample predictions saved to {OUTPUT_DIR / 'sample_predictions.png'}")

# --- Error analysis ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("FNO Error Analysis", fontsize=14, fontweight="bold")

per_sample_errors = np.array(per_sample_rel_l2)

axes[0].hist(
    per_sample_errors, bins=20, alpha=0.7, color="steelblue", edgecolor="black"
)
axes[0].set_title("Relative L2 Error Distribution")
axes[0].set_xlabel("Relative L2 Error")
axes[0].set_ylabel("Frequency")
axes[0].grid(True, alpha=0.3)

axes[1].plot(per_sample_errors, "o-", alpha=0.7, color="coral", markersize=3)
axes[1].set_title("Relative L2 Error per Sample")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Relative L2 Error")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Error analysis saved to {OUTPUT_DIR / 'error_analysis.png'}")

# %% [markdown]
"""
## Results Summary

After running this example you should observe:
- Decreasing training and validation loss over epochs
- Reasonable L2 relative error on the Darcy flow test set
- Visualizations showing input permeability, ground truth pressure,
  FNO predictions, and pointwise error maps

## Next Steps

- Increase resolution and training epochs for better accuracy
- Experiment with different numbers of Fourier modes and layers
- Try the UNO architecture for multi-scale Darcy flow problems
- Add physics-informed loss terms for conservation constraints
- Explore `TensorizedFourierNeuralOperator` for parameter-efficient training
"""

# %%
print()
print("=" * 70)
print(f"FNO Darcy Flow example completed in {training_time:.1f}s")
print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)
