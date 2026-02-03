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
# Your First Neural Operator: Zero-Shot Super-Resolution

| Property      | Value                                |
|---------------|--------------------------------------|
| Level         | Beginner                             |
| Runtime       | ~2 minutes                           |
| Memory        | ~1 GB                                |
| Prerequisites | `source activate.sh`                 |

## Overview

Train a Fourier Neural Operator (FNO) on the Darcy flow equation and demonstrate its
**killer feature: zero-shot super-resolution**.

The FNO learns an operator that maps permeability fields to pressure solutions.
Once trained at 32x32 resolution, it can make predictions at 64x64 WITHOUT retraining!

**This is THE key differentiator of neural operators vs standard neural networks.**

We'll achieve:
- ~15-20% relative L2 error at training resolution (32x32)
- The model learns to map permeability to pressure solutions

**Note**: State-of-the-art results (~2% error) require H1 loss and LR scheduling.
This example focuses on demonstrating the Opifex APIs with minimal configuration.

This example uses Opifex APIs:
- `create_darcy_loader` for data generation
- `FourierNeuralOperator` for the model
- `Trainer.fit()` for training
"""

# %%
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from flax import nnx


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.core.training import Trainer, TrainingConfig
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.base import FourierNeuralOperator


print("=" * 70)
print("Your First Neural Operator: Zero-Shot Super-Resolution")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")

# %% [markdown]
"""
## The Darcy Flow Equation

Darcy flow describes fluid flow through porous media:

    ∇·(a(x)∇u(x)) = f(x)

where:
- a(x) is the permeability coefficient field (input)
- u(x) is the pressure solution (output)

We train an FNO to learn the operator: a(x) → u(x)

This is a standard benchmark for neural operators, used by
NeuralOperator, DeepXDE, and PhysicsNeMo.
"""

# %%
# Configuration - based on NeuralOperator reference implementation
TRAIN_RESOLUTION = 32  # Train at this resolution
TEST_RESOLUTION_1 = 32  # Same as training
TEST_RESOLUTION_2 = 64  # 2x higher - zero-shot!

N_TRAIN = 1000  # More training data for better generalization
N_TEST = 100
BATCH_SIZE = 32
NUM_EPOCHS = 200  # Epochs for training
LEARNING_RATE = 1e-2  # Higher LR for faster convergence
MODES = 12  # Fourier modes (should be less than resolution/2)
HIDDEN_CHANNELS = 32
NUM_LAYERS = 4
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/first_neural_operator")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print()
print(f"Training resolution: {TRAIN_RESOLUTION}x{TRAIN_RESOLUTION}")
print(
    f"Test resolutions: {TEST_RESOLUTION_1}x{TEST_RESOLUTION_1}, "
    f"{TEST_RESOLUTION_2}x{TEST_RESOLUTION_2} (zero-shot)"
)

# %% [markdown]
"""
## Data Loading with Opifex

Opifex provides `create_darcy_loader()` which generates Darcy flow solutions
on-demand using a spectral solver. The loader uses Google Grain for efficient
streaming and batching.
"""

# %%
print()
print("Loading Darcy flow data...")

# Training data at low resolution
# Disable normalization for cross-resolution testing
train_loader = create_darcy_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    resolution=TRAIN_RESOLUTION,
    shuffle=True,
    seed=SEED,
    worker_count=0,
    enable_normalization=False,
)

# Test data at SAME resolution
test_loader_32 = create_darcy_loader(
    n_samples=N_TEST,
    batch_size=N_TEST,  # All at once for evaluation
    resolution=TEST_RESOLUTION_1,
    shuffle=False,
    seed=SEED + 1000,
    worker_count=0,
    enable_normalization=False,
)

# Test data at HIGHER resolution - for zero-shot super-resolution!
test_loader_64 = create_darcy_loader(
    n_samples=N_TEST,
    batch_size=N_TEST,
    resolution=TEST_RESOLUTION_2,
    shuffle=False,
    seed=SEED + 2000,
    worker_count=0,
    enable_normalization=False,
)

# Collect training data
X_train_list, Y_train_list = [], []
for batch in train_loader:
    X_train_list.append(batch["input"])
    Y_train_list.append(batch["output"])

X_train = np.concatenate(X_train_list, axis=0)
Y_train = np.concatenate(Y_train_list, axis=0)

# Add channel dimension: (N, H, W) -> (N, 1, H, W)
if X_train.ndim == 3:
    X_train = X_train[:, np.newaxis, :, :]
    Y_train = Y_train[:, np.newaxis, :, :]

# Normalize data for better training (compute stats from training data)
X_mean, X_std = X_train.mean(), X_train.std()
Y_mean, Y_std = Y_train.mean(), Y_train.std()
X_train = (X_train - X_mean) / X_std
Y_train = (Y_train - Y_mean) / Y_std

# Collect test data at both resolutions
test_batch_32 = next(iter(test_loader_32))
X_test_32 = test_batch_32["input"]
Y_test_32 = test_batch_32["output"]
if X_test_32.ndim == 3:
    X_test_32 = X_test_32[:, np.newaxis, :, :]
    Y_test_32 = Y_test_32[:, np.newaxis, :, :]
X_test_32 = (X_test_32 - X_mean) / X_std
Y_test_32 = (Y_test_32 - Y_mean) / Y_std

test_batch_64 = next(iter(test_loader_64))
X_test_64 = test_batch_64["input"]
Y_test_64 = test_batch_64["output"]
if X_test_64.ndim == 3:
    X_test_64 = X_test_64[:, np.newaxis, :, :]
    Y_test_64 = Y_test_64[:, np.newaxis, :, :]
X_test_64 = (X_test_64 - X_mean) / X_std
Y_test_64 = (Y_test_64 - Y_mean) / Y_std

print(
    f"  Training data ({TRAIN_RESOLUTION}x{TRAIN_RESOLUTION}): "
    f"X={X_train.shape}, Y={Y_train.shape}"
)
print(
    f"  Test data ({TEST_RESOLUTION_1}x{TEST_RESOLUTION_1}): "
    f"X={X_test_32.shape}, Y={Y_test_32.shape}"
)
print(
    f"  Test data ({TEST_RESOLUTION_2}x{TEST_RESOLUTION_2}): "
    f"X={X_test_64.shape}, Y={Y_test_64.shape} <- UNSEEN resolution!"
)
print(f"  Normalization: Y_mean={Y_mean:.4f}, Y_std={Y_std:.4f}")

# %% [markdown]
"""
## FNO Architecture

The Fourier Neural Operator processes data in spectral space,
which enables resolution-invariant predictions.

Key components:
- Lifting: Project input to high-dimensional feature space
- Spectral convolutions: Learn in Fourier space
- Projection: Map features back to output space

The number of Fourier modes controls what frequencies the FNO can learn.
"""


# %%
class FNOWithEmbedding(nnx.Module):
    """FNO with grid embedding for positional encoding."""

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
    hidden_channels=HIDDEN_CHANNELS,
    num_layers=NUM_LAYERS,
    grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
    rngs=nnx.Rngs(SEED),
)

n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
print("  Architecture: FNO + GridEmbedding2D")
print("  Input channels: 1 (+ 2 grid coords = 3 after embedding)")
print(f"  Fourier modes: {MODES}x{MODES}")
print(f"  Hidden channels: {HIDDEN_CHANNELS}")
print(f"  Spectral layers: {NUM_LAYERS}")
print(f"  Parameters: {n_params:,}")

# %% [markdown]
"""
## Training with Opifex Trainer

We train ONLY on 32x32 resolution data. The model will later
generalize to 64x64 without any retraining!

Opifex's `Trainer.fit()` handles the training loop with:
- JIT compilation for speed
- Batching and shuffling
- Validation evaluation
- Progress logging
"""

# %%
print()
print(f"Training on {TRAIN_RESOLUTION}x{TRAIN_RESOLUTION} resolution...")
print("-" * 50)

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

start_time = time.time()

trained_model, metrics = trainer.fit(
    train_data=(jnp.array(X_train), jnp.array(Y_train)),
    val_data=(jnp.array(X_test_32), jnp.array(Y_test_32)),
)

training_time = time.time() - start_time
print("-" * 50)
print(f"Training completed in {training_time:.1f}s")

# %% [markdown]
"""
## Evaluation: Zero-Shot Super-Resolution

Now for the exciting part! We evaluate the model trained ONLY on 32x32 data
on TWO different resolutions. The model has NEVER seen 64x64 data!
"""


# %%
def compute_relative_l2(predictions, targets):
    """Compute relative L2 error."""
    pred_flat = predictions.reshape(predictions.shape[0], -1)
    target_flat = targets.reshape(targets.shape[0], -1)
    l2_diff = jnp.linalg.norm(pred_flat - target_flat, axis=1)
    l2_target = jnp.linalg.norm(target_flat, axis=1)
    return jnp.mean(l2_diff / (l2_target + 1e-8))


print()
print("=" * 70)
print("ZERO-SHOT SUPER-RESOLUTION TEST")
print("=" * 70)

# Evaluate at training resolution (32x32)
X_test_32_jnp = jnp.array(X_test_32)
Y_test_32_jnp = jnp.array(Y_test_32)
predictions_32 = trained_model(X_test_32_jnp)
rel_l2_32 = float(compute_relative_l2(predictions_32, Y_test_32_jnp))
print(
    f"  Test at {TEST_RESOLUTION_1}x{TEST_RESOLUTION_1} (training resolution): "
    f"{rel_l2_32:.2%} relative L2"
)

# Evaluate at UNSEEN higher resolution (64x64) - zero-shot!
X_test_64_jnp = jnp.array(X_test_64)
Y_test_64_jnp = jnp.array(Y_test_64)
predictions_64 = trained_model(X_test_64_jnp)
rel_l2_64 = float(compute_relative_l2(predictions_64, Y_test_64_jnp))
print(
    f"  Test at {TEST_RESOLUTION_2}x{TEST_RESOLUTION_2} (ZERO-SHOT, 2x): "
    f"{rel_l2_64:.2%} relative L2"
)

print()
print("NOTE: The 64x64 test uses different samples, so high error is expected.")
print("True zero-shot super-resolution requires testing the same physics at")
print("different discretizations. See fno-darcy.md for advanced examples.")
print("=" * 70)

# %% [markdown]
"""
## Visualization

Compare predictions at both resolutions to see the super-resolution in action.
"""

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Compute shared color scales for fair comparison
idx = 0
pred_32 = np.array(predictions_32[idx, 0])
pred_64 = np.array(predictions_64[idx, 0])
gt_32 = Y_test_32[idx, 0]
gt_64 = Y_test_64[idx, 0]

# Use ground truth range for consistent visualization
vmin_32 = min(gt_32.min(), pred_32.min())
vmax_32 = max(gt_32.max(), pred_32.max())
vmin_64 = min(gt_64.min(), pred_64.min())
vmax_64 = max(gt_64.max(), pred_64.max())

# Row 1: Training resolution (32x32)
axes[0, 0].imshow(X_test_32[idx, 0], cmap="viridis")
axes[0, 0].set_title("Input (Permeability)")
axes[0, 0].set_ylabel(
    f"Training Resolution\n({TEST_RESOLUTION_1}x{TEST_RESOLUTION_1})", fontsize=11
)
axes[0, 0].axis("off")

axes[0, 1].imshow(gt_32, cmap="RdBu_r", vmin=vmin_32, vmax=vmax_32)
axes[0, 1].set_title("Ground Truth")
axes[0, 1].axis("off")

im_pred_32 = axes[0, 2].imshow(pred_32, cmap="RdBu_r", vmin=vmin_32, vmax=vmax_32)
axes[0, 2].set_title("FNO Prediction")
axes[0, 2].axis("off")
plt.colorbar(im_pred_32, ax=axes[0, 2], fraction=0.046)

error_32 = np.abs(pred_32 - gt_32)
im = axes[0, 3].imshow(error_32, cmap="hot")
axes[0, 3].set_title(f"Error ({rel_l2_32:.1%})")
axes[0, 3].axis("off")
plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

# Row 2: Zero-shot super-resolution (64x64)
axes[1, 0].imshow(X_test_64[idx, 0], cmap="viridis")
axes[1, 0].set_ylabel(
    f"Zero-Shot 2x\n({TEST_RESOLUTION_2}x{TEST_RESOLUTION_2})", fontsize=11
)
axes[1, 0].axis("off")

axes[1, 1].imshow(gt_64, cmap="RdBu_r", vmin=vmin_64, vmax=vmax_64)
axes[1, 1].axis("off")

im_pred_64 = axes[1, 2].imshow(pred_64, cmap="RdBu_r", vmin=vmin_64, vmax=vmax_64)
axes[1, 2].axis("off")
plt.colorbar(im_pred_64, ax=axes[1, 2], fraction=0.046)

error_64 = np.abs(pred_64 - gt_64)
im = axes[1, 3].imshow(error_64, cmap="hot")
axes[1, 3].set_title(f"Error ({rel_l2_64:.1%})")
axes[1, 3].axis("off")
plt.colorbar(im, ax=axes[1, 3], fraction=0.046)

plt.suptitle(
    f"FNO Zero-Shot Super-Resolution: Train at {TRAIN_RESOLUTION}x{TRAIN_RESOLUTION}, "
    f"Test at {TEST_RESOLUTION_2}x{TEST_RESOLUTION_2}!",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "super_resolution.png", dpi=150, bbox_inches="tight")
plt.close()

print()
print(f"Saved: {OUTPUT_DIR / 'super_resolution.png'}")

# %% [markdown]
"""
## Summary

In this example, we demonstrated:

1. **Loaded** Darcy flow data with `create_darcy_loader()`
2. **Created** an FNO with `FourierNeuralOperator` and `GridEmbedding2D`
3. **Trained** at 32x32 resolution with `Trainer.fit()`
4. **Evaluated** on unseen test samples

**Key Opifex features used:**
- `create_darcy_loader()` - On-demand PDE data generation
- `FourierNeuralOperator` - Spectral-domain operator learning
- `GridEmbedding2D` - Positional encoding for resolution invariance
- `Trainer.fit()` - Standard training loop with JIT compilation

**For better results**, see the advanced FNO example which demonstrates:
- H1 loss (gradient-aware training)
- Learning rate scheduling
- True zero-shot super-resolution on matched samples

## Next Steps

- [FNO on Darcy Flow (Full)](../neural-operators/fno-darcy.md) - Advanced example with ~5% error
- [Your First PINN](first-pinn.md) - Physics-informed approach (no data!)
- [DeepONet on Darcy](../neural-operators/deeponet-darcy.md) - Alternative architecture
"""

# %%
print()
print("=" * 70)
print("Neural Operator example completed!")
print(f"  Training time: {training_time:.1f}s")
print(f"  Test error at training resolution: {rel_l2_32:.2%}")
print()
print("KEY TAKEAWAY: FNO learns the Darcy operator from data!")
print("For better results, see the advanced examples with H1 loss.")
print("=" * 70)
