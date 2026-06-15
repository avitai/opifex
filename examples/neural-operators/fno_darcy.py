# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.6
# ---

# %% [markdown]
"""
# FNO on Darcy Flow

| Property      | Value                                    |
|---------------|------------------------------------------|
| Level         | Intermediate                             |
| Runtime       | ~3 min (CPU) / ~1 min (GPU)              |
| Memory        | ~2 GB                                    |
| Prerequisites | JAX, Flax NNX, Neural Operators basics   |

## Overview

Train a Fourier Neural Operator (FNO) on the Darcy flow equation, a 2D elliptic
PDE that maps a permeability coefficient field to the pressure solution. This is
the standard-FNO showcase on Opifex's own Darcy data: it uses
`create_darcy_loader` (smooth Darcy, solved with the accurate direct solver) and
reaches a low relative L2 error with the standard operator-learning recipe.

This example demonstrates:

- **GridEmbedding2D** for spatial positional encoding
- **FourierNeuralOperator** for spectral operator learning
- **Gaussian normalization** of inputs and outputs (fit on train, un-normalized
  predictions for physical-space error)
- **relative-L2 loss** via `LossConfig`, the standard operator-learning objective
- **AdamW + exponential LR decay + weight decay** to converge without overfitting
- **Trainer.fit()** for end-to-end training with validation

It is the FNO counterpart to [UNO on Darcy Flow](uno-darcy.md) and
[Your First Neural Operator](../getting-started/first-neural-operator.md), which
use the same synthetic Darcy data and recipe.

## Learning Goals

1. Compose `GridEmbedding2D` with `FourierNeuralOperator`
2. Load Darcy flow data with `create_darcy_loader`
3. Apply Gaussian normalization and the relative-L2 loss
4. Use AdamW + an exponential learning-rate schedule + weight decay
5. Evaluate with L2 relative error and full visualization
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
from opifex.core.training.config import LossConfig, OptimizationConfig
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

We follow the standard operator-learning recipe: ~1000 training samples,
Gaussian normalization, the relative-L2 loss, and `AdamW` with weight decay plus
an exponential learning-rate decay over enough epochs for the spectral weights to
converge. The FNO also uses `domain_padding`, which pads the spatial dimensions
before the spectral layers to reduce the Gibbs phenomenon on this non-periodic
boundary-value problem.
"""

# %%
RESOLUTION = 32  # synthetic Darcy resolution (differentiates from getting-started 32x32 super-res)
N_TRAIN = 1000
N_TEST = 100
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE = 5e-3  # AdamW initial LR
WEIGHT_DECAY = 1e-4  # regularization to combat overfitting
MODES = 12  # retained Fourier modes per axis
HIDDEN_WIDTH = 32
NUM_LAYERS = 4
DOMAIN_PADDING = 8  # pad spatial dims to soften the Gibbs phenomenon (non-periodic)
SEED = 42

# Exponential LR schedule: halve the rate every 60 epochs.
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
LR_DECAY_EPOCHS = 60
LR_TRANSITION_STEPS = LR_DECAY_EPOCHS * STEPS_PER_EPOCH
LR_DECAY_RATE = 0.5

OUTPUT_DIR = Path("docs/assets/examples/fno_darcy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
print(f"FNO config: modes={MODES}, width={HIDDEN_WIDTH}, layers={NUM_LAYERS}")
print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print(f"LR schedule: exponential, x{LR_DECAY_RATE} every {LR_DECAY_EPOCHS} epochs")

# %% [markdown]
"""
## Data Loading with Grain

Opifex provides `create_darcy_loader`, which generates Darcy flow equation data
(permeability-to-pressure mapping) with the accurate direct solver and wraps it
in a Google Grain DataLoader for efficient streaming and batching.
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
    enable_normalization=False,
)

test_loader = create_darcy_loader(
    n_samples=N_TEST,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=False,
    seed=SEED + 1000,
    worker_count=0,
    enable_normalization=False,
)

# Collect data from loaders into arrays for Trainer.fit().
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

# FNO expects channels-first: (batch, channels, H, W).
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
## Normalization

Neural operators train best on standardized fields. We fit Gaussian statistics on
the training set, normalize all splits, and un-normalize predictions before
computing the physical-space relative L2 error.
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

We compose `GridEmbedding2D` with `FourierNeuralOperator` to inject spatial
coordinates as additional input channels. This positional encoding helps the FNO
learn spatially varying operators on this boundary-value problem. The FNO also
uses `domain_padding`, padding the spatial dimensions before the spectral layers
to reduce the Gibbs phenomenon for the non-periodic Darcy problem.
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
        *,
        domain_padding: int,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the grid embedding and the underlying FNO.

        Args:
            in_channels: Number of physical input channels (before the grid).
            out_channels: Number of output channels.
            modes: Number of Fourier modes per spatial dimension.
            hidden_channels: Number of FNO hidden channels.
            num_layers: Number of spectral layers.
            grid_boundaries: Per-axis ``[min, max]`` grid extents.
            domain_padding: Pixels padded on each spatial axis to reduce the
                Gibbs phenomenon for the non-periodic Darcy problem.
            rngs: Random number generators.
        """
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
            domain_padding=domain_padding,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: grid embedding -> FNO.

        Args:
            x: Input of shape ``(batch, channels, H, W)``.

        Returns:
            Output of shape ``(batch, out_channels, H, W)``.
        """
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
    domain_padding=DOMAIN_PADDING,
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

We train with `AdamW`, weight decay, and an exponential learning-rate schedule
that halves the rate every 60 epochs. The data loss is the **relative-L2 loss**
(`loss_type="relative_l2"`), the standard operator-learning objective.
`Trainer.fit()` handles batched training with JIT compilation, validation, and
progress logging.
"""

# %%
print()
print("Setting up Trainer...")
config = TrainingConfig(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_frequency=20,
    verbose=True,
    loss_config=LossConfig(loss_type="relative_l2"),
    optimization_config=OptimizationConfig(
        optimizer="adamw",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        schedule_type="exponential",
        transition_steps=LR_TRANSITION_STEPS,
        decay_rate=LR_DECAY_RATE,
    ),
)

trainer = Trainer(
    model=model,
    config=config,
    rngs=nnx.Rngs(SEED),
)

print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print("Loss: relative L2 (the standard operator-learning objective)")
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
relative L2 error. We run the test and training sets through the model in batches
(to bound memory) and compare the two to confirm the model is not overfitting.
"""


# %%
def predict_in_batches(
    forward: nnx.Module,
    inputs: jax.Array,
    batch_size: int = 128,
) -> jax.Array:
    """Run the model over the inputs in batches to bound memory use.

    Args:
        forward: The trained model.
        inputs: Inputs of shape ``(N, channels, H, W)``.
        batch_size: Number of samples per forward pass.

    Returns:
        Concatenated predictions of shape ``(N, out_channels, H, W)``.
    """
    outputs = [forward(inputs[i : i + batch_size]) for i in range(0, inputs.shape[0], batch_size)]
    return jnp.concatenate(outputs, axis=0)


print()
print("Running evaluation...")
X_test_jnp = jnp.array(X_test_n)
Y_test_jnp = jnp.array(Y_test)
X_train_jnp = jnp.array(X_train_n)
Y_train_jnp = jnp.array(Y_train)

# Un-normalize predictions back to physical pressure units.
predictions = predict_in_batches(trained_model, X_test_jnp) * y_std + y_mean
train_predictions = predict_in_batches(trained_model, X_train_jnp) * y_std + y_mean

# Overall metrics (in physical units).
test_mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))

pred_diff = (predictions - Y_test_jnp).reshape(predictions.shape[0], -1)
Y_flat = Y_test_jnp.reshape(Y_test_jnp.shape[0], -1)
per_sample_rel_l2 = jnp.linalg.norm(pred_diff, axis=1) / jnp.linalg.norm(Y_flat, axis=1)
mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))

train_diff = (train_predictions - Y_train_jnp).reshape(train_predictions.shape[0], -1)
Y_train_flat = Y_train_jnp.reshape(Y_train_jnp.shape[0], -1)
train_rel_l2 = float(
    jnp.mean(jnp.linalg.norm(train_diff, axis=1) / jnp.linalg.norm(Y_train_flat, axis=1))
)

print(f"Train Relative L2: {train_rel_l2:.6f}")
print(f"Test  Relative L2: {mean_rel_l2:.6f}")
print(f"Overfitting gap (test - train): {mean_rel_l2 - train_rel_l2:+.6f}")
print(f"Test MSE:         {test_mse:.6e}")
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

axes[0].hist(per_sample_errors, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
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
- A decreasing relative-L2 training loss with the exponential learning-rate decay
- A low relative L2 error on the Darcy flow test set, with a small
  train-vs-test gap (the relative-L2 loss + weight decay + LR schedule prevent
  overfitting)
- Visualizations showing input permeability, ground truth pressure,
  FNO predictions, and pointwise error maps

## Next Steps

- Increase resolution and training epochs for even better accuracy
- Experiment with different numbers of Fourier modes and layers
- Compare the relative-L2 objective against an H1 (Sobolev) gradient-aware loss
- Try the UNO architecture for multi-scale Darcy flow problems
- Explore `TensorizedFourierNeuralOperator` for parameter-efficient training
"""

# %%
print()
print("=" * 70)
print(f"FNO Darcy Flow example completed in {training_time:.1f}s")
print(f"Test MSE: {test_mse:.6e}, Relative L2: {mean_rel_l2:.6f}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)
