# %% [markdown]
# # TFNO on Darcy Flow
#
# | Property      | Value                                    |
# |---------------|------------------------------------------|
# | Level         | Intermediate                             |
# | Runtime       | ~3 min (CPU), ~30s (GPU)                 |
# | Memory        | ~1 GB                                    |
# | Prerequisites | JAX, Flax NNX, Neural Operators basics   |
#
# ## Overview
#
# Train a Tensorized Fourier Neural Operator (TFNO) on the Darcy flow problem.
# TFNO extends the FNO architecture with complex-valued spectral convolution
# weights that operate directly on Fourier coefficients.
#
# This example demonstrates:
#
# - **Complex spectral weights** for enhanced frequency-domain learning
# - **create_tucker_fno()** factory for simplified model creation
# - **Mode truncation** for efficient spectral convolutions
# - **Comparison** with standard FNO architecture
#
# Equivalent to `neuraloperator` Tucker FNO examples,
# reimplemented using Opifex APIs.
#
# ## Learning Goals
#
# 1. Use `create_tucker_fno()` factory for parameter-efficient FNO
# 2. Understand Tucker decomposition compression in spectral layers
# 3. Compare TFNO vs FNO parameter counts and accuracy tradeoffs
# 4. Analyze compression statistics per layer

# %% [markdown]
# ## Imports and Setup

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
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.fno.tensorized import create_tucker_fno


print("=" * 70)
print("Opifex Example: TFNO (Tucker-Factorized FNO) on Darcy Flow")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## Configuration
#
# The rank parameter controls compression: rank=0.1 means ~10% of parameters
# compared to equivalent non-factorized weight tensors.

# %%
RESOLUTION = 64
N_TRAIN = 200
N_TEST = 50
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
MODES = (12, 12)
HIDDEN_WIDTH = 32
NUM_LAYERS = 4
RANK = 0.1  # Tucker compression ratio (10% of full parameters)
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/tfno_darcy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
print(f"FNO config: modes={MODES}, width={HIDDEN_WIDTH}, layers={NUM_LAYERS}")
print(f"Tucker rank: {RANK} (target ~{int(RANK * 100)}% compression)")

# %% [markdown]
# ## Data Loading
#
# Use the standard Darcy flow loader - data format is identical for FNO and TFNO.

# %%
print()
print("Generating Darcy flow data...")
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

# Collect batches into arrays
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

# Add channel dimension for TFNO (expects batch, channels, H, W)
X_train = X_train[:, np.newaxis, :, :]
Y_train = Y_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
Y_test = Y_test[:, np.newaxis, :, :]

print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test data:     X={X_test.shape}, Y={Y_test.shape}")

# %% [markdown]
# ## Model Creation and Comparison
#
# Create both TFNO and standard FNO to compare parameter counts.

# %%
print()
print("Creating TFNO model (Tucker-factorized)...")
tfno_model = create_tucker_fno(
    in_channels=1,
    out_channels=1,
    hidden_channels=HIDDEN_WIDTH,
    modes=MODES,
    rank=RANK,
    num_layers=NUM_LAYERS,
    rngs=nnx.Rngs(SEED),
)

# Count TFNO parameters
tfno_params = nnx.state(tfno_model, nnx.Param)
tfno_param_count = sum(x.size for x in jax.tree_util.tree_leaves(tfno_params))

# Create equivalent standard FNO for comparison
print("Creating standard FNO for comparison...")
fno_model = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=HIDDEN_WIDTH,
    modes=max(MODES),  # FNO takes single mode value
    num_layers=NUM_LAYERS,
    rngs=nnx.Rngs(SEED + 1),
)

fno_params = nnx.state(fno_model, nnx.Param)
fno_param_count = sum(x.size for x in jax.tree_util.tree_leaves(fno_params))

print()
print("Model: Tucker-Factorized FNO (TFNO)")
print(f"  Modes: {MODES}, Hidden width: {HIDDEN_WIDTH}, Layers: {NUM_LAYERS}")
print(f"  Tucker rank: {RANK}")
print(f"  TFNO parameters: {tfno_param_count:,}")
print(f"  Standard FNO parameters: {fno_param_count:,}")
print(f"  Parameter reduction: {(1 - tfno_param_count / fno_param_count) * 100:.1f}%")

# Get compression stats from first layer
layer_stats = tfno_model.tfno_layers[0].get_compression_stats()
print()
print("Per-layer compression stats:")
print(f"  Factorized params: {layer_stats['factorized_parameters']:,}")
print(f"  Dense equivalent:  {layer_stats['equivalent_dense_parameters']:,}")
print(f"  Compression ratio: {layer_stats['compression_ratio']:.3f}")

# %% [markdown]
# ## Training with Opifex Trainer
#
# TFNO uses the same `Trainer.fit()` interface as standard FNO.

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
    model=tfno_model,
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
# ## Evaluation

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
# ## Visualization

# %%
print()
print("Generating visualizations...")

# --- Sample predictions ---
n_vis = min(4, len(X_test))
fig, axes = plt.subplots(n_vis, 4, figsize=(16, 4 * n_vis))
fig.suptitle("TFNO Darcy Flow Predictions (Opifex)", fontsize=14, fontweight="bold")

for i in range(n_vis):
    axes[i, 0].imshow(X_test[i, 0], cmap="viridis")
    axes[i, 0].set_title("Input (Permeability)" if i == 0 else "")
    axes[i, 0].set_ylabel(f"Sample {i}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(Y_test[i, 0], cmap="RdBu_r")
    axes[i, 1].set_title("Ground Truth" if i == 0 else "")
    axes[i, 1].axis("off")

    pred_np = np.array(predictions[i, 0])
    axes[i, 2].imshow(pred_np, cmap="RdBu_r")
    axes[i, 2].set_title("TFNO Prediction" if i == 0 else "")
    axes[i, 2].axis("off")

    error = np.abs(pred_np - Y_test[i, 0])
    im = axes[i, 3].imshow(error, cmap="Reds")
    axes[i, 3].set_title("Absolute Error" if i == 0 else "")
    axes[i, 3].axis("off")
    plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Sample predictions saved to {OUTPUT_DIR / 'predictions.png'}")

# --- Error and compression analysis ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("TFNO Analysis", fontsize=14, fontweight="bold")

# Error distribution
per_sample_errors = np.array(per_sample_rel_l2)
axes[0].hist(
    per_sample_errors, bins=20, alpha=0.7, color="steelblue", edgecolor="black"
)
axes[0].set_xlabel("Relative L2 Error")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Error Distribution")
axes[0].grid(True, alpha=0.3)

# Parameter comparison
models = ["Standard\nFNO", "Tucker\nTFNO"]
params = [fno_param_count, tfno_param_count]
colors = ["coral", "steelblue"]
bars = axes[1].bar(models, params, color=colors, edgecolor="black", alpha=0.7)
axes[1].set_ylabel("Number of Parameters")
axes[1].set_title("Parameter Comparison")
for bar, count in zip(bars, params, strict=False):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
axes[1].grid(True, alpha=0.3, axis="y")

# Per-sample error
axes[2].plot(per_sample_errors, "o-", alpha=0.7, color="coral", markersize=3)
axes[2].set_xlabel("Sample Index")
axes[2].set_ylabel("Relative L2 Error")
axes[2].set_title("Error per Sample")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis saved to {OUTPUT_DIR / 'analysis.png'}")

# %% [markdown]
# ## Results Summary
#
# TFNO achieves similar accuracy to FNO with significantly fewer parameters.
# The Tucker decomposition compresses spectral convolution weights while
# preserving the essential frequency components.
#
# ## Next Steps
#
# - Try different rank values (0.05, 0.2) to explore accuracy-compression tradeoffs
# - Compare with CP (`create_cp_fno()`) and Tensor Train (`create_tt_fno()`) factorizations
# - Apply TFNO to larger problems where memory savings are more significant
# - Experiment with progressive rank training (start low, increase during training)
#
# ### Related Examples
#
# - [FNO on Darcy Flow](fno-darcy.md) — Standard FNO baseline
# - [FNO on Burgers Equation](fno-burgers.md) — 1D temporal evolution
# - [Operator Comparison Tour](operator-tour.md) — Compare all operators

# %%
print()
print("=" * 70)
print(f"TFNO Darcy example completed in {training_time:.1f}s")
print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
print(f"Parameters: TFNO={tfno_param_count:,} vs FNO={fno_param_count:,}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)
