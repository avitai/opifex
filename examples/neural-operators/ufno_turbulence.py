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
# Comprehensive U-FNO for Turbulence Modeling

| Property    | Value                                                      |
|-------------|------------------------------------------------------------|
| Level       | Advanced                                                   |
| Runtime     | ~15 min (CPU/GPU)                                          |
| Prerequisites | JAX, Flax NNX, Multi-scale Analysis, Energy Conservation |

## Overview
This example demonstrates U-FNO functionality for multi-scale turbulence modeling
using the Opifex framework with JAX/Flax NNX. Features include grid embeddings,
physics-aware energy conservation loss, and comprehensive turbulence analysis.

We use Opifex's `create_turbulence_ufno` factory, `GridEmbedding2D` for positional
encoding, `create_burgers_loader` for streaming data, and `Trainer.fit()` with
custom energy conservation loss via `trainer.custom_losses`.

## Learning Goals
1. Build U-FNO with `create_turbulence_ufno` and `GridEmbedding2D`
2. Train with Opifex `Trainer` using custom physics loss
3. Analyze multi-scale frequency content
4. Evaluate energy conservation and prediction quality
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
from opifex.data.loaders.factory import create_burgers_loader
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.ufno import create_turbulence_ufno


print("=" * 70)
print("Opifex Example: Comprehensive U-FNO for Turbulence Modeling")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Configuration
"""

# %%
RESOLUTION = 64
N_TRAIN = 300
N_TEST = 60
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
IN_CHANNELS = 1
OUT_CHANNELS = 1
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/ufno_turbulence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}x{RESOLUTION}, Samples: {N_TRAIN}/{N_TEST}")
print(f"Batch: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

# %% [markdown]
"""
## Data Loading with Grain

We use `create_burgers_loader` for 2D turbulent Burgers equation data,
then collect into arrays for `Trainer.fit()`.
"""

# %%
print("\nLoading 2D Turbulent Burgers data via Grain...")
train_loader = create_burgers_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    dimension="2d",
    resolution=RESOLUTION,
    viscosity_range=(0.001, 0.005),
    time_range=(0.0, 1.0),
    shuffle=True,
    seed=SEED + 2000,
    worker_count=0,
)
test_loader = create_burgers_loader(
    n_samples=N_TEST,
    batch_size=BATCH_SIZE,
    dimension="2d",
    resolution=RESOLUTION,
    viscosity_range=(0.001, 0.005),
    time_range=(0.0, 1.0),
    shuffle=False,
    seed=SEED + 3000,
    worker_count=0,
)

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

# Add channel dimension: (batch, H, W) -> (batch, 1, H, W)
# For Burgers Y: (batch, time_steps, H, W) -> (batch, 1, time_steps, H, W)
if X_train.ndim == 3:
    X_train, Y_train = X_train[:, None, :, :], Y_train[:, None, :, :]
if X_test.ndim == 3:
    X_test, Y_test = X_test[:, None, :, :], Y_test[:, None, :, :]

# Handle Burgers output with time steps: (batch, 1, time_steps, H, W) -> (batch, 1, H, W)
# Use last time step as prediction target
if Y_train.ndim == 5:
    Y_train = Y_train[:, :, -1, :, :]
if Y_test.ndim == 5:
    Y_test = Y_test[:, :, -1, :, :]

print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test:  X={X_test.shape}, Y={Y_test.shape}")

# %% [markdown]
"""
## Model Creation with Grid Embedding

`GridEmbedding2D` appends spatial coordinate channels to the input.
The U-FNO model is created via `create_turbulence_ufno` factory.
"""

# %%
print("\nCreating U-FNO model with grid embedding...")
grid_embedding = GridEmbedding2D(
    in_channels=IN_CHANNELS, grid_boundaries=[[0.0, 1.0], [0.0, 1.0]]
)

model = create_turbulence_ufno(
    in_channels=grid_embedding.out_channels,
    out_channels=OUT_CHANNELS,
    rngs=nnx.Rngs(SEED),
)

print(f"GridEmbedding2D: {IN_CHANNELS} -> {grid_embedding.out_channels} channels")
print(f"U-FNO: {grid_embedding.out_channels} -> {OUT_CHANNELS} channels")

# %% [markdown]
"""
## Apply Grid Embedding to Data

We pre-apply the grid embedding to all data before training so that
`Trainer.fit()` can work with the embedded inputs directly.
"""

# %%
print("\nApplying grid embedding to data...")


def apply_embedding(x_data, embedding):
    """Apply grid embedding: (B, C, H, W) -> embed -> (B, C+2, H, W)."""
    x_grid = jnp.moveaxis(jnp.array(x_data), 1, -1)  # (B, H, W, C)
    x_embedded = embedding(x_grid)  # (B, H, W, C+2)
    return np.array(jnp.moveaxis(x_embedded, -1, 1))  # (B, C+2, H, W)


X_train_emb = apply_embedding(X_train, grid_embedding)
X_test_emb = apply_embedding(X_test, grid_embedding)

print(f"Embedded train: {X_train_emb.shape}")
print(f"Embedded test:  {X_test_emb.shape}")

# %% [markdown]
"""
## Training with Opifex Trainer

We use `Trainer.fit()` with a custom energy conservation loss registered
via `trainer.custom_losses["energy"]`.
"""

# %%
print("\nSetting up Trainer...")
config = TrainingConfig(
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    verbose=True,
)
trainer = Trainer(model=model, config=config, rngs=nnx.Rngs(SEED))


# Register energy conservation as custom loss
def energy_loss_fn(model, x, y_pred, y_true):
    """Energy conservation loss: penalize energy deviation."""
    pred_energy = jnp.mean(y_pred**2, axis=(2, 3))
    target_energy = jnp.mean(y_true**2, axis=(2, 3))
    return 0.1 * jnp.mean(jnp.abs(pred_energy - target_energy))


trainer.custom_losses["energy"] = energy_loss_fn
print(f"Optimizer: Adam (lr={LEARNING_RATE})")
print("Custom loss: energy conservation (weight=0.1)")

print("\nStarting training...")
start_time = time.time()
trained_model, metrics = trainer.fit(
    train_data=(jnp.array(X_train_emb), jnp.array(Y_train)),
    val_data=(jnp.array(X_test_emb), jnp.array(Y_test)),
)
training_time = time.time() - start_time
print(
    f"Done in {training_time:.1f}s | Train: {metrics.get('final_train_loss', 'N/A')} | Val: {metrics.get('final_val_loss', 'N/A')}"
)

# %% [markdown]
"""
## Comprehensive Evaluation
"""

# %%
print("\nRunning comprehensive evaluation...")
X_test_jnp = jnp.array(X_test_emb)
Y_test_jnp = jnp.array(Y_test)
predictions = trained_model(X_test_jnp)

test_mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))

per_sample_errors = []
for i in range(Y_test_jnp.shape[0]):
    p, t = predictions[i : i + 1], Y_test_jnp[i : i + 1]
    per_sample_errors.append(
        float(jnp.sqrt(jnp.sum((p - t) ** 2)) / jnp.sqrt(jnp.sum(t**2)))
    )
mean_error, std_error = (
    float(np.mean(per_sample_errors)),
    float(np.std(per_sample_errors)),
)

pred_energy = jnp.mean(predictions**2, axis=(2, 3))
target_energy = jnp.mean(Y_test_jnp**2, axis=(2, 3))
energy_conservation = float(jnp.mean(jnp.abs(pred_energy - target_energy)))

print(f"MSE: {test_mse:.6f} | Rel L2: {mean_error:.6f}+/-{std_error:.6f}")
print(f"Energy Conservation: {energy_conservation:.6f}")

# %% [markdown]
"""
## Visualization: Training Curves
"""

# %%
print("\nGenerating training curves...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("U-FNO Training - Turbulence Modeling", fontsize=16, fontweight="bold")

axes[0, 0].bar(
    ["Train", "Val"],
    [metrics.get("final_train_loss", 0), metrics.get("final_val_loss", 0)],
    color=["steelblue", "indianred"],
)
axes[0, 0].set_title("Final Loss", fontweight="bold")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar(["Test MSE"], [test_mse], color="steelblue")
axes[0, 1].set_title("Test MSE", fontweight="bold")
axes[0, 1].set_ylabel("MSE")
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].bar(["Rel L2"], [mean_error], yerr=[std_error], color="steelblue", capsize=5)
axes[0, 2].set_title("Relative L2", fontweight="bold")
axes[0, 2].set_ylabel("Rel L2")
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].bar(["Energy"], [energy_conservation], color="darkorange")
axes[1, 0].set_title("Energy Conservation Error", fontweight="bold")
axes[1, 0].set_ylabel("Error")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(per_sample_errors, "o-", ms=4, lw=1, color="darkblue")
axes[1, 1].axhline(mean_error, color="red", ls="--", label=f"Mean: {mean_error:.4f}")
axes[1, 1].set_title("Per-Sample Error", fontweight="bold")
axes[1, 1].set_xlabel("Sample")
axes[1, 1].set_ylabel("Rel L2")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

s = (
    f"\nMetrics Summary:\n\nMSE: {test_mse:.6f}\nRel L2: {mean_error:.6f}+/-{std_error:.6f}\n"
    f"Energy: {energy_conservation:.6f}\nTime: {training_time:.1f}s\n"
)
axes[1, 2].text(
    0.1,
    0.5,
    s,
    fontsize=11,
    transform=axes[1, 2].transAxes,
    va="center",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7},
)
axes[1, 2].set_title("Summary", fontweight="bold")
axes[1, 2].axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=300, bbox_inches="tight")
plt.close()

# %% [markdown]
"""
## Visualization: Sample Predictions
"""

# %%
print("Generating sample predictions...")
pred_np = np.array(predictions)
n_show = min(3, pred_np.shape[0])

fig, axes = plt.subplots(n_show, 4, figsize=(20, 5 * n_show))
if n_show == 1:
    axes = axes[None, :]
fig.suptitle("U-FNO Turbulence Predictions", fontsize=16, fontweight="bold")

for i in range(n_show):
    for ax, d, t in [
        (axes[i, 0], X_test[i, 0], f"Input {i + 1}"),
        (axes[i, 1], Y_test[i, 0], f"Truth {i + 1}"),
        (axes[i, 2], pred_np[i, 0], f"U-FNO {i + 1}"),
    ]:
        im = ax.imshow(d, cmap="RdBu_r", aspect="equal")
        ax.set_title(t, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.8)
    err = np.abs(pred_np[i, 0] - Y_test[i, 0])
    im3 = axes[i, 3].imshow(err, cmap="plasma", aspect="equal")
    axes[i, 3].set_title(f"Error {i + 1}", fontweight="bold")
    axes[i, 3].set_xlabel("x")
    plt.colorbar(im3, ax=axes[i, 3], shrink=0.8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=300, bbox_inches="tight")
plt.close()

# %% [markdown]
"""
## Visualization: Multi-Scale Analysis

Analyze frequency content and multi-scale error behavior of the U-FNO predictions.
"""

# %%
print("Generating multi-scale analysis...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("U-FNO Multi-Scale Analysis", fontsize=16, fontweight="bold")

pred_fft = np.abs(np.fft.fft2(pred_np[0, 0]))
target_fft = np.abs(np.fft.fft2(Y_test[0, 0]))

axes[0, 0].semilogy(np.mean(pred_fft, axis=0), "b-", label="U-FNO", lw=2)
axes[0, 0].semilogy(np.mean(target_fft, axis=0), "r--", label="Truth", lw=2)
axes[0, 0].set_title("Frequency Content (x)", fontweight="bold")
axes[0, 0].set_xlabel("Mode")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].semilogy(np.mean(pred_fft, axis=1), "b-", label="U-FNO", lw=2)
axes[0, 1].semilogy(np.mean(target_fft, axis=1), "r--", label="Truth", lw=2)
axes[0, 1].set_title("Frequency Content (y)", fontweight="bold")
axes[0, 1].set_xlabel("Mode")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

scales = [1, 2, 4, 8]
scale_errors = []
for sc in scales:
    ps = pred_np[0, 0, ::sc, ::sc]
    ts = Y_test[0, 0, ::sc, ::sc]
    scale_errors.append(np.mean((ps - ts) ** 2))
axes[1, 0].loglog(scales, scale_errors, "go-", lw=2, ms=8)
axes[1, 0].set_title("Multi-Scale Error", fontweight="bold")
axes[1, 0].set_xlabel("Scale")
axes[1, 0].set_ylabel("MSE")
axes[1, 0].grid(True, alpha=0.3)

pred_es = np.mean(pred_fft**2, axis=0)
target_es = np.mean(target_fft**2, axis=0)
freqs = np.fft.fftfreq(pred_np.shape[-1])
pf = freqs[freqs > 0]
axes[1, 1].loglog(pf, pred_es[: len(pf)], "b-", label="U-FNO", lw=2)
axes[1, 1].loglog(pf, target_es[: len(pf)], "r--", label="Truth", lw=2)
axes[1, 1].set_title("Energy Spectrum", fontweight="bold")
axes[1, 1].set_xlabel("Frequency")
axes[1, 1].set_ylabel("Energy")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "multiscale_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# %% [markdown]
"""
## Visualization: Error Analysis
"""

# %%
print("Generating error analysis...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("U-FNO Error Analysis", fontsize=16, fontweight="bold")

axes[0, 0].hist(
    per_sample_errors, bins=15, alpha=0.7, color="lightcoral", edgecolor="black"
)
axes[0, 0].axvline(
    mean_error, color="red", ls="--", label=f"Mean: {mean_error:.4f}", lw=2
)
axes[0, 0].set_title("Error Distribution", fontweight="bold")
axes[0, 0].set_xlabel("Rel L2")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(per_sample_errors, "o-", ms=6, lw=2, color="darkblue")
axes[0, 1].set_title("Error vs Sample", fontweight="bold")
axes[0, 1].set_xlabel("Sample")
axes[0, 1].grid(True, alpha=0.3)

se = np.sort(per_sample_errors)
cu = np.arange(1, len(se) + 1) / len(se)
axes[1, 0].plot(se, cu, lw=3, color="forestgreen")
axes[1, 0].set_title("Cumulative Error", fontweight="bold")
axes[1, 0].set_xlabel("Rel L2")
axes[1, 0].grid(True, alpha=0.3)

es = (
    f"\nError Statistics:\n\nMean: {mean_error:.6f}\nStd: {std_error:.6f}\n"
    f"Min: {np.min(per_sample_errors):.6f}\nMax: {np.max(per_sample_errors):.6f}\n"
    f"\n95th Pct: {np.percentile(per_sample_errors, 95):.6f}\n"
)
axes[1, 1].text(
    0.1,
    0.5,
    es,
    fontsize=12,
    transform=axes[1, 1].transAxes,
    va="center",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgreen", "alpha": 0.7},
)
axes[1, 1].set_title("Statistics", fontweight="bold")
axes[1, 1].axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# %% [markdown]
"""
## Results Summary + Next Steps

After running this example you should observe:
- Decreasing training loss with physics-aware energy conservation
- Multi-scale frequency analysis showing U-FNO captures turbulence structure
- Comprehensive error statistics across the test set

**Next steps:**
- Increase resolution and training epochs for better convergence
- Experiment with stronger energy conservation loss weights
- Try different viscosity ranges for more/less turbulent regimes
- Compare U-FNO with standard FNO on the same turbulence data
"""

# %%
print()
print("=" * 70)
print(f"Comprehensive U-FNO Turbulence example completed in {training_time:.1f}s")
print(f"Mean Relative L2 Error: {mean_error:.6f}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)
