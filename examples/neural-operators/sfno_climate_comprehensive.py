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
# Comprehensive SFNO for Climate Modeling

| Property    | Value                                                  |
|-------------|--------------------------------------------------------|
| Level       | Advanced                                               |
| Runtime     | ~10 min (CPU/GPU)                                      |
| Prerequisites | JAX, Flax NNX, Spherical Harmonics, Conservation Laws |

## Overview
This example demonstrates comprehensive Spherical FNO functionality for climate
modeling using the Opifex framework with JAX/Flax NNX. Features include spherical
harmonic analysis, conservation laws, and comprehensive climate data visualization.

We use Opifex's `create_climate_sfno` factory, the `create_shallow_water_loader`
for streaming data via Google Grain, and the `Trainer` with `TrainingConfig`
(including `ConservationConfig`) for physics-aware training.

## Learning Goals
1. Build comprehensive SFNO with conservation-aware loss via `TrainingConfig`
2. Analyze spherical harmonic spectra
3. Evaluate energy and mass conservation
4. Visualize climate fields on spherical domains
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

from opifex.core.training import ConservationConfig, Trainer, TrainingConfig
from opifex.data.loaders import create_shallow_water_loader
from opifex.neural.operators.fno.spherical import create_climate_sfno


print("=" * 70)
print("Opifex Example: Comprehensive Spherical FNO for Climate Modeling")
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
N_TEST = 40
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
LMAX = 8
IN_CHANNELS = 3
OUT_CHANNELS = 3
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/sfno_climate_comprehensive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}x{RESOLUTION}, Samples: {N_TRAIN}/{N_TEST}")
print(f"Batch: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, lmax: {LMAX}")

# %% [markdown]
"""
## Data Loading with Grain
"""

# %%
print("\nLoading shallow water equation data via Grain...")
train_loader = create_shallow_water_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=True,
    seed=SEED + 3000,
    worker_count=0,
)
test_loader = create_shallow_water_loader(
    n_samples=N_TEST,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=False,
    seed=SEED + 4000,
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

if X_train.ndim == 3:
    X_train, Y_train = X_train[:, None, :, :], Y_train[:, None, :, :]
if X_test.ndim == 3:
    X_test, Y_test = X_test[:, None, :, :], Y_test[:, None, :, :]

print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test:  X={X_test.shape}, Y={Y_test.shape}")

# %% [markdown]
"""
## Model Creation
"""

# %%
print("\nCreating Spherical FNO model...")
model = create_climate_sfno(
    in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, lmax=LMAX, rngs=nnx.Rngs(SEED)
)
print(f"Model: SFNO (lmax={LMAX}), channels: {IN_CHANNELS}->{OUT_CHANNELS}")

# %% [markdown]
"""
## Training with Opifex Trainer

`Trainer.fit()` with `ConservationConfig` adds energy and mass conservation loss.
"""

# %%
print("\nSetting up Trainer with conservation-aware loss...")
config = TrainingConfig(
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    verbose=True,
    conservation_config=ConservationConfig(
        laws=["energy", "mass"], energy_tolerance=1e-6, energy_monitoring=True
    ),
)
trainer = Trainer(model=model, config=config, rngs=nnx.Rngs(SEED))
print(f"Optimizer: Adam (lr={LEARNING_RATE}), Conservation: energy, mass")

print("\nStarting training...")
start_time = time.time()
trained_model, metrics = trainer.fit(
    train_data=(jnp.array(X_train), jnp.array(Y_train)),
    val_data=(jnp.array(X_test), jnp.array(Y_test)),
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
X_test_jnp, Y_test_jnp = jnp.array(X_test), jnp.array(Y_test)
predictions = trained_model(X_test_jnp)
test_mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))

per_sample_errors = []
for i in range(X_test_jnp.shape[0]):
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
pred_mass = jnp.mean(predictions, axis=(2, 3))
target_mass = jnp.mean(Y_test_jnp, axis=(2, 3))
mass_conservation = float(jnp.mean(jnp.abs(pred_mass - target_mass)))

print(f"MSE: {test_mse:.6f} | Rel L2: {mean_error:.6f}+/-{std_error:.6f}")
print(
    f"Energy Conserv: {energy_conservation:.6f} | Mass Conserv: {mass_conservation:.6f}"
)

# %% [markdown]
"""
## Visualization: Training Curves
"""

# %%
print("\nGenerating training curves...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    "Spherical FNO Training - Climate Modeling", fontsize=16, fontweight="bold"
)

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

axes[1, 0].bar(
    ["Energy", "Mass"],
    [energy_conservation, mass_conservation],
    color=["darkorange", "seagreen"],
)
axes[1, 0].set_title("Conservation Error", fontweight="bold")
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
    f"Energy: {energy_conservation:.6f}\nMass: {mass_conservation:.6f}\nTime: {training_time:.1f}s\n"
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
## Visualization: Spherical Predictions
"""

# %%
print("Generating spherical predictions...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Spherical FNO Climate Predictions", fontsize=16, fontweight="bold")
pred_np = np.array(predictions)

for ax, d, t in [
    (axes[0], X_test[0, 0], "Input"),
    (axes[1], Y_test[0, 0], "Ground Truth"),
    (axes[2], pred_np[0, 0], "SFNO Prediction"),
]:
    im = ax.imshow(d, cmap="RdBu_r", aspect="equal")
    ax.set_title(t, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, shrink=0.8)

err = np.abs(pred_np[0, 0] - Y_test[0, 0])
im3 = axes[3].imshow(err, cmap="plasma", aspect="equal")
axes[3].set_title("Absolute Error", fontweight="bold")
axes[3].set_xlabel("Longitude")
plt.colorbar(im3, ax=axes[3], shrink=0.8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "spherical_predictions.png", dpi=300, bbox_inches="tight")
plt.close()

# %% [markdown]
"""
## Visualization: Spectral Analysis
"""

# %%
print("Generating spectral analysis...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Spherical Harmonic Spectral Analysis", fontsize=16, fontweight="bold")

pred_fft = np.abs(np.fft.fft2(pred_np[0, 0]))
target_fft = np.abs(np.fft.fft2(Y_test[0, 0]))


def radial_average(data):
    """Compute radial average of 2D data for spectral analysis."""
    y, x = np.ogrid[: data.shape[0], : data.shape[1]]
    c = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - c[0], y - c[1]).astype(int)
    tb = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    nr[nr == 0] = 1
    return tb / nr


pr, tr = radial_average(pred_fft**2), radial_average(target_fft**2)
deg = np.arange(len(pr))
mi = min(20, len(deg))

axes[0, 0].loglog(deg[1:mi], pr[1:mi], "b-", label="SFNO", lw=2)
axes[0, 0].loglog(deg[1:mi], tr[1:mi], "r--", label="Truth", lw=2)
axes[0, 0].set_title("Power Spectrum", fontweight="bold")
axes[0, 0].set_xlabel("Degree l")
axes[0, 0].set_ylabel("Power")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

er = radial_average(np.abs(np.fft.fft2(pred_np[0, 0] - Y_test[0, 0])) ** 2)
axes[0, 1].loglog(deg[1:mi], er[1:mi], "g-", lw=2)
axes[0, 1].set_title("Error Spectrum", fontweight="bold")
axes[0, 1].set_xlabel("Degree l")
axes[0, 1].grid(True, alpha=0.3)

ratio = pr / (tr + 1e-10)
axes[1, 0].semilogx(deg[1:mi], ratio[1:mi], "purple", lw=2)
axes[1, 0].axhline(1.0, color="k", ls="--", alpha=0.5)
axes[1, 0].set_title("Energy Ratio by Degree", fontweight="bold")
axes[1, 0].set_xlabel("Degree l")
axes[1, 0].grid(True, alpha=0.3)

ecr = np.sum(pr[1:]) / (np.sum(tr[1:]) + 1e-10)
i5, i10, i15, i20 = (min(n, len(er)) for n in [5, 10, 15, 20])
st = (
    f"\nSpherical Harmonic Analysis:\n\nEnergy Conservation: {ecr:.4f}\n"
    f"Low-deg Error: {np.mean(er[1:i5]):.2e}\nMid-deg Error: {np.mean(er[5:i10]):.2e}\n"
    f"High-deg Error: {np.mean(er[10:i15]):.2e}\n\nPeak Degree: {np.argmax(tr[1:i20]) + 1}\n"
    f"Peak Energy: {np.max(tr[1:i20]):.2e}\n"
)
axes[1, 1].text(
    0.1,
    0.5,
    st,
    fontsize=11,
    transform=axes[1, 1].transAxes,
    va="center",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7},
)
axes[1, 1].set_title("Spectral Summary", fontweight="bold")
axes[1, 1].axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "spectral_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# %% [markdown]
"""
## Visualization: Error Analysis
"""

# %%
print("Generating error analysis...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Spherical FNO Error Analysis", fontsize=16, fontweight="bold")

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
- Decreasing training and validation loss with conservation-aware objectives
- Energy and mass conservation metrics from physics-informed training
- Spherical harmonic spectral analysis of predictions vs ground truth
- Comprehensive error statistics across the test set

**Next steps:**
- Increase `lmax` and resolution for higher-fidelity climate modeling
- Experiment with stronger conservation loss weights via `ConservationConfig`
- Compare with standard FNO on the same spherical domain data
- Integrate real climate reanalysis data (e.g., ERA5)
"""

# %%
print()
print("=" * 70)
print(f"Comprehensive SFNO Climate example completed in {training_time:.1f}s")
print(f"Mean Relative L2 Error: {mean_error:.6f}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)
