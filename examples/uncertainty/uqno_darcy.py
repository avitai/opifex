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
# UQNO on Darcy Flow

| Property      | Value                                    |
|---------------|------------------------------------------|
| Level         | Intermediate                             |
| Runtime       | ~3 min (GPU) / ~15 min (CPU)             |
| Memory        | ~1.5 GB                                  |
| Prerequisites | JAX, Flax NNX, Bayesian Neural Networks  |

## Overview

Train an Uncertainty Quantification Neural Operator (UQNO) on the Darcy flow
equation to predict both the pressure solution and uncertainty estimates.

**Opifex's UQNO** uses Bayesian spectral convolutions with learned weight
distributions, providing:
- **Epistemic uncertainty**: Model uncertainty from weight distributions
- **Aleatoric uncertainty**: Data uncertainty from learned noise parameters
- **Monte Carlo sampling**: Probabilistic predictions via weight sampling

This differs from conformal prediction approaches (e.g., neuraloperator's UQNO)
which use a two-stage training with base + residual models.

**Reference**: Ma et al. (2024), "Calibrated Uncertainty Quantification for
Operator Learning via Conformal Prediction", TMLR.

## Learning Goals

1. Use `UncertaintyQuantificationNeuralOperator` for Bayesian predictions
2. Train with ELBO loss (data likelihood + KL divergence)
3. Compute epistemic vs aleatoric uncertainty via Monte Carlo
4. Analyze uncertainty calibration quality
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
import optax
from flax import nnx

from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.specialized.uqno import (
    UncertaintyQuantificationNeuralOperator,
)


print("=" * 70)
print("Opifex Example: UQNO on Darcy Flow")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Configuration

Key hyperparameters for Bayesian UQNO training.
"""

# %%
# Data configuration
RESOLUTION = 64
N_TRAIN = 150
N_TEST = 30
BATCH_SIZE = 8

# Model configuration
MODES = (12, 12)
HIDDEN_CHANNELS = 32
NUM_LAYERS = 4

# Training configuration
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
KL_WEIGHT = 1e-4  # Weight for KL divergence term in ELBO

# Uncertainty configuration
MC_SAMPLES = 10  # Monte Carlo samples for uncertainty estimation

SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/uqno_darcy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print()
print("Configuration:")
print(f"  Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"  Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"  Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
print(f"  UQNO: modes={MODES}, hidden={HIDDEN_CHANNELS}, layers={NUM_LAYERS}")
print(f"  KL weight: {KL_WEIGHT}, MC samples: {MC_SAMPLES}")

# %% [markdown]
"""
## Load Darcy Flow Data

The Darcy flow equation is an elliptic PDE:
$$-\\nabla \\cdot (a(x) \\nabla u(x)) = f(x)$$

where $a(x)$ is the permeability coefficient and $u(x)$ is the pressure.
"""

# %%
print()
print("Loading Darcy flow data...")

train_loader = create_darcy_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=True,
    seed=SEED,
    worker_count=0,
    enable_normalization=True,
    num_epochs=NUM_EPOCHS,
)

test_loader = create_darcy_loader(
    n_samples=N_TEST,
    batch_size=N_TEST,
    resolution=RESOLUTION,
    shuffle=False,
    seed=SEED + 1,
    worker_count=0,
    enable_normalization=True,
    num_epochs=1,
)

# Get test batch
test_batch = next(iter(test_loader))
test_inputs = jnp.array(test_batch["input"])
test_targets = jnp.array(test_batch["output"])

print(f"  Test input shape: {test_inputs.shape}")
print(f"  Test target shape: {test_targets.shape}")

# %% [markdown]
"""
## Data Preprocessing

Ensure data is in the correct format: (batch, height, width, channels).
"""


# %%
def preprocess_batch(x, y, resolution):
    """Ensure correct shape: (batch, H, W, C)."""
    if x.ndim == 3:
        x = x[..., jnp.newaxis]
        y = y[..., jnp.newaxis]
    elif x.ndim == 4 and x.shape[1] != resolution:
        # Shape is (batch, C, H, W), transpose to (batch, H, W, C)
        x = x.transpose(0, 2, 3, 1)
        y = y.transpose(0, 2, 3, 1)
    return x, y


test_inputs, test_targets = preprocess_batch(test_inputs, test_targets, RESOLUTION)

in_channels = test_inputs.shape[-1]
out_channels = test_targets.shape[-1]

print(f"  Preprocessed test input: {test_inputs.shape}")
print(f"  Input channels: {in_channels}, Output channels: {out_channels}")

# %% [markdown]
"""
## Create UQNO Model

The UQNO uses Bayesian spectral convolutions where weights are distributions
(mean + variance) rather than point estimates. This enables uncertainty
quantification through Monte Carlo sampling.
"""

# %%
print()
print("Creating UQNO model...")

model = UncertaintyQuantificationNeuralOperator(
    in_channels=in_channels,
    out_channels=out_channels,
    hidden_channels=HIDDEN_CHANNELS,
    modes=MODES,
    num_layers=NUM_LAYERS,
    use_epistemic=True,
    use_aleatoric=True,
    ensemble_size=MC_SAMPLES,
    rngs=nnx.Rngs(SEED),
)

# Warmup call to initialize dynamic modules (epistemic_head is lazily initialized)
# This must happen before JIT compilation to establish fixed model structure
dummy_input = jnp.zeros((1, RESOLUTION, RESOLUTION, in_channels))
_ = model(dummy_input, training=True)

# Count parameters (after initialization)
param_count = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
print(f"  Total parameters: {param_count:,}")
print("  Epistemic uncertainty: enabled")
print("  Aleatoric uncertainty: enabled")

# %% [markdown]
"""
## ELBO Loss Function

The Evidence Lower BOund (ELBO) combines:
- **Data likelihood**: How well predictions match targets (MSE)
- **KL divergence**: Regularization towards prior distributions

$$\\mathcal{L} = \\mathbb{E}_{q(w)}[\\log p(y|x,w)] - \\beta \\cdot KL(q(w) || p(w))$$
"""


# %%
def compute_elbo_loss(model, inputs, targets, kl_weight=KL_WEIGHT):
    """
    Compute ELBO loss for Bayesian UQNO.

    Returns:
        total_loss: ELBO loss (minimize)
        metrics: Dictionary with data_loss and kl_div
    """
    # Forward pass
    output = model(inputs, training=True)
    predictions = output["mean"]

    # Data loss (negative log likelihood ~ MSE)
    data_loss = jnp.mean((predictions - targets) ** 2)

    # KL divergence from Bayesian layers
    kl_div = model.kl_divergence()

    # ELBO loss
    total_loss = data_loss + kl_weight * kl_div

    return total_loss, {"data_loss": data_loss, "kl_div": kl_div}


# %% [markdown]
"""
## Training Loop
"""

# %%
print()
print("Training UQNO...")

opt = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(model, opt, x, y):
    """Single training step."""

    def loss_fn(m):
        loss, _ = compute_elbo_loss(m, x, y)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


start_time = time.time()
train_losses = []
epoch_count = 0
batches_per_epoch = N_TRAIN // BATCH_SIZE

for batch_count, batch in enumerate(train_loader, start=1):
    # Get and preprocess batch data
    x = jnp.array(batch["input"])
    y = jnp.array(batch["output"])
    x, y = preprocess_batch(x, y, RESOLUTION)

    loss = train_step(model, opt, x, y)
    train_losses.append(float(loss))

    if batch_count % batches_per_epoch == 0:
        epoch_count += 1
        avg_loss = np.mean(train_losses[-batches_per_epoch:])
        if epoch_count % 3 == 0 or epoch_count == 1:
            print(f"  Epoch {epoch_count:3d}/{NUM_EPOCHS}: loss = {avg_loss:.6f}")

train_time = time.time() - start_time
print(f"Training time: {train_time:.1f}s")
print(f"Final loss: {train_losses[-1]:.6f}")

# %% [markdown]
"""
## Evaluation with Monte Carlo Uncertainty

Use `predict_with_uncertainty()` for Monte Carlo sampling over weight
distributions to estimate prediction uncertainty.
"""

# %%
print()
print("Evaluating with uncertainty estimation...")

# Get predictions with uncertainty via Monte Carlo
output = model.predict_with_uncertainty(
    test_inputs, num_samples=MC_SAMPLES, key=jax.random.PRNGKey(SEED)
)

predictions = output["mean"]
epistemic_uncertainty = output["epistemic_uncertainty"]
aleatoric_uncertainty = output["aleatoric_uncertainty"]
total_uncertainty = output["total_uncertainty"]

# Compute error metrics
mse = jnp.mean((predictions - test_targets) ** 2)
l2_error = jnp.sqrt(jnp.sum((predictions - test_targets) ** 2)) / jnp.sqrt(
    jnp.sum(test_targets**2)
)
rmse = jnp.sqrt(mse)

print()
print("Results:")
print(f"  Relative L2 Error:      {float(l2_error):.4f}")
print(f"  RMSE:                   {float(rmse):.6f}")
print(f"  Mean Epistemic Std:     {float(jnp.mean(epistemic_uncertainty)):.6f}")
print(f"  Mean Aleatoric Std:     {float(jnp.mean(aleatoric_uncertainty)):.6f}")
print(f"  Mean Total Uncertainty: {float(jnp.mean(total_uncertainty)):.6f}")

# %% [markdown]
"""
## Uncertainty Calibration Analysis

Well-calibrated uncertainty should correlate with actual prediction errors.
We analyze this by:
1. Error-uncertainty correlation
2. Coverage: fraction of true values within uncertainty bounds
"""

# %%
print()
print("Uncertainty calibration analysis...")

# Compute per-sample errors
errors = jnp.abs(predictions - test_targets)
mean_error_per_sample = jnp.mean(errors, axis=(1, 2, 3))
mean_uncertainty_per_sample = jnp.mean(total_uncertainty, axis=(1, 2, 3))

# Correlation between error and uncertainty (higher = better calibrated)
correlation = jnp.corrcoef(
    mean_error_per_sample.flatten(), mean_uncertainty_per_sample.flatten()
)[0, 1]
print(f"  Error-Uncertainty Correlation: {float(correlation):.4f}")

# Coverage: fraction of errors within uncertainty bounds
coverage_1sigma = jnp.mean(errors <= total_uncertainty)
coverage_2sigma = jnp.mean(errors <= 2 * total_uncertainty)

print(f"  1-sigma coverage: {float(coverage_1sigma) * 100:.1f}% (expected ~68%)")
print(f"  2-sigma coverage: {float(coverage_2sigma) * 100:.1f}% (expected ~95%)")

# %% [markdown]
"""
## Visualization
"""

# %%
print()
print("Creating visualizations...")

sample_idx = 0

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Input, Target, Prediction, Error
ax = axes[0, 0]
im = ax.imshow(test_inputs[sample_idx, :, :, 0], cmap="viridis")
ax.set_title("Input (Permeability)")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 1]
im = ax.imshow(test_targets[sample_idx, :, :, 0], cmap="RdBu_r")
ax.set_title("Target (Pressure)")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 2]
im = ax.imshow(predictions[sample_idx, :, :, 0], cmap="RdBu_r")
ax.set_title("Prediction")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 3]
error = jnp.abs(predictions[sample_idx, :, :, 0] - test_targets[sample_idx, :, :, 0])
im = ax.imshow(error, cmap="hot")
ax.set_title("Absolute Error")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, fraction=0.046)

# Row 2: Epistemic, Aleatoric, Total Uncertainty, Calibration
ax = axes[1, 0]
im = ax.imshow(epistemic_uncertainty[sample_idx, :, :, 0], cmap="Purples")
ax.set_title("Epistemic Uncertainty")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 1]
im = ax.imshow(aleatoric_uncertainty[sample_idx, :, :, 0], cmap="Greens")
ax.set_title("Aleatoric Uncertainty")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 2]
im = ax.imshow(total_uncertainty[sample_idx, :, :, 0], cmap="Oranges")
ax.set_title("Total Uncertainty")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, fraction=0.046)

# Calibration scatter plot
ax = axes[1, 3]
ax.scatter(
    mean_uncertainty_per_sample,
    mean_error_per_sample,
    alpha=0.7,
    c="steelblue",
    edgecolors="white",
    linewidths=0.5,
)
max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, max_val], [0, max_val], "r--", label="Perfect calibration")
ax.set_xlabel("Predicted Uncertainty")
ax.set_ylabel("Actual Error")
ax.set_title(f"Calibration (r={float(correlation):.2f})")
ax.legend()

plt.suptitle("UQNO on Darcy Flow: Predictions and Uncertainty", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "solution.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"  Saved: {OUTPUT_DIR / 'solution.png'}")

# %% [markdown]
"""
## Training Analysis
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training loss
ax = axes[0]
ax.semilogy(train_losses)
ax.set_xlabel("Batch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss (ELBO)")
ax.grid(True, alpha=0.3)

# Uncertainty distribution
ax = axes[1]
ax.hist(
    epistemic_uncertainty.flatten(),
    bins=50,
    alpha=0.7,
    label="Epistemic",
    color="purple",
    density=True,
)
ax.hist(
    aleatoric_uncertainty.flatten(),
    bins=50,
    alpha=0.7,
    label="Aleatoric",
    color="green",
    density=True,
)
ax.set_xlabel("Uncertainty")
ax.set_ylabel("Density")
ax.set_title("Uncertainty Distributions")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("UQNO Training Analysis", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"  Saved: {OUTPUT_DIR / 'analysis.png'}")

# %% [markdown]
"""
## Summary

| Metric                    | Value              |
|---------------------------|--------------------|
| Relative L2 Error         | ~134 (needs tuning)|
| RMSE                      | ~0.37              |
| Mean Epistemic Std        | ~0.31              |
| Mean Aleatoric Std        | ~0.72              |
| Error-Uncertainty Corr    | ~0.92 (excellent!) |
| Training Time             | ~30s               |

**Key Observations:**
- **Uncertainty quantification works**: High correlation (0.92) between uncertainty and error
- **Epistemic uncertainty is non-zero**: Bayesian weight sampling produces varied predictions
- **L2 error is high**: Model needs more training (epochs/data) for accurate predictions
- The architecture correctly separates epistemic (model) and aleatoric (data) uncertainty

**Note on Accuracy:**
The high L2 error indicates the model is undertrained. For production use:
- Increase `NUM_EPOCHS` to 50-100
- Increase `N_TRAIN` to 500+
- Consider tuning `HIDDEN_CHANNELS`, `MODES`, `NUM_LAYERS`

**Comparison to Conformal Prediction (neuraloperator UQNO):**
- Opifex: Bayesian weights, single-stage, ELBO loss, direct uncertainty estimation
- neuraloperator: Base + residual models, two-stage, quantile loss + calibration
- Both approaches are valid; conformal provides coverage guarantees, Bayesian provides
  decomposition into epistemic/aleatoric components
"""
