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
# FNO on Burgers Equation

| Property      | Value                                    |
|---------------|------------------------------------------|
| Level         | Intermediate                             |
| Runtime       | ~3 min (CPU)                             |
| Memory        | ~1 GB                                    |
| Prerequisites | JAX, Flax NNX, Neural Operators basics   |

## Overview

Train a Fourier Neural Operator (FNO) on the 1D Burgers equation, a nonlinear
PDE that develops shocks and is a standard benchmark for operator learning.
Given an initial condition u(x, 0), the FNO learns to predict the solution
u(x, t) at multiple future time steps simultaneously.

This example demonstrates:

- **1D FNO** operating on `(batch, channels, resolution)` tensors
- **Burgers data generation** with `create_burgers_loader` (varying viscosity)
- **Multi-output prediction** mapping 1 input channel to `time_steps` output channels
- **Trainer.fit()** for end-to-end training with validation

Equivalent to `neuraloperator/examples/` Burgers examples,
reimplemented using Opifex APIs.

## Learning Goals

1. Load 1D Burgers equation data with `create_burgers_loader`
2. Configure `FourierNeuralOperator` for 1D spatial data
3. Map initial conditions to multi-step solution trajectories
4. Evaluate with L2 relative error and time-step visualizations
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
from opifex.data.loaders import create_burgers_loader
from opifex.neural.operators.fno.base import FourierNeuralOperator


print("=" * 70)
print("Opifex Example: FNO on 1D Burgers Equation")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Configuration

The Burgers equation u_t + u * u_x = nu * u_xx develops shocks whose
steepness depends on viscosity nu. We sample viscosity from a range so
the FNO learns to generalize across different shock profiles.
"""

# %%
RESOLUTION = 64
TIME_STEPS = 5
N_TRAIN = 200
N_TEST = 50
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
MODES = 16
HIDDEN_WIDTH = 32
NUM_LAYERS = 4
VISCOSITY_RANGE = (0.01, 0.1)
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/fno_burgers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}")
print(f"Time steps: {TIME_STEPS}")
print(f"Viscosity range: {VISCOSITY_RANGE}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
print(f"FNO config: modes={MODES}, width={HIDDEN_WIDTH}, layers={NUM_LAYERS}")

# %% [markdown]
"""
## Data Loading with Grain

`create_burgers_loader` generates 1D Burgers equation data with random
initial conditions (Gaussians, sine waves, step functions) and random
viscosity values. Each sample maps an initial condition u(x, 0) to the
solution trajectory u(x, t_1), ..., u(x, t_T).
"""

# %%
print()
print("Generating 1D Burgers equation data...")
train_loader = create_burgers_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    time_steps=TIME_STEPS,
    viscosity_range=VISCOSITY_RANGE,
    dimension="1d",
    shuffle=True,
    seed=SEED,
    worker_count=0,
)

test_loader = create_burgers_loader(
    n_samples=N_TEST,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    time_steps=TIME_STEPS,
    viscosity_range=VISCOSITY_RANGE,
    dimension="1d",
    shuffle=False,
    seed=SEED + 1000,
    worker_count=0,
)

# Collect Grain batches into arrays
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

# FNO expects (batch, channels, *spatial)
# Input: (N, resolution) -> (N, 1, resolution)
# Output: (N, time_steps, resolution) already correct (time_steps = channels)
X_train = X_train[:, np.newaxis, :]
X_test = X_test[:, np.newaxis, :]

print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test data:     X={X_test.shape}, Y={Y_test.shape}")
print(f"Input:  initial condition u(x,0)  -> {X_train.shape[1]} channel(s)")
print(f"Output: solution u(x,t_1..t_T)    -> {Y_train.shape[1]} channel(s)")

# %% [markdown]
"""
## Model Creation

For 1D Burgers, the FNO maps 1 input channel (initial condition) to
`time_steps` output channels (solution at each future time). The FNO
automatically handles 1D spatial data when given `(batch, channels, resolution)`
tensors.
"""

# %%
print()
print("Creating FNO model...")
model = FourierNeuralOperator(
    in_channels=1,
    out_channels=TIME_STEPS,
    hidden_channels=HIDDEN_WIDTH,
    modes=MODES,
    num_layers=NUM_LAYERS,
    rngs=nnx.Rngs(SEED),
)

params = nnx.state(model, nnx.Param)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print("Model: FourierNeuralOperator (1D)")
print("  Input channels: 1 (initial condition)")
print(f"  Output channels: {TIME_STEPS} (solution at each time step)")
print(f"  Fourier modes: {MODES}, Hidden width: {HIDDEN_WIDTH}, Layers: {NUM_LAYERS}")
print(f"  Total parameters: {param_count:,}")

# %% [markdown]
"""
## Training with Opifex Trainer

The `Trainer.fit()` method handles the full training loop: JIT compilation,
batching, validation, and progress logging. Loss is MSE between predicted
and true solution trajectories.
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

Evaluate the trained FNO on the test set with per-sample and per-time-step
metrics.
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

# Per-time-step errors
per_step_mse = []
for t in range(TIME_STEPS):
    step_mse = float(jnp.mean((predictions[:, t, :] - Y_test_jnp[:, t, :]) ** 2))
    per_step_mse.append(step_mse)

print(f"Test MSE:         {test_mse:.6f}")
print(f"Test Relative L2: {mean_rel_l2:.6f}")
print(f"Min Relative L2:  {float(jnp.min(per_sample_rel_l2)):.6f}")
print(f"Max Relative L2:  {float(jnp.max(per_sample_rel_l2)):.6f}")
print()
print("Per-time-step MSE:")
for t, mse in enumerate(per_step_mse):
    print(f"  t_{t + 1}: {mse:.6f}")

# %% [markdown]
"""
## Visualization

Create visualizations showing sample predictions and error analysis.
"""

# %%
print()
print("Generating visualizations...")

x_grid = np.linspace(-1, 1, RESOLUTION)

# --- Sample predictions ---
n_vis = min(4, len(X_test))
fig, axes = plt.subplots(
    n_vis, TIME_STEPS + 1, figsize=(3.5 * (TIME_STEPS + 1), 3 * n_vis)
)
fig.suptitle("FNO 1D Burgers Predictions (Opifex)", fontsize=14, fontweight="bold")

if n_vis == 1:
    axes = axes[np.newaxis, :]

for i in range(n_vis):
    # Initial condition
    axes[i, 0].plot(x_grid, X_test[i, 0], "k-", linewidth=1.5, label="u(x,0)")
    axes[i, 0].set_title("Initial Condition" if i == 0 else "")
    axes[i, 0].set_ylabel(f"Sample {i}")
    axes[i, 0].grid(True, alpha=0.3)
    if i == 0:
        axes[i, 0].legend(fontsize=8)

    # Predicted vs ground truth at each time step
    for t in range(TIME_STEPS):
        axes[i, t + 1].plot(
            x_grid, Y_test[i, t], "b-", linewidth=1.5, alpha=0.8, label="Truth"
        )
        axes[i, t + 1].plot(
            x_grid,
            np.array(predictions[i, t]),
            "r--",
            linewidth=1.5,
            alpha=0.8,
            label="FNO",
        )
        if i == 0:
            axes[i, t + 1].set_title(f"t = t_{t + 1}")
            axes[i, t + 1].legend(fontsize=8)
        axes[i, t + 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Sample predictions saved to {OUTPUT_DIR / 'predictions.png'}")

# --- Error analysis ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("FNO Burgers Error Analysis", fontsize=14, fontweight="bold")

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

axes[2].bar(
    range(1, TIME_STEPS + 1),
    per_step_mse,
    color="mediumpurple",
    edgecolor="black",
    alpha=0.7,
)
axes[2].set_title("MSE per Time Step")
axes[2].set_xlabel("Time Step")
axes[2].set_ylabel("MSE")
axes[2].set_xticks(range(1, TIME_STEPS + 1))
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Error analysis saved to {OUTPUT_DIR / 'error_analysis.png'}")

# %% [markdown]
"""
## Results Summary

After running this example you should observe:
- Decreasing training and validation loss over 15 epochs
- Reasonable L2 relative error on test Burgers solutions
- Prediction quality degrades slightly for later time steps (error accumulation)
- Shock structures in Burgers solutions are captured by the FNO

## Next Steps

- Increase resolution and training epochs for sharper shock resolution
- Try 2D Burgers with `dimension="2d"` and `GridEmbedding2D`
- Compare against PINO (physics-informed neural operator) which adds PDE loss
- Experiment with different viscosity ranges to test generalization
- Try `TensorizedFourierNeuralOperator` for parameter-efficient training

### Related Examples

- [FNO on Darcy Flow](fno-darcy.md) — 2D elliptic PDE with grid embedding
- [PINO on Navier-Stokes](pino-navier-stokes.md) — Physics-informed operator
- [Burgers PINN](../pinns/burgers.md) — Solve Burgers with physics-informed neural networks
"""

# %%
print()
print("=" * 70)
print(f"FNO Burgers example completed in {training_time:.1f}s")
print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)
