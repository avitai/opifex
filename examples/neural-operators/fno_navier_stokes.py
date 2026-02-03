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
# # FNO on 2D Navier-Stokes Equations
#
# | Property      | Value                                    |
# |---------------|------------------------------------------|
# | Level         | Intermediate                             |
# | Runtime       | ~3 min (CPU) / ~30 sec (GPU)             |
# | Memory        | ~3 GB                                    |
# | Prerequisites | JAX, Flax NNX, FNO basics, CFD concepts  |
#
# ## Overview
#
# Train a Fourier Neural Operator (FNO) to learn the solution operator for the
# 2D incompressible Navier-Stokes equations:
#
#     du/dt + (u*nabla)u = -nabla(p)/rho + nu*laplacian(u)
#     div(u) = 0  (incompressibility)
#
# where u = (u, v) is the velocity field, p is pressure, rho is density, and nu is
# kinematic viscosity. The Reynolds number Re = UL/nu characterizes the flow regime.
#
# This example demonstrates:
#
# - **FNO for CFD** — mapping initial velocity to future time steps
# - **Taylor-Green vortex** — analytically incompressible initial conditions
# - **Reynolds number variation** — training across different flow regimes
# - **Velocity field prediction** — 2-channel input/output for (u, v) components
#
# ## Learning Goals
#
# 1. Understand FNO for time-dependent PDEs
# 2. Work with multi-channel velocity fields
# 3. Use the Navier-Stokes data loader
# 4. Evaluate prediction accuracy across time steps
# 5. Visualize 2D velocity fields and vorticity

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
import optax
from flax import nnx


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.data.loaders import create_navier_stokes_loader
from opifex.neural.operators.fno.base import FourierNeuralOperator


print("=" * 70)
print("Opifex Example: FNO on 2D Navier-Stokes Equations")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## Configuration

# %%
RESOLUTION = 32  # Spatial grid resolution
TIME_STEPS = 5  # Number of future time steps to predict
N_TRAIN = 100  # Training samples
N_TEST = 30  # Test samples
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
MODES = 12  # Fourier modes
HIDDEN_WIDTH = 32
NUM_LAYERS = 4
REYNOLDS_RANGE = (100.0, 500.0)  # Reynolds number range
TIME_RANGE = (0.0, 1.0)  # Time interval
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/fno_navier_stokes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Time steps: {TIME_STEPS}")
print(f"Reynolds number range: {REYNOLDS_RANGE}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"FNO config: modes={MODES}, width={HIDDEN_WIDTH}, layers={NUM_LAYERS}")

# %% [markdown]
# ## Data Loading
#
# The Navier-Stokes data loader generates solutions starting from Taylor-Green
# vortex initial conditions with varying Reynolds numbers.
#
# - **Input**: Initial velocity field (2, resolution, resolution) = (u, v)
# - **Output**: Future velocity (time_steps, 2, resolution, resolution)

# %%
print()
print("Generating Navier-Stokes data...")
print("  (Using Taylor-Green vortex initial conditions)")

train_loader = create_navier_stokes_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    time_steps=TIME_STEPS,
    reynolds_range=REYNOLDS_RANGE,
    time_range=TIME_RANGE,
    shuffle=True,
    seed=SEED,
    worker_count=0,
)

test_loader = create_navier_stokes_loader(
    n_samples=N_TEST,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    time_steps=TIME_STEPS,
    reynolds_range=REYNOLDS_RANGE,
    time_range=TIME_RANGE,
    shuffle=False,
    seed=SEED + 1000,
    worker_count=0,
)

# Collect all batches
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

print(f"Training: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test:     X={X_test.shape}, Y={Y_test.shape}")
print("  X = (batch, 2=[u,v], res, res) = initial velocity")
print("  Y = (batch, time_steps, 2=[u,v], res, res) = future velocity")

# %% [markdown]
# ## Model Creation
#
# FNO maps the 2-channel initial velocity field to future time step velocities.
# We flatten the output to (time_steps * 2) channels and reshape after.

# %%
print()
print("Creating FNO model...")

IN_CHANNELS = 2  # (u, v) velocity components
OUT_CHANNELS = TIME_STEPS * 2  # (time_steps, 2) flattened

model = FourierNeuralOperator(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    hidden_channels=HIDDEN_WIDTH,
    modes=MODES,
    num_layers=NUM_LAYERS,
    rngs=nnx.Rngs(SEED),
)

params = nnx.state(model, nnx.Param)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Model parameters: {param_count:,}")

# %% [markdown]
# ## Training

# %%
print()
print("Setting up training...")

optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)


def loss_fn(model, x, y):
    """MSE loss for velocity field prediction."""
    y_pred = model(x)  # (batch, time_steps*2, res, res)
    # Reshape to (batch, time_steps, 2, res, res)
    batch_size = y_pred.shape[0]
    y_pred = y_pred.reshape(batch_size, TIME_STEPS, 2, RESOLUTION, RESOLUTION)
    return jnp.mean((y_pred - y) ** 2)


@nnx.jit
def train_step(model, optimizer, x_batch, y_batch):
    """Single training step."""
    loss, grads = nnx.value_and_grad(loss_fn)(model, x_batch, y_batch)
    optimizer.update(model, grads)
    return loss


# %%
print("Starting training...")
print()

X_train_jnp = jnp.array(X_train)
Y_train_jnp = jnp.array(Y_train)
X_test_jnp = jnp.array(X_test)
Y_test_jnp = jnp.array(Y_test)

n_samples = X_train_jnp.shape[0]
n_batches = n_samples // BATCH_SIZE

train_losses = []

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0

    perm = jax.random.permutation(jax.random.PRNGKey(epoch), n_samples)
    X_shuffled = X_train_jnp[perm]
    Y_shuffled = Y_train_jnp[perm]

    for i in range(n_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        x_batch = X_shuffled[start_idx:end_idx]
        y_batch = Y_shuffled[start_idx:end_idx]

        loss = train_step(model, optimizer, x_batch, y_batch)
        epoch_loss += float(loss)

    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS}: Loss = {avg_loss:.6f}")

training_time = time.time() - start_time
print()
print(f"Training completed in {training_time:.1f}s")

# %% [markdown]
# ## Evaluation

# %%
print()
print("Running evaluation...")

# Predict on test set
predictions = model(X_test_jnp)  # (batch, time_steps*2, res, res)
predictions = predictions.reshape(-1, TIME_STEPS, 2, RESOLUTION, RESOLUTION)

# Compute metrics
test_mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))

# Relative L2 error per sample
pred_flat = predictions.reshape(predictions.shape[0], -1)
true_flat = Y_test_jnp.reshape(Y_test_jnp.shape[0], -1)
rel_l2 = jnp.linalg.norm(pred_flat - true_flat, axis=1) / jnp.linalg.norm(
    true_flat, axis=1
)
mean_rel_l2 = float(jnp.mean(rel_l2))

print(f"Test MSE:         {test_mse:.6f}")
print(f"Test Relative L2: {mean_rel_l2:.6f}")

# Per-time-step and per-component errors
print()
print("Per-time-step, per-component MSE:")
for t in range(TIME_STEPS):
    u_mse = float(jnp.mean((predictions[:, t, 0] - Y_test_jnp[:, t, 0]) ** 2))
    v_mse = float(jnp.mean((predictions[:, t, 1] - Y_test_jnp[:, t, 1]) ** 2))
    print(f"  t={t + 1}: u-MSE={u_mse:.6f}, v-MSE={v_mse:.6f}")

# %% [markdown]
# ## Visualization

# %%
print()
print("Generating visualizations...")


def compute_vorticity(u, v, dx):
    """Compute vorticity ω = ∂v/∂x - ∂u/∂y using central differences."""
    dv_dx = (jnp.roll(v, -1, axis=0) - jnp.roll(v, 1, axis=0)) / (2 * dx)
    du_dy = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2 * dx)
    return dv_dx - du_dy


dx = 2 * np.pi / RESOLUTION

# --- Sample prediction comparison ---
sample_idx = 0
fig, axes = plt.subplots(3, TIME_STEPS + 1, figsize=(3.5 * (TIME_STEPS + 1), 9))
fig.suptitle(
    "FNO 2D Navier-Stokes Prediction (Opifex)",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)

row_labels = ["u-velocity", "v-velocity", "vorticity"]

for row, label in enumerate(row_labels):
    if row < 2:  # velocity components
        comp = row
        initial = X_test[sample_idx, comp]
        truth = Y_test[sample_idx, :, comp]
        pred = np.array(predictions[sample_idx, :, comp])
        vmin, vmax = -1.5, 1.5
        cmap = "RdBu_r"
    else:  # vorticity
        u_init = X_test[sample_idx, 0]
        v_init = X_test[sample_idx, 1]
        initial = np.array(compute_vorticity(u_init, v_init, dx))
        truth_vort = []
        pred_vort = []
        for t in range(TIME_STEPS):
            truth_vort.append(
                np.array(
                    compute_vorticity(
                        Y_test[sample_idx, t, 0], Y_test[sample_idx, t, 1], dx
                    )
                )
            )
            pred_vort.append(
                np.array(
                    compute_vorticity(
                        predictions[sample_idx, t, 0], predictions[sample_idx, t, 1], dx
                    )
                )
            )
        truth = np.stack(truth_vort)
        pred = np.stack(pred_vort)
        vmin, vmax = -3, 3
        cmap = "coolwarm"

    # Initial condition
    im = axes[row, 0].imshow(initial.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[row, 0].set_title("t=0" if row == 0 else "")
    axes[row, 0].set_ylabel(label)
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])

    # Time steps
    for t in range(TIME_STEPS):
        data = pred[t]

        im = axes[row, t + 1].imshow(
            data.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
        )
        if row == 0:
            axes[row, t + 1].set_title(f"t={t + 1}")
        axes[row, t + 1].set_xticks([])
        axes[row, t + 1].set_yticks([])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Predictions saved to {OUTPUT_DIR / 'predictions.png'}")

# --- Training history ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("FNO Navier-Stokes Training", fontsize=14, fontweight="bold")

axes[0].semilogy(range(1, NUM_EPOCHS + 1), train_losses, "b-", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss (log scale)")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)

# Error distribution
axes[1].hist(np.array(rel_l2), bins=15, alpha=0.7, color="steelblue", edgecolor="black")
axes[1].set_xlabel("Relative L2 Error")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Test Error Distribution")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Training analysis saved to {OUTPUT_DIR / 'training_analysis.png'}")

# %% [markdown]
# ## Results Summary
#
# The FNO successfully learns the Navier-Stokes solution operator, mapping
# initial velocity fields to future time steps. Key observations:
#
# - **Multi-channel handling**: FNO naturally handles (u, v) velocity components
# - **Reynolds number variation**: Model generalizes across different flow regimes
# - **Vorticity preservation**: Predicted flow structures maintain physical features
#
# ## Next Steps
#
# - Increase resolution and training data for better accuracy
# - Add physics-informed loss for PINO on Navier-Stokes
# - Try longer prediction horizons
# - Use spectral convergence analysis for solution quality
#
# ### Related Examples
#
# - [FNO on Darcy Flow](fno-darcy.md) — Elliptic PDE baseline
# - [FNO on Burgers Equation](fno-burgers.md) — 1D time-dependent PDE
# - [PINO on Burgers](pino-burgers.md) — Physics-informed operator learning
# - [Navier-Stokes PINN](../pinns/navier-stokes.md) — Physics-only approach

# %%
print()
print("=" * 70)
print(f"FNO Navier-Stokes example completed in {training_time:.1f}s")
print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)
