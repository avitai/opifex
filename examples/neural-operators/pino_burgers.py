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
# # Physics-Informed Neural Operator (PINO) on Burgers Equation
#
# | Property      | Value                                    |
# |---------------|------------------------------------------|
# | Level         | Advanced                                 |
# | Runtime       | ~5 min (CPU) / ~1 min (GPU)              |
# | Memory        | ~2 GB                                    |
# | Prerequisites | JAX, Flax NNX, FNO, PDEs basics          |
#
# ## Overview
#
# Train a Physics-Informed Neural Operator (PINO) on the 1D Burgers equation.
# PINO combines the FNO architecture with physics-informed loss, enabling
# training with reduced data requirements by enforcing PDE constraints.
#
# The Burgers equation is:
#     u_t + u * u_x = nu * u_xx
#
# where u is velocity, nu is viscosity, and subscripts denote partial derivatives.
#
# This example demonstrates:
#
# - **FNO backbone** for operator learning
# - **Physics loss** via finite difference PDE residual computation
# - **Multi-objective training** balancing data and physics losses
# - **Comparison** of data-only vs physics-informed training
#
# Equivalent to `neuraloperator/scripts/train_burgers_pino.py`,
# reimplemented using Opifex APIs.
#
# ## Learning Goals
#
# 1. Understand PINO architecture: FNO backbone + physics loss
# 2. Implement PDE residual computation using finite differences
# 3. Configure multi-objective loss weighting
# 4. Analyze physics loss contribution to training dynamics
# 5. Compare data-only FNO vs physics-informed PINO

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

from opifex.data.loaders import create_burgers_loader
from opifex.neural.operators.fno.base import FourierNeuralOperator


# %% [markdown]
# ## Configuration
#
# PINO uses a physics loss weight to balance data fitting and PDE enforcement.
# Higher physics_weight encourages PDE consistency but may sacrifice data fit.

# %%
RESOLUTION = 64
TIME_STEPS = 5
N_TRAIN = 200
N_TEST = 50
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
MODES = 16
HIDDEN_WIDTH = 32
NUM_LAYERS = 4
VISCOSITY = 0.05  # Fixed viscosity for PINO
VISCOSITY_RANGE = (0.05, 0.05)  # Fixed for physics loss
DATA_WEIGHT = 1.0
PHYSICS_WEIGHT = 0.1  # Weight for PDE residual loss
SEED = 42

# Grid parameters for the physics loss
DOMAIN_LENGTH = 2.0  # x in [-1, 1]
TIME_LENGTH = 1.0  # t in [0, 1]
DX = DOMAIN_LENGTH / RESOLUTION
DT = TIME_LENGTH / TIME_STEPS

OUTPUT_DIR = Path("docs/assets/examples/pino_burgers")


# %% [markdown]
# ## Physics Loss: Burgers Equation Residual
#
# The physics loss computes the PDE residual using finite differences:
#
# Residual = u_t + u * u_x - nu * u_xx
#
# A perfect solution has residual = 0 everywhere.


# %%
def compute_burgers_residual(
    u: jax.Array,
    dx: float,
    dt: float,
    nu: float,
) -> jax.Array:
    """Compute Burgers equation residual using finite differences.

    Args:
        u: Solution tensor of shape (batch, time_steps, resolution)
        dx: Spatial step size
        dt: Time step size
        nu: Viscosity coefficient

    Returns:
        Residual tensor of shape (batch, time_steps-1, resolution-2)
    """
    # Time derivative: (u(t+1) - u(t)) / dt -> (batch, time_steps-1, resolution)
    u_t = (u[:, 1:, :] - u[:, :-1, :]) / dt

    # For spatial derivatives, use u at midpoint in time
    u_mid = 0.5 * (u[:, 1:, :] + u[:, :-1, :])

    # Spatial first derivative (central difference)
    u_x = (u_mid[:, :, 2:] - u_mid[:, :, :-2]) / (2 * dx)

    # Spatial second derivative (central difference)
    u_xx = (u_mid[:, :, 2:] - 2 * u_mid[:, :, 1:-1] + u_mid[:, :, :-2]) / (dx**2)

    # u value at interior points, and u_t trimmed to match
    u_interior = u_mid[:, :, 1:-1]
    u_t_interior = u_t[:, :, 1:-1]

    # Burgers residual: u_t + u * u_x - nu * u_xx = 0
    return u_t_interior + u_interior * u_x - nu * u_xx


def physics_loss(pred: jax.Array, dx: float, dt: float, nu: float) -> jax.Array:
    """Compute mean squared PDE residual loss."""
    residual = compute_burgers_residual(pred, dx, dt, nu)
    return jnp.mean(residual**2)


def pino_loss_fn(
    model: FourierNeuralOperator,
    x: jax.Array,
    y_true: jax.Array,
    dx: float,
    dt: float,
    nu: float,
    data_weight: float,
    physics_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Combined data + physics loss for PINO training.

    The model predicts the full space-time trajectory ``(batch, time_steps,
    resolution)``. Supervision is sparse: an MSE data term anchors the predicted
    initial frame to the input IC and the predicted final frame to the
    final-time target ``y_true`` ``(batch, 1, resolution)``; the physics term
    enforces the Burgers PDE residual across every predicted time step.
    """
    y_pred = model(x)
    ic_loss = jnp.mean((y_pred[:, :1, :] - x) ** 2)
    final_loss = jnp.mean((y_pred[:, -1:, :] - y_true) ** 2)
    data_loss = ic_loss + final_loss
    pde_loss = physics_loss(y_pred, dx, dt, nu)
    total_loss = data_weight * data_loss + physics_weight * pde_loss
    return total_loss, {"data_loss": data_loss, "physics_loss": pde_loss}


# %% [markdown]
# ## Run the Example
#
# `main()` loads the Burgers data, builds the FNO backbone, trains with the
# combined data + physics loss, evaluates the PDE residual on the test set,
# saves the figures, and returns a small dict of finite metrics.


# %%
def main() -> dict[str, float | int]:
    """Train and evaluate a PINO on the 1D Burgers equation."""
    print("=" * 70)
    print("Opifex Example: PINO on 1D Burgers Equation")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Resolution: {RESOLUTION}, Time steps: {TIME_STEPS}, Viscosity: {VISCOSITY}")
    print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
    print(f"FNO config: modes={MODES}, width={HIDDEN_WIDTH}, layers={NUM_LAYERS}")
    print(f"Loss weights: data={DATA_WEIGHT}, physics={PHYSICS_WEIGHT}")
    print(f"Grid: dx={DX:.4f}, dt={DT:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data loading ---
    print()
    print("Generating 1D Burgers data (jit+vmap) and serving via datarax...")
    n_samples = N_TRAIN + N_TEST
    loaders = create_burgers_loader(
        n_samples=n_samples,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        viscosity_range=VISCOSITY_RANGE,
        val_fraction=N_TEST / n_samples,
        seed=SEED,
    )

    # The datarax loader follows the uniform operator contract: input is the
    # initial condition u(x,0) and output is the FINAL-time solution u(x,T),
    # both channels-first (N, 1, resolution). PINO supervises only this sparse
    # final-time target; the physics (PDE residual) loss constrains the full
    # predicted space-time trajectory in between (semi-supervised learning).
    def _collect(pipeline) -> tuple[np.ndarray, np.ndarray]:
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    X_train, Y_train = _collect(loaders.train)
    X_test, Y_test = _collect(loaders.val)

    print(f"Training data: X={X_train.shape}, Y(final-time)={Y_train.shape}")
    print(f"Test data:     X={X_test.shape}, Y(final-time)={Y_test.shape}")

    # --- Model creation ---
    print()
    print("Creating PINO model (FNO backbone)...")
    model = FourierNeuralOperator(
        in_channels=1,
        out_channels=TIME_STEPS,
        hidden_channels=HIDDEN_WIDTH,
        modes=MODES,
        num_layers=NUM_LAYERS,
        spatial_dims=1,
        rngs=nnx.Rngs(SEED),
    )
    params = nnx.state(model, nnx.Param)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")

    # --- Training setup ---
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)

    @nnx.jit
    def train_step(
        model: FourierNeuralOperator,
        optimizer: nnx.Optimizer,
        x_batch: jax.Array,
        y_batch: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Single PINO training step (data + physics loss)."""

        def loss_fn(model: FourierNeuralOperator) -> tuple[jax.Array, dict[str, jax.Array]]:
            return pino_loss_fn(
                model, x_batch, y_batch, DX, DT, VISCOSITY, DATA_WEIGHT, PHYSICS_WEIGHT
            )

        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return loss, aux

    # --- Training loop ---
    print()
    print("Starting PINO training...")
    print(f"Optimizer: Adam (lr={LEARNING_RATE})")

    X_train_jnp = jnp.array(X_train)
    Y_train_jnp = jnp.array(Y_train)
    X_test_jnp = jnp.array(X_test)
    Y_test_jnp = jnp.array(Y_test)

    n_samples = X_train_jnp.shape[0]
    n_batches = n_samples // BATCH_SIZE
    train_history: dict[str, list[float]] = {"total": [], "data": [], "physics": []}

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0

        perm = jax.random.permutation(jax.random.PRNGKey(epoch), n_samples)
        X_shuffled = X_train_jnp[perm]
        Y_shuffled = Y_train_jnp[perm]

        for i in range(n_batches):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            loss, aux = train_step(
                model, optimizer, X_shuffled[start_idx:end_idx], Y_shuffled[start_idx:end_idx]
            )
            epoch_loss += float(loss)
            epoch_data_loss += float(aux["data_loss"])
            epoch_physics_loss += float(aux["physics_loss"])

        avg_loss = epoch_loss / n_batches
        avg_data = epoch_data_loss / n_batches
        avg_physics = epoch_physics_loss / n_batches
        train_history["total"].append(avg_loss)
        train_history["data"].append(avg_data)
        train_history["physics"].append(avg_physics)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{NUM_EPOCHS}: "
                f"Total={avg_loss:.6f}, Data={avg_data:.6f}, Physics={avg_physics:.6f}"
            )

    training_time = time.time() - start_time
    print()
    print(f"Training completed in {training_time:.1f}s")

    # --- Evaluation ---
    print()
    print("Running evaluation...")
    predictions = model(X_test_jnp)  # (N, TIME_STEPS, resolution)
    final_pred = predictions[:, -1:, :]  # predicted u(x, T)

    # Accuracy is measured at the supervised final time; the physics residual is
    # evaluated over the whole predicted trajectory.
    test_mse = float(jnp.mean((final_pred - Y_test_jnp) ** 2))
    pred_diff = (final_pred - Y_test_jnp).reshape(final_pred.shape[0], -1)
    Y_flat = Y_test_jnp.reshape(Y_test_jnp.shape[0], -1)
    per_sample_rel_l2 = jnp.linalg.norm(pred_diff, axis=1) / jnp.linalg.norm(Y_flat, axis=1)
    mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))
    test_physics_loss = float(physics_loss(predictions, DX, DT, VISCOSITY))

    print(f"Test MSE (final time):  {test_mse:.6f}")
    print(f"Test Relative L2:       {mean_rel_l2:.6f}")
    print(f"Test Physics Loss:      {test_physics_loss:.6f}")

    print()
    print("Per-time-step PDE residual (physics consistency of the rollout):")
    step_residual = jnp.mean(
        compute_burgers_residual(predictions, DX, DT, VISCOSITY) ** 2, axis=(0, 2)
    )
    for t in range(int(step_residual.shape[0])):
        print(f"  t_{t + 1}: {float(step_residual[t]):.6e}")

    # --- Visualization: sample predictions ---
    print()
    print("Generating visualizations...")
    x_grid = np.linspace(-1, 1, RESOLUTION)

    n_vis = min(4, len(X_test))
    fig, axes = plt.subplots(n_vis, TIME_STEPS + 1, figsize=(3.5 * (TIME_STEPS + 1), 3 * n_vis))
    fig.suptitle("PINO 1D Burgers Predictions (Opifex)", fontsize=14, fontweight="bold", y=1.02)

    if n_vis == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_vis):
        axes[i, 0].plot(x_grid, X_test[i, 0], "k-", linewidth=1.5, label="u(x,0)")
        axes[i, 0].set_title("Initial Condition" if i == 0 else "")
        axes[i, 0].set_ylabel(f"Sample {i}")
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].legend(fontsize=8)

        # The PINO predicts the full rollout; ground truth is only available
        # (and supervised) at the final time, overlaid on the last frame.
        for t in range(TIME_STEPS):
            axes[i, t + 1].plot(
                x_grid, np.array(predictions[i, t]), "r--", linewidth=1.5, alpha=0.8, label="PINO"
            )
            if t == TIME_STEPS - 1:
                axes[i, t + 1].plot(
                    x_grid, Y_test[i, 0], "b-", linewidth=1.5, alpha=0.8, label="Truth (final)"
                )
            if i == 0:
                axes[i, t + 1].set_title(f"t = t_{t + 1}")
                axes[i, t + 1].legend(fontsize=8)
            axes[i, t + 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions saved to {OUTPUT_DIR / 'predictions.png'}")

    # --- Visualization: training history ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("PINO Training Dynamics", fontsize=14, fontweight="bold")

    epochs_arr = np.arange(1, NUM_EPOCHS + 1)
    axes[0].semilogy(epochs_arr, train_history["total"], "k-", linewidth=2, label="Total")
    axes[0].semilogy(epochs_arr, train_history["data"], "b--", linewidth=1.5, label="Data")
    axes[0].semilogy(epochs_arr, train_history["physics"], "r--", linewidth=1.5, label="Physics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Training Loss Components")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    per_sample_errors = np.array(per_sample_rel_l2)
    axes[1].hist(per_sample_errors, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Relative L2 Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Test Error Distribution")
    axes[1].grid(True, alpha=0.3)

    per_step_residual = np.array(
        jnp.mean(compute_burgers_residual(predictions, DX, DT, VISCOSITY) ** 2, axis=(0, 2))
    )
    axes[2].bar(
        range(1, len(per_step_residual) + 1),
        per_step_residual,
        color="mediumpurple",
        edgecolor="black",
        alpha=0.7,
    )
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Mean Squared PDE Residual")
    axes[2].set_title("Per-Step Physics Consistency")
    axes[2].set_xticks(range(1, len(per_step_residual) + 1))
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training analysis saved to {OUTPUT_DIR / 'training_analysis.png'}")

    print()
    print("=" * 70)
    print(f"PINO Burgers example completed in {training_time:.1f}s")
    print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
    print(f"Test Physics Loss: {test_physics_loss:.6f}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return {
        "test_mse": test_mse,
        "test_rel_l2": mean_rel_l2,
        "test_physics_loss": test_physics_loss,
        "final_total_loss": train_history["total"][-1],
        "num_parameters": int(param_count),
        "training_time_s": training_time,
    }


# %% [markdown]
# ## Results Summary
#
# PINO combines FNO's operator learning capability with physics-informed
# constraints. The physics loss ensures predictions satisfy the Burgers PDE,
# which can:
#
# - Improve generalization with limited training data
# - Produce physically consistent predictions
# - Enable semi-supervised learning with partial observations
#
# ## Next Steps
#
# - Experiment with physics_weight values (0.01, 0.1, 1.0)
# - Compare PINO vs data-only FNO performance
# - Try adaptive loss weighting (SoftAdapt, ReLoBRaLo)
# - Apply to 2D Burgers or Navier-Stokes equations
# - Use spectral differentiation instead of finite differences
#
# ### Related Examples
#
# - [FNO on Burgers Equation](fno-burgers.md) — Data-only FNO baseline
# - [FNO on Darcy Flow](fno-darcy.md) — 2D elliptic PDE
# - [Burgers PINN](../pinns/burgers.md) — Physics-only neural network
# - [TFNO on Darcy Flow](tfno-darcy.md) — Tensorized FNO with compression

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
