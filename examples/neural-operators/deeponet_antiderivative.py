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
# # DeepONet on Antiderivative Operator
#
# | Property      | Value                                    |
# |---------------|------------------------------------------|
# | Level         | Beginner                                 |
# | Runtime       | ~30s (CPU), ~5s (GPU)                    |
# | Memory        | ~500 MB                                  |
# | Prerequisites | JAX, Flax NNX, Neural Operators basics   |
#
# ## Overview
#
# Train a DeepONet to learn the antiderivative operator, the canonical benchmark
# from the original DeepONet paper (Lu et al., 2021). Given a function v(x),
# the operator learns to predict u(x) = ∫₀ˣ v(t) dt.
#
# This example demonstrates:
#
# - **DeepONet architecture** with branch (function encoder) and trunk (location encoder)
# - **Custom training loop** for operators with two distinct inputs
# - **Antiderivative data generation** using Gaussian Random Field (GRF) basis
# - **Zero initial condition constraint** via output transformation
#
# Equivalent to DeepXDE's `antiderivative_aligned.py` example,
# reimplemented using Opifex APIs.
#
# ## Learning Goals
#
# 1. Understand branch-trunk DeepONet architecture
# 2. Generate synthetic operator learning data
# 3. Implement custom training loop for multi-input operators
# 4. Apply physics constraints via output transformations

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

from opifex.neural.operators.deeponet import DeepONet


# %% [markdown]
# ## Configuration
#
# The antiderivative operator maps v(x) → u(x) where du/dx = v(x) and u(0) = 0.
# We generate input functions v using a Gaussian Random Field (GRF) basis
# with random coefficients, ensuring diverse function shapes.

# %%
N_SENSORS = 50  # Number of sensor points for input function
N_TRAIN = 1000  # Training samples
N_TEST = 200  # Test samples
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
LATENT_DIM = 64  # Shared dimension for branch/trunk outputs

SEED = 42
OUTPUT_DIR = Path("docs/assets/examples/deeponet_antiderivative")

# %% [markdown]
# ## Data Generation
#
# Generate input functions v(x) using a truncated Fourier series with random
# coefficients. For each v, compute the antiderivative u(x) = ∫₀ˣ v(t) dt
# using cumulative trapezoidal integration.


# %%
def generate_grf_function(x: np.ndarray, n_modes: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a smooth random function using Gaussian Random Field basis.

    Uses sine basis with decaying random coefficients to ensure smoothness.
    """
    coeffs = rng.standard_normal(n_modes)
    # Decay coefficients for smoothness (higher modes contribute less)
    decay = 1.0 / (np.arange(1, n_modes + 1) ** 0.5)
    coeffs = coeffs * decay

    # Sum sine basis functions
    v = np.zeros_like(x)
    for k in range(n_modes):
        v += coeffs[k] * np.sin((k + 1) * np.pi * x)
    return v


def compute_antiderivative(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute antiderivative u(x) = ∫₀ˣ v(t) dt using trapezoidal rule."""
    dx = x[1] - x[0]
    u = np.zeros_like(v)
    u[1:] = np.cumsum(0.5 * (v[:-1] + v[1:])) * dx
    return u


def generate_dataset(
    n_samples: int, n_sensors: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate antiderivative operator dataset.

    Returns:
        branch_input: (n_samples, n_sensors) - input function values v(x)
        trunk_input: (n_sensors, 1) - evaluation locations x
        targets: (n_samples, n_sensors) - antiderivative values u(x)
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_sensors)

    branch_inputs = []
    targets = []

    for _ in range(n_samples):
        v = generate_grf_function(x, n_modes=10, rng=rng)
        u = compute_antiderivative(x, v)
        branch_inputs.append(v)
        targets.append(u)

    branch_input = np.stack(branch_inputs, axis=0)  # (n_samples, n_sensors)
    trunk_input = x[:, np.newaxis]  # (n_sensors, 1)
    targets = np.stack(targets, axis=0)  # (n_samples, n_sensors)

    return branch_input, trunk_input, targets


# %% [markdown]
# ## Training and Evaluation Helpers
#
# The zero initial condition u(0) = 0 is enforced by multiplying predictions by
# the x-coordinate. ``train_step`` and ``eval_model`` are JIT-compiled.


# %%
def apply_zero_ic(predictions: jnp.ndarray, x_coords: jnp.ndarray) -> jnp.ndarray:
    """Apply zero initial condition: u(0) = 0 by multiplying by x."""
    # x_coords shape: (n_locations, 1), predictions: (batch, n_locations)
    return predictions * x_coords.squeeze()


@nnx.jit
def train_step(
    model: DeepONet,
    opt: nnx.Optimizer,
    x_branch: jax.Array,
    x_trunk: jax.Array,
    y_target: jax.Array,
) -> jax.Array:
    """Single training step with MSE loss and the zero-IC constraint."""

    def loss_fn(model: DeepONet) -> jax.Array:
        batch_size = x_branch.shape[0]
        trunk_batch = jnp.broadcast_to(x_trunk[None], (batch_size, *x_trunk.shape))
        y_pred = model(x_branch, trunk_batch)
        y_pred = apply_zero_ic(y_pred, x_trunk)
        return jnp.mean((y_pred - y_target) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


@nnx.jit
def eval_model(model: DeepONet, x_branch: jax.Array, x_trunk: jax.Array) -> jax.Array:
    """Evaluate model with the zero-IC constraint."""
    batch_size = x_branch.shape[0]
    trunk_batch = jnp.broadcast_to(x_trunk[None], (batch_size, *x_trunk.shape))
    return apply_zero_ic(model(x_branch, trunk_batch), x_trunk)


# %% [markdown]
# ## Run the Example
#
# `main()` generates the dataset, builds and trains the DeepONet, evaluates on
# the test split, saves the figures, and returns a small dict of finite metrics.


# %%
def main() -> dict[str, float | int]:
    """Train and evaluate a DeepONet on the antiderivative operator."""
    print("=" * 70)
    print("Opifex Example: DeepONet on Antiderivative Operator")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Sensors: {N_SENSORS}")
    print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}, Latent dim: {LATENT_DIM}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data generation ---
    print()
    print("Generating antiderivative dataset...")
    X_branch_train, trunk_coords, Y_train = generate_dataset(N_TRAIN, N_SENSORS, SEED)
    X_branch_test, _, Y_test = generate_dataset(N_TEST, N_SENSORS, SEED + 1000)

    print(f"Training data: branch={X_branch_train.shape}, trunk={trunk_coords.shape}")
    print(f"Training targets: {Y_train.shape}")
    print(f"Test data: branch={X_branch_test.shape}, targets={Y_test.shape}")

    # --- Model creation ---
    print()
    print("Creating DeepONet model...")
    model = DeepONet(
        branch_sizes=[N_SENSORS, 128, 128, LATENT_DIM],  # v(x) → latent
        trunk_sizes=[1, 128, 128, LATENT_DIM],  # x → latent
        activation="tanh",
        rngs=nnx.Rngs(SEED),
    )

    params = nnx.state(model, nnx.Param)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("Model: DeepONet")
    print(f"  Branch network: {N_SENSORS} → 128 → 128 → {LATENT_DIM}")
    print(f"  Trunk network: 1 → 128 → 128 → {LATENT_DIM}")
    print(f"  Total parameters: {param_count:,}")

    # --- Training setup ---
    X_branch_train_jax = jnp.array(X_branch_train)
    X_branch_test_jax = jnp.array(X_branch_test)
    trunk_jax = jnp.array(trunk_coords)  # shared across all samples
    Y_train_jax = jnp.array(Y_train)
    Y_test_jax = jnp.array(Y_test)

    opt = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    print(f"Optimizer: Adam (lr={LEARNING_RATE})")

    # --- Training loop ---
    print()
    print("Starting training...")
    start_time = time.time()

    n_batches = N_TRAIN // BATCH_SIZE
    train_losses: list[float] = []
    val_losses: list[float] = []

    rng = np.random.default_rng(SEED)
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        perm = rng.permutation(N_TRAIN)
        X_shuffled = X_branch_train_jax[perm]
        Y_shuffled = Y_train_jax[perm]

        for i in range(n_batches):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            loss = train_step(
                model, opt, X_shuffled[start_idx:end_idx], trunk_jax, Y_shuffled[start_idx:end_idx]
            )
            epoch_losses.append(float(loss))

        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        val_pred = eval_model(model, X_branch_test_jax, trunk_jax)
        val_loss = float(jnp.mean((val_pred - Y_test_jax) ** 2))
        val_losses.append(val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{NUM_EPOCHS}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss:   {val_losses[-1]:.6f}")

    # --- Evaluation ---
    print()
    print("Running evaluation...")
    predictions = eval_model(model, X_branch_test_jax, trunk_jax)

    test_mse = float(jnp.mean((predictions - Y_test_jax) ** 2))
    pred_diff = predictions - Y_test_jax
    per_sample_rel_l2 = jnp.linalg.norm(pred_diff, axis=1) / jnp.linalg.norm(Y_test_jax, axis=1)
    mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))

    print(f"Test MSE:         {test_mse:.6f}")
    print(f"Test Relative L2: {mean_rel_l2:.6f}")
    print(f"Min Relative L2:  {float(jnp.min(per_sample_rel_l2)):.6f}")
    print(f"Max Relative L2:  {float(jnp.max(per_sample_rel_l2)):.6f}")

    # --- Visualization: sample predictions ---
    print()
    print("Generating visualizations...")
    x_grid = np.linspace(0, 1, N_SENSORS)

    n_vis = 4
    fig, axes = plt.subplots(n_vis, 3, figsize=(12, 3 * n_vis))
    fig.suptitle("DeepONet Antiderivative Predictions (Opifex)", fontsize=14, fontweight="bold")

    for i in range(n_vis):
        axes[i, 0].plot(x_grid, X_branch_test[i], "b-", linewidth=1.5)
        axes[i, 0].set_title("Input v(x)" if i == 0 else "")
        axes[i, 0].set_ylabel(f"Sample {i}")
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(x_grid, Y_test[i], "b-", linewidth=1.5, label="Truth")
        axes[i, 1].plot(x_grid, np.array(predictions[i]), "r--", linewidth=1.5, label="DeepONet")
        axes[i, 1].set_title("Antiderivative u(x)" if i == 0 else "")
        if i == 0:
            axes[i, 1].legend(fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)

        error = np.array(predictions[i]) - Y_test[i]
        axes[i, 2].plot(x_grid, error, "k-", linewidth=1.0)
        axes[i, 2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[i, 2].set_title("Error" if i == 0 else "")
        axes[i, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample predictions saved to {OUTPUT_DIR / 'predictions.png'}")

    # --- Visualization: training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("DeepONet Training Progress", fontsize=14, fontweight="bold")

    epochs_arr = np.arange(1, NUM_EPOCHS + 1)
    axes[0].semilogy(epochs_arr, train_losses, "b-", linewidth=1.5, label="Train")
    axes[0].semilogy(epochs_arr, val_losses, "r-", linewidth=1.5, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (log scale)")
    axes[0].set_title("Training Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    per_sample_errors = np.array(per_sample_rel_l2)
    axes[1].hist(per_sample_errors, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Relative L2 Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Error Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {OUTPUT_DIR / 'training.png'}")

    print()
    print("=" * 70)
    print(f"DeepONet Antiderivative example completed in {training_time:.1f}s")
    print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return {
        "test_mse": test_mse,
        "test_rel_l2": mean_rel_l2,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "num_parameters": int(param_count),
        "training_time_s": training_time,
    }


# %% [markdown]
# ## Results Summary
#
# The DeepONet learns to approximate the antiderivative operator with low error.
# Key observations:
# - The zero IC constraint (multiplying by x) ensures u(0) = 0
# - Smooth GRF-based input functions are well-captured by the learned operator
# - Error is typically largest near x=1 where the integral accumulates
#
# ## Next Steps
#
# - Try different input function distributions (step functions, polynomials)
# - Experiment with physics-informed loss (adding du/dx = v constraint)
# - Scale to higher-dimensional problems
# - Compare against `FourierEnhancedDeepONet` for spectral input functions
#
# ### Related Examples
#
# - [DeepONet on Darcy Flow](deeponet-darcy.md) — 2D operator learning
# - [FNO on Burgers Equation](fno-burgers.md) — Temporal evolution operator
# - [Operator Comparison Tour](operator-tour.md) — Compare all operators

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
