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
# DeepONet on Darcy Flow

| Property      | Value                                    |
|---------------|------------------------------------------|
| Level         | Intermediate                             |
| Runtime       | ~2 min (CPU/GPU)                         |
| Memory        | ~1 GB                                    |
| Prerequisites | JAX, Flax NNX, Neural Operators basics   |

## Overview

Train a Deep Operator Network (DeepONet) to learn the Darcy flow operator,
which maps permeability coefficient fields to pressure solutions. Unlike FNO
which operates on grids, DeepONet uses a branch-trunk architecture:

- **Branch network**: Encodes the input function (permeability) at fixed sensors
- **Trunk network**: Encodes evaluation locations (query coordinates)
- **Output**: Dot product of branch and trunk embeddings

This makes DeepONet resolution-independent -- once trained, it can be queried
at arbitrary spatial locations.

This example uses the standard operator-learning recipe -- ~1000 training
samples, Gaussian input/output normalization, and the relative-L2 objective --
to reach a low relative L2 error on Darcy flow.
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import optax
from flax import nnx


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.deeponet import DeepONet


# %% [markdown]
"""
## Configuration
"""

# %%
# Data configuration
RESOLUTION = 32
N_TRAIN = 1000
N_TEST = 100
N_SENSORS = RESOLUTION * RESOLUTION  # Flatten grid as sensor values
LOCATION_DIM = 2  # (x, y) coordinates

# Model configuration
LATENT_DIM = 128
BRANCH_HIDDEN = [256, 256]
TRUNK_HIDDEN = [128, 128, 128]

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3
EVAL_BATCH_SIZE = 128  # Batch the test forward pass to bound memory use
SEED = 42

ASSETS_DIR = Path("docs/assets/examples/deeponet_darcy")


# %% [markdown]
"""
## Loss and Inference Helpers

These pure helpers are shared by training and evaluation: ``relative_l2_loss``
is the operator-learning objective (mirroring ``Trainer``), and
``predict_in_batches`` runs the test forward pass in chunks to bound memory.
"""


# %%
def relative_l2_loss(y_pred: jax.Array, y_target: jax.Array) -> jax.Array:
    """Mean per-sample relative L2 error ``||y_pred - y||_2 / ||y||_2``.

    Mirrors the operator-learning objective used by ``Trainer`` so DeepONet's
    custom loop optimizes the same scale-invariant criterion.

    Args:
        y_pred: Model prediction of shape (batch, n_locations).
        y_target: Target of shape (batch, n_locations).

    Returns:
        Scalar mean relative L2 loss.
    """
    numerator = jnp.linalg.norm(y_pred - y_target, axis=-1)
    denominator = jnp.linalg.norm(y_target, axis=-1) + 1e-8
    return jnp.mean(numerator / denominator)


def predict_in_batches(
    model: DeepONet,
    x_branch: jax.Array,
    trunk: jax.Array,
    batch_size: int = EVAL_BATCH_SIZE,
) -> jax.Array:
    """Run DeepONet over the branch inputs in batches to bound memory use.

    Args:
        model: Trained DeepONet.
        x_branch: Normalized branch inputs of shape (n_samples, n_sensors).
        trunk: Trunk coordinates of shape (n_locations, 2).
        batch_size: Maximum number of samples per forward pass.

    Returns:
        Predictions of shape (n_samples, n_locations) in normalized space.
    """
    outputs = []
    for i in range(0, x_branch.shape[0], batch_size):
        x_chunk = x_branch[i : i + batch_size]
        trunk_chunk = jnp.broadcast_to(trunk[None], (x_chunk.shape[0], *trunk.shape))
        outputs.append(model(x_chunk, trunk_chunk))
    return jnp.concatenate(outputs, axis=0)


@nnx.jit
def train_step(
    model: DeepONet,
    opt: nnx.Optimizer,
    x_branch: jax.Array,
    trunk: jax.Array,
    y_target: jax.Array,
) -> jax.Array:
    """Single relative-L2 training step for DeepONet."""

    def loss_fn(model: DeepONet) -> jax.Array:
        # Expand trunk for batch: (n_locations, 2) -> (batch, n_locations, 2)
        batch_size = x_branch.shape[0]
        trunk_batch = jnp.broadcast_to(trunk[None], (batch_size, *trunk.shape))
        y_pred = model(x_branch, trunk_batch)  # (batch, n_locations)
        return relative_l2_loss(y_pred, y_target)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


# %% [markdown]
"""
## Run the Example

`main()` loads the Darcy data, builds and trains the DeepONet, evaluates on the
test split, saves the figures, and returns a small dict of finite metrics.
"""


# %%
def main() -> dict[str, float | int]:
    """Train and evaluate a DeepONet on the Darcy flow operator."""
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
    print(f"Sensors: {N_SENSORS}, Latent dim: {LATENT_DIM}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Darcy flow data via datarax ---
    # `create_darcy_loader` returns a frozen `PDELoaders` with `.train`/`.val`
    # datarax pipelines. Batches are channels-first dicts with "input"/"output"
    # of shape (b, 1, res, res) -- no add-channel reshape is needed.
    n_samples = N_TRAIN + N_TEST
    loaders = create_darcy_loader(
        n_samples=n_samples,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        val_fraction=N_TEST / n_samples,
        seed=SEED,
    )

    def _collect(pipeline) -> tuple[np.ndarray, np.ndarray]:
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    X_train_grid, Y_train_grid = _collect(loaders.train)  # (N, 1, res, res)
    X_test_grid, Y_test_grid = _collect(loaders.val)  # (N, 1, res, res)

    # Drop the channel axis so the viz/grid views are (N, res, res) as before.
    X_train_grid = X_train_grid[:, 0]
    Y_train_grid = Y_train_grid[:, 0]
    X_test_grid = X_test_grid[:, 0]
    Y_test_grid = Y_test_grid[:, 0]

    # Flatten grids for DeepONet branch input
    X_train_branch = X_train_grid.reshape(X_train_grid.shape[0], -1)  # (N, n_sensors)
    X_test_branch = X_test_grid.reshape(X_test_grid.shape[0], -1)

    # Create coordinate grid for trunk input
    x_coords = np.linspace(0, 1, RESOLUTION)
    y_coords = np.linspace(0, 1, RESOLUTION)
    xx, yy = np.meshgrid(x_coords, y_coords)
    trunk_coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (n_locations, 2)

    # Flatten target grids to match trunk locations
    Y_train_flat = Y_train_grid.reshape(Y_train_grid.shape[0], -1)  # (N, n_locations)
    Y_test_flat = Y_test_grid.reshape(Y_test_grid.shape[0], -1)

    print(f"Branch input: {X_train_branch.shape}")
    print(f"Trunk input:  {trunk_coords.shape}")
    print(f"Target:       {Y_train_flat.shape}")

    # --- Normalization ---
    x_mean, x_std = X_train_branch.mean(), X_train_branch.std()
    y_mean, y_std = Y_train_flat.mean(), Y_train_flat.std()

    X_train_branch_n = (X_train_branch - x_mean) / x_std
    X_test_branch_n = (X_test_branch - x_mean) / x_std
    Y_train_flat_n = (Y_train_flat - y_mean) / y_std

    print(f"Input mean/std:  {x_mean:.4f} / {x_std:.4f}")
    print(f"Output mean/std: {y_mean:.6f} / {y_std:.6f}")

    # --- Create DeepONet model ---
    branch_sizes = [N_SENSORS, *BRANCH_HIDDEN, LATENT_DIM]
    trunk_sizes = [LOCATION_DIM, *TRUNK_HIDDEN, LATENT_DIM]

    model = DeepONet(
        branch_sizes=branch_sizes,
        trunk_sizes=trunk_sizes,
        activation="gelu",
        rngs=nnx.Rngs(SEED),
    )

    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(model)))
    print(f"Model: DeepONet (latent_dim={LATENT_DIM})")
    print(f"Branch: {branch_sizes}")
    print(f"Trunk:  {trunk_sizes}")
    print(f"Total parameters: {n_params:,}")

    # --- Training setup ---
    # Adam with a short warmup then cosine decay. The warmup helps the
    # branch/trunk escape the early plateau; the decay refines the solution late.
    steps_per_epoch = N_TRAIN // BATCH_SIZE
    total_steps = NUM_EPOCHS * steps_per_epoch
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=LEARNING_RATE / 10,
        peak_value=LEARNING_RATE,
        warmup_steps=10 * steps_per_epoch,
        decay_steps=total_steps,
        end_value=LEARNING_RATE / 50,
    )
    opt = nnx.Optimizer(model, optax.adam(lr_schedule), wrt=nnx.Param)
    print(f"Optimizer: Adam + warmup-cosine (peak lr={LEARNING_RATE}), loss: relative L2")

    trunk_jax = jnp.array(trunk_coords)  # shared across all samples
    X_train_jax = jnp.array(X_train_branch_n)
    Y_train_jax = jnp.array(Y_train_flat_n)
    X_test_jax = jnp.array(X_test_branch_n)
    Y_test_jax = jnp.array(Y_test_flat)

    # --- Training loop ---
    print(f"\nStarting training ({NUM_EPOCHS} epochs)...")
    t0 = time.time()

    key = jax.random.PRNGKey(SEED)
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(NUM_EPOCHS):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, X_train_jax.shape[0])
        X_shuffled = X_train_jax[perm]
        Y_shuffled = Y_train_jax[perm]

        epoch_losses = []
        n_batches = X_train_jax.shape[0] // BATCH_SIZE
        for i in range(n_batches):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            loss = train_step(model, opt, X_shuffled[start:end], trunk_jax, Y_shuffled[start:end])
            epoch_losses.append(float(loss))

        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        val_pred = predict_in_batches(model, X_test_jax, trunk_jax) * y_std + y_mean
        val_loss = float(relative_l2_loss(val_pred, Y_test_jax))
        val_losses.append(val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{NUM_EPOCHS}: "
                f"train_rel_l2={train_loss:.6f}, val_rel_l2={val_loss:.6f}"
            )

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Final train rel-L2: {train_losses[-1]:.6e}")
    print(f"Final val rel-L2:   {val_losses[-1]:.6e}")

    # --- Evaluation ---
    predictions = predict_in_batches(model, X_test_jax, trunk_jax) * y_std + y_mean
    test_mse = float(jnp.mean((predictions - Y_test_jax) ** 2))

    per_sample_l2 = jnp.sqrt(jnp.sum((predictions - Y_test_jax) ** 2, axis=-1))
    per_sample_norm = jnp.sqrt(jnp.sum(Y_test_jax**2, axis=-1))
    relative_l2 = per_sample_l2 / (per_sample_norm + 1e-8)
    mean_rel_l2 = float(jnp.mean(relative_l2))

    print(f"\nTest MSE:         {test_mse:.6e}")
    print(f"Test Relative L2: {mean_rel_l2:.6f}")
    print(f"Min Relative L2:  {float(jnp.min(relative_l2)):.6f}")
    print(f"Max Relative L2:  {float(jnp.max(relative_l2)):.6f}")

    # --- Visualization: sample predictions ---
    pred_grid = np.array(predictions).reshape(-1, RESOLUTION, RESOLUTION)
    truth_grid = Y_test_grid

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row in range(3):
        idx = row
        vmin = float(truth_grid[idx].min())
        vmax = float(truth_grid[idx].max())

        axes[row, 0].imshow(X_test_grid[idx], cmap="viridis")
        axes[row, 0].set_title("Input (Permeability)" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(truth_grid[idx], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title("Ground Truth" if row == 0 else "")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_grid[idx], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[row, 2].set_title("DeepONet Prediction" if row == 0 else "")
        axes[row, 2].axis("off")

        err = np.abs(truth_grid[idx] - pred_grid[idx])
        im = axes[row, 3].imshow(err, cmap="hot")
        axes[row, 3].set_title("Absolute Error" if row == 0 else "")
        axes[row, 3].axis("off")
        fig.colorbar(im, ax=axes[row, 3], shrink=0.8)

    plt.suptitle("DeepONet on Darcy Flow: Sample Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions saved to {ASSETS_DIR / 'predictions.png'}")

    # --- Visualization: error analysis ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(np.array(relative_l2), bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Relative L2 Error")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error Distribution")
    axes[0].axvline(mean_rel_l2, color="red", linestyle="--", label=f"Mean: {mean_rel_l2:.4f}")
    axes[0].legend()

    axes[1].semilogy(
        range(1, NUM_EPOCHS + 1), train_losses, label="Train rel-L2", color="steelblue"
    )
    axes[1].semilogy(range(1, NUM_EPOCHS + 1), val_losses, label="Val rel-L2", color="coral")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative L2 Loss")
    axes[1].set_title("Training Progress")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("DeepONet Error Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Error analysis saved to {ASSETS_DIR / 'error_analysis.png'}")

    # --- Visualization: branch-trunk embeddings ---
    branch_out = model.get_branch_output(X_test_jax[:5])
    trunk_out = model.get_trunk_output(trunk_jax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    branch_np = np.array(branch_out)
    similarity = branch_np @ branch_np.T
    similarity /= np.max(np.abs(similarity))
    im0 = axes[0].imshow(similarity, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Branch Embedding Similarity (5 samples)")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Sample")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    trunk_np = np.array(trunk_out)  # (n_locations, latent_dim)
    trunk_rgb = trunk_np[:, :3]
    trunk_rgb = (trunk_rgb - trunk_rgb.min(axis=0)) / (
        trunk_rgb.max(axis=0) - trunk_rgb.min(axis=0) + 1e-8
    )
    trunk_img = trunk_rgb.reshape(RESOLUTION, RESOLUTION, 3)
    axes[1].imshow(trunk_img)
    axes[1].set_title("Trunk Embedding (first 3 dims as RGB)")
    axes[1].axis("off")

    plt.suptitle("DeepONet Learned Representations", fontsize=14)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "branch_trunk.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Branch-trunk analysis saved to {ASSETS_DIR / 'branch_trunk.png'}")

    print()
    print("=" * 70)
    print(f"DeepONet Darcy Flow example completed in {elapsed:.1f}s")
    print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
    print(f"Results saved to: {ASSETS_DIR}")
    print("=" * 70)

    return {
        "test_mse": test_mse,
        "test_rel_l2": mean_rel_l2,
        "final_train_rel_l2": train_losses[-1],
        "final_val_rel_l2": val_losses[-1],
        "num_parameters": int(n_params),
        "training_time_s": elapsed,
    }


# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
