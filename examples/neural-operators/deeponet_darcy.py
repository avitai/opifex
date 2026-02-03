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
"""

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Configuration
"""

# %%
# Data configuration
RESOLUTION = 32
N_TRAIN = 200
N_TEST = 50
N_SENSORS = RESOLUTION * RESOLUTION  # Flatten grid as sensor values
LOCATION_DIM = 2  # (x, y) coordinates

# Model configuration
LATENT_DIM = 64
BRANCH_HIDDEN = [256, 128]
TRUNK_HIDDEN = [128, 128]

# Training configuration
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42

ASSETS_DIR = Path("docs/assets/examples/deeponet_darcy")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"Sensors: {N_SENSORS}, Latent dim: {LATENT_DIM}")

# %% [markdown]
"""
## Load Darcy Flow Data

We use the Grain-based Darcy loader and reshape data for DeepONet:
- **Branch input**: Flattened permeability field `(batch, n_sensors)`
- **Trunk input**: Grid coordinate pairs `(n_locations, 2)`
- **Target**: Pressure values at those locations `(batch, n_locations)`
"""

# %%
from opifex.data.loaders import create_darcy_loader


# Load training data
train_loader = create_darcy_loader(
    n_samples=N_TRAIN,
    batch_size=N_TRAIN,
    resolution=RESOLUTION,
    shuffle=True,
    seed=SEED,
    worker_count=0,
)
train_batch = next(iter(train_loader))
X_train_grid = np.array(train_batch["input"])  # (N, res, res)
Y_train_grid = np.array(train_batch["output"])  # (N, res, res)

# Load test data
test_loader = create_darcy_loader(
    n_samples=N_TEST,
    batch_size=N_TEST,
    resolution=RESOLUTION,
    shuffle=False,
    seed=99,
    worker_count=0,
)
test_batch = next(iter(test_loader))
X_test_grid = np.array(test_batch["input"])
Y_test_grid = np.array(test_batch["output"])

# Flatten grids for DeepONet
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

# %% [markdown]
"""
## Create DeepONet Model

The branch and trunk networks are MLPs with matching output dimensions.
Their outputs are combined via dot product to produce the operator output.
"""

# %%
from opifex.neural.operators.deeponet import DeepONet


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

# %% [markdown]
"""
## Training Loop

DeepONet has a different input structure (branch + trunk) than grid-based
operators, so we use a custom training loop with optax directly.
"""

# %%
opt = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)
print(f"Optimizer: Adam (lr={LEARNING_RATE})")

# Convert trunk to JAX array (shared across all samples)
trunk_jax = jnp.array(trunk_coords)

# Convert data to JAX arrays
X_train_jax = jnp.array(X_train_branch)
Y_train_jax = jnp.array(Y_train_flat)


@nnx.jit
def train_step(model, opt, x_branch, y_target):
    """Single training step for DeepONet."""

    def loss_fn(model):
        # Expand trunk for batch: (n_locations, 2) -> (batch, n_locations, 2)
        batch_size = x_branch.shape[0]
        trunk_batch = jnp.broadcast_to(trunk_jax[None], (batch_size, *trunk_jax.shape))
        y_pred = model(x_branch, trunk_batch)  # (batch, n_locations)
        return jnp.mean((y_pred - y_target) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


print(f"\nStarting training ({NUM_EPOCHS} epochs)...")
import time


t0 = time.time()

key = jax.random.PRNGKey(SEED)
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    # Shuffle training data
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, X_train_jax.shape[0])
    X_shuffled = X_train_jax[perm]
    Y_shuffled = Y_train_jax[perm]

    # Mini-batch training
    epoch_losses = []
    n_batches = X_train_jax.shape[0] // BATCH_SIZE
    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        loss = train_step(model, opt, X_shuffled[start:end], Y_shuffled[start:end])
        epoch_losses.append(float(loss))

    train_loss = np.mean(epoch_losses)
    train_losses.append(train_loss)

    # Validation loss on test set
    X_test_jax = jnp.array(X_test_branch)
    Y_test_jax = jnp.array(Y_test_flat)
    trunk_test = jnp.broadcast_to(
        trunk_jax[None], (X_test_jax.shape[0], *trunk_jax.shape)
    )
    test_pred = model(X_test_jax, trunk_test)
    val_loss = float(jnp.mean((test_pred - Y_test_jax) ** 2))
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"  Epoch {epoch + 1:3d}/{NUM_EPOCHS}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )

elapsed = time.time() - t0
print(f"\nTraining completed in {elapsed:.1f}s")
print(f"Final train loss: {train_losses[-1]:.6e}")
print(f"Final val loss:   {val_losses[-1]:.6e}")

# %% [markdown]
"""
## Evaluation
"""

# %%
# Full test evaluation
X_test_jax = jnp.array(X_test_branch)
Y_test_jax = jnp.array(Y_test_flat)
trunk_test = jnp.broadcast_to(trunk_jax[None], (X_test_jax.shape[0], *trunk_jax.shape))
predictions = model(X_test_jax, trunk_test)

test_mse = float(jnp.mean((predictions - Y_test_jax) ** 2))

# Per-sample relative L2 error
per_sample_l2 = jnp.sqrt(jnp.sum((predictions - Y_test_jax) ** 2, axis=-1))
per_sample_norm = jnp.sqrt(jnp.sum(Y_test_jax**2, axis=-1))
relative_l2 = per_sample_l2 / (per_sample_norm + 1e-8)
mean_rel_l2 = float(jnp.mean(relative_l2))

print(f"\nTest MSE:         {test_mse:.6f}")
print(f"Test Relative L2: {mean_rel_l2:.6f}")
print(f"Min Relative L2:  {float(jnp.min(relative_l2)):.6f}")
print(f"Max Relative L2:  {float(jnp.max(relative_l2)):.6f}")

# %% [markdown]
"""
## Visualizations

### Sample Predictions
"""

# %%
import matplotlib as mpl


mpl.use("Agg")
import matplotlib.pyplot as plt


# Reshape predictions back to grid
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

# %% [markdown]
"""
### Error Analysis
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Relative L2 error distribution
axes[0].hist(np.array(relative_l2), bins=20, color="steelblue", edgecolor="white")
axes[0].set_xlabel("Relative L2 Error")
axes[0].set_ylabel("Count")
axes[0].set_title("Error Distribution")
axes[0].axvline(
    mean_rel_l2, color="red", linestyle="--", label=f"Mean: {mean_rel_l2:.4f}"
)
axes[0].legend()

# Training and validation loss curves
axes[1].semilogy(
    range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss", color="steelblue"
)
axes[1].semilogy(range(1, NUM_EPOCHS + 1), val_losses, label="Val Loss", color="coral")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")
axes[1].set_title("Training Progress")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("DeepONet Error Analysis", fontsize=14)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "error_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Error analysis saved to {ASSETS_DIR / 'error_analysis.png'}")

# %% [markdown]
"""
### Branch-Trunk Embedding Analysis
"""

# %%
# Analyze learned embeddings
branch_out = model.get_branch_output(X_test_jax[:5])
trunk_out = model.get_trunk_output(trunk_jax)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Branch embedding similarity
branch_np = np.array(branch_out)
similarity = branch_np @ branch_np.T
similarity /= np.max(np.abs(similarity))
im0 = axes[0].imshow(similarity, cmap="coolwarm", vmin=-1, vmax=1)
axes[0].set_title("Branch Embedding Similarity (5 samples)")
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("Sample")
fig.colorbar(im0, ax=axes[0], shrink=0.8)

# Trunk embedding: first 3 principal components as RGB
trunk_np = np.array(trunk_out)  # (n_locations, latent_dim)
# Use first 3 dims as RGB channels
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

# %% [markdown]
"""
## Summary
"""

# %%
print()
print("=" * 70)
print(f"DeepONet Darcy Flow example completed in {elapsed:.1f}s")
print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
print(f"Results saved to: {ASSETS_DIR}")
print("=" * 70)
