# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Local FNO on Darcy Flow
#
# This example demonstrates training a Local Fourier Neural Operator (LocalFNO)
# on the Darcy flow problem. LocalFNO combines global Fourier spectral convolutions
# with local spatial convolutions to capture both long-range dependencies and
# fine-grained local features.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.fno.local import LocalFourierNeuralOperator


# %%
# Configuration
print("=" * 70)
print("Opifex Example: Local FNO on Darcy Flow")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Problem configuration
RESOLUTION = 32
N_TRAIN = 200
N_TEST = 50
BATCH_SIZE = 16
EPOCHS = 20

# Model configuration
MODES = (12, 12)
HIDDEN_CHANNELS = 32
NUM_LAYERS = 4
KERNEL_SIZE = 3

print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
print(f"FNO config: modes={MODES}, width={HIDDEN_CHANNELS}, layers={NUM_LAYERS}")
print(f"Local kernel size: {KERNEL_SIZE}")

# %% [markdown]
# ## Data Loading
#
# Generate Darcy flow data: permeability fields (input) mapped to pressure solutions (output).

# %%
print()
print("Generating Darcy flow data...")

train_loader = create_darcy_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=True,
    seed=42,
    worker_count=0,
)

test_loader = create_darcy_loader(
    n_samples=N_TEST,
    batch_size=N_TEST,
    resolution=RESOLUTION,
    shuffle=False,
    seed=123,
    worker_count=0,
)

# Get data as arrays
train_batch = next(iter(train_loader))
X_train = jnp.array(train_batch["input"])
Y_train = jnp.array(train_batch["output"])

test_batch = next(iter(test_loader))
X_test = jnp.array(test_batch["input"])
Y_test = jnp.array(test_batch["output"])

# Ensure channel dimension exists (NCHW format)
if X_train.ndim == 3:
    X_train = X_train[:, None, :, :]  # Add channel dimension
    Y_train = Y_train[:, None, :, :]
    X_test = X_test[:, None, :, :]
    Y_test = Y_test[:, None, :, :]

print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test data:     X={X_test.shape}, Y={Y_test.shape}")

# %% [markdown]
# ## Model Creation
#
# Create both LocalFNO and standard FNO for comparison.

# %%
print()
print("Creating LocalFNO model...")

local_fno = LocalFourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=HIDDEN_CHANNELS,
    modes=MODES,
    num_layers=NUM_LAYERS,
    kernel_size=KERNEL_SIZE,
    use_residual_connections=True,
    rngs=nnx.Rngs(42),
)

# Count parameters
local_fno_params = sum(
    x.size for x in jax.tree_util.tree_leaves(nnx.state(local_fno, nnx.Param))
)
print(f"LocalFNO parameters: {local_fno_params:,}")

# Create standard FNO for comparison
print()
print("Creating standard FNO for comparison...")
standard_fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=HIDDEN_CHANNELS,
    modes=MODES[0],
    num_layers=NUM_LAYERS,
    rngs=nnx.Rngs(42),
)

fno_params = sum(
    x.size for x in jax.tree_util.tree_leaves(nnx.state(standard_fno, nnx.Param))
)
print(f"Standard FNO parameters: {fno_params:,}")
print(f"LocalFNO overhead: {(local_fno_params / fno_params - 1) * 100:.1f}%")

# %% [markdown]
# ## Training
#
# Train both models with Adam optimizer.


# %%
def train_model(model, X_train, Y_train, epochs, lr=1e-3, model_name="Model"):
    """Train a model with MSE loss."""
    opt = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, opt, x, y):
        def loss_fn(model):
            y_pred = model(x)
            return jnp.mean((y_pred - y) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    print(f"Training {model_name}...")
    losses = []

    for epoch in range(epochs):
        loss = train_step(model, opt, X_train, Y_train)
        losses.append(float(loss))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: loss={loss:.6f}")

    return losses


# %%
print()
local_fno_losses = train_model(
    local_fno, X_train, Y_train, EPOCHS, model_name="LocalFNO"
)
print(f"Final LocalFNO loss: {local_fno_losses[-1]:.6e}")

print()
fno_losses = train_model(
    standard_fno, X_train, Y_train, EPOCHS, model_name="Standard FNO"
)
print(f"Final FNO loss: {fno_losses[-1]:.6e}")

# %% [markdown]
# ## Evaluation
#
# Compare LocalFNO and standard FNO on test data.


# %%
def evaluate_model(model, X_test, Y_test, model_name="Model"):
    """Evaluate model on test data."""
    predictions = model(X_test)
    mse = float(jnp.mean((predictions - Y_test) ** 2))

    # Relative L2 error per sample
    rel_l2_per_sample = jnp.sqrt(
        jnp.sum((predictions - Y_test) ** 2, axis=(1, 2, 3))
        / jnp.sum(Y_test**2, axis=(1, 2, 3))
    )
    rel_l2_mean = float(jnp.mean(rel_l2_per_sample))
    rel_l2_min = float(jnp.min(rel_l2_per_sample))
    rel_l2_max = float(jnp.max(rel_l2_per_sample))

    print(f"{model_name} Results:")
    print(f"  Test MSE:         {mse:.6f}")
    print(
        f"  Relative L2:      {rel_l2_mean:.6f} (min={rel_l2_min:.6f}, max={rel_l2_max:.6f})"
    )

    return predictions, mse, rel_l2_mean


# %%
print()
print("Running evaluation...")
local_pred, local_mse, local_rel_l2 = evaluate_model(
    local_fno, X_test, Y_test, "LocalFNO"
)
print()
fno_pred, fno_mse, fno_rel_l2 = evaluate_model(
    standard_fno, X_test, Y_test, "Standard FNO"
)

# Compare
print()
print("Comparison:")
mse_improvement = (fno_mse - local_mse) / fno_mse * 100
rel_l2_improvement = (fno_rel_l2 - local_rel_l2) / fno_rel_l2 * 100
print(f"  MSE improvement (LocalFNO vs FNO): {mse_improvement:+.1f}%")
print(f"  Rel L2 improvement: {rel_l2_improvement:+.1f}%")

# %% [markdown]
# ## Visualization
#
# Compare predictions and analyze local vs global feature capture.

# %%
# Create output directory
output_dir = Path("docs/assets/examples/local_fno_darcy")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot predictions for a sample
mpl.use("Agg")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

sample_idx = 0

# Row 1: LocalFNO
axes[0, 0].imshow(np.array(X_test[sample_idx, 0]), cmap="viridis")
axes[0, 0].set_title("Input (Permeability)")
axes[0, 0].axis("off")

axes[0, 1].imshow(np.array(Y_test[sample_idx, 0]), cmap="RdBu_r")
axes[0, 1].set_title("Ground Truth")
axes[0, 1].axis("off")

axes[0, 2].imshow(np.array(local_pred[sample_idx, 0]), cmap="RdBu_r")
axes[0, 2].set_title("LocalFNO Prediction")
axes[0, 2].axis("off")

local_error = np.abs(np.array(local_pred[sample_idx, 0] - Y_test[sample_idx, 0]))
im1 = axes[0, 3].imshow(local_error, cmap="hot")
axes[0, 3].set_title(f"LocalFNO Error (max={local_error.max():.4f})")
axes[0, 3].axis("off")
plt.colorbar(im1, ax=axes[0, 3], fraction=0.046)

# Row 2: Standard FNO
axes[1, 0].imshow(np.array(X_test[sample_idx, 0]), cmap="viridis")
axes[1, 0].set_title("Input (Permeability)")
axes[1, 0].axis("off")

axes[1, 1].imshow(np.array(Y_test[sample_idx, 0]), cmap="RdBu_r")
axes[1, 1].set_title("Ground Truth")
axes[1, 1].axis("off")

axes[1, 2].imshow(np.array(fno_pred[sample_idx, 0]), cmap="RdBu_r")
axes[1, 2].set_title("Standard FNO Prediction")
axes[1, 2].axis("off")

fno_error = np.abs(np.array(fno_pred[sample_idx, 0] - Y_test[sample_idx, 0]))
im2 = axes[1, 3].imshow(fno_error, cmap="hot")
axes[1, 3].set_title(f"FNO Error (max={fno_error.max():.4f})")
axes[1, 3].axis("off")
plt.colorbar(im2, ax=axes[1, 3], fraction=0.046)

plt.tight_layout()
plt.savefig(output_dir / "predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Predictions saved to {output_dir / 'predictions.png'}")

# %%
# Training comparison plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].semilogy(local_fno_losses, label="LocalFNO", linewidth=2)
axes[0].semilogy(fno_losses, label="Standard FNO", linewidth=2, linestyle="--")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training Loss Comparison")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Error comparison
models = ["LocalFNO", "Standard FNO"]
mse_values = [local_mse, fno_mse]
rel_l2_values = [local_rel_l2, fno_rel_l2]

x = np.arange(len(models))
width = 0.35

bars1 = axes[1].bar(x - width / 2, mse_values, width, label="MSE", color="steelblue")
ax2 = axes[1].twinx()
bars2 = ax2.bar(x + width / 2, rel_l2_values, width, label="Rel L2", color="coral")

axes[1].set_ylabel("MSE", color="steelblue")
ax2.set_ylabel("Relative L2", color="coral")
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].set_title("Test Error Comparison")
axes[1].legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Comparison saved to {output_dir / 'comparison.png'}")

# %%
# Summary
print()
print("=" * 70)
print("Local FNO Darcy Flow example completed")
print("=" * 70)
print()
print("Results Summary:")
print(
    f"  LocalFNO:     MSE={local_mse:.6f}, Rel L2={local_rel_l2:.4f}, Params={local_fno_params:,}"
)
print(
    f"  Standard FNO: MSE={fno_mse:.6f}, Rel L2={fno_rel_l2:.4f}, Params={fno_params:,}"
)
print(f"  Improvement:  MSE {mse_improvement:+.1f}%, Rel L2 {rel_l2_improvement:+.1f}%")
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
