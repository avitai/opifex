# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
"""
# Training a Neural Exchange-Correlation Functional

This example demonstrates training a neural exchange-correlation (XC) functional
from electron density data. Neural XC functionals can learn complex
exchange-correlation energy patterns beyond traditional LDA/GGA approximations.

**Key Concepts:**
- Exchange-correlation energy in DFT
- Density feature extraction
- Attention mechanisms for non-local correlations
- Physics constraints (negative XC energy, proper scaling)
"""

# %%
# Configuration
SEED = 42
HIDDEN_SIZES = (64, 64, 32)  # Neural network hidden layer sizes
NUM_ATTENTION_HEADS = 4
USE_ATTENTION = True
USE_ADVANCED_FEATURES = True
DROPOUT_RATE = 0.0  # No dropout for inference stability

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_TRAIN_SAMPLES = 500
NUM_TEST_SAMPLES = 100
GRID_POINTS = 32  # Points per density sample

# Output directory
OUTPUT_DIR = "docs/assets/examples/neural_xc_functional"

# %%
print("=" * 70)
print("Opifex Example: Training Neural XC Functional")
print("=" * 70)

# %%
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx


print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %%
from opifex.neural.quantum import NeuralXCFunctional


# %% [markdown]
"""
## Step 1: Generate Training Data

We generate synthetic electron density data and compute reference XC energies
using the Local Density Approximation (LDA). The neural XC functional will
learn to predict these energies from the density.

**LDA XC Energy:**
$$E_{xc}^{LDA} = -C_x \\int \\rho^{4/3} dr$$

where $C_x \\approx 0.738$ for exchange.
"""

# %%
print()
print("Generating training data...")
print("-" * 50)


def generate_density_sample(key: jax.Array, grid_points: int) -> jax.Array:
    """Generate a physically reasonable electron density sample.

    Uses a superposition of Gaussian functions to mimic atomic densities.
    """
    key1, key2, key3 = jax.random.split(key, 3)

    # Number of atomic centers (1-4)
    n_centers = int(jax.random.randint(key1, (), 1, 5))

    # Generate density as sum of Gaussians
    x = jnp.linspace(-5.0, 5.0, grid_points)
    density = jnp.zeros(grid_points)

    for i in range(n_centers):
        # Random center position and width
        center = jax.random.uniform(
            jax.random.fold_in(key2, i), (), minval=-3.0, maxval=3.0
        )
        width = jax.random.uniform(
            jax.random.fold_in(key3, i), (), minval=0.5, maxval=2.0
        )
        amplitude = jax.random.uniform(
            jax.random.fold_in(key1, i + 10), (), minval=0.5, maxval=2.0
        )

        # Add Gaussian contribution
        density = density + amplitude * jnp.exp(-((x - center) ** 2) / (2 * width**2))

    # Ensure positive density
    density = jnp.maximum(density, 1e-10)

    # Normalize to reasonable electron count (1-10 electrons)
    target_electrons = jax.random.uniform(key1, (), minval=1.0, maxval=10.0)

    return density / jnp.sum(density) * target_electrons


def compute_density_gradients(density: jax.Array) -> jax.Array:
    """Compute density gradients (simplified 1D -> 3D gradient)."""
    # Compute 1D gradient using finite differences
    grad_1d = jnp.gradient(density)

    # Expand to 3D gradient format (gradient only in first direction)
    gradients = jnp.zeros((density.shape[0], 3))

    return gradients.at[:, 0].set(grad_1d)


def compute_lda_xc_energy(density: jax.Array) -> jax.Array:
    """Compute LDA exchange-correlation energy per grid point.

    Uses Dirac exchange formula: E_x = -C_x * rho^(4/3)
    Simplified correlation: E_c = -C_c * rho * log(1 + rho)
    """
    # Exchange energy (Dirac)
    c_x = 0.738  # Exchange coefficient
    exchange = -c_x * jnp.power(jnp.maximum(density, 1e-12), 4 / 3)

    # Correlation energy (simplified Wigner-like)
    c_c = 0.044  # Correlation coefficient
    correlation = -c_c * density * jnp.log1p(density + 1e-12)

    return exchange + correlation


# Generate training data
print(f"  Training samples: {NUM_TRAIN_SAMPLES}")
print(f"  Test samples: {NUM_TEST_SAMPLES}")
print(f"  Grid points per sample: {GRID_POINTS}")

key = jax.random.PRNGKey(SEED)

# Generate training densities
train_keys = jax.random.split(key, NUM_TRAIN_SAMPLES + 1)
key = train_keys[0]
train_densities = jnp.stack(
    [generate_density_sample(k, GRID_POINTS) for k in train_keys[1:]]
)

# Generate test densities
test_keys = jax.random.split(key, NUM_TEST_SAMPLES + 1)
key = test_keys[0]
test_densities = jnp.stack(
    [generate_density_sample(k, GRID_POINTS) for k in test_keys[1:]]
)

# Compute gradients
train_gradients = jnp.stack([compute_density_gradients(d) for d in train_densities])
test_gradients = jnp.stack([compute_density_gradients(d) for d in test_densities])

# Compute reference LDA XC energies
train_xc_ref = jnp.stack([compute_lda_xc_energy(d) for d in train_densities])
test_xc_ref = jnp.stack([compute_lda_xc_energy(d) for d in test_densities])

print()
print(f"  Train densities shape: {train_densities.shape}")
print(f"  Train gradients shape: {train_gradients.shape}")
print(f"  Train XC reference shape: {train_xc_ref.shape}")
print()
print(f"  Test densities shape: {test_densities.shape}")
print(f"  Test gradients shape: {test_gradients.shape}")
print(f"  Test XC reference shape: {test_xc_ref.shape}")

# %% [markdown]
"""
## Step 2: Create Neural XC Functional

The Neural XC Functional uses:
1. **Density Feature Extractor**: Extracts physics-informed features
2. **Attention Mechanism**: Captures non-local correlations
3. **Physics Constraints**: Ensures negative XC energy and proper scaling
"""

# %%
print()
print("Creating Neural XC Functional...")
print("-" * 50)

rngs = nnx.Rngs(SEED)

model = NeuralXCFunctional(
    hidden_sizes=HIDDEN_SIZES,
    activation=nnx.gelu,
    use_attention=USE_ATTENTION,
    num_attention_heads=NUM_ATTENTION_HEADS,
    use_advanced_features=USE_ADVANCED_FEATURES,
    dropout_rate=DROPOUT_RATE,
    rngs=rngs,
)

# Count parameters
param_count = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model)))
print(f"  Hidden sizes: {HIDDEN_SIZES}")
print(f"  Use attention: {USE_ATTENTION}")
print(f"  Attention heads: {NUM_ATTENTION_HEADS}")
print(f"  Use advanced features: {USE_ADVANCED_FEATURES}")
print(f"  Total parameters: {param_count:,}")

# Test forward pass
test_output = model(train_densities[:1], train_gradients[:1], deterministic=True)
print(f"  Test output shape: {test_output.shape}")

# %% [markdown]
"""
## Step 3: Define Loss Function and Training Loop

We train the neural XC functional to minimize the mean squared error
between predicted and reference LDA XC energies.
"""

# %%
print()
print("Setting up training...")
print("-" * 50)


def loss_fn(model, densities, gradients, targets):
    """Mean squared error loss for XC energy prediction."""
    predictions = model(densities, gradients, deterministic=True)
    return jnp.mean((predictions - targets) ** 2)


# Create optimizer
optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)

print("  Optimizer: Adam")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Number of epochs: {NUM_EPOCHS}")

# %% [markdown]
"""
## Step 4: Train the Model
"""

# %%
print()
print("Training Neural XC Functional...")
print("-" * 50)


@nnx.jit
def train_step(model, optimizer, densities, gradients, targets):
    """Perform a single training step."""
    loss, grads = nnx.value_and_grad(loss_fn)(model, densities, gradients, targets)
    optimizer.update(model, grads)
    return loss


# Training loop
train_losses = []
test_losses = []
n_batches = NUM_TRAIN_SAMPLES // BATCH_SIZE

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_losses = []

    # Shuffle training data
    key, shuffle_key = jax.random.split(key)
    perm = jax.random.permutation(shuffle_key, NUM_TRAIN_SAMPLES)
    shuffled_densities = train_densities[perm]
    shuffled_gradients = train_gradients[perm]
    shuffled_targets = train_xc_ref[perm]

    # Mini-batch training
    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        batch_densities = shuffled_densities[start_idx:end_idx]
        batch_gradients = shuffled_gradients[start_idx:end_idx]
        batch_targets = shuffled_targets[start_idx:end_idx]

        loss = train_step(
            model, optimizer, batch_densities, batch_gradients, batch_targets
        )
        epoch_losses.append(float(loss))

    # Record epoch loss
    epoch_loss = jnp.mean(jnp.array(epoch_losses))
    train_losses.append(float(epoch_loss))

    # Compute test loss
    test_loss = float(loss_fn(model, test_densities, test_gradients, test_xc_ref))
    test_losses.append(test_loss)

    # Log progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"  Epoch {epoch + 1:3d}/{NUM_EPOCHS}: "
            f"train_loss = {epoch_loss:.6f}, test_loss = {test_loss:.6f}"
        )

training_time = time.time() - start_time

print()
print("Training complete!")
print(f"  Training time: {training_time:.1f}s")
print(f"  Final train loss: {train_losses[-1]:.6f}")
print(f"  Final test loss: {test_losses[-1]:.6f}")

# %% [markdown]
"""
## Step 5: Evaluate Model Performance
"""

# %%
print()
print("Evaluating model performance...")
print("-" * 50)

# Get predictions on test set
test_predictions = model(test_densities, test_gradients, deterministic=True)

# Compute metrics
mse = float(jnp.mean((test_predictions - test_xc_ref) ** 2))
mae = float(jnp.mean(jnp.abs(test_predictions - test_xc_ref)))
r2 = float(
    1
    - jnp.sum((test_xc_ref - test_predictions) ** 2)
    / jnp.sum((test_xc_ref - jnp.mean(test_xc_ref)) ** 2)
)

# Per-sample correlation
correlations = []
for i in range(NUM_TEST_SAMPLES):
    corr = jnp.corrcoef(test_predictions[i], test_xc_ref[i])[0, 1]
    if jnp.isfinite(corr):
        correlations.append(float(corr))

mean_correlation = jnp.mean(jnp.array(correlations)) if correlations else 0.0

print(f"  Mean Squared Error (MSE): {mse:.6e}")
print(f"  Mean Absolute Error (MAE): {mae:.6e}")
print(f"  R-squared (R2): {r2:.4f}")
print(f"  Mean Correlation: {mean_correlation:.4f}")

# Check physics constraints
print()
print("Physics Constraint Verification:")
all_negative = float(jnp.mean(test_predictions < 0))
print(f"  XC energy negative: {all_negative * 100:.1f}% of predictions")

# %% [markdown]
"""
## Step 6: Visualization
"""

# %%
print()
print("Generating visualizations...")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# %%
# Figure 1: Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Training curves
ax1 = axes[0]
epochs = jnp.arange(1, NUM_EPOCHS + 1)
ax1.semilogy(epochs, train_losses, "b-", linewidth=2, label="Train Loss")
ax1.semilogy(epochs, test_losses, "r--", linewidth=2, label="Test Loss")
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("MSE Loss", fontsize=12)
ax1.set_title("Training Progress", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Prediction vs Reference scatter
ax2 = axes[1]
ax2.scatter(
    test_xc_ref.flatten(),
    test_predictions.flatten(),
    alpha=0.3,
    s=5,
    c="blue",
)
# Perfect prediction line
min_val = min(test_xc_ref.min(), test_predictions.min())
max_val = max(test_xc_ref.max(), test_predictions.max())
ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect")
ax2.set_xlabel("Reference XC Energy (LDA)", fontsize=12)
ax2.set_ylabel("Predicted XC Energy", fontsize=12)
ax2.set_title(f"Prediction vs Reference (R2 = {r2:.3f})", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR}/training_curves.png")

# %%
# Figure 2: Sample predictions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Select 6 test samples
sample_indices = [0, 10, 20, 30, 40, 50]

for idx, ax in zip(sample_indices, axes.flatten(), strict=False):
    x = jnp.arange(GRID_POINTS)

    # Plot density (scaled for visibility)
    density_scaled = test_densities[idx] / test_densities[idx].max() * 0.5
    ax.fill_between(x, 0, density_scaled, alpha=0.3, color="gray", label="Density")

    # Plot XC energies
    ax.plot(x, test_xc_ref[idx], "b-", linewidth=2, label="Reference (LDA)")
    ax.plot(x, test_predictions[idx], "r--", linewidth=2, label="Predicted")

    # Compute sample correlation
    sample_corr = float(jnp.corrcoef(test_predictions[idx], test_xc_ref[idx])[0, 1])

    ax.set_xlabel("Grid Point", fontsize=10)
    ax.set_ylabel("XC Energy", fontsize=10)
    ax.set_title(f"Sample {idx} (corr = {sample_corr:.3f})", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Sample XC Energy Predictions", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sample_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR}/sample_predictions.png")

# %%
# Figure 3: Error analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Error distribution
ax1 = axes[0]
errors = (test_predictions - test_xc_ref).flatten()
ax1.hist(errors, bins=50, density=True, alpha=0.7, color="blue", edgecolor="black")
ax1.axvline(0, color="r", linestyle="--", linewidth=2)
ax1.axvline(
    jnp.mean(errors),
    color="g",
    linestyle="-",
    linewidth=2,
    label=f"Mean: {jnp.mean(errors):.4f}",
)
ax1.set_xlabel("Prediction Error", fontsize=12)
ax1.set_ylabel("Density", fontsize=12)
ax1.set_title("Error Distribution", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Error vs density magnitude
ax2 = axes[1]
density_mag = jnp.abs(test_densities).flatten()
errors_flat = jnp.abs(errors)
ax2.scatter(density_mag, errors_flat, alpha=0.2, s=3, c="blue")
ax2.set_xlabel("Density Magnitude", fontsize=12)
ax2.set_ylabel("Absolute Error", fontsize=12)
ax2.set_title("Error vs Density", fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/error_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR}/error_analysis.png")

# %% [markdown]
"""
## Step 7: Assess Chemical Accuracy
"""

# %%
print()
print("Chemical Accuracy Assessment:")
print("-" * 50)

# Use built-in assessment method
accuracy_metrics = model.assess_chemical_accuracy(
    test_densities[:10],
    test_gradients[:10],
    reference_energy=test_xc_ref[:10],
    deterministic=True,
)

for key, value in accuracy_metrics.items():
    if isinstance(value, float):
        if abs(value) < 1e-3 or abs(value) > 1e3:
            print(f"  {key}: {value:.6e}")
        else:
            print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# %% [markdown]
"""
## Results Summary
"""

# %%
print()
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()
print("Model Configuration:")
print(f"  Hidden sizes: {HIDDEN_SIZES}")
print(f"  Attention: {USE_ATTENTION} ({NUM_ATTENTION_HEADS} heads)")
print(f"  Advanced features: {USE_ADVANCED_FEATURES}")
print(f"  Parameters: {param_count:,}")
print()
print("Training:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Training samples: {NUM_TRAIN_SAMPLES}")
print(f"  Training time: {training_time:.1f}s")
print(f"  Final train loss: {train_losses[-1]:.6e}")
print(f"  Final test loss: {test_losses[-1]:.6e}")
print()
print("Evaluation:")
print(f"  MSE: {mse:.6e}")
print(f"  MAE: {mae:.6e}")
print(f"  R-squared: {r2:.4f}")
print(f"  Mean correlation: {mean_correlation:.4f}")
print()
print("Physics Constraints:")
print(f"  Negative XC energy: {all_negative * 100:.1f}%")
print("=" * 70)

# %%
print()
print("Neural XC Functional training example completed successfully!")
