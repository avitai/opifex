"""
Darcy Flow UNO - Opifex Framework Implementation.

==============================================

Reproduces the neuraloperator UNO Darcy Flow example using Opifex framework.
This implements training a U-Net Neural Operator (UNO) on the Darcy Flow equation (2D elliptic PDE).

Equivalent to: neuraloperator/examples/models/plot_UNO_darcy.py

The UNO combines U-Net architecture with spectral convolutions for multi-scale
operator learning with zero-shot super-resolution capabilities.
"""

# CRITICAL: Set JAX environment variables before importing JAX
import os


os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["JAX_DEBUG_NANS"] = "False"
# Suppress XLA slow operation warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress all TensorFlow logging
os.environ["JAX_LOG_COMPILES"] = "0"  # Suppress JAX compilation logs
# Note: XLA slow operation warnings are normal during first compilation and can be ignored

from datetime import datetime, UTC
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

# Opifex imports
from opifex.core.training import Trainer, TrainingConfig
from opifex.neural.operators.specialized.uno import UNeuralOperator
from opifex.training.basic_trainer import create_progress_bar_callback


def create_results_directory():
    """Create a timestamped results directory for this run."""
    # Create base output directory if it doesn't exist
    base_dir = Path("examples_output")
    base_dir.mkdir(exist_ok=True)

    # Create timestamped subdirectory for this run
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_dir = base_dir / f"darcy_uno_run_{timestamp}"
    results_dir.mkdir(exist_ok=True)

    print(f"Results will be saved to: {results_dir}")
    return results_dir


def _create_spatial_grid(resolution: int) -> tuple[jax.Array, jax.Array]:
    """Create spatial grid for Darcy flow problem."""
    x = jnp.linspace(0, 1, resolution)
    y = jnp.linspace(0, 1, resolution)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y


def _generate_coefficient_field(key: jax.Array, resolution: int) -> jax.Array:
    """Generate permeability coefficient field using GRF (vectorized)."""
    # Simple GRF approximation using Fourier series
    n_modes = 8  # Number of Fourier modes

    # Generate random Fourier coefficients
    modes_x = jnp.arange(1, n_modes + 1)
    modes_y = jnp.arange(1, n_modes + 1)

    k1, k2 = jax.random.split(key)

    # Random amplitudes with power-law decay
    amp_x = jax.random.normal(k1, (n_modes,)) / (modes_x**1.5)
    amp_y = jax.random.normal(k2, (n_modes,)) / (modes_y**1.5)

    # Spatial grid
    x = jnp.linspace(0, 1, resolution)
    y = jnp.linspace(0, 1, resolution)

    # Precompute sine bases on 1D grids then combine via outer products
    sx = jnp.sin(jnp.pi * modes_x[:, None] * x[None, :])  # (n_modes, res)
    sy = jnp.sin(jnp.pi * modes_y[:, None] * y[None, :])  # (n_modes, res)

    # Vectorized reconstruction: sum_{i,j} amp_x[i]*amp_y[j]*sx[i,x]*sy[j,y]
    field = jnp.einsum("i,j,ix,jy->xy", amp_x, amp_y, sx, sy)

    # Ensure positive coefficients
    return jnp.exp(1.0 + 0.1 * field)


def _jacobi_sweep(u: jax.Array, a: jax.Array, f: jax.Array, h: float) -> jax.Array:
    """One vectorized Jacobi sweep updating interior points only."""
    # Neighbor slices
    u_c = u[1:-1, 1:-1]
    u_ip1 = u[2:, 1:-1]
    u_im1 = u[:-2, 1:-1]
    u_jp1 = u[1:-1, 2:]
    u_jm1 = u[1:-1, :-2]

    a_c = a[1:-1, 1:-1]
    a_ip1 = a[2:, 1:-1]
    a_jp1 = a[1:-1, 2:]
    f_c = f[1:-1, 1:-1]

    inv_h2 = 1.0 / (h * h)

    coeff = 2.0 * (a_ip1 + a_c + a_jp1 + a_c) * inv_h2
    rhs = (
        f_c
        + (a_ip1 * u_ip1 + a_c * u_im1) * inv_h2
        + (a_jp1 * u_jp1 + a_c * u_jm1) * inv_h2
    )

    # Pure Jacobi update (interior only)
    u_new_center = rhs / coeff

    # Write back to full array
    return u.at[1:-1, 1:-1].set(u_new_center)


@jax.jit
def _solve_darcy_pde(a: jax.Array, resolution: int) -> jax.Array:
    """Solve Darcy equation using vectorized weighted-Jacobi with JIT acceleration."""
    # Get resolution from input array shape instead of using the parameter
    resolution_from_shape = a.shape[0]

    # Grid spacing
    h = 1.0 / (resolution_from_shape - 1)

    # Right-hand side (forcing term) - use shape from input array
    f = jnp.ones_like(a)

    # Initialize solution (Dirichlet boundary: zeros)
    u0 = jnp.zeros_like(f)

    # Weighted-Jacobi relaxation factor (empirically good for Poisson-like problems)
    omega = 0.8
    max_iters = 200
    tol = 1e-4

    def body(carry):
        u, _u_prev, it = carry
        u_jacobi = _jacobi_sweep(u, a, f, h)
        # Weighted relaxation on interior points only
        u_relaxed = u + omega * (u_jacobi - u)
        return (u_relaxed, u, it + 1)

    def cond(carry):
        u, u_prev, it = carry
        # Relative change as stopping criterion
        num = jnp.linalg.norm(u - u_prev)
        den = jnp.linalg.norm(u_prev) + 1e-8
        rel_change = num / den
        return jnp.logical_and(it < max_iters, rel_change > tol)

    u_sol, _, _ = jax.lax.while_loop(cond, body, (u0, u0 + 1.0, 0))
    return u_sol


def _generate_dataset_samples(
    keys: jax.Array, resolution: int, n_samples: int
) -> tuple[jax.Array, jax.Array]:
    """Generate coefficient-solution pairs for dataset."""

    def generate_sample(key):
        a = _generate_coefficient_field(key, resolution)
        u = _solve_darcy_pde(a, resolution)
        return a, u

    # Vectorized generation with progress feedback
    print(f"  - Generating {n_samples} coefficient fields and solving PDEs...")
    data = jax.vmap(generate_sample)(keys)
    X, y = data
    print("  - Data generation completed!")
    return X, y


def load_darcy_flow_data(
    n_train: int = 1000, n_test: int = 200, resolution: int = 64
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Load or generate Darcy Flow dataset using JAX transformations for efficiency.

    Darcy's equation: -∇·(a(x)∇u(x)) = f(x) in Ω
    with homogeneous Dirichlet boundary conditions.

    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        resolution: Spatial resolution (resolution x resolution grid)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) arrays
    """
    print(f"Generating Darcy Flow dataset: {n_train} train, {n_test} test samples")
    print(f"Resolution: {resolution}x{resolution}")

    key = jax.random.key(0)
    keys = jax.random.split(key, 4)

    # Generate training data
    print("Generating training data...")
    train_keys = jax.random.split(keys[0], n_train)
    X_train, y_train = _generate_dataset_samples(train_keys, resolution, n_train)

    # Generate test data
    print("Generating test data...")
    test_keys = jax.random.split(keys[1], n_test)
    X_test, y_test = _generate_dataset_samples(test_keys, resolution, n_test)

    # Reshape for neural operator (add channel dimension)
    # UNO expects (batch, height, width, channels) format
    X_train = X_train[..., None]  # Add channel dimension
    y_train = y_train[..., None]
    X_test = X_test[..., None]
    y_test = y_test[..., None]

    print(f"Training data shape: {X_train.shape} -> {y_train.shape}")
    print(f"Test data shape: {X_test.shape} -> {y_test.shape}")

    return X_train, y_train, X_test, y_test


def create_uno_model(*, rngs: nnx.Rngs) -> UNeuralOperator:
    """Create UNO model for Darcy flow problem."""
    return UNeuralOperator(
        input_channels=1,  # Permeability field
        output_channels=1,  # Solution field
        hidden_channels=16,  # Reduced from 32 for memory efficiency
        modes=16,  # Reduced from 32 for memory efficiency
        n_layers=3,  # Reduced from 4 for memory efficiency
        use_spectral=True,  # Enable spectral convolutions
        activation=nnx.gelu,  # Activation function
        rngs=rngs,
    )


def _plot_single_sample(
    axes, i: int, X_viz: jax.Array, y_true: jax.Array, y_pred: jax.Array
):
    """Plot input, true solution, and prediction for a single sample."""
    # Input (permeability field)
    im1 = axes[i, 0].imshow(X_viz[i, :, :, 0], cmap="viridis")
    axes[i, 0].set_title(f"Input {i + 1}: Permeability Field")
    axes[i, 0].set_xlabel("x")
    axes[i, 0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[i, 0])

    # True solution
    im2 = axes[i, 1].imshow(y_true[i, :, :, 0], cmap="plasma")
    axes[i, 1].set_title(f"True Solution {i + 1}")
    axes[i, 1].set_xlabel("x")
    axes[i, 1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[i, 1])

    # Predicted solution
    im3 = axes[i, 2].imshow(y_pred[i, :, :, 0], cmap="plasma")
    axes[i, 2].set_title(f"UNO Prediction {i + 1}")
    axes[i, 2].set_xlabel("x")
    axes[i, 2].set_ylabel("y")
    plt.colorbar(im3, ax=axes[i, 2])


def _plot_errors(axes, i: int, errors: jax.Array, relative_errors: jax.Array):
    """Plot absolute and relative errors for a single sample."""
    # Absolute error
    im1 = axes[i, 0].imshow(errors[i, :, :, 0], cmap="Reds")
    axes[i, 0].set_title(f"Absolute Error {i + 1}")
    axes[i, 0].set_xlabel("x")
    axes[i, 0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[i, 0])

    # Relative error
    im2 = axes[i, 1].imshow(relative_errors[i, :, :, 0], cmap="Reds")
    axes[i, 1].set_title(f"Relative Error {i + 1}")
    axes[i, 1].set_xlabel("x")
    axes[i, 1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[i, 1])


def visualize_predictions(
    model: UNeuralOperator,
    X_test: jax.Array,
    y_test: jax.Array,
    results_dir: Path,
    n_samples: int = 3,
):
    """Visualize UNO predictions on test data."""
    print("    - Selecting test samples for visualization...")

    # Select samples for visualization
    indices = jnp.linspace(0, len(X_test) - 1, n_samples, dtype=int)
    X_viz = X_test[indices]
    y_true = y_test[indices]

    # Generate predictions
    print("    - Running model predictions...")
    y_pred = jax.vmap(lambda x: model(x[None], deterministic=True)[0])(X_viz)

    # Create prediction visualization
    _, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]  # Ensure 2D array

    for i in range(n_samples):
        _plot_single_sample(axes, i, X_viz, y_true, y_pred)

    plt.tight_layout()
    print("    - Saving prediction plots...")
    plt.savefig(results_dir / "uno_darcy_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Compute and visualize errors
    print("    - Computing error analysis...")
    errors = jnp.abs(y_pred - y_true)
    relative_errors = errors / (jnp.abs(y_true) + 1e-8)

    print("    - Creating error visualization...")
    _, axes = plt.subplots(1, 4, figsize=(20, 5))
    if n_samples == 1:
        axes = axes[None, :]

    for i in range(n_samples):
        _plot_errors(axes, i, errors, relative_errors)

    plt.tight_layout()
    print("    - Saving error plots...")
    plt.savefig(results_dir / "uno_darcy_errors.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print error statistics
    print("    - Computing performance metrics...")
    mse = jnp.mean((y_pred - y_true) ** 2)
    mae = jnp.mean(jnp.abs(y_pred - y_true))
    relative_error = jnp.mean(relative_errors)

    print("\n📊 UNO Performance on Test Samples:")
    print(f"   📉 Mean Squared Error: {mse:.6f}")
    print(f"   📏 Mean Absolute Error: {mae:.6f}")
    print(f"   📐 Mean Relative Error: {relative_error:.6f}")


def _setup_training_config() -> TrainingConfig:
    """Setup training configuration with progress bar."""
    config = TrainingConfig(
        learning_rate=5e-4,
        num_epochs=20,  # Reduced for demonstration
        batch_size=16,  # Reduced for memory efficiency
        validation_frequency=5,
        checkpoint_frequency=10,
        progress_callback=create_progress_bar_callback("UNO Training"),
        verbose=False,  # Disable verbose since we're using progress bar
    )

    print("\nTraining configuration:")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Number of epochs: {config.num_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Evaluation frequency: every {config.validation_frequency} epochs")
    print(f"  - Checkpoint frequency: every {config.checkpoint_frequency} epochs")

    return config


def _save_training_history(results_dir: Path, metrics):
    """Save training history to JSON file."""
    print("\nSaving results...")
    import json

    history_file = results_dir / "training_history.json"
    print("  - Saving training history...")
    with open(history_file, "w") as f:
        history_serializable = {
            "train_losses": [float(x) for x in metrics.train_losses],
            "val_losses": [float(x) for x in metrics.val_losses],
            "epochs": list(range(len(metrics.train_losses))),
        }
        json.dump(history_serializable, f, indent=2)


def _print_final_summary(metrics, param_count: int):
    """Print final training summary."""
    print("\n🎉 Training completed successfully!")
    print("   📈 uno_darcy_predictions.png: Model predictions")
    print("   🎯 uno_darcy_errors.png: Error analysis")
    print("   📉 training_history.json: Training metrics")
    print("   💾 checkpoints/: Model checkpoints")

    # Print summary statistics
    final_train_loss = (
        metrics.train_losses[-1] if metrics.train_losses else float("nan")
    )
    final_val_loss = metrics.val_losses[-1] if metrics.val_losses else float("nan")
    print("\n📋 Training Summary:")
    print(f"   🏁 Final training loss: {final_train_loss:.6f}")
    print(f"   🎯 Final validation loss: {final_val_loss:.6f}")
    print(f"   ⚡ Total epochs: {len(metrics.train_losses)}")
    print(f"   💾 Model parameters: {param_count:,}")


def main():
    """Main training and evaluation loop."""
    print("=== UNO Darcy Flow - Opifex Framework ===")

    # Verify JAX x64 configuration
    print(
        f"JAX configured: {jax.default_backend()} backend, x64 enabled: {jax.config.read('jax_enable_x64')}"
    )
    print(
        "Note: XLA slow operation warnings during first compilation are normal and can be ignored"
    )

    # Create results directory
    results_dir = create_results_directory()

    # Load data
    X_train, y_train, X_test, y_test = load_darcy_flow_data(
        n_train=200,  # Reduced for memory efficiency
        n_test=50,
        resolution=32,  # Reduced resolution
    )

    # Initialize model
    print("\nInitializing UNO model...")
    print("  - Creating U-Net Neural Operator...")
    rngs = nnx.Rngs(0)
    model = create_uno_model(rngs=rngs)

    param_count = sum(x.size for x in jax.tree.leaves(nnx.state(model)))
    print(f"  - Model created with {param_count:,} parameters")
    print("  - Model architecture: U-Net with spectral convolutions")

    # Setup training configuration
    config = _setup_training_config()

    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(model=model, config=config)
    print("  - Trainer initialized successfully")

    # Train model with integrated progress bar
    print("  - Training progress will be displayed with progress bar")
    trained_model, metrics = trainer.train(
        (X_train, y_train), (X_test[:50], y_test[:50])
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    print(f"  - Running evaluation on {len(X_test)} test samples...")
    test_pred = trained_model(X_test, deterministic=True)
    test_loss = jnp.mean((test_pred - y_test) ** 2)
    print(f"  - Final test loss: {float(test_loss):.6f}")

    # Visualize results
    print("\nGenerating visualizations...")
    print("  - Creating prediction plots...")
    visualize_predictions(trained_model, X_test, y_test, results_dir)
    print("  - Visualizations completed!")

    # Save results and print summary
    _save_training_history(results_dir, metrics)
    print(f"📁 Results saved to: {results_dir}")
    print("\n📊 Files generated:")
    _print_final_summary(metrics, param_count)


if __name__ == "__main__":
    main()
