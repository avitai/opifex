#!/usr/bin/env python3
"""
U-Net Neural Operator (UNO) for Darcy Flow - Opifex Framework Example

This example demonstrates the UNO architecture using the Opifex framework's
neural operator infrastructure, showcasing proper integration with the
framework's components.

Key Features:
- Uses Opifex framework's UNO implementation
- Integration with existing training infrastructure
- Proper use of Opifex patterns and best practices
- Comprehensive evaluation and visualization
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

# Opifex Framework imports
from opifex.neural.operators.foundations import create_uno, UNeuralOperator


def create_synthetic_darcy_data(
    batch_size: int,
    resolution: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Create synthetic Darcy flow data for demonstration."""
    # Generate smooth permeability field using random Fourier modes
    k1, k2 = jax.random.split(key)

    # Create coordinate grid
    x = jnp.linspace(0, 1, resolution)
    y = jnp.linspace(0, 1, resolution)
    X, Y = jnp.meshgrid(x, y)

    # Generate random Fourier coefficients
    n_modes = 8
    coeffs_real = jax.random.normal(k1, (batch_size, n_modes, n_modes)) * 0.5
    coeffs_imag = jax.random.normal(k2, (batch_size, n_modes, n_modes)) * 0.5

    # Create permeability field using Fourier series
    permeability = jnp.zeros((batch_size, resolution, resolution))

    for i in range(n_modes):
        for j in range(n_modes):
            mode_real = coeffs_real[:, i, j : j + 1, None] * jnp.cos(
                2 * jnp.pi * (i * X + j * Y)
            )
            mode_imag = coeffs_imag[:, i, j : j + 1, None] * jnp.sin(
                2 * jnp.pi * (i * X + j * Y)
            )
            permeability = permeability + mode_real + mode_imag

    # Ensure positivity
    permeability = jnp.exp(permeability)

    # Create simple pressure solution (for demonstration)
    # In reality, this would solve the Darcy equation
    pressure = (
        jnp.sin(jnp.pi * X)
        * jnp.sin(jnp.pi * Y)
        / permeability.mean(axis=(1, 2), keepdims=True)
    )

    # Add channel dimensions
    x_data = permeability[..., None]  # (batch, height, width, 1)
    y_data = pressure[..., None]  # (batch, height, width, 1)

    return x_data, y_data


class UNODarcyTrainer:
    """Trainer for UNO on Darcy flow problems."""

    def __init__(
        self,
        model: UNeuralOperator,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model

        # Create optimizer
        self.optimizer = nnx.Optimizer(
            model, optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        )

        # Training state
        self.step = 0
        self.losses = []
        self.val_errors = []

    def train_step(
        self,
        x_batch: jax.Array,
        y_batch: jax.Array,
    ) -> float:
        """Single training step."""

        def loss_fn(model):
            y_pred = model(x_batch, deterministic=False)
            return jnp.mean((y_pred - y_batch) ** 2)

        # Compute loss and gradients
        loss, grads = nnx.value_and_grad(loss_fn)(self.model)

        # Update model
        self.optimizer.update(grads)

        # Track loss
        self.losses.append(float(loss))
        self.step += 1

        return float(loss)

    def evaluate(
        self,
        x_val: jax.Array,
        y_val: jax.Array,
    ) -> dict[str, float]:
        """Evaluate model on validation data."""

        # Forward pass
        y_pred = self.model(x_val, deterministic=True)

        # Compute metrics
        mse_loss = float(jnp.mean((y_pred - y_val) ** 2))

        # L2 relative error
        l2_error = float(
            jnp.sqrt(jnp.sum((y_pred - y_val) ** 2)) / jnp.sqrt(jnp.sum(y_val**2))
        )

        # Maximum absolute error
        max_error = float(jnp.max(jnp.abs(y_pred - y_val)))

        return {
            "mse_loss": mse_loss,
            "l2_relative_error": l2_error,
            "max_absolute_error": max_error,
        }


def train_uno_model(
    model: UNeuralOperator,
    resolution: int = 64,
    num_epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    val_frequency: int = 10,
) -> dict:
    """Train UNO model on synthetic Darcy data."""

    print(f"🏋️  Training UNO for {num_epochs} epochs...")

    # Create trainer
    trainer = UNODarcyTrainer(model, learning_rate=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Generate training batch
        train_key = jax.random.key(epoch)
        x_train, y_train = create_synthetic_darcy_data(
            batch_size, resolution, train_key
        )

        # Training step
        loss = trainer.train_step(x_train, y_train)

        # Validation
        if epoch % val_frequency == 0:
            val_key = jax.random.key(epoch + 10000)
            x_val, y_val = create_synthetic_darcy_data(batch_size, resolution, val_key)

            metrics = trainer.evaluate(x_val, y_val)
            trainer.val_errors.append(metrics["l2_relative_error"])

            print(
                f"Epoch {epoch:3d}: Loss = {loss:.6f}, "
                f"L2 Error = {metrics['l2_relative_error']:.6f}"
            )

    return {
        "training_losses": trainer.losses,
        "validation_errors": trainer.val_errors,
        "final_loss": trainer.losses[-1],
        "final_val_error": trainer.val_errors[-1] if trainer.val_errors else None,
    }


def demonstrate_zero_shot_superresolution(
    model: UNeuralOperator,
    base_resolution: int = 64,
    target_resolution: int = 128,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Demonstrate zero-shot super-resolution capabilities."""

    print(
        f"🔍 Testing zero-shot super-resolution: {base_resolution} → {target_resolution}"
    )

    # Generate test data at base resolution
    test_key = jax.random.key(999)
    x_test, y_test = create_synthetic_darcy_data(1, base_resolution, test_key)

    # Upsample input to target resolution
    x_high_res = jax.image.resize(
        x_test, (1, target_resolution, target_resolution, 1), method="bilinear"
    )

    # Predict at high resolution
    y_pred_high = model(x_high_res, deterministic=True)

    # Upsample ground truth for comparison
    y_test_high = jax.image.resize(
        y_test, (1, target_resolution, target_resolution, 1), method="bilinear"
    )

    return x_high_res, y_pred_high, y_test_high


def visualize_uno_results(
    x: jax.Array,
    y_true: jax.Array,
    y_pred: jax.Array,
    title: str = "UNO Results",
    save_path: Path | None = None,
):
    """Visualize UNO prediction results."""

    _, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Extract data from first batch item
    x_plot = np.array(x[0, :, :, 0])
    y_true_plot = np.array(y_true[0, :, :, 0])
    y_pred_plot = np.array(y_pred[0, :, :, 0])

    # Input (permeability)
    im1 = axes[0].imshow(x_plot, cmap="viridis", origin="lower")
    axes[0].set_title("Input (Permeability)")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Ground truth
    im2 = axes[1].imshow(y_true_plot, cmap="RdBu_r", origin="lower")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Prediction
    im3 = axes[2].imshow(y_pred_plot, cmap="RdBu_r", origin="lower")
    axes[2].set_title("UNO Prediction")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    # Error
    error = np.abs(y_true_plot - y_pred_plot)
    im4 = axes[3].imshow(error, cmap="Reds", origin="lower")
    axes[3].set_title("Absolute Error")
    axes[3].axis("off")
    plt.colorbar(im4, ax=axes[3], shrink=0.8)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Visualization saved to {save_path}")

    plt.show()


def analyze_uno_performance(
    model: UNeuralOperator,
    resolutions: list[int] | None = None,
) -> dict:
    """Analyze UNO performance across different resolutions."""
    if resolutions is None:
        resolutions = [32, 64, 128]

    print("📈 Analyzing UNO performance across resolutions...")

    results = {}

    for resolution in resolutions:
        print(f"   Testing {resolution}x{resolution}...")

        # Generate test data
        test_key = jax.random.key(resolution)
        x_test, y_test = create_synthetic_darcy_data(1, resolution, test_key)

        # Time the prediction
        start_time = time.time()
        y_pred = model(x_test, deterministic=True)
        y_pred_blocked = jax.block_until_ready(y_pred)
        end_time = time.time()

        # Compute metrics
        mse_loss = float(jnp.mean((y_pred - y_test) ** 2))
        l2_error = float(
            jnp.sqrt(jnp.sum((y_pred - y_test) ** 2)) / jnp.sqrt(jnp.sum(y_test**2))
        )

        results[resolution] = {
            "mse_loss": mse_loss,
            "l2_relative_error": l2_error,
            "inference_time": end_time - start_time,
            "throughput": 1.0 / (end_time - start_time),
        }

    return results


def plot_training_curves(
    training_results: dict,
    save_path: Path | None = None,
):
    """Plot training loss and validation error curves."""

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Training loss
    ax1.plot(training_results["training_losses"])
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Validation error
    if training_results["validation_errors"]:
        val_steps = np.arange(
            0,
            len(training_results["training_losses"]),
            len(training_results["training_losses"])
            // len(training_results["validation_errors"]),
        )
        val_steps = val_steps[: len(training_results["validation_errors"])]

        ax2.plot(val_steps, training_results["validation_errors"], "o-")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("L2 Relative Error")
        ax2.set_title("Validation Error")
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📈 Training curves saved to {save_path}")

    plt.show()


def main():  # noqa: PLR0915
    """Main UNO demonstration using Opifex framework."""

    print("🚀 UNO (U-Net Neural Operator) - Opifex Framework Example")
    print("=" * 65)

    # Configuration
    resolution = 64
    hidden_channels = 32  # Smaller for faster demo
    modes = 12
    n_layers = 3
    num_epochs = 50
    batch_size = 4
    learning_rate = 1e-3

    # Create output directory
    output_dir = Path("examples/models/examples_output/uno_framework")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Problem: Darcy flow at {resolution}x{resolution} resolution")
    print(
        f"🏗️  Architecture: UNO with {hidden_channels} channels, {modes} modes, {n_layers} layers"
    )

    # Create UNO model using Opifex framework
    print("\n1️⃣ Creating UNO model using Opifex framework...")
    rngs = nnx.Rngs(42)

    model = create_uno(
        input_channels=1,
        output_channels=1,
        hidden_channels=hidden_channels,
        modes=modes,
        n_layers=n_layers,
        rngs=rngs,
    )

    # Re-enable spectral layers for full UNO functionality
    # model.use_spectral = False

    # Count parameters
    params = nnx.state(model, nnx.Param)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"   ✅ UNO model created with {param_count:,} parameters")

    # Test forward pass
    print("\n2️⃣ Testing model forward pass...")
    test_key = jax.random.key(123)
    x_test, y_test = create_synthetic_darcy_data(2, resolution, test_key)

    try:
        y_pred = model(x_test, deterministic=True)
        print(f"   ✅ Forward pass successful: {x_test.shape} → {y_pred.shape}")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return

    # Training
    print(f"\n3️⃣ Training UNO for {num_epochs} epochs...")
    start_time = time.time()

    training_results = train_uno_model(
        model=model,
        resolution=resolution,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_frequency=10,
    )

    training_time = time.time() - start_time
    print(f"   ✅ Training completed in {training_time:.2f}s")
    print(f"   📊 Final training loss: {training_results['final_loss']:.6f}")
    if training_results["final_val_error"]:
        print(
            f"   📊 Final validation error: {training_results['final_val_error']:.6f}"
        )

    # Plot training curves
    plot_training_curves(training_results, save_path=output_dir / "training_curves.png")

    # Performance analysis
    print("\n4️⃣ Analyzing UNO performance...")
    performance_results = analyze_uno_performance(
        model=model,
        resolutions=[32, 64, 96],
    )

    print("   📊 Performance across resolutions:")
    for res, metrics in performance_results.items():
        print(
            f"     {res}x{res}: L2 error = {metrics['l2_relative_error']:.4f}, "
            f"time = {metrics['inference_time'] * 1000:.1f}ms"
        )

    # Zero-shot super-resolution demonstration
    print("\n5️⃣ Demonstrating zero-shot super-resolution...")
    x_hr, y_pred_hr, y_true_hr = demonstrate_zero_shot_superresolution(
        model=model,
        base_resolution=64,
        target_resolution=128,
    )

    # Compute super-resolution error
    sr_error = float(
        jnp.sqrt(jnp.sum((y_pred_hr - y_true_hr) ** 2))
        / jnp.sqrt(jnp.sum(y_true_hr**2))
    )
    print(f"   📊 Super-resolution L2 error: {sr_error:.6f}")

    # Visualize results
    visualize_uno_results(
        x=x_hr,
        y_true=y_true_hr,
        y_pred=y_pred_hr,
        title="UNO Zero-Shot Super-Resolution (64→128)",
        save_path=output_dir / "uno_superresolution.png",
    )

    # Regular resolution results
    test_key = jax.random.key(777)
    x_test, y_test = create_synthetic_darcy_data(1, resolution, test_key)
    y_pred_test = model(x_test, deterministic=True)

    visualize_uno_results(
        x=x_test,
        y_true=y_test,
        y_pred=y_pred_test,
        title=f"UNO Results at {resolution}x{resolution}",
        save_path=output_dir / "uno_results.png",
    )

    # Architecture summary
    print("\n6️⃣ UNO Architecture Summary")
    print("=" * 40)
    print(f"   • Input channels: {model.input_channels}")
    print(f"   • Output channels: {model.output_channels}")
    print(f"   • Hidden channels: {model.hidden_channels}")
    print(f"   • Fourier modes: {model.modes}")
    print(f"   • U-Net layers: {model.n_layers}")
    print(
        f"   • Spectral layers: {len(model.spectral_layers) if model.use_spectral else 0}"
    )
    print(f"   • Total parameters: {param_count:,}")
    print("   • Uses Opifex framework: ✅")

    # Summary
    print("\n✅ UNO Demonstration Complete!")
    print("=" * 65)
    print("🔬 Scientific Impact:")
    print("   • U-Net architecture enables multi-scale feature extraction")
    print("   • Spectral convolutions provide global operator learning")
    print("   • Skip connections preserve spatial details across scales")
    print("   • Zero-shot super-resolution demonstrates true operator learning")
    print("   • Integrated with Opifex framework for extensibility")

    print("\n🛠️  Opifex Framework Integration:")
    print("   • Uses opifex.neural.operators.specialized.uno.UNeuralOperator")
    print("   • Follows Opifex framework patterns and best practices")
    print("   • Compatible with existing training infrastructure")
    print("   • Leverages framework's Fourier spectral convolutions")

    print("\n🚀 Next Steps:")
    print("   • Apply to real PDE datasets (PDEBench, etc.)")
    print("   • Compare with other Opifex neural operators (FNO, DeepONet)")
    print("   • Extend to 3D problems and irregular geometries")
    print("   • Integrate with physics-informed training")


if __name__ == "__main__":
    main()
