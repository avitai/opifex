#!/usr/bin/env python3
"""Simple Spherical FNO for Climate Modeling - Neural Operator Example.

This example demonstrates Spherical FNO functionality using the Opifex framework
with JAX/Flax NNX for spherical domain problems like climate modeling.
"""

import argparse
import time
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax  # type: ignore[import-untyped]
from flax import nnx

# Opifex imports
from opifex.data.loaders import create_shallow_water_loader
from opifex.neural.operators.fno.spherical import create_climate_sfno


def setup_experiment(args):
    """Setup experiment configuration and output directory."""
    output_dir = Path("examples_output/sfno_climate_simple")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🌍 Opifex Neural Operator Example: Simple Spherical FNO for Climate")
    print("=" * 80)
    print(f"📁 Output directory: {output_dir}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"💻 JAX backend: {jax.default_backend()}")
    print(f"🔧 JAX devices: {jax.devices()}")

    return output_dir


def setup_dataset_and_model(args):
    """Setup dataset and model components."""
    # Setup dataset with Grain loaders
    print("\n📊 Setting up Climate dataset with Grain...")
    train_loader = create_shallow_water_loader(
        n_samples=50,  # Small for demo
        batch_size=args.batch_size,
        resolution=args.resolution,
        shuffle=True,
        seed=args.seed + 3000,
        worker_count=0,
    )

    test_loader = create_shallow_water_loader(
        n_samples=10,
        batch_size=args.batch_size,
        resolution=args.resolution,
        shuffle=False,
        seed=args.seed + 4000,
        worker_count=0,
    )

    print("   ✅ Dataset: Shallow Water Equations (Climate Proxy, Grain streaming)")
    print(f"   ✅ Resolution: {args.resolution}x{args.resolution}")
    print("   ✅ Training samples: 50, Test samples: 10")

    # Setup model
    print("\n🏗️ Setting up Spherical FNO model...")
    model = create_climate_sfno(
        in_channels=2,  # height, vorticity (from shallow water dataset)
        out_channels=2,  # next state
        lmax=8,  # Small for demo
        rngs=nnx.Rngs(args.seed),
    )

    print("   ✅ Model: Spherical FNO")
    print("   ✅ Input/Output channels: 2")
    print("   ✅ Max spherical harmonic degree: 8")

    # Setup optimizer
    print("\n⚙️ Setting up optimizer...")
    optimizer = optax.adam(learning_rate=args.learning_rate)
    opt_state = optimizer.init(nnx.state(model))

    print("   ✅ Optimizer: Adam")
    print(f"   ✅ Learning rate: {args.learning_rate}")

    return (train_loader, test_loader), model, optimizer, opt_state


def prepare_data(loaders):
    """Prepare training and test data from loaders."""
    train_loader, test_loader = loaders

    # Collect training data
    X_train_list = []
    Y_train_list = []
    for batch in train_loader:
        X_train_list.append(batch["input"])
        Y_train_list.append(batch["output"])

    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)

    # Collect test data
    X_test_list = []
    Y_test_list = []
    for batch in test_loader:
        X_test_list.append(batch["input"])
        Y_test_list.append(batch["output"])

    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)

    # Ensure 4D tensors: (batch, channels, height, width)
    if X_train.ndim == 3:
        X_train = X_train[:, None, :, :]
        Y_train = Y_train[:, None, :, :]
    if X_test.ndim == 3:
        X_test = X_test[:, None, :, :]
        Y_test = Y_test[:, None, :, :]

    return X_train, Y_train, X_test, Y_test


def train_model(model, optimizer, opt_state, X_train, Y_train, args):
    """Train the model and return final optimizer state."""

    def train_step(batch_x, batch_y):
        def loss_fn(params):
            nnx.update(model, params)
            predictions = model(batch_x)  # type: ignore[arg-type]
            return jnp.mean((predictions - batch_y) ** 2)

        # Get current state
        model_state = nnx.state(model)

        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(model_state)  # type: ignore[arg-type]

        # Update optimizer state
        updates, new_opt_state = optimizer.update(grads, opt_state)

        # Apply updates to get new model state
        new_model_state = optax.apply_updates(model_state, updates)  # type: ignore[arg-type]

        # Update model with new state
        nnx.update(model, new_model_state)

        return loss, new_opt_state

    print("\n🚀 Starting training...")
    start_time = time.time()

    batch_size = args.batch_size
    n_train = X_train.shape[0]

    for epoch in range(args.epochs):
        epoch_losses = []

        # Simple batching
        for i in range(0, n_train, batch_size):
            end_idx = min(i + batch_size, n_train)
            batch_x = X_train[i:end_idx]
            batch_y = Y_train[i:end_idx]

            loss, opt_state = train_step(batch_x, batch_y)
            epoch_losses.append(float(loss))

        avg_loss = np.mean(epoch_losses)
        elapsed = time.time() - start_time
        print(
            f"   Epoch {epoch + 1:2d}/{args.epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s"
        )

    return start_time


def evaluate_model(model, X_test, Y_test):
    """Evaluate the trained model."""
    print("\n🔍 Final evaluation...")
    predictions = model(X_test)
    test_mse = float(jnp.mean((predictions - Y_test) ** 2))
    # Compute relative L2 error with proper tensor handling
    pred_diff = predictions - Y_test

    # Flatten spatial and channel dimensions for each sample
    if pred_diff.ndim == 4:
        # 4D tensor: (batch, channels, height, width) -> (batch, -1)
        pred_diff_flat = pred_diff.reshape(pred_diff.shape[0], -1)
        Y_test_flat = Y_test.reshape(Y_test.shape[0], -1)
    else:
        # 3D tensor: (batch, height, width) -> (batch, -1)
        pred_diff_flat = pred_diff.reshape(pred_diff.shape[0], -1)
        Y_test_flat = Y_test.reshape(Y_test.shape[0], -1)

    # Compute L2 norm for each sample (axis=1 gives vector norm)
    test_rel_l2 = float(
        jnp.mean(
            jnp.linalg.norm(pred_diff_flat, axis=1)
            / jnp.linalg.norm(Y_test_flat, axis=1)
        )
    )

    print(f"   ✅ Test MSE: {test_mse:.6f}")
    print(f"   ✅ Test Relative L2: {test_rel_l2:.6f}")

    return predictions


def create_visualization(predictions, X_test, Y_test, output_dir):
    """Create and save visualization."""
    print("\n📊 Generating visualization...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Spherical FNO Climate Prediction", fontsize=14, fontweight="bold")

    # Show first test sample
    sample_idx = 0

    # Input
    im0 = axes[0].imshow(X_test[sample_idx, 0], cmap="RdBu_r", aspect="equal")
    axes[0].set_title("Input")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Ground truth
    im1 = axes[1].imshow(Y_test[sample_idx, 0], cmap="RdBu_r", aspect="equal")
    axes[1].set_title("Ground Truth")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Prediction
    im2 = axes[2].imshow(predictions[sample_idx, 0], cmap="RdBu_r", aspect="equal")
    axes[2].set_title("SFNO Prediction")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    # Error
    error = np.abs(predictions[sample_idx, 0] - Y_test[sample_idx, 0])
    im3 = axes[3].imshow(error, cmap="plasma", aspect="equal")
    axes[3].set_title("Absolute Error")
    axes[3].set_xlabel("Longitude")
    axes[3].set_ylabel("Latitude")
    plt.colorbar(im3, ax=axes[3], shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "sfno_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ✅ Visualization saved to {output_dir / 'sfno_results.png'}")


def main():
    """Main function to run the simple Spherical FNO climate example."""
    parser = argparse.ArgumentParser(description="Simple Spherical FNO Climate Example")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup experiment
    output_dir = setup_experiment(args)

    # Setup dataset, model, and optimizer
    loaders, model, optimizer, opt_state = setup_dataset_and_model(args)

    # Prepare data
    X_train, Y_train, X_test, Y_test = prepare_data(loaders)

    # Train model
    start_time = train_model(model, optimizer, opt_state, X_train, Y_train, args)

    # Evaluate model
    predictions = evaluate_model(model, X_test, Y_test)

    # Create visualization
    create_visualization(predictions, X_test, Y_test, output_dir)

    # Final summary
    training_time = time.time() - start_time
    print(
        f"\n🎉 Spherical FNO Climate example completed in {training_time:.2f} seconds!"
    )
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
