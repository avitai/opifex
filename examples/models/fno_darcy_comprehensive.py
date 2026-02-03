#!/usr/bin/env python3
"""
Comprehensive FNO for Darcy Flow - Opifex High-Level API Demo

========================================================================

This example showcases Opifex's high-level APIs for scientific ML:

1. **Grain Data Loaders**: On-demand PDE solution generation with
   create_darcy_loader() - no manual data management
2. **Pre-built FNO Model**: FourierNeuralOperator with grid embeddings
   built-in - no manual architecture
3. **Unified Trainer**: opifex.core.training.Trainer handles optimization,
   checkpointing, and metrics - no manual training loops
4. **Comprehensive Evaluation**: Automatic metrics collection and visualization

**Key Pattern**: Use Opifex's high-level APIs for rapid prototyping. The
framework handles data loading, model creation, training, and evaluation.
Focus on your problem, not boilerplate.

For custom physics losses, see the physics-informed examples in examples/training/.
For data generation details, see examples/data/.

Based on neuraloperator FNO examples but demonstrating Opifex convenience APIs.
"""

import argparse
import time
import warnings
from pathlib import Path
from typing import Any


warnings.filterwarnings("ignore")

import grain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from jax import random


try:
    from tqdm import tqdm  # type: ignore[import-untyped]
except ImportError:
    # Fallback for when tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        """Fallback tqdm function when tqdm is not available."""
        return iterable


# Opifex imports
from opifex.core.training import Trainer, TrainingConfig
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.base import FourierNeuralOperator


class FNOWithEmbedding(nnx.Module):
    """FNO model with built-in grid embedding.

    This is a convenience wrapper that shows how to combine the grid embedding
    with the FNO model so the Trainer can call it directly with model(x).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        hidden_channels: int,
        num_layers: int,
        grid_boundaries: list[list[float]],
        rngs: nnx.Rngs,
    ):
        super().__init__()

        # Grid embedding for positional encoding
        self.grid_embedding = GridEmbedding2D(
            in_channels=in_channels,
            grid_boundaries=grid_boundaries,
        )

        # FNO model
        self.fno = FourierNeuralOperator(
            in_channels=self.grid_embedding.out_channels,  # Input + grid coordinates
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with automatic grid embedding.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Output tensor (batch, out_channels, height, width)
        """
        # Convert to format expected by grid embedding: (batch, height, width, channels)
        x_grid_format = jnp.moveaxis(x, 1, -1)

        # Apply grid embedding
        x_embedded = self.grid_embedding(x_grid_format)

        # Convert back to FNO expected format: (batch, channels, height, width)
        x_fno_format = jnp.moveaxis(x_embedded, -1, 1)

        # Forward through FNO
        return self.fno(x_fno_format)


class ComprehensiveFNODarcy:
    """
    Comprehensive FNO implementation for Darcy flow problems.

    This class provides a complete implementation including:
    - Model architecture with grid embeddings
    - Training and evaluation pipelines
    - Multi-resolution support
    - Physics-informed loss functions
    - Comprehensive visualization
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        data_config: dict[str, Any],
        seed: int = 42,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.seed = seed

        # Initialize random key
        self.key = random.PRNGKey(seed)

        # Initialize components
        self.model: FourierNeuralOperator | None = None
        self.trainer: Trainer | None = None
        self.train_loader = None
        self.test_loader = None

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_relative_l2": [],
            "val_relative_l2": [],
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging and output directories."""
        self.output_dir = Path("examples_output/fno_darcy_comprehensive")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("🧠 Opifex Neural Operator Example: Comprehensive FNO for Darcy Flow")
        print("=" * 80)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎲 Random seed: {self.seed}")
        print(f"💻 JAX backend: {jax.default_backend()}")
        print(f"🔧 JAX devices: {jax.devices()}")

    def setup_dataset(self) -> tuple[grain.DataLoader, grain.DataLoader]:
        """Setup Darcy flow dataset with specified configuration."""
        print("\n📊 Setting up Darcy Flow dataset...")

        # Calculate number of samples based on batch size
        n_train = 200  # Reasonable size for demonstration
        n_test = 40
        batch_size = self.data_config["batch_size"]
        self.train_loader = create_darcy_loader(
            n_samples=n_train,
            batch_size=batch_size,
            resolution=self.data_config["resolution"],
            shuffle=True,
            seed=self.seed + 3000,
            worker_count=0,
        )
        self.test_loader = create_darcy_loader(
            n_samples=n_test,
            batch_size=batch_size,
            resolution=self.data_config["resolution"],
            shuffle=False,
            seed=self.seed + 4000,
            worker_count=0,
        )

        # Store batch size for later use
        self.batch_size = self.data_config["batch_size"]

        print(
            f"   ✅ Dataset resolution: {self.data_config['resolution']}x{self.data_config['resolution']}"
        )
        print(f"   ✅ Batch size: {self.data_config['batch_size']}")
        print(f"   ✅ Training samples: {n_train}")
        print(f"   ✅ Test samples: {n_test}")

        return self.train_loader, self.test_loader

    def setup_model(self) -> nnx.Module:
        """Setup FNO model with grid embedding.

        This demonstrates using the FNOWithEmbedding wrapper so the model
        can be used directly with the Trainer.
        """
        print("\n🏗️ Setting up FNO model...")

        # Create FNO with built-in grid embedding
        self.model = FNOWithEmbedding(
            in_channels=self.model_config["in_channels"],
            out_channels=self.model_config["out_channels"],
            modes=self.model_config["modes"],
            hidden_channels=self.model_config["width"],
            num_layers=self.model_config["n_layers"],
            grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
            rngs=nnx.Rngs(self.seed),
        )

        print(
            f"   ✅ Model input channels: {self.model_config['in_channels']} (grid embedding adds +2)"
        )
        print(f"   ✅ Model output channels: {self.model_config['out_channels']}")
        print(f"   ✅ Fourier modes: {self.model_config['modes']}")
        print(f"   ✅ Hidden width: {self.model_config['width']}")
        print(f"   ✅ Number of layers: {self.model_config['n_layers']}")

        return self.model

    def setup_trainer(self) -> Trainer:
        """Setup unified trainer with physics-informed configuration."""
        print("\n⚙️ Setting up trainer...")

        if self.model is None:
            raise ValueError("Model must be setup before trainer")

        # Create training configuration
        config = TrainingConfig(
            num_epochs=self.training_config["num_epochs"],
            batch_size=self.data_config["batch_size"],
            learning_rate=self.training_config["learning_rate"],
            validation_frequency=1,  # Validate every epoch
            checkpoint_frequency=10,
            verbose=False,  # We'll handle progress bars ourselves
        )

        # Create unified trainer
        rngs = nnx.Rngs(self.seed)
        self.trainer = Trainer(model=self.model, config=config, rngs=rngs)

        print("   ✅ Trainer: Unified Opifex Trainer")
        print("   ✅ Optimizer: Adam with automatic state management")
        print(f"   ✅ Learning rate: {config.learning_rate}")
        print("   ✅ Gradient clipping: 1.0 (via optax chain)")

        return self.trainer

    def forward(self, x: jax.Array) -> jax.Array:
        """Forward pass - just call the model!

        This is a helper for evaluation. During training, the Trainer handles
        the forward pass automatically by calling model(x).

        Thanks to the FNOWithEmbedding wrapper, we don't need to manually
        handle grid embeddings anymore!
        """
        if self.model is None:
            raise ValueError("Model must be setup before forward pass")

        return self.model(x)

    # Note: Custom training steps removed! The unified Trainer handles all of this.
    # For custom physics losses, see examples/training/physics_informed_*.py

    def get_train_batches(self):
        """Generate training batches."""
        # Get training data
        if self.train_loader is None:
            raise ValueError("Dataset must be setup before getting training data")
        for batch in self.train_loader:
            # Add channel dimension if needed: (batch, h, w) -> (batch, 1, h, w)
            batch_x = batch["input"]
            batch_y = batch["output"]
            if batch_x.ndim == 3:  # (batch, h, w)
                batch_x = jnp.expand_dims(batch_x, axis=1)  # (batch, 1, h, w)
            if batch_y.ndim == 3:  # (batch, h, w)
                batch_y = jnp.expand_dims(batch_y, axis=1)  # (batch, 1, h, w)
            yield batch_x, batch_y

    def get_test_batches(self):
        """Generate test batches."""
        # Get test data
        if self.test_loader is None:
            raise ValueError("Dataset must be setup before getting test data")
        for batch in self.test_loader:
            # Add channel dimension if needed: (batch, h, w) -> (batch, 1, h, w)
            batch_x = batch["input"]
            batch_y = batch["output"]
            if batch_x.ndim == 3:  # (batch, h, w)
                batch_x = jnp.expand_dims(batch_x, axis=1)  # (batch, 1, h, w)
            if batch_y.ndim == 3:  # (batch, h, w)
                batch_y = jnp.expand_dims(batch_y, axis=1)  # (batch, 1, h, w)
            yield batch_x, batch_y

    # Note: Manual epoch training removed! The Trainer.train() method handles this.
    # It automatically batches data, computes losses, optimizes, and collects metrics.

    def train(self) -> dict[str, list[float]]:
        """Full training loop using Opifex's Trainer methods.

        This demonstrates using the Trainer's training_step() and validation_step()
        methods with Grain data loaders. The Trainer handles gradients and optimization!
        """
        if self.trainer is None:
            raise ValueError(
                "Trainer must be setup before training. Call setup_trainer() first."
            )

        print("\n🚀 Starting training...")
        print("   (Trainer handles gradients, optimization, and state)")

        start_time = time.time()
        best_val_loss = float("inf")

        for epoch in range(self.training_config["num_epochs"]):
            epoch_start = time.time()

            # Training epoch - iterate through batches
            train_losses = []
            for batch_x, batch_y in self.get_train_batches():
                # Trainer's training_step handles gradients and optimization!
                loss, _ = self.trainer.training_step(batch_x, batch_y)
                train_losses.append(float(loss))

            avg_train_loss = np.mean(train_losses) if train_losses else 0.0

            # Validation epoch
            val_losses = []
            for batch_x, batch_y in self.get_test_batches():
                # Trainer's validation_step (no gradients)
                loss, _ = self.trainer.validation_step(batch_x, batch_y)
                val_losses.append(float(loss))

            avg_val_loss = np.mean(val_losses) if val_losses else 0.0

            # Track best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_model_state = nnx.state(self.model)

            epoch_time = time.time() - epoch_start

            # Log progress
            if epoch % self.training_config.get("log_every", 1) == 0:
                print(
                    f"Epoch {epoch:3d}/{self.training_config['num_epochs']} "
                    f"({epoch_time:.1f}s) | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}"
                )

            # Store history
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)

        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.1f}s")
        print(f"📊 Best validation loss: {best_val_loss:.4f}")

        # Restore best model
        if hasattr(self, "best_model_state"):
            nnx.update(self.model, self.best_model_state)

        return self.history

    def evaluate_comprehensive(self) -> dict[str, Any]:
        """Comprehensive evaluation with detailed metrics.

        This demonstrates how to evaluate after training with the Opifex Trainer.
        The framework makes predictions easy - just call the model!
        """
        print("\n📈 Running comprehensive evaluation...")

        results = {"metrics": {}, "predictions": [], "targets": [], "inputs": []}

        # Evaluate on test set
        all_mse = []
        all_rel_l2 = []

        test_batches = self.get_test_batches()
        for batch_x, batch_y in test_batches:
            # Predictions using our forward helper (or just call self.model(batch_x)!)
            predictions = self.forward(batch_x)

            # Store for visualization
            results["predictions"].append(predictions)
            results["targets"].append(batch_y)
            results["inputs"].append(batch_x)

            # Compute simple metrics
            mse = float(jnp.mean((predictions - batch_y) ** 2))
            rel_l2 = float(
                jnp.mean(
                    jnp.linalg.norm(predictions - batch_y, axis=(-2, -1))
                    / jnp.linalg.norm(batch_y, axis=(-2, -1))
                )
            )

            all_mse.append(mse)
            all_rel_l2.append(rel_l2)

        # Average metrics
        results["metrics"]["mse"] = np.mean(all_mse)
        results["metrics"]["relative_l2"] = np.mean(all_rel_l2)

        print(f"   📊 Test MSE: {results['metrics']['mse']:.6f}")
        print(f"   📊 Test Relative L2: {results['metrics']['relative_l2']:.6f}")

        return results

    def visualize_results(self, results: dict[str, Any]):
        """Create comprehensive visualizations."""
        print("\n🎨 Creating visualizations...")

        # Training curves
        self._plot_training_curves()

        # Sample predictions
        self._plot_sample_predictions(results)

        # Error analysis
        self._plot_error_analysis(results)

        # Model architecture visualization
        self._plot_model_summary()

        print(f"   💾 Visualizations saved to {self.output_dir}")

    def _plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("FNO Training Progress", fontsize=16, fontweight="bold")

        epochs = range(len(self.history["train_loss"]))

        # Total loss
        axes[0, 0].plot(
            epochs, self.history["train_loss"], label="Train", color="blue", alpha=0.7
        )
        axes[0, 0].plot(
            epochs, self.history["val_loss"], label="Validation", color="red", alpha=0.7
        )
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MSE
        axes[0, 1].plot(
            epochs, self.history["train_mse"], label="Train", color="blue", alpha=0.7
        )
        axes[0, 1].plot(
            epochs, self.history["val_mse"], label="Validation", color="red", alpha=0.7
        )
        axes[0, 1].set_title("Mean Squared Error")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MSE")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")

        # Relative L2
        axes[1, 0].plot(
            epochs,
            self.history["train_relative_l2"],
            label="Train",
            color="blue",
            alpha=0.7,
        )
        axes[1, 0].plot(
            epochs,
            self.history["val_relative_l2"],
            label="Validation",
            color="red",
            alpha=0.7,
        )
        axes[1, 0].set_title("Relative L2 Error")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Relative L2")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning rate (if available)
        axes[1, 1].text(
            0.5,
            0.5,
            "Model Architecture:\n\n"
            f"Input Channels: {self.model_config['in_channels']} + 2 (grid)\n"
            f"Output Channels: {self.model_config['out_channels']}\n"
            f"Fourier Modes: {self.model_config['modes']}\n"
            f"Hidden Width: {self.model_config['width']}\n"
            f"Layers: {self.model_config['n_layers']}",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray"},
        )
        axes[1, 1].set_title("Model Configuration")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_sample_predictions(self, results: dict[str, Any]):
        """Plot sample predictions vs targets."""
        # Get first batch for visualization
        predictions = results["predictions"][0][:4]  # First 4 samples
        targets = results["targets"][0][:4]
        inputs = results["inputs"][0][:4]

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle("FNO Predictions vs Ground Truth", fontsize=16, fontweight="bold")

        for i in range(4):
            # Input (permeability)
            im1 = axes[i, 0].imshow(inputs[i, :, :, 0], cmap="viridis")
            axes[i, 0].set_title("Input (Permeability)" if i == 0 else "")
            axes[i, 0].axis("off")
            if i == 0:
                plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)

            # Target
            im2 = axes[i, 1].imshow(targets[i, :, :, 0], cmap="RdBu_r")
            axes[i, 1].set_title("Ground Truth" if i == 0 else "")
            axes[i, 1].axis("off")
            if i == 0:
                plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)

            # Prediction
            im3 = axes[i, 2].imshow(predictions[i, :, :, 0], cmap="RdBu_r")
            axes[i, 2].set_title("FNO Prediction" if i == 0 else "")
            axes[i, 2].axis("off")
            if i == 0:
                plt.colorbar(im3, ax=axes[i, 2], shrink=0.8)

            # Error
            error = np.abs(predictions[i, :, :, 0] - targets[i, :, :, 0])
            im4 = axes[i, 3].imshow(error, cmap="Reds")
            axes[i, 3].set_title("Absolute Error" if i == 0 else "")
            axes[i, 3].axis("off")
            if i == 0:
                plt.colorbar(im4, ax=axes[i, 3], shrink=0.8)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "sample_predictions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_error_analysis(self, results: dict[str, Any]):
        """Plot detailed error analysis."""
        # Compute per-sample errors
        all_relative_errors = []
        all_mse_errors = []

        for pred_batch, target_batch in zip(
            results["predictions"], results["targets"], strict=False
        ):
            for i in range(pred_batch.shape[0]):
                pred = pred_batch[i, :, :, 0]
                target = target_batch[i, :, :, 0]

                # Relative L2 error
                rel_error = float(
                    jnp.linalg.norm(pred - target) / jnp.linalg.norm(target)
                )
                # Check for infinite or NaN values
                if jnp.isfinite(rel_error):
                    all_relative_errors.append(rel_error)

                # MSE
                mse_error = float(jnp.mean((pred - target) ** 2))
                if jnp.isfinite(mse_error):
                    all_mse_errors.append(mse_error)

        # Ensure we have some valid data
        if len(all_relative_errors) == 0:
            all_relative_errors = [0.0]
        if len(all_mse_errors) == 0:
            all_mse_errors = [0.0]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Error Analysis", fontsize=16, fontweight="bold")

        # Error distribution
        axes[0, 0].hist(
            all_relative_errors, bins=30, alpha=0.7, color="blue", edgecolor="black"
        )
        axes[0, 0].set_title("Relative L2 Error Distribution")
        axes[0, 0].set_xlabel("Relative L2 Error")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

        # Error vs sample index
        axes[0, 1].plot(all_relative_errors, alpha=0.7, color="red")
        axes[0, 1].set_title("Relative L2 Error per Sample")
        axes[0, 1].set_xlabel("Sample Index")
        axes[0, 1].set_ylabel("Relative L2 Error")
        axes[0, 1].grid(True, alpha=0.3)

        # MSE distribution
        axes[1, 0].hist(
            all_mse_errors, bins=30, alpha=0.7, color="green", edgecolor="black"
        )
        axes[1, 0].set_title("MSE Distribution")
        axes[1, 0].set_xlabel("MSE")
        axes[1, 0].set_ylabel("Frequency")
        if len(all_mse_errors) > 1 and max(all_mse_errors) > min(all_mse_errors):
            axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics - handle potential issues with empty lists
        if len(all_relative_errors) > 0 and len(all_mse_errors) > 0:
            stats_text = f"""Error Statistics:

Relative L2 Error:
  Mean: {np.mean(all_relative_errors):.4f}
  Std:  {np.std(all_relative_errors):.4f}
  Min:  {np.min(all_relative_errors):.4f}
  Max:  {np.max(all_relative_errors):.4f}
  Count: {len(all_relative_errors)}

MSE:
  Mean: {np.mean(all_mse_errors):.2e}
  Std:  {np.std(all_mse_errors):.2e}
  Min:  {np.min(all_mse_errors):.2e}
  Max:  {np.max(all_mse_errors):.2e}
  Count: {len(all_mse_errors)}"""
        else:
            stats_text = "Error Statistics:\n\nNo valid error data available"

        axes[1, 1].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray"},
        )
        axes[1, 1].set_title("Error Statistics")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "error_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_model_summary(self):
        """Plot model architecture summary."""
        # Create a simple architecture diagram
        _, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Architecture flow
        layers = [
            f"Input\n({self.model_config['in_channels']} channels)",
            "Grid Embedding\n(+2 coordinates)",
            f"FNO Layers\n({self.model_config['n_layers']} layers)",
            f"Fourier Modes\n({self.model_config['modes']}x{self.model_config['modes']})",
            f"Hidden Width\n({self.model_config['width']})",
            f"Output\n({self.model_config['out_channels']} channels)",
        ]

        y_positions = np.linspace(0.8, 0.2, len(layers))

        for i, (layer, y) in enumerate(zip(layers, y_positions, strict=False)):
            # Draw box
            bbox = {"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7}
            ax.text(
                0.5,
                y,
                layer,
                ha="center",
                va="center",
                bbox=bbox,
                fontsize=12,
                fontweight="bold",
            )

            # Draw arrow
            if i < len(layers) - 1:
                ax.annotate(
                    "",
                    xy=(0.5, y_positions[i + 1] + 0.05),
                    xytext=(0.5, y - 0.05),
                    arrowprops={"arrowstyle": "->", "lw": 2, "color": "black"},
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("FNO Model Architecture", fontsize=16, fontweight="bold")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "model_architecture.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Comprehensive FNO for Darcy Flow")
    parser.add_argument("--resolution", type=int, default=64, help="Grid resolution")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--modes", type=int, default=12, help="Fourier modes")
    parser.add_argument("--width", type=int, default=32, help="Hidden width")
    parser.add_argument("--layers", type=int, default=4, help="Number of FNO layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Configuration
    model_config = {
        "in_channels": 1,
        "out_channels": 1,
        "modes": args.modes,
        "width": args.width,
        "n_layers": args.layers,
    }

    training_config = {
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "grad_clip": 1.0,
        "patience": 15,
        "log_every": 5,
    }

    data_config = {
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "normalize": True,
        "test_split": 0.2,
    }

    # Create and run experiment
    experiment = ComprehensiveFNODarcy(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        seed=args.seed,
    )

    # Setup components
    experiment.setup_dataset()
    experiment.setup_model()
    experiment.setup_trainer()

    # Train model
    _ = experiment.train()

    # Comprehensive evaluation
    results = experiment.evaluate_comprehensive()

    # Create visualizations
    experiment.visualize_results(results)

    print("\n" + "=" * 80)
    print("🎉 Comprehensive FNO Darcy Flow example completed successfully!")
    print("=" * 80)
    print("📊 Final validation metrics:")
    print(f"   • MSE: {results['metrics']['mse']:.6f}")
    print(f"   • Relative L2: {results['metrics']['relative_l2']:.6f}")
    print(f"📁 Results saved to: {experiment.output_dir}")
    print("=" * 80)
    print("\n💡 Key Takeaway: With Opifex's high-level APIs, you get:")
    print("   - Grain data loaders (on-demand PDE solutions)")
    print("   - Pre-built FNO model (with grid embeddings)")
    print("   - Unified Trainer (automatic optimization)")
    print("   - Easy evaluation and visualization")
    print("=" * 80)


if __name__ == "__main__":
    main()
