#!/usr/bin/env python3
"""Comprehensive U-FNO for Turbulence Modeling - Neural Operator Example.

This example demonstrates U-FNO functionality for multi-scale turbulence modeling
using the Opifex framework with JAX/Flax NNX. Features include adaptive training,
physics-aware loss functions, and comprehensive turbulence analysis.
"""

import argparse
import time
import warnings
from pathlib import Path
from typing import Any


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax  # type: ignore[import-untyped]
from flax import nnx
from jax import random

from opifex.data.loaders.factory import create_burgers_loader

# Opifex imports
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.ufno import (
    create_turbulence_ufno,
)


class ComprehensiveUFNOTurbulence:
    """
    Comprehensive U-Neural Operator implementation for turbulent flow problems.

    This class provides a complete implementation including:
    - Multi-scale U-FNO architecture with hierarchical processing
    - Training and evaluation pipelines optimized for turbulence
    - Multi-resolution support for scale-aware learning
    - Physics-informed loss functions for energy conservation
    - Comprehensive visualization with multi-scale analysis
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
        self.model: nnx.Module | None = None
        self.optimizer: optax.GradientTransformation | None = None
        self.train_state = None
        self.train_loader = None
        self.test_loader = None

        # Training history with multi-scale metrics
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_relative_l2": [],
            "val_relative_l2": [],
            "train_energy_conservation": [],
            "val_energy_conservation": [],
            "train_vorticity_preservation": [],
            "val_vorticity_preservation": [],
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging and output directories."""
        self.output_dir = Path("examples_output/ufno_turbulence_comprehensive")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("🌪️ Opifex Neural Operator Example: Comprehensive U-FNO for Turbulence")
        print("=" * 80)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎲 Random seed: {self.seed}")
        print(f"💻 JAX backend: {jax.default_backend()}")
        print(f"🔧 JAX devices: {jax.devices()}")

    def setup_dataset(self):
        """Setup turbulent Burgers equation dataset with Grain loaders."""
        print("\n📊 Setting up Turbulent Burgers dataset with Grain...")

        # Create streaming data loaders
        n_train = 300
        n_test = 60
        batch_size = self.data_config["batch_size"]

        self.train_loader = create_burgers_loader(
            n_samples=n_train,
            batch_size=batch_size,
            dimension="2d",
            resolution=self.data_config["resolution"],
            viscosity_range=(0.001, 0.005),
            time_range=(0.0, 1.0),
            shuffle=True,
            seed=self.seed + 2000,
            worker_count=0,
        )

        self.test_loader = create_burgers_loader(
            n_samples=n_test,
            batch_size=batch_size,
            dimension="2d",
            resolution=self.data_config["resolution"],
            viscosity_range=(0.001, 0.005),
            time_range=(0.0, 1.0),
            shuffle=False,
            seed=self.seed + 3000,
            worker_count=0,
        )

        self.batch_size = batch_size

        print("   ✅ Dataset type: 2D Turbulent Burgers (Grain streaming)")
        print(
            f"   ✅ Resolution: {self.data_config['resolution']}x{self.data_config['resolution']}"
        )
        print(f"   ✅ Batch size: {batch_size}")
        print(f"   ✅ Training samples: {n_train}")
        print(f"   ✅ Test samples: {n_test}")
        print("   ✅ Viscosity: 0.001 (turbulent regime)")

        return self.train_loader, self.test_loader

    def setup_model(self) -> nnx.Module:
        """Setup U-FNO model with multi-scale architecture."""
        print("\n🏗️ Setting up U-FNO model...")

        # Grid embedding for positional encoding
        grid_embedding: GridEmbedding2D = GridEmbedding2D(
            in_channels=self.model_config["in_channels"],
            grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        )

        # Use specialized turbulence U-FNO configuration
        self.model = create_turbulence_ufno(
            in_channels=grid_embedding.out_channels,  # Input + grid coordinates
            out_channels=self.model_config["out_channels"],
            rngs=nnx.Rngs(self.seed),
        )

        # Store embedding separately
        self.grid_embedding = grid_embedding

        print("   ✅ Model type: U-FNO (Multi-scale)")
        print(
            f"   ✅ Input channels: {grid_embedding.out_channels} (data: {self.model_config['in_channels']} + grid: 2)"
        )
        print(f"   ✅ Output channels: {self.model_config['out_channels']}")
        print(f"   ✅ Fourier modes: {self.model_config['modes']}")
        print(f"   ✅ Hidden width: {self.model_config['width']}")
        print(f"   ✅ U-Net levels: {self.model_config['num_levels']}")
        print(f"   ✅ Downsample factor: {self.model_config['downsample_factor']}")

        return self.model

    def setup_optimizer(self) -> optax.GradientTransformation:
        """Setup optimizer and training state."""
        print("\n⚙️ Setting up optimizer...")

        # Create optimizer with learning rate schedule
        schedule = optax.cosine_decay_schedule(
            init_value=self.training_config["learning_rate"],
            decay_steps=self.training_config["num_epochs"]
            * self.training_config.get("steps_per_epoch", 100),
            alpha=0.1,
        )

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.training_config.get("grad_clip", 1.0)),
            optax.adam(learning_rate=schedule),
        )

        # Initialize optimizer state
        self.opt_state = self.optimizer.init(nnx.state(self.model))

        print("   ✅ Optimizer: Adam with cosine decay")
        print(f"   ✅ Learning rate: {self.training_config['learning_rate']}")
        print(f"   ✅ Gradient clipping: {self.training_config.get('grad_clip', 1.0)}")

        return self.optimizer

    def forward_with_embedding(self, x: jax.Array) -> jax.Array:
        """Forward pass with grid embedding."""
        # Input x is in format: (batch, channels, height, width)
        # Convert to format expected by grid embedding: (batch, height, width, channels)
        x_grid_format = jnp.moveaxis(x, 1, -1)

        # Apply grid embedding
        x_embedded = self.grid_embedding(x_grid_format)

        # Convert back to U-FNO expected format: (batch, channels, height, width)
        x_fno_format = jnp.moveaxis(x_embedded, -1, 1)

        # Forward through U-FNO
        return self.model(x_fno_format)

    def compute_losses(
        self, predictions: jax.Array, targets: jax.Array
    ) -> dict[str, float]:
        """Compute comprehensive loss including physics constraints.

        Calculates MSE, relative L2, energy conservation, and vorticity preservation
        losses for turbulence modeling applications.
        """
        # Basic losses
        mse_loss = jnp.mean((predictions - targets) ** 2)

        # Relative L2 loss
        relative_l2 = jnp.mean(
            jnp.linalg.norm(predictions - targets, axis=(1, 2, 3))
            / jnp.linalg.norm(targets, axis=(1, 2, 3))
        )

        # Physics-informed losses
        def compute_gradients(field):
            """Compute spatial gradients for physics metrics."""
            # field shape: (batch, channels, height, width)
            # Compute gradients in spatial dimensions
            grad_y = jnp.gradient(field, axis=2)  # df/dy
            grad_x = jnp.gradient(field, axis=3)  # df/dx
            return grad_x, grad_y

        # Energy conservation (kinetic energy should be preserved)
        pred_energy = jnp.mean(predictions**2, axis=(2, 3))  # (batch, channels)
        target_energy = jnp.mean(targets**2, axis=(2, 3))
        energy_conservation = jnp.mean(jnp.abs(pred_energy - target_energy))

        # Gradient preservation (for scalar field, track gradient magnitude)
        if predictions.shape[1] >= 1:  # Scalar field
            pred_grad_x, pred_grad_y = compute_gradients(predictions)
            target_grad_x, target_grad_y = compute_gradients(targets)

            # For scalar field, use gradient magnitude preservation
            pred_grad_mag = jnp.sqrt(pred_grad_x[:, 0] ** 2 + pred_grad_y[:, 0] ** 2)
            target_grad_mag = jnp.sqrt(
                target_grad_x[:, 0] ** 2 + target_grad_y[:, 0] ** 2
            )

            vorticity_preservation = jnp.mean(jnp.abs(pred_grad_mag - target_grad_mag))
        else:
            vorticity_preservation = 0.0

        return {
            "mse": float(mse_loss),
            "relative_l2": float(relative_l2),
            "energy_conservation": float(energy_conservation),
            "vorticity_preservation": float(vorticity_preservation),
        }

    def train_step(self, batch_x: jax.Array, batch_y: jax.Array) -> dict[str, float]:
        """Training step with physics-informed loss."""

        def loss_fn(params):
            # Temporarily update model with params
            nnx.update(self.model, params)

            # Forward pass
            predictions = self.forward_with_embedding(batch_x)

            # Compute losses
            losses = self.compute_losses(predictions, batch_y)

            # Combined loss with physics terms
            total_loss = (
                losses["mse"]
                + 0.1 * losses["energy_conservation"]
                + 0.05 * losses["vorticity_preservation"]
            )

            return total_loss, losses

        # Compute gradients
        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            nnx.state(self.model)
        )

        # Update parameters
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        nnx.update(self.model, optax.apply_updates(nnx.state(self.model), updates))

        return {
            "loss": float(loss),
            **losses,
        }

    def eval_step(self, batch_x: jax.Array, batch_y: jax.Array) -> dict[str, float]:
        """Evaluation step."""
        predictions = self.forward_with_embedding(batch_x)
        losses = self.compute_losses(predictions, batch_y)
        total_loss = (
            losses["mse"]
            + 0.1 * losses["energy_conservation"]
            + 0.05 * losses["vorticity_preservation"]
        )
        return {"loss": float(total_loss), **losses}

    def get_train_batches(self):
        """Get training data batches."""
        x_data, y_data = self.dataset.get_train_data()

        # Ensure 4D tensor: (batch, channels, height, width)
        if x_data.ndim == 3:
            x_data = x_data[:, None, :, :]  # Add channel dimension
        if y_data.ndim == 3:
            y_data = y_data[:, None, :, :]

        n_samples = x_data.shape[0]

        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            yield x_data[i:end_idx], y_data[i:end_idx]

    def get_test_batches(self):
        """Get test data batches."""
        x_data, y_data = self.dataset.get_test_data()

        # Ensure 4D tensor: (batch, channels, height, width)
        if x_data.ndim == 3:
            x_data = x_data[:, None, :, :]  # Add channel dimension
        if y_data.ndim == 3:
            y_data = y_data[:, None, :, :]

        n_samples = x_data.shape[0]

        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            yield x_data[i:end_idx], y_data[i:end_idx]

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        epoch_metrics: dict[str, list[float]] = {
            "loss": [],
            "mse": [],
            "relative_l2": [],
            "energy_conservation": [],
            "vorticity_preservation": [],
        }

        for batch_x, batch_y in self.get_train_batches():
            metrics = self.train_step(batch_x, batch_y)
            for key, value in metrics.items():
                epoch_metrics[key].append(value)

        # Average metrics
        return {key: float(np.mean(values)) for key, values in epoch_metrics.items()}

    def eval_epoch(self) -> dict[str, float]:
        """Evaluate for one epoch."""
        epoch_metrics: dict[str, list[float]] = {
            "loss": [],
            "mse": [],
            "relative_l2": [],
            "energy_conservation": [],
            "vorticity_preservation": [],
        }

        for batch_x, batch_y in self.get_test_batches():
            metrics = self.eval_step(batch_x, batch_y)
            for key, value in metrics.items():
                epoch_metrics[key].append(value)

        # Average metrics
        return {key: float(np.mean(values)) for key, values in epoch_metrics.items()}

    def train(self) -> dict[str, list[float]]:
        """Complete training loop."""
        print("\n🚀 Starting training...")
        print(f"   Epochs: {self.training_config['num_epochs']}")
        print(f"   Batch size: {self.data_config['batch_size']}")

        start_time = time.time()

        for epoch in range(self.training_config["num_epochs"]):
            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.eval_epoch()

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_mse"].append(train_metrics["mse"])
            self.history["val_mse"].append(val_metrics["mse"])
            self.history["train_relative_l2"].append(train_metrics["relative_l2"])
            self.history["val_relative_l2"].append(val_metrics["relative_l2"])
            self.history["train_energy_conservation"].append(
                train_metrics["energy_conservation"]
            )
            self.history["val_energy_conservation"].append(
                val_metrics["energy_conservation"]
            )
            self.history["train_vorticity_preservation"].append(
                train_metrics["vorticity_preservation"]
            )
            self.history["val_vorticity_preservation"].append(
                val_metrics["vorticity_preservation"]
            )

            # Progress logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(
                    f"   Epoch {epoch + 1:3d}/{self.training_config['num_epochs']:3d} | "
                    f"Train Loss: {train_metrics['loss']:.6f} | "
                    f"Val Loss: {val_metrics['loss']:.6f} | "
                    f"Val Rel L2: {val_metrics['relative_l2']:.6f} | "
                    f"Energy: {val_metrics['energy_conservation']:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )

        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds")

        return self.history

    def evaluate_comprehensive(self) -> dict[str, Any]:
        """Comprehensive evaluation with multi-scale metrics."""
        print("\n🔍 Running comprehensive evaluation...")

        # Get sample test data for detailed analysis
        X_test, Y_test = self.dataset.get_test_data()

        # Ensure 4D tensor
        if X_test.ndim == 3:
            X_test = X_test[:, None, :, :]
        if Y_test.ndim == 3:
            Y_test = Y_test[:, None, :, :]

        # Get predictions
        predictions = self.forward_with_embedding(X_test)

        # Overall metrics
        overall_metrics = self.compute_losses(predictions, Y_test)

        # Per-sample analysis
        per_sample_errors = []
        for i in range(X_test.shape[0]):
            sample_metrics = self.compute_losses(
                predictions[i : i + 1], Y_test[i : i + 1]
            )
            per_sample_errors.append(sample_metrics["relative_l2"])

        results = {
            "overall_metrics": overall_metrics,
            "per_sample_errors": per_sample_errors,
            "mean_error": float(np.mean(per_sample_errors)),
            "std_error": float(np.std(per_sample_errors)),
            "predictions": predictions,
            "targets": Y_test,
            "inputs": X_test,
        }

        print(
            f"   ✅ Mean Relative L2 Error: {results['mean_error']:.6f} ± {results['std_error']:.6f}"
        )
        print(
            f"   ✅ Energy Conservation: {overall_metrics['energy_conservation']:.6f}"
        )
        print(
            f"   ✅ Vorticity Preservation: {overall_metrics['vorticity_preservation']:.6f}"
        )

        return results

    def visualize_results(self, results: dict[str, Any]):
        """Comprehensive visualization of U-FNO results."""
        print("\n📊 Generating comprehensive visualizations...")

        # Create subplots
        self._plot_training_curves()
        self._plot_sample_predictions(results)
        self._plot_multi_scale_analysis(results)
        self._plot_error_analysis(results)
        self._plot_model_summary()

        print(f"   ✅ All plots saved to {self.output_dir}")

    def _plot_loss_curves(self, axes, epochs):
        """Plot loss curves subplot."""
        axes[0, 0].plot(
            epochs, self.history["train_loss"], "b-", label="Train", linewidth=2
        )
        axes[0, 0].plot(
            epochs, self.history["val_loss"], "r--", label="Validation", linewidth=2
        )
        axes[0, 0].set_title("Total Loss", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale("log")

    def _plot_mse_curves(self, axes, epochs):
        """Plot MSE curves subplot."""
        axes[0, 1].plot(
            epochs, self.history["train_mse"], "b-", label="Train", linewidth=2
        )
        axes[0, 1].plot(
            epochs, self.history["val_mse"], "r--", label="Validation", linewidth=2
        )
        axes[0, 1].set_title("MSE Loss", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MSE")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")

    def _plot_relative_l2_curves(self, axes, epochs):
        """Plot relative L2 error curves subplot."""
        axes[0, 2].plot(
            epochs, self.history["train_relative_l2"], "b-", label="Train", linewidth=2
        )
        axes[0, 2].plot(
            epochs,
            self.history["val_relative_l2"],
            "r--",
            label="Validation",
            linewidth=2,
        )
        axes[0, 2].set_title("Relative L2 Error", fontweight="bold")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Relative L2")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    def _plot_physics_conservation_metrics(self, axes, epochs):
        """Plot energy conservation and vorticity preservation subplots."""
        # Energy conservation
        axes[1, 0].plot(
            epochs,
            self.history["train_energy_conservation"],
            "b-",
            label="Train",
            linewidth=2,
        )
        axes[1, 0].plot(
            epochs,
            self.history["val_energy_conservation"],
            "r--",
            label="Validation",
            linewidth=2,
        )
        axes[1, 0].set_title("Energy Conservation Error", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Energy Error")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale("log")

        # Vorticity preservation
        axes[1, 1].plot(
            epochs,
            self.history["train_vorticity_preservation"],
            "b-",
            label="Train",
            linewidth=2,
        )
        axes[1, 1].plot(
            epochs,
            self.history["val_vorticity_preservation"],
            "r--",
            label="Validation",
            linewidth=2,
        )
        axes[1, 1].set_title("Vorticity Preservation Error", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Vorticity Error")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale("log")

    def _plot_final_metrics_summary(self, axes):
        """Plot final metrics summary subplot."""
        final_val_loss = self.history["val_loss"][-1]
        final_val_rel_l2 = self.history["val_relative_l2"][-1]
        final_energy = self.history["val_energy_conservation"][-1]
        final_vorticity = self.history["val_vorticity_preservation"][-1]

        axes[1, 2].text(
            0.1,
            0.8,
            f"Final Validation Loss: {final_val_loss:.6f}",
            fontsize=12,
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].text(
            0.1,
            0.7,
            f"Final Rel L2 Error: {final_val_rel_l2:.6f}",
            fontsize=12,
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].text(
            0.1,
            0.6,
            f"Energy Conservation: {final_energy:.6f}",
            fontsize=12,
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].text(
            0.1,
            0.5,
            f"Vorticity Preservation: {final_vorticity:.6f}",
            fontsize=12,
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].set_title("Final Metrics", fontweight="bold")
        axes[1, 2].axis("off")

    def _plot_training_curves(self):
        """Plot training curves with physics metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "U-FNO Training Progress - Multi-Scale Turbulence",
            fontsize=16,
            fontweight="bold",
        )

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Plot all sections using helper functions
        self._plot_loss_curves(axes, epochs)
        self._plot_mse_curves(axes, epochs)
        self._plot_relative_l2_curves(axes, epochs)
        self._plot_physics_conservation_metrics(axes, epochs)
        self._plot_final_metrics_summary(axes)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_sample_predictions(self, results: dict[str, Any]):
        """Plot sample predictions with multi-scale analysis."""
        predictions = results["predictions"]
        targets = results["targets"]
        inputs = results["inputs"]

        # Select first few samples for visualization
        n_samples = min(3, predictions.shape[0])

        fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
        if n_samples == 1:
            axes = axes[None, :]  # Add batch dimension for consistent indexing

        fig.suptitle(
            "U-FNO Turbulence Predictions - Sample Results",
            fontsize=16,
            fontweight="bold",
        )

        for i in range(n_samples):
            # Input
            im0 = axes[i, 0].imshow(inputs[i, 0], cmap="RdBu_r", aspect="equal")
            axes[i, 0].set_title(f"Input (Sample {i + 1})", fontweight="bold")
            axes[i, 0].set_xlabel("x")
            axes[i, 0].set_ylabel("y")
            plt.colorbar(im0, ax=axes[i, 0], shrink=0.8)

            # Ground truth
            im1 = axes[i, 1].imshow(targets[i, 0], cmap="RdBu_r", aspect="equal")
            axes[i, 1].set_title(f"Ground Truth (Sample {i + 1})", fontweight="bold")
            axes[i, 1].set_xlabel("x")
            axes[i, 1].set_ylabel("y")
            plt.colorbar(im1, ax=axes[i, 1], shrink=0.8)

            # Prediction
            im2 = axes[i, 2].imshow(predictions[i, 0], cmap="RdBu_r", aspect="equal")
            axes[i, 2].set_title(
                f"U-FNO Prediction (Sample {i + 1})", fontweight="bold"
            )
            axes[i, 2].set_xlabel("x")
            axes[i, 2].set_ylabel("y")
            plt.colorbar(im2, ax=axes[i, 2], shrink=0.8)

            # Error
            error = np.abs(predictions[i, 0] - targets[i, 0])
            im3 = axes[i, 3].imshow(error, cmap="plasma", aspect="equal")
            axes[i, 3].set_title(f"Absolute Error (Sample {i + 1})", fontweight="bold")
            axes[i, 3].set_xlabel("x")
            axes[i, 3].set_ylabel("y")
            plt.colorbar(im3, ax=axes[i, 3], shrink=0.8)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "sample_predictions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_multi_scale_analysis(self, results: dict[str, Any]):
        """Plot multi-scale analysis specific to U-FNO architecture."""
        predictions = results["predictions"]
        targets = results["targets"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("U-FNO Multi-Scale Analysis", fontsize=16, fontweight="bold")

        # Spectral analysis
        pred_fft = np.abs(np.fft.fft2(predictions[0, 0]))
        target_fft = np.abs(np.fft.fft2(targets[0, 0]))

        # Frequency content comparison
        axes[0, 0].semilogy(
            np.mean(pred_fft, axis=0), "b-", label="U-FNO Prediction", linewidth=2
        )
        axes[0, 0].semilogy(
            np.mean(target_fft, axis=0), "r--", label="Ground Truth", linewidth=2
        )
        axes[0, 0].set_title("Frequency Content (x-direction)", fontweight="bold")
        axes[0, 0].set_xlabel("Frequency Mode")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].semilogy(
            np.mean(pred_fft, axis=1), "b-", label="U-FNO Prediction", linewidth=2
        )
        axes[0, 1].semilogy(
            np.mean(target_fft, axis=1), "r--", label="Ground Truth", linewidth=2
        )
        axes[0, 1].set_title("Frequency Content (y-direction)", fontweight="bold")
        axes[0, 1].set_xlabel("Frequency Mode")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Multi-scale error analysis
        # Compute error at different scales by applying different levels of smoothing
        scales = [1, 2, 4, 8]
        scale_errors = []

        for scale in scales:
            # Simple downsampling for different scales
            step = scale
            pred_scaled = predictions[0, 0, ::step, ::step]
            target_scaled = targets[0, 0, ::step, ::step]

            error = np.mean((pred_scaled - target_scaled) ** 2)
            scale_errors.append(error)

        axes[1, 0].loglog(scales, scale_errors, "go-", linewidth=2, markersize=8)
        axes[1, 0].set_title("Multi-Scale Error Analysis", fontweight="bold")
        axes[1, 0].set_xlabel("Scale (downsampling factor)")
        axes[1, 0].set_ylabel("MSE")
        axes[1, 0].grid(True, alpha=0.3)

        # Physics conservation analysis
        # Energy spectrum
        pred_energy_spectrum = np.mean(pred_fft**2, axis=(0, 1))
        target_energy_spectrum = np.mean(target_fft**2, axis=(0, 1))

        freqs = np.fft.fftfreq(predictions.shape[-1])
        positive_freqs = freqs[freqs > 0]

        axes[1, 1].loglog(
            positive_freqs,
            pred_energy_spectrum[: len(positive_freqs)],
            "b-",
            label="U-FNO Prediction",
            linewidth=2,
        )
        axes[1, 1].loglog(
            positive_freqs,
            target_energy_spectrum[: len(positive_freqs)],
            "r--",
            label="Ground Truth",
            linewidth=2,
        )
        axes[1, 1].set_title("Energy Spectrum", fontweight="bold")
        axes[1, 1].set_xlabel("Frequency")
        axes[1, 1].set_ylabel("Energy")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "multiscale_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_error_analysis(self, results: dict[str, Any]):
        """Plot comprehensive error analysis."""
        per_sample_errors = results["per_sample_errors"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("U-FNO Error Analysis", fontsize=16, fontweight="bold")

        # Error distribution
        axes[0, 0].hist(
            per_sample_errors, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].axvline(
            np.mean(per_sample_errors),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(per_sample_errors):.4f}",
            linewidth=2,
        )
        axes[0, 0].set_title("Error Distribution", fontweight="bold")
        axes[0, 0].set_xlabel("Relative L2 Error")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Error vs sample index
        axes[0, 1].plot(
            per_sample_errors, "o-", markersize=6, linewidth=2, color="orange"
        )
        axes[0, 1].set_title("Error vs Sample Index", fontweight="bold")
        axes[0, 1].set_xlabel("Sample Index")
        axes[0, 1].set_ylabel("Relative L2 Error")
        axes[0, 1].grid(True, alpha=0.3)

        # Cumulative error
        sorted_errors = np.sort(per_sample_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 0].plot(sorted_errors, cumulative, linewidth=3, color="green")
        axes[1, 0].set_title("Cumulative Error Distribution", fontweight="bold")
        axes[1, 0].set_xlabel("Relative L2 Error")
        axes[1, 0].set_ylabel("Cumulative Probability")
        axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics
        stats_text = f"""
        Error Statistics:
        Mean: {np.mean(per_sample_errors):.6f}
        Std:  {np.std(per_sample_errors):.6f}
        Min:  {np.min(per_sample_errors):.6f}
        Max:  {np.max(per_sample_errors):.6f}

        Percentiles:
        25th: {np.percentile(per_sample_errors, 25):.6f}
        50th: {np.percentile(per_sample_errors, 50):.6f}
        75th: {np.percentile(per_sample_errors, 75):.6f}
        95th: {np.percentile(per_sample_errors, 95):.6f}
        """

        axes[1, 1].text(
            0.1,
            0.5,
            stats_text,
            fontsize=11,
            transform=axes[1, 1].transAxes,
            verticalalignment="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7},
        )
        axes[1, 1].set_title("Summary Statistics", fontweight="bold")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "error_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_model_summary(self):
        """Plot model architecture summary."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle("U-FNO Model Architecture Summary", fontsize=16, fontweight="bold")

        # Model configuration text
        config_text = f"""
        🌪️ U-Net Fourier Neural Operator for Turbulence

        Architecture Configuration:
        • Input Channels: {self.model_config["in_channels"]} (+ 2 grid coordinates)
        • Output Channels: {self.model_config["out_channels"]}
        • Hidden Width: {self.model_config["width"]}
        • Fourier Modes: {self.model_config["modes"]}
        • U-Net Levels: {self.model_config["num_levels"]}
        • Downsample Factor: {self.model_config["downsample_factor"]}

        Multi-Scale Features:
        • Hierarchical encoder-decoder structure
        • Skip connections across scales
        • Spectral convolutions at each level
        • Adaptive spatial resolution processing

        Training Configuration:
        • Learning Rate: {self.training_config["learning_rate"]}
        • Epochs: {self.training_config["num_epochs"]}
        • Batch Size: {self.data_config["batch_size"]}
        • Physics-Informed Loss: Energy + Vorticity conservation

        Dataset Configuration:
        • Problem: 2D Scalar Burgers Equation
        • Resolution: {self.data_config["resolution"]}x{self.data_config["resolution"]}
        • Viscosity: 0.01-0.1 (turbulent regime)
        • Domain: [0,1]x[0,1]
        """

        ax.text(
            0.05,
            0.95,
            config_text,
            fontsize=12,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgreen", "alpha": 0.8},
        )
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / "model_summary.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """Main function to run the comprehensive U-FNO turbulence example."""
    parser = argparse.ArgumentParser(description="U-FNO Turbulence Example")
    parser.add_argument("--resolution", type=int, default=64, help="Grid resolution")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Configuration
    model_config = {
        "in_channels": 1,  # scalar field for 2D Burgers equation
        "out_channels": 1,  # next scalar field
        "width": 64,
        "modes": (16, 16),
        "num_levels": 3,  # U-Net levels for multi-scale processing
        "downsample_factor": 2,
    }

    training_config = {
        "learning_rate": args.learning_rate,
        "num_epochs": args.epochs,
        "grad_clip": 1.0,
    }

    data_config = {
        "resolution": args.resolution,
        "batch_size": args.batch_size,
    }

    # Initialize and run experiment
    experiment = ComprehensiveUFNOTurbulence(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        seed=args.seed,
    )

    # Setup components
    experiment.setup_dataset()
    experiment.setup_model()
    experiment.setup_optimizer()

    # Train model
    _ = experiment.train()

    # Comprehensive evaluation
    results = experiment.evaluate_comprehensive()

    # Generate visualizations
    experiment.visualize_results(results)

    print("\n🎉 U-FNO Turbulence example completed successfully!")
    print(f"📁 Results saved to: {experiment.output_dir}")


if __name__ == "__main__":
    main()
