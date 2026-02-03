#!/usr/bin/env python3
"""Comprehensive Spherical FNO for Climate Modeling - Neural Operator Example.

This example demonstrates comprehensive Spherical FNO functionality for climate
modeling using the Opifex framework with JAX/Flax NNX. Features include spherical
harmonic analysis, conservation laws, and comprehensive climate data visualization.
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

from opifex.data.loaders.factory import create_shallow_water_loader

# Opifex imports
from opifex.neural.operators.fno.spherical import (
    create_climate_sfno,
)


class ComprehensiveSFNOClimate:
    """
    Comprehensive Spherical FNO implementation for climate modeling problems.

    This class provides a complete implementation including:
    - Spherical FNO architecture with spherical harmonic transforms
    - Training and evaluation pipelines optimized for spherical domains
    - Climate data simulation and preprocessing
    - Physics-informed loss functions for energy conservation
    - Comprehensive visualization with spherical projections
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
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None

        # Training history with spherical domain metrics
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_relative_l2": [],
            "val_relative_l2": [],
            "train_energy_conservation": [],
            "val_energy_conservation": [],
            "train_mass_conservation": [],
            "val_mass_conservation": [],
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging and output directories."""
        self.output_dir = Path("examples_output/sfno_climate_comprehensive")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(
            "🌍 Opifex Neural Operator Example: Comprehensive Spherical FNO for Climate"
        )
        print("=" * 80)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎲 Random seed: {self.seed}")
        print(f"💻 JAX backend: {jax.default_backend()}")
        print(f"🔧 JAX devices: {jax.devices()}")

    def setup_dataset(self):
        """Setup climate-like dataset using shallow water equations on sphere."""
        print("\n📊 Setting up Climate dataset with Grain (Shallow Water on Sphere)...")

        # Use shallow water equations as a proxy for climate dynamics
        n_train = 200
        n_test = 40
        batch_size = self.training_config.get("batch_size", 16)

        self.train_loader = create_shallow_water_loader(
            n_samples=n_train,
            batch_size=batch_size,
            resolution=self.data_config["resolution"],
            shuffle=True,
            seed=self.seed + 3000,
            worker_count=0,
        )

        self.test_loader = create_shallow_water_loader(
            n_samples=n_test,
            batch_size=batch_size,
            resolution=self.data_config["resolution"],
            shuffle=False,
            seed=self.seed + 4000,
            worker_count=0,
        )

        # Store batch size for later use
        self.batch_size = self.data_config["batch_size"]

        print("   ✅ Dataset type: Shallow Water on Sphere (Climate Proxy)")
        print(
            f"   ✅ Resolution: {self.data_config['resolution']}x{self.data_config['resolution']}"
        )
        print(f"   ✅ Batch size: {self.data_config['batch_size']}")
        print(f"   ✅ Training samples: {n_train}")
        print(f"   ✅ Test samples: {n_test}")

        return self.dataset

    def setup_model(self):
        """Setup Spherical FNO model."""
        print("\n🏗️ Setting up Spherical FNO model...")

        # Use specialized climate SFNO configuration
        self.model = create_climate_sfno(
            in_channels=self.model_config["in_channels"],
            out_channels=self.model_config["out_channels"],
            lmax=self.model_config["lmax"],
            rngs=nnx.Rngs(self.seed),
        )

        print("   ✅ Model type: Spherical FNO")
        print(f"   ✅ Input channels: {self.model_config['in_channels']}")
        print(f"   ✅ Output channels: {self.model_config['out_channels']}")
        print(
            f"   ✅ Max spherical harmonic degree (lmax): {self.model_config['lmax']}"
        )
        print(f"   ✅ Hidden width: {self.model_config['width']}")
        print(f"   ✅ Number of layers: {self.model_config['n_layers']}")

        return self.model

    def setup_optimizer(self):
        """Setup optimizer and training state."""
        print("\n⚙️ Setting up optimizer...")

        # Create optimizer with learning rate schedule
        schedule = optax.cosine_decay_schedule(
            init_value=self.training_config["learning_rate"],
            decay_steps=self.training_config["num_epochs"] * 100,
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

    def compute_losses(
        self, predictions: jax.Array, targets: jax.Array
    ) -> dict[str, float]:
        """Compute comprehensive loss metrics for spherical domain climate data."""
        # Basic losses
        mse_loss = jnp.mean((predictions - targets) ** 2)

        # Relative L2 loss
        relative_l2 = jnp.mean(
            jnp.linalg.norm(predictions - targets, axis=(1, 2, 3))
            / jnp.linalg.norm(targets, axis=(1, 2, 3))
        )

        # Physics-informed losses for climate/spherical domains
        # Energy conservation (kinetic + potential energy)
        pred_energy = jnp.mean(predictions**2, axis=(2, 3))  # (batch, channels)
        target_energy = jnp.mean(targets**2, axis=(2, 3))
        energy_conservation = jnp.mean(jnp.abs(pred_energy - target_energy))

        # Mass conservation (total mass should be preserved)
        pred_mass = jnp.mean(predictions, axis=(2, 3))  # (batch, channels)
        target_mass = jnp.mean(targets, axis=(2, 3))
        mass_conservation = jnp.mean(jnp.abs(pred_mass - target_mass))

        return {
            "mse": float(mse_loss),
            "relative_l2": float(relative_l2),
            "energy_conservation": float(energy_conservation),
            "mass_conservation": float(mass_conservation),
        }

    def train_step(self, batch_x: jax.Array, batch_y: jax.Array) -> dict[str, float]:
        """Training step with physics-informed loss."""

        def loss_fn(params):
            # Temporarily update model with params
            nnx.update(self.model, params)

            # Forward pass
            predictions = self.model(batch_x)

            # Compute losses
            losses = self.compute_losses(predictions, batch_y)

            # Combined loss with physics terms
            total_loss = (
                losses["mse"]
                + 0.1 * losses["energy_conservation"]
                + 0.05 * losses["mass_conservation"]
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
        predictions = self.model(batch_x)
        losses = self.compute_losses(predictions, batch_y)
        total_loss = (
            losses["mse"]
            + 0.1 * losses["energy_conservation"]
            + 0.05 * losses["mass_conservation"]
        )
        return {"loss": float(total_loss), **losses}

    def get_train_batches(self):
        """Get training data batches."""
        if self.dataset is None:
            raise ValueError("Dataset must be setup before getting training batches")
        x_data, y_data = self.dataset.get_data("train")

        # Ensure 4D tensor: (batch, channels, height, width)
        if x_data.ndim == 3:
            x_data = x_data[:, None, :, :]
        if y_data.ndim == 3:
            y_data = y_data[:, None, :, :]

        n_samples = x_data.shape[0]

        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            yield x_data[i:end_idx], y_data[i:end_idx]

    def get_test_batches(self):
        """Get test data batches."""
        if self.dataset is None:
            raise ValueError("Dataset must be setup before getting test batches")
        x_data, y_data = self.dataset.get_data("test")

        # Ensure 4D tensor: (batch, channels, height, width)
        if x_data.ndim == 3:
            x_data = x_data[:, None, :, :]
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
            "mass_conservation": [],
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
            "mass_conservation": [],
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
            self.history["train_mass_conservation"].append(
                train_metrics["mass_conservation"]
            )
            self.history["val_mass_conservation"].append(
                val_metrics["mass_conservation"]
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
        """Comprehensive evaluation with spherical domain metrics."""
        print("\n🔍 Running comprehensive evaluation...")

        # Get sample test data for detailed analysis
        if self.dataset is None:
            raise ValueError("Dataset must be setup before evaluation")
        X_test, Y_test = self.dataset.get_data("test")

        # Ensure 4D tensor
        if X_test.ndim == 3:
            X_test = X_test[:, None, :, :]
        if Y_test.ndim == 3:
            Y_test = Y_test[:, None, :, :]

        # Get predictions
        predictions = self.model(X_test)

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
        print(f"   ✅ Mass Conservation: {overall_metrics['mass_conservation']:.6f}")

        return results

    def visualize_results(self, results: dict[str, Any]):
        """Comprehensive visualization of Spherical FNO results."""
        print("\n📊 Generating comprehensive visualizations...")

        # Create subplots
        self._plot_training_curves()
        self._plot_spherical_predictions(results)
        self._plot_spectral_analysis(results)
        self._plot_error_analysis(results)

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

    def _plot_conservation_metrics(self, axes, epochs):
        """Plot energy and mass conservation subplots."""
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

        # Mass conservation
        axes[1, 1].plot(
            epochs,
            self.history["train_mass_conservation"],
            "b-",
            label="Train",
            linewidth=2,
        )
        axes[1, 1].plot(
            epochs,
            self.history["val_mass_conservation"],
            "r--",
            label="Validation",
            linewidth=2,
        )
        axes[1, 1].set_title("Mass Conservation Error", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Mass Error")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale("log")

    def _plot_final_metrics_summary(self, axes):
        """Plot final metrics summary subplot."""
        final_val_loss = self.history["val_loss"][-1]
        final_val_rel_l2 = self.history["val_relative_l2"][-1]
        final_energy = self.history["val_energy_conservation"][-1]
        final_mass = self.history["val_mass_conservation"][-1]

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
            f"Mass Conservation: {final_mass:.6f}",
            fontsize=12,
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].set_title("Final Metrics", fontweight="bold")
        axes[1, 2].axis("off")

    def _plot_training_curves(self):
        """Plot training curves with physics metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Spherical FNO Training Progress - Climate Modeling",
            fontsize=16,
            fontweight="bold",
        )

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Plot all sections using helper functions
        self._plot_loss_curves(axes, epochs)
        self._plot_mse_curves(axes, epochs)
        self._plot_relative_l2_curves(axes, epochs)
        self._plot_conservation_metrics(axes, epochs)
        self._plot_final_metrics_summary(axes)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_spherical_predictions(self, results: dict[str, Any]):
        """Plot predictions with spherical projections."""
        predictions = results["predictions"]
        targets = results["targets"]
        inputs = results["inputs"]

        # Select first sample for visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(
            "Spherical FNO Climate Predictions", fontsize=16, fontweight="bold"
        )

        # Input
        im0 = axes[0].imshow(inputs[0, 0], cmap="RdBu_r", aspect="equal")
        axes[0].set_title("Input", fontweight="bold")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        # Ground truth
        im1 = axes[1].imshow(targets[0, 0], cmap="RdBu_r", aspect="equal")
        axes[1].set_title("Ground Truth", fontweight="bold")
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        # Prediction
        im2 = axes[2].imshow(predictions[0, 0], cmap="RdBu_r", aspect="equal")
        axes[2].set_title("Spherical FNO Prediction", fontweight="bold")
        axes[2].set_xlabel("Longitude")
        axes[2].set_ylabel("Latitude")
        plt.colorbar(im2, ax=axes[2], shrink=0.8)

        # Error
        error = np.abs(predictions[0, 0] - targets[0, 0])
        im3 = axes[3].imshow(error, cmap="plasma", aspect="equal")
        axes[3].set_title("Absolute Error", fontweight="bold")
        axes[3].set_xlabel("Longitude")
        axes[3].set_ylabel("Latitude")
        plt.colorbar(im3, ax=axes[3], shrink=0.8)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "spherical_predictions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_spectral_analysis(self, results: dict[str, Any]):
        """Plot spherical harmonic spectral analysis."""
        predictions = results["predictions"]
        targets = results["targets"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Spherical Harmonic Spectral Analysis", fontsize=16, fontweight="bold"
        )

        # Compute power spectra (approximation using 2D FFT)
        pred_fft = np.abs(np.fft.fft2(predictions[0, 0]))
        target_fft = np.abs(np.fft.fft2(targets[0, 0]))

        # Radial average for spherical harmonic degrees
        def radial_average(data):
            y, x = np.ogrid[: data.shape[0], : data.shape[1]]
            center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
            r = np.hypot(x - center[0], y - center[1])
            r = r.astype(int)

            tbin = np.bincount(r.ravel(), data.ravel())
            nr = np.bincount(r.ravel())
            return tbin / nr

        pred_radial = radial_average(pred_fft**2)
        target_radial = radial_average(target_fft**2)

        # Spherical harmonic degree approximation
        degrees = np.arange(len(pred_radial))

        axes[0, 0].loglog(
            degrees[1:20], pred_radial[1:20], "b-", label="SFNO Prediction", linewidth=2
        )
        axes[0, 0].loglog(
            degrees[1:20], target_radial[1:20], "r--", label="Ground Truth", linewidth=2
        )
        axes[0, 0].set_title(
            "Power Spectrum vs Spherical Harmonic Degree", fontweight="bold"
        )
        axes[0, 0].set_xlabel("Degree l")
        axes[0, 0].set_ylabel("Power")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Error spectrum
        error_fft = np.abs(np.fft.fft2(predictions[0, 0] - targets[0, 0]))
        error_radial = radial_average(error_fft**2)

        axes[0, 1].loglog(degrees[1:20], error_radial[1:20], "g-", linewidth=2)
        axes[0, 1].set_title("Error Power Spectrum", fontweight="bold")
        axes[0, 1].set_xlabel("Degree l")
        axes[0, 1].set_ylabel("Error Power")
        axes[0, 1].grid(True, alpha=0.3)

        # Energy conservation by degree
        energy_ratio = pred_radial / (target_radial + 1e-10)
        axes[1, 0].semilogx(degrees[1:20], energy_ratio[1:20], "purple", linewidth=2)
        axes[1, 0].axhline(y=1.0, color="k", linestyle="--", alpha=0.5)
        axes[1, 0].set_title(
            "Energy Ratio by Spherical Harmonic Degree", fontweight="bold"
        )
        axes[1, 0].set_xlabel("Degree l")
        axes[1, 0].set_ylabel("Pred Energy / True Energy")
        axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics
        total_energy_pred = np.sum(pred_radial[1:])
        total_energy_target = np.sum(target_radial[1:])
        energy_conservation_ratio = total_energy_pred / total_energy_target

        stats_text = f"""
        Spherical Harmonic Analysis:

        Total Energy Conservation: {energy_conservation_ratio:.4f}
        Low-degree Error (l<5): {np.mean(error_radial[1:5]):.2e}
        Mid-degree Error (l<10): {np.mean(error_radial[5:10]):.2e}
        High-degree Error (l<15): {np.mean(error_radial[10:15]):.2e}

        Peak Energy Degree: {np.argmax(target_radial[1:20]) + 1}
        Energy at Peak: {np.max(target_radial[1:20]):.2e}
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
        axes[1, 1].set_title("Spectral Analysis Summary", fontweight="bold")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "spectral_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_error_analysis(self, results: dict[str, Any]):
        """Plot comprehensive error analysis."""
        per_sample_errors = results["per_sample_errors"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Spherical FNO Error Analysis", fontsize=16, fontweight="bold")

        # Error distribution
        axes[0, 0].hist(
            per_sample_errors, bins=15, alpha=0.7, color="lightcoral", edgecolor="black"
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
            per_sample_errors, "o-", markersize=6, linewidth=2, color="darkblue"
        )
        axes[0, 1].set_title("Error vs Sample Index", fontweight="bold")
        axes[0, 1].set_xlabel("Sample Index")
        axes[0, 1].set_ylabel("Relative L2 Error")
        axes[0, 1].grid(True, alpha=0.3)

        # Cumulative error
        sorted_errors = np.sort(per_sample_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 0].plot(sorted_errors, cumulative, linewidth=3, color="forestgreen")
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
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgreen", "alpha": 0.7},
        )
        axes[1, 1].set_title("Summary Statistics", fontweight="bold")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "error_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    """Main function to run the comprehensive Spherical FNO climate example."""
    parser = argparse.ArgumentParser(description="Spherical FNO Climate Example")
    parser.add_argument("--resolution", type=int, default=64, help="Grid resolution")
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Configuration
    model_config = {
        "in_channels": 3,  # height, u_velocity, v_velocity
        "out_channels": 3,  # next state
        "width": 64,
        "lmax": 16,  # Maximum spherical harmonic degree
        "n_layers": 4,
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
    experiment = ComprehensiveSFNOClimate(
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

    print("\n🎉 Spherical FNO Climate example completed successfully!")
    print(f"📁 Results saved to: {experiment.output_dir}")


if __name__ == "__main__":
    main()
