# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Local FNO on Darcy Flow

| Property      | Value                                          |
|---------------|------------------------------------------------|
| Level         | Intermediate                                   |
| Runtime       | ~5 min (CPU) / ~1 min (GPU)                     |
| Memory        | ~2 GB                                          |
| Prerequisites | JAX, Flax NNX, Neural Operators basics         |

## Overview

Train a Local Fourier Neural Operator (LocalFNO) on the Darcy flow problem and
compare it against a standard FNO. LocalFNO combines global Fourier spectral
convolutions with local spatial convolutions, capturing both long-range
dependencies and fine-grained local features.

This example uses the standard operator-learning recipe — grid positional
embedding, Gaussian input/output normalization, and the relative-L2 loss — to
reach a low relative L2 error on Darcy flow.

This example demonstrates:

- **LocalFourierNeuralOperator** with spectral + local convolution branches
- **GridEmbedding2D** positional encoding fed as extra input channels
- **Gaussian normalization** of inputs and outputs
- **relative-L2 loss** via `LossConfig`, the standard operator-learning objective
- **Trainer.fit()** for end-to-end training with validation
- A head-to-head comparison of LocalFNO vs standard FNO
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import time
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from flax import nnx


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.core.evaluation import predict_in_batches
from opifex.core.metrics import per_sample_relative_l2
from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.config import LossConfig
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.fno.local import LocalFourierNeuralOperator


# %% [markdown]
"""
## Configuration

We follow the standard operator-learning recipe: ~1000 training samples,
Gaussian normalization, the relative-L2 loss, and enough epochs for the
spectral weights to converge.
"""

# %%
RESOLUTION = 32
N_TRAIN = 1000
N_TEST = 100
BATCH_SIZE = 32
NUM_EPOCHS = 120
LEARNING_RATE = 1e-3
SEED = 42

# Model configuration
MODES = (12, 12)
HIDDEN_CHANNELS = 32
NUM_LAYERS = 4
KERNEL_SIZE = 3

OUTPUT_DIR = Path("docs/assets/examples/local_fno_darcy")


# %% [markdown]
"""
## Model Definitions

LocalFNO operates on channels-first tensors and does not append grid
coordinates internally, so we wrap it with `GridEmbedding2D`. The embedding
appends normalized ``(x, y)`` coordinate channels to the permeability input —
the standard positional encoding that lets spectral operators resolve the
Dirichlet boundary of the Darcy problem. We build a standard FNO with the same
embedding for a fair head-to-head comparison.
"""


# %%
class LocalFNOWithGrid(nnx.Module):
    """LocalFNO with a 2D grid positional embedding on a channels-first input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        modes: tuple[int, int],
        num_layers: int,
        kernel_size: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the grid embedding and the underlying LocalFNO.

        Args:
            in_channels: Number of physical input channels (before the grid).
            out_channels: Number of output channels.
            hidden_channels: LocalFNO hidden width.
            modes: Fourier modes for the spectral branch, one per spatial axis.
            num_layers: Number of local Fourier layers.
            kernel_size: Kernel size for the local convolution branch.
            rngs: Random number generators.
        """
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=in_channels,
            grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        )
        self.local_fno = LocalFourierNeuralOperator(
            in_channels=self.grid_embedding.out_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            kernel_size=kernel_size,
            use_residual_connections=True,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Append grid coordinates, then apply the LocalFNO.

        Args:
            x: Input of shape (batch, in_channels, height, width).

        Returns:
            Output of shape (batch, out_channels, height, width).
        """
        # (batch, channels, H, W) -> (batch, H, W, channels) for the embedding
        x_hwc = jnp.moveaxis(x, 1, -1)
        x_embedded = self.grid_embedding(x_hwc)
        # (batch, H, W, channels) -> (batch, channels, H, W) for the operator
        x_chw = jnp.moveaxis(x_embedded, -1, 1)
        result = self.local_fno(x_chw)
        if isinstance(result, tuple):
            return result[0]
        return result


class FNOWithGrid(nnx.Module):
    """Standard FNO with a 2D grid positional embedding (channels-first input)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        modes: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the grid embedding and the underlying FNO.

        Args:
            in_channels: Number of physical input channels (before the grid).
            out_channels: Number of output channels.
            hidden_channels: FNO hidden width.
            modes: Number of Fourier modes per spatial axis.
            num_layers: Number of Fourier layers.
            rngs: Random number generators.
        """
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=in_channels,
            grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        )
        self.fno = FourierNeuralOperator(
            in_channels=self.grid_embedding.out_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Append grid coordinates, then apply the FNO.

        Args:
            x: Input of shape (batch, in_channels, height, width).

        Returns:
            Output of shape (batch, out_channels, height, width).
        """
        x_hwc = jnp.moveaxis(x, 1, -1)
        x_embedded = self.grid_embedding(x_hwc)
        x_chw = jnp.moveaxis(x_embedded, -1, 1)
        return self.fno(x_chw)


def count_params(model: nnx.Module) -> int:
    """Count the trainable parameters of an NNX module."""
    return sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


# %% [markdown]
"""
## Run the Example

`main()` loads and normalizes the Darcy data, trains both LocalFNO and a
standard FNO with the relative-L2 loss, evaluates and compares them, saves the
figures, and returns a small dict of finite metrics.
"""


# %%
def main() -> dict[str, float | int]:
    """Train and compare LocalFNO vs a standard FNO on the Darcy flow problem."""
    print("=" * 70)
    print("Opifex Example: Local FNO on Darcy Flow")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
    print(f"FNO config: modes={MODES}, width={HIDDEN_CHANNELS}, layers={NUM_LAYERS}")
    print(f"Local kernel size: {KERNEL_SIZE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data loading via datarax ---
    print()
    print("Generating Darcy flow data and serving via datarax...")
    n_samples = N_TRAIN + N_TEST
    loaders = create_darcy_loader(
        n_samples=n_samples,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        val_fraction=N_TEST / n_samples,
        seed=SEED,
    )

    # Collect the datarax pipelines into arrays. Batches are channels-first
    # {"input": (b, 1, H, W), "output": (b, 1, H, W)} for Darcy.
    def _collect(pipeline) -> tuple[np.ndarray, np.ndarray]:
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    X_train, Y_train = _collect(loaders.train)
    X_test, Y_test = _collect(loaders.val)

    print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test data:     X={X_test.shape}, Y={Y_test.shape}")

    # --- Normalization ---
    x_mean, x_std = X_train.mean(), X_train.std()
    y_mean, y_std = Y_train.mean(), Y_train.std()

    X_train_n = (X_train - x_mean) / x_std
    Y_train_n = (Y_train - y_mean) / y_std
    X_test_n = (X_test - x_mean) / x_std
    Y_test_n = (Y_test - y_mean) / y_std

    print(f"Input mean/std:  {x_mean:.4f} / {x_std:.4f}")
    print(f"Output mean/std: {y_mean:.6f} / {y_std:.6f}")

    # --- Model creation ---
    print()
    print("Creating LocalFNO model with grid embedding...")
    local_fno = LocalFNOWithGrid(
        in_channels=1,
        out_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        modes=MODES,
        num_layers=NUM_LAYERS,
        kernel_size=KERNEL_SIZE,
        rngs=nnx.Rngs(SEED),
    )
    local_fno_params = count_params(local_fno)
    print(f"LocalFNO parameters: {local_fno_params:,}")

    print()
    print("Creating standard FNO for comparison...")
    standard_fno = FNOWithGrid(
        in_channels=1,
        out_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        modes=MODES[0],
        num_layers=NUM_LAYERS,
        rngs=nnx.Rngs(SEED),
    )
    fno_params = count_params(standard_fno)
    print(f"Standard FNO parameters: {fno_params:,}")
    print(f"LocalFNO overhead: {(local_fno_params / fno_params - 1) * 100:.1f}%")

    # --- Training ---
    def train_operator(model: nnx.Module, model_name: str) -> tuple[nnx.Module, list[float]]:
        """Train an operator on the normalized Darcy data with the relative-L2 loss.

        Args:
            model: The NNX operator to train.
            model_name: Human-readable name used in log messages.

        Returns:
            Tuple of (trained model, per-epoch training-loss history).
        """
        loss_history: list[float] = []

        def record_loss(_epoch: int, logs: dict) -> None:
            loss_history.append(float(logs["train_loss"]))

        config = TrainingConfig(
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            validation_frequency=10,
            verbose=True,
            loss_config=LossConfig(loss_type="relative_l2"),
            progress_callback=record_loss,
        )
        trainer = Trainer(model=model, config=config, rngs=nnx.Rngs(SEED))

        print(f"Training {model_name} (Adam lr={LEARNING_RATE}, relative-L2 loss)...")
        start = time.time()
        trained_model, metrics = trainer.fit(
            train_data=(jnp.array(X_train_n), jnp.array(Y_train_n)),
            val_data=(jnp.array(X_test_n), jnp.array(Y_test_n)),
        )
        elapsed = time.time() - start
        print(f"{model_name} training completed in {elapsed:.1f}s")
        print(f"  Final train loss: {metrics.get('final_train_loss', 'N/A')}")
        print(f"  Final val loss:   {metrics.get('final_val_loss', 'N/A')}")
        return trained_model, loss_history

    print()
    local_fno, local_history = train_operator(local_fno, "LocalFNO")
    print()
    standard_fno, fno_history = train_operator(standard_fno, "Standard FNO")

    # --- Evaluation ---
    X_test_jnp = jnp.array(X_test_n)
    Y_test_jnp = jnp.array(Y_test)

    def evaluate_model(model: nnx.Module, model_name: str) -> tuple[jax.Array, float, float]:
        """Evaluate an operator on the physical-space test set."""
        predictions = predict_in_batches(model, X_test_jnp) * y_std + y_mean
        mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))

        per_sample_rel_l2 = per_sample_relative_l2(predictions, Y_test_jnp)
        rel_l2_mean = float(jnp.mean(per_sample_rel_l2))
        rel_l2_min = float(jnp.min(per_sample_rel_l2))
        rel_l2_max = float(jnp.max(per_sample_rel_l2))

        print(f"{model_name} Results:")
        print(f"  Test MSE:         {mse:.6e}")
        print(f"  Relative L2:      {rel_l2_mean:.6f} (min={rel_l2_min:.6f}, max={rel_l2_max:.6f})")
        return predictions, mse, rel_l2_mean

    print()
    print("Running evaluation...")
    local_pred, local_mse, local_rel_l2 = evaluate_model(local_fno, "LocalFNO")
    print()
    fno_pred, fno_mse, fno_rel_l2 = evaluate_model(standard_fno, "Standard FNO")

    print()
    print("Comparison:")
    mse_improvement = (fno_mse - local_mse) / fno_mse * 100
    rel_l2_improvement = (fno_rel_l2 - local_rel_l2) / fno_rel_l2 * 100
    print(f"  MSE improvement (LocalFNO vs FNO): {mse_improvement:+.1f}%")
    print(f"  Rel L2 improvement: {rel_l2_improvement:+.1f}%")

    # --- Visualization: predictions ---
    print()
    print("Generating visualizations...")
    _fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    sample_idx = 0
    input_field = np.array(X_test[sample_idx, 0])
    ground_truth = np.array(Y_test[sample_idx, 0])

    # Row 1: LocalFNO
    axes[0, 0].imshow(input_field, cmap="viridis")
    axes[0, 0].set_title("Input (Permeability)")
    axes[0, 0].set_ylabel("LocalFNO", fontsize=12)
    axes[0, 0].axis("off")
    axes[0, 1].imshow(ground_truth, cmap="RdBu_r")
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(np.array(local_pred[sample_idx, 0]), cmap="RdBu_r")
    axes[0, 2].set_title("LocalFNO Prediction")
    axes[0, 2].axis("off")
    local_error = np.abs(np.array(local_pred[sample_idx, 0]) - ground_truth)
    im1 = axes[0, 3].imshow(local_error, cmap="hot")
    axes[0, 3].set_title(f"LocalFNO Error (max={local_error.max():.4f})")
    axes[0, 3].axis("off")
    plt.colorbar(im1, ax=axes[0, 3], fraction=0.046)

    # Row 2: Standard FNO
    axes[1, 0].imshow(input_field, cmap="viridis")
    axes[1, 0].set_title("Input (Permeability)")
    axes[1, 0].set_ylabel("Standard FNO", fontsize=12)
    axes[1, 0].axis("off")
    axes[1, 1].imshow(ground_truth, cmap="RdBu_r")
    axes[1, 1].set_title("Ground Truth")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(np.array(fno_pred[sample_idx, 0]), cmap="RdBu_r")
    axes[1, 2].set_title("Standard FNO Prediction")
    axes[1, 2].axis("off")
    fno_error = np.abs(np.array(fno_pred[sample_idx, 0]) - ground_truth)
    im2 = axes[1, 3].imshow(fno_error, cmap="hot")
    axes[1, 3].set_title(f"FNO Error (max={fno_error.max():.4f})")
    axes[1, 3].axis("off")
    plt.colorbar(im2, ax=axes[1, 3], fraction=0.046)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions saved to {OUTPUT_DIR / 'predictions.png'}")

    # --- Visualization: training comparison ---
    _fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if local_history:
        axes[0].semilogy(local_history, label="LocalFNO", linewidth=2)
    if fno_history:
        axes[0].semilogy(fno_history, label="Standard FNO", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Relative-L2 Loss")
    axes[0].set_title("Training Loss Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    models = ["LocalFNO", "Standard FNO"]
    rel_l2_values = [local_rel_l2, fno_rel_l2]
    colors = ["steelblue", "coral"]
    bars = axes[1].bar(models, rel_l2_values, color=colors)
    axes[1].set_ylabel("Test Relative L2")
    axes[1].set_title("Test Error Comparison")
    for bar, value in zip(bars, rel_l2_values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison saved to {OUTPUT_DIR / 'comparison.png'}")

    print()
    print("=" * 70)
    print("Local FNO Darcy Flow example completed")
    print("=" * 70)
    print("Results Summary:")
    print(
        f"  LocalFNO:     MSE={local_mse:.6e}, Rel L2={local_rel_l2:.4f}, Params={local_fno_params:,}"
    )
    print(f"  Standard FNO: MSE={fno_mse:.6e}, Rel L2={fno_rel_l2:.4f}, Params={fno_params:,}")
    print(f"  Improvement:  MSE {mse_improvement:+.1f}%, Rel L2 {rel_l2_improvement:+.1f}%")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return {
        "local_fno_mse": local_mse,
        "local_fno_rel_l2": local_rel_l2,
        "standard_fno_mse": fno_mse,
        "standard_fno_rel_l2": fno_rel_l2,
        "local_fno_parameters": int(local_fno_params),
        "standard_fno_parameters": int(fno_params),
    }


# %% [markdown]
"""
## Results Summary

After running this example you should observe:

- A decreasing relative-L2 training loss over epochs on Darcy flow data
- A low test relative L2 error for both operators on the corrected Darcy data
- Accurate predictions mapping permeability to pressure fields
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
