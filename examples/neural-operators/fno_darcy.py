# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.6
# ---

# %% [markdown]
"""
# FNO on Darcy Flow

| Property      | Value                                    |
|---------------|------------------------------------------|
| Level         | Intermediate                             |
| Runtime       | ~3 min (CPU) / ~1 min (GPU)              |
| Memory        | ~2 GB                                    |
| Prerequisites | JAX, Flax NNX, Neural Operators basics   |

## Overview

Train a Fourier Neural Operator (FNO) on the Darcy flow equation, a 2D elliptic
PDE that maps a permeability coefficient field to the pressure solution. This is
the standard-FNO showcase on Opifex's own Darcy data: it uses
`create_darcy_loader` (smooth Darcy, solved with the accurate direct solver) and
reaches a low relative L2 error with the standard operator-learning recipe.

This example demonstrates:

- **GridEmbedding2D** for spatial positional encoding
- **FourierNeuralOperator** for spectral operator learning
- **Gaussian normalization** of inputs and outputs (fit on train, un-normalized
  predictions for physical-space error)
- **relative-L2 loss** via `LossConfig`, the standard operator-learning objective
- **AdamW + exponential LR decay + weight decay** to converge without overfitting
- **Trainer.fit()** for end-to-end training with validation

It is the FNO counterpart to [UNO on Darcy Flow](uno-darcy.md) and
[Your First Neural Operator](../getting-started/first-neural-operator.md), which
use the same synthetic Darcy data and recipe.

## Learning Goals

1. Compose `GridEmbedding2D` with `FourierNeuralOperator`
2. Load Darcy flow data with `create_darcy_loader`
3. Apply Gaussian normalization and the relative-L2 loss
4. Use AdamW + an exponential learning-rate schedule + weight decay
5. Evaluate with L2 relative error and full visualization
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
from opifex.core.metrics import per_sample_relative_l2, relative_l2_error
from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.config import LossConfig, OptimizationConfig
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.base import FourierNeuralOperator


# %% [markdown]
"""
## Configuration

We follow the standard operator-learning recipe: ~1000 training samples,
Gaussian normalization, the relative-L2 loss, and `AdamW` with weight decay plus
an exponential learning-rate decay over enough epochs for the spectral weights to
converge. The FNO also uses `domain_padding`, which pads the spatial dimensions
before the spectral layers to reduce the Gibbs phenomenon on this non-periodic
boundary-value problem.
"""

# %% [markdown]
"""
## Model Creation

We compose `GridEmbedding2D` with `FourierNeuralOperator` to inject spatial
coordinates as additional input channels. This positional encoding helps the FNO
learn spatially varying operators on this boundary-value problem. The FNO also
uses `domain_padding`, padding the spatial dimensions before the spectral layers
to reduce the Gibbs phenomenon for the non-periodic Darcy problem.
"""


# %%
class FNOWithEmbedding(nnx.Module):
    """FNO model with built-in grid embedding for positional encoding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        hidden_channels: int,
        num_layers: int,
        grid_boundaries: list[list[float]],
        *,
        domain_padding: float,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the grid embedding and the underlying FNO.

        Args:
            in_channels: Number of physical input channels (before the grid).
            out_channels: Number of output channels.
            modes: Number of Fourier modes per spatial dimension.
            hidden_channels: Number of FNO hidden channels.
            num_layers: Number of spectral layers.
            grid_boundaries: Per-axis ``[min, max]`` grid extents.
            domain_padding: Fraction of each spatial axis to pad (resolution-invariant)
                to reduce the Gibbs phenomenon for the non-periodic Darcy problem.
            rngs: Random number generators.
        """
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=in_channels,
            grid_boundaries=grid_boundaries,
        )
        self.fno = FourierNeuralOperator(
            in_channels=self.grid_embedding.out_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            domain_padding=domain_padding,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: grid embedding -> FNO.

        Args:
            x: Input of shape ``(batch, channels, H, W)``.

        Returns:
            Output of shape ``(batch, out_channels, H, W)``.
        """
        # (batch, channels, H, W) -> (batch, H, W, channels) for embedding
        x_hwc = jnp.moveaxis(x, 1, -1)
        x_embedded = self.grid_embedding(x_hwc)
        # (batch, H, W, channels) -> (batch, channels, H, W) for FNO
        x_chw = jnp.moveaxis(x_embedded, -1, 1)
        return self.fno(x_chw)


# %% [markdown]
"""
## Training with Opifex Trainer

We train with `AdamW`, weight decay, and an exponential learning-rate schedule
that halves the rate every 60 epochs. The data loss is the **relative-L2 loss**
(`loss_type="relative_l2"`), the standard operator-learning objective.
`Trainer.fit()` handles batched training with JIT compilation, validation, and
progress logging.
"""

# %% [markdown]
"""
## Evaluation

Predictions are un-normalized back to physical pressure before measuring the
relative L2 error. We run the test and training sets through the model in batches
(to bound memory) and compare the two to confirm the model is not overfitting.
"""


# %% [markdown]
"""
## Run the example

All run logic — configuration, data loading, normalization, model creation,
training, evaluation, and visualization — lives in `main()`. It returns a small
dict of finite scalar metrics and saves the prediction/error plots to
`docs/assets/examples/fno_darcy/`.
"""


# %%
def main() -> dict[str, float | int]:
    """Train an FNO on Darcy flow and report relative-L2 error metrics.

    Returns:
        Finite scalar metrics: parameter count, final train/val loss, test MSE,
        and mean test relative L2 error.
    """
    # --- Configuration ---
    resolution = 32  # synthetic Darcy resolution
    n_train = 1000
    n_test = 100
    batch_size = 32
    num_epochs = 200
    learning_rate = 5e-3  # AdamW initial LR
    weight_decay = 1e-4  # regularization to combat overfitting
    modes = 12  # retained Fourier modes per axis
    hidden_width = 32
    num_layers = 4
    domain_padding = 0.25  # fraction of each spatial dim (resolution-invariant Gibbs padding)
    seed = 42

    # Exponential LR schedule: halve the rate every 60 epochs.
    steps_per_epoch = n_train // batch_size
    lr_decay_epochs = 60
    lr_transition_steps = lr_decay_epochs * steps_per_epoch
    lr_decay_rate = 0.5

    output_dir = Path("docs/assets/examples/fno_darcy")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Opifex Example: FNO on Darcy Flow")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Training samples: {n_train}, Test samples: {n_test}")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"FNO config: modes={modes}, width={hidden_width}, layers={num_layers}")
    print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"LR schedule: exponential, x{lr_decay_rate} every {lr_decay_epochs} epochs")

    # --- Data loading via datarax ---
    print()
    print("Generating Darcy flow data (jit+vmap) and serving via datarax...")
    n_samples = n_train + n_test
    loaders = create_darcy_loader(
        n_samples=n_samples,
        batch_size=batch_size,
        resolution=resolution,
        val_fraction=n_test / n_samples,
        seed=seed,
    )

    # Collect the datarax pipelines into arrays for Trainer.fit(). Batches are
    # channels-first {"input": (b, 1, H, W), "output": (b, 1, H, W)}.
    def _collect(pipeline) -> tuple[np.ndarray, np.ndarray]:
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    x_train, y_train = _collect(loaders.train)
    x_test, y_test = _collect(loaders.val)

    print(f"Training data: X={x_train.shape}, Y={y_train.shape}")
    print(f"Test data:     X={x_test.shape}, Y={y_test.shape}")

    # --- Normalization (fit on train, applied to all splits) ---
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    x_train_n = (x_train - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    x_test_n = (x_test - x_mean) / x_std
    y_test_n = (y_test - y_mean) / y_std

    print(f"Input mean/std:  {x_mean:.4f} / {x_std:.4f}")
    print(f"Output mean/std: {y_mean:.6f} / {y_std:.6f}")

    # --- Model creation ---
    print()
    print("Creating FNO model with grid embedding...")
    model = FNOWithEmbedding(
        in_channels=1,
        out_channels=1,
        modes=modes,
        hidden_channels=hidden_width,
        num_layers=num_layers,
        grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        domain_padding=domain_padding,
        rngs=nnx.Rngs(seed),
    )

    params = nnx.state(model, nnx.Param)
    param_count = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print("Model: FNO + GridEmbedding2D")
    print("  Input channels: 1 (+ 2 grid coords = 3 after embedding)")
    print(f"  Fourier modes: {modes}, Hidden width: {hidden_width}, Layers: {num_layers}")
    print(f"  Total parameters: {param_count:,}")

    # --- Training ---
    print()
    print("Setting up Trainer...")
    config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_frequency=20,
        verbose=True,
        loss_config=LossConfig(loss_type="relative_l2"),
        optimization_config=OptimizationConfig(
            optimizer="adamw",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            schedule_type="exponential",
            transition_steps=lr_transition_steps,
            decay_rate=lr_decay_rate,
        ),
    )
    trainer = Trainer(
        model=model,
        config=config,
        rngs=nnx.Rngs(seed),
    )

    print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print("Loss: relative L2 (the standard operator-learning objective)")
    print()
    print("Starting training...")
    start_time = time.time()

    trained_model, metrics = trainer.fit(
        train_data=(jnp.array(x_train_n), jnp.array(y_train_n)),
        val_data=(jnp.array(x_test_n), jnp.array(y_test_n)),
    )

    training_time = time.time() - start_time
    final_train_loss = float(metrics["final_train_loss"])
    final_val_loss = float(metrics["final_val_loss"])
    print(f"Training completed in {training_time:.1f}s")
    print(f"Final train loss: {final_train_loss}")
    print(f"Final val loss:   {final_val_loss}")

    # --- Evaluation (un-normalized to physical pressure units) ---
    print()
    print("Running evaluation...")
    x_test_jnp = jnp.array(x_test_n)
    y_test_jnp = jnp.array(y_test)
    x_train_jnp = jnp.array(x_train_n)
    y_train_jnp = jnp.array(y_train)

    predictions = predict_in_batches(trained_model, x_test_jnp) * y_std + y_mean
    train_predictions = predict_in_batches(trained_model, x_train_jnp) * y_std + y_mean

    test_mse = float(jnp.mean((predictions - y_test_jnp) ** 2))

    per_sample_rel_l2 = per_sample_relative_l2(predictions, y_test_jnp)
    mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))
    train_rel_l2 = float(relative_l2_error(train_predictions, y_train_jnp))

    print(f"Train Relative L2: {train_rel_l2:.6f}")
    print(f"Test  Relative L2: {mean_rel_l2:.6f}")
    print(f"Overfitting gap (test - train): {mean_rel_l2 - train_rel_l2:+.6f}")
    print(f"Test MSE:         {test_mse:.6e}")
    print(f"Min Relative L2:  {float(jnp.min(per_sample_rel_l2)):.6f}")
    print(f"Max Relative L2:  {float(jnp.max(per_sample_rel_l2)):.6f}")

    # --- Visualization: sample predictions ---
    print()
    print("Generating visualizations...")
    n_vis = min(4, len(x_test))
    fig, axes = plt.subplots(n_vis, 4, figsize=(16, 4 * n_vis))
    fig.suptitle("FNO Darcy Flow Predictions (Opifex)", fontsize=14, fontweight="bold")

    if n_vis == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_vis):
        im0 = axes[i, 0].imshow(x_test[i, 0], cmap="viridis")
        axes[i, 0].set_title("Input (Permeability)" if i == 0 else "")
        axes[i, 0].axis("off")
        if i == 0:
            plt.colorbar(im0, ax=axes[i, 0], shrink=0.8)

        im1 = axes[i, 1].imshow(y_test[i, 0], cmap="RdBu_r")
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")
        axes[i, 1].axis("off")
        if i == 0:
            plt.colorbar(im1, ax=axes[i, 1], shrink=0.8)

        pred_np = np.array(predictions[i, 0])
        im2 = axes[i, 2].imshow(pred_np, cmap="RdBu_r")
        axes[i, 2].set_title("FNO Prediction" if i == 0 else "")
        axes[i, 2].axis("off")
        if i == 0:
            plt.colorbar(im2, ax=axes[i, 2], shrink=0.8)

        error = np.abs(pred_np - y_test[i, 0])
        im3 = axes[i, 3].imshow(error, cmap="Reds")
        axes[i, 3].set_title("Absolute Error" if i == 0 else "")
        axes[i, 3].axis("off")
        if i == 0:
            plt.colorbar(im3, ax=axes[i, 3], shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample predictions saved to {output_dir / 'sample_predictions.png'}")

    # --- Visualization: error analysis ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FNO Error Analysis", fontsize=14, fontweight="bold")

    per_sample_errors = np.array(per_sample_rel_l2)

    axes[0].hist(per_sample_errors, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    axes[0].set_title("Relative L2 Error Distribution")
    axes[0].set_xlabel("Relative L2 Error")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(per_sample_errors, "o-", alpha=0.7, color="coral", markersize=3)
    axes[1].set_title("Relative L2 Error per Sample")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Relative L2 Error")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Error analysis saved to {output_dir / 'error_analysis.png'}")

    print()
    print("=" * 70)
    print(f"FNO Darcy Flow example completed in {training_time:.1f}s")
    print(f"Test MSE: {test_mse:.6e}, Relative L2: {mean_rel_l2:.6f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return {
        "param_count": param_count,
        "final_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "test_mse": test_mse,
        "rel_l2": mean_rel_l2,
        "train_rel_l2": train_rel_l2,
    }


# %% [markdown]
"""
## Results Summary

After running this example you should observe:
- A decreasing relative-L2 training loss with the exponential learning-rate decay
- A low relative L2 error on the Darcy flow test set, with a small
  train-vs-test gap (the relative-L2 loss + weight decay + LR schedule prevent
  overfitting)
- Visualizations showing input permeability, ground truth pressure,
  FNO predictions, and pointwise error maps

## Next Steps

- Increase resolution and training epochs for even better accuracy
- Experiment with different numbers of Fourier modes and layers
- Compare the relative-L2 objective against an H1 (Sobolev) gradient-aware loss
- Try the UNO architecture for multi-scale Darcy flow problems
- Explore `TensorizedFourierNeuralOperator` for parameter-efficient training
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
