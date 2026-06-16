# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# Simple SFNO for Climate Modeling

| Property      | Value                                        |
|---------------|----------------------------------------------|
| Level         | Intermediate                                 |
| Runtime       | ~3 min (CPU/GPU)                             |
| Prerequisites | JAX, Flax NNX, Spherical Harmonics basics    |

## Overview
This example demonstrates the Spherical Fourier Neural Operator (SFNO) for climate
modeling using the Opifex framework. The SFNO operates on spherical domains using
spherical harmonic transforms, making it well-suited for global climate and weather
prediction tasks.

We use Opifex's `create_climate_sfno` factory to build the model, the
`create_shallow_water_loader` for streaming data via datarax, and the
`Trainer` with `TrainingConfig` for the training loop.

## Learning Goals
1. Create an SFNO with `create_climate_sfno` factory
2. Load climate data with `create_shallow_water_loader` (datarax-based)
3. Train with Opifex's `Trainer.fit()` API
4. Evaluate and visualize climate predictions on a spherical domain
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import math
import time
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

# Opifex framework imports
from opifex.core.training import Trainer, TrainingConfig
from opifex.data.loaders import create_shallow_water_loader
from opifex.neural.operators.fno.spherical import create_climate_sfno


# %% [markdown]
"""
## Configuration

We define experiment parameters as simple variables. In production, you might
use `argparse` or configuration files.
"""

# %%
RESOLUTION = 32
N_TRAIN = 50
N_TEST = 10
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/sfno_climate_simple")

# %% [markdown]
"""
## Run the example

All runtime logic — data loading via datarax, model creation, training with the
Opifex `Trainer`, evaluation, and visualization — lives in `main()`. Nothing
heavy runs at import time, so the module can be imported cheaply (e.g. by the
example smoke tests).
"""


# %%
def main() -> dict[str, float | int]:
    """Train a climate SFNO, evaluate it, save plots, and return scalar metrics."""
    print("=" * 70)
    print("Opifex Example: Simple Spherical FNO for Climate Modeling")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    print(f"Output directory: {OUTPUT_DIR}")

    # --- Data loading with datarax ---
    print()
    print("Loading shallow water equation data via datarax...")
    n_samples = N_TRAIN + N_TEST
    loaders = create_shallow_water_loader(
        n_samples=n_samples,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        val_fraction=N_TEST / n_samples,
        seed=SEED,
    )

    def _collect(pipeline) -> tuple[np.ndarray, np.ndarray]:
        # Batches are channels-first dicts: input/output each (batch, 3, H, W).
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    x_train, y_train = _collect(loaders.train)
    x_test, y_test = _collect(loaders.val)

    print(f"Training data: X={x_train.shape}, Y={y_train.shape}")
    print(f"Test data:     X={x_test.shape}, Y={y_test.shape}")

    # --- Model creation ---
    print()
    print("Creating Spherical FNO model...")
    in_channels = x_train.shape[1]
    out_channels = y_train.shape[1]

    model = create_climate_sfno(
        in_channels=in_channels,
        out_channels=out_channels,
        lmax=8,
        rngs=nnx.Rngs(SEED),
    )

    print("Model: Spherical FNO (lmax=8)")
    print(f"Input channels: {in_channels}, Output channels: {out_channels}")

    # --- Training with Opifex Trainer ---
    print()
    print("Setting up Trainer...")
    config = TrainingConfig(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        verbose=True,
    )

    trainer = Trainer(
        model=model,
        config=config,
        rngs=nnx.Rngs(SEED),
    )

    print(f"Optimizer: Adam (lr={LEARNING_RATE})")
    print()
    print("Starting training...")
    start_time = time.time()

    trained_model, metrics = trainer.fit(
        train_data=(jnp.array(x_train), jnp.array(y_train)),
        val_data=(jnp.array(x_test), jnp.array(y_test)),
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")
    print(f"Final train loss: {metrics.get('final_train_loss', 'N/A')}")
    print(f"Final val loss:   {metrics.get('final_val_loss', 'N/A')}")

    # --- Evaluation ---
    print()
    print("Evaluating on test set...")
    x_test_jnp = jnp.array(x_test)
    y_test_jnp = jnp.array(y_test)

    predictions = trained_model(x_test_jnp)

    test_mse = float(jnp.mean((predictions - y_test_jnp) ** 2))

    # Relative L2 error per sample
    pred_diff = (predictions - y_test_jnp).reshape(predictions.shape[0], -1)
    y_flat = y_test_jnp.reshape(y_test_jnp.shape[0], -1)
    rel_l2 = float(
        jnp.mean(jnp.linalg.norm(pred_diff, axis=1) / jnp.linalg.norm(y_flat, axis=1))
    )

    print(f"Test MSE:         {test_mse:.6f}")
    print(f"Test Relative L2: {rel_l2:.6f}")

    # --- Visualization ---
    print()
    print("Generating visualization...")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Spherical FNO Climate Prediction (Opifex)", fontsize=14, fontweight="bold")

    sample_idx = 0

    # Input
    im0 = axes[0].imshow(x_test[sample_idx, 0], cmap="RdBu_r", aspect="equal")
    axes[0].set_title("Input")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Ground truth
    im1 = axes[1].imshow(y_test[sample_idx, 0], cmap="RdBu_r", aspect="equal")
    axes[1].set_title("Ground Truth")
    axes[1].set_xlabel("Longitude")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Prediction
    pred_np = np.array(predictions[sample_idx, 0])
    im2 = axes[2].imshow(pred_np, cmap="RdBu_r", aspect="equal")
    axes[2].set_title("SFNO Prediction")
    axes[2].set_xlabel("Longitude")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    # Error
    error = np.abs(pred_np - y_test[sample_idx, 0])
    im3 = axes[3].imshow(error, cmap="plasma", aspect="equal")
    axes[3].set_title("Absolute Error")
    axes[3].set_xlabel("Longitude")
    plt.colorbar(im3, ax=axes[3], shrink=0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sfno_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to {OUTPUT_DIR / 'sfno_results.png'}")

    print()
    print("=" * 70)
    print(f"Spherical FNO Climate example completed in {training_time:.1f}s")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    # --- Metrics dict (finite scalars only) ---
    param_count = sum(int(p.size) for p in jax.tree.leaves(nnx.state(trained_model, nnx.Param)))
    summary: dict[str, float | int] = {
        "test_mse": test_mse,
        "l2_relative_error": rel_l2,
        "param_count": param_count,
    }

    final_train_loss = metrics.get("final_train_loss", "N/A")
    if isinstance(final_train_loss, (int, float)) and math.isfinite(final_train_loss):
        summary["final_train_loss"] = float(final_train_loss)

    return summary


# %% [markdown]
"""
## Results Summary

After running this example you should observe:
- Decreasing training loss over epochs on spherical domain data
- Reasonable predictions for the shallow water equations proxy
- Visualization comparing input, ground truth, SFNO prediction, and error

## Next Steps
- Try the full SFNO example with conservation-aware loss
- Increase `lmax` for higher spectral resolution
- Experiment with more training samples and epochs
- Explore energy and mass conservation analysis
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
