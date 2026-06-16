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
# # TFNO on Darcy Flow
#
# | Property      | Value                                    |
# |---------------|------------------------------------------|
# | Level         | Intermediate                             |
# | Runtime       | ~5 min (CPU), ~1 min (GPU)               |
# | Memory        | ~2 GB                                    |
# | Prerequisites | JAX, Flax NNX, Neural Operators basics   |
#
# ## Overview
#
# Train a Tensorized Fourier Neural Operator (TFNO) on the Darcy flow problem.
# A TFNO is an ordinary FNO whose spectral-convolution weights are stored as a
# low-rank **tensor factorization** (CP / Tucker / Tensor-Train). At low rank this
# uses a small fraction of the dense weight's parameters while retaining accuracy.
#
# This example demonstrates:
#
# - **`create_tucker_fno()`** factory for a Tucker-factorized FNO
# - **Genuine low-rank compression** of the spectral weights (parameter count
#   ``<<`` the dense FNO) measured with `get_compression_stats()`
# - **Grid positional embedding** + **relative-L2 loss**, the standard recipe for
#   operator learning on boundary-value problems
# - **Comparison** with the dense FNO parameter count
#
# Equivalent to `neuraloperator` Tucker FNO examples, reimplemented with Opifex.
#
# ## Learning Goals
#
# 1. Use `create_tucker_fno()` for a parameter-efficient FNO
# 2. Understand Tucker compression of spectral weights
# 3. Train with the relative-L2 loss and Gaussian input/output normalization
# 4. Compare TFNO vs dense FNO parameter counts and read the accuracy

# %% [markdown]
# ## Imports and Setup

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
from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.fno.tensorized import create_tucker_fno


# %% [markdown]
# ## Configuration
#
# The rank parameter controls compression: `rank=0.5` keeps each Tucker mode at
# half its dense size, giving a large parameter reduction while preserving the
# accuracy needed to resolve the Darcy solution.

# %%
RESOLUTION = 64
N_TRAIN = 1024
N_TEST = 256
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
MODES = (16, 16)
HIDDEN_WIDTH = 32
NUM_LAYERS = 4
RANK = 0.5  # Tucker compression ratio (50% of each mode dimension)
PERMEABILITY_VALUES = (3.0, 12.0)  # binary high-contrast benchmark (Li et al. 2020)
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/tfno_darcy")


# %% [markdown]
# ## Run the Example
#
# `main()` loads the binary Darcy data, builds the Tucker TFNO and a dense FNO
# baseline, trains with the relative-L2 loss, evaluates, saves the figures, and
# returns a small dict of finite metrics.


# %%
def main() -> dict[str, float | int]:
    """Train and evaluate a Tucker-factorized TFNO on the Darcy flow problem."""
    print("=" * 70)
    print("Opifex Example: TFNO (Tucker-Factorized FNO) on Darcy Flow")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
    print(f"FNO config: modes={MODES}, width={HIDDEN_WIDTH}, layers={NUM_LAYERS}, rank={RANK}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data loading ---
    print()
    print("Generating Darcy flow data...")
    n_samples = N_TRAIN + N_TEST
    loaders = create_darcy_loader(
        n_samples=n_samples,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        field_type="binary",  # high-contrast benchmark (a in {3, 12})
        coeff_range=PERMEABILITY_VALUES,
        val_fraction=N_TEST / n_samples,
        seed=SEED,
    )

    def _collect(pipeline) -> tuple[np.ndarray, np.ndarray]:
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    X_train, Y_train = _collect(loaders.train)
    X_test, Y_test = _collect(loaders.val)
    # Batches are already channels-first (N, 1, H, W) for Darcy.

    print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test data:     X={X_test.shape}, Y={Y_test.shape}")

    # --- Normalization ---
    x_mean, x_std = X_train.mean(), X_train.std()
    y_mean, y_std = Y_train.mean(), Y_train.std()

    X_train_n = (X_train - x_mean) / x_std
    Y_train_n = (Y_train - y_mean) / y_std
    X_test_n = (X_test - x_mean) / x_std

    print(f"Input mean/std:  {x_mean:.4f} / {x_std:.4f}")
    print(f"Output mean/std: {y_mean:.6f} / {y_std:.6f}")

    # --- Model creation and comparison ---
    print()
    print("Creating TFNO model (Tucker-factorized)...")
    tfno_model = create_tucker_fno(
        in_channels=1,
        out_channels=1,
        hidden_channels=HIDDEN_WIDTH,
        modes=MODES,
        rank=RANK,
        num_layers=NUM_LAYERS,
        rngs=nnx.Rngs(SEED),
    )
    tfno_params = nnx.state(tfno_model, nnx.Param)
    tfno_param_count = sum(x.size for x in jax.tree_util.tree_leaves(tfno_params))

    print("Creating dense FNO for comparison...")
    fno_model = FourierNeuralOperator(
        in_channels=1,
        out_channels=1,
        hidden_channels=HIDDEN_WIDTH,
        modes=max(MODES),
        num_layers=NUM_LAYERS,
        positional_embedding=True,
        rngs=nnx.Rngs(SEED + 1),
    )
    fno_params = nnx.state(fno_model, nnx.Param)
    fno_param_count = sum(x.size for x in jax.tree_util.tree_leaves(fno_params))

    stats = tfno_model.get_compression_stats()

    print()
    print("Model: Tucker-Factorized FNO (TFNO)")
    print(f"  TFNO parameters: {tfno_param_count:,}")
    print(f"  Dense FNO parameters: {fno_param_count:,}")
    print(f"  Parameter reduction: {(1 - tfno_param_count / fno_param_count) * 100:.1f}%")
    print("Spectral-weight compression (all factorized layers):")
    print(f"  Factorized params: {int(stats['factorized_parameters']):,}")
    print(f"  Dense equivalent:  {int(stats['equivalent_dense_parameters']):,}")
    print(f"  Compression ratio: {stats['compression_ratio']:.4f}")

    # --- Training ---
    print()
    print("Setting up Trainer...")
    config = TrainingConfig(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        verbose=True,
        loss_config=LossConfig(loss_type="relative_l2"),
    )
    trainer = Trainer(model=tfno_model, config=config, rngs=nnx.Rngs(SEED))

    print(f"Optimizer: Adam (lr={LEARNING_RATE}), loss: relative L2")
    print("Starting training...")
    start_time = time.time()
    trained_model, metrics = trainer.fit(
        train_data=(jnp.array(X_train_n), jnp.array(Y_train_n)),
        val_data=(jnp.array(X_test_n), jnp.array((Y_test - y_mean) / y_std)),
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")
    print(f"Final train loss: {metrics.get('final_train_loss', 'N/A')}")
    print(f"Final val loss:   {metrics.get('final_val_loss', 'N/A')}")

    # --- Evaluation ---
    print()
    print("Running evaluation...")
    X_test_jnp = jnp.array(X_test_n)
    Y_test_jnp = jnp.array(Y_test)

    predictions = predict_in_batches(trained_model, X_test_jnp) * y_std + y_mean

    test_mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))
    per_sample_rel_l2 = per_sample_relative_l2(predictions, Y_test_jnp)
    mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))

    print(f"Test MSE:         {test_mse:.6e}")
    print(f"Test Relative L2: {mean_rel_l2:.6f}")
    print(f"Min Relative L2:  {float(jnp.min(per_sample_rel_l2)):.6f}")
    print(f"Max Relative L2:  {float(jnp.max(per_sample_rel_l2)):.6f}")

    # --- Visualization: sample predictions ---
    print()
    print("Generating visualizations...")
    n_vis = min(4, len(X_test))
    fig, axes = plt.subplots(n_vis, 4, figsize=(16, 4 * n_vis))
    fig.suptitle("TFNO Darcy Flow Predictions (Opifex)", fontsize=14, fontweight="bold")

    for i in range(n_vis):
        axes[i, 0].imshow(X_test[i, 0], cmap="viridis")
        axes[i, 0].set_title("Input (Permeability)" if i == 0 else "")
        axes[i, 0].set_ylabel(f"Sample {i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(Y_test[i, 0], cmap="RdBu_r")
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")
        axes[i, 1].axis("off")

        pred_np = np.array(predictions[i, 0])
        axes[i, 2].imshow(pred_np, cmap="RdBu_r")
        axes[i, 2].set_title("TFNO Prediction" if i == 0 else "")
        axes[i, 2].axis("off")

        error = np.abs(pred_np - Y_test[i, 0])
        im = axes[i, 3].imshow(error, cmap="Reds")
        axes[i, 3].set_title("Absolute Error" if i == 0 else "")
        axes[i, 3].axis("off")
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample predictions saved to {OUTPUT_DIR / 'predictions.png'}")

    # --- Visualization: error and compression analysis ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("TFNO Analysis", fontsize=14, fontweight="bold")

    per_sample_errors = np.array(per_sample_rel_l2)
    axes[0].hist(per_sample_errors, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Relative L2 Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].grid(True, alpha=0.3)

    models = ["Dense\nFNO", "Tucker\nTFNO"]
    params = [fno_param_count, tfno_param_count]
    colors = ["coral", "steelblue"]
    bars = axes[1].bar(models, params, color=colors, edgecolor="black", alpha=0.7)
    axes[1].set_ylabel("Number of Parameters")
    axes[1].set_title("Parameter Comparison")
    for bar, count in zip(bars, params, strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].plot(per_sample_errors, "o-", alpha=0.7, color="coral", markersize=3)
    axes[2].set_xlabel("Sample Index")
    axes[2].set_ylabel("Relative L2 Error")
    axes[2].set_title("Error per Sample")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Analysis saved to {OUTPUT_DIR / 'analysis.png'}")

    print()
    print("=" * 70)
    print(f"TFNO Darcy example completed in {training_time:.1f}s")
    print(f"Test MSE: {test_mse:.6e}, Relative L2: {mean_rel_l2:.6f}")
    print(f"Parameters: TFNO={tfno_param_count:,} vs dense FNO={fno_param_count:,}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return {
        "test_mse": test_mse,
        "test_rel_l2": mean_rel_l2,
        "tfno_parameters": int(tfno_param_count),
        "dense_fno_parameters": int(fno_param_count),
        "compression_ratio": float(stats["compression_ratio"]),
        "training_time_s": training_time,
    }


# %% [markdown]
# ## Results Summary
#
# The TFNO reaches a low relative L2 error on Darcy flow while using a small
# fraction of the dense FNO's spectral parameters — the Tucker factorization
# compresses the spectral weights without sacrificing accuracy.
#
# ## Next Steps
#
# - Try different rank values (0.25, 0.75) to explore the accuracy-compression tradeoff
# - Compare with CP (`create_cp_fno()`) and Tensor Train (`create_tt_fno()`) factorizations
# - Apply TFNO to larger problems where the memory savings are most significant
#
# ### Related Examples
#
# - [FNO on Darcy Flow](fno-darcy.md) — Standard FNO baseline
# - [FNO on Burgers Equation](fno-burgers.md) — 1D temporal evolution
# - [Operator Comparison Tour](operator-tour.md) — Compare all operators

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
