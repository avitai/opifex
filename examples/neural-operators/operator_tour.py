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
# Neural Operator Comparison Tour

| Property      | Value                                   |
|---------------|-----------------------------------------|
| **Level**     | Advanced                                |
| **Runtime**   | ~10 min (CPU), ~3 min (GPU)             |
| **Prerequisites** | JAX, Flax NNX, Neural Operators     |

## Overview

This example tours the Fourier-family neural operators in Opifex and runs a
**fair, head-to-head comparison** of them on a single Darcy-flow benchmark. Every
operator is trained with the *same* recipe -- grid positional embedding, Gaussian
input/output normalization, the relative-L2 loss, and an identical optimizer
budget -- so the resulting accuracy and parameter counts are directly comparable.

The tour covers:

- **Operator discovery** with `list_operators()` and `recommend_operator()`
- **Dense FNO** -- the baseline Fourier Neural Operator
- **Tucker / CP Tensorized FNO** -- low-rank spectral weights for parameter
  efficiency (the compression-vs-accuracy story)
- **Local FNO** -- combined global (spectral) + local (convolution) operations
- **U-FNO** -- a U-Net-style multi-scale Fourier operator

Each operator maps a high-contrast permeability field ``a(x)`` to the Darcy
pressure solution of ``-∇·(a∇u) = 1``. We report the test relative-L2 error, the
parameter count, and the training time, then visualise the trade-offs.

## Learning Goals

1. Discover operators with the factory / recommendation system
2. Apply the *one* operator-learning recipe uniformly across architectures
3. Read a fair accuracy-vs-parameters comparison on real Darcy data
4. Understand the Tucker/CP compression trade-off against a dense FNO
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import time
import warnings
from pathlib import Path
from typing import Any


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from flax import nnx


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.config import LossConfig
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators import (
    FourierNeuralOperator,
    list_operators,
    LocalFourierNeuralOperator,
    recommend_operator,
    UFourierNeuralOperator,
)
from opifex.neural.operators.fno._positional import append_grid_coordinates
from opifex.neural.operators.fno.tensorized import create_cp_fno, create_tucker_fno


# %% [markdown]
"""
## Configuration

Every operator shares this configuration so the comparison is fair: identical
data, identical optimizer budget, and the standard operator-learning recipe.
The Tucker/CP ranks (``0.5``) keep each factor at half its dense size for a real
compression-vs-accuracy story.
"""

# %%
RESOLUTION = 64
N_TRAIN = 1024
N_TEST = 256
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
HIDDEN_CHANNELS = 32
MODES = 16
NUM_LAYERS = 4
TUCKER_RANK = 0.5
CP_RANK = 0.5
UFNO_LEVELS = 3
PERMEABILITY_VALUES = (3.0, 12.0)  # binary high-contrast benchmark (Li et al. 2020)
SEED = 42

OUTPUT_DIR = Path("docs/assets/examples/operator_tour")


# %% [markdown]
"""
## Building the Comparison Set

All five operators map ``(batch, 1, H, W)`` permeability to pressure. The dense
FNO and the Tensorized FNOs append grid coordinates internally; Local FNO and
U-FNO have no built-in positional embedding, so we wrap them so that **every**
operator sees the same normalized grid-coordinate channels. This keeps the
comparison fair -- the only difference between models is the architecture.
"""


# %%
class GridWrapped(nnx.Module):
    """Wrap an operator so it receives appended grid-coordinate channels.

    Local FNO and U-FNO do not embed positional information themselves. Prefixing
    the input with normalized ``(x, y)`` coordinate channels gives them the same
    boundary-aware recipe used by the FNO / TFNO family.
    """

    def __init__(self, operator: nnx.Module) -> None:
        """Store the wrapped operator.

        Args:
            operator: A channels-first neural operator to feed grid-augmented input.
        """
        super().__init__()
        self.operator = operator

    def __call__(self, x: jax.Array) -> jax.Array:
        """Append grid coordinates, then apply the wrapped operator.

        Args:
            x: Input of shape ``(batch, in_channels, height, width)``.

        Returns:
            Operator output of shape ``(batch, out_channels, height, width)``.
        """
        return self.operator(append_grid_coordinates(x))


def build_operators() -> dict[str, nnx.Module]:
    """Construct the five comparison operators with a shared, fair configuration.

    Returns:
        A mapping from display name to an initialized operator. FNO and the
        Tensorized FNOs carry positional embedding internally; Local FNO and
        U-FNO are wrapped so they receive the same grid-coordinate channels.
    """
    grid_in_channels = 1 + 2  # permeability + (x, y) coordinate channels
    return {
        "Dense FNO": FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=HIDDEN_CHANNELS,
            modes=MODES,
            num_layers=NUM_LAYERS,
            positional_embedding=True,
            rngs=nnx.Rngs(SEED),
        ),
        "Tucker TFNO": create_tucker_fno(
            in_channels=1,
            out_channels=1,
            hidden_channels=HIDDEN_CHANNELS,
            modes=(MODES, MODES),
            rank=TUCKER_RANK,
            num_layers=NUM_LAYERS,
            rngs=nnx.Rngs(SEED + 1),
        ),
        "CP TFNO": create_cp_fno(
            in_channels=1,
            out_channels=1,
            hidden_channels=HIDDEN_CHANNELS,
            modes=(MODES, MODES),
            rank=CP_RANK,
            num_layers=NUM_LAYERS,
            rngs=nnx.Rngs(SEED + 2),
        ),
        "Local FNO": GridWrapped(
            LocalFourierNeuralOperator(
                in_channels=grid_in_channels,
                out_channels=1,
                hidden_channels=HIDDEN_CHANNELS,
                modes=(MODES, MODES),
                num_layers=NUM_LAYERS,
                rngs=nnx.Rngs(SEED + 3),
            )
        ),
        "U-FNO": GridWrapped(
            UFourierNeuralOperator(
                in_channels=grid_in_channels,
                out_channels=1,
                hidden_channels=HIDDEN_CHANNELS,
                modes=(MODES, MODES),
                num_levels=UFNO_LEVELS,
                rngs=nnx.Rngs(SEED + 4),
            )
        ),
    }


def count_parameters(model: nnx.Module) -> int:
    """Count the trainable parameters of a model."""
    params = nnx.state(model, nnx.Param)
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def predict_in_batches(model: nnx.Module, inputs: jax.Array, batch_size: int = 128) -> jax.Array:
    """Run a model over the inputs in batches to bound memory use."""
    outputs = [model(inputs[i : i + batch_size]) for i in range(0, inputs.shape[0], batch_size)]
    return jnp.concatenate(outputs, axis=0)


def relative_l2(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Per-sample relative-L2 error between physical-space fields."""
    diff = (predictions - targets).reshape(predictions.shape[0], -1)
    target_flat = targets.reshape(targets.shape[0], -1)
    return jnp.linalg.norm(diff, axis=1) / jnp.linalg.norm(target_flat, axis=1)


def collect_darcy_split(pipeline: Any) -> tuple[np.ndarray, np.ndarray]:
    """Drain a datarax pipeline into channels-first ``(N, 1, H, W)`` arrays."""
    inputs, outputs = [], []
    for batch in pipeline:
        inputs.append(np.asarray(batch["input"]))
        outputs.append(np.asarray(batch["output"]))
    return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)


# %% [markdown]
"""
## Run the Example

`main()` runs operator discovery, loads and normalizes the Darcy data, trains
each of the five operators with the shared recipe, evaluates against a mean-
predictor floor, saves the figures, and returns a small dict of finite metrics
(best operator's error and parameter count, plus the operator count).
"""


# %%
def main() -> dict[str, float | int]:
    """Tour and compare the Fourier-family neural operators on Darcy flow."""
    print("=" * 70)
    print("Opifex Example: Neural Operator Comparison Tour on Darcy Flow")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
    print(f"Shared FNO config: modes={MODES}, width={HIDDEN_CHANNELS}, layers={NUM_LAYERS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Operator discovery ---
    print()
    print("=" * 70)
    print("OPERATOR DISCOVERY")
    print("=" * 70)
    print("Available operators by category:")
    for category, operators_in_cat in list_operators().items():
        print(f"  {category}: {', '.join(operators_in_cat)}")

    print()
    print("Recommendations by application:")
    applications = [
        "turbulent_flow",
        "global_climate",
        "molecular_dynamics",
        "cad_geometry",
        "safety_critical",
        "parameter_efficient",
    ]
    for app in applications:
        rec = recommend_operator(app)
        print(f"  {app:20s}: {rec['primary']} - {rec['reason']}")

    # --- Data loading ---
    print()
    print("Generating Darcy flow data...")
    n_samples = N_TRAIN + N_TEST
    loaders = create_darcy_loader(
        n_samples=n_samples,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        field_type="binary",
        coeff_range=PERMEABILITY_VALUES,
        val_fraction=N_TEST / n_samples,
        seed=SEED,
    )

    X_train, Y_train = collect_darcy_split(loaders.train)
    X_test, Y_test = collect_darcy_split(loaders.val)
    print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test data:     X={X_test.shape}, Y={Y_test.shape}")

    # --- Normalization ---
    x_mean, x_std = X_train.mean(), X_train.std()
    y_mean, y_std = Y_train.mean(), Y_train.std()

    X_train_n = jnp.array((X_train - x_mean) / x_std)
    Y_train_n = jnp.array((Y_train - y_mean) / y_std)
    X_test_n = jnp.array((X_test - x_mean) / x_std)
    Y_test_n = jnp.array((Y_test - y_mean) / y_std)
    Y_test_jnp = jnp.array(Y_test)

    print(f"Input mean/std:  {x_mean:.4f} / {x_std:.4f}")
    print(f"Output mean/std: {y_mean:.6f} / {y_std:.6f}")

    # --- Build the comparison set ---
    operators = build_operators()
    print()
    print("Comparison operators (parameter counts):")
    for name, op in operators.items():
        print(f"  {name:14s}: {count_parameters(op):,} params")

    # --- Train and evaluate each operator ---
    def train_and_evaluate(name: str, model: nnx.Module) -> dict[str, Any]:
        """Train one operator with the shared recipe and evaluate on the test set."""
        print()
        print("-" * 70)
        print(f"Training {name} ({count_parameters(model):,} params)...")
        config = TrainingConfig(
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            validation_frequency=10,
            verbose=False,
            loss_config=LossConfig(loss_type="relative_l2"),
        )
        trainer = Trainer(model=model, config=config, rngs=nnx.Rngs(SEED))

        start = time.time()
        trained_model, metrics = trainer.fit(
            train_data=(X_train_n, Y_train_n),
            val_data=(X_test_n, Y_test_n),
        )
        train_time = time.time() - start

        predictions = predict_in_batches(trained_model, X_test_n) * y_std + y_mean
        per_sample = relative_l2(predictions, Y_test_jnp)
        mean_rel_l2 = float(jnp.mean(per_sample))
        mse = float(jnp.mean((predictions - Y_test_jnp) ** 2))

        print(
            f"  {name}: rel-L2={mean_rel_l2:.4f}, MSE={mse:.3e}, "
            f"time={train_time:.1f}s, "
            f"final val loss={metrics.get('final_val_loss', float('nan')):.4f}"
        )
        return {
            "params": count_parameters(trained_model),
            "train_time_s": train_time,
            "final_train_loss": float(metrics.get("final_train_loss", float("nan"))),
            "final_val_loss": float(metrics.get("final_val_loss", float("nan"))),
            "per_sample_rel_l2": np.array(per_sample),
            "mean_rel_l2": mean_rel_l2,
            "mse": mse,
            "predictions": np.array(predictions),
        }

    print()
    print("=" * 70)
    print("TRAINING COMPARISON")
    print("=" * 70)

    results: dict[str, dict[str, Any]] = {}
    for name, op in operators.items():
        results[name] = train_and_evaluate(name, op)

    # --- Baseline: mean predictor ---
    mean_field = jnp.full_like(Y_test_jnp, float(Y_train.mean()))
    mean_baseline_rel_l2 = float(jnp.mean(relative_l2(mean_field, Y_test_jnp)))
    print()
    print(f"Mean-predictor relative-L2: {mean_baseline_rel_l2:.4f}")
    print("Every trained operator must beat this floor to be useful.")

    # --- Comparison summary ---
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY (sorted by relative-L2)")
    print("=" * 70)
    baseline_params = results["Dense FNO"]["params"]

    print()
    header = (
        f"{'Operator':14s} {'Rel-L2':>9s} {'MSE':>11s} "
        f"{'Params':>11s} {'vs FNO':>8s} {'Time(s)':>9s}"
    )
    print(header)
    print("-" * len(header))
    for name, res in sorted(results.items(), key=lambda kv: kv[1]["mean_rel_l2"]):
        ratio = baseline_params / res["params"] if res["params"] > 0 else 0.0
        print(
            f"{name:14s} {res['mean_rel_l2']:9.4f} {res['mse']:11.3e} "
            f"{res['params']:11,d} {ratio:7.2f}x {res['train_time_s']:9.1f}"
        )
    print(f"{'Mean predictor':14s} {mean_baseline_rel_l2:9.4f}")

    best_name = min(results, key=lambda k: results[k]["mean_rel_l2"])
    print()
    print(f"Best accuracy: {best_name} (rel-L2={results[best_name]['mean_rel_l2']:.4f})")

    # --- Visualization: best-operator predictions ---
    print()
    print("Generating visualizations...")
    best_preds = results[best_name]["predictions"]
    n_vis = 3
    indices = np.linspace(0, len(X_test) - 1, n_vis, dtype=int)

    fig, axes = plt.subplots(n_vis, 4, figsize=(16, 4 * n_vis))
    fig.suptitle(
        f"Best Operator ({best_name}) -- Darcy Flow Predictions",
        fontsize=14,
        fontweight="bold",
    )
    for row, idx in enumerate(indices):
        x_field = X_test[idx, 0]
        y_true = Y_test[idx, 0]
        y_pred = best_preds[idx, 0]
        error = np.abs(y_pred - y_true)

        im0 = axes[row, 0].imshow(x_field, cmap="viridis")
        axes[row, 0].set_title(f"Input {row + 1}: Permeability")
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.8)

        im1 = axes[row, 1].imshow(y_true, cmap="RdBu_r")
        axes[row, 1].set_title(f"Ground Truth {row + 1}")
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.8)

        im2 = axes[row, 2].imshow(y_pred, cmap="RdBu_r")
        axes[row, 2].set_title(f"{best_name} Prediction {row + 1}")
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.8)

        im3 = axes[row, 3].imshow(error, cmap="Reds")
        axes[row, 3].set_title(f"Absolute Error {row + 1}")
        axes[row, 3].axis("off")
        plt.colorbar(im3, ax=axes[row, 3], shrink=0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions saved to {OUTPUT_DIR / 'predictions.png'}")

    # --- Visualization: accuracy, parameters, cost ---
    names = list(results.keys())
    rel_l2_values = [results[n]["mean_rel_l2"] for n in names]
    param_values = [results[n]["params"] for n in names]
    time_values = [results[n]["train_time_s"] for n in names]
    colors = ["steelblue", "coral", "mediumseagreen", "goldenrod", "mediumpurple"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Neural Operator Comparison on Darcy Flow", fontsize=14, fontweight="bold")

    bars0 = axes[0].bar(names, rel_l2_values, color=colors, edgecolor="black", alpha=0.8)
    axes[0].axhline(mean_baseline_rel_l2, color="red", linestyle="--", label="Mean predictor")
    axes[0].set_ylabel("Test Relative L2 (lower is better)")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)
    for bar, value in zip(bars0, rel_l2_values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars1 = axes[1].bar(names, param_values, color=colors, edgecolor="black", alpha=0.8)
    axes[1].set_ylabel("Parameters")
    axes[1].set_title("Model Size")
    axes[1].tick_params(axis="x", rotation=30)
    for bar, value in zip(bars1, param_values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value / 1e3:.0f}k",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars2 = axes[2].bar(names, time_values, color=colors, edgecolor="black", alpha=0.8)
    axes[2].set_ylabel("Training Time (s)")
    axes[2].set_title("Training Cost")
    axes[2].tick_params(axis="x", rotation=30)
    for bar, value in zip(bars2, time_values, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.0f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Analysis saved to {OUTPUT_DIR / 'analysis.png'}")

    print()
    print("=" * 70)
    print("Operator comparison tour complete")
    print(f"Best operator: {best_name} (rel-L2={results[best_name]['mean_rel_l2']:.4f})")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    best = results[best_name]
    return {
        "num_operators": len(results),
        "best_rel_l2": best["mean_rel_l2"],
        "best_mse": best["mse"],
        "best_parameters": int(best["params"]),
        "mean_baseline_rel_l2": mean_baseline_rel_l2,
    }


# %% [markdown]
"""
## Results Summary

After running this tour you will have:

- Trained five Fourier-family operators on the *same* Darcy benchmark with the
  *same* recipe, so their accuracy and size are directly comparable
- A clear floor: every operator beats the mean predictor
- The compression-vs-accuracy trade-off of the Tensorized (Tucker / CP) FNOs
  against the dense FNO

## Next Steps

- Sweep the Tucker / CP `rank` (0.25, 0.5, 0.75) to map the accuracy-compression
  curve
- Increase `hidden_channels`, `modes`, or `num_layers` for higher capacity
- Add the Tensor-Train factorization with `create_tt_fno()`
- Try the specialized operators (SFNO, GINO, MGNO, UQNO) on their native domains

### Related Examples

- [FNO on Darcy Flow](fno-darcy.md) -- standard FNO training pipeline
- [TFNO on Darcy Flow](tfno-darcy.md) -- the Tucker compression story in depth
- [UNO on Darcy Flow](uno-darcy.md) -- multi-resolution U-shaped operator
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
