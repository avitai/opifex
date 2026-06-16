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
# Full U-FNO for Turbulence Modeling

| Property    | Value                                                      |
|-------------|------------------------------------------------------------|
| Level       | Advanced                                                   |
| Runtime     | ~15 min (CPU/GPU)                                          |
| Prerequisites | JAX, Flax NNX, Multi-scale Analysis, Energy Conservation |

## Overview
This example demonstrates U-FNO functionality for multi-scale 2D Navier-Stokes
turbulence modeling using the Opifex framework with JAX/Flax NNX. The operator
maps the **initial velocity field** ``(u, v)`` to the **final-time velocity
field** of an incompressible Navier-Stokes flow. Features include grid
embeddings, physics-aware energy conservation loss, and full turbulence analysis.

We use Opifex's `create_turbulence_ufno` factory, `GridEmbedding2D` for positional
encoding, `create_navier_stokes_loader` for data (via datarax), and `Trainer.fit()`
with custom energy conservation loss via `trainer.custom_losses`.

## Learning Goals
1. Build U-FNO with `create_turbulence_ufno` and `GridEmbedding2D`
2. Train with Opifex `Trainer` using custom physics loss
3. Analyze multi-scale frequency content
4. Evaluate energy conservation and prediction quality
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
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from opifex.core.training import Trainer, TrainingConfig
from opifex.data.loaders.factory import create_navier_stokes_loader
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.ufno import create_turbulence_ufno


# %% [markdown]
"""
## Helper functions

`apply_embedding` pre-applies the grid embedding to data arrays so that
`Trainer.fit()` can work with the embedded inputs directly. `energy_loss_fn`
implements the physics-aware energy conservation penalty.
"""


# %%
def apply_embedding(x_data: np.ndarray, embedding: GridEmbedding2D) -> np.ndarray:
    """Apply grid embedding: (B, C, H, W) -> embed -> (B, C+2, H, W)."""
    x_grid = jnp.moveaxis(jnp.array(x_data), 1, -1)  # (B, H, W, C)
    x_embedded = embedding(x_grid)  # (B, H, W, C+2)
    return np.array(jnp.moveaxis(x_embedded, -1, 1))  # (B, C+2, H, W)


def energy_loss_fn(
    model: nnx.Module, x: jax.Array, y_pred: jax.Array, y_true: jax.Array
) -> jax.Array:
    """Energy conservation loss: penalize energy deviation."""
    pred_energy = jnp.mean(y_pred**2, axis=(2, 3))
    target_energy = jnp.mean(y_true**2, axis=(2, 3))
    return 0.1 * jnp.mean(jnp.abs(pred_energy - target_energy))


# %% [markdown]
"""
## Run the example

All run logic (data loading, model creation, training, evaluation, plotting)
lives in ``main`` so nothing heavy executes at import time.
"""


# %%
def main() -> dict[str, float | int]:
    """Train a U-FNO on 2D Navier-Stokes turbulence and return finite scalar metrics."""
    # --- Configuration ---
    resolution = 64
    n_train = 300
    n_test = 60
    batch_size = 16
    num_epochs = 5
    learning_rate = 1e-3
    in_channels = 2  # (u, v) velocity components
    out_channels = 2  # (u, v) velocity components
    seed = 42

    output_dir = Path("docs/assets/examples/ufno_turbulence")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Opifex Example: Full U-FNO for 2D Navier-Stokes Turbulence Modeling")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Resolution: {resolution}x{resolution}, Samples: {n_train}/{n_test}")
    print(f"Batch: {batch_size}, Epochs: {num_epochs}")

    # --- Data Loading with datarax ---
    # Initial velocity field (u, v) -> final-time velocity field (u, v).
    # Batches are channels-first dicts: input/output each shaped (batch, 2, H, W).
    print("\nLoading 2D Navier-Stokes turbulence data via datarax...")
    n_samples = n_train + n_test
    loaders = create_navier_stokes_loader(
        n_samples=n_samples,
        batch_size=batch_size,
        resolution=resolution,
        viscosity_range=(0.001, 0.005),
        time_range=(0.0, 1.0),
        val_fraction=n_test / n_samples,
        seed=seed,
    )

    def _collect(pipeline: object) -> tuple[np.ndarray, np.ndarray]:
        """Materialize a datarax pipeline into channels-first (N, 2, H, W) arrays."""
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    x_train, y_train = _collect(loaders.train)
    x_test, y_test = _collect(loaders.val)

    print(f"Train: X={x_train.shape}, Y={y_train.shape}")
    print(f"Test:  X={x_test.shape}, Y={y_test.shape}")

    # --- Model Creation with Grid Embedding ---
    print("\nCreating U-FNO model with grid embedding...")
    grid_embedding = GridEmbedding2D(
        in_channels=in_channels, grid_boundaries=[[0.0, 1.0], [0.0, 1.0]]
    )

    model = create_turbulence_ufno(
        in_channels=grid_embedding.out_channels,
        out_channels=out_channels,
        rngs=nnx.Rngs(seed),
    )

    print(f"GridEmbedding2D: {in_channels} -> {grid_embedding.out_channels} channels")
    print(f"U-FNO: {grid_embedding.out_channels} -> {out_channels} channels")

    param_count = sum(int(p.size) for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Model parameters: {param_count:,}")

    # --- Apply grid embedding to data ---
    print("\nApplying grid embedding to data...")
    x_train_emb = apply_embedding(x_train, grid_embedding)
    x_test_emb = apply_embedding(x_test, grid_embedding)

    print(f"Embedded train: {x_train_emb.shape}")
    print(f"Embedded test:  {x_test_emb.shape}")

    # --- Training with Opifex Trainer ---
    print("\nSetting up Trainer...")
    config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=True,
    )
    trainer = Trainer(model=model, config=config, rngs=nnx.Rngs(seed))

    # Register energy conservation as custom loss
    trainer.custom_losses["energy"] = energy_loss_fn
    print(f"Optimizer: Adam (lr={learning_rate})")
    print("Custom loss: energy conservation (weight=0.1)")

    print("\nStarting training...")
    start_time = time.time()
    trained_model, metrics = trainer.fit(
        train_data=(jnp.array(x_train_emb), jnp.array(y_train)),
        val_data=(jnp.array(x_test_emb), jnp.array(y_test)),
    )
    training_time = time.time() - start_time
    final_train_loss = float(metrics.get("final_train_loss", 0.0))
    final_val_loss = float(metrics.get("final_val_loss", 0.0))
    print(
        f"Done in {training_time:.1f}s | Train: {final_train_loss:.6f} | Val: {final_val_loss:.6f}"
    )

    # --- Full Evaluation ---
    print("\nRunning full evaluation...")
    x_test_jnp = jnp.array(x_test_emb)
    y_test_jnp = jnp.array(y_test)
    predictions = trained_model(x_test_jnp)

    test_mse = float(jnp.mean((predictions - y_test_jnp) ** 2))

    per_sample_errors = []
    for i in range(y_test_jnp.shape[0]):
        p, t = predictions[i : i + 1], y_test_jnp[i : i + 1]
        per_sample_errors.append(float(jnp.sqrt(jnp.sum((p - t) ** 2)) / jnp.sqrt(jnp.sum(t**2))))
    mean_error, std_error = (
        float(np.mean(per_sample_errors)),
        float(np.std(per_sample_errors)),
    )

    pred_energy = jnp.mean(predictions**2, axis=(2, 3))
    target_energy = jnp.mean(y_test_jnp**2, axis=(2, 3))
    energy_conservation = float(jnp.mean(jnp.abs(pred_energy - target_energy)))

    print(f"MSE: {test_mse:.6f} | Rel L2: {mean_error:.6f}+/-{std_error:.6f}")
    print(f"Energy Conservation: {energy_conservation:.6f}")

    # --- Visualization: Training Curves ---
    print("\nGenerating training curves...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "U-FNO Training - Navier-Stokes Turbulence Modeling",
        fontsize=16,
        fontweight="bold",
    )

    axes[0, 0].bar(
        ["Train", "Val"],
        [final_train_loss, final_val_loss],
        color=["steelblue", "indianred"],
    )
    axes[0, 0].set_title("Final Loss", fontweight="bold")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(["Test MSE"], [test_mse], color="steelblue")
    axes[0, 1].set_title("Test MSE", fontweight="bold")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].bar(["Rel L2"], [mean_error], yerr=[std_error], color="steelblue", capsize=5)
    axes[0, 2].set_title("Relative L2", fontweight="bold")
    axes[0, 2].set_ylabel("Rel L2")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].bar(["Energy"], [energy_conservation], color="darkorange")
    axes[1, 0].set_title("Energy Conservation Error", fontweight="bold")
    axes[1, 0].set_ylabel("Error")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(per_sample_errors, "o-", ms=4, lw=1, color="darkblue")
    axes[1, 1].axhline(mean_error, color="red", ls="--", label=f"Mean: {mean_error:.4f}")
    axes[1, 1].set_title("Per-Sample Error", fontweight="bold")
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].set_ylabel("Rel L2")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    s = (
        f"\nMetrics Summary:\n\nMSE: {test_mse:.6f}\nRel L2: {mean_error:.6f}+/-{std_error:.6f}\n"
        f"Energy: {energy_conservation:.6f}\nTime: {training_time:.1f}s\n"
    )
    axes[1, 2].text(
        0.1,
        0.5,
        s,
        fontsize=11,
        transform=axes[1, 2].transAxes,
        va="center",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7},
    )
    axes[1, 2].set_title("Summary", fontweight="bold")
    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Visualization: Sample Predictions (velocity magnitude) ---
    print("Generating sample predictions...")
    pred_np = np.array(predictions)
    n_show = min(3, pred_np.shape[0])

    def _velocity_magnitude(field: np.ndarray) -> np.ndarray:
        """Channels-first (2, H, W) [u, v] -> speed |(u, v)| = sqrt(u^2 + v^2)."""
        return np.sqrt(field[0] ** 2 + field[1] ** 2)

    fig, axes = plt.subplots(n_show, 4, figsize=(20, 5 * n_show))
    if n_show == 1:
        axes = axes[None, :]
    fig.suptitle(
        "U-FNO Navier-Stokes Predictions (velocity magnitude)",
        fontsize=16,
        fontweight="bold",
    )

    for i in range(n_show):
        input_mag = _velocity_magnitude(x_test[i])
        truth_mag = _velocity_magnitude(y_test[i])
        pred_mag = _velocity_magnitude(pred_np[i])
        for ax, d, t in [
            (axes[i, 0], input_mag, f"Initial |v| {i + 1}"),
            (axes[i, 1], truth_mag, f"Final |v| Truth {i + 1}"),
            (axes[i, 2], pred_mag, f"Final |v| U-FNO {i + 1}"),
        ]:
            im = ax.imshow(d, cmap="viridis", aspect="equal")
            ax.set_title(t, fontweight="bold")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(im, ax=ax, shrink=0.8)
        err = np.abs(pred_mag - truth_mag)
        im3 = axes[i, 3].imshow(err, cmap="plasma", aspect="equal")
        axes[i, 3].set_title(f"|v| Error {i + 1}", fontweight="bold")
        axes[i, 3].set_xlabel("x")
        plt.colorbar(im3, ax=axes[i, 3], shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "sample_predictions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Visualization: Multi-Scale Analysis ---
    print("Generating multi-scale analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("U-FNO Multi-Scale Analysis", fontsize=16, fontweight="bold")

    # Multi-scale analysis on the velocity-magnitude field of the first sample.
    pred_mag0 = _velocity_magnitude(pred_np[0])
    target_mag0 = _velocity_magnitude(y_test[0])
    pred_fft = np.abs(np.fft.fft2(pred_mag0))
    target_fft = np.abs(np.fft.fft2(target_mag0))

    axes[0, 0].semilogy(np.mean(pred_fft, axis=0), "b-", label="U-FNO", lw=2)
    axes[0, 0].semilogy(np.mean(target_fft, axis=0), "r--", label="Truth", lw=2)
    axes[0, 0].set_title("Frequency Content (x)", fontweight="bold")
    axes[0, 0].set_xlabel("Mode")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(np.mean(pred_fft, axis=1), "b-", label="U-FNO", lw=2)
    axes[0, 1].semilogy(np.mean(target_fft, axis=1), "r--", label="Truth", lw=2)
    axes[0, 1].set_title("Frequency Content (y)", fontweight="bold")
    axes[0, 1].set_xlabel("Mode")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    scales = [1, 2, 4, 8]
    scale_errors = []
    for sc in scales:
        ps = pred_mag0[::sc, ::sc]
        ts = target_mag0[::sc, ::sc]
        scale_errors.append(np.mean((ps - ts) ** 2))
    axes[1, 0].loglog(scales, scale_errors, "go-", lw=2, ms=8)
    axes[1, 0].set_title("Multi-Scale Error", fontweight="bold")
    axes[1, 0].set_xlabel("Scale")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(True, alpha=0.3)

    pred_es = np.mean(pred_fft**2, axis=0)
    target_es = np.mean(target_fft**2, axis=0)
    freqs = np.fft.fftfreq(pred_np.shape[-1])
    pf = freqs[freqs > 0]
    axes[1, 1].loglog(pf, pred_es[: len(pf)], "b-", label="U-FNO", lw=2)
    axes[1, 1].loglog(pf, target_es[: len(pf)], "r--", label="Truth", lw=2)
    axes[1, 1].set_title("Energy Spectrum", fontweight="bold")
    axes[1, 1].set_xlabel("Frequency")
    axes[1, 1].set_ylabel("Energy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "multiscale_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Visualization: Error Analysis ---
    print("Generating error analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("U-FNO Error Analysis", fontsize=16, fontweight="bold")

    axes[0, 0].hist(per_sample_errors, bins=15, alpha=0.7, color="lightcoral", edgecolor="black")
    axes[0, 0].axvline(mean_error, color="red", ls="--", label=f"Mean: {mean_error:.4f}", lw=2)
    axes[0, 0].set_title("Error Distribution", fontweight="bold")
    axes[0, 0].set_xlabel("Rel L2")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(per_sample_errors, "o-", ms=6, lw=2, color="darkblue")
    axes[0, 1].set_title("Error vs Sample", fontweight="bold")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].grid(True, alpha=0.3)

    se = np.sort(per_sample_errors)
    cu = np.arange(1, len(se) + 1) / len(se)
    axes[1, 0].plot(se, cu, lw=3, color="forestgreen")
    axes[1, 0].set_title("Cumulative Error", fontweight="bold")
    axes[1, 0].set_xlabel("Rel L2")
    axes[1, 0].grid(True, alpha=0.3)

    es = (
        f"\nError Statistics:\n\nMean: {mean_error:.6f}\nStd: {std_error:.6f}\n"
        f"Min: {np.min(per_sample_errors):.6f}\nMax: {np.max(per_sample_errors):.6f}\n"
        f"\n95th Pct: {np.percentile(per_sample_errors, 95):.6f}\n"
    )
    axes[1, 1].text(
        0.1,
        0.5,
        es,
        fontsize=12,
        transform=axes[1, 1].transAxes,
        va="center",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgreen", "alpha": 0.7},
    )
    axes[1, 1].set_title("Statistics", fontweight="bold")
    axes[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print()
    print("=" * 70)
    print(f"Full U-FNO Navier-Stokes turbulence example completed in {training_time:.1f}s")
    print(f"Mean Relative L2 Error: {mean_error:.6f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return {
        "l2_relative_error": mean_error,
        "test_mse": test_mse,
        "final_loss": final_train_loss,
        "param_count": int(param_count),
    }


# %% [markdown]
"""
## Results Summary + Next Steps

After running this example you should observe:
- Decreasing training loss with physics-aware energy conservation
- Multi-scale frequency analysis of the velocity-magnitude field showing U-FNO
  captures Navier-Stokes turbulence structure
- Full error statistics across the test set for the initial -> final velocity map

**Next steps:**
- Increase resolution and training epochs for better convergence
- Experiment with stronger energy conservation loss weights
- Try different viscosity ranges for more/less turbulent regimes
- Compare U-FNO with standard FNO on the same Navier-Stokes data
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
