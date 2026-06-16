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
# Bayesian FNO on Darcy Flow

| Property      | Value                                    |
|---------------|------------------------------------------|
| Level         | Intermediate                             |
| Runtime       | ~2 min (GPU) / ~10 min (CPU)             |
| Memory        | ~2 GB                                    |
| Prerequisites | JAX, Flax NNX, Deep Ensembles            |

## Overview

Quantify predictive uncertainty for the Darcy permeability-to-pressure
operator with a **heteroscedastic deep ensemble** of Fourier Neural
Operators. Each member is a
:class:`opifex.neural.operators.fno.probabilistic.ProbabilisticFourierNeuralOperator`
— an FNO backbone with twin pointwise heads that emit a per-location
``mean`` and ``log-variance``. Training each member by the
heteroscedastic-Gaussian negative log-likelihood gives the *aleatoric*
axis (input-dependent noise), and the disagreement *across* members gives
the *epistemic* axis (model uncertainty). Their sum is the total
predictive variance, the standard deep-ensemble decomposition.

This is the canonical scalable Bayesian-predictive recipe and it follows
the same accuracy template as the deterministic operator examples — grid
positional embedding, Gaussian input/output normalization, and enough
epochs for the spectral weights to converge — so the predictive *mean*
stays accurate while the predictive *spread* stays meaningful.

**Key Concepts:**
- **Deep Ensemble**: Independently-trained members; their spread is the
  epistemic (model) uncertainty (Lakshminarayanan et al. 2017).
- **Heteroscedastic Head**: Each member predicts a per-location variance,
  the aleatoric (data) uncertainty (Kendall & Gal 2017).
- **Variance Calibration**: A single scale fit on a held-out split
  aligns the predictive std with the observed error spread so the
  ~90% interval covers ~90% of the test residuals.

## Learning Goals

1. Build a heteroscedastic FNO with `ProbabilisticFourierNeuralOperator`
2. Train an ensemble with the heteroscedastic-Gaussian NLL loss
3. Decompose predictive variance into aleatoric + epistemic parts
4. Calibrate and visualize the predictive uncertainty
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
import optax
from flax import nnx

from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.fno._positional import append_grid_coordinates
from opifex.neural.operators.fno.probabilistic import (
    probabilistic_fno_negative_log_likelihood,
    ProbabilisticFourierNeuralOperator,
)
from opifex.uncertainty._predictive import ensemble_predictive


# %% [markdown]
"""
## Configuration

We follow the standard operator-learning recipe — ~1000 training samples,
Gaussian normalization, and enough epochs for the spectral weights to
converge — and add a held-out calibration split plus a small ensemble of
heteroscedastic members for the uncertainty estimate.
"""

# %%
# Data configuration
RESOLUTION = 64
N_TRAIN = 1000
N_TEST = 100
N_CALIBRATION = 100
BATCH_SIZE = 32
EVAL_CHUNK = 64  # Forward-pass chunk size to bound evaluation memory

# FNO model configuration
MODES = 12
HIDDEN_WIDTH = 32
NUM_LAYERS = 4

# Ensemble / uncertainty configuration
NUM_MEMBERS = 4  # Independently-trained ensemble members
TARGET_COVERAGE = 0.9  # Calibrate the 1.64-sigma interval to this coverage

# Training configuration
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3

SEED = 42

# Standard-normal quantile for the target coverage (0.9 -> 1.6449).
COVERAGE_Z = float(jax.scipy.stats.norm.ppf(0.5 + TARGET_COVERAGE / 2.0))


def _find_repo_root() -> Path:
    """Walk up from cwd until we find pyproject.toml — works in scripts AND notebooks."""
    here = Path.cwd().resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return here


_REPO_ROOT = _find_repo_root()
OUTPUT_DIR = _REPO_ROOT / "docs" / "assets" / "examples" / "bayesian_fno"

# %% [markdown]
"""
## Load Darcy Flow Data

`create_darcy_loader` generates the smooth Darcy permeability-to-pressure
dataset. We collect three disjoint splits: train (fits the models),
calibration (fits the variance scale), and test (reports the metrics).
"""


# %%
def _collect_split(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one Darcy split as channels-first ``(batch, 1, H, W)`` arrays.

    The datarax loader splits a generation into train/val pipelines; we want a
    single contiguous split, so both pipelines are drained and concatenated.
    Batches are already channels-first ``(batch, 1, H, W)``.
    """
    loaders = create_darcy_loader(
        n_samples=n_samples,
        batch_size=n_samples,
        resolution=RESOLUTION,
        seed=seed,
    )
    inputs, outputs = [], []
    for pipeline in (loaders.train, loaders.val):
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
    return (
        np.concatenate(inputs, axis=0)[:n_samples],
        np.concatenate(outputs, axis=0)[:n_samples],
    )


# %% [markdown]
"""
## Normalization

Neural operators train best on standardized fields. We fit Gaussian
statistics on the **training** set, normalize every split with them, and
un-normalize the predictive mean (and rescale the predictive std by
``y_std``) before computing physical-space errors.
"""

# %% [markdown]
"""
## Build the Heteroscedastic Ensemble

Each member is a `ProbabilisticFourierNeuralOperator`: an FNO backbone
plus a ``mean`` head and a ``log-variance`` head. The backbone is
translation-equivariant, so we prepend normalized grid-coordinate
channels to the input (the standard positional embedding for
boundary-value problems) — that is why each member is built with
``in_channels = 1 + 2``.
"""

# %%
DATA_CHANNELS = 1  # Darcy permeability / pressure are single-channel fields
GRID_CHANNELS = 2  # append_grid_coordinates adds one channel per spatial axis
IN_CHANNELS = DATA_CHANNELS + GRID_CHANNELS
OUT_CHANNELS = DATA_CHANNELS


def _make_member(seed: int) -> ProbabilisticFourierNeuralOperator:
    """Construct one ensemble member with its own initialization seed."""
    return ProbabilisticFourierNeuralOperator(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        hidden_channels=HIDDEN_WIDTH,
        modes=MODES,
        num_layers=NUM_LAYERS,
        rngs=nnx.Rngs(seed),
    )


# %% [markdown]
"""
## Training Loop

Every member is trained independently by the heteroscedastic-Gaussian
negative log-likelihood

```
- log N(y; mu(x), sigma^2(x))
```

(Kendall & Gal 2017, §3.1). The likelihood jointly fits the mean and the
per-location variance, so a single loss drives both accuracy and the
aleatoric uncertainty. Independent seeds and shuffles give the ensemble
its epistemic spread (Lakshminarayanan et al. 2017).
"""


# %%
@nnx.jit
def _train_step(
    member: ProbabilisticFourierNeuralOperator,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """One NLL gradient step on a grid-augmented batch."""

    def loss_fn(model: ProbabilisticFourierNeuralOperator) -> jax.Array:
        return probabilistic_fno_negative_log_likelihood(model, append_grid_coordinates(x), y)

    loss, grads = nnx.value_and_grad(loss_fn)(member)
    optimizer.update(member, grads)
    return loss


def _train_member(
    seed: int, x_train_n: jax.Array, y_train_n: jax.Array
) -> tuple[ProbabilisticFourierNeuralOperator, list[float]]:
    """Train one member and return it alongside its per-epoch NLL history."""
    member = _make_member(seed)
    optimizer = nnx.Optimizer(member, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    key = jax.random.key(seed)
    history: list[float] = []
    num_samples = x_train_n.shape[0]
    for _epoch in range(NUM_EPOCHS):
        key, shuffle_key = jax.random.split(key)
        permutation = jax.random.permutation(shuffle_key, num_samples)
        epoch_losses = []
        for start in range(0, num_samples, BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            loss = _train_step(member, optimizer, x_train_n[indices], y_train_n[indices])
            epoch_losses.append(float(loss))
        history.append(float(np.mean(epoch_losses)))
    return member, history


# %% [markdown]
"""
## Predictive Distribution

For a batch we collect each member's ``(mean, variance)`` in physical
units. The ensemble predictive mean is the average of the member means;
the predictive variance decomposes as

```
total = aleatoric + epistemic
      = mean_m[sigma_m^2(x)] + var_m[mu_m(x)]
```

so the std is ``sqrt(total)``. Members are evaluated in chunks to bound
memory at this resolution.
"""


# %%
def _member_moments(
    member: ProbabilisticFourierNeuralOperator,
    x_normalized: jax.Array,
    y_mean: float,
    y_std: float,
) -> tuple[jax.Array, jax.Array]:
    """Return one member's physical-unit ``(mean, variance)`` for a batch."""
    mean_n, log_variance_n = member(append_grid_coordinates(x_normalized))
    mean_phys = mean_n * y_std + y_mean
    variance_phys = jnp.exp(log_variance_n) * (y_std**2)
    return mean_phys, variance_phys


def _predict(
    x_normalized: jax.Array,
    members: list[ProbabilisticFourierNeuralOperator],
    y_mean: float,
    y_std: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Chunked ensemble prediction.

    Returns:
        Tuple ``(mean, std, aleatoric_std, epistemic_std)`` in physical
        units, each of shape ``(batch, C, H, W)``.
    """
    member_means_chunks: list[jax.Array] = []
    aleatoric_chunks: list[jax.Array] = []
    for start in range(0, x_normalized.shape[0], EVAL_CHUNK):
        x_chunk = x_normalized[start : start + EVAL_CHUNK]
        means_and_variances = [
            _member_moments(member, x_chunk, y_mean, y_std) for member in members
        ]
        member_means = jnp.stack([mean for mean, _ in means_and_variances], axis=0)
        member_variances = jnp.stack([variance for _, variance in means_and_variances], axis=0)
        member_means_chunks.append(member_means)
        aleatoric_chunks.append(jnp.mean(member_variances, axis=0))

    member_means_all = jnp.concatenate(member_means_chunks, axis=1)
    aleatoric_variance = jnp.concatenate(aleatoric_chunks, axis=0)

    predictive = ensemble_predictive(member_means_all, method="heteroscedastic_deep_ensemble")
    epistemic_variance = predictive.variance
    total_variance = aleatoric_variance + epistemic_variance
    return (
        predictive.mean,
        jnp.sqrt(total_variance + 1e-12),
        jnp.sqrt(aleatoric_variance + 1e-12),
        jnp.sqrt(epistemic_variance + 1e-12),
    )


# %% [markdown]
"""
## Variance Calibration

A raw deep ensemble is typically over-confident. We fit a single positive
scale on the calibration split so that the ``COVERAGE_Z * scale * std``
interval covers ``TARGET_COVERAGE`` of the calibration residuals, then
apply it unchanged at test time.
"""

# %% [markdown]
"""
## Run the example

`main()` performs the full pipeline — data loading, normalization,
ensemble training, variance calibration, evaluation, calibration
analysis, and visualization — and returns a small dictionary of finite
scalar metrics. Nothing heavy runs at import time.
"""


# %%
def main() -> dict[str, float | int]:
    """Run the Bayesian-FNO Darcy pipeline and return summary metrics."""
    print("=" * 70)
    print("Opifex Example: Bayesian FNO on Darcy Flow")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Configuration:")
    print(f"  Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"  Train / calibration / test samples: {N_TRAIN} / {N_CALIBRATION} / {N_TEST}")
    print(f"  FNO: modes={MODES}, width={HIDDEN_WIDTH}, layers={NUM_LAYERS}")
    print(
        f"  Ensemble: members={NUM_MEMBERS}, "
        f"target coverage={TARGET_COVERAGE:.0%} (z={COVERAGE_Z:.3f})"
    )
    print(f"  Training: epochs={NUM_EPOCHS}, lr={LEARNING_RATE}")

    # --- Load Darcy flow data ---------------------------------------------
    print()
    print("Loading Darcy flow data...")
    x_train, y_train = _collect_split(N_TRAIN, SEED)
    x_calibration, y_calibration = _collect_split(N_CALIBRATION, SEED + 1000)
    x_test, y_test = _collect_split(N_TEST, SEED + 2000)
    print(f"  Train input shape: {x_train.shape}")
    print(f"  Test input shape:  {x_test.shape}")

    # --- Normalization ----------------------------------------------------
    x_mean, x_std = float(x_train.mean()), float(x_train.std())
    y_mean, y_std = float(y_train.mean()), float(y_train.std())

    x_train_n = jnp.asarray((x_train - x_mean) / x_std)
    y_train_n = jnp.asarray((y_train - y_mean) / y_std)
    x_calibration_n = jnp.asarray((x_calibration - x_mean) / x_std)
    x_test_n = jnp.asarray((x_test - x_mean) / x_std)

    y_calibration_phys = jnp.asarray(y_calibration)
    y_test_phys = jnp.asarray(y_test)

    print(f"  Input mean/std:  {x_mean:.4f} / {x_std:.4f}")
    print(f"  Output mean/std: {y_mean:.6f} / {y_std:.6f}")

    # --- Build the heteroscedastic ensemble -------------------------------
    print()
    print("Creating heteroscedastic FNO ensemble...")
    probe = _make_member(SEED)
    member_params = sum(p.size for p in jax.tree.leaves(nnx.state(probe, nnx.Param)))
    print(f"  Member: ProbabilisticFNO (modes={MODES}, width={HIDDEN_WIDTH}, layers={NUM_LAYERS})")
    print(f"  Input channels: {DATA_CHANNELS} (+ {GRID_CHANNELS} grid = {IN_CHANNELS})")
    print(f"  Parameters per member: {member_params:,}")
    print(f"  Ensemble parameters:   {member_params * NUM_MEMBERS:,}")

    # --- Training loop ----------------------------------------------------
    print()
    print("Training ensemble members...")
    start_time = time.time()
    members: list[ProbabilisticFourierNeuralOperator] = []
    loss_histories: list[list[float]] = []
    for member_index in range(NUM_MEMBERS):
        trained_member, member_history = _train_member(SEED + member_index, x_train_n, y_train_n)
        members.append(trained_member)
        loss_histories.append(member_history)
        print(
            f"  Member {member_index + 1}/{NUM_MEMBERS}: "
            f"NLL {member_history[0]:+.4f} -> {member_history[-1]:+.4f}"
        )
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s")

    # --- Variance calibration ---------------------------------------------
    print()
    print("Calibrating predictive uncertainty...")
    calibration_mean, calibration_std, _, _ = _predict(x_calibration_n, members, y_mean, y_std)
    calibration_error = jnp.abs(calibration_mean - y_calibration_phys)
    calibration_ratio = (calibration_error / calibration_std).flatten()
    std_scale = float(jnp.quantile(calibration_ratio, TARGET_COVERAGE) / COVERAGE_Z)
    print(f"  Calibration std scale: {std_scale:.4f}")

    # --- Evaluation -------------------------------------------------------
    print()
    print("Evaluating on test set...")
    test_mean, test_std_raw, test_aleatoric_std, test_epistemic_std = _predict(
        x_test_n, members, y_mean, y_std
    )
    test_std = test_std_raw * std_scale
    test_error = jnp.abs(test_mean - y_test_phys)

    flat_diff = (test_mean - y_test_phys).reshape(test_mean.shape[0], -1)
    flat_target = y_test_phys.reshape(y_test_phys.shape[0], -1)
    per_sample_rel_l2 = jnp.linalg.norm(flat_diff, axis=1) / jnp.linalg.norm(flat_target, axis=1)
    mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))
    test_mse = float(jnp.mean((test_mean - y_test_phys) ** 2))

    print()
    print("Results:")
    print(f"  Predictive-mean Relative L2: {mean_rel_l2:.6f}")
    print(
        f"  Min / Max Relative L2:       {float(jnp.min(per_sample_rel_l2)):.6f}"
        f" / {float(jnp.max(per_sample_rel_l2)):.6f}"
    )
    print(f"  Test MSE:                    {test_mse:.6e}")
    print(f"  Mean predictive std:         {float(jnp.mean(test_std)):.6e}")
    print(f"  Mean aleatoric std:          {float(jnp.mean(test_aleatoric_std)):.6e}")
    print(f"  Mean epistemic std:          {float(jnp.mean(test_epistemic_std)):.6e}")

    # --- Calibration analysis ---------------------------------------------
    print()
    print("Uncertainty calibration analysis...")
    coverage_target = float(jnp.mean(test_error <= COVERAGE_Z * test_std))
    coverage_1sigma = float(jnp.mean(test_error <= test_std))
    coverage_2sigma = float(jnp.mean(test_error <= 2.0 * test_std))

    mean_error_per_sample = jnp.mean(test_error, axis=(1, 2, 3))
    mean_std_per_sample = jnp.mean(test_std, axis=(1, 2, 3))
    per_sample_correlation = float(jnp.corrcoef(mean_error_per_sample, mean_std_per_sample)[0, 1])
    per_pixel_correlation = float(jnp.corrcoef(test_error.flatten(), test_std.flatten())[0, 1])

    print(
        f"  Coverage @ {COVERAGE_Z:.2f}-sigma: {coverage_target * 100:.1f}% "
        f"(target {TARGET_COVERAGE:.0%})"
    )
    print(f"  1-sigma coverage:           {coverage_1sigma * 100:.1f}%")
    print(f"  2-sigma coverage:           {coverage_2sigma * 100:.1f}%")
    print(f"  Error-uncertainty corr (per-sample): {per_sample_correlation:.4f}")
    print(f"  Error-uncertainty corr (per-pixel):  {per_pixel_correlation:.4f}")

    # --- Visualization ----------------------------------------------------
    print()
    print("Creating visualizations...")
    sample_idx = int(jnp.argmax(per_sample_rel_l2))  # Show the hardest test sample

    _fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    ax = axes[0, 0]
    im = ax.imshow(np.asarray(x_test[sample_idx, 0, :, :]), cmap="viridis")
    ax.set_title("Input (Permeability)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(np.asarray(y_test_phys[sample_idx, 0, :, :]), cmap="RdBu_r")
    ax.set_title("Target (Pressure)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    im = ax.imshow(np.asarray(test_mean[sample_idx, 0, :, :]), cmap="RdBu_r")
    ax.set_title("Predictive Mean")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    error_map = np.asarray(test_error[sample_idx, 0, :, :])
    im = ax.imshow(error_map, cmap="hot")
    ax.set_title("Absolute Error")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(np.asarray(test_std[sample_idx, 0, :, :]), cmap="Oranges")
    ax.set_title("Predictive Std (calibrated)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    ax.scatter(
        np.asarray(mean_std_per_sample),
        np.asarray(mean_error_per_sample),
        alpha=0.7,
        c="steelblue",
        edgecolors="white",
        linewidths=0.5,
    )
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_val], [0, max_val], "r--", label="Error = std")
    ax.set_xlabel("Mean predictive std")
    ax.set_ylabel("Mean absolute error")
    ax.set_title(f"Calibration (r={per_sample_correlation:.2f})")
    ax.legend()

    plt.suptitle("Bayesian FNO on Darcy Flow", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "solution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'solution.png'}")

    # --- Training analysis ------------------------------------------------
    _fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for member_index, member_history in enumerate(loss_histories):
        ax.plot(member_history, label=f"Member {member_index + 1}", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("Per-Member Training NLL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    aleatoric_per_sample = jnp.mean(test_aleatoric_std, axis=(1, 2, 3))
    epistemic_per_sample = jnp.mean(test_epistemic_std, axis=(1, 2, 3))
    ax.scatter(
        np.asarray(aleatoric_per_sample),
        np.asarray(epistemic_per_sample),
        alpha=0.7,
        c=np.asarray(mean_error_per_sample),
        cmap="viridis",
    )
    ax.set_xlabel("Mean aleatoric std")
    ax.set_ylabel("Mean epistemic std")
    ax.set_title("Uncertainty Decomposition (color = error)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Bayesian FNO Training Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'analysis.png'}")

    return {
        "rel_l2": mean_rel_l2,
        "empirical_coverage": coverage_target,
        "mean_interval_width": float(jnp.mean(2.0 * COVERAGE_Z * test_std)),
        "param_count": int(member_params * NUM_MEMBERS),
    }


# %% [markdown]
"""
## Summary

| Metric                            | Value         |
|-----------------------------------|---------------|
| Predictive-mean Relative L2       | ~0.005        |
| Coverage @ 1.64-sigma             | ~90%          |
| Error-uncertainty corr (sample)   | ~0.6-0.7      |
| Parameters per member             | ~1.3 M        |
| Training time                     | ~1-2 min (GPU)|

**Key Observations:**
- The heteroscedastic deep ensemble keeps the operator-learning accuracy
  of a deterministic FNO (predictive-mean rel-L2 well under 0.08).
- The predictive variance is a genuine aleatoric + epistemic
  decomposition, calibrated to the target coverage on a held-out split.
- Uncertainty is larger where error is larger, both per pixel and per
  sample, so the predictive std is an actionable risk signal.
- Members are independent, so the ensemble trains and evaluates in
  parallel and scales to large FNO backbones.
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
