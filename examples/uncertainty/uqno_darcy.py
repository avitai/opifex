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
# UQNO on Darcy Flow — Conformal Prediction Bands

| Property      | Value                                    |
|---------------|------------------------------------------|
| Level         | Intermediate                             |
| Runtime       | ~3 min (GPU) / ~15 min (CPU)             |
| Memory        | ~1 GB                                    |
| Prerequisites | JAX, Flax NNX, Conformal Prediction      |

## Overview

Train an Uncertainty Quantification Neural Operator (UQNO) on the Darcy
flow equation, then *conformally calibrate* it to produce prediction
intervals with finite-sample coverage guarantees. The opifex
implementation is a JAX-native port of the conformal three-stage UQNO
recipe from Ma, Pitt, Azizzadenesheli, Anandkumar (TMLR 2024,
[arXiv:2402.01960](https://arxiv.org/abs/2402.01960)); the canonical
PyTorch reference lives at
[``neuraloperator/neuralop/models/uqno.py``](https://github.com/neuraloperator/neuraloperator).
The numerical core — ``PointwiseQuantileLoss``,
``get_coeff_quantile_idx``, the scaling-factor derivation — is
cross-checked test-by-test against that reference; opifex-side
ergonomics layer on top (typed `PredictiveDistribution` /
`PredictionInterval` returns, explicit `base=` / `residual=`
constructor, in-class `.calibrate(...)`).

**Stages:**
1. **Base solution operator** — a deterministic FNO trained to predict
   the Darcy pressure field via the relative-L2 loss on
   `(input, target)` pairs. This is the standard operator-learning
   objective and reaches a low relative-L2 error.
2. **Residual quantile operator** — a *second* deterministic FNO
   trained against ``PointwiseQuantileLoss`` to predict per-grid-point
   quantile widths of the base operator's residual.
3. **Scalar conformal calibration** — on a held-out calibration set,
   the ratios ``|y - base(x)| / residual(x)`` are reduced to a single
   ``uncertainty_scaling_factor`` via the canonical
   ``get_coeff_quantile_idx`` rule. Test-time bands are
   ``base(x) ± residual(x) * scaling_factor``.

To reach both **good accuracy** and **meaningful calibrated
uncertainty**, the example uses the proven operator-learning recipe:
Gaussian input/output normalization (fit on the training split), the
relative-L2 loss for the base, grid positional embedding, and enough
data and epochs for the spectral weights to converge. The predicted
mean is un-normalized back to physical pressure before its relative-L2
error and the conformal coverage are reported.

Conformal prediction gives **distribution-free** finite-sample coverage
guarantees: the bands cover the true target on at least ``1 - alpha``
fraction of points (per the chosen ``(alpha, delta)`` configuration),
regardless of how well the base / residual operators fit the data.
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
from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.specialized.uqno import (
    UncertaintyQuantificationNeuralOperator,
    UQNOBaseSolutionOperator,
    UQNOResidualOperator,
)
from opifex.uncertainty.losses import PointwiseQuantileLoss


def _find_repo_root() -> Path:
    """Walk up from cwd until we find pyproject.toml — works in scripts AND notebooks."""
    here = Path.cwd().resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return here


_REPO_ROOT = _find_repo_root()
OUTPUT_DIR = _REPO_ROOT / "docs" / "assets" / "examples" / "uqno_darcy"


# %% [markdown]
"""
## Configuration

We follow the standard operator-learning recipe: ~1000 base training
samples, Gaussian normalization, the relative-L2 loss for the base
operator, and enough epochs for the spectral weights to converge. The
residual / calibration / test splits are large enough for stable
conformal calibration.
"""

# %%
RESOLUTION = 64
N_TRAIN_BASE = 1000  # Base solution operator training samples
N_TRAIN_RESIDUAL = 500  # Residual quantile operator training samples
N_CALIB = 500  # Held-out calibration samples for the scalar scaling factor
N_TEST = 100
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64  # Forward-pass chunk size for batched evaluation

# Both operators share the canonical FNO sizing (Li et al.).
MODES = 12
HIDDEN_CHANNELS = 32
NUM_LAYERS = 4

# Training schedules
BASE_EPOCHS = 120
RESIDUAL_EPOCHS = 80
LEARNING_RATE = 1e-3

# Conformal target: 1 - alpha pointwise coverage; delta is the
# function-level miscoverage budget.
ALPHA = 0.1
DELTA = 0.1

SEED = 42


# %% [markdown]
r"""
## Load Darcy Flow Data

The Darcy flow equation is an elliptic PDE
$-\nabla \cdot (a(x) \nabla u(x)) = f(x)$ where $a(x)$ is the
permeability coefficient and $u(x)$ is the pressure. The synthetic
opifex loader yields `(input, output)` pairs at the configured
resolution. We disable the loader's pass-through normalization and fit
proper Gaussian statistics on the base-train split below.
"""

# %% [markdown]
"""
## Normalization

Neural operators train best on standardized fields. We fit Gaussian
statistics on the base-train split, normalize every split, and
un-normalize the predicted **mean** (and scale the predicted band
widths) before computing physical-space errors and coverage. The
base + residual operators therefore train and calibrate entirely in
normalized space; only the reported metrics live in physical units.
"""


# %% [markdown]
"""
## Stage 1 — Train the Base Solution Operator

A standard deterministic FNO trained with the **relative-L2 loss** — the
canonical operator-learning objective — on normalized `(input, target)`
pairs. ``positional_embedding=True`` appends normalized grid coordinates
as extra input channels, the standard positional encoding that lets the
spectral operator resolve the boundary-value problem. This is the
operator we will later wrap with conformal bands.
"""


# %%
def _relative_l2(pred: jax.Array, target: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Mean per-sample relative-L2 error between ``pred`` and ``target``."""
    diff = (pred - target).reshape(pred.shape[0], -1)
    ref = target.reshape(target.shape[0], -1)
    return jnp.mean(jnp.linalg.norm(diff, axis=1) / (jnp.linalg.norm(ref, axis=1) + eps))


@nnx.jit
def base_train_step(
    model: FourierNeuralOperator,
    opt: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Relative-L2 training step for the base FNO."""

    def loss_fn(m: FourierNeuralOperator) -> jax.Array:
        return _relative_l2(m(x), y)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


def _shuffle_batches(
    x: jax.Array, y: jax.Array, *, batch_size: int, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Shuffle a dataset and chunk into ``batch_size``-sized batches."""
    n = x.shape[0]
    perm = jax.random.permutation(key, n)
    x = x[perm]
    y = y[perm]
    n_full = (n // batch_size) * batch_size
    x = x[:n_full].reshape(n_full // batch_size, batch_size, *x.shape[1:])
    y = y[:n_full].reshape(n_full // batch_size, batch_size, *y.shape[1:])
    return x, y


# %% [markdown]
"""
## Stage 2 — Train the Residual Quantile Operator

A *separate* FNO trained against
:class:`opifex.uncertainty.losses.PointwiseQuantileLoss` on the
residuals ``base(x) - y_true`` from the (frozen) base operator, in
normalized space. The target is for the residual operator to predict
per-grid-point quantile widths consistent with the residual
distribution.
"""


# %%
@nnx.jit
def residual_train_step(
    base: FourierNeuralOperator,
    residual: FourierNeuralOperator,
    opt: nnx.Optimizer,
    quantile_loss: PointwiseQuantileLoss,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Quantile-loss training step for the residual operator.

    The base operator's output is wrapped in ``jax.lax.stop_gradient``
    so its weights stay frozen for the residual stage (matches the
    canonical reference's ``no_grad`` + ``eval`` pattern).
    """

    def loss_fn(r: FourierNeuralOperator) -> jax.Array:
        base_pred = jax.lax.stop_gradient(base(x))
        quantile_widths = jnp.abs(r(x))
        return quantile_loss(y_pred=quantile_widths, y=base_pred - y)

    loss, grads = nnx.value_and_grad(loss_fn)(residual)
    opt.update(residual, grads)
    return loss


# %% [markdown]
"""
## Stage 3 — Conformal Calibration + Test-time Bands

Wrap the trained base + residual operators in
``UncertaintyQuantificationNeuralOperator``, then derive a scalar
``uncertainty_scaling_factor`` from the held-out calibration ratios
``|y - base(x)| / (residual(x) + eps)`` via the canonical
``get_coeff_quantile_idx`` rule. The fitted calibrator is attached to
the model and used by ``predict_with_bands``. Calibration is performed
in normalized space; predicted means and band widths are un-normalized
to physical pressure before coverage is measured.
"""

# %% [markdown]
"""
## Evaluation — Predictive Mean Accuracy + Calibrated Coverage

The predicted mean is un-normalized back to physical pressure and its
relative-L2 error is reported per sample. The conformal band widths are
scaled by ``y_std`` and the empirical coverage is measured against the
physical targets. The test set is run through the operators in chunks
to bound memory at resolution 64.
"""


# %%
def predict_bands_in_batches(
    operator: UncertaintyQuantificationNeuralOperator,
    inputs_n: jax.Array,
    batch_size: int,
    y_mean: float,
    y_std: float,
) -> tuple[jax.Array, jax.Array]:
    """Run ``predict_with_bands`` over normalized inputs in memory-bounded chunks.

    Returns the un-normalized physical predictive mean and the
    un-normalized (one-sided) band half-width for each chunk,
    concatenated along the batch axis.
    """
    means: list[jax.Array] = []
    half_widths: list[jax.Array] = []
    for i in range(0, inputs_n.shape[0], batch_size):
        chunk = inputs_n[i : i + batch_size]
        dist = operator.predict_with_bands(chunk)
        chunk_interval = dist.interval
        if chunk_interval is None:
            raise RuntimeError("predict_with_bands returned no interval; check calibration.")
        means.append(dist.mean * y_std + y_mean)
        half_widths.append((chunk_interval.upper - chunk_interval.lower) * 0.5 * y_std)
    return jnp.concatenate(means, axis=0), jnp.concatenate(half_widths, axis=0)


# %% [markdown]
"""
## Visualisation

The plot below shows, for one representative test sample:

* Input permeability $a(x, y)$
* Target pressure $u(x, y)$
* Base operator prediction (un-normalized)
* Absolute error of the predictive mean
* Calibrated band width (uncertainty heat-map)
* Per-pixel in-band mask
"""

# %% [markdown]
"""
## Training Analysis

Left: per-stage training losses. Centre: the calibrated band-width
distribution. Right: absolute error versus predicted uncertainty — a
positive trend means the operator is *more uncertain where it is less
accurate*, the qualitative signature of a sensible uncertainty surface.
"""


# %% [markdown]
"""
## Run the example

`main()` runs the full three-stage UQNO pipeline — data loading,
normalization, base + residual training, conformal calibration,
evaluation, and visualization — and returns a small dictionary of finite
scalar metrics. Nothing heavy runs at import time.
"""


# %%
def main() -> dict[str, float | int]:
    """Run the UQNO Darcy conformal pipeline and return summary metrics."""
    print("=" * 70)
    print("Opifex Example: UQNO on Darcy Flow (Conformal Prediction Bands)")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Configuration:")
    print(f"  Resolution:    {RESOLUTION}x{RESOLUTION}")
    print(f"  Base train:    {N_TRAIN_BASE}, Residual train: {N_TRAIN_RESIDUAL}")
    print(f"  Calibration:   {N_CALIB}, Test: {N_TEST}")
    print(f"  FNO:           modes={MODES}, hidden={HIDDEN_CHANNELS}, layers={NUM_LAYERS}")
    print(f"  Base epochs:   {BASE_EPOCHS}, Residual epochs: {RESIDUAL_EPOCHS}")
    print(f"  Conformal:     alpha={ALPHA}, delta={DELTA}")

    # --- Load Darcy flow data ---------------------------------------------
    print()
    print("Loading Darcy flow data...")
    n_total = N_TRAIN_BASE + N_TRAIN_RESIDUAL + N_CALIB + N_TEST

    # datarax generates ``n_total`` samples and serves them via train/val
    # pipelines; both are drained for a single contiguous block of samples.
    # Batches are already channels-first ``(batch, 1, H, W)``.
    loaders = create_darcy_loader(
        n_samples=n_total,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        seed=SEED,
    )

    print("  Materialising dataset into memory...")
    inputs_all: list[jax.Array] = []
    outputs_all: list[jax.Array] = []
    for pipeline in (loaders.train, loaders.val):
        for batch in pipeline:
            inputs_all.append(jnp.asarray(np.asarray(batch["input"])))
            outputs_all.append(jnp.asarray(np.asarray(batch["output"])))

    inputs = jnp.concatenate(inputs_all, axis=0)[:n_total]
    outputs = jnp.concatenate(outputs_all, axis=0)[:n_total]
    print(f"  Materialised {inputs.shape[0]} samples (input shape {inputs.shape[1:]})")

    splits = [N_TRAIN_BASE, N_TRAIN_RESIDUAL, N_CALIB]
    cuts = [sum(splits[: i + 1]) for i in range(len(splits))]
    x_base, x_residual, x_calib, x_test = jnp.split(inputs, cuts, axis=0)
    y_base, y_residual, y_calib, y_test = jnp.split(outputs, cuts, axis=0)
    print(
        f"  Splits: base={x_base.shape[0]}, residual={x_residual.shape[0]}, "
        f"calib={x_calib.shape[0]}, test={x_test.shape[0]}"
    )

    # --- Normalization ----------------------------------------------------
    x_mean = float(jnp.mean(x_base))
    x_std = float(jnp.std(x_base))
    y_mean = float(jnp.mean(y_base))
    y_std = float(jnp.std(y_base))

    x_base_n = (x_base - x_mean) / x_std
    x_residual_n = (x_residual - x_mean) / x_std
    x_calib_n = (x_calib - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std

    y_base_n = (y_base - y_mean) / y_std
    y_residual_n = (y_residual - y_mean) / y_std
    y_calib_n = (y_calib - y_mean) / y_std

    print(f"Input mean/std:  {x_mean:.4f} / {x_std:.4f}")
    print(f"Output mean/std: {y_mean:.6f} / {y_std:.6f}")

    # --- Stage 1: base solution operator ----------------------------------
    print()
    print("Stage 1: training base solution operator (relative-L2)...")
    base_fno = FourierNeuralOperator(
        in_channels=1,
        out_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        modes=MODES,
        num_layers=NUM_LAYERS,
        positional_embedding=True,
        rngs=nnx.Rngs(SEED),
    )
    base_opt = nnx.Optimizer(base_fno, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    base_param_count = sum(
        p.size for p in jax.tree_util.tree_leaves(nnx.state(base_fno, nnx.Param))
    )

    start = time.time()
    base_losses: list[float] = []
    for epoch in range(BASE_EPOCHS):
        epoch_key = jax.random.fold_in(jax.random.PRNGKey(SEED), epoch)
        xb, yb = _shuffle_batches(x_base_n, y_base_n, batch_size=BATCH_SIZE, key=epoch_key)
        epoch_loss = 0.0
        for i in range(xb.shape[0]):
            loss = base_train_step(base_fno, base_opt, xb[i], yb[i])
            epoch_loss += float(loss)
        epoch_loss /= xb.shape[0]
        base_losses.append(epoch_loss)
        if epoch % 10 == 0 or epoch == BASE_EPOCHS - 1:
            print(f"  Base epoch {epoch + 1:3d}/{BASE_EPOCHS}: rel-L2 = {epoch_loss:.6f}")
    print(f"Base training time: {time.time() - start:.1f}s")

    # --- Stage 2: residual quantile operator ------------------------------
    print()
    print("Stage 2: training residual quantile operator (PointwiseQuantileLoss)...")
    residual_fno = FourierNeuralOperator(
        in_channels=1,
        out_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        modes=MODES,
        num_layers=NUM_LAYERS,
        positional_embedding=True,
        rngs=nnx.Rngs(SEED + 1),
    )
    residual_opt = nnx.Optimizer(residual_fno, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    quantile_loss = PointwiseQuantileLoss(alpha=ALPHA, reduction="mean")

    start = time.time()
    residual_losses: list[float] = []
    for epoch in range(RESIDUAL_EPOCHS):
        epoch_key = jax.random.fold_in(jax.random.PRNGKey(SEED + 1), epoch)
        xb, yb = _shuffle_batches(x_residual_n, y_residual_n, batch_size=BATCH_SIZE, key=epoch_key)
        epoch_loss = 0.0
        for i in range(xb.shape[0]):
            loss = residual_train_step(
                base_fno, residual_fno, residual_opt, quantile_loss, xb[i], yb[i]
            )
            epoch_loss += float(loss)
        epoch_loss /= xb.shape[0]
        residual_losses.append(epoch_loss)
        if epoch % 10 == 0 or epoch == RESIDUAL_EPOCHS - 1:
            print(f"  Residual epoch {epoch + 1:3d}/{RESIDUAL_EPOCHS}: quantile = {epoch_loss:.6f}")
    print(f"Residual training time: {time.time() - start:.1f}s")

    # --- Stage 3: conformal calibration -----------------------------------
    print()
    print("Stage 3: conformal calibration...")
    uqno = UncertaintyQuantificationNeuralOperator(
        base=UQNOBaseSolutionOperator(base_fno),
        residual=UQNOResidualOperator(residual_fno),
    )
    calibrator = uqno.calibrate(x_calib_n, y_calib_n, alpha=ALPHA, delta=DELTA)
    uqno = uqno.with_calibrator(calibrator)
    print(f"  domain_idx = {calibrator.domain_idx}")
    print(f"  function_idx = {calibrator.function_idx}")
    print(f"  scaling factor = {float(calibrator.scaling_factor):.6f}")

    # --- Evaluation -------------------------------------------------------
    print()
    print("Evaluating predictive mean + coverage on held-out test set...")
    pred_mean, half_width = predict_bands_in_batches(uqno, x_test_n, EVAL_BATCH_SIZE, y_mean, y_std)
    lower = pred_mean - half_width
    upper = pred_mean + half_width

    pred_diff = (pred_mean - y_test).reshape(pred_mean.shape[0], -1)
    y_flat = y_test.reshape(y_test.shape[0], -1)
    per_sample_rel_l2 = jnp.linalg.norm(pred_diff, axis=1) / jnp.linalg.norm(y_flat, axis=1)
    mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))
    min_rel_l2 = float(jnp.min(per_sample_rel_l2))
    max_rel_l2 = float(jnp.max(per_sample_rel_l2))

    in_band = (y_test >= lower) & (y_test <= upper)
    pointwise_coverage = float(jnp.mean(in_band))
    mean_width = float(jnp.mean(upper - lower))

    abs_error = jnp.abs(pred_mean - y_test)
    pred_std = half_width  # one-sided band half-width as a positive uncertainty surface
    std_positive = bool(jnp.all(pred_std >= 0.0))
    high_unc_mask = pred_std >= jnp.median(pred_std)
    mean_err_high_unc = float(jnp.mean(abs_error[high_unc_mask]))
    mean_err_low_unc = float(jnp.mean(abs_error[~high_unc_mask]))
    error_unc_corr = float(jnp.corrcoef(abs_error.reshape(-1), pred_std.reshape(-1))[0, 1])

    print(
        f"  Predictive-mean test rel-L2  = {mean_rel_l2:.6f} "
        f"(min {min_rel_l2:.6f}, max {max_rel_l2:.6f})"
    )
    print(f"  Target coverage (1 - alpha)  = {1 - ALPHA:.3f}")
    print(f"  Empirical pointwise coverage = {pointwise_coverage:.3f}")
    print(f"  Mean band width              = {mean_width:.6f}")
    print(f"  Predicted std all positive   = {std_positive}")
    print(f"  |error| vs predicted-std corr= {error_unc_corr:.3f}")
    print(f"  Mean |error| (high unc / low) = {mean_err_high_unc:.6f} / {mean_err_low_unc:.6f}")

    # --- Visualisation ----------------------------------------------------
    print()
    print("Creating visualisations...")
    sample_idx = 0
    target = np.asarray(y_test[sample_idx, 0])
    prediction = np.asarray(pred_mean[sample_idx, 0])
    lower_s = np.asarray(lower[sample_idx, 0])
    upper_s = np.asarray(upper[sample_idx, 0])
    width = upper_s - lower_s
    error = np.abs(prediction - target)
    input_perm = np.asarray(x_test[sample_idx, 0])
    in_band_sample = (target >= lower_s) & (target <= upper_s)

    _fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    im = ax.imshow(input_perm, cmap="viridis")
    ax.set_title("Input (Permeability)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(target, cmap="RdBu_r")
    ax.set_title("Target (Pressure)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    im = ax.imshow(prediction, cmap="RdBu_r")
    ax.set_title(f"Base Prediction (rel-L2 = {float(per_sample_rel_l2[sample_idx]):.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    im = ax.imshow(error, cmap="Reds")
    ax.set_title("Absolute Error")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(width, cmap="Oranges")
    ax.set_title("Band Width (calibrated)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    im = ax.imshow(in_band_sample.astype(float), cmap="Greens", vmin=0.0, vmax=1.0)
    ax.set_title(f"In-band mask (coverage = {float(np.mean(in_band_sample)):.2f})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(
        f"UQNO on Darcy Flow — Conformal Bands (alpha={ALPHA}, delta={DELTA}, "
        f"rel-L2={mean_rel_l2:.3f}, coverage={pointwise_coverage:.3f})",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "solution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'solution.png'}")

    # --- Training analysis ------------------------------------------------
    _fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    ax = axes[0]
    ax.plot(base_losses, label="Base rel-L2")
    ax.plot(residual_losses, label="Residual quantile", color="C3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training Loss (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    band_widths = np.asarray(upper - lower).flatten()
    ax.hist(band_widths, bins=50, color="C1", alpha=0.7)
    ax.set_xlabel("Band width (per pixel)")
    ax.set_ylabel("Count")
    ax.set_title(f"Calibrated band-width distribution (target coverage {1 - ALPHA:.2f})")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    err_flat = np.asarray(abs_error).reshape(-1)
    std_flat = np.asarray(pred_std).reshape(-1)
    sub = np.linspace(0, err_flat.shape[0] - 1, min(5000, err_flat.shape[0]), dtype=int)
    ax.scatter(std_flat[sub], err_flat[sub], s=3, alpha=0.2, color="C2")
    ax.set_xlabel("Predicted uncertainty (band half-width)")
    ax.set_ylabel("Absolute error")
    ax.set_title(f"Error vs uncertainty (corr = {error_unc_corr:.2f})")
    ax.grid(True, alpha=0.3)

    plt.suptitle("UQNO Training + Calibration Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'analysis.png'}")

    return {
        "rel_l2": mean_rel_l2,
        "coverage": pointwise_coverage,
        "param_count": int(base_param_count),
    }


# %% [markdown]
"""
## Summary + Next Steps

UQNO produces an accurate predictive mean *and* distribution-free
coverage bands via three stages:

1. Train a base FNO with the relative-L2 operator-learning loss on
   Gaussian-normalized data (grid positional embedding included).
2. Train a residual FNO with ``PointwiseQuantileLoss`` on the
   residuals of the (frozen) base.
3. Compute a single scalar ``uncertainty_scaling_factor`` from
   per-grid ratios on a held-out calibration set; bands at test time
   are ``base(x) ± residual(x) * scaling_factor``.

The predictive mean reaches a low relative-L2 error in physical units,
and the empirical coverage proportion on the held-out test set lands
near the target ``1 - alpha = {1 - ALPHA:.2f}`` (the exact target
depends jointly on ``alpha`` and ``delta`` via the canonical
``get_coeff_quantile_idx`` rule). The predicted uncertainty is positive
everywhere and is larger where the error is larger.

For even higher accuracy, scale up the per-stage epoch counts or the
hidden channels / Fourier modes; the canonical Li-style FNO Darcy setup
uses ~1000 / ~500 / ~500 samples with several hundred epochs per stage.
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
