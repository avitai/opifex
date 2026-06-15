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
   the Darcy pressure field via MSE on `(input, target)` pairs.
2. **Residual quantile operator** — a *second* deterministic FNO
   trained against ``PointwiseQuantileLoss`` to predict per-grid-point
   quantile widths of the base operator's residual.
3. **Scalar conformal calibration** — on a held-out calibration set,
   the ratios ``|y - base(x)| / residual(x)`` are reduced to a single
   ``uncertainty_scaling_factor`` via the canonical
   ``get_coeff_quantile_idx`` rule. Test-time bands are
   ``base(x) ± residual(x) * scaling_factor``.

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


print("=" * 70)
print("Opifex Example: UQNO on Darcy Flow (Conformal Prediction Bands)")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")


def _find_repo_root() -> Path:
    """Walk up from cwd until we find pyproject.toml — works in scripts AND notebooks."""
    here = Path.cwd().resolve()
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return here


_REPO_ROOT = _find_repo_root()
OUTPUT_DIR = _REPO_ROOT / "docs" / "assets" / "examples" / "uqno_darcy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
"""
## Configuration
"""

# %%
RESOLUTION = 64
N_TRAIN_BASE = 200  # Base solution operator training samples
N_TRAIN_RESIDUAL = 100  # Residual quantile operator training samples
N_CALIB = 80  # Held-out calibration samples for the scalar scaling factor
N_TEST = 40
BATCH_SIZE = 8

# Both operators share the canonical FNO sizing (Li et al.).
MODES = 12
HIDDEN_CHANNELS = 32
NUM_LAYERS = 4

# Training schedules
BASE_EPOCHS = 30
RESIDUAL_EPOCHS = 20
LEARNING_RATE = 1e-3

# Conformal target: 1 - alpha pointwise coverage; delta is the
# function-level miscoverage budget.
ALPHA = 0.1
DELTA = 0.1

SEED = 42

print()
print("Configuration:")
print(f"  Resolution:    {RESOLUTION}x{RESOLUTION}")
print(f"  Base train:    {N_TRAIN_BASE}, Residual train: {N_TRAIN_RESIDUAL}")
print(f"  Calibration:   {N_CALIB}, Test: {N_TEST}")
print(f"  FNO:           modes={MODES}, hidden={HIDDEN_CHANNELS}, layers={NUM_LAYERS}")
print(f"  Base epochs:   {BASE_EPOCHS}, Residual epochs: {RESIDUAL_EPOCHS}")
print(f"  Conformal:     alpha={ALPHA}, delta={DELTA}")


# %% [markdown]
r"""
## Load Darcy Flow Data

The Darcy flow equation is an elliptic PDE
$-\nabla \cdot (a(x) \nabla u(x)) = f(x)$ where $a(x)$ is the
permeability coefficient and $u(x)$ is the pressure. The synthetic
opifex loader yields `(input, output)` pairs at the configured
resolution.
"""

# %%
print()
print("Loading Darcy flow data...")

n_total = N_TRAIN_BASE + N_TRAIN_RESIDUAL + N_CALIB + N_TEST

train_loader = create_darcy_loader(
    n_samples=n_total,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=True,
    seed=SEED,
    worker_count=0,
    enable_normalization=True,
    num_epochs=max(BASE_EPOCHS, RESIDUAL_EPOCHS) + 4,
)

# Collect everything once into JAX arrays so we can re-iterate cheaply.
print("  Materialising dataset into memory...")
inputs_all: list[jax.Array] = []
outputs_all: list[jax.Array] = []
seen = 0
for batch in train_loader:
    x_batch = jnp.asarray(np.asarray(batch["input"]))
    y_batch = jnp.asarray(np.asarray(batch["output"]))
    if x_batch.ndim == 3:
        x_batch = x_batch[:, None, ...]
        y_batch = y_batch[:, None, ...]
    inputs_all.append(x_batch)
    outputs_all.append(y_batch)
    seen += x_batch.shape[0]
    if seen >= n_total:
        break

inputs = jnp.concatenate(inputs_all, axis=0)[:n_total]
outputs = jnp.concatenate(outputs_all, axis=0)[:n_total]
print(f"  Materialised {inputs.shape[0]} samples (input shape {inputs.shape[1:]})")

# Split into base-train / residual-train / calib / test partitions.
splits = [N_TRAIN_BASE, N_TRAIN_RESIDUAL, N_CALIB]
cuts = [sum(splits[: i + 1]) for i in range(len(splits))]
x_base, x_residual, x_calib, x_test = jnp.split(inputs, cuts, axis=0)
y_base, y_residual, y_calib, y_test = jnp.split(outputs, cuts, axis=0)
print(
    f"  Splits: base={x_base.shape[0]}, residual={x_residual.shape[0]}, "
    f"calib={x_calib.shape[0]}, test={x_test.shape[0]}"
)


# %% [markdown]
"""
## Stage 1 — Train the Base Solution Operator

A standard deterministic FNO trained with MSE loss on `(input, target)`.
This is the operator we will later wrap with conformal bands.
"""

# %%
print()
print("Stage 1: training base solution operator (MSE)...")

base_fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=HIDDEN_CHANNELS,
    modes=MODES,
    num_layers=NUM_LAYERS,
    rngs=nnx.Rngs(SEED),
)
base_opt = nnx.Optimizer(base_fno, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def base_train_step(
    model: FourierNeuralOperator,
    opt: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Plain MSE training step for the base FNO."""

    def loss_fn(m: FourierNeuralOperator) -> jax.Array:
        return jnp.mean((m(x) - y) ** 2)

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


start = time.time()
base_losses: list[float] = []
for epoch in range(BASE_EPOCHS):
    epoch_key = jax.random.fold_in(jax.random.PRNGKey(SEED), epoch)
    xb, yb = _shuffle_batches(x_base, y_base, batch_size=BATCH_SIZE, key=epoch_key)
    epoch_loss = 0.0
    for i in range(xb.shape[0]):
        loss = base_train_step(base_fno, base_opt, xb[i], yb[i])
        epoch_loss += float(loss)
    epoch_loss /= xb.shape[0]
    base_losses.append(epoch_loss)
    if epoch % 5 == 0 or epoch == BASE_EPOCHS - 1:
        print(f"  Base epoch {epoch + 1:3d}/{BASE_EPOCHS}: MSE = {epoch_loss:.6f}")
print(f"Base training time: {time.time() - start:.1f}s")


# %% [markdown]
"""
## Stage 2 — Train the Residual Quantile Operator

A *separate* FNO trained against
:class:`opifex.uncertainty.losses.PointwiseQuantileLoss` on the
residuals ``base(x) - y_true`` from the (frozen) base operator. The
target is for the residual operator to predict per-grid-point quantile
widths consistent with the residual distribution.
"""

# %%
print()
print("Stage 2: training residual quantile operator (PointwiseQuantileLoss)...")

residual_fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=HIDDEN_CHANNELS,
    modes=MODES,
    num_layers=NUM_LAYERS,
    rngs=nnx.Rngs(SEED + 1),
)
residual_opt = nnx.Optimizer(residual_fno, optax.adam(LEARNING_RATE), wrt=nnx.Param)
quantile_loss = PointwiseQuantileLoss(alpha=ALPHA, reduction="mean")


@nnx.jit
def residual_train_step(
    base: FourierNeuralOperator,
    residual: FourierNeuralOperator,
    opt: nnx.Optimizer,
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


start = time.time()
residual_losses: list[float] = []
for epoch in range(RESIDUAL_EPOCHS):
    epoch_key = jax.random.fold_in(jax.random.PRNGKey(SEED + 1), epoch)
    xb, yb = _shuffle_batches(x_residual, y_residual, batch_size=BATCH_SIZE, key=epoch_key)
    epoch_loss = 0.0
    for i in range(xb.shape[0]):
        loss = residual_train_step(base_fno, residual_fno, residual_opt, xb[i], yb[i])
        epoch_loss += float(loss)
    epoch_loss /= xb.shape[0]
    residual_losses.append(epoch_loss)
    if epoch % 5 == 0 or epoch == RESIDUAL_EPOCHS - 1:
        print(f"  Residual epoch {epoch + 1:3d}/{RESIDUAL_EPOCHS}: quantile = {epoch_loss:.6f}")
print(f"Residual training time: {time.time() - start:.1f}s")


# %% [markdown]
"""
## Stage 3 — Conformal Calibration + Test-time Bands

Wrap the trained base + residual operators in
``UncertaintyQuantificationNeuralOperator``, then derive a scalar
``uncertainty_scaling_factor`` from the held-out calibration ratios
``|y - base(x)| / (residual(x) + eps)`` via the canonical
``get_coeff_quantile_idx`` rule. The fitted calibrator is attached to
the model and used by ``predict_with_bands``.
"""

# %%
print()
print("Stage 3: conformal calibration...")

uqno = UncertaintyQuantificationNeuralOperator(
    base=UQNOBaseSolutionOperator(base_fno),
    residual=UQNOResidualOperator(residual_fno),
)
calibrator = uqno.calibrate(x_calib, y_calib, alpha=ALPHA, delta=DELTA)
uqno = uqno.with_calibrator(calibrator)
print(f"  domain_idx = {calibrator.domain_idx}")
print(f"  function_idx = {calibrator.function_idx}")
print(f"  scaling factor = {float(calibrator.scaling_factor):.6f}")

print()
print("Evaluating coverage on held-out test set...")
test_dist = uqno.predict_with_bands(x_test)
# predict_with_bands always returns a populated interval; fall back to
# a zero-volume interval if the field is unexpectedly None (defensive).
interval = test_dist.interval
if interval is None:
    raise RuntimeError("predict_with_bands returned no interval; check calibration.")
in_band = (y_test >= interval.lower) & (y_test <= interval.upper)
pointwise_coverage = float(jnp.mean(in_band))
mean_width = float(jnp.mean(interval.upper - interval.lower))
print(f"  Target coverage (1 - alpha) = {1 - ALPHA:.3f}")
print(f"  Empirical pointwise coverage = {pointwise_coverage:.3f}")
print(f"  Mean band width             = {mean_width:.6f}")


# %% [markdown]
"""
## Visualisation

The plot below shows, for one representative test sample:

* Input permeability $a(x, y)$
* Target pressure $u(x, y)$
* Base operator prediction
* Calibrated lower / upper bands
* Per-pixel band width (uncertainty heat-map)
"""

# %%
print()
print("Creating visualisations...")

sample_idx = 0
target = np.asarray(y_test[sample_idx, 0])
prediction = np.asarray(test_dist.mean[sample_idx, 0])
lower = np.asarray(interval.lower[sample_idx, 0])
upper = np.asarray(interval.upper[sample_idx, 0])
width = upper - lower
input_perm = np.asarray(x_test[sample_idx, 0])
in_band_sample = (target >= lower) & (target <= upper)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

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
ax.set_title("Base Prediction")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 0]
im = ax.imshow(lower, cmap="RdBu_r")
ax.set_title("Lower Band (calibrated)")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 1]
im = ax.imshow(upper, cmap="RdBu_r")
ax.set_title("Upper Band (calibrated)")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 2]
im = ax.imshow(width, cmap="Oranges")
ax.set_title(f"Band Width (sample coverage = {float(np.mean(in_band_sample)):.2f})")
plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle(
    f"UQNO on Darcy Flow — Conformal Bands (alpha={ALPHA}, delta={DELTA}, "
    f"empirical coverage={pointwise_coverage:.3f})",
    fontsize=13,
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "solution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR / 'solution.png'}")


# %% [markdown]
"""
## Training Analysis
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(base_losses, label="Base MSE")
ax.plot(residual_losses, label="Residual quantile", color="C3")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_yscale("log")
ax.set_title("Training Loss (log scale)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
band_widths = np.asarray(interval.upper - interval.lower).flatten()
ax.hist(band_widths, bins=50, color="C1", alpha=0.7)
ax.set_xlabel("Band width (per pixel)")
ax.set_ylabel("Density")
ax.set_title(f"Calibrated band-width distribution (target coverage {1 - ALPHA:.2f})")
ax.grid(True, alpha=0.3)

plt.suptitle("UQNO Training + Calibration Analysis", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR / 'analysis.png'}")


# %% [markdown]
"""
## Summary + Next Steps

UQNO produces *distribution-free* coverage bands via three stages:

1. Train a base FNO with any standard regression loss.
2. Train a residual FNO with ``PointwiseQuantileLoss`` on the
   residuals of the (frozen) base.
3. Compute a single scalar ``uncertainty_scaling_factor`` from
   per-grid ratios on a held-out calibration set; bands at test time
   are ``base(x) ± residual(x) * scaling_factor``.

The empirical coverage proportion on the held-out test set should land
near the target ``1 - alpha = {1 - ALPHA:.2f}`` (the exact target
depends jointly on ``alpha`` and ``delta`` via the canonical
``get_coeff_quantile_idx`` rule).

For higher accuracy, scale up ``N_TRAIN_BASE`` /
``N_TRAIN_RESIDUAL`` / ``N_CALIB`` and the per-stage epoch counts.
The canonical Li-style FNO Darcy setup uses ~1000 / ~500 / ~500 with
~300 epochs per stage.
"""
