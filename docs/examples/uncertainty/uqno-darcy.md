# UQNO on Darcy Flow — Conformal Prediction Bands

| Property      | Value                                    |
|---------------|------------------------------------------|
| Level         | Intermediate                             |
| Runtime       | ~3 min (GPU) / ~15 min (CPU)             |
| Memory        | ~1 GB                                    |
| Prerequisites | JAX, Flax NNX, Conformal Prediction      |

## Overview

This example trains an Uncertainty Quantification Neural Operator
(UQNO) on the Darcy flow equation and produces **calibrated prediction
intervals** with finite-sample coverage guarantees. The implementation
follows the canonical three-stage recipe from Ma, Pitt, Azizzadenesheli,
Anandkumar (TMLR 2024,
[arXiv:2402.01960](https://arxiv.org/abs/2402.01960)):

1. **Base solution operator** — a standard deterministic
   ``FourierNeuralOperator`` trained on `(input, target)` pairs with
   MSE loss.
2. **Residual quantile operator** — a *separate* FNO trained against
   ``opifex.uncertainty.losses.PointwiseQuantileLoss`` on the
   residuals of the (frozen) base.
3. **Scalar conformal calibration** — on a held-out calibration set,
   ratios ``|y - base(x)| / residual(x)`` are reduced to a single
   ``uncertainty_scaling_factor`` via the canonical
   ``get_coeff_quantile_idx`` formula. Test-time bands are
   ``base(x) ± residual(x) * scaling_factor``.

Conformal prediction is **distribution-free**: the bands cover the
true target on at least ``1 - alpha`` fraction of points (per the
chosen ``(alpha, delta)`` configuration), independent of how well the
base / residual operators fit. There is no Bayesian posterior, no
Monte-Carlo sampling, and no KL term — the uncertainty surface lives
entirely in the residual operator + the scalar scaling factor.

## What You'll Learn

1. Compose ``UncertaintyQuantificationNeuralOperator`` from a base +
   residual ``FourierNeuralOperator``.
2. Train the residual operator with
   :class:`opifex.uncertainty.losses.PointwiseQuantileLoss`.
3. Calibrate a scalar uncertainty scaling factor with
   ``UncertaintyQuantificationNeuralOperator.calibrate``.
4. Predict + evaluate coverage with ``predict_with_bands``.

## Files

- **Python Script**: [`examples/uncertainty/uqno_darcy.py`](https://github.com/avitai/opifex/blob/main/examples/uncertainty/uqno_darcy.py)
- **Jupyter Notebook**: [`examples/uncertainty/uqno_darcy.ipynb`](https://github.com/avitai/opifex/blob/main/examples/uncertainty/uqno_darcy.ipynb)

## Quick Start

### Run the Python Script

```bash
source activate.sh && python examples/uncertainty/uqno_darcy.py
```

### Run the Jupyter Notebook

```bash
jupyter lab examples/uncertainty/uqno_darcy.ipynb
```

## Core Concepts

### Three-Stage Conformal UQNO

The canonical UQNO (the PyTorch reference at
[`neuraloperator/neuralop/models/uqno.py`](https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/models/uqno.py))
is **not** Bayesian — neither at the layer level nor
at the predictive-distribution level. It is a conformal-prediction
wrapper around a pair of deterministic FNOs:

| Stage | Object | Loss | Output |
|-------|--------|------|--------|
| 1 | `base: UQNOBaseSolutionOperator(FNO)` | MSE on `(x, y)` | $\hat{u}(x)$ |
| 2 | `residual: UQNOResidualOperator(FNO)` | `PointwiseQuantileLoss(alpha)` on `base(x) - y_true` | width$E(x)$ |
| 3 | `UQNOConformalCalibrator` | scalar factor from held-out ratios | `scaling_factor` |

At test time, ``UncertaintyQuantificationNeuralOperator.predict_with_bands(x)`` returns
a :class:`opifex.uncertainty.types.PredictiveDistribution` whose
:class:`opifex.uncertainty.types.PredictionInterval` is
``[base(x) - E(x) * scaling_factor, base(x) + E(x) * scaling_factor]``.

### Coverage Guarantee

The conformal calibration rule
(``opifex.neural.operators.specialized.uqno.get_coeff_quantile_idx``)
picks two indices ``(domain_idx, function_idx)`` from
``(alpha, delta, n_samples, n_gridpts)`` such that, on average, the
predicted bands cover the true target on at least ``1 - alpha``
fraction of grid points per function, for at least ``1 - delta``
fraction of functions in the calibration distribution.

## Implementation

### Step 1: Train the Base Solution Operator

```python
from flax import nnx
import optax
from opifex.neural.operators.fno.base import FourierNeuralOperator

base_fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    modes=12,
    num_layers=4,
    rngs=nnx.Rngs(42),
)
base_opt = nnx.Optimizer(base_fno, optax.adam(1e-3), wrt=nnx.Param)


@nnx.jit
def base_train_step(model, opt, x, y):
    def loss_fn(m):
        return jnp.mean((m(x) - y) ** 2)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss
```

### Step 2: Train the Residual Quantile Operator

```python
from opifex.uncertainty.losses import PointwiseQuantileLoss

residual_fno = FourierNeuralOperator(
    in_channels=1, out_channels=1,
    hidden_channels=32, modes=12, num_layers=4, rngs=nnx.Rngs(43),
)
residual_opt = nnx.Optimizer(residual_fno, optax.adam(1e-3), wrt=nnx.Param)
quantile_loss = PointwiseQuantileLoss(alpha=0.1, reduction="mean")


@nnx.jit
def residual_train_step(base, residual, opt, x, y):
    def loss_fn(r):
        base_pred = jax.lax.stop_gradient(base(x))
        widths = jnp.abs(r(x))
        return quantile_loss(y_pred=widths, y=base_pred - y)
    loss, grads = nnx.value_and_grad(loss_fn)(residual)
    opt.update(residual, grads)
    return loss
```

### Step 3: Compose, Calibrate, Predict

```python
from opifex.neural.operators.specialized.uqno import (
    UncertaintyQuantificationNeuralOperator,
    UQNOBaseSolutionOperator,
    UQNOResidualOperator,
)

uqno = UncertaintyQuantificationNeuralOperator(
    base=UQNOBaseSolutionOperator(base_fno),
    residual=UQNOResidualOperator(residual_fno),
)

# Calibrate on a held-out set.
calibrator = uqno.calibrate(x_calib, y_calib, alpha=0.1, delta=0.1)
uqno = uqno.with_calibrator(calibrator)
print(f"scaling factor: {float(calibrator.scaling_factor):.6f}")

# Predict with bands.
dist = uqno.predict_with_bands(x_test)
lower, upper = dist.interval.lower, dist.interval.upper
mean_prediction = dist.mean
coverage = float(jnp.mean((y_test >= lower) & (y_test <= upper)))
print(f"empirical coverage: {coverage:.3f} (target 1-alpha = {1-0.1:.2f})")
```

## Visualisation

![UQNO Solution](../../assets/examples/uqno_darcy/solution.png)

![UQNO Analysis](../../assets/examples/uqno_darcy/analysis.png)

## Results Summary

| Metric                    | Description                                |
|---------------------------|--------------------------------------------|
| `calibrator.scaling_factor` | Scalar conformal scaling factor          |
| `calibrator.domain_idx`     | Per-function quantile index              |
| `calibrator.function_idx`   | Across-functions quantile index          |
| Empirical pointwise coverage on test set | Should land near $1 - \alpha$ |
| Mean band width             | Per-pixel width of the calibrated interval |

The empirical coverage on the held-out test set should land near the
target $1 - \alpha$; the exact target depends jointly on $\alpha$ and
$\delta$ via the canonical ``get_coeff_quantile_idx`` rule. For
showcase-quality numbers, scale up the per-stage training samples and
epoch counts; canonical Li-style Darcy uses ~1000 / ~500 / ~500 samples
across the three stages and ~300 epochs per training phase.

## Next Steps

### Experiments to Try

1. **Tune $\alpha$ / $\delta$**: Smaller $\alpha$ widens bands;
   smaller $\delta$ raises the function-level coverage demand.
2. **Different base architectures**: The same conformal calibrator
   wraps any deterministic operator that satisfies the
   :class:`opifex.uncertainty.adapters.operators.FNOConformalAdapterSpec`
   capability.
3. **Scale to canonical setup**: 1000 train / 500 residual / 500
   calibration samples; 300 epochs per stage.

### Related Examples

| Example                                            | Level        | What You'll Learn                |
|----------------------------------------------------|--------------|----------------------------------|
| [Bayesian FNO](bayesian-fno.md)                    | Intermediate | Variational framework wrapper    |
| [FNO on Darcy](../neural-operators/fno-darcy.md)   | Beginner     | Standard FNO without uncertainty |
| [Calibration Methods](calibration.md)              | Intermediate | Post-hoc calibration techniques  |

### API Reference

- `UncertaintyQuantificationNeuralOperator`: Three-stage conformal orchestrator
- `UQNOBaseSolutionOperator`: Thin wrapper tagging an FNO as the base
- `UQNOResidualOperator`: Thin wrapper tagging an FNO as the residual
- `UQNOConformalCalibrator`: Fitted scalar scaling factor (pytree)
- `PointwiseQuantileLoss`: Quantile (pinball) loss for the residual stage
- `get_coeff_quantile_idx`: Canonical conformal-index helper

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Coverage well below $1 - \alpha$ | Train base + residual longer; calibration set may be too small |
| Bands too wide | The residual operator hasn't concentrated yet — train it more |
| Calibration `scaling_factor` is `inf` | `residual(x)` is near zero somewhere — increase `eps` in `calibrate(..., eps=...)` |
| Memory issues | Reduce `hidden_channels` or `modes`; `RESOLUTION` if needed |
