# Diagonal Laplace posterior for a small MLP classifier

| Property | Value |
|---|---|
| **Level** | Intermediate |
| **Runtime** | < 5 s (CPU) |
| **Prerequisites** | JAX, Flax NNX, Laplace approximation |

## Overview

The diagonal Laplace approximation (MacKay 1992; Daxberger et al. 2021
arXiv:2106.14806) builds a Gaussian posterior around a MAP point using
the diagonal of the empirical Fisher matrix. It is the cheapest
post-hoc Bayesian deep learning method: one extra pass over the
training data to populate the per-parameter precision diagonal, no
re-training.

This example trains a 2-layer MLP binary classifier with
`optax.adam`, builds the diagonal Laplace posterior at the trained
MAP point via
`opifex.uncertainty.curvature.diagonal_laplace_posterior`, and
reports two calibration summaries:

- **ECE** — `opifex.uncertainty.calibration.expected_calibration_error`
  (CalibraX backend) against the binary class label.
- **ANEES** — `opifex.uncertainty.metrics.anees` on the predicted
  logits with the Laplace posterior variance as a coarse predictive
  covariance proxy.

## What You Will Learn

1. Train a `flax.nnx` classifier with `optax.adam`.
2. Use `nnx.split` / `nnx.merge` + `jax.flatten_util.ravel_pytree` to
   project an NNX module's parameters into a flat array for Laplace.
3. Build the diagonal Laplace posterior in one line via
   `diagonal_laplace_posterior`.
4. Compute ECE (classification) and ANEES (state-space style) on the
   same model's predictions.

## Files

- **Python Script**: `/examples/uncertainty/curvature/laplace_classifier.py`
- **Jupyter Notebook**: `/examples/uncertainty/curvature/laplace_classifier.ipynb`

## Core Concepts

### Diagonal empirical Fisher

The per-parameter Fisher diagonal estimate is

```
F_ii ≈ mean over samples of (∂L_i / ∂θ_i)²
```

where `L_i` is the per-sample loss. The Laplace posterior precision
is `τ + F_ii` for prior precision `τ`.

### NNX parameter flattening

`flax.nnx` modules are pytrees of `Param` leaves. `jax.flatten_util.ravel_pytree`
returns a flat array view of these leaves plus an `unflatten` callback,
so a per-sample loss expressed over a flat parameter vector composes
cleanly with the curvature routines in
`opifex.uncertainty.curvature.fisher`.

## Why This Matters

Diagonal Laplace is the simplest path from a deterministic-trained
neural network to a Bayesian posterior. The opifex implementation
sticks to pure JAX, supports arbitrary per-sample losses, and is
trivially jit-compatible — `diagonal_laplace_posterior` itself is a
single `jax.vmap(jax.grad(loss))` plus a square-and-mean reduction.

## Expected Output

For `num_samples = 128`, two-layer MLP `(4 -> 8 -> 2)`, 100 Adam
steps at lr = `1e-2`, prior precision `τ = 1.0`, seeded by
`jax.random.PRNGKey(0)`:

| Summary | Value (approx) |
|---|---|
| `num_parameters` | 58 |
| `posterior_precision_mean` | ~ 1.01 |
| `ece` | < 0.10 |
| `anees` | finite, positive |

## Next Steps

- See `opifex.uncertainty.curvature.ggn` for generalized Gauss-Newton
  vector products, a stricter PSD curvature operator than the
  empirical Fisher.
- See `opifex.uncertainty.calibration.temperature.TemperatureScaling`
  for post-hoc multiclass calibration on top of a Laplace posterior.
