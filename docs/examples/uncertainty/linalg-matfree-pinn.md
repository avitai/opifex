# Matrix-free PINN curvature: Lanczos log-det + XNysTrace Fisher trace

| Property | Value |
|---|---|
| **Level** | Intermediate |
| **Runtime** | < 5 s (CPU) |
| **Prerequisites** | JAX, Flax NNX |

## Overview

Curvature summaries (Hessian log-determinant, Fisher trace) drive Laplace
posteriors and evidence-bound computations across opifex's uncertainty
stack. For all but the smallest models these matrices are too large to
materialise — opifex exposes matrix-free estimators in
`opifex.uncertainty.linalg` that consume only a mat-vec callable.

This example shows the two most commonly used estimators in action on a
tiny Bayesian classifier head:

- `slq_logdet` — stochastic Lanczos quadrature `log det` (matfree
  reference: Krämer 2024 arXiv:2405.17277).
- `xnys_trace` — XNysTrace exact-on-low-rank trace estimator (traceax
  reference: Nahid et al.).

## What You Will Learn

1. Wrap any Flax NNX module's loss-Hessian as a matrix-free
   `matvec(v) = (F + τI) v` closure using `jax.flatten_util.ravel_pytree`.
2. Estimate `log det(F + τI)` with stochastic Lanczos quadrature.
3. Estimate `trace(F + τI)` with the XNysTrace algorithm.
4. JIT-compile both estimators end-to-end.

## Files

- **Python Script**: `/examples/uncertainty/linalg/matfree_pinn_calibration.py`
- **Jupyter Notebook**: `/examples/uncertainty/linalg/matfree_pinn_calibration.ipynb`

## Core Concepts

### Lanczos log-determinant

The SLQ estimator approximates `log det(A) = trace(log A)` by Hutchinson's
identity and a `num_matvecs`-step Lanczos tridiagonalisation per
Rademacher probe. Variance scales as `O(1 / num_samples)`. Lucky
breakdowns (Lanczos hits an A-invariant subspace) yield exact results;
the implementation handles this case under JIT-friendly static shapes.

### XNysTrace

A Nyström low-rank approximation of `A` plus an exchangeable residual
correction. For PSD matrices whose effective rank is below
`num_samples`, XNysTrace recovers `trace(A)` up to numerical roundoff.

## Why This Matters

Bayesian PINNs and Bayesian neural operators both require `log det`
and trace summaries of the parameter Fisher block for evidence-based
hyperparameter selection. Matrix-free estimators make these summaries
tractable for parameter counts in the tens of thousands without
materialising any `n x n` matrix.

## Expected Output

The example prints three scalars: the matrix dimension, an SLQ log-det
estimate, and an XNysTrace Fisher-trace estimate. All values are
deterministic given `jax.random.PRNGKey(0)`.

## Next Steps

- See `opifex.uncertainty.curvature.laplace` for a full diagonal
  Laplace posterior using the same primitives.
- See `opifex.uncertainty.statespace.cakf` for compute-aware Kalman
  filtering, which reuses the Lanczos / CG infrastructure.
