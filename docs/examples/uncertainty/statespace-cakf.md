# Compute-aware Kalman filtering on sparsely observed linear systems

| Property | Value |
|---|---|
| **Level** | Intermediate |
| **Runtime** | < 5 s (CPU) |
| **Prerequisites** | JAX, linear-Gaussian state-space models |

## Overview

The compute-aware Kalman filter (CAKF, Pförtner et al. 2024
arXiv:2405.08971) replaces the dense Kalman gain computation with a
truncated CG sweep that propagates a low-rank correction factor to the
prior covariance. The number of CG iterations per update step trades
posterior accuracy for compute — `max_iter = obs_dim` recovers the
exact Kalman filter, smaller values run faster at the cost of a
slightly biased posterior mean.

This example simulates a small constant-velocity model with scalar
position observations, masks two-thirds of the observations as
non-informative, and compares the CAKF posterior mean against the
exact sequential Kalman filter from
`opifex.uncertainty.statespace.kalman`.

## What You Will Learn

1. Use `cakf_predict` and `cakf_update` from
   `opifex.uncertainty.statespace.cakf` directly.
2. Drive a sequential CAKF roll-out via `jax.lax.scan` while
   preserving a fixed-shape factor carry (a JAX scan requirement).
3. Compare the CAKF posterior mean against the exact
   `kalman_filter` posterior at matched observation budget.

## Files

- **Python Script**: `/examples/uncertainty/statespace/cakf_smoothing.py`
- **Jupyter Notebook**: `/examples/uncertainty/statespace/cakf_smoothing.ipynb`

## Note on API Surface

The current public surface of `opifex.uncertainty.statespace.cakf` is
the low-level pair `cakf_predict` / `cakf_update`. A higher-level
`cakf_smooth` wrapper is **not** part of this slice — the example
demonstrates a sequential filter via `jax.lax.scan` over the low-level
primitives. The reference implementation in
`../ComputationAwareKalman.jl` provides full smoothing; opifex will
land an analogous wrapper in a future slice.

## Core Concepts

### Low-rank downdated covariance

CAKF maintains the implicit posterior covariance as
`Σ - M M^T` where `Σ` is the prior marginal covariance and `M` is a
growing low-rank factor. After `k` CG iterations per update step,
`M` has `k` extra columns. The example uses a sliding window of width
`max_iter` to keep the JAX-traced carry shape static.

### CG search-direction policy

Each iteration uses the current residual as the CG search direction
(`CGPolicy` in the Julia reference) and conjugates it against
previously selected directions via Gram-Schmidt. The result is a
truncated Lanczos-style basis for the observation-space residual.

## Why This Matters

Filtering and smoothing dominate the compute cost of probabilistic
ODE solvers (Tronarp+ 2019, Krämer+ 2024) and physics-informed
state-space models (Wenger+ 2023). CAKF/CAKS extends the same
compute-aware idea to those settings — the linalg primitives in
`opifex.uncertainty.linalg` plug directly into the CAKF inner loop.

## Expected Output

The example prints the number of steps, the observed fraction
(`1/3`), the CAKF iteration budget (`max_iter = 1`), and three L2
posterior-mean error summaries:

- `cakf_vs_exact_mean_l2` — CAKF vs exact Kalman posterior mean.
- `cakf_vs_truth_mean_l2` — CAKF posterior mean vs ground truth.
- `exact_vs_truth_mean_l2` — exact Kalman posterior mean vs truth.

With `max_iter = 1` and `observation_dim = 1` the CAKF posterior
mean already tracks the exact Kalman filter at typical accuracy.

## Next Steps

- See `opifex.uncertainty.statespace.parallel` for the associative-scan
  parallel Kalman filter / smoother.
- See `opifex.uncertainty.statespace.sqrt_kalman` for square-root form
  filters with stable Cholesky updates.
