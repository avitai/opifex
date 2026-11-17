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

1. Run the computation-aware Kalman filter with
   `opifex.uncertainty.statespace.cakf_filter`, which carries the mean and
   the low-rank factor from step to step.
2. Form the CAKF posterior covariance `Σ - M M^T` from the filter's
   prior covariances and factors.
3. Check that, with one CG iteration on a scalar observation, CAKF is the
   exact `kalman_filter` posterior.

## Files

- **Python Script**: `/examples/uncertainty/statespace/cakf_smoothing.py`
- **Jupyter Notebook**: `/examples/uncertainty/statespace/cakf_smoothing.ipynb`

## Core Concepts

### Low-rank downdated covariance

CAKF maintains the implicit posterior covariance as
`Σ - M M^T` where `Σ` is the prior marginal covariance and `M` is a
low-rank factor. Each update appends one column per CG iteration, and
`cakf_filter` then truncates `M` to its leading singular directions, as
ComputationAwareKalman.jl's `filter!` and `truncate` do, so the factor
keeps the state dimension as its width across steps.

### CG search-direction policy

Each iteration uses the current residual as the CG search direction
and conjugates it against
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
(`1/3`), the CAKF iteration budget (`max_iter = 1`), and:

- `cakf_vs_exact_mean_relative` and `cakf_vs_exact_cov_relative`, the
  relative difference between the CAKF and exact Kalman posterior means
  and covariances over the whole sequence;
- `cakf_vs_truth_mean_l2` and `exact_vs_truth_mean_l2`, each filter's
  mean distance to the simulated states.

With `max_iter = 1` and a scalar observation, one CG iteration spans the
observation, so CAKF is the exact filter: on CPU in float32 the relative
differences are about 2e-6 (mean) and 6e-6 (covariance), and both
filters are 0.4272 from the states.

## Next Steps

- See `opifex.uncertainty.statespace.parallel` for the associative-scan
  parallel Kalman filter / smoother.
- See `opifex.uncertainty.statespace.sqrt_kalman` for square-root form
  filters with stable Cholesky updates.
