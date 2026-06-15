# Bayesian quadrature evidence: VanillaBQ vs WSABI-L vs Monte Carlo

| Property | Value |
|---|---|
| **Level** | Intermediate |
| **Runtime** | < 5 s (CPU) |
| **Prerequisites** | JAX |

## Overview

Bayesian model evidence integrals of the form `Z = ∫ f(x) p(x) dx` are
notoriously sample-inefficient under naive Monte Carlo. Bayesian
quadrature replaces the empirical mean estimator with a Gaussian-process
surrogate of the integrand, yielding closed-form posterior mean and
variance of the integral that converge faster than `O(1/√N)`.

This example targets the closed-form integral
`Z = ∫ exp(-x²/2) N(x; 0, 1) dx = 1 / sqrt(2) ≈ 0.7071068` at a matched
point budget across three estimators.

## What You Will Learn

1. Use `vanilla_bayesian_quadrature` for closed-form GP-BQ posterior
   moments under an RBF kernel and Gaussian measure.
2. Use `wsabi_l_bayesian_quadrature` for the WSABI-L bounded-integrand
   variant (Gunter et al. 2014).
3. Use `bayesian_monte_carlo` as a baseline at the same budget.
4. Verify that GP-based BQ achieves several orders of magnitude lower
   absolute error than MC at `N = 16` samples.

## Files

- **Python Script**: `/examples/uncertainty/quadrature/bayesian_quadrature_evidence.py`
- **Jupyter Notebook**: `/examples/uncertainty/quadrature/bayesian_quadrature_evidence.ipynb`

## Core Concepts

### Vanilla Bayesian quadrature

Briol et al. 2019 §2.4. The posterior mean and variance of `∫ f dπ`
under a GP prior with RBF kernel and Gaussian measure are

```
mean = q_K.T @ (K_XX + σ²I)^{-1} @ y
var  = q_Kq - q_K.T @ (K_XX + σ²I)^{-1} @ q_K
```

with `q_K` the kernel mean against the measure and `q_Kq` the double
kernel mean. Both are closed-form for the RBF kernel and a diagonal
Gaussian measure.

### WSABI-L

Gunter et al. 2014. Models a non-negative integrand as
`f(x) = α + 0.5 g(x)²` with a GP on `g`, then integrates the
linearised Taylor approximation. Useful when the integrand is known
to be non-negative (likelihoods, evidence).

## Why This Matters

Evidence-based hyperparameter selection, posterior model comparison,
and rare-event integration all benefit from Bayesian quadrature's
faster convergence than MC. The vanilla GP-BQ estimator gives an
explicit posterior variance — a built-in uncertainty estimate for the
integral.

## Expected Output

For `N = 16` samples seeded by `jax.random.PRNGKey(0)`:

| Estimator | Absolute error |
|---|---|
| Vanilla BQ | `< 1e-6` |
| WSABI-L | `< 1e-5` |
| Monte Carlo | `~ 0.08` |

Bayesian quadrature beats MC by ~5 orders of magnitude.

## Next Steps

- See `opifex.uncertainty.quadrature.sober` for batch-acquisition BQ.
- See `opifex.uncertainty.quadrature.frank_wolfe_bq` for kernel-mean
  recombination via Frank-Wolfe.
