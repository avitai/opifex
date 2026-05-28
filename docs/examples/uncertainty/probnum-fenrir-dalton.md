# Fenrir vs DALTON likelihoods on a linear-drift ODE inverse problem

| Property | Value |
|---|---|
| **Level** | Advanced |
| **Runtime** | < 5 s (CPU) |
| **Prerequisites** | JAX, linear-Gaussian state-space models, ODE likelihoods |

## Overview

Probabilistic ODE solvers produce a Gaussian-process posterior over the
solution. Combining that posterior with noisy data observations yields
a **data marginal log-likelihood** suitable for parameter inference —
the building block for Bayesian ODE inverse problems.

Two complementary data-likelihood combinators are implemented in
`opifex.uncertainty.scientific._likelihoods`:

- **Fenrir** (Tronarp et al. 2022, arXiv:2202.01287) — backward
  smoothing of an *unconditioned* forward solver pass that conditions
  on the data only during the backward sweep.
- **DALTON** (Wu et al. 2023, arXiv:2306.05566) — three-term
  combinator `data_ll + with_pn_ll - without_pn_ll` that explicitly
  accounts for the differential in the solver's probabilistic-numerics
  log-likelihood between data-conditioned and unconditioned passes.

The example targets the decay ODE `dy/dt = -θ y` with closed-form
solution `y(t) = exp(-θ t)`. We score the *true* parameter
`θ = 0.5` under both likelihoods in two regimes:

1. **Well-specified observation noise** — the filter is told the
   exact data variance.
2. **Misspecified observation noise** — the filter is told 100x less
   variance than the truth.

## What You Will Learn

1. Build a linear-Gaussian state-space approximation of a scalar ODE.
2. Run an unconditioned forward Kalman filter via
   `opifex.uncertainty.statespace.kalman.kalman_filter`.
3. Evaluate Fenrir's backward-smoothing log-likelihood with
   `fenrir_data_loglik`.
4. Evaluate DALTON's three-term combinator with `dalton_data_loglik`.
5. Observe how the two likelihoods rank candidate parameters
   differently when observation noise is misspecified.

## Files

- **Python Script**: `/examples/uncertainty/probabilistic_numerics/fenrir_dalton.py`
- **Jupyter Notebook**: `/examples/uncertainty/probabilistic_numerics/fenrir_dalton.ipynb`

## Core Concepts

### Fenrir backward smoothing

Given filter outputs from an unconditioned forward pass (data variance
inflated to `~1e12`), Fenrir sweeps backward applying a Kalman
measurement update at each step that carries an observation,
accumulating the innovation log-density. The total is the data
marginal log-likelihood under the smoothed posterior.

Reference: Tronarp+ 2022, *Fenrir: Physics-Enhanced Regression for
Initial Value Problems*, ICML.

### DALTON three-term combinator

DALTON sums the per-observation log-likelihood from a *data-conditioned*
forward pass with the differential in the solver's
probabilistic-numerics log-likelihood between conditioned and
unconditioned passes:

```
ℓ_DALTON = ℓ_data + ℓ_PN_with_data - ℓ_PN_without_data
```

Reference: Wu+ 2023, *Data-Adaptive Probabilistic Likelihood
Approximation for ODEs*.

## Why This Matters

Bayesian ODE inverse problems hinge on a tractable, calibrated data
likelihood for parameter inference. Fenrir is preferred under
**well-specified** observation noise (matched filter variance);
DALTON is preferred under **misspecified** noise because its
three-term form partially corrects for the solver's
probabilistic-numerics bias.

## Expected Output

The example reports four log-likelihoods at `θ = 0.5`:

| Regime | Fenrir | DALTON |
|---|---|---|
| Well-specified | finite, positive | finite, positive |
| Misspecified (noise 1/100x) | large negative | larger negative |

The DALTON correction is most informative when comparing parameters
against each other — its absolute scale is not directly comparable to
Fenrir's.

## Next Steps

- See `opifex.uncertainty.scientific.probabilistic_numerics` for the
  adapter catalogue (Probdiffeq, Tornadox, ProbNum).
- See `opifex.uncertainty.scientific._priors_sde` for IWP / IOUP /
  Matérn priors used inside Fenrir / DALTON solver passes.
