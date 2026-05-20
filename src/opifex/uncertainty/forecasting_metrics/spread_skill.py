"""Spread-skill ratio for ensemble calibration.

Fortin et al. 2014, "Why should ensemble spread match the RMSE of the
ensemble mean?" — for a perfectly calibrated ensemble of size ``M``, the
spread should match the RMSE of the ensemble mean against the
verification target. Values < 1 indicate under-dispersion, > 1 over.

Both numerator and denominator use the *unbiased* finite-ensemble
estimators recommended by Fortin 2014:

* numerator: ``sqrt(mean(Var(ensemble, ddof=1)))`` — root mean unbiased
  ensemble variance, matching WeatherBenchX ``EnsembleRootMeanVariance``.
* denominator: ``sqrt(mean(MSE(ensemble_mean, target) − Var/M))`` — the
  bias-corrected mean squared error of the ensemble mean, matching
  WeatherBenchX ``UnbiasedEnsembleMeanSquaredError``.

The previous biased implementation used population std (``ddof=0``) plus
the standard RMSE; both terms underestimated the unbiased quantities for
small ensembles, biasing the ratio.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def spread_skill_ratio(*, ensemble: jax.Array, targets: jax.Array) -> jax.Array:
    """Unbiased spread-skill ratio (Fortin 2014 / WBX ``UnbiasedSpreadSkillRatio``).

    Args:
        ensemble: Shape ``(n_samples, n_members)``.
        targets: Shape ``(n_samples,)``.

    Returns:
        Scalar ratio. ~1.0 indicates calibrated dispersion. Returns
        ``nan`` when the bias-corrected MSE is non-positive (which can
        happen for very small samples — the unbiased estimator is not
        guaranteed positive).
    """
    n_members = ensemble.shape[1]
    ensemble_mean = jnp.mean(ensemble, axis=1)
    # Per-sample unbiased ensemble variance (ddof=1).
    per_sample_variance = jnp.var(ensemble, axis=1, ddof=1)
    mean_variance = jnp.mean(per_sample_variance)
    # Bias-corrected MSE: subtract Var/M from the squared error of the
    # ensemble mean (Ferro 2008 / WBX ``UnbiasedEnsembleMeanSquaredError``).
    squared_error = (ensemble_mean - targets) ** 2
    unbiased_mse = jnp.mean(squared_error - per_sample_variance / n_members)
    return jnp.sqrt(mean_variance / unbiased_mse)
