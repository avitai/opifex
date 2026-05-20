"""Empirical and fair CRPS for ensemble forecasts.

References (canonical):
* Gneiting & Raftery 2007 — strictly proper CRPS.
* Ferro 2014, "Fair scores for ensemble forecasts" — finite-ensemble
  bias-corrected (fair) CRPS.

Cross-checked against ``calibrax.metrics.functional.regression.crps`` for
the empirical-form numerical core, and against WeatherBenchX's
``CRPSEnsemble`` family for the fair-CRPS adjustment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def crps(*, predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Empirical ensemble CRPS averaged over samples.

    ``mean_i(|X_im - y_i|) - 0.5 * mean_i mean_{j,k}(|X_ij - X_ik|)``

    Args:
        predictions: Ensemble forecasts of shape ``(n_samples, n_members)``.
        targets: Observed targets of shape ``(n_samples,)``.

    Returns:
        Scalar mean CRPS. Lower is better.
    """
    forecast_error = jnp.mean(jnp.abs(predictions - targets[:, None]), axis=1)
    pairwise = jnp.abs(predictions[:, :, None] - predictions[:, None, :])
    spread = 0.5 * jnp.mean(pairwise, axis=(1, 2))
    return jnp.mean(forecast_error - spread)


def fair_crps(*, predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Fair (finite-ensemble bias-corrected) CRPS per Ferro 2014.

    Replaces the ``1/M^2`` averaging of pairwise spread with the unbiased
    ``1/(M(M-1))`` estimator. Recovers the population CRPS as ``M → ∞``
    (the empirical form has a downward bias for finite ensembles).
    """
    n_members = predictions.shape[1]
    forecast_error = jnp.mean(jnp.abs(predictions - targets[:, None]), axis=1)
    pairwise = jnp.abs(predictions[:, :, None] - predictions[:, None, :])
    # Sum over upper triangle to drop the i=j diagonal; multiply by 2 / (M(M-1)).
    total_pairs = n_members * (n_members - 1)
    spread = jnp.sum(pairwise, axis=(1, 2)) / total_pairs
    return jnp.mean(forecast_error - 0.5 * spread)
