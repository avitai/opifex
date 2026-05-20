"""Energy score for multivariate ensemble forecasts.

Gneiting & Raftery 2007 §4.2: ``ES(P, y) = mean_i ||X_i - y|| - 0.5
mean_{i,j} ||X_i - X_j||`` with the Euclidean norm. Reduces to CRPS for
1-D outputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def energy_score(*, ensemble: jax.Array, targets: jax.Array) -> jax.Array:
    """Per-sample energy score for a multivariate ensemble.

    Args:
        ensemble: Array of shape ``(n_samples, n_members, n_output_dims)``.
        targets: Array of shape ``(n_samples, n_output_dims)``.

    Returns:
        Array of shape ``(n_samples,)`` — per-sample energy scores.
    """
    forecast_error = jnp.linalg.norm(ensemble - targets[:, None, :], axis=-1)
    mean_forecast_error = jnp.mean(forecast_error, axis=1)
    pairwise = jnp.linalg.norm(ensemble[:, :, None, :] - ensemble[:, None, :, :], axis=-1)
    mean_pairwise = jnp.mean(pairwise, axis=(1, 2))
    return mean_forecast_error - 0.5 * mean_pairwise
