"""Spread-skill ratio for ensemble calibration.

Fortin et al. 2014, "Why should ensemble spread match the RMSE of the
ensemble mean?" — for a perfectly calibrated ensemble of size ``M``, the
spread (std-dev across members) should match the RMSE of the ensemble
mean against the verification target. Ratio is reported as
``spread / RMSE``; values < 1 indicate under-dispersion, > 1 over.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def spread_skill_ratio(*, ensemble: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean ensemble-spread over RMSE of the ensemble mean.

    Args:
        ensemble: Shape ``(n_samples, n_members)``.
        targets: Shape ``(n_samples,)``.

    Returns:
        Scalar ratio. ~1.0 indicates calibrated dispersion.
    """
    ensemble_mean = jnp.mean(ensemble, axis=1)
    # Population std (matches the Fortin definition for ensemble spread).
    ensemble_spread = jnp.std(ensemble, axis=1)
    mean_spread = jnp.mean(ensemble_spread)
    rmse = jnp.sqrt(jnp.mean((ensemble_mean - targets) ** 2))
    return mean_spread / (rmse + 1e-12)
