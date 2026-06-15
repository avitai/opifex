"""Surrogate-uncertainty decomposition.

A small array-only helper that combines three independent variance
sources reported by a surrogate model into a single total-uncertainty
estimate via the standard sum-of-squares decomposition

    sigma_total^2 = sigma_prediction^2 + sigma_residual^2 + sigma_calibration^2

Designed to be model-agnostic: callers pass the three component
arrays explicitly (e.g. surrogate predictive variance, held-out
residual variance, post-hoc calibration variance) and receive a typed
result back. No neural-model internals are accessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True, kw_only=True)
class SurrogateUncertaintyResult:
    """Container for the per-source variance decomposition.

    Attributes:
        total_std: ``sqrt(sigma_pred^2 + sigma_resid^2 + sigma_cal^2)``.
        prediction_std: Surrogate predictive standard deviation.
        residual_std: Held-out residual standard deviation (zero if not
            supplied).
        calibration_std: Post-hoc calibration standard deviation (zero
            if not supplied).
    """

    total_std: jax.Array
    prediction_std: jax.Array
    residual_std: jax.Array
    calibration_std: jax.Array


def decompose_surrogate_uncertainty(
    *,
    prediction_std: jax.Array,
    residual_std: jax.Array | None = None,
    calibration_std: jax.Array | None = None,
) -> SurrogateUncertaintyResult:
    """Combine independent variance sources into a total-uncertainty estimate.

    Args:
        prediction_std: Predictive standard deviation reported by the
            surrogate. Shape ``(N,)`` or ``(N, K)``.
        residual_std: Optional held-out residual standard deviation.
            Defaults to zero (no residual contribution). Must broadcast
            to ``prediction_std`` if supplied.
        calibration_std: Optional calibration standard deviation (e.g.
            from temperature scaling cross-validation). Defaults to
            zero. Must broadcast to ``prediction_std`` if supplied.

    Returns:
        :class:`SurrogateUncertaintyResult` with each component and the
        combined total.

    Raises:
        ValueError: If any standard deviation is negative.
    """
    pred = jnp.asarray(prediction_std)
    resid = jnp.zeros_like(pred) if residual_std is None else jnp.asarray(residual_std)
    calib = jnp.zeros_like(pred) if calibration_std is None else jnp.asarray(calibration_std)

    # Use eager Python checks for negatives — they're cheap and prevent
    # silent garbage propagation. JIT consumers should pass non-negative
    # tensors so the check trips at trace time, not runtime.
    if jnp.any(pred < 0) or jnp.any(resid < 0) or jnp.any(calib < 0):
        raise ValueError("All standard-deviation inputs must be non-negative.")

    total = jnp.sqrt(pred**2 + resid**2 + calib**2)
    return SurrogateUncertaintyResult(
        total_std=total,
        prediction_std=pred,
        residual_std=resid,
        calibration_std=calib,
    )


__all__ = ["SurrogateUncertaintyResult", "decompose_surrogate_uncertainty"]
