"""Regression-side calibration diagnostics.

* :func:`picp` — prediction interval coverage probability.
* :func:`mpiw` — mean prediction interval width.
* :func:`regression_calibration_error` — quantile calibration error of a
  Gaussian predictive against observed targets, averaged across a grid of
  quantile levels. Standard regression analogue of ECE per Kuleshov et al.
  2018 ("Accurate Uncertainties for Deep Learning Using Calibrated Regression",
  arXiv:1807.00263). Lower is better; perfectly calibrated → 0.

Quantile-level CDF inversion uses :func:`jax.scipy.stats.norm.ppf` so the
metric remains transform-friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import cdf as _norm_cdf


def picp(
    *,
    lower: jax.Array,
    upper: jax.Array,
    target: jax.Array,
    validate: bool = False,
) -> jax.Array:
    """Fraction of ``target`` values that fall in ``[lower, upper]``.

    Args:
        lower: Lower interval bound of shape ``(batch, ...)``.
        upper: Upper interval bound, same shape as ``lower``.
        target: Observed values, same shape.
        validate: When ``True``, raise if any ``upper < lower``. Defaults
            to ``False`` so the metric traces cleanly under ``jax.jit`` /
            ``jax.grad`` / ``jax.vmap``; callers that want eager safety
            checks must opt in.

    Returns:
        Scalar coverage fraction in ``[0, 1]``.

    Raises:
        ValueError: If ``validate`` and any interval is inverted.

    """
    if validate and bool(jnp.any(upper < lower)):
        raise ValueError("picp: encountered upper < lower in input interval.")
    covered = (target >= lower) & (target <= upper)
    return jnp.mean(covered.astype(jnp.float32))


def mpiw(*, lower: jax.Array, upper: jax.Array) -> jax.Array:
    """Mean prediction-interval width.

    Args:
        lower: Lower interval bound of shape ``(batch, ...)``.
        upper: Upper interval bound, same shape.

    Returns:
        Scalar mean of ``upper - lower`` over all elements.

    """
    return jnp.mean(upper - lower)


def regression_calibration_error(
    *,
    mean: jax.Array,
    variance: jax.Array,
    target: jax.Array,
    quantile_levels: jax.Array,
) -> jax.Array:
    """Regression calibration error over a grid of quantile levels.

    For each requested nominal level ``q``, computes the empirical
    proportion of targets that fall below the ``q``-quantile of the
    predictive Gaussian ``Normal(mean, sqrt(variance))``. The metric is
    the mean absolute deviation between empirical and nominal levels.

    Args:
        mean: Predictive mean ``μ`` of shape ``(batch, ...)``.
        variance: Predictive variance ``σ²`` of the same shape; must be > 0.
        target: Observed values ``y`` of the same shape.
        quantile_levels: 1-D array of nominal levels in ``(0, 1)``.

    Returns:
        Scalar mean absolute miscalibration over the requested levels.

    """
    std = jnp.sqrt(variance)
    # Empirical CDF of the residual at each nominal level: F(target | mean, std).
    cdf_values = _norm_cdf(target, loc=mean, scale=std)
    # vmap over the levels axis to get the empirical fraction at each.
    flat_cdf = cdf_values.reshape(-1)

    def empirical_at_level(level: jax.Array) -> jax.Array:
        """Return the empirical coverage fraction at one nominal quantile level."""
        return jnp.mean((flat_cdf <= level).astype(jnp.float32))

    empirical = jax.vmap(empirical_at_level)(quantile_levels)
    return jnp.mean(jnp.abs(empirical - quantile_levels))
