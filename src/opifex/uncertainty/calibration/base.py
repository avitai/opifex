"""Base calibration metrics shared across classification and regression.

The classification primitives — Brier score, expected calibration error
(ECE), and the pinball / quantile loss — are thin wrappers around CalibraX
functional metrics
(`calibrax.metrics.functional.calibration.brier_score`,
`...calibration.expected_calibration_error`,
`...regression.quantile_loss`). Opifex does not re-implement those
formulas; the wrappers exist only to give the calibration subsystem a
single import surface and stable keyword-only signatures consistent with
the Opifex value-object contracts.

The regression NLL is implemented locally because CalibraX does not expose
a Gaussian-mean+variance log-likelihood (its closest equivalents are
classification log-loss and CRPS). The formula is the standard univariate
Gaussian negative log-density averaged over the batch:

    NLL = mean_i 0.5 * (log(2π σ_i²) + (y_i - μ_i)² / σ_i²)

All metrics here are pure functions over ``jax.Array``s with no module
state, no rng draws, and no Python-level branching on traced array
contents — they trace cleanly under ``jax.jit`` / ``jax.grad`` / ``jax.vmap``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from calibrax.metrics.functional.calibration import (
    brier_score as _calibrax_brier_score,
    expected_calibration_error as _calibrax_ece,
)
from calibrax.metrics.functional.regression import (
    quantile_loss as _calibrax_quantile_loss,
)


def gaussian_nll(
    *,
    mean: jax.Array,
    variance: jax.Array,
    target: jax.Array,
    validate: bool = False,
) -> jax.Array:
    """Mean Gaussian negative log-likelihood under a diagonal predictive.

    Args:
        mean: Predictive mean ``μ`` of shape ``(batch, ...)``.
        variance: Predictive variance ``σ²`` of the same shape; must be > 0.
        target: Observed values ``y`` of the same shape.
        validate: When ``True``, raise ``ValueError`` if any variance entry
            is non-positive. Defaults to ``False`` so the metric traces
            cleanly under ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` without
            a data-dependent Python branch; callers that want eager safety
            checks must opt in.

    Returns:
        The mean of ``0.5 * (log(2π σ²) + (y - μ)² / σ²)`` over all elements.

    Raises:
        ValueError: If ``validate`` and any variance entry is non-positive.

    """
    if validate and bool(jnp.any(variance <= 0.0)):
        raise ValueError("gaussian_nll: variance must be strictly positive elementwise.")
    diff = target - mean
    elementwise = 0.5 * (jnp.log(2.0 * jnp.pi * variance) + diff * diff / variance)
    return jnp.mean(elementwise)


def brier_score(
    *,
    probabilities: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Brier score — wraps ``calibrax.metrics.functional.calibration.brier_score``.

    Args:
        probabilities: Per-sample probabilities of the positive class, in ``[0, 1]``.
        targets: Per-sample binary ground truth in ``{0, 1}``.

    Returns:
        Scalar Brier score, lower is better.

    """
    return _calibrax_brier_score(probabilities, targets)


def expected_calibration_error(
    *,
    probabilities: jax.Array,
    targets: jax.Array,
    num_bins: int = 10,
) -> jax.Array:
    """Compute expected calibration error via the CalibraX equal-width-bin estimator.

    Args:
        probabilities: Per-sample probabilities of the positive class.
        targets: Binary ground truth.
        num_bins: Number of equal-width bins over ``[0, 1]``.

    Returns:
        Scalar ECE, lower is better.

    """
    return _calibrax_ece(probabilities, targets, num_bins=num_bins)


def pinball_loss(
    *,
    predictions: jax.Array,
    targets: jax.Array,
    quantile: float = 0.5,
) -> jax.Array:
    """Quantile (pinball) loss — wraps ``calibrax.metrics.functional.regression.quantile_loss``.

    Args:
        predictions: Predicted quantile values.
        targets: Ground truth values.
        quantile: Target quantile in ``(0, 1)``. ``0.5`` is the median.

    Returns:
        Mean pinball loss.

    """
    return _calibrax_quantile_loss(predictions, targets, quantile=quantile)
