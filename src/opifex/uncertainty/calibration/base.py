"""Calibration scores: one-release wrappers over calibrax.

``gaussian_nll`` is ``calibrax.metrics.functional.uncertainty.gaussian_nll``;
``brier_score`` and ``expected_calibration_error`` are
``calibrax.metrics.functional.calibration``'s; ``pinball_loss`` is
``calibrax.metrics.functional.regression.quantile_loss``. Each call emits a
``DeprecationWarning``; the names are removed in 0.2.3.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from calibrax.metrics.functional import (
    calibration as _calibration,
    regression as _regression,
    uncertainty as _uncertainty,
)

from opifex.uncertainty._deprecated_metric import warn_deprecated_metric


_HERE = "opifex.uncertainty.calibration"


def gaussian_nll(*, mean: Any, variance: Any, target: Any, validate: bool = False) -> Any:
    """Gaussian negative log-likelihood: calibrax's ``uncertainty.gaussian_nll``.

    Args:
        mean: Predicted means.
        variance: Predicted variances, strictly positive.
        target: Observed values.
        validate: Raise instead of propagating a non-positive variance.

    Returns:
        The mean negative log-likelihood.

    Raises:
        ValueError: If ``validate`` and any variance entry is non-positive.
    """
    warn_deprecated_metric(
        f"{_HERE}.gaussian_nll", "calibrax.metrics.functional.uncertainty.gaussian_nll"
    )
    if validate and bool(jnp.any(jnp.asarray(variance) <= 0.0)):
        raise ValueError("gaussian_nll: variance must be strictly positive elementwise.")
    return _uncertainty.gaussian_nll(mean, variance, target)


def brier_score(*, probabilities: Any, targets: Any) -> Any:
    """Brier score: calibrax's ``calibration.brier_score``."""
    warn_deprecated_metric(
        f"{_HERE}.brier_score", "calibrax.metrics.functional.calibration.brier_score"
    )
    return _calibration.brier_score(probabilities, targets)


def expected_calibration_error(*, probabilities: Any, targets: Any, num_bins: int = 10) -> Any:
    """ECE: calibrax's ``calibration.expected_calibration_error``."""
    warn_deprecated_metric(
        f"{_HERE}.expected_calibration_error",
        "calibrax.metrics.functional.calibration.expected_calibration_error",
    )
    return _calibration.expected_calibration_error(probabilities, targets, num_bins=num_bins)


def pinball_loss(*, predictions: Any, targets: Any, quantile: float) -> Any:
    """Pinball loss: calibrax's ``regression.quantile_loss``."""
    warn_deprecated_metric(
        f"{_HERE}.pinball_loss", "calibrax.metrics.functional.regression.quantile_loss"
    )
    return _regression.quantile_loss(predictions, targets, quantile=quantile)


__all__ = ["brier_score", "expected_calibration_error", "gaussian_nll", "pinball_loss"]
