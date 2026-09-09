"""Interval calibration scores: one-release wrappers over calibrax.

``picp``, ``mpiw`` and ``regression_calibration_error`` are
``calibrax.metrics.functional.uncertainty``'s under opifex's keyword names.
Each call emits a ``DeprecationWarning``; the names are removed in 0.2.3.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from calibrax.metrics.functional import uncertainty as _uncertainty

from opifex.uncertainty._deprecated_metric import warn_deprecated_metric


_HOME = "calibrax.metrics.functional.uncertainty"
_HERE = "opifex.uncertainty.calibration"


def picp(*, lower: Any, upper: Any, target: Any, validate: bool = False) -> Any:
    """Prediction-interval coverage: calibrax's ``picp``.

    Args:
        lower: Lower interval bounds.
        upper: Upper interval bounds.
        target: Observed values.
        validate: Raise instead of scoring an inverted interval.

    Returns:
        The fraction of targets inside their interval.

    Raises:
        ValueError: If ``validate`` and any interval is inverted.
    """
    warn_deprecated_metric(f"{_HERE}.picp", f"{_HOME}.picp")
    if validate and bool(jnp.any(jnp.asarray(upper) < jnp.asarray(lower))):
        raise ValueError("picp: encountered upper < lower in input interval.")
    return _uncertainty.picp(lower, upper, target)


def mpiw(*, lower: Any, upper: Any) -> Any:
    """Mean prediction-interval width: calibrax's ``mpiw``."""
    warn_deprecated_metric(f"{_HERE}.mpiw", f"{_HOME}.mpiw")
    return _uncertainty.mpiw(lower, upper)


def regression_calibration_error(
    *, mean: Any, variance: Any, target: Any, quantile_levels: Any
) -> Any:
    """Regression calibration error: calibrax's ``regression_calibration_error``."""
    warn_deprecated_metric(
        f"{_HERE}.regression_calibration_error", f"{_HOME}.regression_calibration_error"
    )
    return _uncertainty.regression_calibration_error(
        mean, variance, target, quantile_levels=quantile_levels
    )


__all__ = ["mpiw", "picp", "regression_calibration_error"]
