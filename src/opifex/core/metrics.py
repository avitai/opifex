"""Operator-learning error metrics: one-release wrappers over calibrax.

``per_sample_relative_l2`` and ``relative_l2_error`` are
``calibrax.metrics.functional.regression``'s (calibrax 0.1.3 added the
per-sample-mean relative L2 that operator learning reports). Each call emits a
``DeprecationWarning``; the names are removed in 0.2.3, and the trainer and the
examples already call calibrax directly.
"""

from __future__ import annotations

from typing import Any

from calibrax.metrics.functional import regression as _regression

from opifex._deprecated import warn_deprecated


_HOME = "calibrax.metrics.functional.regression"
_HERE = "opifex.core.metrics"


def per_sample_relative_l2(prediction: Any, target: Any) -> Any:
    """Per-sample relative L2 error: calibrax's ``per_sample_relative_l2``."""
    warn_deprecated(f"{_HERE}.per_sample_relative_l2", f"{_HOME}.per_sample_relative_l2")
    return _regression.per_sample_relative_l2(prediction, target)


def relative_l2_error(prediction: Any, target: Any) -> Any:
    """Mean per-sample relative L2 error: calibrax's ``relative_l2_error``."""
    warn_deprecated(f"{_HERE}.relative_l2_error", f"{_HOME}.relative_l2_error")
    return _regression.relative_l2_error(prediction, target)


__all__ = ["per_sample_relative_l2", "relative_l2_error"]
