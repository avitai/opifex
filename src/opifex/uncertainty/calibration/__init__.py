"""Calibration metrics and calibrators.

Public surface:

* Metrics (`opifex.uncertainty.calibration.base`,
  `opifex.uncertainty.calibration.regression`):

  - :func:`gaussian_nll` — mean Gaussian negative log-likelihood (regression).
  - :func:`brier_score` — wraps CalibraX (classification).
  - :func:`expected_calibration_error` — wraps CalibraX (classification, fixed bins).
  - :func:`pinball_loss` — wraps CalibraX quantile loss.
  - :func:`picp` / :func:`mpiw` — interval coverage and width.
  - :func:`regression_calibration_error` — Gaussian quantile-calibration error.

* Calibrators (`opifex.uncertainty.calibration.temperature`):

  - :class:`TemperatureScaling` + :class:`TemperatureScalingState` — Guo et al.
    temperature scaling for multiclass logits.
"""

from __future__ import annotations

from opifex.uncertainty.calibration.base import (
    brier_score,
    expected_calibration_error,
    gaussian_nll,
    pinball_loss,
)
from opifex.uncertainty.calibration.regression import (
    mpiw,
    picp,
    regression_calibration_error,
)
from opifex.uncertainty.calibration.temperature import (
    nll_loss_at_temperature,
    TemperatureScaling,
    TemperatureScalingState,
)


__all__ = [
    "TemperatureScaling",
    "TemperatureScalingState",
    "brier_score",
    "expected_calibration_error",
    "gaussian_nll",
    "mpiw",
    "nll_loss_at_temperature",
    "picp",
    "pinball_loss",
    "regression_calibration_error",
]
