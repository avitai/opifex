"""Calibrators.

The calibration *metrics* (Brier score, expected calibration error, Gaussian NLL,
pinball loss, PICP, MPIW, regression calibration error) are calibrax's:
``calibrax.metrics.functional.{calibration,uncertainty,regression}``.

Public surface (`opifex.uncertainty.calibration.temperature`):

- :class:`TemperatureScaling` + :class:`TemperatureScalingState` — Guo et al.
  temperature scaling for multiclass logits.
- :func:`nll_loss_at_temperature` — the objective the calibrator minimises.
"""

from __future__ import annotations

from opifex.uncertainty.calibration.temperature import (
    nll_loss_at_temperature,
    TemperatureScaling,
    TemperatureScalingState,
)


__all__ = [
    "TemperatureScaling",
    "TemperatureScalingState",
    "nll_loss_at_temperature",
]
