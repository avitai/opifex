"""Surrogate-model uncertainty primitives (Task 6.6).

Currently a single helper:

* :func:`decompose_surrogate_uncertainty` — combine independent
  prediction / residual / calibration variance components into a single
  total uncertainty.

The PCE primitives live under :mod:`opifex.uncertainty.scientific.polynomial_chaos`
(Task 6.6 explicitly forbids an intermediate ``surrogate/pce.py``).
"""

from opifex.uncertainty.surrogate.surrogate_uncertainty import (
    decompose_surrogate_uncertainty,
    SurrogateUncertaintyResult,
)


__all__ = [
    "SurrogateUncertaintyResult",
    "decompose_surrogate_uncertainty",
]
