"""OOD detection scores + residual-shift diagnostics.

Module surface:

* :mod:`opifex.uncertainty.ood.scores` —
  :func:`max_softmax_probability` (Hendrycks & Gimpel 2017),
  :func:`fpr95`.
* :mod:`opifex.uncertainty.ood.shift_diagnostics` —
  :class:`ShiftReport`, :func:`residual_shift_diagnostic`.

AUROC / AUPRC have no shim here — callers import them directly from
``calibrax.metrics.functional.classification``. Predictive entropy /
mutual information OOD scores live in :mod:`opifex.uncertainty.metrics`.
"""

from __future__ import annotations

from opifex.uncertainty.ood.scores import fpr95, max_softmax_probability
from opifex.uncertainty.ood.shift_diagnostics import (
    residual_shift_diagnostic,
    ShiftReport,
)


__all__ = [
    "ShiftReport",
    "fpr95",
    "max_softmax_probability",
    "residual_shift_diagnostic",
]
