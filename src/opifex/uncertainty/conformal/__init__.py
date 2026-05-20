"""Conformal prediction subsystem.

* Score helpers (`scores`): :func:`absolute_residual_score`,
  :func:`cqr_score`, :func:`conformal_quantile`.
* Generic value object (`base`): :class:`ConformalScore`.
* Regression calibrators (`regression`):
  :class:`SplitConformalRegressor` / :class:`SplitConformalState`,
  :class:`ConformalizedQuantileRegressor` / :class:`CQRState`,
  :class:`GroupedSplitConformalRegressor` / :class:`GroupedSplitConformalState`.
* Exchangeability diagnostic (`exchangeability`):
  :class:`ExchangeabilityReport`, :func:`check_exchangeability`,
  :func:`ks_two_sample_pvalue`.
"""

from __future__ import annotations

from opifex.uncertainty.conformal.base import ConformalScore
from opifex.uncertainty.conformal.exchangeability import (
    check_exchangeability,
    ExchangeabilityReport,
    ks_two_sample_pvalue,
)
from opifex.uncertainty.conformal.regression import (
    ConformalizedQuantileRegressor,
    CQRState,
    GroupedSplitConformalRegressor,
    GroupedSplitConformalState,
    SplitConformalRegressor,
    SplitConformalState,
)
from opifex.uncertainty.conformal.scores import (
    absolute_residual_score,
    conformal_quantile,
    cqr_score,
)


__all__ = [
    "CQRState",
    "ConformalScore",
    "ConformalizedQuantileRegressor",
    "ExchangeabilityReport",
    "GroupedSplitConformalRegressor",
    "GroupedSplitConformalState",
    "SplitConformalRegressor",
    "SplitConformalState",
    "absolute_residual_score",
    "check_exchangeability",
    "conformal_quantile",
    "cqr_score",
    "ks_two_sample_pvalue",
]
