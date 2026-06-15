"""Conformal prediction subsystem.

* Score helpers (`scores`): :func:`absolute_residual_score`,
  :func:`cqr_score`, :func:`conformal_quantile`.
* Generic value object (`base`): :class:`ConformalScore`.
* Regression calibrators (`regression`):
  :class:`SplitConformalRegressor` / :class:`SplitConformalState`,
  :class:`ConformalizedQuantileRegressor` / :class:`CQRState`,
  :class:`GroupedSplitConformalRegressor` / :class:`GroupedSplitConformalState`.
* Classification scores and calibrator (`classification`):
  :func:`lac_score`, :func:`aps_score`, :func:`raps_score`,
  :func:`aps_prediction_set`,
  :class:`LACConformalClassifier` / :class:`LACConformalState`.
* Cross-conformal / weighted conformal (`advanced`):
  :func:`jackknife_plus_intervals`, :func:`cv_plus_intervals`,
  :func:`weighted_conformal_quantile`,
  :func:`weighted_split_conformal_intervals`.
* Time-series conformal (`time_series`):
  :class:`EnbPIState` + :func:`enbpi_update` / :func:`enbpi_predict`,
  :class:`AdaptiveConformalState` + :func:`aci_update` /
  :func:`aci_metadata`.
* Field / function-space conformal (`fields`):
  :func:`field_l2_score`, :func:`field_linf_score`, :func:`field_h1_score`,
  :class:`FieldSplitConformalRegressor` / :class:`FieldSplitConformalState`.
* Risk control (`risk_control`):
  :class:`RiskControlConfig`, :class:`RiskControllerState`,
  :func:`hoeffding_upper_bound`, :func:`rcps_threshold_kernel`,
  :func:`select_threshold_rcps`, :func:`bootstrap_threshold_ci`.
* Exchangeability diagnostic (`exchangeability`):
  :class:`ExchangeabilityReport`, :func:`check_exchangeability`,
  :func:`ks_two_sample_pvalue`.
"""

from __future__ import annotations

from opifex.uncertainty.conformal.advanced import (
    cv_plus_intervals,
    jackknife_plus_intervals,
    weighted_conformal_quantile,
    weighted_split_conformal_intervals,
)
from opifex.uncertainty.conformal.base import ConformalScore
from opifex.uncertainty.conformal.classification import (
    aps_prediction_set,
    aps_score,
    lac_score,
    LACConformalClassifier,
    LACConformalState,
    raps_score,
)
from opifex.uncertainty.conformal.exchangeability import (
    check_exchangeability,
    ExchangeabilityReport,
    ks_two_sample_pvalue,
)
from opifex.uncertainty.conformal.fields import (
    field_h1_score,
    field_l2_score,
    field_linf_score,
    FieldSplitConformalRegressor,
    FieldSplitConformalState,
)
from opifex.uncertainty.conformal.regression import (
    ConformalizedQuantileRegressor,
    CQRState,
    GroupedSplitConformalRegressor,
    GroupedSplitConformalState,
    SplitConformalRegressor,
    SplitConformalState,
)
from opifex.uncertainty.conformal.risk_control import (
    bootstrap_threshold_ci,
    hoeffding_upper_bound,
    rcps_threshold_kernel,
    RiskControlConfig,
    RiskControllerState,
    select_threshold_rcps,
)
from opifex.uncertainty.conformal.scores import (
    absolute_residual_score,
    conformal_quantile,
    cqr_score,
)
from opifex.uncertainty.conformal.time_series import (
    aci_metadata,
    aci_update,
    AdaptiveConformalState,
    enbpi_predict,
    enbpi_update,
    EnbPIState,
)


__all__ = [
    "AdaptiveConformalState",
    "CQRState",
    "ConformalScore",
    "ConformalizedQuantileRegressor",
    "EnbPIState",
    "ExchangeabilityReport",
    "FieldSplitConformalRegressor",
    "FieldSplitConformalState",
    "GroupedSplitConformalRegressor",
    "GroupedSplitConformalState",
    "LACConformalClassifier",
    "LACConformalState",
    "RiskControlConfig",
    "RiskControllerState",
    "SplitConformalRegressor",
    "SplitConformalState",
    "absolute_residual_score",
    "aci_metadata",
    "aci_update",
    "aps_prediction_set",
    "aps_score",
    "bootstrap_threshold_ci",
    "check_exchangeability",
    "conformal_quantile",
    "cqr_score",
    "cv_plus_intervals",
    "enbpi_predict",
    "enbpi_update",
    "field_h1_score",
    "field_l2_score",
    "field_linf_score",
    "hoeffding_upper_bound",
    "jackknife_plus_intervals",
    "ks_two_sample_pvalue",
    "lac_score",
    "raps_score",
    "rcps_threshold_kernel",
    "select_threshold_rcps",
    "weighted_conformal_quantile",
    "weighted_split_conformal_intervals",
]
