"""Cross-conformal, Jackknife+, and weighted conformal (covariate shift).

References (canonical):
* Barber, Candes, Ramdas, Tibshirani 2021 — "Predictive Inference with the
  Jackknife+" (arXiv:1905.02928). J+ guarantees coverage >= 1 - 2*alpha;
  CV+ is the K-fold variant.
* Tibshirani, Foygel Barber, Candes, Ramdas 2019 — "Conformal Prediction
  Under Covariate Shift" (arXiv:1904.06019). Weighted quantile of
  calibration scores against likelihood-ratio weights.

Returned intervals are :class:`opifex.uncertainty.types.PredictionInterval`
value objects.
"""

from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
from calibrax.statistics import conformal_plus_ranks
from flax import struct

from opifex.uncertainty.conformal.scores import cqr_score, ipcw_conformal_quantile
from opifex.uncertainty.types import (
    MetadataItems,
    PredictionInterval,
    require_fitted_state,
)


# ---------------------------------------------------------------------------
# Jackknife+ / CV+
# ---------------------------------------------------------------------------


def _plus_order_statistic_bounds(
    lower_candidates: jax.Array,
    upper_candidates: jax.Array,
    *,
    alpha: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute Barber et al. (2021) finite-sample order-statistic interval bounds."""
    n = lower_candidates.shape[0]
    dtype = jnp.result_type(lower_candidates, upper_candidates, jnp.float32)
    lo_cand = jnp.asarray(lower_candidates, dtype=dtype)
    hi_cand = jnp.asarray(upper_candidates, dtype=dtype)
    out_shape = lo_cand.shape[1:]

    k_lo, k_hi = conformal_plus_ranks(n, alpha)
    k_lo = min(k_lo, n)

    lower = (
        jnp.full(out_shape, -jnp.inf, dtype=dtype)
        if k_lo < 1
        else jnp.sort(lo_cand, axis=0)[k_lo - 1]
    )
    upper = (
        jnp.full(out_shape, jnp.inf, dtype=dtype)
        if k_hi > n
        else jnp.sort(hi_cand, axis=0)[k_hi - 1]
    )
    return lower, upper


def jackknife_plus_intervals(
    *,
    loo_val_predictions: jax.Array,
    loo_val_targets: jax.Array,
    loo_test_predictions: jax.Array,
    alpha: float,
) -> PredictionInterval:
    """Jackknife+ prediction intervals per Barber et al. 2021.

    Uses exact finite-sample order statistics ``floor(alpha * (n + 1))`` and
    ``ceil((1 - alpha) * (n + 1))`` over the ``n`` leave-one-out candidate bounds
    (Barber, Candès, Ramdas & Tibshirani 2021, Annals of Statistics Eqs. 2.2 & 3.2),
    emitting ``(-inf, +inf)`` when ``n`` cannot attain ``alpha``.

    Args:
        loo_val_predictions: 1-D array of shape ``(n_calib,)``; the i-th
            entry is the prediction of the leave-one-out model trained
            without sample i, evaluated at sample i.
        loo_val_targets: 1-D array of shape ``(n_calib,)``; ground-truth
            for each LOO held-out sample.
        loo_test_predictions: 2-D array of shape ``(n_calib, n_test)``;
            entry ``[i, j]`` is the prediction of the model that left out
            sample i, evaluated at test input j.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        :class:`PredictionInterval` with ``method="jackknife_plus"`` and
        metadata recording ``method``, ``alpha``, ``loo_size``,
        ``coverage_guarantee="1-2*alpha"``.

    Raises:
        ValueError: When ``alpha`` is outside ``(0, 1)`` or LOO array dimensions mismatch.
    """
    if loo_val_predictions.ndim != 1 or loo_val_targets.ndim != 1 or loo_test_predictions.ndim < 2:
        raise ValueError(
            "loo_val_predictions and loo_val_targets must be 1-D and loo_test_predictions "
            f"must have ndim >= 2; got shapes {loo_val_predictions.shape}, "
            f"{loo_val_targets.shape}, {loo_test_predictions.shape}."
        )
    if loo_val_predictions.shape[0] < 1:
        raise ValueError("loo_val_predictions must contain at least 1 calibration sample.")
    if loo_val_predictions.shape[0] != loo_val_targets.shape[0]:
        raise ValueError(
            "loo_val_predictions and loo_val_targets must share leading dim; "
            f"got {loo_val_predictions.shape[0]} vs {loo_val_targets.shape[0]}."
        )
    if loo_test_predictions.shape[0] != loo_val_predictions.shape[0]:
        raise ValueError(
            "loo_test_predictions leading dim must equal n_calib; "
            f"got {loo_test_predictions.shape[0]} vs {loo_val_predictions.shape[0]}."
        )
    dtype = jnp.result_type(loo_val_predictions, loo_val_targets, loo_test_predictions, jnp.float32)
    val_preds = jnp.asarray(loo_val_predictions, dtype=dtype)
    val_targets = jnp.asarray(loo_val_targets, dtype=dtype)
    test_preds = jnp.asarray(loo_test_predictions, dtype=dtype)
    residuals = jnp.abs(val_targets - val_preds).reshape(
        (val_preds.shape[0], *([1] * (test_preds.ndim - 1)))
    )
    lower_candidates = test_preds - residuals
    upper_candidates = test_preds + residuals
    lower, upper = _plus_order_statistic_bounds(lower_candidates, upper_candidates, alpha=alpha)
    metadata: MetadataItems = (
        ("method", "jackknife_plus"),
        ("alpha", float(alpha)),
        ("loo_size", int(loo_val_predictions.shape[0])),
        ("coverage_guarantee", "1-2*alpha"),
        ("assumption_status", "exchangeable_assumed"),
    )
    return PredictionInterval(
        lower=lower,
        upper=upper,
        coverage=1.0 - alpha,
        method="jackknife_plus",
        metadata=metadata,
    )


def cv_plus_intervals(
    *,
    fold_val_predictions: jax.Array,
    fold_val_targets: jax.Array,
    fold_test_predictions: jax.Array,
    alpha: float,
) -> PredictionInterval:
    """CV+ prediction intervals per Barber et al. 2021 (K-fold extension of J+).

    Uses exact finite-sample order statistics ``floor(alpha * (n + 1))`` and
    ``ceil((1 - alpha) * (n + 1))`` across all ``n = n_folds * n_per_fold`` out-of-fold
    candidates, emitting ``(-inf, +inf)`` when ``n`` cannot attain ``alpha``.

    Args:
        fold_val_predictions: ``(n_folds, n_per_fold)`` per-fold held-out
            predictions.
        fold_val_targets: ``(n_folds, n_per_fold)`` per-fold held-out targets.
        fold_test_predictions: ``(n_folds, n_test)`` per-fold predictions
            on the test set.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        :class:`PredictionInterval` with ``method="cv_plus"`` and metadata
        recording ``method``, ``alpha``, ``num_folds``,
        ``calibration_size_per_fold``, ``coverage_guarantee="1-2*alpha"``.

    Raises:
        ValueError: When ``alpha`` is outside ``(0, 1)`` or fold array shapes mismatch.
    """
    if (
        fold_val_predictions.ndim != 2
        or fold_val_predictions.shape != fold_val_targets.shape
        or fold_test_predictions.ndim < 2
        or fold_val_predictions.shape[0] != fold_test_predictions.shape[0]
    ):
        raise ValueError(
            "fold shapes must match across fold_val_predictions, fold_val_targets, and "
            f"fold_test_predictions; got {fold_val_predictions.shape}, "
            f"{fold_val_targets.shape}, {fold_test_predictions.shape}."
        )
    n_folds, n_per_fold = fold_val_predictions.shape
    if n_folds * n_per_fold < 1:
        raise ValueError("fold_val_predictions must contain at least 1 calibration sample.")
    dtype = jnp.result_type(
        fold_val_predictions, fold_val_targets, fold_test_predictions, jnp.float32
    )
    val_preds = jnp.asarray(fold_val_predictions, dtype=dtype)
    val_targets = jnp.asarray(fold_val_targets, dtype=dtype)
    test_preds = jnp.asarray(fold_test_predictions, dtype=dtype)
    residuals = jnp.abs((val_targets - val_preds).reshape((-1, *([1] * (test_preds.ndim - 1)))))
    test_preds_repeated = jnp.repeat(test_preds, n_per_fold, axis=0)
    lower_candidates = test_preds_repeated - residuals
    upper_candidates = test_preds_repeated + residuals
    lower, upper = _plus_order_statistic_bounds(lower_candidates, upper_candidates, alpha=alpha)
    metadata: MetadataItems = (
        ("method", "cv_plus"),
        ("alpha", float(alpha)),
        ("num_folds", int(n_folds)),
        ("calibration_size_per_fold", int(n_per_fold)),
        ("coverage_guarantee", "1-2*alpha"),
        ("assumption_status", "exchangeable_assumed"),
    )
    return PredictionInterval(
        lower=lower,
        upper=upper,
        coverage=1.0 - alpha,
        method="cv_plus",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Weighted conformal (covariate shift)
# ---------------------------------------------------------------------------


def weighted_conformal_quantile(
    *,
    scores: jax.Array,
    weights: jax.Array,
    test_weight: jax.Array | float,
    alpha: float,
) -> jax.Array:
    """Weighted step-CDF ``(1 - alpha)``-quantile per Tibshirani et al. 2019.

    Implements Eq. (5) of Tibshirani, Foygel Barber, Candès & Ramdas (2019,
    NeurIPS, arXiv:1904.06019) by placing the test-point likelihood-ratio weight
    ``test_weight`` at ``+inf``, mapping zero-weight calibration entries to
    ``+inf``, and returning ``+inf`` whenever calibration probability mass cannot
    reach ``1 - alpha``.

    Args:
        scores: 1-D array of nonconformity scores of shape ``(n,)``.
        weights: 1-D array of non-negative calibration likelihood-ratio weights
            of shape ``(n,)``.
        test_weight: Non-negative scalar test-point likelihood-ratio weight
            ``w_{n+1}(x)`` placed at ``+inf``.
        alpha: Static miscoverage level in ``(0, 1)`` (validated via
            :func:`ipcw_conformal_quantile`, raising ``ValueError`` outside
            ``(0, 1)`` and ``TypeError`` if traced).

    Returns:
        Scalar weighted quantile threshold (finite score, ``+inf``, or ``NaN``).
    """
    return ipcw_conformal_quantile(
        scores=scores,
        calibration_weights=weights,
        test_weight=test_weight,
        alpha=alpha,
    )


def weighted_split_conformal_intervals(
    *,
    calibration_scores: jax.Array,
    calibration_weights: jax.Array,
    predictions: jax.Array,
    test_weights: jax.Array | float,
    alpha: float,
) -> PredictionInterval:
    """Weighted split-conformal intervals using Tibshirani et al. (2019) quantiles.

    Args:
        calibration_scores: 1-D array of calibration nonconformity scores.
        calibration_weights: 1-D array of non-negative calibration weights.
        predictions: Test-point predictions of shape ``(n_test,)``.
        test_weights: Scalar test weight or 1-D per-test-point weight array of
            shape ``predictions.shape`` placed at ``+inf``.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        :class:`PredictionInterval` with bounds ``predictions ± threshold`` and
        metadata recording ``method="weighted_split_conformal"`` and
        ``assumption_status="weights_required"``.

    Raises:
        ValueError: When ``alpha`` lies outside ``(0, 1)`` or ``test_weights``
            cannot match ``predictions``.
    """
    if predictions.ndim != 1:
        raise ValueError(f"predictions must be a 1-D array; got shape {predictions.shape}.")
    dtype = jnp.result_type(
        calibration_scores, calibration_weights, predictions, test_weights, jnp.float32
    )
    preds = jnp.asarray(predictions, dtype=dtype)
    w_test_arr = jnp.asarray(test_weights, dtype=dtype)
    if w_test_arr.ndim == 0:
        threshold = weighted_conformal_quantile(
            scores=calibration_scores,
            weights=calibration_weights,
            alpha=alpha,
            test_weight=w_test_arr,
        )
    elif w_test_arr.ndim == 1 and w_test_arr.shape == preds.shape:
        threshold = jax.vmap(
            lambda w_t: weighted_conformal_quantile(
                scores=calibration_scores,
                weights=calibration_weights,
                alpha=alpha,
                test_weight=w_t,
            )
        )(w_test_arr)
    else:
        raise ValueError(
            f"test_weights must be a scalar or 1-D array matching predictions shape "
            f"{preds.shape}; got shape {w_test_arr.shape}."
        )
    metadata: MetadataItems = (
        ("method", "weighted_split_conformal"),
        ("score_type", "absolute_residual"),
        ("alpha", float(alpha)),
        ("calibration_size", int(calibration_scores.shape[0])),
        ("assumption_status", "weights_required"),
    )
    return PredictionInterval(
        lower=preds - threshold,
        upper=preds + threshold,
        coverage=1.0 - alpha,
        method="weighted_split_conformal",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# K-fold cross-conformalized quantile regression (CV+ CQR)
# ---------------------------------------------------------------------------


def _validate_cv_plus_cqr_fit(
    fold_val_lower: jax.Array,
    fold_val_upper: jax.Array,
    fold_val_targets: jax.Array,
) -> tuple[int, int]:
    """Validate out-of-fold CQR calibration arrays and return ``(n_folds, n_per_fold)``.

    Args:
        fold_val_lower: Out-of-fold lower quantile predictions ``(n_folds, n_per_fold)``.
        fold_val_upper: Out-of-fold upper quantile predictions ``(n_folds, n_per_fold)``.
        fold_val_targets: Out-of-fold targets ``(n_folds, n_per_fold)``.

    Returns:
        ``(n_folds, n_per_fold)``.

    Raises:
        ValueError: When fold shapes are invalid.
    """
    if (
        fold_val_lower.ndim != 2
        or fold_val_lower.shape != fold_val_upper.shape
        or fold_val_lower.shape != fold_val_targets.shape
    ):
        raise ValueError(
            "fold_val_lower, fold_val_upper, and fold_val_targets must be 2-D arrays of "
            f"matching shape; got {fold_val_lower.shape}, {fold_val_upper.shape}, "
            f"{fold_val_targets.shape}."
        )
    n_folds, n_per_fold = fold_val_lower.shape
    if n_folds * n_per_fold < 1:
        raise ValueError("Out-of-fold calibration arrays must contain at least 1 sample.")
    return n_folds, n_per_fold


def _validate_cv_plus_cqr_predict(
    fold_test_lower: jax.Array,
    fold_test_upper: jax.Array,
    *,
    num_folds: int,
) -> None:
    """Validate per-fold test quantile prediction arrays.

    Args:
        fold_test_lower: Per-fold lower quantile test predictions ``(n_folds, n_test)``.
        fold_test_upper: Per-fold upper quantile test predictions ``(n_folds, n_test)``.
        num_folds: Expected number of folds ``n_folds``.

    Raises:
        ValueError: When test fold prediction shapes mismatch.
    """
    if (
        fold_test_lower.ndim != 2
        or fold_test_lower.shape != fold_test_upper.shape
        or fold_test_lower.shape[0] != num_folds
    ):
        raise ValueError(
            f"fold_test_lower and fold_test_upper must be 2-D arrays with leading dim "
            f"{num_folds}; got {fold_test_lower.shape} and {fold_test_upper.shape}."
        )


def _cv_plus_cqr_bounds(
    conformity_scores: jax.Array,
    fold_test_lower: jax.Array,
    fold_test_upper: jax.Array,
    *,
    alpha: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute Barber et al. (2021) CV+ order-statistic bounds from OOF CQR scores."""
    n_per_fold = conformity_scores.shape[1]
    dtype = jnp.result_type(conformity_scores, fold_test_lower, fold_test_upper, jnp.float32)
    scores_flat = jnp.asarray(conformity_scores, dtype=dtype).reshape((-1,))
    test_lo = jnp.asarray(fold_test_lower, dtype=dtype)
    test_hi = jnp.asarray(fold_test_upper, dtype=dtype)
    lower_candidates = jnp.repeat(test_lo, n_per_fold, axis=0) - scores_flat[:, None]
    upper_candidates = jnp.repeat(test_hi, n_per_fold, axis=0) + scores_flat[:, None]
    return _plus_order_statistic_bounds(lower_candidates, upper_candidates, alpha=alpha)


def cv_plus_cqr_intervals(
    *,
    fold_val_lower: jax.Array,
    fold_val_upper: jax.Array,
    fold_val_targets: jax.Array,
    fold_test_lower: jax.Array,
    fold_test_upper: jax.Array,
    alpha: float,
) -> PredictionInterval:
    """K-fold cross-conformalized quantile regression (CV+ CQR) intervals.

    Combines CQR out-of-fold conformity scores ``max(lo - y, y - hi)`` (Romano,
    Patterson & Candès 2019) with the exact finite-sample CV+ order statistics
    ``floor(alpha * (n + 1))`` and ``ceil((1 - alpha) * (n + 1))`` of Barber,
    Candès, Ramdas & Tibshirani (2021).

    Args:
        fold_val_lower: ``(n_folds, n_per_fold)`` out-of-fold lower quantiles.
        fold_val_upper: ``(n_folds, n_per_fold)`` out-of-fold upper quantiles.
        fold_val_targets: ``(n_folds, n_per_fold)`` out-of-fold targets.
        fold_test_lower: ``(n_folds, n_test)`` per-fold test lower quantiles.
        fold_test_upper: ``(n_folds, n_test)`` per-fold test upper quantiles.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        :class:`PredictionInterval` with ``method="cv_plus_cqr"`` and
        ``coverage_guarantee="1-2*alpha"``.
    """
    n_folds, n_per_fold = _validate_cv_plus_cqr_fit(
        fold_val_lower, fold_val_upper, fold_val_targets
    )
    _validate_cv_plus_cqr_predict(fold_test_lower, fold_test_upper, num_folds=n_folds)
    dtype = jnp.result_type(
        fold_val_lower,
        fold_val_upper,
        fold_val_targets,
        fold_test_lower,
        fold_test_upper,
        jnp.float32,
    )
    scores = cqr_score(
        lower=jnp.asarray(fold_val_lower, dtype=dtype),
        upper=jnp.asarray(fold_val_upper, dtype=dtype),
        targets=jnp.asarray(fold_val_targets, dtype=dtype),
    )
    lower, upper = _cv_plus_cqr_bounds(scores, fold_test_lower, fold_test_upper, alpha=alpha)
    metadata: MetadataItems = (
        ("method", "cv_plus_cqr"),
        ("score_type", "cqr"),
        ("alpha", float(alpha)),
        ("num_folds", int(n_folds)),
        ("calibration_size_per_fold", int(n_per_fold)),
        ("coverage_guarantee", "1-2*alpha"),
        ("assumption_status", "exchangeable_assumed"),
    )
    return PredictionInterval(
        lower=lower,
        upper=upper,
        coverage=1.0 - alpha,
        method="cv_plus_cqr",
        metadata=metadata,
    )


@struct.dataclass(slots=True, kw_only=True)
class CrossCQRState:
    """Fitted out-of-fold CQR conformity scores for K-fold cross-CQR."""

    conformity_scores: jax.Array  # shape (n_folds, n_per_fold)
    alpha: float = struct.field(pytree_node=False)
    num_folds: int = struct.field(pytree_node=False)
    calibration_size_per_fold: int = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class CrossConformalizedQuantileRegressor:
    """Stateful K-fold cross-conformalized quantile regressor (CV+ CQR)."""

    alpha: float
    _state: CrossCQRState | None = dc.field(default=None)

    def with_state(self, state: CrossCQRState) -> CrossConformalizedQuantileRegressor:
        """Return a fresh regressor carrying ``state`` (immutable update)."""
        return dc.replace(self, _state=state)

    def fit(
        self,
        *,
        fold_val_lower: jax.Array,
        fold_val_upper: jax.Array,
        fold_val_targets: jax.Array,
    ) -> CrossCQRState:
        """Compute and store out-of-fold CQR conformity scores across ``K`` folds.

        Args:
            fold_val_lower: ``(n_folds, n_per_fold)`` out-of-fold lower quantiles.
            fold_val_upper: ``(n_folds, n_per_fold)`` out-of-fold upper quantiles.
            fold_val_targets: ``(n_folds, n_per_fold)`` out-of-fold targets.

        Returns:
            Fitted :class:`CrossCQRState`.
        """
        n_folds, n_per_fold = _validate_cv_plus_cqr_fit(
            fold_val_lower, fold_val_upper, fold_val_targets
        )
        conformal_plus_ranks(n_folds * n_per_fold, self.alpha)
        dtype = jnp.result_type(fold_val_lower, fold_val_upper, fold_val_targets, jnp.float32)
        scores = cqr_score(
            lower=jnp.asarray(fold_val_lower, dtype=dtype),
            upper=jnp.asarray(fold_val_upper, dtype=dtype),
            targets=jnp.asarray(fold_val_targets, dtype=dtype),
        )
        metadata: MetadataItems = (
            ("method", "cross_cqr"),
            ("score_type", "cqr"),
            ("alpha", float(self.alpha)),
            ("num_folds", int(n_folds)),
            ("calibration_size_per_fold", int(n_per_fold)),
            ("coverage_guarantee", "1-2*alpha"),
            ("assumption_status", "exchangeable_assumed"),
        )
        return CrossCQRState(
            conformity_scores=scores,
            alpha=self.alpha,
            num_folds=int(n_folds),
            calibration_size_per_fold=int(n_per_fold),
            metadata=metadata,
        )

    def predict(
        self,
        *,
        fold_test_lower: jax.Array,
        fold_test_upper: jax.Array,
    ) -> PredictionInterval:
        """Aggregate per-fold test quantile bounds with CV+ order statistics.

        Args:
            fold_test_lower: ``(n_folds, n_test)`` per-fold test lower quantiles.
            fold_test_upper: ``(n_folds, n_test)`` per-fold test upper quantiles.

        Returns:
            :class:`PredictionInterval` with ``method="cross_cqr"``.
        """
        state = require_fitted_state(
            self._state, surface="CrossConformalizedQuantileRegressor.predict"
        )
        _validate_cv_plus_cqr_predict(fold_test_lower, fold_test_upper, num_folds=state.num_folds)
        lower, upper = _cv_plus_cqr_bounds(
            state.conformity_scores,
            fold_test_lower,
            fold_test_upper,
            alpha=state.alpha,
        )
        return PredictionInterval(
            lower=lower,
            upper=upper,
            coverage=1.0 - state.alpha,
            method="cross_cqr",
            metadata=state.metadata,
        )


# ---------------------------------------------------------------------------
# Feedback covariate shift for adaptive design (Fannjiang et al., PNAS 2022)
# ---------------------------------------------------------------------------


def _assemble_factorized_log_ratios(
    log_calibration_ratios: jax.Array | None,
    log_test_ratios: jax.Array | None,
) -> jax.Array:
    """Validate factorized 1-D log-ratios and broadcast to ``(n_test, n_calib + 1)``.

    Args:
        log_calibration_ratios: Optional ``(n_calib,)`` calibration log-ratios.
        log_test_ratios: Optional ``(n_test,)`` test log-ratios.

    Returns:
        Broadcast log-ratio matrix of shape ``(n_test, n_calib + 1)``.

    Raises:
        ValueError: When either array is missing, empty, or not 1-D.
    """
    if log_calibration_ratios is None or log_test_ratios is None:
        raise ValueError(
            "Provide either swapped_log_ratios or both log_calibration_ratios and log_test_ratios."
        )
    if (
        log_calibration_ratios.ndim != 1
        or log_calibration_ratios.shape[0] < 1
        or log_test_ratios.ndim != 1
    ):
        raise ValueError(
            "log_calibration_ratios (n_calib >= 1) and log_test_ratios (n_test,) must "
            f"be 1-D arrays; got {log_calibration_ratios.shape} and {log_test_ratios.shape}."
        )
    dtype = jnp.result_type(log_calibration_ratios, log_test_ratios, jnp.float32)
    cal_lr = jnp.asarray(log_calibration_ratios, dtype=dtype)
    test_lr = jnp.asarray(log_test_ratios, dtype=dtype)
    n_test = test_lr.shape[0]
    cal_broadcast = jnp.broadcast_to(cal_lr[None, :], (n_test, cal_lr.shape[0]))
    return jnp.concatenate([cal_broadcast, test_lr[:, None]], axis=-1)


def _resolve_feedback_log_ratio_matrix(
    *,
    log_calibration_ratios: jax.Array | None,
    log_test_ratios: jax.Array | None,
    swapped_log_ratios: jax.Array | None,
) -> tuple[jax.Array, str]:
    """Validate and assemble the ``(n_test, n_calib + 1)`` log-likelihood-ratio matrix.

    Args:
        log_calibration_ratios: Optional ``(n_calib,)`` log-ratios for fixed policies.
        log_test_ratios: Optional ``(n_test,)`` test log-ratios for fixed policies.
        swapped_log_ratios: Optional ``(n_test, n_calib + 1)`` swapped-policy log-ratio
            matrix per Fannjiang et al. (2022, Theorem 1).

    Returns:
        Tuple ``(row_log_ratios, policy_mode)``.

    Raises:
        ValueError: When both ratio specifications are supplied or ``swapped_log_ratios``
            has invalid shape.
    """
    if swapped_log_ratios is not None:
        if log_calibration_ratios is not None or log_test_ratios is not None:
            raise ValueError(
                "Pass either swapped_log_ratios or (log_calibration_ratios, log_test_ratios), "
                "not both."
            )
        if swapped_log_ratios.ndim != 2 or swapped_log_ratios.shape[1] < 2:
            raise ValueError(
                "swapped_log_ratios must be 2-D of shape (n_test, n_calib + 1) with "
                f"n_calib >= 1; got shape {swapped_log_ratios.shape}."
            )
        dtype = jnp.result_type(swapped_log_ratios, jnp.float32)
        return jnp.asarray(swapped_log_ratios, dtype=dtype), "swapped"

    matrix = _assemble_factorized_log_ratios(log_calibration_ratios, log_test_ratios)
    return matrix, "factorized"


def feedback_covariate_shift_weights(
    *,
    log_calibration_ratios: jax.Array | None = None,
    log_test_ratios: jax.Array | None = None,
    swapped_log_ratios: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute overflow-safe normalized weights under feedback covariate shift.

    Implements the log-space weight stabilization for conformal prediction under
    feedback covariate shift (Fannjiang, Bates, Angelopoulos, Listgarten & Jordan,
    PNAS 2022, 10.1073/pnas.2204569119, Theorem 1). Shifting each
    test row by ``max(row_log_ratios)`` before exponentiation prevents overflow and
    ``0 / 0`` underflow under high-inverse-temperature Boltzmann or acquisition
    design policies.

    Args:
        log_calibration_ratios: Optional ``(n_calib,)`` log-density ratios
            ``log q(X_i) - log p(X_i)`` when the design policy does not depend on
            swapping test candidate ``X_{n+1}``.
        log_test_ratios: Optional ``(n_test,)`` log-density ratios
            ``log q(X_{n+1}) - log p(X_{n+1})`` paired with ``log_calibration_ratios``.
        swapped_log_ratios: Optional ``(n_test, n_calib + 1)`` matrix for
            calibration-dependent design policies (Theorem 1), where entry
            ``[j, i]`` (``i < n_calib``) is the log-ratio at ``X_i`` under the
            dataset swapping ``X_i`` with ``X_{n+1}^{(j)}``, and ``[j, n_calib]``
            is the log-ratio at ``X_{n+1}^{(j)}``.

    Returns:
        ``(calibration_weights, test_weights)`` of shapes ``(n_test, n_calib)`` and
        ``(n_test,)``, normalized so each non-degenerate row sums to ``1``.
    """
    row_log_ratios, _ = _resolve_feedback_log_ratio_matrix(
        log_calibration_ratios=log_calibration_ratios,
        log_test_ratios=log_test_ratios,
        swapped_log_ratios=swapped_log_ratios,
    )
    dtype = row_log_ratios.dtype
    # Guard rows containing NaN or +inf: NaN rows propagate NaN, +inf indicators select
    # +inf atoms, and all -inf rows map cleanly to zero weights.
    has_nan = jnp.any(jnp.isnan(row_log_ratios), axis=-1, keepdims=True)
    has_posinf = jnp.any(jnp.isposinf(row_log_ratios), axis=-1, keepdims=True)
    finite_masked = jnp.where(
        jnp.isposinf(row_log_ratios) | jnp.isnan(row_log_ratios), -jnp.inf, row_log_ratios
    )
    row_max = jnp.max(finite_masked, axis=-1, keepdims=True)
    safe_max = jnp.where(jnp.isfinite(row_max), row_max, jnp.asarray(0.0, dtype=dtype))
    unnorm = jnp.where(
        has_posinf,
        jnp.isposinf(row_log_ratios).astype(dtype),
        jnp.exp(row_log_ratios - safe_max),
    )
    row_sum = jnp.sum(unnorm, axis=-1, keepdims=True)
    safe_sum = jnp.where(row_sum > 0.0, row_sum, jnp.asarray(1.0, dtype=dtype))
    normalized = jnp.where(row_sum > 0.0, unnorm / safe_sum, jnp.zeros_like(unnorm))
    normalized = jnp.where(has_nan, jnp.full_like(normalized, jnp.nan), normalized)
    return normalized[:, :-1], normalized[:, -1]


def feedback_covariate_shift_intervals(
    *,
    calibration_scores: jax.Array,
    predictions: jax.Array,
    alpha: float,
    log_calibration_ratios: jax.Array | None = None,
    log_test_ratios: jax.Array | None = None,
    swapped_log_ratios: jax.Array | None = None,
) -> PredictionInterval:
    """Prediction intervals under feedback covariate shift (Fannjiang et al. 2022).

    Args:
        calibration_scores: 1-D calibration nonconformity scores ``(n_calib,)``.
        predictions: 1-D test point predictions ``(n_test,)``.
        alpha: Miscoverage level in ``(0, 1)``.
        log_calibration_ratios: Optional ``(n_calib,)`` fixed-policy log-ratios.
        log_test_ratios: Optional ``(n_test,)`` fixed-policy test log-ratios.
        swapped_log_ratios: Optional ``(n_test, n_calib + 1)`` swapped-policy
            log-ratio matrix (Fannjiang et al. 2022, Theorem 1).

    Returns:
        :class:`PredictionInterval` with ``method="feedback_covariate_shift"``.

    Raises:
        ValueError: When ``alpha`` lies outside ``(0, 1)`` or score/prediction/ratio
            shapes mismatch.
    """
    if calibration_scores.ndim != 1 or predictions.ndim != 1:
        raise ValueError(
            "calibration_scores and predictions must be 1-D arrays; got shapes "
            f"{calibration_scores.shape} and {predictions.shape}."
        )
    _, policy_mode = _resolve_feedback_log_ratio_matrix(
        log_calibration_ratios=log_calibration_ratios,
        log_test_ratios=log_test_ratios,
        swapped_log_ratios=swapped_log_ratios,
    )
    cal_weights, test_weights = feedback_covariate_shift_weights(
        log_calibration_ratios=log_calibration_ratios,
        log_test_ratios=log_test_ratios,
        swapped_log_ratios=swapped_log_ratios,
    )
    expected_shape = (predictions.shape[0], calibration_scores.shape[0])
    if cal_weights.shape != expected_shape:
        raise ValueError(
            f"Log-ratio matrix shape {cal_weights.shape} must match "
            f"(n_test, n_calib) = {expected_shape}."
        )
    dtype = jnp.result_type(calibration_scores, predictions, cal_weights, test_weights, jnp.float32)
    scores_f = jnp.asarray(calibration_scores, dtype=dtype)
    preds_f = jnp.asarray(predictions, dtype=dtype)
    thresholds = jax.vmap(
        lambda w_cal, w_test: ipcw_conformal_quantile(
            scores=scores_f,
            calibration_weights=w_cal,
            test_weight=w_test,
            alpha=alpha,
        )
    )(cal_weights.astype(dtype), test_weights.astype(dtype))
    metadata: MetadataItems = (
        ("method", "feedback_covariate_shift"),
        ("score_type", "absolute_residual"),
        ("alpha", float(alpha)),
        ("calibration_size", int(calibration_scores.shape[0])),
        ("policy_mode", policy_mode),
        ("assumption_status", "feedback_covariate_shift_weights_required"),
    )
    return PredictionInterval(
        lower=preds_f - thresholds,
        upper=preds_f + thresholds,
        coverage=1.0 - alpha,
        method="feedback_covariate_shift",
        metadata=metadata,
    )


__all__ = [
    "CrossCQRState",
    "CrossConformalizedQuantileRegressor",
    "cv_plus_cqr_intervals",
    "cv_plus_intervals",
    "feedback_covariate_shift_intervals",
    "feedback_covariate_shift_weights",
    "jackknife_plus_intervals",
    "weighted_conformal_quantile",
    "weighted_split_conformal_intervals",
]
