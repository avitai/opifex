"""Split conformal, conformalized quantile regression, and group-conditional variants.

Three regressors, each with a Pattern-B frozen ``flax.struct.dataclass``
fitted state:

* :class:`SplitConformalRegressor` — point-predictor split conformal
  (Lei et al. 2018). ``fit`` returns :class:`SplitConformalState` with
  ``quantile``. ``predict`` produces ``[ŷ - q, ŷ + q]``.
* :class:`ConformalizedQuantileRegressor` — CQR (Romano, Patterson,
  Candes 2019, arXiv:1905.03222). ``fit`` returns :class:`CQRState`;
  ``predict`` produces ``[lo - q, hi + q]``.
* :class:`GroupedSplitConformalRegressor` — per-group split conformal.
  ``fit`` returns :class:`GroupedSplitConformalState` with one quantile per
  group ID. Each group must hold
  :func:`~opifex.uncertainty.conformal.scores.minimum_calibration_size`
  scores on its own; ``fit`` warns and records the groups that do not, whose
  quantile is ``+inf``.

All ``predict`` methods return :class:`opifex.uncertainty.types.PredictionInterval`
so downstream consumers (`PredictiveDistribution.interval`, coverage
metrics, plotting) operate on a single typed surface.
"""

from __future__ import annotations

import dataclasses as dc
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from opifex.uncertainty.conformal.scores import (
    absolute_residual_score,
    conformal_quantile,
    cqr_score,
    effective_sample_size,
    ipcw_conformal_quantile,
    minimum_calibration_size,
    signed_lower_residual_score,
)
from opifex.uncertainty.types import (
    MetadataItems,
    PredictionInterval,
    require_fitted_state,
)


# ---------------------------------------------------------------------------
# Split conformal (point predictor + |y - ŷ| score)
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class SplitConformalState:
    """Fitted scalar quantile for a split-conformal calibrator."""

    quantile: jax.Array
    alpha: float = struct.field(pytree_node=False)
    score_type: str = struct.field(pytree_node=False, default="absolute_residual")
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class SplitConformalRegressor:
    """Absolute-residual split conformal regressor.

    Usage::

        cp = SplitConformalRegressor(alpha=0.1)
        state = cp.fit(predictions=val_preds, targets=val_targets)
        interval = cp.with_state(state).predict(predictions=test_preds)
    """

    alpha: float
    _state: SplitConformalState | None = dc.field(default=None)

    def with_state(self, state: SplitConformalState) -> SplitConformalRegressor:
        """Return a fresh regressor carrying ``state`` (immutable update)."""
        return dc.replace(self, _state=state)

    def fit(self, *, predictions: jax.Array, targets: jax.Array) -> SplitConformalState:
        """Fit the conformal threshold from calibration ``(predictions, targets)``."""
        scores = absolute_residual_score(predictions=predictions, targets=targets)
        threshold = conformal_quantile(scores=scores, alpha=self.alpha)
        metadata: MetadataItems = (
            ("method", "split_conformal"),
            ("score_type", "absolute_residual"),
            ("alpha", float(self.alpha)),
            ("calibration_size", int(predictions.shape[0])),
        )
        return SplitConformalState(quantile=threshold, alpha=self.alpha, metadata=metadata)

    def predict(self, *, predictions: jax.Array) -> PredictionInterval:
        """Return a :class:`PredictionInterval` of width ``2 * state.quantile``."""
        state = require_fitted_state(self._state, surface="SplitConformalRegressor.predict")
        lower = predictions - state.quantile
        upper = predictions + state.quantile
        return PredictionInterval(
            lower=lower,
            upper=upper,
            coverage=1.0 - state.alpha,
            method="split_conformal",
            metadata=state.metadata,
        )


# ---------------------------------------------------------------------------
# Conformalized Quantile Regression (CQR)
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class CQRState:
    """Fitted scalar adjustment for a CQR calibrator."""

    quantile_adjustment: jax.Array
    alpha: float = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class ConformalizedQuantileRegressor:
    """CQR per Romano, Patterson, Candes 2019."""

    alpha: float
    _state: CQRState | None = dc.field(default=None)

    def with_state(self, state: CQRState) -> ConformalizedQuantileRegressor:
        """Return a fresh regressor carrying ``state`` (immutable update)."""
        return dc.replace(self, _state=state)

    def fit(
        self,
        *,
        lower: jax.Array,
        upper: jax.Array,
        targets: jax.Array,
    ) -> CQRState:
        """Fit the CQR adjustment from calibration quantile bounds and targets."""
        scores = cqr_score(lower=lower, upper=upper, targets=targets)
        adjustment = conformal_quantile(scores=scores, alpha=self.alpha)
        metadata: MetadataItems = (
            ("method", "cqr"),
            ("score_type", "cqr"),
            ("alpha", float(self.alpha)),
            ("calibration_size", int(lower.shape[0])),
        )
        return CQRState(quantile_adjustment=adjustment, alpha=self.alpha, metadata=metadata)

    def predict(self, *, lower: jax.Array, upper: jax.Array) -> PredictionInterval:
        """Return calibrated ``[lo - adj, hi + adj]`` as a :class:`PredictionInterval`."""
        state = require_fitted_state(self._state, surface="ConformalizedQuantileRegressor.predict")
        adj = state.quantile_adjustment
        return PredictionInterval(
            lower=lower - adj,
            upper=upper + adj,
            coverage=1.0 - state.alpha,
            method="cqr",
            metadata=state.metadata,
        )


# ---------------------------------------------------------------------------
# Grouped split conformal (per-group threshold)
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class GroupedSplitConformalState:
    """Per-group quantile thresholds keyed by integer group ID."""

    quantiles: jax.Array  # shape (num_groups,)
    group_ids: jax.Array  # shape (num_groups,) — sorted unique IDs
    alpha: float = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class GroupedSplitConformalRegressor:
    """Split-conformal regressor with per-group thresholds.

    ``fit`` partitions calibration scores by group ID and computes a
    finite-sample-corrected quantile per group; ``predict`` looks the
    quantile up by group ID at evaluation time.
    """

    alpha: float
    _state: GroupedSplitConformalState | None = dc.field(default=None)

    def with_state(self, state: GroupedSplitConformalState) -> GroupedSplitConformalRegressor:
        """Return a fresh regressor carrying ``state`` (immutable update)."""
        return dc.replace(self, _state=state)

    def fit(
        self,
        *,
        predictions: jax.Array,
        targets: jax.Array,
        groups: jax.Array,
    ) -> GroupedSplitConformalState:
        """Fit a per-group conformal threshold from calibration data.

        The ``1 - alpha`` guarantee is per group, so each group needs
        :func:`~opifex.uncertainty.conformal.scores.minimum_calibration_size`
        scores of its own however large the whole calibration set is. Groups
        below it get a ``+inf`` quantile — an interval of unbounded width,
        which is the honest statement that this group's level is unattainable —
        and are named in a :class:`UserWarning` and in
        ``metadata['underfilled_groups']``.
        """
        scores = absolute_residual_score(predictions=predictions, targets=targets)
        unique_groups = jnp.unique(groups)
        required = minimum_calibration_size(alpha=self.alpha)
        quantiles_list = []
        underfilled: list[tuple[int, int]] = []
        for gid in unique_groups.tolist():
            mask = groups == gid
            group_scores = scores[mask]
            if group_scores.shape[0] < required:
                underfilled.append((int(gid), int(group_scores.shape[0])))
            quantile_value = conformal_quantile(scores=group_scores, alpha=self.alpha)
            quantiles_list.append(quantile_value)
        quantiles = jnp.stack(quantiles_list)
        if underfilled:
            shown = ", ".join(f"{gid} (n={size})" for gid, size in underfilled[:10])
            if len(underfilled) > 10:
                shown += f", ... and {len(underfilled) - 10} more"
            warnings.warn(
                f"alpha={self.alpha} needs at least {required} calibration scores per "
                f"group; {len(underfilled)} of {int(unique_groups.shape[0])} groups hold "
                f"fewer: {shown}. Their quantile is +inf (unbounded interval). Merge "
                f"groups, raise alpha, or calibrate more points in them.",
                UserWarning,
                stacklevel=2,
            )
        metadata: MetadataItems = (
            ("method", "grouped_split_conformal"),
            ("score_type", "absolute_residual"),
            ("alpha", float(self.alpha)),
            ("calibration_size", int(predictions.shape[0])),
            ("num_groups", int(unique_groups.shape[0])),
            ("min_group_size", required),
            ("underfilled_groups", tuple(underfilled)),
            # Travels on every PredictionInterval predict returns, so a caller holding only
            # the interval can see which groups were calibrated. num_groups alone cannot
            # distinguish a sample whose group was absent from one that was calibrated.
            ("calibrated_groups", tuple(int(gid) for gid in unique_groups.tolist())),
        )
        return GroupedSplitConformalState(
            quantiles=quantiles,
            group_ids=unique_groups,
            alpha=self.alpha,
            metadata=metadata,
        )

    def predict(self, *, predictions: jax.Array, groups: jax.Array) -> PredictionInterval:
        """Return a :class:`PredictionInterval` with per-sample width set by group.

        A sample whose group was not calibrated gets an unbounded interval rather than a
        zero-width one; ``metadata['calibrated_groups']`` names the groups that were.
        """
        state = require_fitted_state(self._state, surface="GroupedSplitConformalRegressor.predict")
        per_sample_quantile = _lookup_per_group_quantile(
            groups=groups,
            group_ids=state.group_ids,
            quantiles=state.quantiles,
        )
        return PredictionInterval(
            lower=predictions - per_sample_quantile,
            upper=predictions + per_sample_quantile,
            coverage=1.0 - state.alpha,
            method="grouped_split_conformal",
            metadata=state.metadata,
        )


def _lookup_per_group_quantile(
    *, groups: jax.Array, group_ids: jax.Array, quantiles: jax.Array
) -> jax.Array:
    """Vectorised gather of per-sample group quantile; ``+inf`` where the group is unknown.

    ``group_ids`` is assumed sorted; for each entry in ``groups`` we find
    the matching slot in ``group_ids`` and gather the corresponding
    ``quantiles`` value.

    A group absent from ``group_ids`` contributed no calibration scores, so no quantile of
    them covers it at any level: it has no threshold rather than a small one. Its row of
    ``matches`` is all-``False``, and the bare sum would be ``0.0`` -- a zero-width interval,
    indistinguishable from a legitimately tiny quantile and carrying the nominal coverage of
    a calibrated group. Such a row yields ``+inf`` instead, the unbounded interval
    :func:`~opifex.uncertainty.conformal.scores.conformal_quantile` already returns for a
    calibration set too small to carry the level.

    Unlike that function, whose ``n`` and ``alpha`` are static so its branch resolves at
    trace time, this condition is per sample and data-dependent; ``jnp.where`` keeps the
    gather jit- and vmap-compatible.

    The two reference implementations of group-conditional (Mondrian) conformal split on
    this condition, and neither path is fully available here. MAPIE 0.9's
    ``MondrianCP.predict`` refused it eagerly -- ``_check_partition_predict`` rejects "at
    least one new group in the prediction" -- which a traced function cannot do, and that
    class was removed in MAPIE 1.0. crepes returns exactly the zero-width interval described
    above; it is open as ``henrikbostrom/crepes#54``, whose reporter proposes warning and
    falling back to ``[-inf, inf]`` to match that package's own too-small-bin behaviour. The
    unbounded interval is where both of those point, and it is what this repository already
    does for the neighbouring condition.
    """
    # (n_samples, n_groups) one-hot match.
    matches = groups[:, None] == group_ids[None, :]
    gathered = jnp.sum(matches * quantiles[None, :], axis=-1)
    return jnp.where(jnp.any(matches, axis=-1), gathered, jnp.inf)


# ---------------------------------------------------------------------------
# Right-censored & doubly-robust split conformal (Candès et al. 2023; Sesia & Svetnik 2025)
# ---------------------------------------------------------------------------

_VALID_CENSORED_SCORE_TYPES: frozenset[str] = frozenset(
    {"signed_residual", "cqr_lower", "scaled_signed_residual"}
)


def _kaplan_meier_left_continuous_survival(
    observed_times: np.ndarray,
    censoring_events: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """Estimate left-continuous censoring survival ``P(C >= t)`` via Kaplan-Meier."""
    cens_times = observed_times[censoring_events]
    if cens_times.size == 0:
        return np.ones_like(eval_times, dtype=np.float64)

    unique_u = np.unique(cens_times)
    # Risk set R(u_k) = #{j : T_tilde_j >= u_k}; censoring drops d_C(u_k) at u_k.
    at_risk = np.sum(observed_times[:, None] >= unique_u[None, :], axis=0).astype(np.float64)
    drops = np.sum(
        (observed_times[:, None] == unique_u[None, :]) & censoring_events[:, None],
        axis=0,
    ).astype(np.float64)
    step_factors = np.where(at_risk > 0.0, np.maximum(1.0 - drops / at_risk, 0.0), 1.0)
    log_factors = np.log(np.maximum(step_factors, np.finfo(np.float64).tiny))
    cum_log = np.concatenate([[0.0], np.cumsum(log_factors)])
    # Left-continuous product over u_k < t corresponds to searchsorted(..., side="left").
    idx = np.searchsorted(unique_u, eval_times, side="left")
    return np.exp(cum_log[idx])


def _validate_censored_hyperparams(
    *,
    alpha: float,
    positivity_floor: float,
    score_type: str,
) -> None:
    """Validate scalar hyperparameters for censored conformal calibration."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    if positivity_floor <= 0.0:
        raise ValueError(f"positivity_floor must be strictly positive; got {positivity_floor!r}.")
    if score_type not in _VALID_CENSORED_SCORE_TYPES:
        raise ValueError(
            f"Unsupported score_type {score_type!r}; choose from "
            f"{sorted(_VALID_CENSORED_SCORE_TYPES)}."
        )


def _validate_censored_fit_inputs(
    *,
    alpha: float,
    positivity_floor: float,
    score_type: str,
    predictions: jax.Array,
    observed_targets: jax.Array,
    censoring_times: jax.Array | None,
    event_indicators: jax.Array | None,
    scales: jax.Array | None,
) -> None:
    """Validate configuration and calibration array shapes for censored conformal fit."""
    _validate_censored_hyperparams(
        alpha=alpha, positivity_floor=positivity_floor, score_type=score_type
    )
    if predictions.ndim != 1 or observed_targets.ndim != 1:
        raise ValueError("predictions and observed_targets must be 1-D calibration arrays.")
    if predictions.shape != observed_targets.shape:
        raise ValueError(
            f"predictions shape {predictions.shape} must match observed_targets shape "
            f"{observed_targets.shape}."
        )
    if censoring_times is None and event_indicators is None:
        raise ValueError(
            "Supply either censoring_times (Type-I / known censoring thresholds) or "
            "event_indicators (general right-censored)."
        )
    if score_type == "scaled_signed_residual" and scales is None:
        raise ValueError("scales is required when score_type='scaled_signed_residual'.")


def _compute_censored_weights(
    *,
    observed_targets: jax.Array,
    cutoff: float,
    positivity_floor: float,
    censoring_times: jax.Array | None,
    event_indicators: jax.Array | None,
    censoring_survival: jax.Array | None,
) -> tuple[jax.Array, jax.Array, float]:
    """Compute ``(active_mask, calibration_weights, default_test_survival)``."""
    dtype = jnp.result_type(observed_targets, jnp.float32)
    c0 = float(cutoff)

    if censoring_times is not None:
        c_arr = jnp.asarray(censoring_times, dtype=dtype)
        active = c_arr >= c0
        if censoring_survival is not None:
            surv = jnp.maximum(jnp.asarray(censoring_survival, dtype=dtype), positivity_floor)
            default_test_surv = 1.0
        else:
            emp_g0 = max(float(jnp.mean(active.astype(dtype))), positivity_floor)
            surv = jnp.full_like(c_arr, emp_g0)
            default_test_surv = emp_g0
        weights = jnp.where(active, 1.0 / surv, 0.0)
        return active, weights, default_test_surv

    events_bool = jnp.asarray(event_indicators, dtype=jnp.bool_)
    active = events_bool | (observed_targets >= c0)
    if censoring_survival is not None:
        surv = jnp.maximum(jnp.asarray(censoring_survival, dtype=dtype), positivity_floor)
    else:
        obs_np = np.asarray(observed_targets, dtype=np.float64)
        cens_events_np = ~np.asarray(events_bool, dtype=np.bool_)
        trunc_np = np.minimum(obs_np, c0)
        km_surv = _kaplan_meier_left_continuous_survival(obs_np, cens_events_np, trunc_np)
        surv = jnp.maximum(jnp.asarray(km_surv, dtype=dtype), positivity_floor)
    weights = jnp.where(active, 1.0 / surv, 0.0)
    return active, weights, 1.0


@partial(jax.jit, static_argnames=("config",))
def _predict_censored_lower_kernel(
    calibration_scores: jax.Array,
    calibration_weights: jax.Array,
    predictions: jax.Array,
    test_survival: jax.Array,
    scales: jax.Array,
    *,
    config: tuple[float, float, float, str, bool, bool],
) -> jax.Array:
    """JIT-compiled per-test-point lower bound reconstruction."""
    alpha, cutoff, positivity_floor, score_type, doubly_robust, underfilled = config
    dtype = predictions.dtype
    if underfilled:
        return jnp.full_like(predictions, -jnp.inf)

    c0 = jnp.asarray(cutoff, dtype=dtype)
    guarded_surv = jnp.maximum(test_survival, jnp.asarray(positivity_floor, dtype=dtype))
    test_weights = 1.0 / guarded_surv

    q_hat = jax.vmap(
        lambda w_t: ipcw_conformal_quantile(
            scores=calibration_scores,
            calibration_weights=calibration_weights,
            test_weight=w_t,
            alpha=alpha,
        )
    )(test_weights)
    q_eff = jnp.maximum(q_hat, 0.0) if doubly_robust else q_hat

    base_pred = (
        jnp.where(jnp.isfinite(predictions), jnp.minimum(predictions, c0), c0)
        if score_type == "cqr_lower"
        else predictions
    )
    offset = (
        q_eff * jnp.maximum(scales, jnp.asarray(positivity_floor, dtype=dtype))
        if score_type == "scaled_signed_residual"
        else q_eff
    )
    raw_lower = base_pred - offset
    return jnp.minimum(raw_lower, c0)


@struct.dataclass(slots=True, kw_only=True)
class CensoredConformalState:
    """Fitted state for right-censored IPCW / doubly-robust split conformal regression."""

    calibration_scores: jax.Array
    calibration_weights: jax.Array
    cutoff: float = struct.field(pytree_node=False)
    alpha: float = struct.field(pytree_node=False)
    effective_n: float = struct.field(pytree_node=False)
    default_test_survival: float = struct.field(pytree_node=False, default=1.0)
    positivity_floor: float = struct.field(pytree_node=False, default=1e-6)
    score_type: str = struct.field(pytree_node=False, default="signed_residual")
    doubly_robust: bool = struct.field(pytree_node=False, default=False)
    underfilled: bool = struct.field(pytree_node=False, default=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class CensoredSplitConformalRegressor:
    """Right-censored IPCW and doubly-robust split-conformal lower-bound regressor.

    Implements censoring-cutoff filtration and inverse-probability-of-censoring weighted
    conformal prediction (Candès, Lei & Ren 2023, JRSS-B 85(1):24-45; Tibshirani et al.
    2019, NeurIPS Eq. 5) plus general right-censoring Kaplan-Meier IPCW and doubly-robust
    quantile intersection (Sesia & Svetnik 2025, ICML, arXiv:2412.09729). ``predict``
    returns one-sided lower prediction intervals ``[lower, +inf)`` clamped above at
    ``cutoff`` so truncated coverage lifts monotonically to the latent uncensored outcome.
    """

    alpha: float
    cutoff: float
    score_type: str = "signed_residual"
    doubly_robust: bool = False
    positivity_floor: float = 1e-6
    _state: CensoredConformalState | None = dc.field(default=None)

    def with_state(self, state: CensoredConformalState) -> CensoredSplitConformalRegressor:
        """Return a fresh regressor carrying ``state`` (immutable update).

        Args:
            state: Fitted :class:`CensoredConformalState` instance.

        Returns:
            New :class:`CensoredSplitConformalRegressor` bound to ``state``.
        """
        return dc.replace(self, _state=state)

    def fit(
        self,
        *,
        predictions: jax.Array,
        observed_targets: jax.Array,
        censoring_times: jax.Array | None = None,
        event_indicators: jax.Array | None = None,
        censoring_survival: jax.Array | None = None,
        scales: jax.Array | None = None,
    ) -> CensoredConformalState:
        """Calibrate IPCW conformal scores and weights from right-censored data.

        Args:
            predictions: 1-D calibration predictions (mean or ``alpha``-quantile).
            observed_targets: 1-D observed outcomes ``min(T_i, C_i)``.
            censoring_times: Optional 1-D censoring thresholds ``C_i`` (Regime I).
            event_indicators: Optional 1-D boolean/0-1 event indicators ``1{T_i <= C_i}``
                used when ``censoring_times`` is partially unobserved (Regime II).
            censoring_survival: Optional 1-D estimated survival probabilities
                ``G_hat(c_0 | X_i)`` at calibration covariates.
            scales: Optional 1-D positive uncertainty scales for
                ``score_type='scaled_signed_residual'``.

        Returns:
            Fitted :class:`CensoredConformalState` PyTree.
        """
        _validate_censored_fit_inputs(
            alpha=self.alpha,
            positivity_floor=self.positivity_floor,
            score_type=self.score_type,
            predictions=predictions,
            observed_targets=observed_targets,
            censoring_times=censoring_times,
            event_indicators=event_indicators,
            scales=scales,
        )
        raw_scores = signed_lower_residual_score(
            predictions=predictions,
            targets=observed_targets,
            cutoff=self.cutoff,
            scales=scales if self.score_type == "scaled_signed_residual" else None,
            truncate_predictions=(self.score_type == "cqr_lower"),
            scale_floor=self.positivity_floor,
        )
        active, weights, default_test_surv = _compute_censored_weights(
            observed_targets=observed_targets,
            cutoff=self.cutoff,
            positivity_floor=self.positivity_floor,
            censoring_times=censoring_times,
            event_indicators=event_indicators,
            censoring_survival=censoring_survival,
        )
        scores = jnp.where(active, raw_scores, jnp.asarray(jnp.inf, dtype=raw_scores.dtype))
        eff_n = float(effective_sample_size(weights))
        required = minimum_calibration_size(alpha=self.alpha)
        underfilled = eff_n < float(required)
        if underfilled:
            warnings.warn(
                f"Censored calibration effective sample size n_eff={eff_n:.2f} is below "
                f"minimum_calibration_size(alpha={self.alpha})={required} at cutoff={self.cutoff}; "
                "predict() will emit unbounded (-inf, +inf) intervals.",
                UserWarning,
                stacklevel=2,
            )
        regime = "type1_cutoff" if censoring_times is not None else "right_censored_ipcw"
        metadata: MetadataItems = (
            ("method", "censored_split_conformal"),
            ("regime", regime),
            ("score_type", self.score_type),
            ("alpha", float(self.alpha)),
            ("cutoff", float(self.cutoff)),
            ("calibration_size", int(predictions.shape[0])),
            ("active_count", int(jnp.sum(active))),
            ("effective_sample_size", eff_n),
            ("minimum_calibration_size", required),
            ("doubly_robust", bool(self.doubly_robust)),
            ("underfilled", underfilled),
        )
        return CensoredConformalState(
            calibration_scores=scores,
            calibration_weights=weights,
            cutoff=float(self.cutoff),
            alpha=float(self.alpha),
            effective_n=eff_n,
            default_test_survival=default_test_surv,
            positivity_floor=float(self.positivity_floor),
            score_type=self.score_type,
            doubly_robust=bool(self.doubly_robust),
            underfilled=underfilled,
            metadata=metadata,
        )

    def predict(
        self,
        *,
        predictions: jax.Array,
        censoring_survival: jax.Array | None = None,
        scales: jax.Array | None = None,
    ) -> PredictionInterval:
        """Predict one-sided ``[lower, +inf)`` conformal intervals for test inputs.

        Args:
            predictions: 1-D test predictions of shape ``(n_test,)``.
            censoring_survival: Optional 1-D censoring survival ``G_hat(c_0 | X_test)`` at
                test points. Defaults to ``state.default_test_survival``.
            scales: Optional 1-D positive scales when ``score_type='scaled_signed_residual'``.

        Returns:
            :class:`PredictionInterval` with ``upper = +inf`` and ``lower <= state.cutoff``.

        Raises:
            ValueError: When ``score_type='scaled_signed_residual'`` and ``scales`` is omitted.
        """
        state = require_fitted_state(self._state, surface="CensoredSplitConformalRegressor.predict")
        dtype = jnp.result_type(predictions, jnp.float32)
        preds = jnp.asarray(predictions, dtype=dtype)
        if state.score_type == "scaled_signed_residual" and scales is None:
            raise ValueError("scales is required when score_type='scaled_signed_residual'.")

        test_surv = (
            jnp.broadcast_to(jnp.asarray(censoring_survival, dtype=dtype), preds.shape)
            if censoring_survival is not None
            else jnp.full_like(preds, state.default_test_survival)
        )
        scale_arr = (
            jnp.broadcast_to(jnp.asarray(scales, dtype=dtype), preds.shape)
            if scales is not None
            else jnp.ones_like(preds)
        )
        lower = _predict_censored_lower_kernel(
            state.calibration_scores,
            state.calibration_weights,
            preds,
            test_surv,
            scale_arr,
            config=(
                state.alpha,
                state.cutoff,
                state.positivity_floor,
                state.score_type,
                state.doubly_robust,
                state.underfilled,
            ),
        )
        return PredictionInterval(
            lower=lower,
            upper=jnp.full_like(lower, jnp.inf),
            coverage=1.0 - state.alpha,
            method="censored_split_conformal",
            metadata=state.metadata,
        )
