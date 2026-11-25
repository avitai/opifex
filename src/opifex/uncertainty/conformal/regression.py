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
from calibrax.metrics.functional import reverse_kaplan_meier
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


def _warn_underfilled_groups(
    underfilled: list[tuple[int, int]],
    *,
    num_groups: int,
    alpha: float,
    required: int,
) -> None:
    """Emit a ``UserWarning`` naming calibration groups smaller than ``required``."""
    if not underfilled:
        return
    shown = ", ".join(f"{gid} (n={size})" for gid, size in underfilled[:10])
    if len(underfilled) > 10:
        shown += f", ... and {len(underfilled) - 10} more"
    warnings.warn(
        f"alpha={alpha} needs at least {required} calibration scores per "
        f"group; {len(underfilled)} of {num_groups} groups hold "
        f"fewer: {shown}. Their quantile is +inf (unbounded interval). Merge "
        f"groups, raise alpha, or calibrate more points in them.",
        UserWarning,
        stacklevel=4,
    )


def _fit_per_group_quantiles(
    *,
    scores: jax.Array,
    groups: jax.Array,
    alpha: float,
    method: str,
    score_type: str,
) -> tuple[jax.Array, jax.Array, MetadataItems]:
    """Compute per-group conformal quantiles and emit underfilled-group warnings.

    Args:
        scores: 1-D nonconformity scores of shape ``(n,)``.
        groups: 1-D integer group identifiers of shape ``(n,)``.
        alpha: Miscoverage level in ``(0, 1)``.
        method: Method identifier stored in returned metadata.
        score_type: Score identifier stored in returned metadata.

    Returns:
        Tuple ``(quantiles, unique_groups, metadata)``.

    Raises:
        ValueError: When ``alpha`` lies outside ``(0, 1)`` or ``scores`` and
            ``groups`` shapes are empty or mismatched.
    """
    required = minimum_calibration_size(alpha=alpha)
    if not jnp.issubdtype(groups.dtype, jnp.integer):
        raise ValueError(f"groups must have an integer dtype; got {groups.dtype}.")
    if scores.ndim != 1 or scores.shape != groups.shape or scores.shape[0] < 1:
        raise ValueError(
            "Calibration scores and groups must be non-empty 1-D arrays of matching shape; "
            f"got {scores.shape} and {groups.shape}."
        )
    score_dtype = jnp.result_type(scores, jnp.float32)
    scores_f = jnp.asarray(scores, dtype=score_dtype)
    unique_groups = jnp.unique(groups)
    group_id_tuple = tuple(int(gid) for gid in unique_groups.tolist())
    quantiles_list: list[jax.Array] = []
    underfilled: list[tuple[int, int]] = []
    for gid in group_id_tuple:
        group_scores = scores_f[groups == gid]
        if group_scores.shape[0] < required:
            underfilled.append((gid, int(group_scores.shape[0])))
        quantile_value = conformal_quantile(scores=group_scores, alpha=alpha).astype(score_dtype)
        quantiles_list.append(quantile_value)
    quantiles = jnp.stack(quantiles_list)
    _warn_underfilled_groups(
        underfilled,
        num_groups=len(group_id_tuple),
        alpha=alpha,
        required=required,
    )
    metadata: MetadataItems = (
        ("method", method),
        ("score_type", score_type),
        ("alpha", float(alpha)),
        ("calibration_size", int(scores_f.shape[0])),
        ("num_groups", len(group_id_tuple)),
        ("min_group_size", required),
        ("underfilled_groups", tuple(underfilled)),
        # Travels on every PredictionInterval predict returns, so a caller holding only
        # the interval can see which groups were calibrated. num_groups alone cannot
        # distinguish a sample whose group was absent from one that was calibrated.
        ("calibrated_groups", group_id_tuple),
    )
    return quantiles, unique_groups, metadata


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

        Args:
            predictions: Calibration point predictions of shape ``(n,)``.
            targets: Calibration targets of shape ``(n,)``.
            groups: Calibration integer group IDs of shape ``(n,)``.

        Returns:
            Fitted :class:`GroupedSplitConformalState`.
        """
        scores = absolute_residual_score(predictions=predictions, targets=targets)
        quantiles, unique_groups, metadata = _fit_per_group_quantiles(
            scores=scores,
            groups=groups,
            alpha=self.alpha,
            method="grouped_split_conformal",
            score_type="absolute_residual",
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

        Args:
            predictions: Test point predictions of shape ``(m,)``.
            groups: Test integer group IDs of shape ``(m,)``.

        Returns:
            Group-conditional :class:`PredictionInterval`.

        Raises:
            ValueError: When ``groups`` does not have an integer dtype or shapes mismatch.
        """
        state = require_fitted_state(self._state, surface="GroupedSplitConformalRegressor.predict")
        if not jnp.issubdtype(groups.dtype, jnp.integer):
            raise ValueError(f"groups must have an integer dtype; got {groups.dtype}.")
        if predictions.ndim != 1 or predictions.shape != groups.shape:
            raise ValueError(
                "predictions and groups must be 1-D arrays of matching shape; "
                f"got {predictions.shape} and {groups.shape}."
            )
        dtype = jnp.result_type(predictions, state.quantiles, jnp.float32)
        preds = jnp.asarray(predictions, dtype=dtype)
        per_sample_quantile = _lookup_per_group_quantile(
            groups=groups,
            group_ids=state.group_ids,
            quantiles=state.quantiles.astype(dtype),
        )
        return PredictionInterval(
            lower=preds - per_sample_quantile,
            upper=preds + per_sample_quantile,
            coverage=1.0 - state.alpha,
            method="grouped_split_conformal",
            metadata=state.metadata,
        )


@struct.dataclass(slots=True, kw_only=True)
class GroupedCQRState:
    """Per-group CQR quantile adjustments keyed by integer group ID."""

    quantiles: jax.Array  # shape (num_groups,)
    group_ids: jax.Array  # shape (num_groups,) — sorted unique IDs
    alpha: float = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class GroupedConformalizedQuantileRegressor:
    """Mondrian (group-conditional) Conformalized Quantile Regressor.

    Combines CQR conformity scores ``max(lo - y, y - hi)`` (Romano, Patterson &
    Candès 2019, NeurIPS) with Mondrian per-group calibration (Vovk et al. 2005).
    Groups with fewer than
    :func:`~opifex.uncertainty.conformal.scores.minimum_calibration_size`
    calibration points emit a :class:`UserWarning` and receive ``+inf`` quantile
    adjustments; unseen groups at test time likewise yield ``(-inf, +inf)``.
    """

    alpha: float
    _state: GroupedCQRState | None = dc.field(default=None)

    def with_state(self, state: GroupedCQRState) -> GroupedConformalizedQuantileRegressor:
        """Return a fresh regressor carrying ``state`` (immutable update)."""
        return dc.replace(self, _state=state)

    def fit(
        self,
        *,
        lower_predictions: jax.Array,
        upper_predictions: jax.Array,
        targets: jax.Array,
        groups: jax.Array,
    ) -> GroupedCQRState:
        """Fit per-group CQR adjustments from calibration quantile bounds and targets.

        Args:
            lower_predictions: Calibration lower-quantile predictions of shape ``(n,)``.
            upper_predictions: Calibration upper-quantile predictions of shape ``(n,)``.
            targets: Calibration ground-truth targets of shape ``(n,)``.
            groups: Calibration integer group IDs of shape ``(n,)``.

        Returns:
            Fitted :class:`GroupedCQRState`.

        Raises:
            ValueError: When calibration array shapes mismatch.
        """
        if (
            lower_predictions.shape != upper_predictions.shape
            or lower_predictions.shape != targets.shape
        ):
            raise ValueError(
                "lower_predictions, upper_predictions, and targets must share shape; "
                f"got {lower_predictions.shape}, {upper_predictions.shape}, {targets.shape}."
            )
        scores = cqr_score(lower=lower_predictions, upper=upper_predictions, targets=targets)
        quantiles, unique_groups, metadata = _fit_per_group_quantiles(
            scores=scores,
            groups=groups,
            alpha=self.alpha,
            method="grouped_cqr",
            score_type="cqr",
        )
        return GroupedCQRState(
            quantiles=quantiles,
            group_ids=unique_groups,
            alpha=self.alpha,
            metadata=metadata,
        )

    def predict(
        self,
        *,
        lower_predictions: jax.Array,
        upper_predictions: jax.Array,
        groups: jax.Array,
    ) -> PredictionInterval:
        """Return group-calibrated ``[lo - q_g, hi + q_g]`` intervals.

        Args:
            lower_predictions: Test lower-quantile predictions of shape ``(m,)``.
            upper_predictions: Test upper-quantile predictions of shape ``(m,)``.
            groups: Test integer group IDs of shape ``(m,)``.

        Returns:
            Group-conditional :class:`PredictionInterval` with ``method="grouped_cqr"``.

        Raises:
            ValueError: When ``groups`` does not have an integer dtype or
                ``lower_predictions``, ``upper_predictions``, and ``groups`` shapes mismatch.
        """
        state = require_fitted_state(
            self._state, surface="GroupedConformalizedQuantileRegressor.predict"
        )
        if not jnp.issubdtype(groups.dtype, jnp.integer):
            raise ValueError(f"groups must have an integer dtype; got {groups.dtype}.")
        if (
            lower_predictions.ndim != 1
            or lower_predictions.shape != upper_predictions.shape
            or lower_predictions.shape != groups.shape
        ):
            raise ValueError(
                "lower_predictions, upper_predictions, and groups must be 1-D arrays of "
                f"matching shape; got {lower_predictions.shape}, "
                f"{upper_predictions.shape}, {groups.shape}."
            )
        dtype = jnp.result_type(lower_predictions, upper_predictions, state.quantiles, jnp.float32)
        lo = jnp.asarray(lower_predictions, dtype=dtype)
        hi = jnp.asarray(upper_predictions, dtype=dtype)
        per_sample_quantile = _lookup_per_group_quantile(
            groups=groups,
            group_ids=state.group_ids,
            quantiles=state.quantiles.astype(dtype),
        )
        return PredictionInterval(
            lower=lo - per_sample_quantile,
            upper=hi + per_sample_quantile,
            coverage=1.0 - state.alpha,
            method="grouped_cqr",
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
    return jnp.where(
        jnp.any(matches, axis=-1),
        gathered,
        jnp.asarray(jnp.inf, dtype=quantiles.dtype),
    )


# ---------------------------------------------------------------------------
# Right-censored & doubly-robust split conformal (Candès et al. 2023; Sesia & Svetnik 2025)
# ---------------------------------------------------------------------------

_VALID_CENSORED_SCORE_TYPES: frozenset[str] = frozenset(
    {"signed_residual", "cqr_lower", "scaled_signed_residual"}
)


def _validate_survival_probabilities(
    censoring_survival: jax.Array,
    *,
    expected_shape: tuple[int, ...],
) -> None:
    """Validate 0-D or matching 1-D censoring survival shape and concrete ``(0, 1]`` bounds."""
    if censoring_survival.ndim not in (0, 1) or (
        censoring_survival.ndim == 1 and censoring_survival.shape != expected_shape
    ):
        raise ValueError(
            f"censoring_survival must be scalar or 1-D of shape {expected_shape}; "
            f"got shape {censoring_survival.shape}."
        )
    try:
        in_bounds = bool(jnp.all((censoring_survival > 0.0) & (censoring_survival <= 1.0)))
    except jax.errors.TracerBoolConversionError:
        in_bounds = True
    if not in_bounds:
        raise ValueError("censoring_survival probabilities must lie in (0, 1].")


def _validate_censored_hyperparams(
    *,
    alpha: float,
    positivity_floor: float,
    score_type: str,
) -> None:
    """Validate scalar hyperparameters for censored conformal calibration."""
    minimum_calibration_size(alpha=alpha)
    if positivity_floor <= 0.0:
        raise ValueError(f"positivity_floor must be strictly positive; got {positivity_floor!r}.")
    if score_type not in _VALID_CENSORED_SCORE_TYPES:
        raise ValueError(
            f"Unsupported score_type {score_type!r}; choose from "
            f"{sorted(_VALID_CENSORED_SCORE_TYPES)}."
        )


def _validate_optional_1d_match(
    arr: jax.Array | None,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> None:
    """Validate an optional 1-D array against ``expected_shape``."""
    if arr is not None and arr.shape != expected_shape:
        raise ValueError(
            f"{name} must be a 1-D array matching shape {expected_shape}; got shape {arr.shape}."
        )


def _validate_censored_fit_inputs(
    regressor: CensoredSplitConformalRegressor,
    *,
    predictions: jax.Array,
    observed_targets: jax.Array,
    censoring_times: jax.Array | None,
    event_indicators: jax.Array | None,
    censoring_survival: jax.Array | None,
    scales: jax.Array | None,
) -> None:
    """Validate configuration and calibration array shapes for censored conformal fit."""
    _validate_censored_hyperparams(
        alpha=regressor.alpha,
        positivity_floor=regressor.positivity_floor,
        score_type=regressor.score_type,
    )
    if predictions.ndim != 1 or observed_targets.ndim != 1 or predictions.shape[0] < 1:
        raise ValueError("predictions and observed_targets must be non-empty 1-D arrays.")
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
    _validate_optional_1d_match(
        censoring_times, name="censoring_times", expected_shape=predictions.shape
    )
    _validate_optional_1d_match(
        event_indicators, name="event_indicators", expected_shape=predictions.shape
    )
    _validate_optional_1d_match(scales, name="scales", expected_shape=predictions.shape)
    if censoring_survival is not None:
        _validate_survival_probabilities(censoring_survival, expected_shape=predictions.shape)
    if regressor.score_type == "scaled_signed_residual" and scales is None:
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
        default_test_surv = 1.0
    else:
        trunc_targets = jnp.minimum(observed_targets, jnp.asarray(c0, dtype=observed_targets.dtype))
        km_surv = reverse_kaplan_meier(
            observed_targets,
            events_bool,
            eval_times=trunc_targets,
            left_continuous=True,
        )
        surv = jnp.maximum(jnp.asarray(km_surv, dtype=dtype), positivity_floor)
        km_c0 = float(
            reverse_kaplan_meier(
                observed_targets,
                events_bool,
                eval_times=jnp.asarray([c0], dtype=dtype),
                left_continuous=True,
            )[0]
        )
        default_test_surv = max(km_c0, positivity_floor)
    weights = jnp.where(active, 1.0 / surv, 0.0)
    return active, weights, default_test_surv


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
    requires_test_censoring_survival: bool = struct.field(pytree_node=False, default=False)
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
            self,
            predictions=predictions,
            observed_targets=observed_targets,
            censoring_times=censoring_times,
            event_indicators=event_indicators,
            censoring_survival=censoring_survival,
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
        requires_test_surv = censoring_survival is not None
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
            ("requires_test_censoring_survival", requires_test_surv),
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
            requires_test_censoring_survival=requires_test_surv,
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
                test points. Required when ``fit`` was called with explicit
                ``censoring_survival``; otherwise defaults to ``state.default_test_survival``.
            scales: Optional 1-D positive scales when ``score_type='scaled_signed_residual'``.

        Returns:
            :class:`PredictionInterval` with ``upper = +inf`` and ``lower <= state.cutoff``.

        Raises:
            ValueError: When ``score_type='scaled_signed_residual'`` and ``scales`` is omitted,
                or when ``state.requires_test_censoring_survival`` is ``True`` and
                ``censoring_survival`` is omitted.
        """
        state = require_fitted_state(self._state, surface="CensoredSplitConformalRegressor.predict")
        if predictions.ndim != 1:
            raise ValueError(
                f"predictions must be a 1-D test array; got shape {predictions.shape}."
            )
        if state.requires_test_censoring_survival and censoring_survival is None:
            raise ValueError(
                "censoring_survival is required in predict() when fit() was calibrated "
                "with explicit censoring_survival."
            )
        if censoring_survival is not None:
            _validate_survival_probabilities(censoring_survival, expected_shape=predictions.shape)
        if state.score_type == "scaled_signed_residual" and scales is None:
            raise ValueError("scales is required when score_type='scaled_signed_residual'.")
        _validate_optional_1d_match(scales, name="scales", expected_shape=predictions.shape)
        dtype = jnp.result_type(predictions, jnp.float32)
        preds = jnp.asarray(predictions, dtype=dtype)

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
