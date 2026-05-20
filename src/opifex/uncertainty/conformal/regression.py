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
  group ID.

All ``predict`` methods return :class:`opifex.uncertainty.types.PredictionInterval`
so downstream consumers (`PredictiveDistribution.interval`, coverage
metrics, plotting) operate on a single typed surface.
"""

from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct

from opifex.uncertainty.conformal.scores import (
    absolute_residual_score,
    conformal_quantile,
    cqr_score,
)
from opifex.uncertainty.types import PredictionInterval


if TYPE_CHECKING:
    from opifex.uncertainty.types import MetadataItems


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
        state = self._state
        if state is None:
            raise RuntimeError(
                "SplitConformalRegressor.predict called before fit; "
                "call fit(...) first or .with_state(state)."
            )
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
        state = self._state
        if state is None:
            raise RuntimeError(
                "ConformalizedQuantileRegressor.predict called before fit; "
                "call fit(...) first or .with_state(state)."
            )
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
        """Fit a per-group conformal threshold from calibration data."""
        scores = absolute_residual_score(predictions=predictions, targets=targets)
        unique_groups = jnp.unique(groups)
        quantiles_list = []
        for gid in unique_groups.tolist():
            mask = groups == gid
            group_scores = scores[mask]
            quantile_value = conformal_quantile(scores=group_scores, alpha=self.alpha)
            quantiles_list.append(quantile_value)
        quantiles = jnp.stack(quantiles_list)
        metadata: MetadataItems = (
            ("method", "grouped_split_conformal"),
            ("score_type", "absolute_residual"),
            ("alpha", float(self.alpha)),
            ("calibration_size", int(predictions.shape[0])),
            ("num_groups", int(unique_groups.shape[0])),
        )
        return GroupedSplitConformalState(
            quantiles=quantiles,
            group_ids=unique_groups,
            alpha=self.alpha,
            metadata=metadata,
        )

    def predict(self, *, predictions: jax.Array, groups: jax.Array) -> PredictionInterval:
        """Return a :class:`PredictionInterval` with per-sample width set by group."""
        state = self._state
        if state is None:
            raise RuntimeError(
                "GroupedSplitConformalRegressor.predict called before fit; "
                "call fit(...) first or .with_state(state)."
            )
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
    """Vectorised gather of per-sample group quantile.

    ``group_ids`` is assumed sorted; for each entry in ``groups`` we find
    the matching slot in ``group_ids`` and gather the corresponding
    ``quantiles`` value.
    """
    # (n_samples, n_groups) one-hot match.
    matches = groups[:, None] == group_ids[None, :]
    return jnp.sum(matches * quantiles[None, :], axis=-1)
