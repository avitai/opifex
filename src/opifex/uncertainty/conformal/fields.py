"""Field / function-space conformal prediction for PDE solution fields.

Score functions use an explicit norm over caller-specified spatial axes:

* :func:`field_l2_score` — ``sqrt(mean((y - ŷ)^2))`` over the spatial axes.
* :func:`field_linf_score` — ``max(|y - ŷ|)`` over the spatial axes.
* :func:`field_h1_score` — ``L2(field)`` plus ``L2(finite-difference grad)``.

The :class:`FieldSplitConformalRegressor` exposes the standard
``with_state(...) / fit(...) / predict(...)`` ergonomics and returns
:class:`opifex.uncertainty.types.PredictionInterval` so downstream code
consumes a single typed surface.

Field metadata is sourced from :class:`opifex.uncertainty.scientific.fields.FieldMetadata`
(the canonical Pattern-A schema).
"""

from __future__ import annotations

import dataclasses as dc
from typing import Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct

from opifex.uncertainty.conformal.scores import conformal_quantile
from opifex.uncertainty.scientific.fields import FieldMetadata
from opifex.uncertainty.types import PredictionInterval


if TYPE_CHECKING:
    from opifex.uncertainty.conformal.exchangeability import ExchangeabilityReport
    from opifex.uncertainty.types import MetadataItems


FieldNorm = Literal["L2", "Linf", "H1"]
_VALID_NORMS: frozenset[str] = frozenset({"L2", "Linf", "H1"})


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------


def field_l2_score(
    *,
    predictions: jax.Array,
    targets: jax.Array,
    spatial_axes: tuple[int, ...],
) -> jax.Array:
    """L2 field score: ``sqrt(mean((y - ŷ)^2))`` over the spatial axes."""
    residual = targets - predictions
    return jnp.sqrt(jnp.mean(residual * residual, axis=spatial_axes))


def field_linf_score(
    *,
    predictions: jax.Array,
    targets: jax.Array,
    spatial_axes: tuple[int, ...],
) -> jax.Array:
    """Linf field score: ``max(|y - ŷ|)`` over the spatial axes."""
    return jnp.max(jnp.abs(targets - predictions), axis=spatial_axes)


def field_h1_score(
    *,
    predictions: jax.Array,
    targets: jax.Array,
    spatial_axes: tuple[int, ...],
) -> jax.Array:
    """H1 field score: ``sqrt(L2(field)^2 + L2(grad)^2)`` over the spatial axes.

    Uses finite-difference gradients along each spatial axis.
    """
    residual = targets - predictions
    l2_sq = jnp.mean(residual * residual, axis=spatial_axes)
    gradient_sq_sum = jnp.zeros_like(l2_sq)
    for axis in spatial_axes:
        grad = jnp.diff(residual, axis=axis)
        # Mean of squared finite difference over the same spatial axes (gradient
        # has one fewer element along ``axis``, but mean is dimensionless).
        gradient_sq_sum = gradient_sq_sum + jnp.mean(grad * grad, axis=spatial_axes)
    return jnp.sqrt(l2_sq + gradient_sq_sum)


# ---------------------------------------------------------------------------
# Field split-conformal calibrator
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class FieldSplitConformalState:
    """Fitted scalar threshold for a field split-conformal calibrator."""

    threshold: jax.Array
    alpha: float = struct.field(pytree_node=False)
    norm: str = struct.field(pytree_node=False)
    spatial_axes: tuple[int, ...] = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class FieldSplitConformalRegressor:
    """Split-conformal regressor for field outputs (PDE solutions).

    Usage::

        cp = FieldSplitConformalRegressor(alpha=0.1, norm="L2",
                                           spatial_axes=(-2, -1))
        state = cp.fit(predictions=val_preds, targets=val_targets)
        interval = cp.with_state(state).predict(predictions=test_preds)
    """

    alpha: float
    norm: FieldNorm
    spatial_axes: tuple[int, ...]
    _state: FieldSplitConformalState | None = dc.field(default=None)

    def __post_init__(self) -> None:
        """Validate the requested norm against the supported set."""
        if self.norm not in _VALID_NORMS:
            raise ValueError(
                f"Unknown field norm {self.norm!r}; valid choices: {sorted(_VALID_NORMS)}"
            )

    def with_state(self, state: FieldSplitConformalState) -> FieldSplitConformalRegressor:
        """Return a fresh regressor carrying ``state`` (immutable update)."""
        return dc.replace(self, _state=state)

    def fit(
        self,
        *,
        predictions: jax.Array,
        targets: jax.Array,
        exchangeability_report: ExchangeabilityReport | None = None,
    ) -> FieldSplitConformalState:
        """Fit the field conformal threshold and package metadata.

        Args:
            predictions: ``(batch, *spatial)`` calibration predictions.
            targets: ``(batch, *spatial)`` calibration targets.
            exchangeability_report: Optional report from
                :func:`opifex.uncertainty.conformal.check_exchangeability`. If
                provided and ``passes is False``, the fitted state records
                ``assumption_status="exchangeability_failed"`` so downstream
                consumers do not overclaim distribution-free coverage.
        """
        score_fn = _score_fn_for_norm(self.norm)
        scores = score_fn(predictions=predictions, targets=targets, spatial_axes=self.spatial_axes)
        threshold = conformal_quantile(scores=scores, alpha=self.alpha)
        if exchangeability_report is None or bool(exchangeability_report.passes):
            assumption_status = "exchangeable_assumed"
        else:
            assumption_status = "exchangeability_failed"
        field_md = FieldMetadata(
            grid_axes=(),
            time_axis=None,
            spatial_axes=self.spatial_axes,
            norm=self.norm,
            alpha=self.alpha,
            calibration_size=int(predictions.shape[0]),
            assumption_status=assumption_status,
        )
        metadata: MetadataItems = (
            ("method", "field_split_conformal"),
            ("norm", field_md.norm),
            ("spatial_axes", field_md.spatial_axes),
            ("alpha", field_md.alpha),
            ("calibration_size", field_md.calibration_size),
            ("assumption_status", field_md.assumption_status),
        )
        return FieldSplitConformalState(
            threshold=threshold,
            alpha=self.alpha,
            norm=self.norm,
            spatial_axes=self.spatial_axes,
            metadata=metadata,
        )

    def predict(self, *, predictions: jax.Array) -> PredictionInterval:
        """Return a :class:`PredictionInterval` of width ``2 * state.threshold``.

        Bounds are broadcast scalar over the prediction field — the
        chosen norm collapses to a per-sample scalar threshold during
        calibration.
        """
        state = self._state
        if state is None:
            raise RuntimeError(
                "FieldSplitConformalRegressor.predict called before fit; "
                "call fit(...) first or .with_state(state)."
            )
        return PredictionInterval(
            lower=predictions - state.threshold,
            upper=predictions + state.threshold,
            coverage=1.0 - state.alpha,
            method="field_split_conformal",
            metadata=state.metadata,
        )


def _score_fn_for_norm(norm: str):
    """Map a norm identifier to its scoring kernel."""
    if norm == "L2":
        return field_l2_score
    if norm == "Linf":
        return field_linf_score
    if norm == "H1":
        return field_h1_score
    raise ValueError(f"Unknown norm: {norm!r}")
