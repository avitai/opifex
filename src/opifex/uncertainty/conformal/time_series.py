"""Time-series conformal: EnbPI and Adaptive Conformal Inference.

References (canonical):
* Xu, Xie 2021 — "Conformal Prediction Interval for Dynamic Time-Series"
  (EnbPI, arXiv:2010.09107). Sliding residual window updated online
  without retraining; coverage is an approximate marginal guarantee.
* Gibbs, Candès 2021 — "Adaptive Conformal Inference Under Distribution
  Shift" (arXiv:2106.00170). Online ``alpha`` update via a stochastic
  recursion that achieves long-run marginal coverage regardless of
  exchangeability.

State containers follow Pattern B (``flax.struct.dataclass(slots=True,
kw_only=True)``); the update kernels are pure pytree-in / pytree-out
transformations and trace cleanly under ``jax.jit``.
"""

from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
from flax import struct

from opifex.uncertainty.types import (
    MetadataItems,
    PredictionInterval,
)


# ---------------------------------------------------------------------------
# EnbPI: sliding residual window
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class EnbPIState:
    """Fitted state for an EnbPI predictor.

    ``residual_window`` is a fixed-size 1-D rolling buffer; ``enbpi_update``
    returns a fresh state with the oldest element dropped and the new
    residual appended.
    """

    residual_window: jax.Array
    alpha: float = struct.field(pytree_node=False)


def enbpi_update(*, state: EnbPIState, new_residual: jax.Array) -> EnbPIState:
    """Slide the residual window: drop the oldest entry, append ``new_residual``."""
    new_window = jnp.concatenate([state.residual_window[1:], new_residual[None]])
    return dc.replace(state, residual_window=new_window)


def enbpi_predict(*, state: EnbPIState, predictions: jax.Array) -> PredictionInterval:
    """Compute an EnbPI prediction interval ``predictions ± quantile(|residuals|, 1 - alpha)``.

    Matches Xu, Xie 2021 (EnbPI, arXiv:2010.09107, Algorithm 1, line 8):
    ``β̂ = ⌈(1-α)(T+1)⌉/T``-th smallest ``|ε̂_t|``. The ``'higher'`` rule
    on ``jnp.quantile`` matches the canonical Angelopoulos conformal
    interpolation choice used in
    ``aangelopoulos/conformal-prediction``.
    """
    threshold = jnp.quantile(jnp.abs(state.residual_window), 1.0 - state.alpha, method="higher")
    metadata: MetadataItems = (
        ("method", "enbpi"),
        ("alpha", float(state.alpha)),
        ("window_size", int(state.residual_window.shape[0])),
        ("assumption_status", "approximate_marginal"),
    )
    return PredictionInterval(
        lower=predictions - threshold,
        upper=predictions + threshold,
        coverage=1.0 - state.alpha,
        method="enbpi",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Adaptive Conformal Inference (ACI)
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class AdaptiveConformalState:
    r"""Running state for Gibbs-Candes Adaptive Conformal Inference.

    ``current_alpha`` is the live miscoverage level driving the prediction
    interval width; it drifts up when observations land outside the
    interval and down when they land inside, following the recursion
    ``alpha_{t+1} = alpha_t + lr * (target_alpha - 1_{y_t \notin C_t})``.
    """

    current_alpha: jax.Array
    target_alpha: float = struct.field(pytree_node=False)
    learning_rate: float = struct.field(pytree_node=False)


def aci_update(
    *, state: AdaptiveConformalState, was_covered: jax.Array | bool
) -> AdaptiveConformalState:
    """One ACI update step (Gibbs & Candès 2021, arXiv:2106.00170).

    Canonical recursion::

        alpha_{t+1} = alpha_t + lr * (target_alpha - 1{y_t not in C_t})

    where the indicator is 1 when *uncovered*. Equivalent to Fortuna's
    ``fortuna.conformal.regression.adaptive_conformal_regressor.update_error``
    formula ``error += gamma * (target_error - 1 + is_in)``.

    Args:
        state: Current ACI state.
        was_covered: ``True`` when the latest observation fell inside the
            interval; ``False`` otherwise. Accepts either a Python ``bool``
            or a 0-d boolean ``jax.Array`` (for jit-compatibility).

    Returns:
        Fresh :class:`AdaptiveConformalState` with the updated
        ``current_alpha``.

    """
    covered_int = jnp.asarray(was_covered, dtype=jnp.float32)
    uncovered_indicator = 1.0 - covered_int
    update_step = state.learning_rate * (state.target_alpha - uncovered_indicator)
    return dc.replace(state, current_alpha=state.current_alpha + update_step)


def aci_metadata(*, state: AdaptiveConformalState) -> MetadataItems:
    """Return ACI metadata for downstream consumers.

    Records the canonical assumption status (``long_run_marginal``) — ACI
    does NOT claim finite-sample coverage; the guarantee is asymptotic.
    """
    return (
        ("method", "adaptive_conformal_inference"),
        ("target_alpha", float(state.target_alpha)),
        ("learning_rate", float(state.learning_rate)),
        ("current_alpha", float(state.current_alpha)),
        ("assumption_status", "long_run_marginal"),
    )
