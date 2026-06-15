"""Distribution-free risk control: RCPS / Learn-then-Test.

References (canonical formulations):
* Bates, Angelopoulos, Lei, Malik, Jordan 2021 — "Distribution-Free,
  Risk-Controlling Prediction Sets" (RCPS, arXiv:2101.02703). For a
  monotonically non-increasing loss in a threshold ``λ``, the largest
  ``λ`` whose Hoeffding upper confidence bound on empirical loss is at
  most ``α`` controls the population risk at level ``α`` with confidence
  ``1 - δ``.
* Angelopoulos et al. 2022 — "Learn then Test" (LTT, arXiv:2110.01052).
  The general framework: RCPS is the monotonic special case.

The risk-controller threshold-selection kernel is a pure jax.Array →
jax.Array transformation; the package-level helper
:func:`select_threshold_rcps` wraps it with `RiskControlConfig`-aware
metadata recording and returns a Pattern-B `RiskControllerState`
pytree.

For confidence-interval reporting on calibration statistics, we reuse
``calibrax.statistics.analyzer.StatisticalAnalyzer.bootstrap_ci`` via the
:func:`bootstrap_threshold_ci` helper — percentile bootstrap CI semantics
match what we need exactly.
"""

from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from calibrax.statistics.analyzer import StatisticalAnalyzer
from flax import struct

from opifex.uncertainty.types import MetadataItems  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Configuration (Pattern A: plain frozen dataclass, scalar-only)
# ---------------------------------------------------------------------------


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class RiskControlConfig:
    """Static configuration for distribution-free risk control.

    All fields are scalar (no array data), so this is a plain
    ``@dataclasses.dataclass`` passed to jitted kernels via
    ``static_argnames``. Validation happens in ``__post_init__`` and raises
    ``ValueError`` — never ``assert``.

    Args:
        alpha: Target risk level in ``(0, 1)``.
        delta: Confidence-bound failure rate in ``(0, 1)``; the
            population-risk guarantee holds with probability
            ``1 - delta``.
        loss_name: Human-readable identifier for the loss being controlled.
        monotonic: ``True`` when the loss is known to be monotonically
            non-increasing in the threshold; required for the RCPS
            finite-sample coverage guarantee.

    """

    alpha: float
    delta: float
    loss_name: str
    monotonic: bool = True

    def __post_init__(self) -> None:
        """Validate fields. Raises :class:`ValueError` on any out-of-range value."""
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha!r}")
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1); got {self.delta!r}")
        if not self.loss_name:
            raise ValueError("loss_name must be a non-empty string")


# ---------------------------------------------------------------------------
# Fitted state (Pattern B: flax.struct pytree)
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class RiskControllerState:
    """Fitted RCPS / LTT controller state."""

    threshold: jax.Array
    empirical_loss_at_threshold: jax.Array
    upper_confidence_bound: jax.Array
    config: RiskControlConfig = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


# ---------------------------------------------------------------------------
# Hoeffding upper confidence bound
# ---------------------------------------------------------------------------


def hoeffding_upper_bound(*, empirical_mean: jax.Array, n: int, delta: float) -> jax.Array:
    """One-sided Hoeffding upper confidence bound for ``[0, 1]``-bounded losses.

    Returns ``empirical_mean + sqrt(log(1/delta) / (2 n))``. Valid for
    bounded losses; for unbounded losses, scale them into ``[0, 1]`` first
    or use a sub-Gaussian concentration bound.
    """
    return empirical_mean + jnp.sqrt(jnp.log(1.0 / delta) / (2.0 * n))


# ---------------------------------------------------------------------------
# RCPS threshold-selection kernel
# ---------------------------------------------------------------------------


def rcps_threshold_kernel(
    *,
    thresholds: jax.Array,
    losses: jax.Array,
    alpha: float,
    delta: float,
) -> jax.Array:
    """Largest threshold whose Hoeffding UCB on per-threshold mean loss is ``<= alpha``.

    Pure jax.Array → scalar kernel; jit / vmap compatible.

    Args:
        thresholds: 1-D array of candidate thresholds, sorted ascending.
        losses: ``(n_samples, n_thresholds)`` per-sample loss for each
            candidate threshold. Each column is the loss vector at that
            threshold.
        alpha: Target risk level in ``(0, 1)``.
        delta: Confidence-bound failure rate in ``(0, 1)``.

    Returns:
        Scalar threshold value — the largest threshold whose UCB ≤ alpha.
        If no threshold is safe, returns ``thresholds[0]`` (conservative
        fallback) — the metadata-aware wrapper :func:`select_threshold_rcps`
        records this case.

    """
    n_samples = losses.shape[0]
    empirical = jnp.mean(losses, axis=0)  # (n_thresholds,)
    ucb = hoeffding_upper_bound(empirical_mean=empirical, n=n_samples, delta=delta)
    safe = ucb <= alpha
    # If any threshold is safe, return the largest; else return the smallest.
    any_safe = jnp.any(safe)
    # Largest safe index: last True in the mask.
    largest_safe_idx = jnp.where(
        any_safe,
        jnp.argmax(safe[::-1].astype(jnp.int32)),
        0,
    )
    chosen_idx = jnp.where(any_safe, thresholds.shape[0] - 1 - largest_safe_idx, 0)
    return thresholds[chosen_idx]


def select_threshold_rcps(
    *,
    thresholds: jax.Array,
    losses: jax.Array,
    config: RiskControlConfig,
) -> RiskControllerState:
    """Select an RCPS threshold and package result + metadata as a fitted state.

    Args:
        thresholds: 1-D array of candidate thresholds, sorted ascending.
        losses: ``(n_samples, n_thresholds)`` per-sample, per-threshold loss.
        config: Risk-control configuration (alpha, delta, loss_name, monotonic).

    Returns:
        :class:`RiskControllerState` carrying the chosen threshold, the
        empirical loss at it, and the Hoeffding UCB. ``metadata`` records
        the canonical ``method``, ``loss_name``, ``alpha``, ``delta``,
        ``calibration_size``, ``num_thresholds``, ``monotonic``, and
        ``coverage_guarantee`` (``"finite_sample"`` when the loss is
        declared monotonic, ``"conservative"`` otherwise).

    """
    threshold = rcps_threshold_kernel(
        thresholds=thresholds,
        losses=losses,
        alpha=config.alpha,
        delta=config.delta,
    )
    threshold_idx = jnp.argmin(jnp.abs(thresholds - threshold))
    empirical_at_threshold = jnp.mean(losses[:, threshold_idx])
    ucb_at_threshold = hoeffding_upper_bound(
        empirical_mean=empirical_at_threshold,
        n=losses.shape[0],
        delta=config.delta,
    )
    coverage_guarantee = "finite_sample" if config.monotonic else "conservative"
    metadata: MetadataItems = (
        ("method", "rcps"),
        ("loss_name", config.loss_name),
        ("alpha", float(config.alpha)),
        ("delta", float(config.delta)),
        ("calibration_size", int(losses.shape[0])),
        ("num_thresholds", int(thresholds.shape[0])),
        ("monotonic", bool(config.monotonic)),
        ("coverage_guarantee", coverage_guarantee),
    )
    return RiskControllerState(
        threshold=threshold,
        empirical_loss_at_threshold=empirical_at_threshold,
        upper_confidence_bound=ucb_at_threshold,
        config=config,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence interval (CalibraX reuse)
# ---------------------------------------------------------------------------


def bootstrap_threshold_ci(
    *,
    samples: Sequence[float],
    confidence: float = 0.95,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a sequence of samples.

    Thin keyword-only wrapper around
    ``calibrax.statistics.analyzer.StatisticalAnalyzer.bootstrap_ci``.

    Args:
        samples: Sequence of measurement values (e.g., per-bootstrap
            chosen thresholds, per-fold empirical losses).
        confidence: Confidence level in ``(0, 1)``; default 0.95 for a
            95% interval.
        bootstrap_resamples: Number of bootstrap resamples passed to the
            CalibraX analyzer.
        seed: Random seed for reproducible bootstrap sampling.

    Returns:
        ``(lower_bound, upper_bound)`` percentile bootstrap interval.

    """
    analyzer = StatisticalAnalyzer(bootstrap_resamples=bootstrap_resamples, seed=seed)
    return analyzer.bootstrap_ci(samples, confidence=confidence)
