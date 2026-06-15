"""Empirical failure probability with Wilson binomial confidence interval.

Operates on a 1-D array of boolean / ``{0, 1}`` failure indicators
produced by the caller (e.g. ``g(x_i) <= 0`` of a limit-state function
``g`` evaluated at Monte Carlo samples ``x_i``).

Reference: Wilson, E. B. (1927), "Probable inference, the law of
succession, and statistical inference", JASA 22(158), 209-212. The
Wilson interval is the canonical small-sample-friendly binomial
proportion confidence interval used throughout OpenTURNS / SciPy.

This module intentionally keeps the implementation simple — advanced
FORM / SORM / subset-simulation adapters are not yet active per the
Task 6.5 implementation requirements.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp


@dataclass(frozen=True, slots=True, kw_only=True)
class ReliabilityResult:
    """Container for empirical failure probability + binomial CI.

    Attributes:
        probability: Empirical failure probability ``p_f = sum / N``.
        confidence_interval: ``(lower, upper)`` Wilson binomial CI at
            the requested confidence level.
        confidence_level: Confidence level of the interval (e.g. 0.95).
        num_samples: Total sample count ``N``.
        num_failures: Sum of failure indicators.
    """

    probability: jax.Array
    confidence_interval: tuple[jax.Array, jax.Array]
    confidence_level: float
    num_samples: int
    num_failures: jax.Array


def failure_probability(
    indicators: jax.Array,
    *,
    confidence_level: float = 0.95,
) -> ReliabilityResult:
    """Estimate the empirical failure probability and its Wilson CI.

    Args:
        indicators: 1-D array of failure indicators (``1.0`` for
            failure, ``0.0`` otherwise). Anything non-zero counts as
            a failure.
        confidence_level: Two-sided Wilson interval confidence
            (must lie strictly in ``(0, 1)``).

    Returns:
        A :class:`ReliabilityResult` carrying the point estimate +
        Wilson interval.

    Raises:
        ValueError: If ``indicators`` is empty, or
            ``confidence_level`` is not strictly in ``(0, 1)``.
    """
    if indicators.shape[0] == 0:
        raise ValueError("Cannot estimate failure probability from empty indicator array.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie strictly in (0, 1); got {confidence_level}.")

    n = indicators.shape[0]
    num_failures = jnp.sum((indicators != 0.0).astype(jnp.float32))
    probability = num_failures / n

    # Wilson interval at requested confidence level.
    # z = Phi^{-1}(1 - alpha/2).
    alpha = 1.0 - confidence_level
    z = jsp.stats.norm.ppf(1.0 - alpha / 2.0)
    denom = 1.0 + z**2 / n
    centre = (probability + z**2 / (2.0 * n)) / denom
    half_width = z * jnp.sqrt(probability * (1.0 - probability) / n + z**2 / (4.0 * n**2)) / denom
    lower = jnp.clip(centre - half_width, 0.0, 1.0)
    upper = jnp.clip(centre + half_width, 0.0, 1.0)

    return ReliabilityResult(
        probability=probability,
        confidence_interval=(lower, upper),
        confidence_level=confidence_level,
        num_samples=n,
        num_failures=num_failures,
    )


__all__ = ["ReliabilityResult", "failure_probability"]
