"""PIT histogram, ranked probability score, and event reliability.

References (canonical):
* Probability Integral Transform (PIT) histogram per Diebold, Gunther,
  Tay 1998 — bin the CDF values of targets under the predictive
  distribution; a calibrated forecast yields a uniform histogram.
* Epstein 1969, "A scoring system for probability forecasts of ranked
  categories" — ranked probability score (RPS).
* Murphy 1973, "A new vector partition of the probability score" —
  reliability component of the Brier decomposition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import cdf as _norm_cdf


def pit_histogram(
    *,
    means: jax.Array,
    variances: jax.Array,
    targets: jax.Array,
    num_bins: int,
) -> jax.Array:
    """Bin counts of PIT values ``F(y | μ, σ)`` under a Gaussian predictive.

    Args:
        means: Predictive means of shape ``(batch,)``.
        variances: Predictive variances of shape ``(batch,)``.
        targets: Observed values of shape ``(batch,)``.
        num_bins: Number of equal-width PIT bins over ``[0, 1]``.

    Returns:
        Integer counts per bin of shape ``(num_bins,)``.
    """
    std = jnp.sqrt(variances)
    pit_values = _norm_cdf(targets, loc=means, scale=std)
    bin_idx = jnp.minimum((pit_values * num_bins).astype(jnp.int32), num_bins - 1)
    return jnp.bincount(bin_idx, length=num_bins)


def ranked_probability_score(
    *,
    probabilities: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Ranked probability score per Epstein 1969.

    ``RPS = sum_k (F_k - O_k)^2`` where ``F_k`` is the cumulative
    predictive probability through class ``k`` and ``O_k`` is the
    cumulative observation (step at the true class).

    Args:
        probabilities: Shape ``(batch, num_classes)`` — per-class
            probabilities (non-negative, sum to 1).
        targets: Integer class indices of shape ``(batch,)``.

    Returns:
        Per-sample RPS of shape ``(batch,)``. Lower is better.
    """
    num_classes = probabilities.shape[-1]
    cumulative_probs = jnp.cumsum(probabilities, axis=-1)
    one_hot = jax.nn.one_hot(targets, num_classes)
    cumulative_obs = jnp.cumsum(one_hot, axis=-1)
    return jnp.sum((cumulative_probs - cumulative_obs) ** 2, axis=-1)


def event_reliability(
    *,
    predicted_event_probabilities: jax.Array,
    event_indicators: jax.Array,
    num_bins: int,
) -> jax.Array:
    """Reliability component of the Brier decomposition (Murphy 1973).

    Bins predicted probabilities, computes the per-bin gap between mean
    predicted probability and the empirical event frequency, and returns
    the count-weighted mean absolute gap.

    Args:
        predicted_event_probabilities: Forecast P(event) in ``[0, 1]``.
        event_indicators: Binary observed indicators (0 / 1).
        num_bins: Equal-width bins over ``[0, 1]``.

    Returns:
        Scalar reliability score in ``[0, 1]``. Lower is better.
    """
    n = predicted_event_probabilities.shape[0]
    bin_idx = jnp.minimum(
        (predicted_event_probabilities * num_bins).astype(jnp.int32), num_bins - 1
    )
    one_hot = jax.nn.one_hot(bin_idx, num_bins)
    counts = jnp.sum(one_hot, axis=0)
    sum_probs = jnp.sum(one_hot * predicted_event_probabilities[:, None], axis=0)
    sum_events = jnp.sum(one_hot * event_indicators[:, None], axis=0)
    safe_counts = jnp.maximum(counts, 1.0)
    mean_probs = sum_probs / safe_counts
    mean_events = sum_events / safe_counts
    gaps = jnp.abs(mean_probs - mean_events)
    return jnp.sum(counts * gaps) / n
