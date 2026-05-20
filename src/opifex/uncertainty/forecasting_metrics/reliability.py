"""PIT histogram, ranked probability score, and event reliability.

References (canonical):
* Probability Integral Transform (PIT) histogram per Diebold, Gunther,
  Tay 1998 — bin the CDF values of targets under the predictive
  distribution; a calibrated forecast yields a uniform histogram.
* Epstein 1969, "A scoring system for probability forecasts of ranked
  categories" — ranked probability score (RPS).
* Murphy 1971, "A note on the ranked probability score" — RPSS skill
  score against a reference forecast.
* Murphy 1973, "A new vector partition of the probability score" —
  reliability component of the Brier decomposition.
* Ferro, Richardson, Weigel 2008 — "fair" finite-ensemble RPS estimator
  matching WeatherBenchX ``EnsembleRankedProbabilityScore(fair=True)``.
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

    Bins predicted probabilities and returns
    ``REL = (1/n) Σ_k n_k (f_k − o_k)²``
    where ``n_k`` is the bin count, ``f_k`` the mean forecast in bin ``k``,
    and ``o_k`` the empirical event frequency in bin ``k`` — the standard
    squared-gap form that decomposes the Brier score as
    ``BS = REL − RES + UNC``.

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
    squared_gaps = (mean_probs - mean_events) ** 2
    return jnp.sum(counts * squared_gaps) / n


def ensemble_ranked_probability_score(
    *,
    samples: jax.Array,
    targets: jax.Array,
    thresholds: jax.Array,
    fair: bool = True,
) -> jax.Array:
    """Ranked probability score for an ensemble of continuous samples.

    Bins predictions and targets at the supplied ``thresholds`` to form
    cumulative probabilities and observed indicators, then sums squared
    CDF gaps across thresholds. When ``fair=True`` (default), applies the
    Ferro / Richardson / Weigel 2008 finite-ensemble debiasing:

    ``RPS_fair = Σ_k [(F̂_k − O_k)² − Var(F̂_k, ddof=1) / M]``

    where ``F̂_k`` is the empirical CDF of the prediction ensemble at
    threshold ``b_k`` and ``M`` is the ensemble size. This matches
    WeatherBenchX ``EnsembleRankedProbabilityScore(fair=True)`` and
    yields a score whose expectation does not depend on ``M`` —
    essential when comparing ensembles of different sizes.

    Args:
        samples: ``(batch, num_members)`` ensemble predictions of a
            real-valued scalar.
        targets: ``(batch,)`` observed values.
        thresholds: ``(num_thresholds,)`` monotonically increasing
            threshold values defining the CDF bins.
        fair: When ``True`` (default), apply Ferro 2008 debiasing.

    Returns:
        Per-sample RPS of shape ``(batch,)``. Lower is better.
    """
    samples_per_threshold = samples[:, :, None] <= thresholds[None, None, :]
    pred_cdf = jnp.mean(samples_per_threshold.astype(jnp.float32), axis=1)
    target_cdf = (targets[:, None] <= thresholds[None, :]).astype(jnp.float32)
    squared_error = (pred_cdf - target_cdf) ** 2
    if not fair:
        return jnp.sum(squared_error, axis=-1)
    num_members = samples.shape[1]
    pred_cdf_var = jnp.var(samples_per_threshold.astype(jnp.float32), axis=1, ddof=1)
    bias_correction = pred_cdf_var / num_members
    return jnp.sum(squared_error - bias_correction, axis=-1)


def ranked_probability_skill_score(
    *,
    rps: jax.Array,
    rps_reference: jax.Array,
) -> jax.Array:
    """Ranked probability skill score per Murphy 1971.

    ``RPSS = 1 − RPS / RPS_reference``. Skill > 0 indicates the forecast
    beats the reference (e.g. climatology); skill ≤ 0 indicates the
    reference does at least as well.

    Args:
        rps: Forecast RPS, any shape.
        rps_reference: Reference-forecast RPS, broadcastable to ``rps``.

    Returns:
        Skill score, same shape as ``rps``. Higher is better.
    """
    return 1.0 - rps / rps_reference
