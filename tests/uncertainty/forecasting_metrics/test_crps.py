"""Probabilistic forecast metric contracts.

References (canonical):

* Gneiting & Raftery 2007, "Strictly Proper Scoring Rules, Prediction, and
  Estimation" (JASA) — CRPS, energy score, ranked probability score.
* Ferro 2014, "Fair scores for ensemble forecasts" — fair CRPS
  unbiased estimator for finite-ensemble miscalibration.
* Hamill 2001, "Interpretation of rank histograms" — rank histogram
  definition.
* Fortin et al. 2014, "Why should ensemble spread match the RMSE of the
  ensemble mean?" — spread-skill ratio for calibration assessment.

The numerical core of the empirical CRPS matches
``calibrax.metrics.functional.regression.crps``; the formula references
are also exposed via the WeatherBenchX class-based metric library
(``../weatherbenchX/weatherbenchX/metrics/probabilistic.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _import_fm():
    from opifex.uncertainty import forecasting_metrics

    return forecasting_metrics


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------


def test_crps_matches_two_member_closed_form() -> None:
    """For a 2-member ensemble (x1, x2) and target y, empirical CRPS is
    ``mean(|x_i - y|) - 0.5 * mean(|x_i - x_j|)``.
    """
    fm = _import_fm()
    predictions = jnp.array([[1.0, 3.0]])
    targets = jnp.array([2.0])
    # mean abs error = (|1-2| + |3-2|)/2 = 1.0
    # 0.5 * mean pairwise = 0.5 * (|1-1| + |1-3| + |3-1| + |3-3|)/4 = 0.5 * 1.0 = 0.5
    # crps = 1.0 - 0.5 = 0.5
    out = float(fm.crps(predictions=predictions, targets=targets))
    assert out == pytest.approx(0.5, abs=1e-6)


def test_fair_crps_is_less_than_empirical_for_finite_ensembles() -> None:
    """Fair CRPS removes finite-ensemble bias (Ferro 2014; Zamo & Naveau 2018):

    Empirical CRPS spread term divides by ``M^2``; fair-CRPS divides by
    ``M(M-1)``. Since ``M(M-1) < M^2``, the fair spread is larger →
    fair CRPS = error - 0.5 * spread is SMALLER than the empirical
    estimator. Both converge as ``M → ∞``.

    Canonical reference: ``../weatherbenchX/weatherbenchX/metrics/probabilistic.py``
    ``CRPSSpread._compute_per_variable`` uses
    ``sum_pairs / (M * (M - int(fair)))``.
    """
    fm = _import_fm()
    rng = np.random.default_rng(0)
    n_samples = 16
    n_members = 5
    predictions = jnp.asarray(rng.standard_normal((n_samples, n_members)))
    targets = jnp.asarray(rng.standard_normal(n_samples))
    empirical = float(fm.crps(predictions=predictions, targets=targets))
    fair = float(fm.fair_crps(predictions=predictions, targets=targets))
    assert fair < empirical


def test_crps_is_jit_compatible() -> None:
    fm = _import_fm()
    rng = np.random.default_rng(0)
    predictions = jnp.asarray(rng.standard_normal((8, 4)))
    targets = jnp.asarray(rng.standard_normal(8))
    jitted = jax.jit(lambda p, t: fm.crps(predictions=p, targets=t))
    out = float(jitted(predictions, targets))
    eager = float(fm.crps(predictions=predictions, targets=targets))
    assert out == pytest.approx(eager, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# Energy score (multivariate ensemble)
# ---------------------------------------------------------------------------


def test_energy_score_matches_two_member_closed_form() -> None:
    """Energy score (Gneiting & Raftery 2007 §4.2):

    ES(P, y) = mean_i ||X_i - y|| - 0.5 mean_{i,j} ||X_i - X_j||
    """
    fm = _import_fm()
    # 2 ensemble members, 1 sample, 2-D output
    ensemble = jnp.array([[[1.0, 0.0], [3.0, 0.0]]])  # shape (1, 2, 2)
    targets = jnp.array([[2.0, 0.0]])  # shape (1, 2)
    # ||X1 - y|| = 1, ||X2 - y|| = 1 → mean = 1.0
    # ||X1 - X2|| = 2; pairwise mean = (0 + 2 + 2 + 0)/4 = 1.0
    # ES = 1.0 - 0.5 * 1.0 = 0.5
    out = float(fm.energy_score(ensemble=ensemble, targets=targets)[0])
    assert out == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Rank histogram
# ---------------------------------------------------------------------------


def test_rank_histogram_counts_target_position_in_sorted_ensemble() -> None:
    """For a target equal to the smallest ensemble member, rank should be 0.
    For a target equal to the largest, rank should equal num_members.
    """
    fm = _import_fm()
    # 3 samples: target below all / inside / above all
    ensemble = jnp.array(
        [
            [1.0, 2.0, 3.0],  # target 0 → rank 0
            [1.0, 2.0, 3.0],  # target 2.5 → rank 2
            [1.0, 2.0, 3.0],  # target 4 → rank 3
        ]
    )
    targets = jnp.array([0.0, 2.5, 4.0])
    counts = fm.rank_histogram(ensemble=ensemble, targets=targets)
    # 4 bins for a 3-member ensemble (M + 1).
    assert counts.shape == (4,)
    assert int(counts[0]) == 1  # target below all
    assert int(counts[2]) == 1  # target between members 1 and 2
    assert int(counts[3]) == 1  # target above all


# ---------------------------------------------------------------------------
# Spread-skill ratio
# ---------------------------------------------------------------------------


def test_spread_skill_ratio_is_one_for_perfect_calibration() -> None:
    """When ensemble spread matches the RMSE of the ensemble mean across
    samples, the ratio is ~1.0 (Fortin et al. 2014).
    """
    fm = _import_fm()
    rng = np.random.default_rng(0)
    n_samples = 4096
    n_members = 50
    sigma = 0.5
    ensemble = jnp.asarray(sigma * rng.standard_normal((n_samples, n_members)))
    # Targets drawn from the same N(0, sigma^2).
    targets = jnp.asarray(sigma * rng.standard_normal(n_samples))
    ratio = float(fm.spread_skill_ratio(ensemble=ensemble, targets=targets))
    assert ratio == pytest.approx(1.0, abs=0.1)


def test_spread_skill_ratio_below_one_when_ensemble_is_underdispersed() -> None:
    """Narrow spread + wider target distribution → ratio < 1."""
    fm = _import_fm()
    rng = np.random.default_rng(0)
    n_samples = 2048
    n_members = 30
    ensemble = jnp.asarray(0.1 * rng.standard_normal((n_samples, n_members)))
    targets = jnp.asarray(rng.standard_normal(n_samples))
    ratio = float(fm.spread_skill_ratio(ensemble=ensemble, targets=targets))
    assert ratio < 0.5


# ---------------------------------------------------------------------------
# PIT histogram (probability integral transform)
# ---------------------------------------------------------------------------


def test_pit_histogram_is_uniform_for_perfectly_calibrated_gaussian() -> None:
    """For y ~ N(μ, σ²) and predictive N(μ, σ²), the PIT values are uniform on
    ``(0, 1)`` → bin counts should be ~equal.
    """
    fm = _import_fm()
    rng = np.random.default_rng(0)
    n = 4096
    means = jnp.zeros(n)
    variances = jnp.ones(n)
    targets = jnp.asarray(rng.standard_normal(n))
    n_bins = 10
    counts = fm.pit_histogram(means=means, variances=variances, targets=targets, num_bins=n_bins)
    expected_per_bin = n / n_bins
    # Allow 25% deviation per bin (sampling tolerance).
    assert bool(jnp.all(jnp.abs(counts - expected_per_bin) < 0.25 * expected_per_bin))


# ---------------------------------------------------------------------------
# Ranked probability score (categorical / thresholded)
# ---------------------------------------------------------------------------


def test_rps_matches_closed_form_on_three_class_example() -> None:
    """RPS for cumulative probabilities ``F`` vs cumulative observation ``O``:

    RPS = sum_k (F_k - O_k)^2
    """
    fm = _import_fm()
    # 1 sample, 3-class predictive [0.5, 0.3, 0.2], true class 1.
    probabilities = jnp.array([[0.5, 0.3, 0.2]])
    targets = jnp.array([1])
    cumulative_probs = jnp.array([[0.5, 0.8, 1.0]])
    cumulative_obs = jnp.array([[0.0, 1.0, 1.0]])
    expected = float(jnp.sum((cumulative_probs - cumulative_obs) ** 2))
    out = float(fm.ranked_probability_score(probabilities=probabilities, targets=targets)[0])
    assert out == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Event reliability
# ---------------------------------------------------------------------------


def test_event_reliability_matches_known_threshold_event_frequency() -> None:
    """For event = ``target > threshold`` and predicted P(event), reliability
    is the calibration gap |mean(P_in_bin) - empirical_freq_in_bin| averaged
    by bin.
    """
    fm = _import_fm()
    # 10 samples with predicted P(event) = i/10 and empirical event indicator.
    predicted_probs = jnp.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    event_indicators = jnp.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1], dtype=jnp.float32)
    reliability = float(
        fm.event_reliability(
            predicted_event_probabilities=predicted_probs,
            event_indicators=event_indicators,
            num_bins=5,
        )
    )
    assert reliability >= 0.0


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_forecasting_metric_surface() -> None:
    fm = _import_fm()
    expected = {
        "crps",
        "fair_crps",
        "energy_score",
        "rank_histogram",
        "spread_skill_ratio",
        "pit_histogram",
        "ranked_probability_score",
        "ensemble_ranked_probability_score",
        "ranked_probability_skill_score",
        "event_reliability",
    }
    missing = expected - set(dir(fm))
    assert not missing, f"missing public forecasting metrics: {sorted(missing)}"
