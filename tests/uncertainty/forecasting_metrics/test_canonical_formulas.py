"""Canonical-formula contract tests for forecasting metrics.

Covers:

* :func:`event_reliability` returns the Murphy 1973 squared-gap
  Brier-reliability decomposition.
* :func:`spread_skill_ratio` uses Fortin 2014 / WeatherBenchX
  unbiased estimators (ddof=1 variance + bias-corrected MSE).
* :func:`ensemble_ranked_probability_score` applies Ferro 2008
  fair-RPS debiasing for ensemble-sample inputs.
* :func:`ranked_probability_skill_score` returns the Murphy 1971 RPSS.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.uncertainty import forecasting_metrics as fm


# ---------------------------------------------------------------------------
# event_reliability (Murphy 1973 squared gaps)
# ---------------------------------------------------------------------------


def test_event_reliability_is_zero_for_perfectly_calibrated_forecast() -> None:
    """A forecast that hits the empirical frequency in every bin has zero
    reliability term (squared gap = 0)."""
    # Bin 0 (probs in [0, 0.2)): 4 samples, forecast 0.1, 0 events → freq = 0
    # Bin 4 (probs in [0.8, 1.0)): 4 samples, forecast 0.9, 4 events → freq = 1
    # In a calibrated forecast every bin's mean prob equals empirical freq.
    probs = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    targets = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    rel = float(
        fm.event_reliability(
            predicted_event_probabilities=probs,
            event_indicators=targets,
            num_bins=5,
        )
    )
    assert rel == pytest.approx(0.0, abs=1e-6)


def test_event_reliability_known_value_squared_gap() -> None:
    """Manually compute REL = (n_k/n) * (f_k - o_k)^2 for a controlled split.

    2 samples in bin 0 (forecast ~0.1), event_indicator = [0, 1] → freq = 0.5,
    gap² = (0.1 − 0.5)² = 0.16. Weighted: (2/2) * 0.16 = 0.16.
    """
    probs = jnp.array([0.1, 0.1])
    events = jnp.array([0.0, 1.0])
    rel = float(
        fm.event_reliability(
            predicted_event_probabilities=probs,
            event_indicators=events,
            num_bins=5,
        )
    )
    assert rel == pytest.approx(0.16, abs=1e-6)


# ---------------------------------------------------------------------------
# spread_skill_ratio (Fortin 2014 unbiased estimators)
# ---------------------------------------------------------------------------


def test_spread_skill_ratio_uses_unbiased_variance() -> None:
    """For a well-calibrated large ensemble of N(0, σ²) members with
    targets also drawn from N(0, σ²), the unbiased ratio converges to 1.0
    more tightly than the biased version did (because the bias terms in
    both numerator and denominator cancel)."""
    rng = np.random.default_rng(1234)
    n_samples = 4096
    n_members = 20
    sigma = 1.0
    ensemble = jnp.asarray(sigma * rng.standard_normal((n_samples, n_members)))
    targets = jnp.asarray(sigma * rng.standard_normal(n_samples))
    ratio = float(fm.spread_skill_ratio(ensemble=ensemble, targets=targets))
    # Tighter tolerance than the biased estimator could deliver.
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_spread_skill_ratio_small_ensemble_does_not_systematically_underestimate() -> None:
    """Sanity: with a tiny 5-member ensemble of calibrated draws, the
    unbiased estimator stays near 1.0; the previous biased version
    biased low."""
    rng = np.random.default_rng(5678)
    n_samples = 8192
    n_members = 5
    sigma = 1.0
    ensemble = jnp.asarray(sigma * rng.standard_normal((n_samples, n_members)))
    targets = jnp.asarray(sigma * rng.standard_normal(n_samples))
    ratio = float(fm.spread_skill_ratio(ensemble=ensemble, targets=targets))
    assert ratio == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# ensemble_ranked_probability_score (Ferro 2008 fair estimator)
# ---------------------------------------------------------------------------


def test_ensemble_rps_biased_matches_manual_for_single_threshold() -> None:
    """With a single threshold, RPS reduces to the squared CDF gap. A
    single ensemble sample equal to the target with threshold ABOVE
    both gives F̂ = 1, O = 1 → gap = 0; a single sample 0, target 0,
    threshold 0.5 → both <= 0.5 → gap = 0."""
    samples = jnp.array([[0.0]])
    targets = jnp.array([0.0])
    thresholds = jnp.array([0.5])
    out = fm.ensemble_ranked_probability_score(
        samples=samples, targets=targets, thresholds=thresholds, fair=False
    )
    assert out.shape == (1,)
    assert float(out[0]) == pytest.approx(0.0, abs=1e-6)


def test_ensemble_rps_biased_known_two_samples() -> None:
    """Two-member ensemble {0, 1}, target 0, threshold 0.5:
    F̂ = mean(1{0<=0.5} + 1{1<=0.5}) = mean(1, 0) = 0.5
    O = 1{0<=0.5} = 1
    Squared gap = (0.5 - 1)² = 0.25.
    """
    samples = jnp.array([[0.0, 1.0]])
    targets = jnp.array([0.0])
    thresholds = jnp.array([0.5])
    out = fm.ensemble_ranked_probability_score(
        samples=samples, targets=targets, thresholds=thresholds, fair=False
    )
    assert float(out[0]) == pytest.approx(0.25, abs=1e-6)


def test_ensemble_rps_fair_subtracts_finite_sample_bias() -> None:
    """Fair RPS subtracts ``var(F̂, ddof=1) / M`` per threshold from the
    biased squared error. For samples {0, 1}, target 0, threshold 0.5,
    indicator var with ddof=1 = 0.5 over M=2 → bias = 0.25; fair score =
    0.25 − 0.25 = 0.0.
    """
    samples = jnp.array([[0.0, 1.0]])
    targets = jnp.array([0.0])
    thresholds = jnp.array([0.5])
    fair = fm.ensemble_ranked_probability_score(
        samples=samples, targets=targets, thresholds=thresholds, fair=True
    )
    assert float(fair[0]) == pytest.approx(0.0, abs=1e-6)


def test_ensemble_rps_fair_expectation_independent_of_ensemble_size() -> None:
    """Fair estimator's mean over many calibrated draws should not change
    materially when ensemble size changes — that is the whole point of
    Ferro 2008 debiasing."""
    rng = np.random.default_rng(0)
    n_batches = 2048
    thresholds = jnp.linspace(-2.0, 2.0, 9)
    targets = jnp.asarray(rng.standard_normal(n_batches))

    def mean_fair_rps(num_members: int) -> float:
        samples = jnp.asarray(rng.standard_normal((n_batches, num_members)))
        return float(
            jnp.mean(
                fm.ensemble_ranked_probability_score(
                    samples=samples,
                    targets=targets,
                    thresholds=thresholds,
                    fair=True,
                )
            )
        )

    score_small = mean_fair_rps(num_members=8)
    score_large = mean_fair_rps(num_members=64)
    # Difference attributable to finite-sample noise only, not to ensemble size bias.
    assert abs(score_small - score_large) < 0.05, (
        f"fair RPS should be ~ensemble-size-invariant: "
        f"M=8 → {score_small:.4f}, M=64 → {score_large:.4f}"
    )


# ---------------------------------------------------------------------------
# ranked_probability_skill_score (Murphy 1971)
# ---------------------------------------------------------------------------


def test_rpss_is_one_for_perfect_forecast() -> None:
    """If RPS = 0 (perfect) and reference > 0, skill = 1."""
    skill = fm.ranked_probability_skill_score(rps=jnp.asarray(0.0), rps_reference=jnp.asarray(0.5))
    assert float(skill) == pytest.approx(1.0, abs=1e-6)


def test_rpss_is_zero_when_forecast_matches_reference() -> None:
    """RPS == reference → skill = 0."""
    skill = fm.ranked_probability_skill_score(rps=jnp.asarray(0.4), rps_reference=jnp.asarray(0.4))
    assert float(skill) == pytest.approx(0.0, abs=1e-6)


def test_rpss_is_negative_when_forecast_worse_than_reference() -> None:
    skill = fm.ranked_probability_skill_score(rps=jnp.asarray(1.0), rps_reference=jnp.asarray(0.5))
    assert float(skill) < 0.0


def test_rpss_supports_array_inputs() -> None:
    skills = fm.ranked_probability_skill_score(
        rps=jnp.asarray([0.0, 0.4, 1.0]),
        rps_reference=jnp.asarray([0.5, 0.4, 0.5]),
    )
    assert skills.shape == (3,)


# ---------------------------------------------------------------------------
# JIT compatibility for the new surfaces
# ---------------------------------------------------------------------------


def test_ensemble_rps_jit_compatible() -> None:
    samples = jnp.array([[0.0, 1.0]])
    targets = jnp.array([0.0])
    thresholds = jnp.array([0.5])

    @jax.jit
    def call(s: jax.Array, t: jax.Array, b: jax.Array) -> jax.Array:
        return fm.ensemble_ranked_probability_score(samples=s, targets=t, thresholds=b, fair=True)

    out = call(samples, targets, thresholds)
    assert out.shape == (1,)
