"""Contract tests for aggregator methods that orchestrate multi-source UQ.

Covers:

* :meth:`EpistemicUncertainty.compute_ensemble_disagreement` —
  variance / std / range / iqr modes.
* :meth:`EpistemicUncertainty.compute_predictive_diversity` —
  pairwise-distance / cosine modes.
* :meth:`DistributionalAleatoricUncertainty.compute_laplace_uncertainty`.
* :meth:`MultiSourceUncertaintyAggregator.adaptive_weighting` —
  reliability / inverse-variance / entropy / uniform modes.
* :meth:`MultiSourceUncertaintyAggregator.assess_uncertainty_quality`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# EpistemicUncertainty.compute_ensemble_disagreement
# ---------------------------------------------------------------------------


def test_ensemble_disagreement_variance_matches_compute_variance() -> None:
    from opifex.uncertainty.aggregators import EpistemicUncertainty

    preds = jax.random.normal(jax.random.key(0), (8, 4, 2))
    via_alias = EpistemicUncertainty.compute_ensemble_disagreement(
        preds, aggregation_method="variance"
    )
    via_base = EpistemicUncertainty.compute_variance(preds)
    assert jnp.allclose(via_alias, via_base)


def test_ensemble_disagreement_std_range_iqr_have_expected_shape() -> None:
    from opifex.uncertainty.aggregators import EpistemicUncertainty

    preds = jax.random.normal(jax.random.key(1), (8, 4, 2))
    for method in ("std", "range", "iqr"):
        out = EpistemicUncertainty.compute_ensemble_disagreement(preds, aggregation_method=method)
        assert out.shape == (4, 2)
        assert bool(jnp.all(out >= 0))


def test_ensemble_disagreement_unknown_method_raises() -> None:
    from opifex.uncertainty.aggregators import EpistemicUncertainty

    preds = jax.random.normal(jax.random.key(2), (8, 4, 2))
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        EpistemicUncertainty.compute_ensemble_disagreement(preds, aggregation_method="bogus")


# ---------------------------------------------------------------------------
# EpistemicUncertainty.compute_predictive_diversity
# ---------------------------------------------------------------------------


def test_predictive_diversity_pairwise_zero_for_identical_predictions() -> None:
    from opifex.uncertainty.aggregators import EpistemicUncertainty

    # Same prediction across all 5 ensemble members → zero diversity.
    same = jnp.tile(jnp.ones((1, 4, 2)), (5, 1, 1))
    out = EpistemicUncertainty.compute_predictive_diversity(
        same, diversity_metric="pairwise_distance"
    )
    assert out.shape == (4,)
    assert jnp.allclose(out, 0.0, atol=1e-6)


def test_predictive_diversity_cosine_zero_for_aligned_predictions() -> None:
    from opifex.uncertainty.aggregators import EpistemicUncertainty

    aligned = jnp.tile(jnp.array([[[1.0, 2.0]]]), (5, 4, 1))
    out = EpistemicUncertainty.compute_predictive_diversity(
        aligned, diversity_metric="cosine_diversity"
    )
    assert out.shape == (4,)
    assert jnp.allclose(out, 0.0, atol=1e-5)


def test_predictive_diversity_unknown_metric_raises() -> None:
    from opifex.uncertainty.aggregators import EpistemicUncertainty

    preds = jax.random.normal(jax.random.key(3), (5, 4, 2))
    with pytest.raises(ValueError, match="Unknown diversity metric"):
        EpistemicUncertainty.compute_predictive_diversity(preds, diversity_metric="bogus")


# ---------------------------------------------------------------------------
# DistributionalAleatoricUncertainty.compute_laplace_uncertainty
# ---------------------------------------------------------------------------


def test_laplace_uncertainty_is_scale_times_sqrt_two() -> None:
    from opifex.uncertainty.aggregators import DistributionalAleatoricUncertainty

    estimator = DistributionalAleatoricUncertainty()
    scale = jnp.array([[1.0, 2.0], [0.5, 0.25]])
    out = estimator.compute_laplace_uncertainty(scale)
    assert jnp.allclose(out, scale * jnp.sqrt(2.0))


# ---------------------------------------------------------------------------
# MultiSourceUncertaintyAggregator.adaptive_weighting
# ---------------------------------------------------------------------------


def test_adaptive_weighting_reliability_normalises_across_sources() -> None:
    from opifex.uncertainty.aggregators import MultiSourceUncertaintyAggregator

    u1 = jax.random.uniform(jax.random.key(4), (8, 2))
    u2 = jax.random.uniform(jax.random.key(5), (8, 2))
    r1 = jnp.ones((8,)) * 0.9
    r2 = jnp.ones((8,)) * 0.1
    weights = MultiSourceUncertaintyAggregator.adaptive_weighting(
        [u1, u2], reliability_scores=[r1, r2], adaptation_method="reliability_based"
    )
    assert weights.shape == (2, 8)
    sums = jnp.sum(weights, axis=0)
    assert jnp.allclose(sums, 1.0, atol=1e-6)


def test_adaptive_weighting_uniform_returns_equal_weights() -> None:
    from opifex.uncertainty.aggregators import MultiSourceUncertaintyAggregator

    sources = [jnp.zeros((8, 2)), jnp.zeros((8, 2)), jnp.zeros((8, 2))]
    weights = MultiSourceUncertaintyAggregator.adaptive_weighting(
        sources, adaptation_method="uniform"
    )
    assert weights.shape == (3, 8)
    assert jnp.allclose(weights, 1.0 / 3.0)


def test_adaptive_weighting_unknown_method_raises() -> None:
    from opifex.uncertainty.aggregators import MultiSourceUncertaintyAggregator

    sources = [jnp.zeros((8, 2)), jnp.zeros((8, 2))]
    with pytest.raises(ValueError, match="Unknown adaptation method"):
        MultiSourceUncertaintyAggregator.adaptive_weighting(sources, adaptation_method="bogus")


# ---------------------------------------------------------------------------
# MultiSourceUncertaintyAggregator.assess_uncertainty_quality
# ---------------------------------------------------------------------------


def test_assess_uncertainty_quality_without_true_values_returns_source_stats_only() -> None:
    from opifex.uncertainty.aggregators import MultiSourceUncertaintyAggregator

    predictions = jnp.zeros((8, 2))
    uncertainties = jnp.ones((8, 2)) * 0.5
    quality = MultiSourceUncertaintyAggregator.assess_uncertainty_quality(
        predictions, uncertainties
    )
    assert set(quality) == {
        "mean_uncertainty",
        "uncertainty_std",
        "uncertainty_range",
        "mean_confidence",
    }
    assert quality["mean_uncertainty"] == pytest.approx(0.5, abs=1e-5)


def test_assess_uncertainty_quality_with_true_values_returns_coverage_metrics() -> None:
    from opifex.uncertainty.aggregators import MultiSourceUncertaintyAggregator

    predictions = jnp.zeros((8, 2))
    uncertainties = jnp.ones((8, 2))
    # Ground truth within ±2σ band of zero — full coverage.
    true_values = jnp.zeros((8, 2)) + 0.5
    quality = MultiSourceUncertaintyAggregator.assess_uncertainty_quality(
        predictions, uncertainties, true_values=true_values
    )
    assert quality["coverage_probability"] == pytest.approx(1.0)
    assert quality["mean_interval_width"] == pytest.approx(4.0, abs=1e-5)
    assert "calibration_error" in quality
