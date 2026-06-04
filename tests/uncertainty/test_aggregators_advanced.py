"""Contract tests for the multi-source uncertainty-aggregator surface.

Covers the canonical aggregator + multi-source classes in
:mod:`opifex.uncertainty.aggregators`:

* :class:`EpistemicUncertainty` — variance / std / range / iqr ensemble
  disagreement modes plus pairwise / cosine predictive diversity.
* :class:`DistributionalAleatoricUncertainty` — Gaussian + Laplace +
  mixture aleatoric quantification.
* :class:`MultiSourceUncertaintyAggregator` — sum / weighted-sum / max
  combination, plus :meth:`adaptive_weighting` (reliability /
  inverse-variance / entropy / uniform) and
  :meth:`assess_uncertainty_quality`.
* :class:`EnhancedUncertaintyQuantifier` — orchestrator that wires the
  three above into a single ``enhanced_decompose_uncertainty`` call.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pytest

from opifex.uncertainty.aggregators import (
    DistributionalAleatoricUncertainty,
    EnhancedUncertaintyComponents,
    EnhancedUncertaintyQuantifier,
    EpistemicUncertainty,
    MultiSourceUncertaintyAggregator,
    UncertaintyQuantifier,
)


# ---------------------------------------------------------------------------
# UncertaintyQuantifier — exact Gaussian prediction intervals
# ---------------------------------------------------------------------------


def test_prediction_intervals_use_exact_gaussian_quantile() -> None:
    """Half-width must be the exact Gaussian quantile for any confidence level.

    The previous lookup-table implementation returned a wrong ``z = 1.0``
    fallback for any level below 0.90; the interval must instead use
    ``norm.ppf(1 - alpha/2)`` (the in-repo convention, e.g.
    ``reliability/failure_probability.py``).
    """
    quantifier = UncertaintyQuantifier()
    mean = jnp.zeros((1, 1))
    variance = jnp.ones((1, 1))  # std == 1 so the half-width equals the z-score

    # 0.80 is OFF the old lookup table -> old code returned z = 1.0 (wrong).
    lower, upper = quantifier.compute_prediction_intervals(mean, variance, confidence_level=0.80)
    expected_z = float(jsp.stats.norm.ppf(0.90))  # 1 - 0.20/2 = 0.90 -> ~1.2816
    assert float(upper[0, 0]) == pytest.approx(expected_z, abs=1e-4)
    assert float(lower[0, 0]) == pytest.approx(-expected_z, abs=1e-4)

    # Standard 95% level still recovers ~1.96.
    _, upper_95 = quantifier.compute_prediction_intervals(mean, variance, confidence_level=0.95)
    assert float(upper_95[0, 0]) == pytest.approx(1.959964, abs=1e-4)


def test_prediction_intervals_jit_compatible() -> None:
    """compute_prediction_intervals must be jittable over (mean, variance)."""
    quantifier = UncertaintyQuantifier()
    jitted = jax.jit(
        lambda m, v: quantifier.compute_prediction_intervals(m, v, confidence_level=0.95)
    )
    _, upper = jitted(jnp.zeros((2, 3)), jnp.ones((2, 3)))
    assert upper.shape == (2, 3)
    assert float(upper[0, 0]) == pytest.approx(1.959964, abs=1e-4)


# ---------------------------------------------------------------------------
# EpistemicUncertainty surface
# ---------------------------------------------------------------------------


def test_epistemic_compute_variance_matches_compute_variance_of_expected() -> None:
    preds = jax.random.normal(jax.random.key(0), (8, 4, 2))
    via_var = EpistemicUncertainty.compute_variance(preds)
    via_alias = EpistemicUncertainty.compute_variance_of_expected(preds)
    assert jnp.allclose(via_var, via_alias)


def test_compute_ensemble_disagreement_iqr_returns_q75_minus_q25() -> None:
    preds = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    out = EpistemicUncertainty.compute_ensemble_disagreement(
        preds[:, :, None], aggregation_method="iqr"
    )
    expected_q75 = jnp.percentile(preds[:, :, None], 75, axis=0)
    expected_q25 = jnp.percentile(preds[:, :, None], 25, axis=0)
    assert jnp.allclose(out, expected_q75 - expected_q25)


# ---------------------------------------------------------------------------
# DistributionalAleatoricUncertainty
# ---------------------------------------------------------------------------


def test_gaussian_uncertainty_returns_exp_two_log_std() -> None:
    estimator = DistributionalAleatoricUncertainty()
    log_std = jnp.array([[-1.0, 0.0, 1.0]])
    out = estimator.compute_gaussian_uncertainty(jnp.zeros_like(log_std), log_std)
    assert jnp.allclose(out, jnp.exp(2.0 * log_std))


def test_mixture_uncertainty_returns_non_negative() -> None:
    estimator = DistributionalAleatoricUncertainty()
    weights = jnp.array([[0.4, 0.6]])
    means = jnp.array([[[0.0, 1.0], [1.0, 2.0]]])
    log_stds = jnp.array([[[0.0, -1.0], [-1.0, 0.0]]])
    out = estimator.compute_mixture_uncertainty(weights, means, log_stds)
    assert bool(jnp.all(out >= 0.0))


# ---------------------------------------------------------------------------
# MultiSourceUncertaintyAggregator
# ---------------------------------------------------------------------------


def test_aggregate_uncertainties_variance_sum_is_pointwise_sum() -> None:
    aggregator = MultiSourceUncertaintyAggregator()
    epi = [jnp.ones((4, 2)) * 0.1]
    ale = [jnp.ones((4, 2)) * 0.3]
    out = aggregator.aggregate_uncertainties(
        epistemic_sources=epi, aleatoric_sources=ale, method="variance_sum"
    )
    assert jnp.allclose(out, 0.4 * jnp.ones((4, 2)))


def test_aggregate_uncertainties_rejects_unknown_method() -> None:
    aggregator = MultiSourceUncertaintyAggregator()
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregator.aggregate_uncertainties(
            epistemic_sources=[jnp.zeros((4, 2))],
            aleatoric_sources=[jnp.zeros((4, 2))],
            method="bogus",
        )


def test_compute_uncertainty_breakdown_assigns_default_names() -> None:
    aggregator = MultiSourceUncertaintyAggregator()
    out = aggregator.compute_uncertainty_breakdown(
        epistemic_sources=[jnp.zeros((4, 2))],
        aleatoric_sources=[jnp.zeros((4, 2))],
    )
    assert set(out) == {"epistemic_0", "aleatoric_0"}


# ---------------------------------------------------------------------------
# EnhancedUncertaintyQuantifier orchestrator
# ---------------------------------------------------------------------------


def test_enhanced_quantifier_returns_components_value_object() -> None:
    quantifier = EnhancedUncertaintyQuantifier(
        ensemble_size=4,
        distributional_output=True,
        multi_source_aggregation=True,
    )
    ensemble_predictions = jax.random.normal(jax.random.key(0), (4, 8, 1))
    distributional_std = jnp.full((8, 1), 0.2)
    result = quantifier.enhanced_decompose_uncertainty(
        ensemble_predictions=ensemble_predictions,
        distributional_std=distributional_std,
    )
    assert isinstance(result, EnhancedUncertaintyComponents)
    assert result.epistemic_ensemble.shape == (8, 1)
    assert result.aleatoric_distributional.shape == (8, 1)
    assert result.total_uncertainty.shape == (8, 1)


def test_enhanced_quantifier_dropout_path_populates_epistemic_dropout() -> None:
    quantifier = EnhancedUncertaintyQuantifier(ensemble_size=4)
    ensemble_predictions = jax.random.normal(jax.random.key(0), (4, 8, 1))
    dropout_predictions = jax.random.normal(jax.random.key(1), (16, 8, 1))
    result = quantifier.enhanced_decompose_uncertainty(
        ensemble_predictions=ensemble_predictions,
        dropout_predictions=dropout_predictions,
    )
    assert result.epistemic_dropout is not None
    assert result.epistemic_dropout.shape == (8, 1)
