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
import pytest

from opifex.uncertainty.aggregators import (
    DistributionalAleatoricUncertainty,
    EnhancedUncertaintyComponents,
    EnhancedUncertaintyQuantifier,
    EpistemicUncertainty,
    MultiSourceUncertaintyAggregator,
)


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
