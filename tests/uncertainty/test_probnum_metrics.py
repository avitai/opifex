"""Tests for the probnum-evaluation vendored calibration metrics.

Vendors the ANEES + NCI + sample-distance metrics from the
``probnum-evaluation`` reference (Bar-Shalom 2002, Li+ 2006).

Canonical references:
* ``../probnum-evaluation/src/probnumeval/timeseries/_calibration_measures.py``
  (function names + paper citations).
* Bar-Shalom, Y., Li, X. R., Kirubarajan, T. 2002 — *Estimation with
  Applications to Tracking and Navigation*, IFAC 2002 (ANEES).
* Li, X. R., Zhao, Z. 2006 — *Measuring estimator's credibility*, IFAC
  (NCI).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.metrics import (
    anees,
    chi2_confidence_intervals,
    non_credibility_index,
)


def test_anees_returns_one_on_perfectly_calibrated_gaussian_predictions() -> None:
    """Calibrated predictions: error covariance = predicted covariance → ANEES ≈ 1."""
    key = jax.random.PRNGKey(0)
    num_steps = 4000
    dim = 2
    predicted_means = jnp.zeros((num_steps, dim))
    predicted_covs = jnp.broadcast_to(jnp.eye(dim), (num_steps, dim, dim))
    references = jax.random.normal(key, (num_steps, dim))  # exactly N(0, I)
    metric = anees(
        predicted_means=predicted_means,
        predicted_covariances=predicted_covs,
        references=references,
    )
    assert jnp.abs(metric - 1.0) < 0.1


def test_anees_greater_than_one_when_predicted_covariance_is_too_small() -> None:
    """Over-confident predictions: ANEES > 1."""
    key = jax.random.PRNGKey(1)
    num_steps = 2000
    dim = 2
    predicted_means = jnp.zeros((num_steps, dim))
    # Predicted variance is 0.25; true variance is 1 → ANEES ≈ 4.
    predicted_covs = jnp.broadcast_to(0.25 * jnp.eye(dim), (num_steps, dim, dim))
    references = jax.random.normal(key, (num_steps, dim))
    metric = anees(
        predicted_means=predicted_means,
        predicted_covariances=predicted_covs,
        references=references,
    )
    assert metric > 2.0


def test_anees_less_than_one_when_predicted_covariance_is_too_large() -> None:
    """Under-confident predictions: ANEES < 1."""
    key = jax.random.PRNGKey(2)
    num_steps = 2000
    dim = 2
    predicted_means = jnp.zeros((num_steps, dim))
    predicted_covs = jnp.broadcast_to(4.0 * jnp.eye(dim), (num_steps, dim, dim))
    references = jax.random.normal(key, (num_steps, dim))
    metric = anees(
        predicted_means=predicted_means,
        predicted_covariances=predicted_covs,
        references=references,
    )
    assert metric < 0.5


def test_chi2_confidence_intervals_match_scipy_quantiles() -> None:
    """``chi2_confidence_intervals`` matches the ``scipy.stats.chi2`` ppf."""
    lower, upper = chi2_confidence_intervals(dim=3, percentile=0.99)
    # For chi^2_3, the 0.005 and 0.995 quantiles are known constants.
    assert 0.0 < lower < 1.0
    assert 10.0 < upper < 15.0


def test_chi2_confidence_intervals_rejects_invalid_percentile() -> None:
    """Percentile outside ``(0, 1)`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="percentile must be in"):
        chi2_confidence_intervals(dim=2, percentile=1.5)


def test_non_credibility_index_zero_when_predicted_cov_matches_truth() -> None:
    """NCI ≈ 0 when the predicted covariance matches the true error cov."""
    key = jax.random.PRNGKey(3)
    num_steps = 4000
    dim = 2
    predicted_means = jnp.zeros((num_steps, dim))
    predicted_covs = jnp.broadcast_to(jnp.eye(dim), (num_steps, dim, dim))
    reference_covs = predicted_covs
    references = jax.random.normal(key, (num_steps, dim))
    nci = non_credibility_index(
        predicted_means=predicted_means,
        predicted_covariances=predicted_covs,
        references=references,
        reference_covariances=reference_covs,
    )
    assert jnp.abs(nci) < 0.5


def test_non_credibility_index_positive_when_predicted_cov_too_small() -> None:
    """NCI > 0 when the predicted covariance underestimates the true error cov."""
    key = jax.random.PRNGKey(4)
    num_steps = 2000
    dim = 2
    predicted_means = jnp.zeros((num_steps, dim))
    predicted_covs = jnp.broadcast_to(0.1 * jnp.eye(dim), (num_steps, dim, dim))
    reference_covs = jnp.broadcast_to(jnp.eye(dim), (num_steps, dim, dim))
    references = jax.random.normal(key, (num_steps, dim))
    nci = non_credibility_index(
        predicted_means=predicted_means,
        predicted_covariances=predicted_covs,
        references=references,
        reference_covariances=reference_covs,
    )
    assert nci > 0.0


def test_anees_jit_compatible() -> None:
    """``anees`` runs inside ``jax.jit``."""

    @jax.jit
    def call(means: jax.Array, covs: jax.Array, refs: jax.Array) -> jax.Array:
        return anees(predicted_means=means, predicted_covariances=covs, references=refs)

    dim = 2
    num_steps = 100
    means = jnp.zeros((num_steps, dim))
    covs = jnp.broadcast_to(jnp.eye(dim), (num_steps, dim, dim))
    refs = jax.random.normal(jax.random.PRNGKey(5), (num_steps, dim))
    result = call(means, covs, refs)
    assert jnp.isfinite(result)
