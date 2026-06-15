"""Tests for the Bayesian Monte Carlo baseline integration loop.

Bayesian Monte Carlo integrates a function ``f`` against a probability
measure ``π`` by Monte Carlo sampling from ``π`` and returning both the
sample mean (point estimate) and the standard-error variance (epistemic
uncertainty in the integral). It is the simplest, no-GP baseline against
which Bayesian quadrature methods (WSABI-L, vanilla BQ) are compared.

Canonical reference:
* ``../emukit/emukit/quadrature/loop/bayesian_monte_carlo_loop.py``
  (line 18) — ``BayesianMonteCarlo``.
* Rasmussen & Ghahramani 2003 — *Bayesian Monte Carlo*, NeurIPS.

References
----------
* Rasmussen, C. E. & Ghahramani, Z. 2003 — *Bayesian Monte Carlo*,
  Advances in Neural Information Processing Systems 16.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.quadrature import (
    bayesian_monte_carlo,
    IntegralEstimate,
)


def test_integral_estimate_is_frozen_dataclass_with_mean_and_variance() -> None:
    """``IntegralEstimate`` carries a scalar mean and standard-error variance."""
    import dataclasses as dc

    estimate = IntegralEstimate(mean=jnp.asarray(1.5), variance=jnp.asarray(0.1), num_samples=100)
    assert dc.is_dataclass(estimate)
    assert estimate.mean.shape == ()
    assert estimate.variance.shape == ()
    assert estimate.num_samples == 100
    with pytest.raises(dc.FrozenInstanceError):
        estimate.mean = jnp.asarray(0.0)  # type: ignore[misc]


def test_bayesian_monte_carlo_constant_function_returns_constant_mean() -> None:
    """``∫ 1 dπ = 1`` for any proper measure π."""

    def constant_one(_x: jax.Array) -> jax.Array:
        return jnp.asarray(1.0)

    samples = jax.random.normal(jax.random.PRNGKey(0), (1000, 2))
    estimate = bayesian_monte_carlo(integrand=constant_one, samples=samples)
    assert jnp.allclose(estimate.mean, jnp.asarray(1.0), atol=1e-5)
    assert estimate.num_samples == 1000


def test_bayesian_monte_carlo_linear_integrand_recovers_first_moment() -> None:
    """``E_π[x_0] ≈ 0`` for samples from a centred Gaussian."""
    samples = jax.random.normal(jax.random.PRNGKey(1), (5000, 3))
    estimate = bayesian_monte_carlo(integrand=lambda x: x[0], samples=samples)
    assert jnp.abs(estimate.mean) < 0.1
    # Standard error of the mean shrinks as 1/sqrt(N).
    assert estimate.variance < 0.01


def test_bayesian_monte_carlo_variance_shrinks_with_sample_count() -> None:
    """Variance of the integral estimate decreases as ``N`` grows."""

    def quadratic(x: jax.Array) -> jax.Array:
        return jnp.sum(x**2)

    key = jax.random.PRNGKey(2)
    small_samples = jax.random.normal(key, (200, 2))
    large_samples = jax.random.normal(key, (5000, 2))
    small = bayesian_monte_carlo(integrand=quadratic, samples=small_samples)
    large = bayesian_monte_carlo(integrand=quadratic, samples=large_samples)
    assert large.variance < small.variance


def test_bayesian_monte_carlo_is_jit_compatible() -> None:
    """``bayesian_monte_carlo`` works inside ``jax.jit``."""

    def integrand(x: jax.Array) -> jax.Array:
        return jnp.sum(x**2)

    @jax.jit
    def call(samples: jax.Array) -> IntegralEstimate:
        return bayesian_monte_carlo(integrand=integrand, samples=samples)

    samples = jax.random.normal(jax.random.PRNGKey(3), (500, 2))
    estimate = call(samples)
    assert jnp.isfinite(estimate.mean)
    assert jnp.isfinite(estimate.variance)


def test_bayesian_monte_carlo_rejects_empty_sample_set() -> None:
    """A zero-sample call is degenerate and must raise."""

    def integrand(x: jax.Array) -> jax.Array:
        return jnp.sum(x)

    with pytest.raises(ValueError, match="at least 2"):
        bayesian_monte_carlo(integrand=integrand, samples=jnp.zeros((1, 2)))


def test_bayesian_monte_carlo_matches_analytic_gaussian_integral() -> None:
    """For ``f(x) = x^2`` and ``π = N(0, 1)``, the integral is exactly 1."""

    def quadratic(x: jax.Array) -> jax.Array:
        return jnp.sum(x**2)

    samples = jax.random.normal(jax.random.PRNGKey(4), (20000, 1))
    estimate = bayesian_monte_carlo(integrand=quadratic, samples=samples)
    # Expected: E[x^2] = 1 (variance of standard normal).
    assert jnp.abs(estimate.mean - 1.0) < 0.05
