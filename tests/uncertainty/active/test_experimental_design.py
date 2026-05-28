"""Bayesian experimental design tests for Task 8.3.

* ``expected_information_gain`` matches the closed-form linear-Gaussian
  EIG: ``0.5 * log(1 + sigma_prior^2 / sigma_noise^2)`` for the trivial
  scalar design ``y = theta + noise``.
* ``bayesian_experimental_design_loop`` monotonically reduces predictive
  variance over rounds on a controlled synthetic regression problem with
  a deterministic oracle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.active.acquisition import (
    AcquisitionStrategy,
    upper_confidence_bound,
)
from opifex.uncertainty.active.experimental_design import (
    bayesian_experimental_design_loop,
    BayesianExperimentalDesignResult,
    expected_information_gain,
)
from opifex.uncertainty.types import PredictiveDistribution


# ---------------------------------------------------------------------------
# EIG
# ---------------------------------------------------------------------------


def _linear_gaussian_model(design: jax.Array, theta_samples: jax.Array) -> jax.Array:
    """Return predictive samples ``y = design @ theta`` for the linear-Gaussian case.

    ``design`` has shape ``(d,)`` and ``theta_samples`` has shape
    ``(num_samples, d)``. Output has shape ``(num_samples,)``.
    """
    return theta_samples @ design


class TestExpectedInformationGain:
    def test_linear_gaussian_scalar_design_matches_closed_form(self) -> None:
        """Closed-form EIG for ``y = theta + eps``, ``theta ~ N(0, sigma_p^2)``."""
        rng = jax.random.PRNGKey(0)
        sigma_prior = 2.0
        sigma_noise = 0.5
        num_prior = 8000
        prior_samples = sigma_prior * jax.random.normal(rng, (num_prior, 1))
        design = jnp.array([1.0])

        rngs = nnx.Rngs(active_eig=0)
        eig = expected_information_gain(
            model=_linear_gaussian_model,
            design=design,
            prior_samples=prior_samples,
            noise_std=sigma_noise,
            rngs=rngs,
        )

        expected = 0.5 * jnp.log(1.0 + (sigma_prior**2) / (sigma_noise**2))
        assert float(eig) == pytest.approx(float(expected), rel=0.1)

    def test_zero_design_gives_zero_information(self) -> None:
        """A design that doesn't depend on theta yields ~0 EIG."""
        rng = jax.random.PRNGKey(1)
        prior_samples = jax.random.normal(rng, (1000, 2))
        design = jnp.array([0.0, 0.0])

        rngs = nnx.Rngs(active_eig=0)
        eig = expected_information_gain(
            model=_linear_gaussian_model,
            design=design,
            prior_samples=prior_samples,
            noise_std=1.0,
            rngs=rngs,
        )
        assert float(eig) == pytest.approx(0.0, abs=5e-2)


# ---------------------------------------------------------------------------
# BO loop driver
# ---------------------------------------------------------------------------


class _MockGPSurrogate:
    """A tiny mock that pretends to be a Bayesian regression surrogate.

    Backs a posterior over a 1-D function ``f(x) = sin(2 pi x)`` observed
    with Gaussian noise. After each ``update`` the surrogate's predictive
    variance at *every* candidate decreases monotonically (we count
    observations and scale variance by ``1 / (1 + N)``).
    """

    def __init__(self) -> None:
        self.num_observations = 0

    def predict(self, candidates: jax.Array) -> PredictiveDistribution:
        mean = jnp.sin(2.0 * jnp.pi * candidates).squeeze(-1)
        variance = jnp.full_like(mean, 1.0 / (1.0 + self.num_observations))
        return PredictiveDistribution(mean=mean, variance=variance)

    def update(self, x: jax.Array, y: jax.Array) -> None:
        self.num_observations += int(x.shape[0])


class TestBayesianExperimentalDesignLoop:
    def test_loop_monotonically_reduces_predictive_variance(self) -> None:
        surrogate = _MockGPSurrogate()
        candidates = jnp.linspace(0.0, 1.0, 32).reshape(-1, 1)
        oracle = lambda x: jnp.sin(2.0 * jnp.pi * x).squeeze(-1)  # deterministic
        rngs = nnx.Rngs(active_acquire=0)

        result = bayesian_experimental_design_loop(
            surrogate=surrogate,
            candidates=candidates,
            oracle=oracle,
            acquisition=lambda pd: upper_confidence_bound(pd, beta=2.0),
            num_rounds=5,
            rngs=rngs,
        )

        assert isinstance(result, BayesianExperimentalDesignResult)
        variances = result.history_variance
        # Mean predictive variance must be non-increasing over rounds.
        for prev, curr in zip(variances[:-1], variances[1:], strict=True):
            assert curr <= prev + 1e-7

    def test_loop_records_acquired_indices(self) -> None:
        surrogate = _MockGPSurrogate()
        candidates = jnp.linspace(-1.0, 1.0, 16).reshape(-1, 1)
        oracle = lambda x: x.squeeze(-1)
        rngs = nnx.Rngs(active_acquire=0)

        result = bayesian_experimental_design_loop(
            surrogate=surrogate,
            candidates=candidates,
            oracle=oracle,
            acquisition=lambda pd: upper_confidence_bound(pd, beta=1.5),
            num_rounds=3,
            rngs=rngs,
        )

        assert len(result.acquired_indices) == 3
        # All indices must be valid candidates.
        for idx in result.acquired_indices:
            assert 0 <= int(idx) < candidates.shape[0]
