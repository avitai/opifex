"""Tests for :class:`opifex.uncertainty.sbi.likelihood_estimation.NeuralLikelihoodEstimator`.

NLE fits ``q(x | theta)`` (a conditional density over observations given
parameters) and samples the posterior by MCMC over ``log q(x_obs | theta)
+ log prior(theta)``. The MCMC step routes through ``BlackJAXBackend``
from Task 2.5 — no direct ``blackjax`` import in ``opifex.uncertainty.sbi``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.sbi.likelihood_estimation import (
    NeuralLikelihoodEstimator,
    NLEState,
)
from opifex.uncertainty.sbi.simulators import Simulator
from opifex.uncertainty.types import PredictiveDistribution


_TOY_DIM = 2
_TOY_SIGMA_LIK = 0.5
_TOY_POST_SCALE = 1.0 / (1.0 + _TOY_SIGMA_LIK**2)  # = 0.8


def _gaussian_linear_simulator() -> Simulator:
    def prior_sampler(rng: jax.Array, num_samples: int) -> jax.Array:
        return jax.random.normal(rng, (num_samples, _TOY_DIM))

    def simulate_fn(rng: jax.Array, theta: jax.Array) -> jax.Array:
        return theta + _TOY_SIGMA_LIK * jax.random.normal(rng, theta.shape)

    return Simulator(prior_sampler=prior_sampler, simulate_fn=simulate_fn)


def _gaussian_log_prior(theta: jax.Array) -> jax.Array:
    return -0.5 * jnp.sum(theta * theta)


def test_nle_fit_returns_typed_state() -> None:
    est = NeuralLikelihoodEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=30)
    sim = _gaussian_linear_simulator()
    fitted = est.fit(sim, num_simulations=64, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))
    assert fitted.state is not None
    assert isinstance(fitted.state, NLEState)
    assert fitted.state.train_losses.shape == (30,)


def test_nle_fit_deterministic_under_fixed_key() -> None:
    sim = _gaussian_linear_simulator()
    a = NeuralLikelihoodEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=20).fit(
        sim, num_simulations=64, rngs=nnx.Rngs(sbi_simulate=3, sbi_train=4)
    )
    b = NeuralLikelihoodEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=20).fit(
        sim, num_simulations=64, rngs=nnx.Rngs(sbi_simulate=3, sbi_train=4)
    )
    assert a.state is not None and b.state is not None
    assert jnp.allclose(a.state.train_losses, b.state.train_losses, atol=1e-6)


def test_nle_predict_distribution_returns_predictive_distribution() -> None:
    sim = _gaussian_linear_simulator()
    fitted = NeuralLikelihoodEstimator(
        theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=50, mcmc_samples=100, mcmc_burnin=20
    ).fit(sim, num_simulations=128, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))
    x_obs = jnp.ones(_TOY_DIM)
    pred = fitted.predict_distribution(
        x_obs, rngs=nnx.Rngs(sbi_sample=0), num_samples=100, log_prior=_gaussian_log_prior
    )
    assert isinstance(pred, PredictiveDistribution)
    assert pred.mean.shape == (_TOY_DIM,)
    assert dict(pred.metadata).get("method") == "nle"


def test_nle_posterior_mean_matches_analytic_gaussian_toy() -> None:
    sim = _gaussian_linear_simulator()
    fitted = NeuralLikelihoodEstimator(
        theta_dim=_TOY_DIM,
        x_dim=_TOY_DIM,
        num_steps=500,
        learning_rate=1e-3,
        mcmc_samples=800,
        mcmc_burnin=200,
        mcmc_method="nuts",
        mcmc_step_size=0.3,
    ).fit(sim, num_simulations=1000, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))
    x_obs = jnp.array([1.0, 1.0])
    pred = fitted.predict_distribution(
        x_obs, rngs=nnx.Rngs(sbi_sample=0), num_samples=800, log_prior=_gaussian_log_prior
    )
    expected = _TOY_POST_SCALE * x_obs
    err = float(jnp.max(jnp.abs(pred.mean - expected)))
    assert err < 0.4, f"NLE mean {pred.mean} too far from analytic {expected} (err={err})"


def test_nle_optional_backend_unavailable_raises_import_error() -> None:
    sim = _gaussian_linear_simulator()
    est = NeuralLikelihoodEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, backend="bijx")
    with pytest.raises(ImportError, match=r"install bijx|use the default Artifex-flow backend"):
        est.fit(sim, num_simulations=8, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))


def test_nle_predict_before_fit_raises() -> None:
    est = NeuralLikelihoodEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM)
    with pytest.raises(RuntimeError, match="before fit"):
        est.predict_distribution(
            jnp.ones(_TOY_DIM),
            rngs=nnx.Rngs(sbi_sample=0),
            num_samples=8,
            log_prior=_gaussian_log_prior,
        )
