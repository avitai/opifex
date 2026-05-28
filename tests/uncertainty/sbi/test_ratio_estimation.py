"""Tests for :class:`opifex.uncertainty.sbi.ratio_estimation.NeuralRatioEstimator`.

NRE fits a classifier ``f(theta, x)`` to distinguish joint samples
``(theta, x) ~ p(theta) p(x|theta)`` from marginal pairs
``(theta', x) ~ p(theta) p(x)``. The classifier logits approximate the
log density ratio ``log r(theta, x) = log p(x|theta) - log p(x)``, and
posterior samples follow from MCMC on ``log r(theta, x_obs) + log
prior(theta)`` — routed through ``BlackJAXBackend``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.sbi.ratio_estimation import (
    NeuralRatioEstimator,
    NREState,
)
from opifex.uncertainty.sbi.simulators import Simulator
from opifex.uncertainty.types import PredictiveDistribution


_TOY_DIM = 2
_TOY_SIGMA_LIK = 0.5
_TOY_POST_SCALE = 1.0 / (1.0 + _TOY_SIGMA_LIK**2)


def _gaussian_linear_simulator() -> Simulator:
    def prior_sampler(rng: jax.Array, num_samples: int) -> jax.Array:
        return jax.random.normal(rng, (num_samples, _TOY_DIM))

    def simulate_fn(rng: jax.Array, theta: jax.Array) -> jax.Array:
        return theta + _TOY_SIGMA_LIK * jax.random.normal(rng, theta.shape)

    return Simulator(prior_sampler=prior_sampler, simulate_fn=simulate_fn)


def _gaussian_log_prior(theta: jax.Array) -> jax.Array:
    return -0.5 * jnp.sum(theta * theta)


def test_nre_fit_returns_typed_state() -> None:
    est = NeuralRatioEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=30)
    sim = _gaussian_linear_simulator()
    fitted = est.fit(sim, num_simulations=64, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))
    assert fitted.state is not None
    assert isinstance(fitted.state, NREState)
    assert fitted.state.train_losses.shape == (30,)


def test_nre_fit_deterministic_under_fixed_key() -> None:
    sim = _gaussian_linear_simulator()
    a = NeuralRatioEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=20).fit(
        sim, num_simulations=64, rngs=nnx.Rngs(sbi_simulate=9, sbi_train=10)
    )
    b = NeuralRatioEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=20).fit(
        sim, num_simulations=64, rngs=nnx.Rngs(sbi_simulate=9, sbi_train=10)
    )
    assert a.state is not None and b.state is not None
    assert jnp.allclose(a.state.train_losses, b.state.train_losses, atol=1e-6)


def test_nre_predict_distribution_returns_predictive_distribution() -> None:
    sim = _gaussian_linear_simulator()
    fitted = NeuralRatioEstimator(
        theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=50, mcmc_samples=100, mcmc_burnin=20
    ).fit(sim, num_simulations=128, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))
    x_obs = jnp.ones(_TOY_DIM)
    pred = fitted.predict_distribution(
        x_obs, rngs=nnx.Rngs(sbi_sample=0), num_samples=100, log_prior=_gaussian_log_prior
    )
    assert isinstance(pred, PredictiveDistribution)
    assert pred.mean.shape == (_TOY_DIM,)
    assert dict(pred.metadata).get("method") == "nre"


def test_nre_posterior_mean_matches_analytic_gaussian_toy() -> None:
    sim = _gaussian_linear_simulator()
    fitted = NeuralRatioEstimator(
        theta_dim=_TOY_DIM,
        x_dim=_TOY_DIM,
        num_steps=600,
        learning_rate=1e-3,
        hidden_dim=64,
        mcmc_samples=1000,
        mcmc_burnin=300,
        mcmc_method="nuts",
        mcmc_step_size=0.3,
    ).fit(sim, num_simulations=1500, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))
    x_obs = jnp.array([1.0, 1.0])
    pred = fitted.predict_distribution(
        x_obs, rngs=nnx.Rngs(sbi_sample=0), num_samples=1000, log_prior=_gaussian_log_prior
    )
    expected = _TOY_POST_SCALE * x_obs
    err = float(jnp.max(jnp.abs(pred.mean - expected)))
    assert err < 0.5, f"NRE mean {pred.mean} too far from analytic {expected} (err={err})"


def test_nre_optional_backend_unavailable_raises_import_error() -> None:
    sim = _gaussian_linear_simulator()
    est = NeuralRatioEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, backend="flowMC")
    with pytest.raises(ImportError, match=r"install flowMC|use the default Artifex-flow backend"):
        est.fit(sim, num_simulations=8, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))


def test_nre_predict_before_fit_raises() -> None:
    est = NeuralRatioEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM)
    with pytest.raises(RuntimeError, match="before fit"):
        est.predict_distribution(
            jnp.ones(_TOY_DIM),
            rngs=nnx.Rngs(sbi_sample=0),
            num_samples=8,
            log_prior=_gaussian_log_prior,
        )
