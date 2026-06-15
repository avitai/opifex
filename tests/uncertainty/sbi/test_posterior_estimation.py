"""Tests for :class:`opifex.uncertainty.sbi.posterior_estimation.NeuralPosteriorEstimator`.

NPE fits a conditional density estimator ``q(theta | x)`` on
``(theta, x)`` simulation pairs, then samples ``q`` at a fixed observation
to approximate the posterior.

The default density-estimator backend wraps Artifex's
``ConditionalRealNVP``. Optional backends (``bijx``, ``sbiax``, ``flowMC``)
are routed via :class:`BackendRouter`; when not installed, requesting them
must raise :class:`ImportError` with the canonical install hint.

Numerical accuracy is tested on a Gaussian linear toy with known posterior:

    theta ~ N(0, 1),  x | theta ~ N(theta, sigma_lik^2)
    => theta | x ~ N(x * sigma_lik^{-2} / (1 + sigma_lik^{-2}), ...)

For ``sigma_lik = 0.5`` the analytic posterior mean is ``0.8 * x_obs``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.sbi.posterior_estimation import (
    NeuralPosteriorEstimator,
    NPEState,
)
from opifex.uncertainty.sbi.simulators import Simulator
from opifex.uncertainty.types import PredictiveDistribution


_TOY_DIM = 2
_TOY_SIGMA_LIK = 0.5
_TOY_POST_SCALE = 1.0 / (1.0 + _TOY_SIGMA_LIK**2)  # = 0.8


def _gaussian_linear_simulator() -> Simulator:
    """Gaussian linear toy with prior N(0, I_2) and likelihood N(theta, 0.25 I_2)."""

    def prior_sampler(rng: jax.Array, num_samples: int) -> jax.Array:
        return jax.random.normal(rng, (num_samples, _TOY_DIM))

    def simulate_fn(rng: jax.Array, theta: jax.Array) -> jax.Array:
        return theta + _TOY_SIGMA_LIK * jax.random.normal(rng, theta.shape)

    return Simulator(
        prior_sampler=prior_sampler,
        simulate_fn=simulate_fn,
        metadata=(("toy", "gaussian_linear"),),
    )


def test_npe_fit_returns_npe_state_with_array_fields() -> None:
    estimator = NeuralPosteriorEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=50)
    sim = _gaussian_linear_simulator()
    fitted = estimator.fit(sim, num_simulations=100, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))
    assert isinstance(fitted, NeuralPosteriorEstimator)
    assert fitted.state is not None
    assert isinstance(fitted.state, NPEState)
    # ``train_losses`` is an array field on the fitted state (pattern (B)).
    assert isinstance(fitted.state.train_losses, jax.Array)
    assert fitted.state.train_losses.shape == (50,)


def test_npe_fit_deterministic_under_fixed_key() -> None:
    sim = _gaussian_linear_simulator()
    a = NeuralPosteriorEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=20).fit(
        sim, num_simulations=64, rngs=nnx.Rngs(sbi_simulate=7, sbi_train=11)
    )
    b = NeuralPosteriorEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=20).fit(
        sim, num_simulations=64, rngs=nnx.Rngs(sbi_simulate=7, sbi_train=11)
    )
    assert a.state is not None and b.state is not None
    assert jnp.allclose(a.state.train_losses, b.state.train_losses, atol=1e-6)


def test_npe_predict_distribution_returns_predictive_distribution() -> None:
    sim = _gaussian_linear_simulator()
    fitted = NeuralPosteriorEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=50).fit(
        sim, num_simulations=128, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1)
    )
    x_obs = jnp.ones(_TOY_DIM)
    pred = fitted.predict_distribution(x_obs, rngs=nnx.Rngs(sbi_sample=0), num_samples=256)
    assert isinstance(pred, PredictiveDistribution)
    assert pred.mean.shape == (_TOY_DIM,)
    assert pred.samples is not None and pred.samples.shape == (256, _TOY_DIM)
    assert dict(pred.metadata).get("method") == "npe"


def test_npe_posterior_mean_matches_analytic_gaussian_toy() -> None:
    """Sanity: trained NPE posterior mean within tolerance of analytic mean."""
    sim = _gaussian_linear_simulator()
    fitted = NeuralPosteriorEstimator(
        theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=500, learning_rate=1e-3
    ).fit(sim, num_simulations=1000, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))
    x_obs = jnp.array([1.0, 1.0])
    pred = fitted.predict_distribution(x_obs, rngs=nnx.Rngs(sbi_sample=0), num_samples=2000)
    expected = _TOY_POST_SCALE * x_obs
    # Empirical NPE posterior mean within ~0.4 of analytic (training budget
    # is small; tolerance accommodates Monte Carlo + finite-step bias).
    err = float(jnp.max(jnp.abs(pred.mean - expected)))
    assert err < 0.4, f"NPE mean {pred.mean} too far from analytic {expected} (err={err})"


def test_npe_sequential_rounds_reduce_posterior_variance() -> None:
    """Sequential SBI: second round trained on prior-narrowed simulations
    should produce a tighter posterior (lower variance) at the same observation.
    """
    sim = _gaussian_linear_simulator()
    x_obs = jnp.array([1.0, 1.0])
    round1 = NeuralPosteriorEstimator(
        theta_dim=_TOY_DIM, x_dim=_TOY_DIM, num_steps=200, learning_rate=1e-3
    ).fit(sim, num_simulations=400, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))
    round2 = round1.refine_round(
        sim,
        observation=x_obs,
        num_simulations=400,
        num_steps=200,
        rngs=nnx.Rngs(sbi_simulate=2, sbi_train=3),
    )
    pred1 = round1.predict_distribution(x_obs, rngs=nnx.Rngs(sbi_sample=0), num_samples=1500)
    pred2 = round2.predict_distribution(x_obs, rngs=nnx.Rngs(sbi_sample=0), num_samples=1500)
    assert pred1.variance is not None and pred2.variance is not None
    # Sum-trace of the empirical covariance shrinks across rounds.
    tr1 = float(jnp.sum(pred1.variance))
    tr2 = float(jnp.sum(pred2.variance))
    assert tr2 < tr1, f"Sequential round did not shrink variance: {tr1=} -> {tr2=}"


def test_npe_optional_backend_unavailable_raises_import_error() -> None:
    """Requesting an uninstalled optional backend raises ``ImportError`` with hint."""
    estimator = NeuralPosteriorEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, backend="bijx")
    sim = _gaussian_linear_simulator()
    with pytest.raises(ImportError, match=r"install bijx|use the default Artifex-flow backend"):
        estimator.fit(sim, num_simulations=8, rngs=nnx.Rngs(sbi_simulate=0, sbi_train=1))


def test_npe_unknown_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        NeuralPosteriorEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM, backend="not-a-backend")


def test_npe_predict_before_fit_raises() -> None:
    estimator = NeuralPosteriorEstimator(theta_dim=_TOY_DIM, x_dim=_TOY_DIM)
    with pytest.raises(RuntimeError, match="before fit"):
        estimator.predict_distribution(
            jnp.ones(_TOY_DIM), rngs=nnx.Rngs(sbi_sample=0), num_samples=8
        )


def test_npe_state_validate_rejects_inconsistent_loss_history() -> None:
    state = NPEState(
        train_losses=jnp.full((4,), jnp.nan),
        num_simulations=jnp.asarray(0),
    )
    with pytest.raises(ValueError, match="train_losses"):
        state.validate()
