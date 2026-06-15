r"""Tests for heteroscedastic-Gaussian exact GP regression.

A heteroscedastic-Gaussian GP places a per-observation noise scale
``σ_i`` on each training target so the noise covariance is the
*diagonal* matrix ``diag(σ_1², …, σ_n²)`` instead of the scalar
``σ² I`` consumed by :func:`fit_exact_gp`. The Cholesky algorithm is
unchanged except that the right-hand-side perturbation becomes
``diag(σ_i²)``; the closed-form predictive moments

    α = (K + diag(σ_i²))^{-1} y,
    mean(X*) = K(X*, X) α,
    var(X*)  = K(X*, X*) - K(X*, X) (K + diag(σ_i²))^{-1} K(X, X*),

follow directly from RW06 §2.2 by substitution. When every ``σ_i = σ``
the heteroscedastic GP collapses to the existing
:func:`fit_exact_gp` exactly.

References
----------
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes for
  Machine Learning*, MIT Press; §2.2 (homoscedastic baseline that
  this heteroscedastic variant generalises).
* GPJax ``likelihoods.py:HeteroscedasticGaussian`` — reference
  implementation (adapter-only; not imported).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    fit_exact_gp,
    fit_heteroscedastic_exact_gp,
    predict_exact_gp,
    rbf_kernel,
)
from opifex.uncertainty.types import PredictiveDistribution


def test_heteroscedastic_fit_with_constant_noise_matches_exact_gp() -> None:
    """Constant per-observation noise reproduces ``fit_exact_gp`` exactly."""
    x_train = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1))
    state_homo = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.07,
    )
    state_hetero = fit_heteroscedastic_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=jnp.full((6,), 0.07),
    )
    assert jnp.allclose(state_homo.cholesky, state_hetero.cholesky, atol=1e-6)
    assert jnp.allclose(state_homo.alpha, state_hetero.alpha, atol=1e-6)


def test_heteroscedastic_predict_matches_closed_form() -> None:
    """Predictive moments equal the direct closed-form on a 4-point toy."""
    x_train = jax.random.normal(jax.random.PRNGKey(0), (4, 1))
    y_train = jax.random.normal(jax.random.PRNGKey(1), (4,))
    x_test = jax.random.normal(jax.random.PRNGKey(2), (3, 1))
    noise_std = jnp.asarray([0.05, 0.1, 0.2, 0.4])
    lengthscale, output_scale = 0.5, 1.0

    state = fit_heteroscedastic_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
    )
    predictive = predict_exact_gp(state=state, x_test=x_test)

    k_train = rbf_kernel(x_train, x_train, lengthscale=lengthscale, output_scale=output_scale)
    k_test = rbf_kernel(x_test, x_train, lengthscale=lengthscale, output_scale=output_scale)
    k_diag = jnp.full((3,), output_scale**2)
    k_train_inv = jnp.linalg.inv(k_train + jnp.diag(noise_std**2))
    expected_mean = k_test @ k_train_inv @ y_train
    expected_var = k_diag - jnp.sum((k_test @ k_train_inv) * k_test, axis=-1)

    assert predictive.variance is not None
    assert jnp.allclose(predictive.mean, expected_mean, atol=1e-5)
    assert jnp.allclose(predictive.variance, expected_var, atol=1e-5)


def test_heteroscedastic_high_noise_observation_loses_influence() -> None:
    """Inflating one observation's noise shrinks its influence on a nearby test point."""
    x_train = jnp.asarray([[-0.3], [0.0], [0.3]])
    y_train = jnp.asarray([0.0, 5.0, 0.0])
    x_test = jnp.asarray([[0.0]])

    low_noise = fit_heteroscedastic_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.5,
        output_scale=1.0,
        noise_std=jnp.asarray([0.01, 0.01, 0.01]),
    )
    high_centre_noise = fit_heteroscedastic_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.5,
        output_scale=1.0,
        noise_std=jnp.asarray([0.01, 5.0, 0.01]),
    )
    mean_low = float(predict_exact_gp(state=low_noise, x_test=x_test).mean[0])
    mean_high = float(predict_exact_gp(state=high_centre_noise, x_test=x_test).mean[0])
    # The high-noise centre observation contributes less; the predictive mean
    # at the centre stays far from 5.0 under high noise but is close under low noise.
    assert abs(mean_high) < abs(mean_low)


def test_heteroscedastic_fit_returns_exact_gp_state() -> None:
    """Returned state reuses :class:`ExactGPState` (predict path stays the same)."""
    from opifex.uncertainty.gp import ExactGPState

    state = fit_heteroscedastic_exact_gp(
        x_train=jnp.zeros((4, 1)),
        y_train=jnp.zeros((4,)),
        lengthscale=1.0,
        output_scale=1.0,
        noise_std=jnp.full((4,), 0.1),
    )
    assert isinstance(state, ExactGPState)


def test_heteroscedastic_fit_is_jit_compatible() -> None:
    """Heteroscedastic fit compiles end-to-end under ``jax.jit``."""
    x_train = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1))
    noise_std = jnp.linspace(0.05, 0.3, 6)
    x_test = jnp.linspace(-0.5, 0.5, 3).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, sigma: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_heteroscedastic_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.4,
            output_scale=1.0,
            noise_std=sigma,
        )
        pd = predict_exact_gp(state=state, x_test=x_q)
        assert pd.variance is not None
        return pd.mean + pd.variance

    out = fit_predict(x_train, y_train, noise_std, x_test)
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_heteroscedastic_rejects_nonpositive_noise() -> None:
    """All ``noise_std`` entries must be strictly positive."""
    with pytest.raises(ValueError, match="noise_std"):
        fit_heteroscedastic_exact_gp(
            x_train=jnp.zeros((3, 1)),
            y_train=jnp.zeros((3,)),
            lengthscale=1.0,
            output_scale=1.0,
            noise_std=jnp.asarray([0.1, 0.0, 0.1]),
        )


def test_heteroscedastic_rejects_shape_mismatched_noise() -> None:
    """``noise_std`` length must equal the training-set size."""
    with pytest.raises(ValueError, match="noise_std"):
        fit_heteroscedastic_exact_gp(
            x_train=jnp.zeros((4, 1)),
            y_train=jnp.zeros((4,)),
            lengthscale=1.0,
            output_scale=1.0,
            noise_std=jnp.asarray([0.1, 0.1, 0.1]),
        )


def test_heteroscedastic_predict_returns_predictive_distribution() -> None:
    """Predict path returns a ``PredictiveDistribution`` end-to-end."""
    state = fit_heteroscedastic_exact_gp(
        x_train=jnp.zeros((3, 1)),
        y_train=jnp.zeros((3,)),
        lengthscale=1.0,
        output_scale=1.0,
        noise_std=jnp.full((3,), 0.1),
    )
    predictive = predict_exact_gp(state=state, x_test=jnp.zeros((2, 1)))
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert predictive.mean.shape == (2,)
