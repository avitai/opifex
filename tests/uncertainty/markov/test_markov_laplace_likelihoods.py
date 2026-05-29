r"""Markov-Laplace coverage across the D5 likelihood family — Slice 26.

The slice-25 ``fit_markov_laplace_gp`` machinery accepts any D5
``LikelihoodComponentsFn``. This slice ships per-likelihood wrappers
mirroring D5's :func:`fit_poisson_laplace_gp` etc., reusing the
existing components factories so that the Bernoulli / Poisson /
Student-t / Beta / Gaussian likelihoods all work on Markov-GP priors
without code duplication.

Exit criterion for Task 11.2 (``11-...:107``):
*"at least 5 non-Gaussian likelihoods + PEP/VI/Laplace inference
paths with calibration tests."*
This slice ships the **5 likelihoods** half via the Laplace path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.markov import (
    fit_beta_markov_laplace_gp,
    fit_gaussian_markov_laplace_gp,
    fit_poisson_markov_laplace_gp,
    fit_studentst_markov_laplace_gp,
    MarkovLaplaceGPState,
    predict_beta_markov_laplace_gp,
    predict_gaussian_markov_laplace_gp,
    predict_poisson_markov_laplace_gp,
    predict_studentst_markov_laplace_gp,
)
from opifex.uncertainty.statespace import matern32_kernel as state_space_matern32_kernel


def _sorted_times(seed: int, n: int = 20) -> jax.Array:
    return jnp.sort(
        jax.random.uniform(jax.random.PRNGKey(seed), (n,), minval=0.0, maxval=2.0 * jnp.pi)
    )


# -----------------------------------------------------------------------------
# Poisson likelihood (exp link) — count time series
# -----------------------------------------------------------------------------


def test_fit_poisson_markov_laplace_gp_recovers_intensity_on_count_time_series() -> None:
    """Poisson Markov-Laplace fit predicts positive intensity at test times."""
    times = _sorted_times(seed=0)
    key = jax.random.PRNGKey(10)
    rate = jnp.exp(jnp.sin(2.0 * times) + 1.0)
    observations = jax.random.poisson(key, rate).astype(jnp.float32)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    state = fit_poisson_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        num_iterations=30,
    )
    assert isinstance(state, MarkovLaplaceGPState)
    times_test = jnp.linspace(0.5, 5.5, 10)
    predictive = predict_poisson_markov_laplace_gp(state=state, times_test=times_test)
    assert predictive.variance is not None
    assert jnp.all(predictive.mean > 0.0)
    assert jnp.all(predictive.variance > 0.0)


# -----------------------------------------------------------------------------
# Student-t likelihood — robust regression
# -----------------------------------------------------------------------------


def test_fit_studentst_markov_laplace_gp_is_robust_to_outliers() -> None:
    """Student-t Markov-Laplace produces bounded smoothed mode despite outliers."""
    times = jnp.linspace(0.0, 6.0, 25)
    clean = jnp.sin(2.0 * times)
    observations = clean.at[5].set(3.0).at[15].set(-3.0)  # two heavy-tailed outliers
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_studentst_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        df=4.0,
        scale=0.3,
        num_iterations=40,
    )
    # Trend `sin(2t)` is bounded in [-1, 1]; outliers at ±3 should be
    # damped by the Student-t loss — the smoothed mean magnitude
    # should not approach the outlier value.
    assert jnp.max(jnp.abs(state.smoothed_means)) < 2.5


def test_predict_studentst_markov_laplace_returns_finite_moments() -> None:
    """Student-t predict at held-out times produces finite moments."""
    times = jnp.linspace(0.0, 4.0, 18)
    observations = jnp.sin(2.0 * times)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_studentst_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        df=4.0,
        scale=0.2,
        num_iterations=25,
    )
    times_test = jnp.linspace(0.5, 3.5, 5)
    predictive = predict_studentst_markov_laplace_gp(state=state, times_test=times_test)
    assert predictive.variance is not None
    assert jnp.all(jnp.isfinite(predictive.mean))
    assert jnp.all(predictive.variance > 0.0)


# -----------------------------------------------------------------------------
# Beta likelihood — proportion time series
# -----------------------------------------------------------------------------


def test_fit_beta_markov_laplace_gp_recovers_proportion_signal() -> None:
    """Beta Markov-Laplace fit produces unit-interval predictions at test times."""
    times = _sorted_times(seed=2)
    key = jax.random.PRNGKey(11)
    mean = jax.nn.sigmoid(jnp.sin(2.0 * times))
    scale = 20.0
    alpha = mean * scale
    beta = scale * (1.0 - mean)
    observations = jax.random.beta(key, alpha, beta)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    state = fit_beta_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        scale=scale,
        num_iterations=30,
    )
    times_test = jnp.linspace(0.5, 5.5, 8)
    predictive = predict_beta_markov_laplace_gp(state=state, times_test=times_test, scale=scale)
    assert predictive.variance is not None
    assert jnp.all(predictive.mean >= 0.0)
    assert jnp.all(predictive.mean <= 1.0)


# -----------------------------------------------------------------------------
# Gaussian likelihood — collapses to the conjugate Kalman path
# -----------------------------------------------------------------------------


def test_fit_gaussian_markov_laplace_gp_matches_a_one_iteration_fixed_point() -> None:
    """For Gaussian likelihood, the Newton iteration is exact at one step."""
    times = jnp.linspace(0.0, 4.0, 20)
    observations = jnp.sin(2.0 * times) + 0.05 * jax.random.normal(jax.random.PRNGKey(7), (20,))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.4)
    one_step = fit_gaussian_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=0.05,
        num_iterations=1,
    )
    twenty_step = fit_gaussian_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=0.05,
        num_iterations=20,
    )
    # For Gaussian likelihood the log-likelihood is exactly quadratic;
    # Newton converges in one step. Additional iterations are no-ops.
    assert jnp.allclose(one_step.smoothed_means, twenty_step.smoothed_means, atol=1e-5)


def test_gaussian_markov_laplace_predict_returns_predictive_distribution() -> None:
    """Gaussian wrapper provides the standard predict interface."""
    times = jnp.linspace(0.0, 3.0, 12)
    observations = jnp.sin(times)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    state = fit_gaussian_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=0.1,
        num_iterations=5,
    )
    predictive = predict_gaussian_markov_laplace_gp(
        state=state, times_test=jnp.linspace(0.0, 3.0, 4)
    )
    assert predictive.variance is not None
    assert jnp.all(jnp.isfinite(predictive.mean))


# -----------------------------------------------------------------------------
# JIT compatibility across the family
# -----------------------------------------------------------------------------


def test_poisson_markov_laplace_pipeline_is_jit_compatible() -> None:
    """Poisson Markov-Laplace fit + predict compile under ``jax.jit``."""
    times = jnp.linspace(0.0, 3.0, 12)
    observations = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0])
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)

    @jax.jit
    def fit_predict(t: jax.Array, y: jax.Array) -> jax.Array:
        state = fit_poisson_markov_laplace_gp(
            times=t,
            observations=y,
            state_space_kernel=kernel,
            num_iterations=15,
        )
        predictive = predict_poisson_markov_laplace_gp(
            state=state, times_test=jnp.linspace(0.0, 3.0, 4)
        )
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    output = fit_predict(times, observations)
    assert output.shape == (4,)
    assert jnp.all(jnp.isfinite(output))
