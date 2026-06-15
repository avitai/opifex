r"""Posterior Linearisation on Markov GPs — Slice 30 (Task 11.2).

Posterior Linearisation (PL, Garcia-Fernandez, Tronarp, Sarkka 2018)
is an iterated smoothing algorithm based on Statistical Linear
Regression (SLR): each iteration linearises the likelihood
``p(y | f)`` against the current posterior moments via cubature,
yielding a Gaussian linearisation ``y ~ N(A f + b, omega)`` with

* ``mu_i  = E_{q(f_i)}[E[y_i | f_i]]``,
* ``omega_i = E_q[Var[y | f]] + Var_q[E[y | f]] - C^T cov^-1 C`` —
  the residual response covariance after subtracting the
  cross-covariance via the linear regression,
* ``A_i   = dmu_i / dmean_f_i`` — the SLR design slope, obtained by
  ``jax.grad`` on the cubature-evaluated ``mu_i``.

A pseudo-Gaussian Kalman pass on the linearised model then refines
the posterior, and the process iterates until convergence. For
Gaussian likelihood the linearisation is exact (``A = 1, b = 0,
omega = sigma_squared``) so PL recovers the conjugate Kalman path in one
iteration — verified by the slice 30 cross-check.

References
----------
* Garcia-Fernandez, Tronarp, Sarkka 2018 — *Gaussian process
  classification using posterior linearisation*, IEEE SPL.
* Sarkka 2013 — *Bayesian Filtering and Smoothing*, CUP, §6
  (Iterated Extended Kalman Smoother).
* Wilkinson, Solin, Adam 2020+ — ``bayesnewton/inference.py``
  ``PosteriorLinearisation`` (PRIMARY reference).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import matern32_kernel as state_space_matern32_kernel
from opifex.uncertainty.types import PredictiveDistribution


# -----------------------------------------------------------------------------
# Generic fit/predict surface
# -----------------------------------------------------------------------------


def test_fit_markov_pl_gp_returns_finite_smoothed_state() -> None:
    """PL iterations converge to finite smoothed moments."""
    from opifex.uncertainty.markov import fit_markov_pl_gp, MarkovPLGPState

    times = jnp.sort(
        jax.random.uniform(jax.random.PRNGKey(0), (20,), minval=0.0, maxval=2.0 * jnp.pi)
    )
    targets = jnp.sign(jnp.sin(2.0 * times))

    def bernoulli_conditional_moments(f: jax.Array) -> tuple[jax.Array, jax.Array]:
        # ±1 labels, logit link: E[y|f] = 2 sigmoid(f) - 1, Var[y|f] = 1 - mean²
        mean = 2.0 * jax.nn.sigmoid(f) - 1.0
        variance = 1.0 - mean**2
        return mean, variance

    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    fitted = fit_markov_pl_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        conditional_moments_fn=bernoulli_conditional_moments,
        num_iterations=20,
    )
    assert isinstance(fitted, MarkovPLGPState)
    assert fitted.smoothed_means.shape == (times.shape[0],)
    assert fitted.smoothed_variances.shape == (times.shape[0],)
    assert jnp.all(jnp.isfinite(fitted.smoothed_means))
    assert jnp.all(fitted.smoothed_variances > 0.0)


def test_predict_markov_pl_gp_returns_predictive_distribution() -> None:
    """Predict at held-out times returns a populated PredictiveDistribution."""
    from opifex.uncertainty.markov import fit_markov_pl_gp, predict_markov_pl_gp

    times = jnp.linspace(0.0, 4.0, 18)
    targets = jnp.sign(jnp.sin(2.0 * times))

    def bernoulli_conditional_moments(f: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = 2.0 * jax.nn.sigmoid(f) - 1.0
        variance = 1.0 - mean**2
        return mean, variance

    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_markov_pl_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        conditional_moments_fn=bernoulli_conditional_moments,
        num_iterations=15,
    )
    times_test = jnp.linspace(0.5, 3.5, 6)
    predictive = predict_markov_pl_gp(state=state, times_test=times_test)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert predictive.mean.shape == (6,)
    assert jnp.all(jnp.isfinite(predictive.mean))
    assert jnp.all(predictive.variance > 0.0)


def test_markov_pl_full_pipeline_is_jit_compatible() -> None:
    """Fit + predict compile end-to-end under ``jax.jit``."""
    from opifex.uncertainty.markov import fit_markov_pl_gp, predict_markov_pl_gp

    times = jnp.linspace(0.0, 4.0, 14)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)

    def bernoulli_conditional_moments(f: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = 2.0 * jax.nn.sigmoid(f) - 1.0
        variance = 1.0 - mean**2
        return mean, variance

    @jax.jit
    def fit_predict(t: jax.Array, y: jax.Array) -> jax.Array:
        state = fit_markov_pl_gp(
            times=t,
            observations=y,
            state_space_kernel=kernel,
            conditional_moments_fn=bernoulli_conditional_moments,
            num_iterations=10,
        )
        predictive = predict_markov_pl_gp(state=state, times_test=jnp.linspace(0.0, 4.0, 5))
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(times, targets)
    assert out.shape == (5,)
    assert jnp.all(jnp.isfinite(out))


# -----------------------------------------------------------------------------
# Gaussian cross-check: PL reduces to exact conjugate Kalman
# -----------------------------------------------------------------------------


def test_markov_pl_gaussian_likelihood_matches_markov_laplace_gaussian() -> None:
    r"""For Gaussian likelihood, SLR linearisation is exact in one step.

    ``E[y|f] = f`` and ``Var[y|f] = sigma_squared`` so the SLR slope is
    identically 1 and ``omega = sigma_squared``. PL then reduces to
    exact conjugate Kalman in one iteration — must coincide with the
    Markov-Laplace Gaussian path.
    """
    from opifex.uncertainty.markov import (
        fit_gaussian_markov_laplace_gp,
        fit_gaussian_markov_pl_gp,
    )

    times = jnp.linspace(0.0, 3.5, 20)
    observations = jnp.sin(2.0 * times) + 0.05 * jax.random.normal(jax.random.PRNGKey(17), (20,))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    laplace_state = fit_gaussian_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=0.05,
        num_iterations=5,
    )
    pl_state = fit_gaussian_markov_pl_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=0.05,
        num_iterations=3,
    )
    assert jnp.allclose(pl_state.smoothed_means, laplace_state.smoothed_means, atol=1e-4)
    assert jnp.allclose(pl_state.smoothed_variances, laplace_state.smoothed_variances, atol=1e-4)


# -----------------------------------------------------------------------------
# Per-likelihood PL wrappers
# -----------------------------------------------------------------------------


def test_fit_bernoulli_markov_pl_gp_returns_class_probabilities_in_unit_interval() -> None:
    """Bernoulli PL predict yields class probabilities in [0, 1]."""
    from opifex.uncertainty.markov import (
        fit_bernoulli_markov_pl_gp,
        predict_bernoulli_markov_pl_gp,
    )

    times = jnp.linspace(0.0, 4.0, 16)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_bernoulli_markov_pl_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        num_iterations=15,
    )
    predictive = predict_bernoulli_markov_pl_gp(state=state, times_test=jnp.linspace(0.0, 4.0, 8))
    assert jnp.all(predictive.mean >= 0.0)
    assert jnp.all(predictive.mean <= 1.0)


def test_fit_poisson_markov_pl_gp_recovers_positive_intensity() -> None:
    """Poisson PL yields strictly positive predictive intensity."""
    from opifex.uncertainty.markov import (
        fit_poisson_markov_pl_gp,
        predict_poisson_markov_pl_gp,
    )

    key = jax.random.PRNGKey(19)
    times = jnp.sort(jax.random.uniform(key, (20,), minval=0.0, maxval=2.0 * jnp.pi))
    rate = jnp.exp(jnp.sin(2.0 * times) + 1.0)
    observations = jax.random.poisson(jax.random.PRNGKey(20), rate).astype(jnp.float32)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    state = fit_poisson_markov_pl_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        num_iterations=20,
    )
    predictive = predict_poisson_markov_pl_gp(state=state, times_test=jnp.linspace(0.5, 5.5, 8))
    assert predictive.variance is not None
    assert jnp.all(predictive.mean > 0.0)


def test_markov_pl_state_advertises_pl_estimator_metadata() -> None:
    """``predict_markov_pl_gp`` metadata advertises the PL inference path."""
    from opifex.uncertainty.markov import fit_markov_pl_gp, predict_markov_pl_gp

    times = jnp.linspace(0.0, 4.0, 12)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)

    def bernoulli_conditional_moments(f: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = 2.0 * jax.nn.sigmoid(f) - 1.0
        variance = 1.0 - mean**2
        return mean, variance

    state = fit_markov_pl_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        conditional_moments_fn=bernoulli_conditional_moments,
        num_iterations=8,
    )
    predictive = predict_markov_pl_gp(state=state, times_test=jnp.zeros((3,)))
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "markov_pl_gp"
    assert metadata.get("paper") is not None
