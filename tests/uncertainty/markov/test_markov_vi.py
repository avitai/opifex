r"""Variational Inference on Markov GPs — Slice 27 (Task 11.2).

Conjugate-Computation VI (Khan & Lin 2017; Chang, Wilkinson, Khan,
Solin 2020 — *Fast variational learning in state-space GP models*).
Mirrors the slice-25 Markov-Laplace bridge but replaces the
mode-evaluated likelihood quadruple ``(grad, W)`` with the **expected**
quadruple under the current variational posterior ``q(f_t)``:

* ``E_grad_t = E_{q(f_t)}[∂ log p(y_t | f_t) / ∂ f_t]``,
* ``E_W_t   = -E_{q(f_t)}[∂² log p(y_t | f_t) / ∂ f_t²]``.

Both expectations are estimated by Gauss-Hermite quadrature over the
current marginal ``q(f_t) = N(mean_t, var_t)``. The expected pair
then plugs into the same pseudo-Gaussian Kalman linearisation used
by Markov-Laplace; iterating to convergence yields the conjugate-
computation-VI posterior. For Gaussian likelihood (linear-in-f
log-lik) the expected and mode-evaluated components coincide, so VI
and Laplace produce identical answers — the slice-27 cross-check.

References
----------
* Khan, Lin 2017 — *Conjugate-Computation Variational Inference*,
  ICML.
* Chang, Wilkinson, Khan, Solin 2020 — *Fast variational learning
  in state-space Gaussian process models*, ICML.
* Wilkinson, Solin, Adam 2020+ — ``bayesnewton/inference.py``
  ``VariationalInference`` (PRIMARY reference).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.gp.laplace_classification import (
    _bernoulli_log_likelihood_components,
)
from opifex.uncertainty.statespace import matern32_kernel as state_space_matern32_kernel
from opifex.uncertainty.types import PredictiveDistribution


# -----------------------------------------------------------------------------
# Generic fit/predict surface
# -----------------------------------------------------------------------------


def test_fit_markov_vi_gp_returns_finite_smoothed_state() -> None:
    """The VI iterations converge to finite smoothed moments."""
    from opifex.uncertainty.markov import fit_markov_vi_gp, MarkovVIGPState

    times = jnp.sort(
        jax.random.uniform(jax.random.PRNGKey(0), (20,), minval=0.0, maxval=2.0 * jnp.pi)
    )
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    fitted = fit_markov_vi_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        num_iterations=25,
    )
    assert isinstance(fitted, MarkovVIGPState)
    assert fitted.smoothed_means.shape == (times.shape[0],)
    assert fitted.smoothed_variances.shape == (times.shape[0],)
    assert jnp.all(jnp.isfinite(fitted.smoothed_means))
    assert jnp.all(fitted.smoothed_variances > 0.0)
    assert jnp.isfinite(fitted.evidence_lower_bound)


def test_predict_markov_vi_gp_returns_predictive_distribution() -> None:
    """Predict at held-out times returns a populated PredictiveDistribution."""
    from opifex.uncertainty.markov import fit_markov_vi_gp, predict_markov_vi_gp

    times = jnp.linspace(0.0, 4.0, 18)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_markov_vi_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        num_iterations=20,
    )
    times_test = jnp.linspace(0.5, 3.5, 6)
    predictive = predict_markov_vi_gp(state=state, times_test=times_test)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert predictive.mean.shape == (6,)
    assert jnp.all(jnp.isfinite(predictive.mean))
    assert jnp.all(predictive.variance > 0.0)


def test_markov_vi_full_pipeline_is_jit_compatible() -> None:
    """Fit + predict compile end-to-end under ``jax.jit``."""
    from opifex.uncertainty.markov import fit_markov_vi_gp, predict_markov_vi_gp

    times = jnp.linspace(0.0, 4.0, 15)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)

    @jax.jit
    def fit_predict(t: jax.Array, y: jax.Array) -> jax.Array:
        state = fit_markov_vi_gp(
            times=t,
            observations=y,
            state_space_kernel=kernel,
            log_likelihood_components_fn=_bernoulli_log_likelihood_components,
            num_iterations=15,
        )
        predictive = predict_markov_vi_gp(state=state, times_test=jnp.linspace(0.0, 4.0, 5))
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(times, targets)
    assert out.shape == (5,)
    assert jnp.all(jnp.isfinite(out))


# -----------------------------------------------------------------------------
# Cross-check: Gaussian likelihood VI matches Markov-Laplace (both = exact conjugate)
# -----------------------------------------------------------------------------


def test_markov_vi_gaussian_likelihood_matches_markov_laplace_gaussian() -> None:
    """For Gaussian likelihood, VI's expected curvature ≡ mode curvature.

    The Gaussian log-likelihood ``log N(y; f, σ²) = -½ (y - f)² / σ²``
    has constant Hessian ``-1/σ²`` and gradient ``(y - f) / σ²`` linear
    in ``f``. So ``E_{q(f)}[grad]`` and ``E_{q(f)}[Hessian]`` coincide
    exactly with the mode-evaluated versions — VI and Laplace produce
    identical state-space posteriors.
    """
    from opifex.uncertainty.markov import (
        fit_gaussian_markov_laplace_gp,
        fit_gaussian_markov_vi_gp,
    )

    times = jnp.linspace(0.0, 3.5, 20)
    observations = jnp.sin(2.0 * times) + 0.05 * jax.random.normal(jax.random.PRNGKey(11), (20,))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    laplace_state = fit_gaussian_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=0.05,
        num_iterations=5,
    )
    vi_state = fit_gaussian_markov_vi_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=0.05,
        num_iterations=5,
    )
    assert jnp.allclose(vi_state.smoothed_means, laplace_state.smoothed_means, atol=1e-5)
    assert jnp.allclose(vi_state.smoothed_variances, laplace_state.smoothed_variances, atol=1e-5)


# -----------------------------------------------------------------------------
# Per-likelihood Markov-VI wrappers (5 likelihoods to mirror slice 26)
# -----------------------------------------------------------------------------


def test_fit_bernoulli_markov_vi_gp_returns_class_probabilities_in_unit_interval() -> None:
    """Bernoulli VI predict returns class probabilities in [0, 1]."""
    from opifex.uncertainty.markov import (
        fit_bernoulli_markov_vi_gp,
        predict_bernoulli_markov_vi_gp,
    )

    times = jnp.linspace(0.0, 4.0, 16)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_bernoulli_markov_vi_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        num_iterations=20,
    )
    predictive = predict_bernoulli_markov_vi_gp(state=state, times_test=jnp.linspace(0.0, 4.0, 8))
    assert jnp.all(predictive.mean >= 0.0)
    assert jnp.all(predictive.mean <= 1.0)


def test_fit_poisson_markov_vi_gp_recovers_positive_intensity() -> None:
    """Poisson VI yields strictly positive predictive intensity."""
    from opifex.uncertainty.markov import (
        fit_poisson_markov_vi_gp,
        predict_poisson_markov_vi_gp,
    )

    key = jax.random.PRNGKey(13)
    times = jnp.sort(jax.random.uniform(key, (20,), minval=0.0, maxval=2.0 * jnp.pi))
    rate = jnp.exp(jnp.sin(2.0 * times) + 1.0)
    observations = jax.random.poisson(jax.random.PRNGKey(14), rate).astype(jnp.float32)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    state = fit_poisson_markov_vi_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        num_iterations=25,
    )
    predictive = predict_poisson_markov_vi_gp(state=state, times_test=jnp.linspace(0.5, 5.5, 8))
    assert predictive.variance is not None
    assert jnp.all(predictive.mean > 0.0)


def test_fit_studentst_markov_vi_gp_is_robust_to_outliers() -> None:
    """Student-t VI dampens the influence of heavy-tailed outliers."""
    from opifex.uncertainty.markov import fit_studentst_markov_vi_gp

    times = jnp.linspace(0.0, 6.0, 25)
    clean = jnp.sin(2.0 * times)
    observations = clean.at[5].set(3.0).at[15].set(-3.0)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_studentst_markov_vi_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        df=4.0,
        scale=0.3,
        num_iterations=30,
    )
    assert jnp.max(jnp.abs(state.smoothed_means)) < 2.5


def test_fit_beta_markov_vi_gp_recovers_unit_interval_predictions() -> None:
    """Beta VI predict yields means in [0, 1]."""
    from opifex.uncertainty.markov import (
        fit_beta_markov_vi_gp,
        predict_beta_markov_vi_gp,
    )

    times = jnp.linspace(0.0, 4.0, 18)
    mean = jax.nn.sigmoid(jnp.sin(2.0 * times))
    scale = 20.0
    alpha = mean * scale
    beta = scale * (1.0 - mean)
    observations = jax.random.beta(jax.random.PRNGKey(15), alpha, beta)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_beta_markov_vi_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        scale=scale,
        num_iterations=25,
    )
    predictive = predict_beta_markov_vi_gp(
        state=state, times_test=jnp.linspace(0.5, 3.5, 6), scale=scale
    )
    assert jnp.all(predictive.mean >= 0.0)
    assert jnp.all(predictive.mean <= 1.0)


# -----------------------------------------------------------------------------
# ELBO sanity
# -----------------------------------------------------------------------------


def test_markov_vi_elbo_is_finite_scalar() -> None:
    """The variational free energy / ELBO is a finite scalar."""
    from opifex.uncertainty.markov import fit_markov_vi_gp

    times = jnp.linspace(0.0, 4.0, 14)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_markov_vi_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        num_iterations=20,
    )
    assert state.evidence_lower_bound.shape == ()
    assert jnp.isfinite(state.evidence_lower_bound)
