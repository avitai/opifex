r"""Power Expectation Propagation on Markov GPs — Slice 28 (Task 11.2).

Power EP (Minka 2004) generalises classical Expectation Propagation
via a power parameter ``α ∈ (0, 1]``:

* ``α = 1.0`` recovers standard EP (Minka 2001).
* ``α < 1.0`` (typically ``α = 0.5``) yields *power EP* — better
  convergence on heavy-tailed / multi-modal likelihoods.
* ``α → 0`` limits to variational inference.

On a Markov-GP prior the algorithm maintains per-site Gaussian
approximations to each observation's likelihood contribution and
updates them iteratively via cavity-and-moment-matching:

1. Run Kalman filter + smoother with the current sites as
   pseudo-observations → posterior marginal moments at every time.
2. Form the **cavity** at each ``i`` by subtracting ``α × site_i``
   from the posterior in natural-parameter space.
3. Moment-match the **tilted** distribution ``cavity_i · p(y_i|f_i)^α``
   via Gauss-Hermite quadrature to get tilted-Gaussian moments.
4. Update site_i so that ``cavity_i × site_i^α`` matches the tilted
   moments. Apply a small ``learning_rate`` damping for stability.

For Gaussian likelihood the algorithm reduces exactly to the
conjugate Kalman path at ``α = 1`` (the tilted moments coincide with
the cavity-times-Gaussian moment-matching closed form).

References
----------
* Minka 2001 — *Expectation Propagation for Approximate Bayesian
  Inference*, UAI.
* Minka 2004 — *Power EP*, Microsoft Research TR.
* Wilkinson, Solin, Adam 2020+ — ``bayesnewton/inference.py``
  ``ExpectationPropagation`` (PRIMARY reference).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.gp import bernoulli_log_likelihood, poisson_log_likelihood
from opifex.uncertainty.statespace import matern32_kernel as state_space_matern32_kernel
from opifex.uncertainty.types import PredictiveDistribution


# -----------------------------------------------------------------------------
# Generic fit/predict surface
# -----------------------------------------------------------------------------


def test_fit_markov_pep_gp_returns_finite_smoothed_state() -> None:
    """PEP iterations converge to finite smoothed moments."""
    from opifex.uncertainty.markov import fit_markov_pep_gp, MarkovPEPGPState

    times = jnp.sort(
        jax.random.uniform(jax.random.PRNGKey(0), (20,), minval=0.0, maxval=2.0 * jnp.pi)
    )
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    fitted = fit_markov_pep_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        log_likelihood_fn=bernoulli_log_likelihood,
        power=0.5,
        num_iterations=30,
        learning_rate=0.5,
    )
    assert isinstance(fitted, MarkovPEPGPState)
    assert fitted.smoothed_means.shape == (times.shape[0],)
    assert fitted.smoothed_variances.shape == (times.shape[0],)
    assert jnp.all(jnp.isfinite(fitted.smoothed_means))
    assert jnp.all(fitted.smoothed_variances > 0.0)
    assert jnp.isfinite(fitted.log_marginal_likelihood)


def test_predict_markov_pep_gp_returns_predictive_distribution() -> None:
    """Predict at held-out times returns a populated PredictiveDistribution."""
    from opifex.uncertainty.markov import fit_markov_pep_gp, predict_markov_pep_gp

    times = jnp.linspace(0.0, 4.0, 18)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_markov_pep_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        log_likelihood_fn=bernoulli_log_likelihood,
        power=0.5,
        num_iterations=25,
    )
    times_test = jnp.linspace(0.5, 3.5, 6)
    predictive = predict_markov_pep_gp(state=state, times_test=times_test)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert predictive.mean.shape == (6,)
    assert jnp.all(jnp.isfinite(predictive.mean))
    assert jnp.all(predictive.variance > 0.0)


def test_markov_pep_full_pipeline_is_jit_compatible() -> None:
    """Fit + predict compile under ``jax.jit``."""
    from opifex.uncertainty.markov import fit_markov_pep_gp, predict_markov_pep_gp

    times = jnp.linspace(0.0, 4.0, 14)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)

    @jax.jit
    def fit_predict(t: jax.Array, y: jax.Array) -> jax.Array:
        state = fit_markov_pep_gp(
            times=t,
            observations=y,
            state_space_kernel=kernel,
            log_likelihood_fn=bernoulli_log_likelihood,
            power=0.5,
            num_iterations=15,
        )
        predictive = predict_markov_pep_gp(state=state, times_test=jnp.linspace(0.0, 4.0, 5))
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(times, targets)
    assert out.shape == (5,)
    assert jnp.all(jnp.isfinite(out))


# -----------------------------------------------------------------------------
# Gaussian cross-check: PEP at α=1 reduces to exact conjugate Kalman
# -----------------------------------------------------------------------------


def test_markov_pep_gaussian_likelihood_matches_markov_laplace_gaussian() -> None:
    r"""For Gaussian + ``α = 1.0`` PEP reduces to exact conjugate Kalman.

    The tilted distribution ``cavity · N(y; f, σ²)^1`` is Gaussian
    in ``f``, so moment-matching is exact and one iteration gives the
    posterior. The result must coincide with both Markov-Laplace and
    Markov-VI on Gaussian likelihood (the conjugate fixed point).
    """
    from opifex.uncertainty.markov import (
        fit_gaussian_markov_laplace_gp,
        fit_gaussian_markov_pep_gp,
    )

    times = jnp.linspace(0.0, 3.5, 18)
    observations = jnp.sin(2.0 * times) + 0.05 * jax.random.normal(jax.random.PRNGKey(13), (18,))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    laplace_state = fit_gaussian_markov_laplace_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=0.05,
        num_iterations=5,
    )
    pep_state = fit_gaussian_markov_pep_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        noise_std=0.05,
        power=1.0,
        num_iterations=10,
        learning_rate=1.0,
    )
    # Tolerance allows for one or two iterations of approximation
    # before the EP fixed point is exact (PEP at α=1 converges in
    # one full sweep for conjugate likelihoods).
    assert jnp.allclose(pep_state.smoothed_means, laplace_state.smoothed_means, atol=1e-3)


# -----------------------------------------------------------------------------
# Per-likelihood PEP wrappers (Bernoulli + Poisson + Gaussian)
# -----------------------------------------------------------------------------


def test_fit_bernoulli_markov_pep_gp_returns_class_probabilities_in_unit_interval() -> None:
    """Bernoulli PEP predict yields class probabilities in [0, 1]."""
    from opifex.uncertainty.markov import (
        fit_bernoulli_markov_pep_gp,
        predict_bernoulli_markov_pep_gp,
    )

    times = jnp.linspace(0.0, 4.0, 16)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_bernoulli_markov_pep_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        power=0.5,
        num_iterations=20,
    )
    predictive = predict_bernoulli_markov_pep_gp(state=state, times_test=jnp.linspace(0.0, 4.0, 8))
    assert jnp.all(predictive.mean >= 0.0)
    assert jnp.all(predictive.mean <= 1.0)


def test_fit_poisson_markov_pep_gp_recovers_positive_intensity() -> None:
    """Poisson PEP yields strictly positive predictive intensity."""
    from opifex.uncertainty.markov import (
        fit_poisson_markov_pep_gp,
        predict_poisson_markov_pep_gp,
    )

    key = jax.random.PRNGKey(15)
    times = jnp.sort(jax.random.uniform(key, (20,), minval=0.0, maxval=2.0 * jnp.pi))
    rate = jnp.exp(jnp.sin(2.0 * times) + 1.0)
    observations = jax.random.poisson(jax.random.PRNGKey(16), rate).astype(jnp.float32)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    state = fit_poisson_markov_pep_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        power=0.5,
        num_iterations=25,
        learning_rate=0.3,
    )
    predictive = predict_poisson_markov_pep_gp(state=state, times_test=jnp.linspace(0.5, 5.5, 8))
    assert predictive.variance is not None
    assert jnp.all(predictive.mean > 0.0)


def test_markov_pep_state_advertises_pep_estimator_metadata() -> None:
    """``predict_markov_pep_gp`` metadata advertises the PEP inference path."""
    from opifex.uncertainty.markov import fit_markov_pep_gp, predict_markov_pep_gp

    times = jnp.linspace(0.0, 4.0, 12)
    targets = jnp.sign(jnp.sin(2.0 * times))
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_markov_pep_gp(
        times=times,
        observations=targets,
        state_space_kernel=kernel,
        log_likelihood_fn=bernoulli_log_likelihood,
        power=0.5,
        num_iterations=10,
    )
    predictive = predict_markov_pep_gp(state=state, times_test=jnp.zeros((3,)))
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "markov_pep_gp"
    assert metadata.get("paper") is not None


_ = poisson_log_likelihood
