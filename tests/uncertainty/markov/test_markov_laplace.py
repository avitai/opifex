r"""Markov-Laplace inference bridge — Slice 25 (Task 11.2).

Generalises D5's exact-GP Laplace machinery
(:func:`opifex.uncertainty.gp.fit_laplace_gp`) to the state-space /
Markov-GP form. Reuses:

* D5's :class:`LikelihoodComponentsFn` for the per-observation
  ``(log_lik, ∇log_lik, W, √W)`` quadruple.
* opifex's tested :func:`kalman_filter` / :func:`kalman_smoother`
  primitives for the Newton-step linear solve at every iteration.
* The existing :class:`StateSpaceKernel` family
  (matern12/32/52/72, cosine, periodic, quasi-periodic) for the
  Markov-GP prior.

Algorithm (bayesnewton ``inference.py:Laplace`` reference; Sarkka 2013
§9 *Bayesian Filtering and Smoothing* for the Iterated-EKS analogue):

For iteration ``t``:
    1. Compute per-step components ``(log_lik_i, grad_i, W_i)`` from
       the current mean trajectory.
    2. Form linearised pseudo-observations
       ``y_i^{pseudo} = f_i^{(t)} + grad_i / W_i`` with
       pseudo-noise covariance ``R_i = 1 / W_i``.
    3. Run :func:`kalman_filter` + :func:`kalman_smoother` on the
       linearised model.
    4. Update ``f^{(t+1)}`` to the smoothed mean.

The slice-25 correctness anchor is **cross-validation against D5's
exact Laplace path** at the same kernel and data: with matched
Matern-3/2 + Bernoulli, the Markov-Laplace posterior mean and
variance match D5's :func:`fit_bernoulli_laplace_gp` /
:func:`predict_bernoulli_laplace_gp` to within numerical tolerance.

References
----------
* Sarkka 2013 — *Bayesian Filtering and Smoothing*, CUP §9 (Iterated
  Extended Kalman Smoother).
* Wilkinson, Solin, Adam 2020+ — ``bayesnewton`` (PRIMARY for the
  inference-on-Markov-GPs design).
* Rasmussen & Williams 2006 §3.4 (the Laplace algorithm being
  bridged into state-space form).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.gp import (
    bernoulli_log_likelihood,
    fit_bernoulli_laplace_gp,
    matern32_kernel as gp_matern32_kernel,
    predict_bernoulli_laplace_gp,
)
from opifex.uncertainty.gp.laplace_classification import (
    _bernoulli_log_likelihood_components,
)
from opifex.uncertainty.statespace import matern32_kernel as state_space_matern32_kernel
from opifex.uncertainty.types import PredictiveDistribution


def _toy_binary_time_series(seed: int = 0, num_train: int = 20) -> tuple[jax.Array, jax.Array]:
    """1-D Bernoulli classification on a sorted time grid: ``y = sign(sin(2 t))``."""
    key = jax.random.PRNGKey(seed)
    times = jnp.sort(jax.random.uniform(key, (num_train,), minval=0.0, maxval=2.0 * jnp.pi))
    targets = jnp.sign(jnp.sin(2.0 * times))
    return times, targets


# -----------------------------------------------------------------------------
# Sanity: state shape, finite log marginal, JIT compatibility
# -----------------------------------------------------------------------------


def test_fit_markov_laplace_gp_returns_finite_smoothed_state() -> None:
    """Newton iterations converge to a finite smoothed posterior."""
    from opifex.uncertainty.markov.markov_laplace import (
        fit_markov_laplace_gp,
        MarkovLaplaceGPState,
    )

    times, targets = _toy_binary_time_series(seed=0)
    state_space_kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    fitted = fit_markov_laplace_gp(
        times=times,
        observations=targets,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        num_iterations=30,
    )
    assert isinstance(fitted, MarkovLaplaceGPState)
    assert fitted.smoothed_means.shape == (times.shape[0],)
    assert fitted.smoothed_variances.shape == (times.shape[0],)
    assert jnp.all(jnp.isfinite(fitted.smoothed_means))
    assert jnp.all(fitted.smoothed_variances > 0.0)
    assert jnp.isfinite(fitted.log_marginal_likelihood)


def test_markov_laplace_predict_returns_predictive_distribution() -> None:
    """Predict at held-out times returns a populated PredictiveDistribution."""
    from opifex.uncertainty.markov.markov_laplace import (
        fit_markov_laplace_gp,
        predict_markov_laplace_gp,
    )

    times, targets = _toy_binary_time_series(seed=1)
    state_space_kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    fitted = fit_markov_laplace_gp(
        times=times,
        observations=targets,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        num_iterations=25,
    )
    times_test = jnp.linspace(0.5, 5.5, 8)
    predictive = predict_markov_laplace_gp(state=fitted, times_test=times_test)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert predictive.mean.shape == (8,)
    assert jnp.all(jnp.isfinite(predictive.mean))
    assert jnp.all(predictive.variance > 0.0)


def test_fit_markov_laplace_gp_is_jit_compatible() -> None:
    """The full fit pipeline compiles under ``jax.jit``."""
    from opifex.uncertainty.markov.markov_laplace import fit_markov_laplace_gp

    times, targets = _toy_binary_time_series(seed=2, num_train=15)
    state_space_kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)

    @jax.jit
    def fit_call(times_arr: jax.Array, targets_arr: jax.Array) -> jax.Array:
        fitted = fit_markov_laplace_gp(
            times=times_arr,
            observations=targets_arr,
            state_space_kernel=state_space_kernel,
            log_likelihood_components_fn=_bernoulli_log_likelihood_components,
            num_iterations=20,
        )
        return fitted.smoothed_means + fitted.smoothed_variances

    output = fit_call(times, targets)
    assert output.shape == times.shape
    assert jnp.all(jnp.isfinite(output))


# -----------------------------------------------------------------------------
# Cross-validation against D5 exact-GP Laplace path (the strong correctness anchor)
# -----------------------------------------------------------------------------


def test_markov_laplace_bernoulli_matches_d5_exact_path_at_training_points() -> None:
    """Bernoulli + Matern-3/2 state-space matches D5 exact at training points."""
    from opifex.uncertainty.markov.markov_laplace import fit_markov_laplace_gp

    times, targets = _toy_binary_time_series(seed=3, num_train=15)
    lengthscale = 0.6
    output_scale = 1.0

    # D5 exact-GP Laplace path (O(n^2) Cholesky + Newton).
    d5_state = fit_bernoulli_laplace_gp(
        x_train=times.reshape(-1, 1),
        y_train=targets,
        lengthscale=lengthscale,
        output_scale=output_scale,
        num_newton_iterations=40,
        kernel_fn=gp_matern32_kernel,
    )
    d5_predictive = predict_bernoulli_laplace_gp(state=d5_state, x_test=times.reshape(-1, 1))

    # Markov-Laplace state-space path (O(n) Kalman per iteration).
    state_space_kernel = state_space_matern32_kernel(
        variance=output_scale**2, lengthscale=lengthscale
    )
    markov_state = fit_markov_laplace_gp(
        times=times,
        observations=targets,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        num_iterations=40,
    )

    # At training points the smoothed latent mean must match the D5
    # latent posterior mode (both paths are Newton iterates on the
    # same Laplace MAP problem).
    d5_latent_mean = d5_state.f_mode
    assert jnp.allclose(markov_state.smoothed_means, d5_latent_mean, atol=5e-2)

    # The class-probability prediction at training points should
    # therefore match within the same tolerance.
    sigmoid_arg = markov_state.smoothed_means / jnp.sqrt(
        1.0 + jnp.pi * markov_state.smoothed_variances / 8.0
    )
    markov_class_prob = jax.nn.sigmoid(sigmoid_arg)
    assert jnp.allclose(markov_class_prob, d5_predictive.mean, atol=5e-2)


def test_markov_laplace_predict_class_probabilities_lie_in_unit_interval() -> None:
    """Class-probability predictions through the MacKay-probit mapping stay in [0, 1]."""
    from opifex.uncertainty.markov.markov_laplace import (
        fit_markov_laplace_gp,
        predict_markov_laplace_gp,
    )

    times, targets = _toy_binary_time_series(seed=5)
    state_space_kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    fitted = fit_markov_laplace_gp(
        times=times,
        observations=targets,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        num_iterations=30,
    )
    times_test = jnp.linspace(0.0, 6.0, 12)
    predictive = predict_markov_laplace_gp(state=fitted, times_test=times_test)
    # The MacKay probit collapse `σ(μ / sqrt(1 + π V / 8))` must remain
    # bounded in (0, 1) at every test point.
    assert predictive.variance is not None
    sigmoid_arg = predictive.mean / jnp.sqrt(1.0 + jnp.pi * predictive.variance / 8.0)
    class_prob = jax.nn.sigmoid(sigmoid_arg)
    assert jnp.all(class_prob >= 0.0)
    assert jnp.all(class_prob <= 1.0)


def test_markov_laplace_metadata_advertises_state_space_provenance() -> None:
    """The predictive ``metadata`` records ``estimator=markov_laplace_gp``."""
    from opifex.uncertainty.markov.markov_laplace import (
        fit_markov_laplace_gp,
        predict_markov_laplace_gp,
    )

    times, targets = _toy_binary_time_series(seed=6)
    state_space_kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.6)
    fitted = fit_markov_laplace_gp(
        times=times,
        observations=targets,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        num_iterations=20,
    )
    predictive = predict_markov_laplace_gp(state=fitted, times_test=jnp.zeros((3,)))
    metadata = dict(predictive.metadata)
    assert metadata.get("estimator") == "markov_laplace_gp"
    assert metadata.get("paper") is not None


_ = bernoulli_log_likelihood
