"""Tests for the standard Kalman primitives.

Sibling reference (line-by-line port): ``bayesnewton/bayesnewton/ops.py``
``_sequential_kf`` (line 154) and ``_sequential_rts`` (line 288).

References
----------
* Kalman 1960 — *A New Approach to Linear Filtering and Prediction Problems*.
* Rauch, Tung, Striebel 1965 — *Maximum Likelihood Estimates of Linear
  Dynamic Systems*, AIAA J.
* Särkkä — *Bayesian Filtering and Smoothing* (2013).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import (
    kalman_filter,
    kalman_log_likelihood,
    kalman_predict,
    kalman_smoother,
    kalman_update,
)


def test_kalman_predict_propagates_mean_and_covariance() -> None:
    """``kalman_predict`` implements ``m' = A m, P' = A P A^T + Q``."""
    mean = jnp.asarray([1.0, 2.0])
    cov = jnp.eye(2)
    transition = jnp.asarray([[1.0, 0.1], [0.0, 1.0]])
    process_noise = 0.01 * jnp.eye(2)

    predicted_mean, predicted_cov = kalman_predict(
        mean=mean, cov=cov, transition=transition, process_noise=process_noise
    )
    assert jnp.allclose(predicted_mean, transition @ mean, atol=1e-6)
    assert jnp.allclose(predicted_cov, transition @ cov @ transition.T + process_noise, atol=1e-6)


def test_kalman_update_returns_correct_posterior_for_1d_linear_gaussian() -> None:
    """Closed-form posterior matches the analytic Gaussian conjugate update.

    For prior ``N(m, p)`` and observation ``y ~ N(H m, r)``, the posterior
    has mean ``m + p H (H p H + r)^{-1} (y - H m)`` and covariance
    ``p - p H (H p H + r)^{-1} H p``.
    """
    prior_mean = jnp.asarray([0.0])
    prior_cov = jnp.asarray([[1.0]])
    observation = jnp.asarray([2.0])
    observation_matrix = jnp.asarray([[1.0]])
    observation_cov = jnp.asarray([[0.25]])

    posterior_mean, posterior_cov = kalman_update(
        mean=prior_mean,
        cov=prior_cov,
        observation=observation,
        observation_matrix=observation_matrix,
        observation_cov=observation_cov,
    )
    # Analytic posterior: μ = (σ_obs² m + σ_prior² y) / (σ_prior² + σ_obs²)
    # = (0.25 * 0 + 1.0 * 2.0) / (1.0 + 0.25) = 1.6
    assert jnp.allclose(posterior_mean, jnp.asarray([1.6]), atol=1e-5)
    # Analytic variance: σ² = (σ_prior² σ_obs²) / (σ_prior² + σ_obs²) = 0.2
    assert jnp.allclose(posterior_cov, jnp.asarray([[0.2]]), atol=1e-5)


def test_kalman_filter_recovers_true_state_on_noiseless_linear_dynamics() -> None:
    """With perfect observations and consistent dynamics, filter mean = truth.

    For noiseless dynamics the true state evolves deterministically under
    the transition matrix; with noiseless observations the filter
    posterior mean must equal the true state at every step.
    """
    state_dim = 2
    num_steps = 5
    transition = jnp.asarray([[1.0, 0.1], [0.0, 1.0]])
    process_noise = 1e-8 * jnp.eye(state_dim)
    observation_matrix = jnp.eye(state_dim)
    observation_cov = 1e-8 * jnp.eye(state_dim)

    initial_state = jax.random.normal(jax.random.PRNGKey(0), (state_dim,))

    def simulate_step(state: jax.Array, _: jax.Array) -> tuple[jax.Array, jax.Array]:
        next_state = transition @ state
        return next_state, next_state

    _, true_states = jax.lax.scan(simulate_step, initial_state, jnp.arange(num_steps))
    transitions = jnp.broadcast_to(transition, (num_steps, state_dim, state_dim))
    process_noises = jnp.broadcast_to(process_noise, (num_steps, state_dim, state_dim))
    observation_covs = jnp.broadcast_to(observation_cov, (num_steps, state_dim, state_dim))

    means, covs = kalman_filter(
        transitions=transitions,
        process_noises=process_noises,
        observations=true_states,
        observation_matrix=observation_matrix,
        observation_covs=observation_covs,
        initial_mean=initial_state,
        initial_cov=1e-8 * jnp.eye(state_dim),
    )
    assert means.shape == (num_steps, state_dim)
    assert covs.shape == (num_steps, state_dim, state_dim)
    assert jnp.allclose(means, true_states, atol=1e-3)


def test_kalman_smoother_lowers_variance_below_filter() -> None:
    """RTS smoother strictly reduces (or preserves) the filter covariance.

    Cite: Rauch, Tung, Striebel 1965. The smoothed posterior conditions on
    the full observation sequence, so ``P_smooth <= P_filter`` in the
    positive-semidefinite ordering.
    """
    state_dim = 2
    num_steps = 6
    transition = jnp.asarray([[1.0, 0.1], [0.0, 0.95]])
    process_noise = 0.05 * jnp.eye(state_dim)
    observation_matrix = jnp.eye(state_dim)
    observation_cov = 0.1 * jnp.eye(state_dim)

    rng = jax.random.PRNGKey(1)
    observations = jax.random.normal(rng, (num_steps, state_dim))
    transitions = jnp.broadcast_to(transition, (num_steps, state_dim, state_dim))
    process_noises = jnp.broadcast_to(process_noise, (num_steps, state_dim, state_dim))
    observation_covs = jnp.broadcast_to(observation_cov, (num_steps, state_dim, state_dim))

    filter_means, filter_covs = kalman_filter(
        transitions=transitions,
        process_noises=process_noises,
        observations=observations,
        observation_matrix=observation_matrix,
        observation_covs=observation_covs,
        initial_mean=jnp.zeros(state_dim),
        initial_cov=jnp.eye(state_dim),
    )
    _smoother_means, smoother_covs = kalman_smoother(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=transitions,
        process_noises=process_noises,
    )
    # Final-step filter and smoother covariance agree exactly (no
    # additional future info), so compare strictly-earlier steps.
    for step in range(num_steps - 1):
        filter_trace = jnp.trace(filter_covs[step])
        smoother_trace = jnp.trace(smoother_covs[step])
        assert smoother_trace <= filter_trace + 1e-6


def test_kalman_log_likelihood_matches_analytic_value_on_1d_gaussian() -> None:
    """Marginal log-likelihood matches ``mvn_logpdf`` on a single-step model."""
    initial_mean = jnp.zeros(1)
    initial_cov = jnp.asarray([[1.0]])
    transition = jnp.asarray([[1.0]])
    process_noise = jnp.asarray([[0.0]])
    observation_matrix = jnp.asarray([[1.0]])
    observation_cov = jnp.asarray([[0.5]])
    observation = jnp.asarray([1.0])

    transitions = transition[None, :, :]
    process_noises = process_noise[None, :, :]
    observations = observation[None, :]
    observation_covs = observation_cov[None, :, :]

    ll = kalman_log_likelihood(
        transitions=transitions,
        process_noises=process_noises,
        observations=observations,
        observation_matrix=observation_matrix,
        observation_covs=observation_covs,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )
    # Analytic marginal: y ~ N(H A m, H A P A^T H^T + H Q H^T + R) = N(0, 1 + 0.5)
    # log p(1.0) = -0.5 * log(2*pi*1.5) - 0.5 * 1.0^2 / 1.5
    expected = -0.5 * jnp.log(2.0 * jnp.pi * 1.5) - 0.5 / 1.5
    assert jnp.allclose(ll, expected, atol=1e-5)


def test_kalman_filter_is_jit_compatible() -> None:
    """The combined Kalman chain compiles under ``jax.jit``."""
    state_dim = 2
    num_steps = 4
    transitions = jnp.broadcast_to(jnp.eye(state_dim), (num_steps, state_dim, state_dim))
    process_noises = jnp.broadcast_to(0.1 * jnp.eye(state_dim), (num_steps, state_dim, state_dim))
    observation_covs = jnp.broadcast_to(0.1 * jnp.eye(state_dim), (num_steps, state_dim, state_dim))

    def call(observations: jax.Array) -> tuple[jax.Array, jax.Array]:
        return kalman_filter(
            transitions=transitions,
            process_noises=process_noises,
            observations=observations,
            observation_matrix=jnp.eye(state_dim),
            observation_covs=observation_covs,
            initial_mean=jnp.zeros(state_dim),
            initial_cov=jnp.eye(state_dim),
        )

    observations = jax.random.normal(jax.random.PRNGKey(0), (num_steps, state_dim))
    jitted = jax.jit(call)
    means, covs = jitted(observations)
    assert jnp.all(jnp.isfinite(means))
    assert jnp.all(jnp.isfinite(covs))
