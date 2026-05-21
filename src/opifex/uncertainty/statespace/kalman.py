"""Standard Kalman filter, RTS smoother, and marginal log-likelihood.

The primitives are pure JAX and operate on dense state-transition /
observation matrices. The sequential filter/smoother are implemented via
``jax.lax.scan`` so the iteration length stays static under ``jax.jit``.

Public API
----------
* ``kalman_predict`` — one-step ``(m, P) -> (A m, A P A^T + Q)``.
* ``kalman_update`` — one-step Gaussian conditioning given a linear
  observation operator and Gaussian noise.
* ``kalman_filter`` — full forward pass; returns posterior mean and
  covariance at every step.
* ``kalman_smoother`` — RTS backward pass; conditions each step on the
  full observation sequence.
* ``kalman_log_likelihood`` — marginal log-likelihood of the observation
  sequence under the linear-Gaussian state-space model.

Sibling reference (line-by-line port): ``bayesnewton/bayesnewton/ops.py``
``_sequential_kf`` (line 154) and ``_sequential_rts`` (line 288).

References
----------
* Kalman 1960; Rauch, Tung, Striebel 1965; Särkkä 2013.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def kalman_predict(
    *,
    mean: jax.Array,
    cov: jax.Array,
    transition: jax.Array,
    process_noise: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Linear-Gaussian Kalman prediction step.

    Returns ``(transition @ mean, transition @ cov @ transition.T + process_noise)``.

    Args:
        mean: prior mean of shape ``(state_dim,)``.
        cov: prior covariance of shape ``(state_dim, state_dim)``.
        transition: state-transition matrix ``A`` of shape
            ``(state_dim, state_dim)``.
        process_noise: process-noise covariance ``Q`` of shape
            ``(state_dim, state_dim)``.

    Returns:
        Predicted ``(mean, cov)``.
    """
    predicted_mean = transition @ mean
    predicted_cov = transition @ cov @ transition.T + process_noise
    return predicted_mean, predicted_cov


def kalman_update(
    *,
    mean: jax.Array,
    cov: jax.Array,
    observation: jax.Array,
    observation_matrix: jax.Array,
    observation_cov: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Linear-Gaussian Kalman update (innovation + Joseph-form covariance).

    Args:
        mean: predicted mean ``m_-`` of shape ``(state_dim,)``.
        cov: predicted covariance ``P_-`` of shape ``(state_dim, state_dim)``.
        observation: observed measurement ``y`` of shape ``(obs_dim,)``.
        observation_matrix: linear observation operator ``H`` of shape
            ``(obs_dim, state_dim)``.
        observation_cov: observation-noise covariance ``R`` of shape
            ``(obs_dim, obs_dim)``.

    Returns:
        Posterior ``(mean, cov)`` after conditioning on ``observation``.
    """
    innovation = observation - observation_matrix @ mean
    cov_obs = observation_matrix @ cov
    innovation_cov = cov_obs @ observation_matrix.T + observation_cov
    gain = jnp.linalg.solve(innovation_cov, cov_obs).T
    updated_mean = mean + gain @ innovation
    updated_cov = cov - gain @ cov_obs
    return updated_mean, updated_cov


def kalman_filter(
    *,
    transitions: jax.Array,
    process_noises: jax.Array,
    observations: jax.Array,
    observation_matrix: jax.Array,
    observation_covs: jax.Array,
    initial_mean: jax.Array,
    initial_cov: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run a sequential Kalman filter over a fixed-length sequence.

    Sibling reference: ``bayesnewton/bayesnewton/ops.py:154 _sequential_kf``.

    Args:
        transitions: shape ``(num_steps, state_dim, state_dim)``.
        process_noises: shape ``(num_steps, state_dim, state_dim)``.
        observations: shape ``(num_steps, obs_dim)``.
        observation_matrix: shape ``(obs_dim, state_dim)`` (time-invariant).
        observation_covs: shape ``(num_steps, obs_dim, obs_dim)``.
        initial_mean: shape ``(state_dim,)``.
        initial_cov: shape ``(state_dim, state_dim)``.

    Returns:
        ``(means, covs)`` of shapes ``(num_steps, state_dim)`` and
        ``(num_steps, state_dim, state_dim)`` containing the posterior
        mean and covariance at each step *after* the update.
    """

    def body(
        carry: tuple[jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        mean, cov = carry
        observation, transition, process_noise, observation_cov = inputs
        predicted_mean, predicted_cov = kalman_predict(
            mean=mean, cov=cov, transition=transition, process_noise=process_noise
        )
        updated_mean, updated_cov = kalman_update(
            mean=predicted_mean,
            cov=predicted_cov,
            observation=observation,
            observation_matrix=observation_matrix,
            observation_cov=observation_cov,
        )
        return (updated_mean, updated_cov), (updated_mean, updated_cov)

    _, (means, covs) = jax.lax.scan(
        body,
        (initial_mean, initial_cov),
        (observations, transitions, process_noises, observation_covs),
    )
    return means, covs


def kalman_smoother(
    *,
    filter_means: jax.Array,
    filter_covs: jax.Array,
    transitions: jax.Array,
    process_noises: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Rauch-Tung-Striebel backward smoother.

    Sibling reference: ``bayesnewton/bayesnewton/ops.py:288 _sequential_rts``.

    Args:
        filter_means: forward-pass posterior means ``(num_steps, state_dim)``.
        filter_covs: forward-pass posterior covariances
            ``(num_steps, state_dim, state_dim)``.
        transitions: same shape as in ``kalman_filter``.
        process_noises: same shape as in ``kalman_filter``.

    Returns:
        ``(smoothed_means, smoothed_covs)`` — posteriors conditioned on
        the full observation sequence.
    """
    num_steps = filter_means.shape[0]
    last_mean = filter_means[-1]
    last_cov = filter_covs[-1]

    def body(
        carry: tuple[jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        next_smoothed_mean, next_smoothed_cov = carry
        current_filter_mean, current_filter_cov, next_transition, next_process_noise = (
            inputs
        )
        predicted_mean = next_transition @ current_filter_mean
        predicted_cov = (
            next_transition @ current_filter_cov @ next_transition.T + next_process_noise
        )
        gain = jnp.linalg.solve(
            predicted_cov, next_transition @ current_filter_cov
        ).T
        smoothed_mean = current_filter_mean + gain @ (
            next_smoothed_mean - predicted_mean
        )
        smoothed_cov = current_filter_cov + gain @ (
            next_smoothed_cov - predicted_cov
        ) @ gain.T
        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    _, (back_means, back_covs) = jax.lax.scan(
        body,
        (last_mean, last_cov),
        (
            filter_means[: num_steps - 1],
            filter_covs[: num_steps - 1],
            transitions[1:num_steps],
            process_noises[1:num_steps],
        ),
        reverse=True,
    )
    smoothed_means = jnp.concatenate([back_means, last_mean[None, :]], axis=0)
    smoothed_covs = jnp.concatenate([back_covs, last_cov[None, :, :]], axis=0)
    return smoothed_means, smoothed_covs


def _mvn_logpdf(
    observation: jax.Array, mean: jax.Array, cov: jax.Array
) -> jax.Array:
    """Multivariate Gaussian log-density at ``observation``."""
    obs_dim = observation.shape[0]
    cholesky = jnp.linalg.cholesky(cov)
    delta = observation - mean
    solve = jax.scipy.linalg.solve_triangular(cholesky, delta, lower=True)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
    return -0.5 * (jnp.sum(solve**2) + log_det + obs_dim * jnp.log(2.0 * jnp.pi))


def kalman_log_likelihood(
    *,
    transitions: jax.Array,
    process_noises: jax.Array,
    observations: jax.Array,
    observation_matrix: jax.Array,
    observation_covs: jax.Array,
    initial_mean: jax.Array,
    initial_cov: jax.Array,
) -> jax.Array:
    """Marginal log-likelihood of the observation sequence.

    Returns ``sum_k log p(y_k | y_{0:k-1})`` evaluated at each step via
    the innovation Gaussian whose covariance is ``H P_- H^T + R``. Used
    for hyperparameter learning via ``jax.grad``.

    Sibling reference: ``bayesnewton/bayesnewton/ops.py:170-180`` (the
    ``mvn_logpdf`` accumulator inside ``_sequential_kf``).
    """

    def body(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        mean, cov, ll = carry
        observation, transition, process_noise, observation_cov = inputs
        predicted_mean, predicted_cov = kalman_predict(
            mean=mean, cov=cov, transition=transition, process_noise=process_noise
        )
        innovation_cov = (
            observation_matrix @ predicted_cov @ observation_matrix.T + observation_cov
        )
        step_ll = _mvn_logpdf(
            observation, observation_matrix @ predicted_mean, innovation_cov
        )
        updated_mean, updated_cov = kalman_update(
            mean=predicted_mean,
            cov=predicted_cov,
            observation=observation,
            observation_matrix=observation_matrix,
            observation_cov=observation_cov,
        )
        return (updated_mean, updated_cov, ll + step_ll), None

    (_, _, total_ll), _ = jax.lax.scan(
        body,
        (initial_mean, initial_cov, jnp.asarray(0.0)),
        (observations, transitions, process_noises, observation_covs),
    )
    return total_ll


__all__ = [
    "kalman_filter",
    "kalman_log_likelihood",
    "kalman_predict",
    "kalman_smoother",
    "kalman_update",
]
