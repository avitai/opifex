"""Tests for the compute-aware Kalman filter primitives.

The compute-aware Kalman filter (CAKF) propagates a low-rank correction
factor ``M`` to the prior marginal covariance ``Σ_k`` so that the
posterior covariance is the LowRankDowndatedMatrix ``Σ_k - M @ M.T``. The
update step iteratively expands ``M`` using a chosen search-direction
policy (CG / coordinate / random) and terminates at a residual tolerance
or maximum iteration count.

Canonical reference (line-by-line port):
* ``../ComputationAwareKalman.jl/src/low_rank.jl`` —
  ``LowRankDowndatedMatrix`` constructor and matvec.
* ``../ComputationAwareKalman.jl/src/filter/predict.jl`` — ``predict``.
* ``../ComputationAwareKalman.jl/src/filter/update.jl`` — ``update``.
* ``../ComputationAwareKalman.jl/src/filter/policy.jl`` — ``CGPolicy``.

References
----------
* Pförtner, Wenger, Cockayne, Hennig 2024 — *Computation-Aware Kalman
  Filtering and Smoothing*, arXiv:2405.08971.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import (
    cakf_predict,
    cakf_update,
    LowRankDowndatedMatrix,
)


def test_low_rank_downdated_matrix_matvec_matches_explicit_form() -> None:
    """``(A - U V^T) @ x`` matches the implicit matvec."""
    rng = jax.random.PRNGKey(0)
    key_a, key_u, key_v, key_x = jax.random.split(rng, 4)
    n, r = 5, 2
    matrix_a = jax.random.normal(key_a, (n, n))
    matrix_u = jax.random.normal(key_u, (n, r))
    matrix_v = jax.random.normal(key_v, (n, r))
    operator = LowRankDowndatedMatrix(dense=matrix_a, left=matrix_u, right=matrix_v)
    vector = jax.random.normal(key_x, (n,))
    explicit = matrix_a @ vector - matrix_u @ (matrix_v.T @ vector)
    assert jnp.allclose(operator @ vector, explicit, atol=1e-5)


def test_low_rank_downdated_matrix_empty_correction_is_dense_matvec() -> None:
    """An empty ``U`` and ``V`` reduces the operator to ``A``."""
    matrix_a = jnp.asarray([[1.0, 0.5], [0.5, 2.0]])
    operator = LowRankDowndatedMatrix(
        dense=matrix_a,
        left=jnp.zeros((2, 0)),
        right=jnp.zeros((2, 0)),
    )
    vector = jnp.asarray([1.0, -1.0])
    assert jnp.allclose(operator @ vector, matrix_a @ vector, atol=1e-7)


def test_cakf_predict_propagates_mean_and_factor() -> None:
    """``cakf_predict`` propagates ``(m, M)`` through the transition matrix."""
    state_dim = 3
    rank = 2
    rng = jax.random.PRNGKey(0)
    key_m, key_factor = jax.random.split(rng)
    prior_mean = jax.random.normal(key_m, (state_dim,))
    prior_factor = jax.random.normal(key_factor, (state_dim, rank))
    transition = jnp.eye(state_dim) + 0.1 * jnp.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    predicted_mean, predicted_factor = cakf_predict(
        mean=prior_mean,
        factor=prior_factor,
        transition=transition,
    )
    assert jnp.allclose(predicted_mean, transition @ prior_mean, atol=1e-6)
    assert jnp.allclose(predicted_factor, transition @ prior_factor, atol=1e-6)


def test_cakf_update_full_rank_recovers_exact_kalman_update() -> None:
    """At maximum rank, CAKF reproduces the dense Kalman posterior mean.

    For a non-singular innovation covariance, running the CAKF iteration
    until ``max_iter == obs_dim`` recovers the exact Kalman posterior
    because the search directions span the entire observation space.
    """
    obs_dim = 3
    state_dim = 3
    prior_mean = jnp.zeros(state_dim)
    prior_cov = jnp.eye(state_dim) + 0.1 * jnp.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    observation_matrix = jnp.eye(state_dim)
    observation_cov = 0.5 * jnp.eye(obs_dim)
    observation = jnp.asarray([1.0, -0.5, 0.3])

    posterior_mean, _ = cakf_update(
        mean=prior_mean,
        prior_cov=prior_cov,
        factor=jnp.zeros((state_dim, 0)),
        observation=observation,
        observation_matrix=observation_matrix,
        observation_cov=observation_cov,
        max_iter=obs_dim,
    )
    # Reference dense Kalman update.
    cov_obs = observation_matrix @ prior_cov
    innovation_cov = cov_obs @ observation_matrix.T + observation_cov
    gain = jnp.linalg.solve(innovation_cov, cov_obs).T
    expected_mean = prior_mean + gain @ (observation - observation_matrix @ prior_mean)
    assert jnp.allclose(posterior_mean, expected_mean, atol=1e-4)


def test_cakf_update_zero_iterations_returns_prior_mean() -> None:
    """``max_iter = 0`` performs no update; the posterior equals the prior."""
    state_dim = 2
    prior_mean = jnp.asarray([1.0, -2.0])
    posterior_mean, posterior_factor = cakf_update(
        mean=prior_mean,
        prior_cov=jnp.eye(state_dim),
        factor=jnp.zeros((state_dim, 0)),
        observation=jnp.asarray([0.0, 0.0]),
        observation_matrix=jnp.eye(state_dim),
        observation_cov=0.1 * jnp.eye(state_dim),
        max_iter=0,
    )
    assert jnp.allclose(posterior_mean, prior_mean, atol=1e-7)
    assert posterior_factor.shape == (state_dim, 0)


def test_cakf_update_residual_decreases_monotonically() -> None:
    """Each CAKF iteration reduces the innovation residual norm."""
    state_dim = 4
    obs_dim = 4
    prior_mean = jnp.zeros(state_dim)
    prior_cov = jnp.eye(state_dim)
    observation_matrix = jnp.eye(state_dim)
    observation_cov = 0.5 * jnp.eye(obs_dim)
    observation = jnp.asarray([1.0, 0.5, -0.5, 0.3])

    norms = []
    for max_iter in (0, 1, 2, 3, 4):
        posterior_mean, _ = cakf_update(
            mean=prior_mean,
            prior_cov=prior_cov,
            factor=jnp.zeros((state_dim, 0)),
            observation=observation,
            observation_matrix=observation_matrix,
            observation_cov=observation_cov,
            max_iter=max_iter,
        )
        residual = observation - observation_matrix @ posterior_mean
        norms.append(float(jnp.linalg.norm(residual)))
    import itertools

    for previous, current in itertools.pairwise(norms):
        assert current <= previous + 1e-5


def test_cakf_update_factor_columns_grow_with_max_iter() -> None:
    """Each iteration appends one column to the low-rank factor."""
    state_dim = 3
    obs_dim = 3
    prior_factor = jnp.zeros((state_dim, 0))
    _, factor_one_iter = cakf_update(
        mean=jnp.zeros(state_dim),
        prior_cov=jnp.eye(state_dim),
        factor=prior_factor,
        observation=jnp.ones(obs_dim),
        observation_matrix=jnp.eye(state_dim),
        observation_cov=0.1 * jnp.eye(obs_dim),
        max_iter=1,
    )
    _, factor_two_iters = cakf_update(
        mean=jnp.zeros(state_dim),
        prior_cov=jnp.eye(state_dim),
        factor=prior_factor,
        observation=jnp.ones(obs_dim),
        observation_matrix=jnp.eye(state_dim),
        observation_cov=0.1 * jnp.eye(obs_dim),
        max_iter=2,
    )
    assert factor_one_iter.shape == (state_dim, 1)
    assert factor_two_iters.shape == (state_dim, 2)


def test_cakf_predict_update_chain_jit_compatible() -> None:
    """A CAKF predict + update chain compiles under ``jax.jit``."""
    state_dim = 2
    obs_dim = 2

    def chain(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = jnp.zeros(state_dim)
        factor = jnp.zeros((state_dim, 0))
        transition = jnp.eye(state_dim)
        prior_cov = jnp.eye(state_dim)
        observation_matrix = jnp.eye(obs_dim)
        observation_cov = 0.1 * jnp.eye(obs_dim)
        predicted_mean, predicted_factor = cakf_predict(
            mean=mean, factor=factor, transition=transition
        )
        return cakf_update(
            mean=predicted_mean,
            prior_cov=prior_cov,
            factor=predicted_factor,
            observation=observation,
            observation_matrix=observation_matrix,
            observation_cov=observation_cov,
            max_iter=obs_dim,
        )

    observation = jax.random.normal(jax.random.PRNGKey(0), (obs_dim,))
    jitted = jax.jit(chain)
    mean, factor = jitted(observation)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(factor))


def test_cakf_update_posterior_covariance_is_positive_semidefinite() -> None:
    """The implicit posterior covariance ``Σ - M M^T`` is PSD."""
    state_dim = 4
    obs_dim = 4
    prior_cov = jnp.eye(state_dim) + 0.1 * jnp.ones((state_dim, state_dim))
    _, posterior_factor = cakf_update(
        mean=jnp.zeros(state_dim),
        prior_cov=prior_cov,
        factor=jnp.zeros((state_dim, 0)),
        observation=jnp.ones(obs_dim),
        observation_matrix=jnp.eye(state_dim),
        observation_cov=0.5 * jnp.eye(obs_dim),
        max_iter=obs_dim,
    )
    posterior_cov = prior_cov - posterior_factor @ posterior_factor.T
    symmetric = 0.5 * (posterior_cov + posterior_cov.T)
    eigenvalues = jnp.linalg.eigvalsh(symmetric)
    assert jnp.all(eigenvalues >= -1e-5)
