"""The CAKF filter loop: predict, update, truncate, carrying the mean and the low-rank factor.

The reference is ComputationAwareKalman.jl (Pförtner, Wenger, Cockayne, Hennig 2024) at
``fc6b4109``: ``src/filter/loop.jl`` (the loop), ``src/filter/update.jl`` (the residual tolerance
and the clamp of ``max_iter`` to the observation dimension), ``src/truncate.jl`` (the truncation).
Its own test (``test/gp_regression.jl``) runs CAKF on scalar observations and asserts the posterior
equals the exact one with Julia's ``≈``, whose default relative tolerance is ``sqrt(eps)`` of the
element type; the same rule is applied here in float32.
"""

from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from opifex.uncertainty.statespace import cakf_filter, cakf_truncate, cakf_update
from opifex.uncertainty.statespace.kalman import kalman_filter


RTOL = float(np.sqrt(np.finfo(np.float32).eps))
STATE_DIM = 3


class Model(TypedDict):
    """The keyword arguments ``kalman_filter`` and ``cakf_filter`` share."""

    transitions: jax.Array
    process_noises: jax.Array
    observations: jax.Array
    observation_matrix: jax.Array
    observation_covs: jax.Array
    initial_mean: jax.Array
    initial_cov: jax.Array


def _model(num_steps: int, obs_dim: int, seed: int) -> Model:
    """A stable random linear-Gaussian state-space model and observations drawn from it."""
    keys = jax.random.split(jax.random.key(seed), 5)
    rotation, _ = jnp.linalg.qr(jax.random.normal(keys[0], (STATE_DIM, STATE_DIM)))
    transition = 0.95 * rotation
    process_noise = 0.05 * jnp.eye(STATE_DIM)
    observation_matrix = jax.random.normal(keys[1], (obs_dim, STATE_DIM))
    observation_cov = 0.1 * jnp.eye(obs_dim)
    initial_cov = jnp.eye(STATE_DIM)

    def step(state: jax.Array, noise: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        process_sample, observation_sample = noise
        next_state = transition @ state + jnp.sqrt(0.05) * process_sample
        observation = observation_matrix @ next_state + jnp.sqrt(0.1) * observation_sample
        return next_state, observation

    _, observations = jax.lax.scan(
        step,
        jax.random.normal(keys[2], (STATE_DIM,)),
        (
            jax.random.normal(keys[3], (num_steps, STATE_DIM)),
            jax.random.normal(keys[4], (num_steps, obs_dim)),
        ),
    )
    return Model(
        transitions=jnp.broadcast_to(transition, (num_steps, STATE_DIM, STATE_DIM)),
        process_noises=jnp.broadcast_to(process_noise, (num_steps, STATE_DIM, STATE_DIM)),
        observations=observations,
        observation_matrix=observation_matrix,
        observation_covs=jnp.broadcast_to(observation_cov, (num_steps, obs_dim, obs_dim)),
        initial_mean=jnp.zeros(STATE_DIM),
        initial_cov=initial_cov,
    )


def _relative(difference: jax.Array, reference: jax.Array) -> float:
    return float(jnp.linalg.norm(difference) / jnp.linalg.norm(reference))


def _assert_equals_exact_filter(model: Model, max_iter: int) -> None:
    exact_means, exact_covs = kalman_filter(**model)
    means, factors, prior_covs = cakf_filter(**model, max_iter=max_iter)
    covs = prior_covs - factors @ jnp.swapaxes(factors, -1, -2)
    assert _relative(means - exact_means, exact_means) <= RTOL
    assert _relative(covs - exact_covs, exact_covs) <= RTOL


def test_cakf_filter_on_scalar_observations_equals_the_exact_kalman_filter() -> None:
    """One CG iteration spans a scalar observation, so CAKF is the exact filter."""
    _assert_equals_exact_filter(_model(num_steps=20, obs_dim=1, seed=0), max_iter=1)


def test_cakf_filter_with_full_iterations_equals_the_exact_kalman_filter() -> None:
    """``max_iter`` equal to the observation dimension recovers the exact filter."""
    _assert_equals_exact_filter(_model(num_steps=20, obs_dim=2, seed=1), max_iter=2)


def test_cakf_filter_with_masked_observations_equals_the_exact_kalman_filter() -> None:
    """Unobserved steps given a large noise variance behave as in the exact filter."""
    model = _model(num_steps=24, obs_dim=1, seed=2)
    observed = (jnp.arange(24) % 3 == 0)[:, None, None]
    model["observation_covs"] = jnp.where(observed, model["observation_covs"], 1e6)
    _assert_equals_exact_filter(model, max_iter=1)


def test_cakf_filter_factor_keeps_the_state_dimension_as_its_width() -> None:
    """Truncation keeps the carried factor at ``state_dim`` columns, as ``truncate.jl`` does."""
    model = _model(num_steps=5, obs_dim=2, seed=3)
    means, factors, prior_covs = cakf_filter(**model, max_iter=2)
    assert means.shape == (5, STATE_DIM)
    assert factors.shape == (5, STATE_DIM, STATE_DIM)
    assert prior_covs.shape == (5, STATE_DIM, STATE_DIM)


def test_cakf_filter_is_jit_and_vmap_compatible() -> None:
    """The filter compiles once and maps over a batch of observation sequences."""
    model = _model(num_steps=10, obs_dim=1, seed=4)
    eager_means, _, _ = cakf_filter(**model, max_iter=1)
    jit_means, _, _ = jax.jit(lambda m: cakf_filter(**m, max_iter=1))(model)
    assert _relative(jit_means - eager_means, eager_means) <= RTOL

    def means_for(observations: jax.Array) -> jax.Array:
        replaced = Model(**model)
        replaced["observations"] = observations
        return cakf_filter(**replaced, max_iter=1)[0]

    batch = jnp.stack([model["observations"], 2.0 * model["observations"]])
    batched = jax.vmap(means_for)(batch)
    assert batched.shape == (2, 10, STATE_DIM)
    assert _relative(batched[0] - eager_means, eager_means) <= RTOL


def test_cakf_update_clamps_max_iter_to_the_observation_dimension() -> None:
    """``max_iter`` above the observation dimension runs exactly ``obs_dim`` iterations."""
    obs_dim = 2

    def update(max_iter: int) -> tuple[jax.Array, jax.Array]:
        return cakf_update(
            mean=jnp.zeros(STATE_DIM),
            prior_cov=jnp.eye(STATE_DIM),
            factor=jnp.zeros((STATE_DIM, 0)),
            observation=jnp.array([1.0, -0.5]),
            observation_matrix=jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            observation_cov=0.1 * jnp.eye(obs_dim),
            max_iter=max_iter,
        )

    mean_at_dim, factor_at_dim = update(obs_dim)
    mean_above, factor_above = update(obs_dim + 3)
    assert factor_above.shape == (STATE_DIM, obs_dim)
    assert _relative(mean_above - mean_at_dim, mean_at_dim) <= RTOL
    assert _relative(factor_above - factor_at_dim, factor_at_dim) <= RTOL


def test_cakf_update_stops_once_the_residual_is_within_tolerance() -> None:
    """An observation the prior already explains leaves the mean and adds no information."""
    observation_matrix = jnp.array([[1.0, 0.0, 0.0]])
    mean = jnp.array([0.7, -0.2, 0.1])
    posterior_mean, factor = cakf_update(
        mean=mean,
        prior_cov=jnp.eye(STATE_DIM),
        factor=jnp.zeros((STATE_DIM, 0)),
        observation=observation_matrix @ mean,
        observation_matrix=observation_matrix,
        observation_cov=0.1 * jnp.eye(1),
        max_iter=1,
    )
    assert bool(jnp.all(posterior_mean == mean))
    assert bool(jnp.all(factor == 0.0))


def test_cakf_truncate_keeps_the_product_and_drops_small_singular_values() -> None:
    """``M M^T`` survives truncation; singular values below ``sqrt(eps)`` are removed."""
    min_sval = RTOL
    directions, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(5), (STATE_DIM, STATE_DIM)))
    singular_values = jnp.array([2.0, 0.5, 0.1 * min_sval])
    factor = jnp.concatenate([directions * singular_values, jnp.zeros((STATE_DIM, 2))], axis=1)
    truncated = cakf_truncate(factor)
    assert truncated.shape == (STATE_DIM, STATE_DIM)
    kept = directions[:, :2] * singular_values[:2]
    assert _relative(truncated @ truncated.T - kept @ kept.T, kept @ kept.T) <= RTOL
    assert float(jnp.min(jnp.linalg.svd(truncated, compute_uv=False))) == 0.0
