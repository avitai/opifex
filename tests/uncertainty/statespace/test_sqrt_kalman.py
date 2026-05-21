"""Tests for the square-root Kalman primitives.

The square-root form propagates a lower-triangular factor ``L`` such that
``P = L @ L.T`` instead of the symmetric covariance ``P`` directly. This
avoids the squared condition number of ``P`` and guarantees positive
semi-definiteness on every step — critical for poorly-conditioned
problems and long observation chains.

Canonical reference (line-by-line port):
* QR-based revert step — ``../probdiffeq/probdiffeq/util/cholesky_util.py``
  ``revert_conditional`` (line 51) and ``triu_via_qr`` (line 138).
* Closed-form Kalman comparison — ``../bayesnewton/bayesnewton/ops.py``
  ``_sequential_kf`` (line 154).

References
----------
* Kaminski, Bryson, Schmidt 1971 — *Discrete square root filtering: a
  survey of current techniques*, IEEE TAC 16(6).
* Bierman, G. J. 1977 — *Factorization Methods for Discrete Sequential
  Estimation*.
* Grewal & Andrews 2014 — *Kalman Filtering: Theory and Practice* §6.5.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import (
    kalman_predict,
    kalman_update,
    sqrt_kalman_predict,
    sqrt_kalman_update,
)


def _to_cov(cov_sqrt: jax.Array) -> jax.Array:
    """Reconstruct the symmetric covariance from a left square-root."""
    return cov_sqrt @ cov_sqrt.T


def test_sqrt_kalman_predict_matches_standard_kalman_predict() -> None:
    """Square-root predict reproduces ``A P A^T + Q`` on a well-conditioned system."""
    mean = jnp.asarray([1.0, -2.0, 0.5])
    cov = jnp.asarray([[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 0.8]])
    cov_sqrt = jnp.linalg.cholesky(cov)
    transition = jnp.asarray([[1.0, 0.1, 0.0], [0.0, 0.95, 0.05], [0.0, 0.0, 0.9]])
    process_noise = jnp.asarray([[0.1, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.02]])
    process_noise_sqrt = jnp.linalg.cholesky(process_noise)

    sqrt_mean, sqrt_cov_factor = sqrt_kalman_predict(
        mean=mean,
        cov_sqrt=cov_sqrt,
        transition=transition,
        process_noise_sqrt=process_noise_sqrt,
    )
    ref_mean, ref_cov = kalman_predict(
        mean=mean, cov=cov, transition=transition, process_noise=process_noise
    )
    assert jnp.allclose(sqrt_mean, ref_mean, atol=1e-6)
    assert jnp.allclose(_to_cov(sqrt_cov_factor), ref_cov, atol=1e-5)


def test_sqrt_kalman_update_matches_standard_kalman_update() -> None:
    """Square-root update reproduces the joseph-form posterior covariance."""
    prior_mean = jnp.asarray([0.5, 1.0])
    prior_cov = jnp.asarray([[1.0, 0.2], [0.2, 0.8]])
    prior_cov_sqrt = jnp.linalg.cholesky(prior_cov)
    observation = jnp.asarray([0.3, 1.1])
    observation_matrix = jnp.asarray([[1.0, 0.0], [0.5, 1.0]])
    observation_cov = jnp.asarray([[0.05, 0.0], [0.0, 0.07]])
    observation_cov_sqrt = jnp.linalg.cholesky(observation_cov)

    sqrt_mean, sqrt_cov_factor = sqrt_kalman_update(
        mean=prior_mean,
        cov_sqrt=prior_cov_sqrt,
        observation=observation,
        observation_matrix=observation_matrix,
        observation_cov_sqrt=observation_cov_sqrt,
    )
    ref_mean, ref_cov = kalman_update(
        mean=prior_mean,
        cov=prior_cov,
        observation=observation,
        observation_matrix=observation_matrix,
        observation_cov=observation_cov,
    )
    assert jnp.allclose(sqrt_mean, ref_mean, atol=1e-5)
    assert jnp.allclose(_to_cov(sqrt_cov_factor), ref_cov, atol=1e-5)


def test_sqrt_kalman_preserves_positive_definiteness_under_ill_conditioning() -> None:
    """Sqrt form returns a valid factor on an ill-conditioned prior.

    By construction the reconstructed covariance ``L_post L_post^T`` is
    symmetric positive semi-definite — the QR factorisation produces an
    upper-triangular ``R`` whose ``R^T R`` is automatically PSD. The dense
    joseph-form update can produce slightly non-PSD matrices under the
    same roundoff conditions.
    """
    state_dim = 3
    # Span six orders of magnitude — comfortably inside float32 precision
    # but enough to expose roundoff problems in dense Kalman.
    eigenvalues = jnp.asarray([1.0, 1e-3, 1e-6])
    rotation = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (state_dim, state_dim)))[0]
    prior_cov = (rotation * eigenvalues) @ rotation.T
    prior_cov_sqrt = jnp.linalg.cholesky(prior_cov)
    observation_matrix = jnp.eye(state_dim)
    observation_cov_sqrt = jnp.sqrt(1e-4) * jnp.eye(state_dim)

    _, post_cov_factor = sqrt_kalman_update(
        mean=jnp.zeros(state_dim),
        cov_sqrt=prior_cov_sqrt,
        observation=jnp.zeros(state_dim),
        observation_matrix=observation_matrix,
        observation_cov_sqrt=observation_cov_sqrt,
    )
    assert jnp.all(jnp.isfinite(post_cov_factor))
    posterior_cov = _to_cov(post_cov_factor)
    eigenvalues_post = jnp.linalg.eigvalsh(posterior_cov)
    # All eigenvalues must be non-negative (PSD guarantee by construction).
    assert jnp.all(eigenvalues_post >= -1e-7)


def test_sqrt_kalman_predict_handles_zero_process_noise() -> None:
    """Lucky case: deterministic dynamics. Sqrt predict must still produce a
    valid left-triangular factor (no NaN, PSD reconstruction)."""
    mean = jnp.asarray([1.0, 2.0])
    cov_sqrt = jnp.asarray([[1.0, 0.0], [0.0, 0.5]])
    transition = jnp.asarray([[1.0, 0.1], [0.0, 1.0]])
    process_noise_sqrt = jnp.zeros((2, 2))

    _, new_cov_factor = sqrt_kalman_predict(
        mean=mean,
        cov_sqrt=cov_sqrt,
        transition=transition,
        process_noise_sqrt=process_noise_sqrt,
    )
    reconstructed = _to_cov(new_cov_factor)
    expected = transition @ (cov_sqrt @ cov_sqrt.T) @ transition.T
    assert jnp.all(jnp.isfinite(new_cov_factor))
    assert jnp.allclose(reconstructed, expected, atol=1e-6)


def test_sqrt_kalman_chain_jit_compatible() -> None:
    """A predict-update chain compiles under ``jax.jit`` end-to-end."""
    state_dim = 2
    obs_dim = 2

    def chain(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = jnp.zeros(state_dim)
        cov_sqrt = jnp.eye(state_dim)
        transition = jnp.eye(state_dim)
        process_noise_sqrt = 0.1 * jnp.eye(state_dim)
        observation_matrix = jnp.eye(obs_dim)
        observation_cov_sqrt = 0.1 * jnp.eye(obs_dim)
        pred_mean, pred_cov_sqrt = sqrt_kalman_predict(
            mean=mean,
            cov_sqrt=cov_sqrt,
            transition=transition,
            process_noise_sqrt=process_noise_sqrt,
        )
        return sqrt_kalman_update(
            mean=pred_mean,
            cov_sqrt=pred_cov_sqrt,
            observation=observation,
            observation_matrix=observation_matrix,
            observation_cov_sqrt=observation_cov_sqrt,
        )

    observation = jax.random.normal(jax.random.PRNGKey(0), (obs_dim,))
    jitted = jax.jit(chain)
    mean, cov_sqrt = jitted(observation)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(cov_sqrt))
