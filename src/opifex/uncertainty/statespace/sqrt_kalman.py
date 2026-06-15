"""Square-root Kalman primitives — predict and update on Cholesky factors.

Propagates a lower-triangular factor ``L`` with ``P = L @ L.T`` instead of
the symmetric covariance ``P``. Avoids the squared condition number of the
covariance form and preserves positive semi-definiteness by construction.

Canonical reference (line-by-line port):
* ``../probdiffeq/probdiffeq/util/cholesky_util.py`` — ``revert_conditional``
  (line 51) for the update step and ``triu_via_qr`` (line 138) for the
  predict step.

References
----------
* Kaminski, Bryson, Schmidt 1971 — *Discrete square root filtering: a
  survey of current techniques*, IEEE TAC 16(6).
* Bierman, G. J. 1977 — *Factorization Methods for Discrete Sequential
  Estimation*.
* Grewal & Andrews 2014 — *Kalman Filtering: Theory and Practice* §6.5.

Notation:
    All ``cov_sqrt`` arguments and return values are **left** square roots:
    ``P = cov_sqrt @ cov_sqrt.T``. The QR-based internal pipeline uses
    right square roots ``R = L.T`` because QR produces upper-triangular
    factors naturally — the conversion is a transpose.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _triu_via_qr(matrix: jax.Array) -> jax.Array:
    """Upper-triangularise via QR — ports probdiffeq triu_via_qr (line 138)."""
    _, upper = jnp.linalg.qr(matrix, mode="reduced")
    return upper


def sqrt_kalman_predict(
    *,
    mean: jax.Array,
    cov_sqrt: jax.Array,
    transition: jax.Array,
    process_noise_sqrt: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Square-root predict step.

    Computes ``(A m, L')`` where ``L' L'^T = A L L^T A^T + Q_sqrt Q_sqrt^T``.

    The new factor is obtained by stacking the right square roots
    ``[(A L)^T; Q_sqrt^T]`` and reading off the upper-triangular ``R`` of
    a thin QR factorisation: ``R^T R = (A L)(A L)^T + Q_sqrt Q_sqrt^T``.
    The returned left factor is ``L' = R^T``.

    Args:
        mean: Prior mean ``m`` with shape ``(n,)``.
        cov_sqrt: Prior left square root ``L`` with shape ``(n, n)`` such
            that ``P = L @ L.T``.
        transition: Transition matrix ``A`` with shape ``(n, n)``.
        process_noise_sqrt: Left square root of the process-noise
            covariance, shape ``(n, n)``.

    Returns:
        Predicted mean ``A m`` and the lower-triangular factor ``L'`` of
        the predicted covariance.
    """
    predicted_mean = transition @ mean
    stacked = jnp.concatenate([(transition @ cov_sqrt).T, process_noise_sqrt.T], axis=0)
    upper = _triu_via_qr(stacked)
    return predicted_mean, upper.T


def sqrt_kalman_update(
    *,
    mean: jax.Array,
    cov_sqrt: jax.Array,
    observation: jax.Array,
    observation_matrix: jax.Array,
    observation_cov_sqrt: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Square-root update step via the joint-QR revert formula.

    Given prior ``N(m, L L^T)`` and observation model
    ``y | x ~ N(H x, R_sqrt R_sqrt^T)``, computes the posterior
    ``N(m', L'_post L'_post^T)`` using the QR-revert identity
    (probdiffeq ``revert_conditional`` line 51):

    Stack
    ::

        R_block = [[R_YX,    0  ],
                   [R_X_F,  R_X ]]

    where ``R_X_F = (H L)^T`` (shape ``(n, k)``),
    ``R_X = L^T`` (shape ``(n, n)``) and ``R_YX = R_sqrt^T`` (shape
    ``(k, k)``). Computing ``R_full = qr_r(R_block)`` and partitioning
    yields ``r_obs`` (innovation sqrt), ``r_cor`` (posterior sqrt) and
    the Kalman gain ``G = solve_triangular(r_obs, R12, lower=False).T``.

    Args:
        mean: Prior mean with shape ``(n,)``.
        cov_sqrt: Prior left square root ``L`` with shape ``(n, n)``.
        observation: Observed value with shape ``(k,)``.
        observation_matrix: Observation matrix ``H`` with shape ``(k, n)``.
        observation_cov_sqrt: Left square root of observation noise,
            shape ``(k, k)``.

    Returns:
        Posterior mean and lower-triangular posterior covariance factor.
    """
    obs_dim = observation_matrix.shape[0]
    state_dim = mean.shape[0]

    r_x_f = (observation_matrix @ cov_sqrt).T
    r_x = cov_sqrt.T
    r_yx = observation_cov_sqrt.T

    upper_left = jnp.concatenate([r_yx, jnp.zeros((obs_dim, state_dim), dtype=r_yx.dtype)], axis=1)
    lower_left = jnp.concatenate([r_x_f, r_x], axis=1)
    block = jnp.concatenate([upper_left, lower_left], axis=0)
    upper = _triu_via_qr(block)

    r_obs = upper[:obs_dim, :obs_dim]
    r_cross = upper[:obs_dim, obs_dim:]
    r_post = upper[obs_dim:, obs_dim:]

    gain = jax.scipy.linalg.solve_triangular(r_obs, r_cross, lower=False).T
    innovation = observation - observation_matrix @ mean
    posterior_mean = mean + gain @ innovation
    return posterior_mean, r_post.T
