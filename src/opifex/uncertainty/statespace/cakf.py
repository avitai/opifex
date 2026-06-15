r"""Compute-aware Kalman filter (CAKF) primitives.

The CAKF propagates a low-rank correction factor :math:`M` to the prior
marginal covariance :math:`\Sigma_k` so that the posterior covariance is
the implicit ``LowRankDowndatedMatrix`` :math:`\Sigma_k - M\,M^\top`. The
update step iteratively expands :math:`M` by one column per CG iteration
until either ``max_iter`` is reached or the residual norm falls below a
tolerance.

Canonical reference (line-by-line port):
* ``../ComputationAwareKalman.jl/src/low_rank.jl`` —
  ``LowRankDowndatedMatrix``.
* ``../ComputationAwareKalman.jl/src/filter/predict.jl`` — ``predict``.
* ``../ComputationAwareKalman.jl/src/filter/update.jl`` — ``update``.
* ``../ComputationAwareKalman.jl/src/filter/policy.jl`` — ``CGPolicy``
  (default search direction is the residual vector).

References
----------
* Pförtner, Wenger, Cockayne, Hennig 2024 — *Computation-Aware Kalman
  Filtering and Smoothing*, arXiv:2405.08971.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True, kw_only=True)
class LowRankDowndatedMatrix:
    """Implicit ``A - U V^T`` matrix for low-rank-corrected covariances.

    Attributes:
        dense: The base dense matrix ``A`` of shape ``(n, n)`` (typically
            the prior marginal covariance ``Σ``).
        left: Left low-rank factor ``U`` of shape ``(n, r)``.
        right: Right low-rank factor ``V`` of shape ``(n, r)``.
    """

    dense: jax.Array
    left: jax.Array
    right: jax.Array

    def __matmul__(self, vector: jax.Array) -> jax.Array:
        """Compute ``(A - U V^T) @ vector``."""
        return self.dense @ vector - self.left @ (self.right.T @ vector)


def cakf_predict(
    *,
    mean: jax.Array,
    factor: jax.Array,
    transition: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Propagate the CAKF state through a transition matrix.

    Ports ``ComputationAwareKalman.jl/src/filter/predict.jl``. The prior
    marginal covariance ``Σ_{k+1}`` is determined by the Gauss-Markov
    chain externally — only the mean and the low-rank correction factor
    flow through the transition.

    Args:
        mean: Filter mean at time ``k`` with shape ``(n,)``.
        factor: Low-rank correction factor ``M`` with shape ``(n, r)``.
        transition: Transition matrix ``A_k`` with shape ``(n, n)``.

    Returns:
        Predicted ``(mean, factor)`` for time ``k + 1``.
    """
    return transition @ mean, transition @ factor


def cakf_update(
    *,
    mean: jax.Array,
    prior_cov: jax.Array,
    factor: jax.Array,
    observation: jax.Array,
    observation_matrix: jax.Array,
    observation_cov: jax.Array,
    max_iter: int,
) -> tuple[jax.Array, jax.Array]:
    """CAKF update step using the CG search-direction policy.

    Iterates up to ``max_iter`` times. Each iteration:

    1. Uses the current residual ``r`` as the search direction
       (``CGPolicy``).
    2. Conjugates ``r`` against previously selected directions via
       ``d = r - U (U^T S r)`` where ``S`` is the symmetric operator
       ``H Σ H^T - H M M^T H^T + Λ`` and ``U`` is the running
       Gram-Schmidt factor.
    3. Normalises ``d`` by ``η = r^T S d`` and appends ``sqrt(1/η) d``
       to ``U``.
    4. Updates the action ``u`` so that ``H^T u`` is the cumulative
       posterior-mean correction.

    Args:
        mean: Prior mean with shape ``(n,)``.
        prior_cov: Prior marginal covariance ``Σ`` of shape ``(n, n)``.
        factor: Existing low-rank correction factor ``M`` of shape
            ``(n, r0)``. May be empty (``r0 = 0``).
        observation: Observed value with shape ``(k,)``.
        observation_matrix: Observation matrix ``H`` of shape ``(k, n)``.
        observation_cov: Observation noise ``Λ`` of shape ``(k, k)``.
        max_iter: Maximum number of CG iterations (must be a static
            Python ``int`` so the shapes of ``U`` and ``M_new`` are known
            at trace time).

    Returns:
        Posterior ``(mean, factor)``. The returned factor has shape
        ``(n, r0 + max_iter)``.
    """
    obs_dim = observation_matrix.shape[0]
    cov_obs_t = prior_cov @ observation_matrix.T  # (n, k)
    factor_obs = observation_matrix @ factor  # (k, r0)

    def s_apply(vector: jax.Array) -> jax.Array:
        """Apply ``S = H Σ H^T - (H M)(H M)^T + Λ`` to ``vector``."""
        return (
            observation_matrix @ (cov_obs_t @ vector)
            - factor_obs @ (factor_obs.T @ vector)
            + observation_cov @ vector
        )

    def body(
        carry: tuple[jax.Array, jax.Array], _: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        action, gram_factor = carry
        residual = observation - observation_matrix @ mean - s_apply(action)
        direction = residual - gram_factor @ (gram_factor.T @ s_apply(residual))
        eta = residual @ s_apply(direction)
        # Avoid division by zero when the search direction is exhausted;
        # the step becomes a no-op (the residual is already in the span of
        # previously selected directions).
        safe_eta = jnp.where(eta > 0.0, eta, jnp.ones_like(eta))
        valid = eta > 0.0
        alpha = residual @ residual
        scaled_alpha = jnp.where(valid, alpha / safe_eta, 0.0)
        normaliser = jnp.where(valid, jnp.sqrt(1.0 / safe_eta), 0.0)
        new_action = action + scaled_alpha * direction
        new_column = normaliser * direction
        new_gram_factor = gram_factor + jnp.outer(new_column, jax.nn.one_hot(_, obs_dim))
        return (new_action, new_gram_factor), new_column

    initial_action = jnp.zeros(obs_dim)
    initial_gram_factor = jnp.zeros((obs_dim, obs_dim))
    if max_iter == 0:
        return mean, factor
    (final_action, _), _ = jax.lax.scan(
        body, (initial_action, initial_gram_factor), jnp.arange(max_iter)
    )

    # Posterior mean update: m + Σ H^T u - M (M^T H^T u)
    cov_obs_action = cov_obs_t @ final_action
    factor_correction = factor @ (factor_obs.T @ final_action)
    posterior_mean = mean + cov_obs_action - factor_correction

    # Posterior factor: append the rank-one corrections from each iteration.
    # The implicit posterior cov is Σ - M_post @ M_post.T where
    # M_post = [M, (Σ - M M^T) H^T U].
    initial_action_carry = jnp.zeros(obs_dim)
    initial_gram_carry = jnp.zeros((obs_dim, obs_dim))

    def collect(
        carry: tuple[jax.Array, jax.Array], _: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        action, gram_factor = carry
        residual = observation - observation_matrix @ mean - s_apply(action)
        direction = residual - gram_factor @ (gram_factor.T @ s_apply(residual))
        eta = residual @ s_apply(direction)
        safe_eta = jnp.where(eta > 0.0, eta, jnp.ones_like(eta))
        valid = eta > 0.0
        alpha = residual @ residual
        scaled_alpha = jnp.where(valid, alpha / safe_eta, 0.0)
        normaliser = jnp.where(valid, jnp.sqrt(1.0 / safe_eta), 0.0)
        new_action = action + scaled_alpha * direction
        new_column = normaliser * direction
        new_gram = gram_factor + jnp.outer(new_column, jax.nn.one_hot(_, obs_dim))
        return (new_action, new_gram), new_column

    _, all_columns = jax.lax.scan(
        collect, (initial_action_carry, initial_gram_carry), jnp.arange(max_iter)
    )
    # all_columns has shape (max_iter, obs_dim)
    u_matrix = all_columns.T  # (obs_dim, max_iter)
    cov_minus_low_rank_obs = cov_obs_t - factor @ factor_obs.T  # (n, k)
    new_columns = cov_minus_low_rank_obs @ u_matrix  # (n, max_iter)
    posterior_factor = jnp.concatenate([factor, new_columns], axis=1)
    return posterior_mean, posterior_factor
