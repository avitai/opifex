r"""JAX-native data likelihoods for probabilistic ODE solvers.

Two log-likelihood combinators referenced by the probabilistic-numerics
adapter catalogue:

* :func:`fenrir_data_loglik` — Fenrir post-solve smoothing likelihood
  (Tronarp et al, "Fenrir: Physics-Enhanced Regression for Initial
  Value Problems", ICML 2022, arXiv:2202.01287). Backward-conditioning
  variant of the RTS smoother: given filter outputs from an
  unconditioned forward pass, the routine sweeps backward and applies
  a Kalman measurement update at every index that carries an
  observation, accumulating the innovation log-density. The result
  equals the data marginal log-likelihood under the smoothed
  state-space model.

  Sibling reference (READ-ONLY port — never imported at runtime):
  ``ProbNumDiffEq.jl/src/data_likelihoods/fenrir.jl:30-128`` —
  specifically ``fenrir_data_loglik`` (lines 30-64) and the
  ``fit_pnsolution_to_data!`` helper (lines 67-128).

* :func:`dalton_data_loglik` — DALTON three-term combinator
  ``data_ll + with_pn_ll - without_pn_ll`` (Wu et al, "Data-Adaptive
  Probabilistic Likelihood Approximation for Ordinary Differential
  Equations", arXiv 2306.05566). The data log-likelihood from a
  data-conditioned solver pass is combined with the differential in
  the solver's probabilistic-numerics likelihood between the
  data-conditioned and unconditioned passes.

  Sibling reference (READ-ONLY port — never imported at runtime):
  ``ProbNumDiffEq.jl/src/data_likelihoods/dalton.jl:69-75``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _measure_and_update_with_loglik(
    *,
    mean: jax.Array,
    cov: jax.Array,
    observation: jax.Array,
    observation_matrix: jax.Array,
    observation_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single Kalman measurement update returning the innovation log-density.

    Mirrors ``measure_and_update!`` from the Julia reference (line 130 of
    ``fenrir.jl``): the innovation Gaussian
    ``N(0, H P H^T + R)`` evaluated at ``observation - H mean`` is the
    Fenrir per-step likelihood contribution.
    """
    innovation = observation - observation_matrix @ mean
    cov_obs = observation_matrix @ cov
    innovation_cov = cov_obs @ observation_matrix.T + observation_cov
    cholesky = jnp.linalg.cholesky(innovation_cov)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
    whitened = jax.scipy.linalg.solve_triangular(cholesky, innovation, lower=True)
    obs_dim = observation.shape[0]
    log_density = -0.5 * (jnp.sum(whitened**2) + log_det + obs_dim * jnp.log(2.0 * jnp.pi))

    gain = jnp.linalg.solve(innovation_cov, cov_obs).T
    updated_mean = mean + gain @ innovation
    updated_cov = cov - gain @ cov_obs
    return updated_mean, updated_cov, log_density


def _conditional_update(
    *,
    mean: jax.Array,
    cov: jax.Array,
    observation: jax.Array,
    observation_matrix: jax.Array,
    observation_cov: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply a measurement update only at indices where ``mask`` is ``True``.

    Both branches are traced (jit-friendly) and the mask selects between
    the updated and pre-update tuples. ``observation`` must be finite at
    every index — masked-False entries are tolerated but unused.
    """
    updated_mean, updated_cov, log_density = _measure_and_update_with_loglik(
        mean=mean,
        cov=cov,
        observation=observation,
        observation_matrix=observation_matrix,
        observation_cov=observation_cov,
    )
    selected_mean = jnp.where(mask, updated_mean, mean)
    selected_cov = jnp.where(mask, updated_cov, cov)
    selected_ll = jnp.where(mask, log_density, jnp.asarray(0.0))
    return selected_mean, selected_cov, selected_ll


def fenrir_data_loglik(
    *,
    filter_means: jax.Array,
    filter_covs: jax.Array,
    transitions: jax.Array,
    process_noises: jax.Array,
    data: jax.Array,
    data_mask: jax.Array,
    observation_matrix: jax.Array,
    observation_cov: jax.Array,
) -> jax.Array:
    r"""Fenrir post-solve smoothing data log-likelihood (Tronarp et al, 2022).

    Given filter outputs from an *unconditioned* forward pass (no data
    seen), this function sweeps backward applying a Kalman update at
    each index with an observation. The total log-likelihood
    accumulated is the data marginal log-likelihood under the smoothed
    posterior. For a pure linear-Gaussian model with data at every
    step, the result equals the standard forward Kalman marginal
    log-likelihood (Bayes' chain rule applied in reverse).

    Sibling reference (READ-ONLY port — no runtime import):
    ``ProbNumDiffEq.jl/src/data_likelihoods/fenrir.jl:30-128``.

    Args:
        filter_means: Forward-pass filter means, shape
            ``(num_steps, state_dim)``.
        filter_covs: Forward-pass filter covariances, shape
            ``(num_steps, state_dim, state_dim)``.
        transitions: State-transition matrices, shape
            ``(num_steps, state_dim, state_dim)``. ``transitions[i]``
            propagates step ``i-1`` to step ``i``.
        process_noises: Process-noise covariances, shape
            ``(num_steps, state_dim, state_dim)``.
        data: Observations, shape ``(num_steps, obs_dim)``. Entries at
            indices where ``data_mask`` is ``False`` are ignored but
            must be finite.
        data_mask: Boolean mask, shape ``(num_steps,)``. ``True`` at
            indices that carry an observation.
        observation_matrix: Linear observation operator ``H``, shape
            ``(obs_dim, state_dim)``.
        observation_cov: Observation noise covariance ``R``, shape
            ``(obs_dim, obs_dim)``.

    Returns:
        Scalar log-likelihood of ``data`` (where ``data_mask`` is
        ``True``) under the smoothed posterior.
    """
    last_mean = filter_means[-1]
    last_cov = filter_covs[-1]
    last_mean_post, last_cov_post, last_ll = _conditional_update(
        mean=last_mean,
        cov=last_cov,
        observation=data[-1],
        observation_matrix=observation_matrix,
        observation_cov=observation_cov,
        mask=data_mask[-1],
    )

    def body(
        carry: tuple[jax.Array, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        """Run one backward smoothing step and accumulate its log-likelihood term."""
        next_mean_post, next_cov_post = carry
        m_filt, p_filt, transition, process_noise, observation, mask = inputs
        predicted_mean = transition @ m_filt
        predicted_cov = transition @ p_filt @ transition.T + process_noise
        gain = jnp.linalg.solve(predicted_cov, transition @ p_filt).T
        marginal_mean = m_filt + gain @ (next_mean_post - predicted_mean)
        marginal_cov = p_filt + gain @ (next_cov_post - predicted_cov) @ gain.T
        new_mean, new_cov, log_density = _conditional_update(
            mean=marginal_mean,
            cov=marginal_cov,
            observation=observation,
            observation_matrix=observation_matrix,
            observation_cov=observation_cov,
            mask=mask,
        )
        return (new_mean, new_cov), log_density

    _, log_densities = jax.lax.scan(
        body,
        (last_mean_post, last_cov_post),
        (
            filter_means[:-1],
            filter_covs[:-1],
            transitions[1:],
            process_noises[1:],
            data[:-1],
            data_mask[:-1],
        ),
        reverse=True,
    )
    return last_ll + jnp.sum(log_densities)


def dalton_data_loglik(
    data_ll: jax.Array,
    with_pn_ll: jax.Array,
    without_pn_ll: jax.Array,
) -> jax.Array:
    r"""DALTON two-solve data log-likelihood combinator (Wu et al, 2023).

    The DALTON likelihood is the sum of (i) the per-observation
    log-likelihood accumulated by a data-conditioned solver pass and
    (ii) the differential in the solver's probabilistic-numerics
    log-likelihood between the data-conditioned and unconditioned
    passes:

    .. math::

        \ell_{\mathrm{DALTON}} = \ell_{\mathrm{data}}
            + \ell_{\mathrm{PN, with\ data}}
            - \ell_{\mathrm{PN, without\ data}}.

    Sibling reference (READ-ONLY port — no runtime import):
    ``ProbNumDiffEq.jl/src/data_likelihoods/dalton.jl:69-75``.

    Args:
        data_ll: Log-likelihood of observations under the
            data-conditioned solver pass.
        with_pn_ll: Probabilistic-numerics log-likelihood of the
            data-conditioned solver pass.
        without_pn_ll: Probabilistic-numerics log-likelihood of the
            unconditioned solver pass.

    Returns:
        Scalar DALTON log-likelihood.
    """
    return data_ll + with_pn_ll - without_pn_ll
