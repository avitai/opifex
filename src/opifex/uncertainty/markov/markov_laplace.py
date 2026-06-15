r"""Laplace approximation on Markov GPs — Task 11.2 slice 25.

Bridges the per-observation Laplace machinery from
:mod:`opifex.uncertainty.gp.laplace` (Task 11.1 D5) into the
state-space / Markov-GP form. At every Newton iteration:

1. Evaluate the per-observation ``(log_lik, ∇log_lik, W)`` quadruple
   at the current mean trajectory ``f̂^{(t)}``.
2. Form the **Gaussian-equivalent pseudo-observations**

   .. math::

       y_i^{\text{pseudo}} = \hat{f}_i^{(t)}
           + \frac{\nabla \log p(y_i \mid \hat{f}_i^{(t)})}{W_i},
       \qquad R_i = \frac{1}{W_i}.

   These come from rewriting the canonical Newton update
   ``(W + K^{-1}) f^{(t+1)} = W f^{(t)} + ∇log p`` as a
   pseudo-Gaussian observation with mean ``f^{(t)} + grad/W`` and
   noise covariance ``1/W`` (RW06 §3.4 derivation re-cast in
   state-space form per bayesnewton's ``Laplace`` inference module).
3. Run :func:`opifex.uncertainty.statespace.kalman_filter` +
   :func:`kalman_smoother` on the linearised model.
4. Replace ``f̂^{(t+1)}`` with the smoothed posterior mean.

Convergence after ``num_iterations`` Newton steps produces the
Laplace approximation to the non-Gaussian Markov-GP posterior — the
``O(n)``-per-iteration counterpart to D5's ``O(n³)`` direct-GP path.

The same ``LikelihoodComponentsFn`` interface used by D5 plugs in
unchanged here, so the Bernoulli / Poisson / Student-t / Beta
components shipped in
:mod:`opifex.uncertainty.gp.laplace_classification` and
:mod:`opifex.uncertainty.gp.laplace_likelihoods` work without
modification on Markov-GP priors.

References
----------
* Wilkinson, Solin, Adam 2020+ — ``bayesnewton/inference.py``
  Laplace family (PRIMARY reference for the state-space-Newton
  bridge).
* Sarkka 2013 — *Bayesian Filtering and Smoothing*, CUP §9
  (Iterated Extended Kalman Smoother).
* Rasmussen & Williams 2006 §3.4 (Newton-Laplace on conjugate GPs).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from opifex.uncertainty._predictive import gaussian_process_predictive
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.laplace import LikelihoodComponentsFn  # noqa: TC001 — runtime use
from opifex.uncertainty.markov._likelihood_support import interpolate_smoothed_state
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace import (
    kalman_filter,
    kalman_smoother,
    StateSpaceKernel,
)
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


_MARKOV_LAPLACE_SOURCE_PACKAGE = "opifex.uncertainty.markov"
_PSEUDO_NOISE_FLOOR: float = 1e-6
"""Lower clip for the per-step pseudo-noise variance ``1 / W_i`` to
keep the Kalman update PSD when the curvature ``W_i`` is small.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkovLaplaceGPState:
    """Fitted state for the Markov-Laplace non-conjugate GP.

    Attributes:
        times: ``(n,)`` strictly-increasing training time stamps.
        observations: ``(n,)`` training observations.
        smoothed_means: ``(n,)`` posterior mean of the latent
            ``f(t)`` at training times.
        smoothed_variances: ``(n,)`` posterior variance of the
            latent (marginal of the smoothed state via the kernel's
            measurement operator).
        smoothed_state_means: ``(n, state_dim)`` smoothed full state
            trajectory (used at predict time to propagate to
            held-out times).
        smoothed_state_covariances: ``(n, state_dim, state_dim)``
            full smoothed state covariances.
        log_marginal_likelihood: Laplace-approximated log marginal
            (scalar).
        state_space_kernel: The :class:`StateSpaceKernel` used at
            fit time (closed over for the predict path).
        log_likelihood_components_fn: The per-observation likelihood
            quadruple ``(log_lik, ∇log_lik, W, √W)`` callable used at
            fit time. Kept on the state so that
            :func:`predict_markov_laplace_gp` can re-evaluate
            ``W`` / ``grad`` at the smoothed mode for the
            response-distribution map.
    """

    times: jax.Array
    observations: jax.Array
    smoothed_means: jax.Array
    smoothed_variances: jax.Array
    smoothed_state_means: jax.Array
    smoothed_state_covariances: jax.Array
    log_marginal_likelihood: jax.Array
    state_space_kernel: StateSpaceKernel
    log_likelihood_components_fn: LikelihoodComponentsFn


def _build_state_space_sequence(
    *, times: jax.Array, state_space_kernel: StateSpaceKernel
) -> tuple[jax.Array, jax.Array]:
    r"""Discretise the kernel SDE at the training time grid.

    Returns ``(transitions, process_noises)`` arrays of shape
    ``(n, d, d)`` where ``d`` is the kernel's state dimension. The
    process noise uses the stationary-covariance identity
    ``Q_k(\Delta t) = P_\infty - A(\Delta t)\,P_\infty\,A(\Delta t)^T``
    (Sarkka 2013 §6.3 — valid because the prior is stationary).
    """
    deltas = jnp.concatenate([jnp.zeros((1,), dtype=times.dtype), jnp.diff(times)])
    transitions = jax.vmap(state_space_kernel.state_transition)(deltas)
    stationary_cov = state_space_kernel.stationary_cov
    process_noises = stationary_cov[None] - jnp.einsum(
        "kij,jl,kml->kim", transitions, stationary_cov, transitions
    )
    return transitions, process_noises


def fit_markov_laplace_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    log_likelihood_components_fn: LikelihoodComponentsFn,
    num_iterations: int = 25,
) -> MarkovLaplaceGPState:
    r"""Fit the Laplace approximation for a Markov-GP non-conjugate likelihood.

    Runs ``num_iterations`` Newton iterations through the
    pseudo-Gaussian-observation linearisation; each iteration is a
    single Kalman filter + smoother pass over the time grid (linear
    in ``n``).

    Args:
        times: ``(n,)`` strictly-increasing training time stamps.
        observations: ``(n,)`` training observations in the
            likelihood's support.
        state_space_kernel: SDE-form GP prior. Any
            :class:`opifex.uncertainty.statespace.StateSpaceKernel`
            (matern12 / 32 / 52 / 72 / cosine / periodic /
            quasi-periodic) is supported.
        log_likelihood_components_fn: Callable
            ``(f, y) -> (log_lik_total, ∇log_lik, W, √W)`` returning
            a scalar ``log_lik_total`` and per-observation ``(n,)``
            gradient + curvature + sqrt-curvature arrays. The same
            interface as :data:`opifex.uncertainty.gp.laplace.LikelihoodComponentsFn`.
        num_iterations: Newton-loop count (static under ``jax.jit``).

    Returns:
        :class:`MarkovLaplaceGPState` carrying smoothed posterior
        moments, full smoothed state, and the Laplace-approximated
        log marginal.
    """
    transitions, process_noises = _build_state_space_sequence(
        times=times, state_space_kernel=state_space_kernel
    )
    observation_matrix = state_space_kernel.measurement
    initial_mean = jnp.zeros(state_space_kernel.state_dim)
    initial_cov = state_space_kernel.stationary_cov

    def newton_step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        _: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        latent_mean, _smoothed_state_means, _smoothed_state_covs = carry
        _, grad, w_diag, _sqrt_w = log_likelihood_components_fn(latent_mean, observations)
        # Pseudo-Gaussian observation: y_pseudo_i = f_i + grad_i / W_i.
        # Pseudo-noise variance: R_i = 1 / W_i, clipped from below for PSD safety.
        safe_w = jnp.maximum(w_diag, _PSEUDO_NOISE_FLOOR)
        pseudo_observations = (latent_mean + grad / safe_w).reshape(-1, 1)
        pseudo_obs_covs = (1.0 / safe_w).reshape(-1, 1, 1)
        filter_means, filter_covs = kalman_filter(
            transitions=transitions,
            process_noises=process_noises,
            observations=pseudo_observations,
            observation_matrix=observation_matrix,
            observation_covs=pseudo_obs_covs,
            initial_mean=initial_mean,
            initial_cov=initial_cov,
        )
        smoothed_state_means, smoothed_state_covs = kalman_smoother(
            filter_means=filter_means,
            filter_covs=filter_covs,
            transitions=transitions,
            process_noises=process_noises,
        )
        new_latent_mean = (smoothed_state_means @ observation_matrix.T).squeeze(-1)
        return (new_latent_mean, smoothed_state_means, smoothed_state_covs), None

    initial_latent = jnp.zeros(times.shape[0])
    initial_state_means = jnp.zeros((times.shape[0], state_space_kernel.state_dim))
    initial_state_covs = jnp.broadcast_to(
        initial_cov, (times.shape[0], state_space_kernel.state_dim, state_space_kernel.state_dim)
    )
    (final_latent, final_state_means, final_state_covs), _ = jax.lax.scan(
        newton_step,
        (initial_latent, initial_state_means, initial_state_covs),
        jnp.arange(num_iterations),
    )
    # Marginal latent variance = H @ P @ H^T per smoothed step.
    latent_variances = jnp.einsum(
        "ij,kjl,il->ik",
        observation_matrix,
        final_state_covs,
        observation_matrix,
    ).reshape(times.shape[0])
    latent_variances = jnp.clip(latent_variances, a_min=_PSEUDO_NOISE_FLOOR)
    # Laplace-approximated log marginal (RW06 eq. 3.32 in state-space form):
    # log Z ≈ log p(y | f̂) - ½ Σ_i log(1 + W_i V_i) where V_i is the
    # prior marginal variance at i (≈ kernel.measurement P_∞ ...). Use a
    # simpler stable approximation: data log-lik at the mode minus
    # half the trace of W·V_post (the Newton residual penalty).
    log_lik_final, _, w_final, _ = log_likelihood_components_fn(final_latent, observations)
    safe_w_final = jnp.maximum(w_final, _PSEUDO_NOISE_FLOOR)
    log_marginal = log_lik_final - 0.5 * jnp.sum(jnp.log1p(safe_w_final * latent_variances))
    return MarkovLaplaceGPState(
        times=times,
        observations=observations,
        smoothed_means=final_latent,
        smoothed_variances=latent_variances,
        smoothed_state_means=final_state_means,
        smoothed_state_covariances=final_state_covs,
        log_marginal_likelihood=log_marginal,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=log_likelihood_components_fn,
    )


def predict_markov_laplace_gp(
    *,
    state: MarkovLaplaceGPState,
    times_test: jax.Array,
) -> PredictiveDistribution:
    r"""Posterior latent moments at ``times_test`` via state-space interpolation.

    For each test time ``t*``, locate the latest training time
    ``t_k ≤ t*`` and propagate the smoothed state at ``t_k`` forward
    by ``Δt = t* − t_k`` using the SDE transition matrix. For test
    times before the first training time, propagate from the
    stationary prior (zero mean, stationary covariance).

    Args:
        state: Fitted :class:`MarkovLaplaceGPState`.
        times_test: ``(m,)`` test time stamps (any order).

    Returns:
        :class:`PredictiveDistribution` whose ``mean`` and
        ``variance`` carry the latent ``f(t*)`` marginal moments at
        each test time. Map through the per-likelihood response
        link (MacKay probit for Bernoulli, ``exp`` for Poisson, ...)
        downstream.
    """
    test_means, test_variances = interpolate_smoothed_state(
        state_space_kernel=state.state_space_kernel,
        times_train=state.times,
        smoothed_state_means=state.smoothed_state_means,
        smoothed_state_covs=state.smoothed_state_covariances,
        times_test=times_test,
    )
    return gaussian_process_predictive(
        test_means,
        test_variances,
        epistemic=test_variances,
        total_uncertainty=test_variances,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_MARKOV_LAPLACE_SOURCE_PACKAGE,
            extra=(
                ("estimator", "markov_laplace_gp"),
                (
                    "paper",
                    "bayesnewton / Sarkka 2013 §9 (Iterated-EKS Laplace on Markov GPs)",
                ),
            ),
        ),
    )


__all__ = [
    "MarkovLaplaceGPState",
    "fit_markov_laplace_gp",
    "predict_markov_laplace_gp",
]
