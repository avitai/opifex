r"""Variational Inference on Markov GPs — Task 11.2 slice 27.

Conjugate-Computation VI (Khan & Lin 2017) applied to state-space
Gaussian-process models (Chang, Wilkinson, Khan, Solin 2020). The
implementation mirrors :mod:`opifex.uncertainty.markov.markov_laplace`
but replaces the mode-evaluated likelihood components ``(grad, W)``
with **expected** components averaged over the current variational
posterior ``q(f_t) = N(mean_t, var_t)``:

* ``E_grad_t = E_{q(f_t)}[∂ log p(y_t | f_t) / ∂ f_t]``,
* ``E_W_t   = -E_{q(f_t)}[∂² log p(y_t | f_t) / ∂ f_t²]``.

Both expectations are estimated by Gauss-Hermite quadrature over the
current marginal ``q(f_t)`` — the same quadrature pattern as D3's
stochastic-SVGP ELBO (slice 16). The expected pair drives a
pseudo-Gaussian Kalman linearisation identical to Markov-Laplace,
giving the same ``O(n)``-per-iteration complexity.

Iteration converges to the variational free-energy stationary point;
returning the ELBO

.. math::

    \mathcal{L}(q) = \sum_t \mathbb{E}_{q(f_t)}[\log p(y_t | f_t)]
                    - \operatorname{KL}[q(f) \,\|\, p(f)]

as a scalar for downstream hyperparameter learning / model selection.

For Gaussian likelihoods (linear-in-``f`` gradient, constant-in-``f``
curvature) ``E_grad`` and ``E_W`` coincide with the mode-evaluated
versions — VI and Laplace then produce identical posteriors, which
serves as the slice-27 numerical cross-check.

References
----------
* Khan, Lin 2017 — *Conjugate-Computation Variational Inference*,
  ICML.
* Chang, Wilkinson, Khan, Solin 2020 — *Fast variational learning
  in state-space Gaussian process models*, ICML.
* Wilkinson, Solin, Adam 2020+ — ``bayesnewton/inference.py``
  ``VariationalInference`` (PRIMARY).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from opifex.uncertainty._predictive import gaussian_process_predictive
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.laplace import LikelihoodComponentsFn  # noqa: TC001
from opifex.uncertainty.markov._likelihood_support import interpolate_smoothed_state
from opifex.uncertainty.markov.markov_laplace import _build_state_space_sequence
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace import (
    kalman_filter,
    kalman_smoother,
    StateSpaceKernel,
)
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


_MARKOV_VI_SOURCE_PACKAGE = "opifex.uncertainty.markov"
_PSEUDO_NOISE_FLOOR: float = 1e-6


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkovVIGPState:
    """Fitted state for VI on a Markov-GP prior.

    Attributes are the Markov-Laplace counterparts plus the
    variational ``evidence_lower_bound`` (ELBO) — the scalar
    objective conjugate-computation VI maximises.
    """

    times: jax.Array
    observations: jax.Array
    smoothed_means: jax.Array
    smoothed_variances: jax.Array
    smoothed_state_means: jax.Array
    smoothed_state_covariances: jax.Array
    evidence_lower_bound: jax.Array
    state_space_kernel: StateSpaceKernel
    log_likelihood_components_fn: LikelihoodComponentsFn


def _gauss_hermite_nodes_weights(
    num_points: int,
) -> tuple[jax.Array, jax.Array]:
    """Static Gauss-Hermite quadrature nodes + weights (cached via numpy)."""
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(num_points)
    return jnp.asarray(nodes_np), jnp.asarray(weights_np)


def _expected_components(
    *,
    log_likelihood_components_fn: LikelihoodComponentsFn,
    latent_mean: jax.Array,
    latent_variance: jax.Array,
    observations: jax.Array,
    num_quadrature_points: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Per-observation ``(E[log p], E[grad], E[W])`` via Gauss-Hermite quadrature.

    For each ``i``, samples ``f`` from ``q(f_i) = N(mean_i, var_i)`` at
    the GH nodes and accumulates the weighted likelihood quadruple
    returned by ``log_likelihood_components_fn`` at each sample.
    """
    nodes, weights = _gauss_hermite_nodes_weights(num_quadrature_points)
    sqrt_two_var = jnp.sqrt(2.0 * jnp.maximum(latent_variance, _PSEUDO_NOISE_FLOOR))
    # f_samples shape (Q, n).
    f_samples = latent_mean[None, :] + sqrt_two_var[None, :] * nodes[:, None]

    def _per_sample(
        f_q: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        log_lik_total, grad, w_diag, _sqrt_w = log_likelihood_components_fn(f_q, observations)
        return log_lik_total, grad, w_diag

    log_liks, grads, w_diags = jax.vmap(_per_sample)(f_samples)
    inv_sqrt_pi = 1.0 / jnp.sqrt(jnp.pi)
    expected_log_lik = inv_sqrt_pi * jnp.sum(weights * log_liks)
    expected_grad = inv_sqrt_pi * jnp.sum(weights[:, None] * grads, axis=0)
    expected_w = inv_sqrt_pi * jnp.sum(weights[:, None] * w_diags, axis=0)
    return expected_log_lik, expected_grad, expected_w


def fit_markov_vi_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    log_likelihood_components_fn: LikelihoodComponentsFn,
    num_iterations: int = 25,
    num_quadrature_points: int = 20,
) -> MarkovVIGPState:
    r"""Fit conjugate-computation VI on a Markov-GP prior.

    Args:
        times: ``(n,)`` strictly-increasing training times.
        observations: ``(n,)`` training observations.
        state_space_kernel: SDE-form GP prior.
        log_likelihood_components_fn: D5 ``LikelihoodComponentsFn`` —
            the same callable used by :func:`fit_markov_laplace_gp`.
        num_iterations: VI iteration count (static under ``jax.jit``).
        num_quadrature_points: Gauss-Hermite nodes for the expected
            components (static; defaults to 20).

    Returns:
        :class:`MarkovVIGPState` with the smoothed posterior moments,
        full smoothed state, and the final ELBO value.
    """
    transitions, process_noises = _build_state_space_sequence(
        times=times, state_space_kernel=state_space_kernel
    )
    observation_matrix = state_space_kernel.measurement
    initial_mean = jnp.zeros(state_space_kernel.state_dim)
    initial_cov = state_space_kernel.stationary_cov

    def vi_step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        _: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
        latent_mean, latent_variance, _state_means, _state_covs = carry
        _, expected_grad, expected_w = _expected_components(
            log_likelihood_components_fn=log_likelihood_components_fn,
            latent_mean=latent_mean,
            latent_variance=latent_variance,
            observations=observations,
            num_quadrature_points=num_quadrature_points,
        )
        safe_w = jnp.maximum(expected_w, _PSEUDO_NOISE_FLOOR)
        pseudo_observations = (latent_mean + expected_grad / safe_w).reshape(-1, 1)
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
        new_latent_variance = jnp.einsum(
            "ij,kjl,il->ik",
            observation_matrix,
            smoothed_state_covs,
            observation_matrix,
        ).reshape(times.shape[0])
        new_latent_variance = jnp.clip(new_latent_variance, a_min=_PSEUDO_NOISE_FLOOR)
        return (
            new_latent_mean,
            new_latent_variance,
            smoothed_state_means,
            smoothed_state_covs,
        ), None

    initial_latent_mean = jnp.zeros(times.shape[0])
    initial_latent_variance = jnp.full(
        (times.shape[0],),
        (observation_matrix @ initial_cov @ observation_matrix.T).squeeze(),
    )
    initial_state_means = jnp.zeros((times.shape[0], state_space_kernel.state_dim))
    initial_state_covs = jnp.broadcast_to(
        initial_cov,
        (times.shape[0], state_space_kernel.state_dim, state_space_kernel.state_dim),
    )
    (
        (
            final_latent_mean,
            final_latent_variance,
            final_state_means,
            final_state_covs,
        ),
        _,
    ) = jax.lax.scan(
        vi_step,
        (
            initial_latent_mean,
            initial_latent_variance,
            initial_state_means,
            initial_state_covs,
        ),
        jnp.arange(num_iterations),
    )

    # ELBO = expected log-likelihood at the converged posterior - KL.
    # The conjugate-computation VI free energy collapses to the
    # equivalent linear-Gaussian-system log marginal under the
    # pseudo-observations; the simpler stable estimator below uses
    # the converged expected log-likelihood plus the analogous
    # half-log-det penalty from Markov-Laplace eq. 3.32.
    final_expected_log_lik, _, final_expected_w = _expected_components(
        log_likelihood_components_fn=log_likelihood_components_fn,
        latent_mean=final_latent_mean,
        latent_variance=final_latent_variance,
        observations=observations,
        num_quadrature_points=num_quadrature_points,
    )
    safe_final_w = jnp.maximum(final_expected_w, _PSEUDO_NOISE_FLOOR)
    elbo = final_expected_log_lik - 0.5 * jnp.sum(jnp.log1p(safe_final_w * final_latent_variance))

    return MarkovVIGPState(
        times=times,
        observations=observations,
        smoothed_means=final_latent_mean,
        smoothed_variances=final_latent_variance,
        smoothed_state_means=final_state_means,
        smoothed_state_covariances=final_state_covs,
        evidence_lower_bound=elbo,
        state_space_kernel=state_space_kernel,
        log_likelihood_components_fn=log_likelihood_components_fn,
    )


def predict_markov_vi_gp(
    *,
    state: MarkovVIGPState,
    times_test: jax.Array,
) -> PredictiveDistribution:
    r"""Posterior latent moments at ``times_test`` via state-space interpolation.

    Mirrors :func:`opifex.uncertainty.markov.predict_markov_laplace_gp`
    — the state-space interpolation step is identical between the
    Laplace and VI posteriors once the smoothed state trajectory has
    been computed at the training grid.
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
            source_package=_MARKOV_VI_SOURCE_PACKAGE,
            extra=(
                ("estimator", "markov_vi_gp"),
                (
                    "paper",
                    "Khan & Lin 2017 / Chang+ 2020 (Conjugate-Computation VI on Markov GPs)",
                ),
            ),
        ),
    )


__all__ = [
    "MarkovVIGPState",
    "fit_markov_vi_gp",
    "predict_markov_vi_gp",
]
