r"""Power Expectation Propagation on Markov GPs — Task 11.2 slice 28.

Implements EP / Power-EP (Minka 2001, 2004) on a Markov-GP prior via the
canonical bayesnewton recipe (Wilkinson, Solin, Adam 2020+,
``bayesnewton/cubature.py::log_density_power_cubature`` +
``bayesnewton/likelihoods.py::moment_match``):

1. Maintain per-site natural parameters ``(site_eta_1_i, site_eta_2_i)``
   representing each observation's Gaussian site approximation.
2. **Pseudo-observation Kalman pass** — convert sites to pseudo-Gaussian
   observations ``(y_pseudo, R) = (site_eta_1 / -2 site_eta_2,
   -1 / (2 site_eta_2))`` and run a Kalman filter + smoother to get the
   per-time posterior moments ``(post_mean_i, post_var_i)``.
3. **Form cavity** in natural-parameter space:
   ``cavity_eta = post_eta - power * site_eta``.
4. **Moment-match the tilted distribution** ``cavity_i * p(y_i | f_i)^power``
   via Gauss-Hermite cubature on the partition function
   ``log Z_i(m, v) = log ∫ p(y_i | f)^power N(f; m, v) df`` (the
   ``log_density_power_cubature`` term). Derivatives ``∂log Z / ∂m`` and
   ``∂² log Z / ∂m²`` are obtained by ``jax.grad`` — this is the
   numerically stable bayesnewton trick that sidesteps direct
   moment-extraction errors.
5. **Bonnet/Price formulas** convert log-Z derivatives to tilted moments:

       tilted_mean = cavity_mean + cavity_var * dlZ_dm,
       tilted_var  = cavity_var + cavity_var^2 * d2lZ_dm2.

6. **Site update** (Power-EP):

       site_eta_2_new = 0.5 * d2lZ_dm2 / ((1 + cavity_var * d2lZ_dm2) * power),
       site_eta_1_new = (dlZ_dm - cavity_mean * d2lZ_dm2)
                        / ((1 + cavity_var * d2lZ_dm2) * power),

   with damping by ``learning_rate``. For ``power = 1`` this is classical
   EP; for ``power = 0.5`` it is power EP — better convergence on heavy-
   tailed/multi-modal likelihoods.

7. **Predict** — interpolate the converged smoothed state forward via
   the same Kalman state-space machinery as Markov-Laplace and Markov-VI.

References
----------
* Minka 2001 — *Expectation Propagation for Approximate Bayesian
  Inference*, UAI.
* Minka 2004 — *Power EP*, Microsoft Research TR-2004-149.
* Wilkinson, Solin, Adam 2020+ — ``bayesnewton/inference.py``
  ``ExpectationPropagation`` (PRIMARY).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from opifex.uncertainty._predictive import gaussian_process_predictive
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.markov._likelihood_support import interpolate_smoothed_state
from opifex.uncertainty.markov.markov_laplace import _build_state_space_sequence
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace import (
    kalman_filter,
    kalman_smoother,
    StateSpaceKernel,
)
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


_MARKOV_PEP_SOURCE_PACKAGE = "opifex.uncertainty.markov"
_PSEUDO_NOISE_FLOOR: float = 1e-6
_NATURAL_PARAM_CLIP: float = -1e-6
"""``site_eta_2`` and ``cavity_eta_2`` are clipped from above by this
negative value so that the implied variance ``-1 / (2 eta_2)`` stays
positive and finite."""
PerObservationLogLikelihoodFn = Callable[[jax.Array, jax.Array], jax.Array]
"""``(f, y) -> per-observation log p(y_i | f_i)`` — same interface as
:func:`opifex.uncertainty.gp.bernoulli_log_likelihood`.
"""

LogZAndDerivativesFn = Callable[
    [jax.Array, jax.Array, jax.Array, float],
    tuple[jax.Array, jax.Array, jax.Array],
]
"""Closed-form override for the EP partition function.

Signature ``(cavity_mean, cavity_variance, observations, power) ->
(log_Z, dlogZ_dm, d2logZ_dm2)`` — each return value has shape
``(n,)``. Used by the Gaussian wrapper to bypass cubature with the
exact analytical formula. When ``None``, the generic Gauss-Hermite +
``jax.grad`` path is used.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkovPEPGPState:
    """Fitted state for Power EP on a Markov-GP prior.

    Carries the smoothed posterior moments, the converged per-site
    natural parameters, and the approximate log marginal likelihood.
    """

    times: jax.Array
    observations: jax.Array
    smoothed_means: jax.Array
    smoothed_variances: jax.Array
    smoothed_state_means: jax.Array
    smoothed_state_covariances: jax.Array
    site_eta_1: jax.Array
    site_eta_2: jax.Array
    log_marginal_likelihood: jax.Array
    state_space_kernel: StateSpaceKernel
    log_likelihood_fn: PerObservationLogLikelihoodFn
    power: float


def _gauss_hermite_nodes_weights(num_points: int) -> tuple[jax.Array, jax.Array]:
    r"""Normalised GH cubature for integrating against ``N(m, v)``.

    Mirrors ``bayesnewton.cubature.gauss_hermite``: nodes scaled by
    ``sqrt(2)`` and weights by ``1/sqrt(pi)`` so that for a function
    ``g(f)`` and proposal ``N(m, v)`` we have

        E_{N(m, v)}[g(f)] = sum_q weight_q * g(m + sqrt(v) * node_q).

    Returns the static (numpy-backed) arrays cached at module load.
    """
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(num_points)
    nodes_np = np.sqrt(2.0) * nodes_np
    weights_np = weights_np / np.sqrt(np.pi)
    return jnp.asarray(nodes_np), jnp.asarray(weights_np)


def _log_partition_per_observation(
    cavity_mean_scalar: jax.Array,
    cavity_variance_scalar: jax.Array,
    observation_scalar: jax.Array,
    *,
    power: float,
    log_likelihood_fn: PerObservationLogLikelihoodFn,
    nodes: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    r"""Scalar log-partition ``log Z_i(m, v) = log ∫ p(y_i|f)^α N(f;m,v) df``.

    Cubature on the cavity ``N(m, v)`` with normalised GH weights
    (``log_density_power_cubature`` in bayesnewton). The likelihood
    accepts batched ``(f, y)`` arrays and returns per-observation
    log-likelihoods, so we broadcast a single ``observation_scalar``
    against the ``Q``-length sigma-point vector.
    """
    sigma_points = (
        cavity_mean_scalar
        + jnp.sqrt(jnp.maximum(cavity_variance_scalar, _PSEUDO_NOISE_FLOOR)) * nodes
    )
    y_broadcast = jnp.broadcast_to(observation_scalar, sigma_points.shape)
    log_lik = log_likelihood_fn(sigma_points, y_broadcast)
    log_weighted = jnp.log(weights) + power * log_lik
    return jax.scipy.special.logsumexp(log_weighted)


def _gauss_hermite_log_partition_and_derivatives(
    cavity_means: jax.Array,
    cavity_variances: jax.Array,
    observations: jax.Array,
    *,
    power: float,
    log_likelihood_fn: PerObservationLogLikelihoodFn,
    num_quadrature_points: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Per-observation ``(log_Z, dlogZ_dm, d2logZ_dm2)`` via cubature.

    Per-observation Hessian and gradient w.r.t. the cavity mean are
    obtained by ``jax.grad`` on the scalar ``_log_partition_per_observation``;
    this is the bayesnewton-canonical numerically-stable form (the
    derivative of ``log Z`` is smooth even when the cubature samples
    don't densely cover the high-likelihood region — see
    ``bayesnewton.likelihoods.Likelihood.moment_match``).
    """
    nodes, weights = _gauss_hermite_nodes_weights(num_quadrature_points)

    def per_obs_log_partition(m: jax.Array, v: jax.Array, y: jax.Array) -> jax.Array:
        """Return the tilted log-partition for one observation's cavity."""
        return _log_partition_per_observation(
            m,
            v,
            y,
            power=power,
            log_likelihood_fn=log_likelihood_fn,
            nodes=nodes,
            weights=weights,
        )

    log_Z_vec = jax.vmap(per_obs_log_partition)(cavity_means, cavity_variances, observations)
    dlogZ_dm_vec = jax.vmap(jax.grad(per_obs_log_partition, argnums=0))(
        cavity_means, cavity_variances, observations
    )
    d2logZ_dm2_vec = jax.vmap(jax.grad(jax.grad(per_obs_log_partition, argnums=0), argnums=0))(
        cavity_means, cavity_variances, observations
    )
    return log_Z_vec, dlogZ_dm_vec, d2logZ_dm2_vec


def _site_update_from_log_partition_derivatives(
    *,
    cavity_means: jax.Array,
    cavity_variances: jax.Array,
    dlogZ_dm: jax.Array,
    d2logZ_dm2: jax.Array,
    power: float,
) -> tuple[jax.Array, jax.Array]:
    r"""Stable closed-form site update from log-Z derivatives.

    Derivation (Bonnet/Price + EP power):

        tilted_var  = cavity_var + cavity_var^2 * d2lZ,
        tilted_mean = cavity_mean + cavity_var * dlZ,
        site_prec_new = (1/tilted_var - 1/cavity_var) / power
                      = -d2lZ / (1 + cavity_var * d2lZ) / power.

    For Gaussian likelihood at ``power = 1`` this gives
    ``site_eta_2_new = -0.5/sigma^2`` and ``site_eta_1_new = y/sigma^2``
    exactly — verified by
    ``test_markov_pep_gaussian_likelihood_matches_markov_laplace_gaussian``.

    The denominator ``1 + cavity_var * d2lZ`` is bounded in ``(0, 1]``
    for any valid tilted distribution (tilted_var <= cavity_var), so
    the formula is numerically safe.
    """
    denominator = 1.0 + cavity_variances * d2logZ_dm2
    safe_denominator = jnp.where(
        jnp.abs(denominator) < _PSEUDO_NOISE_FLOOR,
        jnp.sign(denominator) * _PSEUDO_NOISE_FLOOR + (denominator == 0.0) * _PSEUDO_NOISE_FLOOR,
        denominator,
    )
    site_eta_2_new = 0.5 * d2logZ_dm2 / (safe_denominator * power)
    site_eta_1_new = (dlogZ_dm - cavity_means * d2logZ_dm2) / (safe_denominator * power)
    site_eta_2_new = jnp.minimum(site_eta_2_new, _NATURAL_PARAM_CLIP)
    return site_eta_1_new, site_eta_2_new


def _kalman_with_sites(
    *,
    site_eta_1: jax.Array,
    site_eta_2: jax.Array,
    transitions: jax.Array,
    process_noises: jax.Array,
    observation_matrix: jax.Array,
    initial_mean: jax.Array,
    initial_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Kalman filter + smoother treating sites as pseudo-Gaussian observations.

    Returns ``(smoothed_state_means, smoothed_state_covs,
    smoothed_obs_means, smoothed_obs_variances)`` — the second pair
    are the per-observation projected moments
    ``H @ smoothed_state @ H.T`` used in EP cavity computation.
    """
    n = site_eta_1.shape[0]
    safe_eta_2 = jnp.minimum(site_eta_2, _NATURAL_PARAM_CLIP)
    site_variance = -1.0 / (2.0 * safe_eta_2)
    site_mean = site_eta_1 * site_variance
    filter_means, filter_covs = kalman_filter(
        transitions=transitions,
        process_noises=process_noises,
        observations=site_mean.reshape(-1, 1),
        observation_matrix=observation_matrix,
        observation_covs=site_variance.reshape(-1, 1, 1),
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )
    smoothed_state_means, smoothed_state_covs = kalman_smoother(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=transitions,
        process_noises=process_noises,
    )
    smoothed_obs_means = (smoothed_state_means @ observation_matrix.T).squeeze(-1)
    smoothed_obs_variances = jnp.einsum(
        "ij,kjl,il->ik",
        observation_matrix,
        smoothed_state_covs,
        observation_matrix,
    ).reshape(n)
    smoothed_obs_variances = jnp.clip(smoothed_obs_variances, a_min=_PSEUDO_NOISE_FLOOR)
    return (
        smoothed_state_means,
        smoothed_state_covs,
        smoothed_obs_means,
        smoothed_obs_variances,
    )


def fit_markov_pep_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    log_likelihood_fn: PerObservationLogLikelihoodFn,
    power: float = 0.5,
    num_iterations: int = 25,
    learning_rate: float = 0.5,
    num_quadrature_points: int = 20,
    log_partition_fn: LogZAndDerivativesFn | None = None,
) -> MarkovPEPGPState:
    r"""Fit Power EP on a Markov-GP prior.

    Args:
        times: ``(n,)`` strictly-increasing training times.
        observations: ``(n,)`` training observations.
        state_space_kernel: SDE-form GP prior.
        log_likelihood_fn: ``(f, y) -> log p(y|f)`` per-observation.
        power: EP power ``α ∈ (0, 1]``. Defaults to ``0.5`` (Power EP).
        num_iterations: Number of full Kalman-and-site sweeps.
        learning_rate: Damping for the site update (1.0 = no damping).
        num_quadrature_points: GH nodes for cubature.
        log_partition_fn: Optional closed-form ``log Z`` and its first two
            derivatives w.r.t. cavity mean. When provided, cubature is
            bypassed — used by the Gaussian wrapper, which has the
            exact analytical form.

    Returns:
        :class:`MarkovPEPGPState` carrying smoothed posterior moments,
        converged sites, and approximate log marginal likelihood.
    """
    transitions, process_noises = _build_state_space_sequence(
        times=times, state_space_kernel=state_space_kernel
    )
    observation_matrix = state_space_kernel.measurement
    initial_mean = jnp.zeros(state_space_kernel.state_dim)
    initial_cov = state_space_kernel.stationary_cov
    n = times.shape[0]

    def compute_log_partition_pack(
        cavity_means: jax.Array, cavity_variances: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the log-partition and its first two derivatives over all cavities."""
        if log_partition_fn is not None:
            return log_partition_fn(cavity_means, cavity_variances, observations, power)
        return _gauss_hermite_log_partition_and_derivatives(
            cavity_means,
            cavity_variances,
            observations,
            power=power,
            log_likelihood_fn=log_likelihood_fn,
            num_quadrature_points=num_quadrature_points,
        )

    def ep_step(
        carry: tuple[jax.Array, jax.Array],
        _: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        """Run one power-EP sweep, updating the site parameters."""
        site_eta_1, site_eta_2 = carry
        _, _, post_means, post_variances = _kalman_with_sites(
            site_eta_1=site_eta_1,
            site_eta_2=site_eta_2,
            transitions=transitions,
            process_noises=process_noises,
            observation_matrix=observation_matrix,
            initial_mean=initial_mean,
            initial_cov=initial_cov,
        )
        post_eta_2 = -0.5 / post_variances
        post_eta_1 = post_means / post_variances
        cavity_eta_2 = post_eta_2 - power * site_eta_2
        cavity_eta_1 = post_eta_1 - power * site_eta_1
        safe_cavity_eta_2 = jnp.minimum(cavity_eta_2, _NATURAL_PARAM_CLIP)
        cavity_variances = -0.5 / safe_cavity_eta_2
        cavity_means = cavity_eta_1 * cavity_variances
        _, dlogZ_dm, d2logZ_dm2 = compute_log_partition_pack(cavity_means, cavity_variances)
        new_site_eta_1, new_site_eta_2 = _site_update_from_log_partition_derivatives(
            cavity_means=cavity_means,
            cavity_variances=cavity_variances,
            dlogZ_dm=dlogZ_dm,
            d2logZ_dm2=d2logZ_dm2,
            power=power,
        )
        site_eta_1_updated = (1.0 - learning_rate) * site_eta_1 + learning_rate * new_site_eta_1
        site_eta_2_updated = (1.0 - learning_rate) * site_eta_2 + learning_rate * new_site_eta_2
        site_eta_2_updated = jnp.minimum(site_eta_2_updated, _NATURAL_PARAM_CLIP)
        return (site_eta_1_updated, site_eta_2_updated), None

    initial_eta_1 = jnp.zeros(n)
    initial_eta_2 = jnp.full((n,), _NATURAL_PARAM_CLIP)
    (final_eta_1, final_eta_2), _ = jax.lax.scan(
        ep_step, (initial_eta_1, initial_eta_2), jnp.arange(num_iterations)
    )

    (
        smoothed_state_means,
        smoothed_state_covs,
        smoothed_means,
        smoothed_variances,
    ) = _kalman_with_sites(
        site_eta_1=final_eta_1,
        site_eta_2=final_eta_2,
        transitions=transitions,
        process_noises=process_noises,
        observation_matrix=observation_matrix,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )

    post_eta_2 = -0.5 / smoothed_variances
    post_eta_1 = smoothed_means / smoothed_variances
    cavity_eta_2 = post_eta_2 - power * final_eta_2
    cavity_eta_1 = post_eta_1 - power * final_eta_1
    safe_cavity_eta_2 = jnp.minimum(cavity_eta_2, _NATURAL_PARAM_CLIP)
    cavity_variances_final = -0.5 / safe_cavity_eta_2
    cavity_means_final = cavity_eta_1 * cavity_variances_final
    log_Z_final, _, _ = compute_log_partition_pack(cavity_means_final, cavity_variances_final)
    log_marginal = jnp.sum(log_Z_final) / power

    return MarkovPEPGPState(
        times=times,
        observations=observations,
        smoothed_means=smoothed_means,
        smoothed_variances=smoothed_variances,
        smoothed_state_means=smoothed_state_means,
        smoothed_state_covariances=smoothed_state_covs,
        site_eta_1=final_eta_1,
        site_eta_2=final_eta_2,
        log_marginal_likelihood=log_marginal,
        state_space_kernel=state_space_kernel,
        log_likelihood_fn=log_likelihood_fn,
        power=power,
    )


def predict_markov_pep_gp(
    *,
    state: MarkovPEPGPState,
    times_test: jax.Array,
) -> PredictiveDistribution:
    r"""Posterior latent moments at ``times_test`` via state-space interpolation.

    Identical predict path as :func:`predict_markov_laplace_gp` and
    :func:`predict_markov_vi_gp` — the inference algorithm differs but
    the smoothed-state interpolation is shared.
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
            source_package=_MARKOV_PEP_SOURCE_PACKAGE,
            extra=(
                ("estimator", "markov_pep_gp"),
                (
                    "paper",
                    "Minka 2001/2004 + Wilkinson, Solin, Adam 2020+ (PEP on Markov GPs)",
                ),
            ),
        ),
    )


__all__ = [
    "LogZAndDerivativesFn",
    "MarkovPEPGPState",
    "PerObservationLogLikelihoodFn",
    "fit_markov_pep_gp",
    "predict_markov_pep_gp",
]
