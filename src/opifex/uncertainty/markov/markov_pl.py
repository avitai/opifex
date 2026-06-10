r"""Posterior Linearisation on Markov GPs — Task 11.2 slice 30.

Implements iterated posterior linearisation (Garcia-Fernandez, Tronarp,
Sarkka 2018) on a Markov-GP prior via the canonical bayesnewton recipe
(``bayesnewton/inference.py::PosteriorLinearisation`` +
``bayesnewton/cubature.py::statistical_linear_regression_cubature``):

1. **Statistical Linear Regression (SLR)** linearises the likelihood
   ``p(y | f)`` against the current posterior moments
   ``q(f) = N(post_mean, post_var)`` via Gauss-Hermite cubature on
   the per-likelihood **conditional moments** ``(E[y|f], Var[y|f])``::

       mu_i    = E_{q(f_i)}[E[y_i | f_i]],
       S_i     = E_q[Var[y|f]] + Var_q[E[y|f]],
       C_i     = Cov_q(f, E[y|f]),
       omega_i = S_i - C_i^T cov^-1 C_i,
       A_i     = d mu_i / d post_mean_i  (jax.grad).

2. **Pseudo-Gaussian linearisation** ``y ~ N(A f + b, omega)`` with
   intercept ``b = mu - A * post_mean``, rearranged into Kalman
   pseudo-observation form

       y_pseudo = (y - b) / A   (when A != 0),
       R_pseudo = omega / A^2.

3. **Kalman filter + smoother** on the linearised model gives new
   smoothed moments; iterate by re-linearising around the updated
   posterior until convergence (the Iterated Extended Kalman
   Smoother of Sarkka 2013 §6, generalised to non-Taylor SLR).

4. **Predict** — interpolate the converged smoothed state forward
   via the same state-space machinery as Markov-Laplace, Markov-VI,
   and Markov-PEP.

For Gaussian likelihood the linearisation is exact:
``A = 1, b = 0, omega = sigma_squared``, so PL reduces to the conjugate
Kalman path in one iteration — verified by the slice 30 cross-check.

References
----------
* Garcia-Fernandez, Tronarp, Sarkka 2018 — *Gaussian process
  classification using posterior linearisation*, IEEE SPL.
* Sarkka 2013 — *Bayesian Filtering and Smoothing*, CUP, §6.
* Wilkinson, Solin, Adam 2020+ — ``bayesnewton/inference.py``
  ``PosteriorLinearisation`` (PRIMARY).
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


_MARKOV_PL_SOURCE_PACKAGE = "opifex.uncertainty.markov"
_PSEUDO_NOISE_FLOOR: float = 1e-6
_SLR_SLOPE_FLOOR: float = 1e-6
"""Floor on ``|A|`` to keep the pseudo-observation ``(y - b) / A`` finite
when the SLR slope vanishes near a saturated likelihood."""


ConditionalMomentsFn = Callable[[jax.Array], tuple[jax.Array, jax.Array]]
"""``f -> (E[y|f], Var[y|f])`` per-observation conditional moments.

Both ``f`` and the returned arrays share the same shape — typically
``()`` for scalar evaluation under :func:`jax.grad`, or ``(Q,)`` for
batched cubature evaluation across ``Q`` sigma points.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkovPLGPState:
    """Fitted state for Posterior Linearisation on a Markov-GP prior."""

    times: jax.Array
    observations: jax.Array
    smoothed_means: jax.Array
    smoothed_variances: jax.Array
    smoothed_state_means: jax.Array
    smoothed_state_covariances: jax.Array
    state_space_kernel: StateSpaceKernel
    conditional_moments_fn: ConditionalMomentsFn


def _gauss_hermite_nodes_weights(num_points: int) -> tuple[jax.Array, jax.Array]:
    r"""Normalised GH cubature (matches ``bayesnewton.cubature.gauss_hermite``).

    Returns ``(nodes, weights)`` with the convention

        E_{N(m, v)}[g(f)] = sum_q weight_q * g(m + sqrt(v) * node_q).
    """
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(num_points)
    nodes_np = np.sqrt(2.0) * nodes_np
    weights_np = weights_np / np.sqrt(np.pi)
    return jnp.asarray(nodes_np), jnp.asarray(weights_np)


def _expected_conditional_mean_scalar(
    mean_f_scalar: jax.Array,
    variance_f_scalar: jax.Array,
    *,
    conditional_moments_fn: ConditionalMomentsFn,
    nodes: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    r"""Scalar ``mu(mean_f, var_f) = E_{N(mean_f, var_f)}[E[y|f]]``.

    Cubature with the canonical normalised GH weights. Returns a
    scalar suitable for :func:`jax.grad` to produce the SLR slope
    ``A = d mu / d mean_f``.
    """
    sigma_points = (
        mean_f_scalar + jnp.sqrt(jnp.maximum(variance_f_scalar, _PSEUDO_NOISE_FLOOR)) * nodes
    )
    conditional_mean, _ = conditional_moments_fn(sigma_points)
    return jnp.sum(weights * conditional_mean)


def _slr_linearisation(
    *,
    mean_f: jax.Array,
    variance_f: jax.Array,
    conditional_moments_fn: ConditionalMomentsFn,
    num_quadrature_points: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Per-observation SLR linearisation ``(mu, omega, A)``.

    Returns three length-``n`` arrays where each entry is the scalar
    SLR design for that observation (response mean, residual
    variance, design slope).
    """
    nodes, weights = _gauss_hermite_nodes_weights(num_quadrature_points)

    def per_obs_mu(m: jax.Array, v: jax.Array) -> jax.Array:
        """Return the expected conditional mean under one observation's marginal."""
        return _expected_conditional_mean_scalar(
            m,
            v,
            conditional_moments_fn=conditional_moments_fn,
            nodes=nodes,
            weights=weights,
        )

    def per_obs_response_and_cross(m: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the response variance and input-output cross-covariance for one observation."""
        sigma_points = m + jnp.sqrt(jnp.maximum(v, _PSEUDO_NOISE_FLOOR)) * nodes
        cond_mean, cond_var = conditional_moments_fn(sigma_points)
        mu = jnp.sum(weights * cond_mean)
        var_q_cond_mean = jnp.sum(weights * (cond_mean - mu) ** 2)
        e_q_cond_var = jnp.sum(weights * cond_var)
        big_s = var_q_cond_mean + e_q_cond_var
        c_term = jnp.sum(weights * (sigma_points - m) * (cond_mean - mu))
        return big_s, c_term

    mu_vec = jax.vmap(per_obs_mu)(mean_f, variance_f)
    a_vec = jax.vmap(jax.grad(per_obs_mu, argnums=0))(mean_f, variance_f)
    s_vec, c_vec = jax.vmap(per_obs_response_and_cross)(mean_f, variance_f)
    safe_variance = jnp.maximum(variance_f, _PSEUDO_NOISE_FLOOR)
    omega_vec = s_vec - c_vec * c_vec / safe_variance
    omega_vec = jnp.maximum(omega_vec, _PSEUDO_NOISE_FLOOR)
    return mu_vec, omega_vec, a_vec


def _kalman_with_pseudo_observations(
    *,
    pseudo_observations: jax.Array,
    pseudo_variances: jax.Array,
    transitions: jax.Array,
    process_noises: jax.Array,
    observation_matrix: jax.Array,
    initial_mean: jax.Array,
    initial_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Kalman filter + smoother on a pre-linearised pseudo-Gaussian model.

    Returns ``(smoothed_state_means, smoothed_state_covs,
    smoothed_obs_means, smoothed_obs_variances)``.
    """
    n = pseudo_observations.shape[0]
    filter_means, filter_covs = kalman_filter(
        transitions=transitions,
        process_noises=process_noises,
        observations=pseudo_observations.reshape(-1, 1),
        observation_matrix=observation_matrix,
        observation_covs=pseudo_variances.reshape(-1, 1, 1),
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


def fit_markov_pl_gp(
    *,
    times: jax.Array,
    observations: jax.Array,
    state_space_kernel: StateSpaceKernel,
    conditional_moments_fn: ConditionalMomentsFn,
    num_iterations: int = 20,
    num_quadrature_points: int = 20,
) -> MarkovPLGPState:
    r"""Fit Posterior Linearisation on a Markov-GP prior.

    Args:
        times: ``(n,)`` strictly-increasing training times.
        observations: ``(n,)`` training observations.
        state_space_kernel: SDE-form GP prior.
        conditional_moments_fn: Per-likelihood ``f -> (E[y|f], Var[y|f])``.
        num_iterations: Number of full re-linearise-and-Kalman sweeps.
        num_quadrature_points: GH nodes for the cubature in SLR.

    Returns:
        :class:`MarkovPLGPState` with the converged smoothed posterior
        moments and the full smoothed state trajectory.
    """
    transitions, process_noises = _build_state_space_sequence(
        times=times, state_space_kernel=state_space_kernel
    )
    observation_matrix = state_space_kernel.measurement
    initial_mean = jnp.zeros(state_space_kernel.state_dim)
    initial_cov = state_space_kernel.stationary_cov

    initial_posterior_mean = jnp.zeros_like(observations)
    initial_posterior_variance = jnp.full_like(
        observations,
        state_space_kernel.stationary_cov[0, 0],
    )

    def pl_step(
        carry: tuple[jax.Array, jax.Array],
        _: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        """Run one posterior-linearisation sweep, refreshing the Gaussian posterior."""
        post_mean, post_var = carry
        mu, omega, slope = _slr_linearisation(
            mean_f=post_mean,
            variance_f=post_var,
            conditional_moments_fn=conditional_moments_fn,
            num_quadrature_points=num_quadrature_points,
        )
        safe_slope = jnp.where(
            jnp.abs(slope) < _SLR_SLOPE_FLOOR,
            jnp.sign(slope) * _SLR_SLOPE_FLOOR + (slope == 0.0) * _SLR_SLOPE_FLOOR,
            slope,
        )
        intercept = mu - safe_slope * post_mean
        pseudo_obs = (observations - intercept) / safe_slope
        pseudo_var = omega / (safe_slope * safe_slope)
        _, _, new_post_mean, new_post_var = _kalman_with_pseudo_observations(
            pseudo_observations=pseudo_obs,
            pseudo_variances=pseudo_var,
            transitions=transitions,
            process_noises=process_noises,
            observation_matrix=observation_matrix,
            initial_mean=initial_mean,
            initial_cov=initial_cov,
        )
        return (new_post_mean, new_post_var), None

    (final_post_mean, final_post_var), _ = jax.lax.scan(
        pl_step,
        (initial_posterior_mean, initial_posterior_variance),
        jnp.arange(num_iterations),
    )

    mu, omega, slope = _slr_linearisation(
        mean_f=final_post_mean,
        variance_f=final_post_var,
        conditional_moments_fn=conditional_moments_fn,
        num_quadrature_points=num_quadrature_points,
    )
    safe_slope = jnp.where(
        jnp.abs(slope) < _SLR_SLOPE_FLOOR,
        jnp.sign(slope) * _SLR_SLOPE_FLOOR + (slope == 0.0) * _SLR_SLOPE_FLOOR,
        slope,
    )
    intercept = mu - safe_slope * final_post_mean
    pseudo_obs = (observations - intercept) / safe_slope
    pseudo_var = omega / (safe_slope * safe_slope)
    (
        smoothed_state_means,
        smoothed_state_covs,
        smoothed_means,
        smoothed_variances,
    ) = _kalman_with_pseudo_observations(
        pseudo_observations=pseudo_obs,
        pseudo_variances=pseudo_var,
        transitions=transitions,
        process_noises=process_noises,
        observation_matrix=observation_matrix,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )

    return MarkovPLGPState(
        times=times,
        observations=observations,
        smoothed_means=smoothed_means,
        smoothed_variances=smoothed_variances,
        smoothed_state_means=smoothed_state_means,
        smoothed_state_covariances=smoothed_state_covs,
        state_space_kernel=state_space_kernel,
        conditional_moments_fn=conditional_moments_fn,
    )


def predict_markov_pl_gp(
    *,
    state: MarkovPLGPState,
    times_test: jax.Array,
) -> PredictiveDistribution:
    r"""Posterior latent moments at ``times_test`` via state-space interpolation.

    Identical predict path as the Laplace / VI / PEP routes — only the
    inference algorithm differs.
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
            source_package=_MARKOV_PL_SOURCE_PACKAGE,
            extra=(
                ("estimator", "markov_pl_gp"),
                (
                    "paper",
                    "Garcia-Fernandez, Tronarp, Sarkka 2018 + Wilkinson, "
                    "Solin, Adam 2020+ (Posterior Linearisation on Markov GPs)",
                ),
            ),
        ),
    )


__all__ = [
    "ConditionalMomentsFn",
    "MarkovPLGPState",
    "fit_markov_pl_gp",
    "predict_markov_pl_gp",
]
