r"""Shared support helpers for the Markov-GP inference paths.

Two pieces of logic are byte-identical across the four inference paths
(Laplace / VI / PEP / PL) and are centralised here (Rule 1 — DRY):

* :func:`latent_variance` — unwraps the latent variance that every
  ``predict_markov_*_gp`` always sets, raising :class:`RuntimeError` (not an
  ``assert``, which ``python -O`` strips) when it is ``None``.
* :func:`interpolate_smoothed_state` — the state-space interpolation that
  propagates a smoothed state trajectory from the training grid to held-out
  test times. The four ``predict_markov_*_gp`` paths share this exact
  ``predict_one`` body; only the metadata stamped on the returned
  :class:`PredictiveDistribution` differs.

Three evidence building blocks are shared by the VI, Laplace and power-EP evidence values:

* :func:`gaussian_expected_log_density` — ``E_{N(f | m, v)} log N(y | f, R)``.
* :func:`pseudo_model_log_normaliser` — the Kalman-filter log normaliser of the
  pseudo-observation model, ``log Z = log ∫ p(f) Π N(ỹ_n | f_n, R_n) df``.
* :func:`power_ep_constant` — the power-EP normalising constant of a Gaussian site.

They follow Chang, Wilkinson, Khan & Solin (2020) and Wilkinson, Sarkka & Solin (JMLR 2023).

References:
----------
* Sarkka 2013 — *Bayesian Filtering and Smoothing*, CUP §9 (state-space GP
  interpolation via the SDE transition matrix).
* Chang, Wilkinson, Khan, Solin 2020 — *Fast Variational Learning in State-Space Gaussian
  Process Models*, MLSP, arXiv:2007.04731.
* Wilkinson, Sarkka, Solin 2023 — *Bayes-Newton Methods for Approximate Bayesian Inference
  with PSD Guarantees*, JMLR 24(83), arXiv:2111.01721.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import (
    kalman_log_likelihood,
    StateSpaceKernel,
)
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


# Variance floor shared by every Markov predict path: clip marginal latent
# variances away from zero to keep the downstream response maps well-defined.
_PSEUDO_NOISE_FLOOR: float = 1e-6
_LOG_2PI: float = math.log(2.0 * math.pi)


def gaussian_expected_log_density(
    observations: jax.Array,
    means: jax.Array,
    variances: jax.Array | float,
    noise_variances: jax.Array | float,
) -> jax.Array:
    r"""Return ``E_{N(f | m, v)} log N(y | f, R)`` element-wise.

    Expected Gaussian log density (Chang et al. 2020). With
    ``variances = 0`` this is the Gaussian log density ``log N(y | m, R)``.

    Args:
        observations: Observations ``y``.
        means: Means ``m`` of the distribution over ``f``.
        variances: Variances ``v`` of the distribution over ``f``.
        noise_variances: Observation noise variances ``R``.

    Returns:
        The expected log densities, broadcast from the inputs.
    """
    return (
        -0.5 * _LOG_2PI
        - 0.5 * jnp.log(noise_variances)
        - 0.5 * ((observations - means) ** 2 + variances) / noise_variances
    )


def pseudo_model_log_normaliser(
    *,
    transitions: jax.Array,
    process_noises: jax.Array,
    observation_matrix: jax.Array,
    initial_mean: jax.Array,
    initial_cov: jax.Array,
    site_observations: jax.Array,
    site_variances: jax.Array,
) -> jax.Array:
    r"""Return ``log Z = log ∫ p(f) Π_n N(ỹ_n | f_n, R_n) df`` for scalar Gaussian sites.

    Runs the Kalman filter on the pseudo-observations and returns its log likelihood
    (Wilkinson, Sarkka & Solin 2023).

    Args:
        transitions: ``(n, d, d)`` per-step transitions.
        process_noises: ``(n, d, d)`` per-step process noises.
        observation_matrix: ``(1, d)`` measurement matrix.
        initial_mean: ``(d,)`` prior state mean.
        initial_cov: ``(d, d)`` prior state covariance.
        site_observations: ``(n,)`` site means ``ỹ``.
        site_variances: ``(n,)`` site variances ``R``.

    Returns:
        The scalar log normaliser.
    """
    return kalman_log_likelihood(
        transitions=transitions,
        process_noises=process_noises,
        observations=site_observations.reshape(-1, 1),
        observation_matrix=observation_matrix,
        observation_covs=site_variances.reshape(-1, 1, 1),
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )


def power_ep_constant(variances: jax.Array | float, power: float) -> jax.Array:
    r"""Return the power-EP constant ``½ ((1 - α) log 2π - log α) + ½ (1 - α) log R``.

    For scalar sites (Wilkinson, Sarkka & Solin 2023). It is
    the log normaliser of ``N(y | f, R)^α`` relative to ``N(y | f, R / α)``.

    Args:
        variances: Site or likelihood variances ``R``.
        power: EP power ``α``.

    Returns:
        The constants, broadcast from ``variances``.
    """
    return 0.5 * ((1.0 - power) * _LOG_2PI - jnp.log(power)) + 0.5 * (1.0 - power) * jnp.log(
        variances
    )


def latent_variance(predictive: PredictiveDistribution) -> jax.Array:
    """Return the latent variance set by every ``predict_markov_*_gp``.

    Args:
        predictive: Latent predictive returned by a Markov predict path.

    Returns:
        The marginal latent variance array.

    Raises:
        RuntimeError: When ``predictive.variance`` is ``None`` — the predict
            paths always populate it, so a ``None`` signals a contract
            violation. Uses an explicit raise (not ``assert``) so the guard
            survives ``python -O``.
    """
    if predictive.variance is None:
        raise RuntimeError("predict_markov_*_gp returned a PredictiveDistribution with no variance")
    return predictive.variance


def interpolate_smoothed_state(
    *,
    state_space_kernel: StateSpaceKernel,
    times_train: jax.Array,
    smoothed_state_means: jax.Array,
    smoothed_state_covs: jax.Array,
    times_test: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Interpolate the smoothed state to ``times_test`` and return latent moments.

    For each test time ``t*``, locate the latest training time ``t_k <= t*`` and
    propagate the smoothed state at ``t_k`` forward by ``dt = t* - t_k`` using
    the SDE transition matrix. For test times before the first training time,
    propagate from the stationary prior (zero mean, stationary covariance).

    Shared verbatim by ``predict_markov_{laplace,vi,pep,pl}_gp`` — the inference
    algorithm differs but this smoothed-state interpolation is identical.

    Args:
        state_space_kernel: The fitted :class:`StateSpaceKernel` (provides the
            measurement operator, stationary covariance, and transition map).
        times_train: ``(n,)`` strictly-increasing training time stamps.
        smoothed_state_means: ``(n, state_dim)`` smoothed full-state trajectory.
        smoothed_state_covs: ``(n, state_dim, state_dim)`` smoothed full-state
            covariances.
        times_test: ``(m,)`` test time stamps (any order).

    Returns:
        ``(test_means, test_variances)`` — the latent ``f(t*)`` marginal mean
        and variance at each test time, variances clipped at
        :data:`_PSEUDO_NOISE_FLOOR`.
    """
    observation_matrix = state_space_kernel.measurement
    stationary_cov = state_space_kernel.stationary_cov
    state_dim = state_space_kernel.state_dim

    # For each test time t*, find the closest preceding training index via
    # right-sided searchsorted; use index 0 (with mean=0, cov=P_inf) when t*
    # precedes the first training point.
    bucket_indices = jnp.searchsorted(times_train, times_test, side="right") - 1

    is_before_first = bucket_indices < 0
    clipped_indices = jnp.maximum(bucket_indices, 0)
    anchor_times = jnp.where(is_before_first, times_test, times_train[clipped_indices])
    # One call over every test gap lets the kernel discretise them as a sequence.
    transitions, process_noises = state_space_kernel.discretize_steps(times_test - anchor_times)

    def predict_one(
        is_first: jax.Array,
        anchor_index: jax.Array,
        transition_matrix: jax.Array,
        process_noise: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return the predictive mean and variance at one test time via SDE interpolation."""
        anchor_mean = jnp.where(is_first, jnp.zeros(state_dim), smoothed_state_means[anchor_index])
        anchor_cov = jnp.where(is_first, stationary_cov, smoothed_state_covs[anchor_index])
        predicted_state_mean = transition_matrix @ anchor_mean
        predicted_state_cov = transition_matrix @ anchor_cov @ transition_matrix.T + process_noise
        latent_mean = (observation_matrix @ predicted_state_mean).squeeze(-1)
        latent_var = (observation_matrix @ predicted_state_cov @ observation_matrix.T).squeeze()
        return latent_mean, latent_var

    test_means, test_variances = jax.vmap(predict_one)(
        is_before_first, clipped_indices, transitions, process_noises
    )
    test_variances = jnp.clip(test_variances, min=_PSEUDO_NOISE_FLOOR)
    return test_means, test_variances


__all__ = [
    "gaussian_expected_log_density",
    "interpolate_smoothed_state",
    "latent_variance",
    "power_ep_constant",
    "pseudo_model_log_normaliser",
]
