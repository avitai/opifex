r"""Spatio-temporal variational Gaussian process (ST-VGP) — Feature F17.

A *separable space-time GP* whose temporal factor is represented as a
Markovian (linear-time-invariant SDE / state-space) GP — solved in
``O(T)`` by Kalman filtering and Rauch-Tung-Striebel smoothing — and
whose spatial factor is a standard GP over the spatial inputs. The
separable prior covariance is

.. math::

    k\bigl((t, R), (t', R')\bigr) = k_t(t, t')\, k_s(R, R'),

so the latent process at a fixed time is a vector ``u(t) \in
\mathbb{R}^{M}`` over ``M`` spatial points, and the whole field evolves
as the Kronecker-lifted temporal SDE

.. math::

    F = I_M \otimes F_t, \quad
    A(\Delta t) = I_M \otimes A_t(\Delta t), \quad
    H = I_M \otimes H_t, \quad
    P_\infty = K_{zz} \otimes P_\infty^{t},

where ``K_{zz} = k_s(R, R)`` is the spatial Gram matrix at the ``M``
spatial locations (Solin & Sarkka 2014; Hamelijnck et al. 2021 §3). The
discrete process noise inherits the same Kronecker structure,
``Q = K_{zz} \otimes Q_t``.

Scope (bounded, conjugate Gaussian likelihood)
----------------------------------------------
This module implements the **conjugate** ST-VGP on *gridded* data: the
same ``M`` spatial locations are observed at every one of the ``T``
times (observations shaped ``(T, M)``). Spatial inducing points are
taken to coincide with the observed spatial locations, which makes the
spatial conditional exact (``B = I_M \otimes H_t``, conditional
covariance ``C = 0``) and the variational posterior collapse to the
closed-form conjugate posterior. The latent state at each time is then
the field at the ``M`` points, and inference is exactly the Kronecker
state-space Kalman filter/smoother. Non-gridded data, sparse spatial
inducing points (``M_z < M``), and non-Gaussian likelihoods (the full
variational Newton / power-EP inference of Hamelijnck et al. 2021) are
out of scope for this slice and are deferred to a follow-up.

Reference (binding)
-------------------
* Hamelijnck, O., Wilkinson, W. J., Loka, N. A. B., Solin, A.,
  Damoulas, T. 2021 — *Spatio-Temporal Variational Gaussian Processes*,
  NeurIPS, arXiv:2111.01732 (PRIMARY — separable space-time GP with a
  state-space temporal prior).
* Solin, A., Sarkka, S. 2014 — *Explicit link between periodic
  covariance functions and state space models* / Kronecker space-time
  state-space construction.

Sibling reference implementations consulted (READ-ONLY — never imported):
* ``../bayesnewton/bayesnewton/kernels.py:SpatioTemporalKernel`` (line
  385) — ``K`` (separable product, line 462), ``stationary_covariance``
  (``Kzz ⊗ Pinf_t``), ``state_transition`` (``I_M ⊗ A_t``),
  ``measurement_model`` (``I_M ⊗ H_t``), ``spatial_conditional`` (line
  486).
* ``../bayesnewton/bayesnewton/basemodels.py:MarkovGaussianProcess``
  (line 626) — ``predict`` (line 766) and
  ``conditional_posterior_to_data`` (line 745).

References
----------
* Sarkka, S. 2013 — *Bayesian Filtering and Smoothing*, CUP.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.exact import rbf_kernel
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.statespace.kalman import (
    kalman_filter,
    kalman_log_likelihood,
    kalman_smoother,
)
from opifex.uncertainty.statespace.kernels import StateSpaceKernel  # noqa: TC001 — eager
from opifex.uncertainty.statespace.lti_sde import discretize_lti_sde
from opifex.uncertainty.types import PredictiveDistribution


_ST_VGP_SOURCE_PACKAGE = "opifex.uncertainty.gp"
_ST_VGP_METHOD = DefaultStrategy.GAUSSIAN_PROCESS.value


def separable_spatiotemporal_kernel(
    *,
    times: jax.Array,
    space: jax.Array,
    temporal_kernel: StateSpaceKernel,
    spatial_lengthscale: float,
    spatial_output_scale: float,
) -> jax.Array:
    r"""Dense separable space-time covariance on the flattened ``(T, M)`` grid.

    Returns the ``(T M, T M)`` Gram matrix of the separable kernel
    ``k((t, R), (t', R')) = k_t(t, t') k_s(R, R')`` evaluated on the
    Cartesian product of ``times`` and ``space``, ordered with the
    spatial index varying fastest (``time``-major flattening). On a
    regular grid this equals the Kronecker product ``K_t \otimes K_s``
    (bayesnewton ``SpatioTemporalKernel.K`` line 462).

    The temporal Gram matrix is reconstructed from the state-space kernel
    via its stationary covariance and closed-form state transition:
    ``k_t(t, t') = H P_\infty A(|t - t'|)^\top H^\top``
    (Sarkka 2013 §6; bayesnewton ``Kernel.K`` for SDE kernels).

    Args:
        times: Time stamps of shape ``(T, 1)`` (or ``(T,)``).
        space: Spatial locations of shape ``(M, d)``.
        temporal_kernel: A :class:`StateSpaceKernel` temporal prior.
        spatial_lengthscale: RBF spatial length-scale (strictly positive).
        spatial_output_scale: RBF spatial output-scale (strictly positive).

    Returns:
        ``(T M, T M)`` separable covariance matrix.
    """
    time_column = times.reshape(-1, 1)
    temporal_gram = _state_space_temporal_gram(time_column, temporal_kernel)
    spatial_gram = rbf_kernel(
        space,
        space,
        lengthscale=spatial_lengthscale,
        output_scale=spatial_output_scale,
    )
    return jnp.kron(temporal_gram, spatial_gram)


def _state_space_temporal_gram(times: jax.Array, temporal_kernel: StateSpaceKernel) -> jax.Array:
    r"""Temporal Gram ``k_t(t_i, t_j)`` from the state-space kernel.

    For a stationary state-space GP, ``k_t(t, t') = H P_\infty
    A(|t - t'|)^\top H^\top`` where ``A`` is the closed-form discrete
    transition (Sarkka 2013 eqn 6.7). Vectorised over the pairwise lag
    matrix so it stays ``jax.jit``-compatible.
    """
    measurement = temporal_kernel.measurement
    stationary_cov = temporal_kernel.stationary_cov
    flat_times = times.reshape(-1)
    lags = jnp.abs(flat_times[:, None] - flat_times[None, :])

    def covariance_at_lag(lag: jax.Array) -> jax.Array:
        transition = temporal_kernel.state_transition(lag)
        return (measurement @ stationary_cov @ transition.T @ measurement.T)[0, 0]

    return jax.vmap(jax.vmap(covariance_at_lag))(lags)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatioTemporalGPState:
    r"""Fitted state of a conjugate spatio-temporal variational GP.

    Attributes:
        smoothed_means: RTS-smoothed latent state means per time, shape
            ``(T, M * state_dim_t)`` (spatial index outer, temporal SDE
            state inner — the ``I_M \\otimes ·`` Kronecker layout).
        smoothed_covs: RTS-smoothed latent state covariances per time,
            shape ``(T, M * state_dim_t, M * state_dim_t)``.
        measurement: Lifted measurement matrix ``H = I_M \\otimes H_t``
            of shape ``(M, M * state_dim_t)``.
        noise_std: Observation noise scale ``σ`` used at fit time.
        num_space: Number of spatial points ``M``.
    """

    smoothed_means: jax.Array
    smoothed_covs: jax.Array
    measurement: jax.Array
    noise_std: jax.Array
    num_space: int


def _require_positive(name: str, value: jax.Array | float) -> None:
    """Fail fast on a non-positive scalar hyperparameter (static check)."""
    # Only a Python ``float`` can be range-checked at trace time; traced
    # arrays (under ``jax.jit``) are validated upstream by the caller.
    if isinstance(value, float) and value <= 0.0:
        raise ValueError(f"{name} must be strictly positive; got {value!r}.")


def _build_kronecker_state_space(
    *,
    times: jax.Array,
    space: jax.Array,
    temporal_kernel: StateSpaceKernel,
    spatial_lengthscale: float,
    spatial_output_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Assemble the Kronecker-lifted space-time state-space matrices.

    Returns ``(transitions, process_noises, measurement, initial_mean,
    initial_cov)`` for the ``T``-step Kalman filter, where the per-step
    transition is ``A_k = I_M \otimes A_t(\Delta t_k)`` and the process
    noise is ``Q_k = K_{zz} \otimes Q_t(\Delta t_k)``. The initial state
    is the stationary distribution ``\mathcal{N}(0, K_{zz} \otimes
    P_\infty^t)`` (bayesnewton ``stationary_covariance`` /
    ``state_transition`` / ``measurement_model``).
    """
    time_column = times.reshape(-1, 1)
    flat_times = time_column.reshape(-1)
    deltas = jnp.concatenate([jnp.zeros((1,), dtype=flat_times.dtype), jnp.diff(flat_times)])
    num_space = space.shape[0]
    spatial_gram = rbf_kernel(
        space,
        space,
        lengthscale=spatial_lengthscale,
        output_scale=spatial_output_scale,
    )
    identity_space = jnp.eye(num_space, dtype=spatial_gram.dtype)

    def per_step(delta: jax.Array) -> tuple[jax.Array, jax.Array]:
        transition_time = temporal_kernel.state_transition(delta)
        _, process_noise_time = discretize_lti_sde(
            drift_matrix=temporal_kernel.feedback,
            dispersion_matrix=temporal_kernel.noise_effect,
            dt=delta,
            diffusion=temporal_kernel.diffusion,
        )
        transition = jnp.kron(identity_space, transition_time)
        process_noise = jnp.kron(spatial_gram, process_noise_time)
        return transition, process_noise

    transitions, process_noises = jax.vmap(per_step)(deltas)

    measurement = jnp.kron(identity_space, temporal_kernel.measurement)
    initial_cov = jnp.kron(spatial_gram, temporal_kernel.stationary_cov)
    initial_mean = jnp.zeros((initial_cov.shape[0],), dtype=initial_cov.dtype)
    return transitions, process_noises, measurement, initial_mean, initial_cov


def fit_spatiotemporal_vgp(
    *,
    times: jax.Array,
    space: jax.Array,
    observations: jax.Array,
    temporal_kernel: StateSpaceKernel,
    spatial_lengthscale: float,
    spatial_output_scale: float,
    noise_std: jax.Array | float,
) -> SpatioTemporalGPState:
    r"""Fit the conjugate spatio-temporal variational GP on gridded data.

    Runs the Kronecker-lifted Kalman filter and RTS smoother over the
    ``T`` time steps; the observation at each step is the ``M``-vector of
    spatial measurements with diagonal Gaussian noise ``σ^2 I_M``
    (bayesnewton ``MarkovGaussianProcess.update_posterior`` line 692,
    specialised to the conjugate Gaussian case where the pseudo
    likelihood equals the true likelihood).

    Args:
        times: Time stamps of shape ``(T, 1)`` (or ``(T,)``), assumed
            sorted ascending.
        space: Spatial locations of shape ``(M, d)``.
        observations: Field observations of shape ``(T, M)``.
        temporal_kernel: A :class:`StateSpaceKernel` temporal prior.
        spatial_lengthscale: RBF spatial length-scale (strictly positive).
        spatial_output_scale: RBF spatial output-scale (strictly positive).
        noise_std: Observation noise scale ``σ`` (strictly positive).

    Returns:
        A :class:`SpatioTemporalGPState` carrying the smoothed latent
        posterior and the lifted measurement model.

    Raises:
        ValueError: If a static (non-traced) hyperparameter is
            non-positive, or ``observations`` is not ``(T, M)``.
    """
    _require_positive("spatial_lengthscale", spatial_lengthscale)
    _require_positive("spatial_output_scale", spatial_output_scale)
    _require_positive("noise_std", noise_std)
    if observations.ndim != 2:
        raise ValueError(
            "observations must be a 2-D (T, M) array of gridded field values; "
            f"got shape {observations.shape}."
        )

    num_space = space.shape[0]
    (
        transitions,
        process_noises,
        measurement,
        initial_mean,
        initial_cov,
    ) = _build_kronecker_state_space(
        times=times,
        space=space,
        temporal_kernel=temporal_kernel,
        spatial_lengthscale=spatial_lengthscale,
        spatial_output_scale=spatial_output_scale,
    )

    noise_std_array = jnp.asarray(noise_std)
    observation_cov = (noise_std_array**2) * jnp.eye(num_space)
    observation_covs = jnp.broadcast_to(
        observation_cov, (observations.shape[0], num_space, num_space)
    )

    filter_means, filter_covs = kalman_filter(
        transitions=transitions,
        process_noises=process_noises,
        observations=observations,
        observation_matrix=measurement,
        observation_covs=observation_covs,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )
    smoothed_means, smoothed_covs = kalman_smoother(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=transitions,
        process_noises=process_noises,
    )
    return SpatioTemporalGPState(
        smoothed_means=smoothed_means,
        smoothed_covs=smoothed_covs,
        measurement=measurement,
        noise_std=noise_std_array,
        num_space=num_space,
    )


def predict_spatiotemporal_vgp(
    *,
    state: SpatioTemporalGPState,
    include_observation_noise: bool = False,
) -> PredictiveDistribution:
    r"""Predict the field at the training space-time grid from a fitted state.

    Projects the smoothed latent state to function space at every time
    via the measurement model ``f(t) = H x(t)`` and extracts the per-point
    marginal variance ``\mathrm{diag}(H P(t) H^\top)`` (bayesnewton
    ``MarkovGaussianProcess.predict`` line 800, spatio-temporal branch:
    ``test_var = diag(W P W^\top)`` with ``W = H`` for inducing-at-data).

    Args:
        state: A fitted :class:`SpatioTemporalGPState`.
        include_observation_noise: When ``True``, adds ``σ^2`` to each
            marginal variance to form the *observation* predictive
            (``p(y_*)``); when ``False`` (default) returns the *latent*
            predictive (``p(f_*)``).

    Returns:
        A :class:`PredictiveDistribution` whose ``mean`` and ``variance``
        are shaped ``(T, M)``.
    """
    measurement = state.measurement

    def project(mean: jax.Array, cov: jax.Array) -> tuple[jax.Array, jax.Array]:
        field_mean = measurement @ mean
        field_cov = measurement @ cov @ measurement.T
        return field_mean, jnp.diag(field_cov)

    means, variances = jax.vmap(project)(state.smoothed_means, state.smoothed_covs)
    if include_observation_noise:
        variances = variances + state.noise_std**2

    return PredictiveDistribution(
        mean=means,
        variance=variances,
        epistemic=variances,
        total_uncertainty=variances,
        metadata=compose_method_metadata(
            method=_ST_VGP_METHOD,
            source_package=_ST_VGP_SOURCE_PACKAGE,
            extra=(
                ("estimator", "spatiotemporal_variational_gp"),
                ("paper", "Hamelijnck+ 2021 arXiv:2111.01732"),
                ("covariance_form", "diag"),
            ),
        ),
    )


def spatiotemporal_vgp_log_marginal(
    *,
    times: jax.Array,
    space: jax.Array,
    observations: jax.Array,
    temporal_kernel: StateSpaceKernel,
    spatial_lengthscale: float,
    spatial_output_scale: float,
    noise_std: jax.Array | float,
) -> jax.Array:
    r"""Marginal log-likelihood of the gridded observations under the ST-VGP.

    Equals the Kalman innovation log-likelihood of the Kronecker-lifted
    state-space model (bayesnewton ``compute_log_lik`` line 729 — the
    conjugate-Gaussian case where the pseudo likelihood is exact). Used
    for hyperparameter learning via ``jax.grad``.

    Args:
        times: Time stamps of shape ``(T, 1)`` (or ``(T,)``).
        space: Spatial locations of shape ``(M, d)``.
        observations: Field observations of shape ``(T, M)``.
        temporal_kernel: A :class:`StateSpaceKernel` temporal prior.
        spatial_lengthscale: RBF spatial length-scale (strictly positive).
        spatial_output_scale: RBF spatial output-scale (strictly positive).
        noise_std: Observation noise scale ``σ`` (strictly positive).

    Returns:
        Scalar marginal log-likelihood ``log p(Y)``.
    """
    num_space = space.shape[0]
    (
        transitions,
        process_noises,
        measurement,
        initial_mean,
        initial_cov,
    ) = _build_kronecker_state_space(
        times=times,
        space=space,
        temporal_kernel=temporal_kernel,
        spatial_lengthscale=spatial_lengthscale,
        spatial_output_scale=spatial_output_scale,
    )
    noise_std_array = jnp.asarray(noise_std)
    observation_cov = (noise_std_array**2) * jnp.eye(num_space)
    observation_covs = jnp.broadcast_to(
        observation_cov, (observations.shape[0], num_space, num_space)
    )
    return kalman_log_likelihood(
        transitions=transitions,
        process_noises=process_noises,
        observations=observations,
        observation_matrix=measurement,
        observation_covs=observation_covs,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )


__all__ = [
    "SpatioTemporalGPState",
    "fit_spatiotemporal_vgp",
    "predict_spatiotemporal_vgp",
    "separable_spatiotemporal_kernel",
    "spatiotemporal_vgp_log_marginal",
]
