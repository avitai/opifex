r"""Tests for the spatio-temporal variational GP (ST-VGP) — Feature F17.

Strict-TDD specification for :mod:`opifex.uncertainty.gp.spatiotemporal`.

The model implements the separable space-time GP of Hamelijnck, Wilkinson,
Loka, Solin, Damoulas 2021 (NeurIPS, arXiv:2111.01732): a separable
covariance ``k((t, R), (t', R')) = k_t(t, t') k_s(R, R')`` whose temporal
factor is a Markovian (state-space) GP solved in ``O(T)`` by Kalman
filtering/smoothing, while the spatial factor is a standard GP over the
spatial inputs. The canonical reference is
``../bayesnewton/bayesnewton/kernels.py:SpatioTemporalKernel`` (line 385)
and ``../bayesnewton/bayesnewton/basemodels.py:MarkovGaussianProcess``
(``predict`` line 766, ``conditional_posterior_to_data`` line 745).

Test contract
-------------
* the separable ST covariance has the right shape and equals the Hadamard
  product of the temporal and spatial Gram matrices;
* it reduces to the temporal kernel (scaled by ``k_s(R, R)``) when the
  spatial inputs coincide;
* predictive mean / variance have ``mean.shape`` matching the query grid;
* a degenerate single-spatial-point ST-VGP recovers a pure temporal exact
  GP marginal (mean + variance) up to numerical precision;
* on a small ST dataset predictions are finite and the empirical
  standardised residuals are roughly calibrated (~unit variance);
* the marginal log-likelihood is jit- and grad-able w.r.t. hyperparameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    fit_exact_gp,
    fit_spatiotemporal_vgp,
    matern32_kernel as direct_matern32,
    predict_exact_gp,
    predict_spatiotemporal_vgp,
    rbf_kernel,
    separable_spatiotemporal_kernel,
    spatiotemporal_vgp_log_marginal,
    SpatioTemporalGPState,
)
from opifex.uncertainty.statespace import matern32_kernel
from opifex.uncertainty.types import PredictiveDistribution


# -----------------------------------------------------------------------------
# Toy gridded spatio-temporal data: f(t, x) = sin(t) cos(x) on a (T x M) grid.
# -----------------------------------------------------------------------------


def _toy_st_data(
    *, num_times: int = 12, num_space: int = 5, noise: float = 0.05, seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(times (T,1), space (M,d), observations (T,M))``."""
    times = jnp.linspace(0.0, 2.0 * jnp.pi, num_times).reshape(-1, 1)
    space = jnp.linspace(-2.0, 2.0, num_space).reshape(-1, 1)
    clean = jnp.sin(times) * jnp.cos(space.reshape(1, -1))  # (T, M)
    key = jax.random.PRNGKey(seed)
    observations = clean + noise * jax.random.normal(key, clean.shape)
    return times, space, observations


def _temporal_kernel():
    return matern32_kernel(variance=1.0, lengthscale=1.0)


# -----------------------------------------------------------------------------
# Separable kernel: shape + Hadamard structure + temporal reduction
# -----------------------------------------------------------------------------


def test_separable_kernel_shape_matches_flattened_grid() -> None:
    """The dense ST covariance is ``(T*M, T*M)`` for a flattened space-time grid."""
    times, space, _ = _toy_st_data(num_times=4, num_space=3)
    cov = separable_spatiotemporal_kernel(
        times=times,
        space=space,
        temporal_kernel=_temporal_kernel(),
        spatial_lengthscale=1.0,
        spatial_output_scale=1.0,
    )
    assert cov.shape == (4 * 3, 4 * 3)
    assert jnp.allclose(cov, cov.T, atol=1e-6)


def test_separable_kernel_is_kron_of_temporal_and_spatial() -> None:
    """The separable covariance equals ``K_t ⊗ K_s`` on a regular grid."""
    times, space, _ = _toy_st_data(num_times=4, num_space=3)
    cov = separable_spatiotemporal_kernel(
        times=times,
        space=space,
        temporal_kernel=_temporal_kernel(),
        spatial_lengthscale=0.7,
        spatial_output_scale=1.3,
    )
    # Build the reference temporal Gram from the state-space kernel directly.
    temporal_gram = direct_matern32(times, times, lengthscale=1.0, output_scale=1.0)
    spatial_gram = rbf_kernel(space, space, lengthscale=0.7, output_scale=1.3)
    expected = jnp.kron(temporal_gram, spatial_gram)
    assert jnp.allclose(cov, expected, atol=1e-5)


def test_separable_kernel_reduces_to_temporal_when_space_coincides() -> None:
    """One spatial point ⇒ ST covariance == spatial-variance-scaled temporal Gram."""
    times, _, _ = _toy_st_data(num_times=5, num_space=1)
    single_space = jnp.zeros((1, 1))
    cov = separable_spatiotemporal_kernel(
        times=times,
        space=single_space,
        temporal_kernel=_temporal_kernel(),
        spatial_lengthscale=1.0,
        spatial_output_scale=2.0,
    )
    temporal_gram = direct_matern32(times, times, lengthscale=1.0, output_scale=1.0)
    assert cov.shape == (5, 5)
    assert jnp.allclose(cov, 4.0 * temporal_gram, atol=1e-5)  # output_scale**2 = 4


# -----------------------------------------------------------------------------
# Fit / predict: shapes + finiteness + calibration
# -----------------------------------------------------------------------------


def test_fit_returns_state() -> None:
    times, space, observations = _toy_st_data()
    state = fit_spatiotemporal_vgp(
        times=times,
        space=space,
        observations=observations,
        temporal_kernel=_temporal_kernel(),
        spatial_lengthscale=1.0,
        spatial_output_scale=1.0,
        noise_std=0.05,
    )
    assert isinstance(state, SpatioTemporalGPState)
    num_times, num_space = observations.shape
    state_dim = _temporal_kernel().state_dim * num_space
    assert state.smoothed_means.shape == (num_times, state_dim)
    assert state.smoothed_covs.shape == (num_times, state_dim, state_dim)


def test_predict_shapes_and_finite() -> None:
    times, space, observations = _toy_st_data()
    state = fit_spatiotemporal_vgp(
        times=times,
        space=space,
        observations=observations,
        temporal_kernel=_temporal_kernel(),
        spatial_lengthscale=1.0,
        spatial_output_scale=1.0,
        noise_std=0.05,
    )
    predictive = predict_spatiotemporal_vgp(state=state)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.mean.shape == observations.shape
    assert predictive.variance is not None
    assert predictive.variance.shape == observations.shape
    assert bool(jnp.all(jnp.isfinite(predictive.mean)))
    assert bool(jnp.all(jnp.isfinite(predictive.variance)))
    assert bool(jnp.all(predictive.variance > 0.0))


def test_predictions_track_signal() -> None:
    """The smoothed posterior mean should be close to the noiseless signal."""
    times, space, observations = _toy_st_data(noise=0.02)
    clean = jnp.sin(times) * jnp.cos(space.reshape(1, -1))
    state = fit_spatiotemporal_vgp(
        times=times,
        space=space,
        observations=observations,
        temporal_kernel=matern32_kernel(variance=1.0, lengthscale=1.5),
        spatial_lengthscale=1.5,
        spatial_output_scale=1.0,
        noise_std=0.02,
    )
    predictive = predict_spatiotemporal_vgp(state=state)
    rmse = jnp.sqrt(jnp.mean((predictive.mean - clean) ** 2))
    assert float(rmse) < 0.3


def test_predictions_are_calibrated() -> None:
    """Standardised residuals should have roughly unit variance (calibration)."""
    times, space, observations = _toy_st_data(num_times=20, num_space=6, noise=0.1)
    clean = jnp.sin(times) * jnp.cos(space.reshape(1, -1))
    state = fit_spatiotemporal_vgp(
        times=times,
        space=space,
        observations=observations,
        temporal_kernel=matern32_kernel(variance=1.0, lengthscale=1.5),
        spatial_lengthscale=1.5,
        spatial_output_scale=1.0,
        noise_std=0.1,
    )
    predictive = predict_spatiotemporal_vgp(state=state, include_observation_noise=True)
    assert predictive.variance is not None
    standardised = (predictive.mean - clean) / jnp.sqrt(predictive.variance)
    empirical_var = jnp.var(standardised)
    # Loose calibration band: not under-confident, not wildly over-confident.
    assert 0.2 < float(empirical_var) < 5.0


# -----------------------------------------------------------------------------
# Degenerate single-spatial-point recovers a pure temporal exact GP
# -----------------------------------------------------------------------------


def test_single_space_point_recovers_temporal_exact_gp() -> None:
    """One spatial location ⇒ marginal matches a 1-D temporal exact GP."""
    times = jnp.linspace(0.0, 4.0 * jnp.pi, 15).reshape(-1, 1)
    space = jnp.zeros((1, 1))
    y_temporal = jnp.sin(times[:, 0]) * jnp.exp(-0.1 * times[:, 0])
    observations = y_temporal.reshape(-1, 1)
    noise_std = 0.1

    st_state = fit_spatiotemporal_vgp(
        times=times,
        space=space,
        observations=observations,
        temporal_kernel=matern32_kernel(variance=1.0, lengthscale=1.2),
        spatial_lengthscale=1.0,
        spatial_output_scale=1.0,
        noise_std=noise_std,
    )
    st_pred = predict_spatiotemporal_vgp(state=st_state)

    # Reference: pure temporal exact GP with the matching Matern-3/2 kernel.
    exact_state = fit_exact_gp(
        x_train=times,
        y_train=y_temporal,
        lengthscale=1.2,
        output_scale=1.0,
        noise_std=noise_std,
        kernel_fn=direct_matern32,
    )
    exact_pred = predict_exact_gp(state=exact_state, x_test=times)

    assert st_pred.variance is not None and exact_pred.variance is not None
    assert jnp.allclose(st_pred.mean.reshape(-1), exact_pred.mean.reshape(-1), atol=1e-3)
    assert jnp.allclose(st_pred.variance.reshape(-1), exact_pred.variance.reshape(-1), atol=1e-3)


# -----------------------------------------------------------------------------
# jit / grad smoke tests (JAX/NNX compatibility — required)
# -----------------------------------------------------------------------------


def test_log_marginal_is_jittable() -> None:
    # Kernel length/output scales are static ``float`` hyperparameters in
    # opifex (matching the exact / SVGP convention); ``noise_std`` is the
    # traced argument, exercising the full Kronecker state-space pipeline.
    times, space, observations = _toy_st_data()

    def loss(noise_std: jax.Array) -> jax.Array:
        return spatiotemporal_vgp_log_marginal(
            times=times,
            space=space,
            observations=observations,
            temporal_kernel=_temporal_kernel(),
            spatial_lengthscale=1.0,
            spatial_output_scale=1.0,
            noise_std=noise_std,
        )

    jitted = jax.jit(loss)
    value = jitted(jnp.asarray(0.05))
    assert bool(jnp.isfinite(value))


def test_log_marginal_is_grad_able() -> None:
    times, space, observations = _toy_st_data()

    def loss(noise_std: jax.Array) -> jax.Array:
        return spatiotemporal_vgp_log_marginal(
            times=times,
            space=space,
            observations=observations,
            temporal_kernel=_temporal_kernel(),
            spatial_lengthscale=1.0,
            spatial_output_scale=1.0,
            noise_std=noise_std,
        )

    grad = jax.grad(loss)(jnp.asarray(0.1))
    assert bool(jnp.isfinite(grad))


def test_fit_predict_under_jit() -> None:
    times, space, observations = _toy_st_data()

    @jax.jit
    def run(noise_std: jax.Array) -> tuple[jax.Array, jax.Array]:
        state = fit_spatiotemporal_vgp(
            times=times,
            space=space,
            observations=observations,
            temporal_kernel=_temporal_kernel(),
            spatial_lengthscale=1.0,
            spatial_output_scale=1.0,
            noise_std=noise_std,
        )
        predictive = predict_spatiotemporal_vgp(state=state)
        assert predictive.variance is not None
        return predictive.mean, predictive.variance

    mean, variance = run(jnp.asarray(0.05))
    assert mean.shape == observations.shape
    assert bool(jnp.all(jnp.isfinite(variance)))


def test_negative_spatial_lengthscale_raises() -> None:
    times, space, observations = _toy_st_data()
    with pytest.raises(ValueError, match="spatial_lengthscale"):
        fit_spatiotemporal_vgp(
            times=times,
            space=space,
            observations=observations,
            temporal_kernel=_temporal_kernel(),
            spatial_lengthscale=-1.0,
            spatial_output_scale=1.0,
            noise_std=0.05,
        )
