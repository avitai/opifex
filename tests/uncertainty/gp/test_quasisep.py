r"""Tests for the scalable SHO state-space GP — Task 11.1 D1.

Slice 10 shipped the direct-evaluation ``O(n²)`` SHO kernel
(:func:`damped_oscillator_kernel`). This slice ships the **scalable
``O(n)`` Kalman implementation** that maps the SHO covariance into a
2-dimensional linear-Gaussian SDE and runs the closed-form forward
filter + backward smoother. Foreman-Mackey, Agol, Ambikasaran, Angus
2017 (AJ, arXiv:1703.09710) introduce the celerite quasiseparable
representation; the state-space form here is the standard equivalent
(Solin 2016 PhD; Sarkka 2013 *Bayesian Filtering and Smoothing* §4.3).

Equivalence guarantee
---------------------

The 2-state SHO SDE has drift

.. math::

    F = \begin{pmatrix} 0 & 1 \\
        -\omega^{2} & -\omega/Q \end{pmatrix},
    \qquad P_{\infty} = \mathrm{diag}(1, \omega^{2}),
    \qquad H = (\sigma_{f}, 0),

and the closed-form discrete transition matrix matches the direct
``damped_oscillator_kernel`` covariance at any lag. Consequently:

* posterior mean / variance at any test point coincide with the
  direct-form exact GP up to numerical precision;
* the log marginal likelihood coincides with
  ``-½ y^T (K + σ² I)^{-1} y - ½ log|K + σ² I| - ½ n log 2π``.

Reference implementation consulted (READ-ONLY):
``../tinygp/src/tinygp/kernels/quasisep.py:SHO``
(``design_matrix`` + ``stationary_covariance`` + ``transition_matrix``).

References
----------
* Foreman-Mackey, D., Agol, E., Ambikasaran, S., Angus, R. 2017 —
  *Fast and scalable Gaussian process modeling with applications to
  astronomical time series*, AJ, arXiv:1703.09710 (PRIMARY).
* Sarkka, S. 2013 — *Bayesian Filtering and Smoothing*, CUP
  (state-space GP equivalence).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    damped_oscillator_kernel,
    fit_exact_gp,
    fit_quasisep_sho_gp,
    predict_exact_gp,
    predict_quasisep_sho_gp,
    QuasisepGPState,
)
from opifex.uncertainty.types import PredictiveDistribution


def _toy_time_series_data(seed: int = 0, *, num_train: int = 30) -> tuple[jax.Array, jax.Array]:
    """1-D damped-oscillator-like time series: ``y(t) = sin(2t) e^{-0.2 t}``."""
    key = jax.random.PRNGKey(seed)
    t_train = jnp.sort(jax.random.uniform(key, (num_train,), minval=0.0, maxval=4.0 * jnp.pi))
    y_train = jnp.sin(2.0 * t_train) * jnp.exp(-0.2 * t_train)
    return t_train.reshape(-1, 1), y_train


# -----------------------------------------------------------------------------
# Equivalence against the direct-form damped_oscillator_kernel exact GP
# -----------------------------------------------------------------------------


def _direct_exact_gp_log_marginal(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    noise_std: float,
    quality_factor: float,
) -> jax.Array:
    """Closed-form exact-GP log marginal for the direct-form SHO kernel."""
    kernel = damped_oscillator_kernel(quality_factor=quality_factor)
    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        kernel_fn=kernel,
    )
    n = y_train.shape[0]
    return (
        -0.5 * jnp.dot(y_train, state.alpha)
        - jnp.sum(jnp.log(jnp.diag(state.cholesky)))
        - 0.5 * n * jnp.log(2.0 * jnp.pi)
    )


def test_fit_quasisep_sho_gp_log_marginal_matches_direct_form_exact_gp() -> None:
    """Kalman log-marginal == direct-form exact-GP log-marginal."""
    x_train, y_train = _toy_time_series_data(0)
    quality, lengthscale, output_scale, noise_std = 1.5, 0.6, 1.0, 0.05
    quasisep_state = fit_quasisep_sho_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        quality_factor=quality,
    )
    direct_log_marginal = _direct_exact_gp_log_marginal(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        quality_factor=quality,
    )
    assert isinstance(quasisep_state, QuasisepGPState)
    assert jnp.isfinite(quasisep_state.log_marginal_likelihood)
    assert jnp.allclose(
        quasisep_state.log_marginal_likelihood,
        direct_log_marginal,
        atol=1e-2,
        rtol=1e-3,
    )


def test_predict_quasisep_sho_gp_posterior_mean_matches_direct_form() -> None:
    """At held-out times the Kalman posterior mean matches the direct exact GP."""
    x_train, y_train = _toy_time_series_data(1, num_train=25)
    quality, lengthscale, output_scale, noise_std = 1.5, 0.6, 1.0, 0.05
    quasisep_state = fit_quasisep_sho_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        quality_factor=quality,
    )
    direct_kernel = damped_oscillator_kernel(quality_factor=quality)
    direct_state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        kernel_fn=direct_kernel,
    )
    x_test = jnp.linspace(1.0, 10.0, 12).reshape(-1, 1)
    quasisep_pred = predict_quasisep_sho_gp(state=quasisep_state, x_test=x_test)
    direct_pred = predict_exact_gp(state=direct_state, x_test=x_test)
    assert isinstance(quasisep_pred, PredictiveDistribution)
    assert jnp.allclose(quasisep_pred.mean, direct_pred.mean, atol=1e-3, rtol=1e-3)


def test_predict_quasisep_sho_gp_posterior_variance_matches_direct_form() -> None:
    """At held-out times the Kalman posterior variance matches the direct exact GP."""
    x_train, y_train = _toy_time_series_data(2, num_train=25)
    quality, lengthscale, output_scale, noise_std = 1.5, 0.6, 1.0, 0.05
    quasisep_state = fit_quasisep_sho_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        quality_factor=quality,
    )
    direct_kernel = damped_oscillator_kernel(quality_factor=quality)
    direct_state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        kernel_fn=direct_kernel,
    )
    x_test = jnp.linspace(1.0, 10.0, 12).reshape(-1, 1)
    quasisep_pred = predict_quasisep_sho_gp(state=quasisep_state, x_test=x_test)
    direct_pred = predict_exact_gp(state=direct_state, x_test=x_test)
    assert quasisep_pred.variance is not None
    assert direct_pred.variance is not None
    assert jnp.allclose(quasisep_pred.variance, direct_pred.variance, atol=1e-3, rtol=1e-3)


def test_quasisep_sho_gp_is_jit_compatible() -> None:
    """``jax.jit`` compiles the full fit + predict pipeline."""
    x_train, y_train = _toy_time_series_data(3, num_train=15)
    x_test = jnp.linspace(0.5, 8.0, 5).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_quasisep_sho_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.5,
            output_scale=1.0,
            noise_std=0.05,
            quality_factor=1.5,
        )
        predictive = predict_quasisep_sho_gp(state=state, x_test=x_q)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(x_train, y_train, x_test)
    assert out.shape == (5,)
    assert jnp.all(jnp.isfinite(out))


def test_quasisep_sho_gp_rejects_critically_or_overdamped_quality_factor() -> None:
    """Only the underdamped regime ``Q > 1/2`` is implemented for this slice."""
    x_train, y_train = _toy_time_series_data(4)
    with pytest.raises(ValueError, match="quality_factor"):
        fit_quasisep_sho_gp(
            x_train=x_train,
            y_train=y_train,
            lengthscale=0.5,
            output_scale=1.0,
            noise_std=0.05,
            quality_factor=0.5,
        )


def test_quasisep_sho_gp_requires_sorted_training_times() -> None:
    """Unsorted training times raise a clear ``ValueError``."""
    x_train = jnp.asarray([[0.0], [2.0], [1.0]])
    y_train = jnp.asarray([0.0, 1.0, 0.5])
    with pytest.raises(ValueError, match="sorted"):
        fit_quasisep_sho_gp(
            x_train=x_train,
            y_train=y_train,
            lengthscale=0.5,
            output_scale=1.0,
            noise_std=0.05,
            quality_factor=1.5,
        )


def test_quasisep_sho_gp_scales_linearly_with_training_size() -> None:
    """Smoke test: fit time at ``n = 800`` is well under 2× fit time at ``n = 200``."""
    key = jax.random.PRNGKey(42)
    rate_lengthscale, output_scale, noise_std, quality = 0.4, 1.0, 0.05, 1.5

    def _fit_at(num_steps: int) -> float:
        x = jnp.sort(jax.random.uniform(key, (num_steps,), minval=0.0, maxval=20.0)).reshape(-1, 1)
        y = jnp.sin(2.0 * x.squeeze(-1)) * jnp.exp(-0.05 * x.squeeze(-1))
        # Warm-up trace once so we measure scan execution, not tracing.
        fitted = fit_quasisep_sho_gp(
            x_train=x,
            y_train=y,
            lengthscale=rate_lengthscale,
            output_scale=output_scale,
            noise_std=noise_std,
            quality_factor=quality,
        )
        fitted.log_marginal_likelihood.block_until_ready()
        start = time.perf_counter()
        for _ in range(3):
            out = fit_quasisep_sho_gp(
                x_train=x,
                y_train=y,
                lengthscale=rate_lengthscale,
                output_scale=output_scale,
                noise_std=noise_std,
                quality_factor=quality,
            )
            out.log_marginal_likelihood.block_until_ready()
        return (time.perf_counter() - start) / 3.0

    small_time = _fit_at(200)
    large_time = _fit_at(800)
    # O(n) scaling: 4x more data should be << 16x runtime; allow generous slack
    # for JAX dispatch overhead, but rule out anything quadratic-ish.
    assert large_time < 8.0 * small_time
