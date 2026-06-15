r"""Tests for the celerite Complex term + kernel summation (Task 11.1 D4).

Foreman-Mackey, Agol, Ambikasaran, Angus 2017 (AJ, arXiv:1703.09710)
introduce the **celerite** family of quasiseparable kernels for fast
one-dimensional GP regression. Slice 10 shipped the underdamped SHO
specialisation; this slice ships the remaining direct-evaluation
parametrisations:

* **Real term** ``k(τ) = a exp(-c τ)`` — already provided by
  :func:`matern12_kernel` with ``a = output_scale²``, ``c = 1 / ℓ``.
  No new symbol is required.
* **Complex term** ``k(τ) = exp(-c τ) [a cos(d τ) + b sin(d τ)]`` —
  the building block of *every* celerite kernel including the SHO
  (Foreman-Mackey+ 2017 §4.1; Foreman-Mackey 2018 RNAAS).
* **Sum-of-terms composition** ``k = Σ_q k_q`` — every celerite
  predictive surface is a sum of Real / Complex terms.

The direct-evaluation forms below run in ``O(n²)`` per Gram; the
scalable quasiseparable Kalman ports (``O(n)``) are deferred to D1.

Reference implementation consulted (READ-ONLY):
``../tinygp/src/tinygp/kernels/quasisep.py:Celerite``.

References
----------
* Foreman-Mackey, D., Agol, E., Ambikasaran, S., Angus, R. 2017 —
  *Fast and scalable Gaussian process modeling with applications to
  astronomical time series*, AJ, arXiv:1703.09710 (PRIMARY).
* Foreman-Mackey, D. 2018 — *Scalable backpropagation for Gaussian
  processes using celerite*, RNAAS (Real / Complex parametrisation).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    celerite_complex_kernel,
    fit_exact_gp,
    kernel_sum,
    matern12_kernel,
    matern32_kernel,
    predict_exact_gp,
    rbf_kernel,
)


# -----------------------------------------------------------------------------
# Celerite Complex term
# -----------------------------------------------------------------------------


def test_celerite_complex_kernel_matches_closed_form_at_known_lags() -> None:
    r"""``k(τ) = exp(-c τ) [a cos(d τ) + b sin(d τ)]`` at fixed parameters."""
    kernel = celerite_complex_kernel(sine_amplitude=0.3, frequency=2.0)
    x1 = jnp.asarray([[0.0]])
    x2 = jnp.asarray([[0.4], [1.2]])
    # output_scale = 1.0 => a = 1.0; lengthscale = 1.0 => c = 1.0; b = 0.3, d = 2.0
    gram = kernel(x1, x2, lengthscale=1.0, output_scale=1.0)
    tau = jnp.array([0.4, 1.2])
    expected = jnp.exp(-tau) * (jnp.cos(2.0 * tau) + 0.3 * jnp.sin(2.0 * tau))
    assert jnp.allclose(gram[0], expected, atol=1e-6)


def test_celerite_complex_kernel_returns_a_at_zero_lag() -> None:
    """``k(0) = a = output_scale²`` (cos(0) = 1, sin(0) = 0)."""
    kernel = celerite_complex_kernel(sine_amplitude=0.5, frequency=1.5)
    x = jnp.asarray([[0.0], [1.0], [-2.0]])
    gram = kernel(x, x, lengthscale=0.5, output_scale=2.0)
    diag = jnp.diag(gram)
    assert jnp.allclose(diag, 4.0, atol=1e-6)


def test_celerite_complex_kernel_is_symmetric_in_its_arguments() -> None:
    """``k(x1, x2) = k(x2, x1)^T`` because the kernel depends only on ``|τ|``."""
    kernel = celerite_complex_kernel(sine_amplitude=0.2, frequency=3.0)
    x1 = jnp.asarray([[0.1], [0.7], [-0.5]])
    x2 = jnp.asarray([[0.4], [1.2]])
    forward = kernel(x1, x2, lengthscale=0.6, output_scale=1.0)
    backward = kernel(x2, x1, lengthscale=0.6, output_scale=1.0)
    assert jnp.allclose(forward, backward.T, atol=1e-6)


def test_celerite_complex_kernel_with_zero_frequency_reduces_to_matern12() -> None:
    r"""At ``d = 0`` the cosine collapses to 1 and the sine vanishes."""
    complex_kernel = celerite_complex_kernel(sine_amplitude=0.0, frequency=0.0)
    x = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    actual = complex_kernel(x, x, lengthscale=0.5, output_scale=1.5)
    expected = matern12_kernel(x, x, lengthscale=0.5, output_scale=1.5)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_celerite_complex_kernel_is_jit_compatible() -> None:
    """End-to-end ``jax.jit`` compatibility through ``fit_exact_gp``."""
    kernel = celerite_complex_kernel(sine_amplitude=0.4, frequency=2.5)
    x_train = jnp.linspace(0.0, 2.0, 8).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1)) * jnp.exp(-0.2 * x_train.squeeze(-1))
    x_test = jnp.linspace(0.0, 2.0, 4).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.6,
            output_scale=1.0,
            noise_std=0.05,
            kernel_fn=kernel,
        )
        predictive = predict_exact_gp(state=state, x_test=x_q)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(x_train, y_train, x_test)
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))


# -----------------------------------------------------------------------------
# kernel_sum — superposition of arbitrary opifex kernels
# -----------------------------------------------------------------------------


def test_kernel_sum_two_kernels_equals_manual_addition_of_grams() -> None:
    r"""``kernel_sum(k1, k2)(x, x') == k1(x, x') + k2(x, x')``."""
    summed = kernel_sum(kernel_fns=(rbf_kernel, matern32_kernel))
    x = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    expected = rbf_kernel(x, x, lengthscale=0.5, output_scale=1.0) + matern32_kernel(
        x, x, lengthscale=0.5, output_scale=1.0
    )
    actual = summed(x, x, lengthscale=0.5, output_scale=1.0)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_kernel_sum_supports_celerite_real_plus_complex_superposition() -> None:
    """Full celerite-style sum: Matern12 (Real term) + Complex term."""
    summed = kernel_sum(
        kernel_fns=(
            matern12_kernel,
            celerite_complex_kernel(sine_amplitude=0.3, frequency=2.0),
        )
    )
    x = jnp.linspace(0.0, 1.5, 5).reshape(-1, 1)
    real_part = matern12_kernel(x, x, lengthscale=0.5, output_scale=1.0)
    complex_kernel = celerite_complex_kernel(sine_amplitude=0.3, frequency=2.0)
    complex_part = complex_kernel(x, x, lengthscale=0.5, output_scale=1.0)
    assert jnp.allclose(
        summed(x, x, lengthscale=0.5, output_scale=1.0),
        real_part + complex_part,
        atol=1e-6,
    )


def test_kernel_sum_single_kernel_collapses_to_that_kernel() -> None:
    """``kernel_sum((k,))(x, x') == k(x, x')``."""
    summed = kernel_sum(kernel_fns=(rbf_kernel,))
    x = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    assert jnp.allclose(
        summed(x, x, lengthscale=0.5, output_scale=1.0),
        rbf_kernel(x, x, lengthscale=0.5, output_scale=1.0),
        atol=1e-6,
    )


def test_kernel_sum_rejects_empty_kernel_tuple() -> None:
    """At least one component is required."""
    with pytest.raises(ValueError, match="kernel"):
        kernel_sum(kernel_fns=())


def test_kernel_sum_is_jit_compatible() -> None:
    """End-to-end ``jax.jit`` compatibility through ``fit_exact_gp``."""
    summed = kernel_sum(
        kernel_fns=(
            matern12_kernel,
            celerite_complex_kernel(sine_amplitude=0.25, frequency=2.0),
        )
    )
    x_train = jnp.linspace(0.0, 2.0, 6).reshape(-1, 1)
    y_train = jnp.cos(2.0 * x_train.squeeze(-1))
    x_test = jnp.linspace(0.0, 2.0, 3).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.6,
            output_scale=1.0,
            noise_std=0.05,
            kernel_fn=summed,
        )
        predictive = predict_exact_gp(state=state, x_test=x_q)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(x_train, y_train, x_test)
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))
