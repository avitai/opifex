r"""Six closed-form kernel-measure crosses for Bayesian quadrature.

Tests for :mod:`opifex.uncertainty.quadrature.kernels`. Hand-derived
closed-form ground truths for each cross verify the line-by-line port
of the emukit reference at ``../emukit/emukit/quadrature/kernels/``.

The six crosses (per design fix #190's catalogue):

* ``qk_rbf_gaussian`` / ``qkq_rbf_gaussian`` — RBF kernel × diagonal
  Gaussian measure. Reference: emukit
  ``quadrature_rbf.py:QuadratureRBFGaussianMeasure`` (lines 150, 158).
* ``qk_rbf_lebesgue`` / ``qkq_rbf_lebesgue`` — RBF kernel × product
  Lebesgue measure. Reference: emukit
  ``quadrature_rbf.py:QuadratureRBFLebesgueMeasure`` (lines 105, 113).
* ``qk_matern12_product_lebesgue`` / ``qkq_matern12_product_lebesgue``
  — product Matern-1/2 (exponential) kernel × product Lebesgue.
  Reference: emukit
  ``quadrature_matern12.py:QuadratureProductMatern12LebesgueMeasure``
  (lines 93, 103).
* ``qk_matern32_product_lebesgue`` / ``qkq_matern32_product_lebesgue``
  — product Matern-3/2 kernel × product Lebesgue.
  Reference: emukit
  ``quadrature_matern32.py:QuadratureProductMatern32LebesgueMeasure``
  (lines 93, 103).
* ``qk_matern52_product_lebesgue`` / ``qkq_matern52_product_lebesgue``
  — product Matern-5/2 kernel × product Lebesgue.
  Reference: emukit
  ``quadrature_matern52.py:QuadratureProductMatern52LebesgueMeasure``
  (lines 93, 111).
* ``qk_brownian_lebesgue`` / ``qkq_brownian_lebesgue`` — Brownian
  motion kernel × 1-D Lebesgue. Reference: emukit
  ``quadrature_brownian.py:QuadratureBrownianLebesgueMeasure``
  (lines 89, 95).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from opifex.uncertainty.quadrature.kernels import (
    qk_brownian_lebesgue,
    qk_matern12_product_lebesgue,
    qk_matern32_product_lebesgue,
    qk_matern52_product_lebesgue,
    qk_rbf_gaussian,
    qk_rbf_lebesgue,
    qkq_brownian_lebesgue,
    qkq_matern12_product_lebesgue,
    qkq_matern32_product_lebesgue,
    qkq_matern52_product_lebesgue,
    qkq_rbf_gaussian,
    qkq_rbf_lebesgue,
)


# ---------------------------------------------------------------------------
# RBF × Gaussian (already used internally by bayesian_quadrature.py)
# ---------------------------------------------------------------------------


def test_qk_rbf_gaussian_single_point_at_measure_mean_recovers_one_over_sqrt_two() -> None:
    r"""``qK(0) = 1/√2`` for unit RBF + ``N(0, 1)`` measure."""
    result = qk_rbf_gaussian(
        points=jnp.array([[0.0]]),
        measure_mean=jnp.array([0.0]),
        measure_variance=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
    )
    assert jnp.allclose(result, 1.0 / jnp.sqrt(2.0), atol=1e-6)


def test_qkq_rbf_gaussian_recovers_one_over_sqrt_three() -> None:
    r"""``qKq = 1/√3`` for unit RBF + ``N(0, 1)`` measure."""
    result = qkq_rbf_gaussian(
        measure_variance=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
    )
    assert jnp.allclose(result, 1.0 / jnp.sqrt(3.0), atol=1e-6)


# ---------------------------------------------------------------------------
# RBF × Lebesgue
# ---------------------------------------------------------------------------


def test_qk_rbf_lebesgue_single_point_at_interval_centre() -> None:
    r"""For unit RBF on ``[-1, 1]`` at ``x'=0``:

    ``qK(0) = σ² density √(π/2) ℓ [erf(1/(√2 ℓ)) - erf(-1/(√2 ℓ))]
             = √(π/2) · 2 erf(1/√2)``.
    """
    result = qk_rbf_lebesgue(
        points=jnp.array([[0.0]]),
        lower=jnp.array([-1.0]),
        upper=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
        density=jnp.asarray(1.0),
    )
    expected = jnp.sqrt(jnp.pi / 2.0) * 2.0 * erf(1.0 / jnp.sqrt(2.0))
    assert jnp.allclose(result, expected, atol=1e-6)


def test_qkq_rbf_lebesgue_closed_form_on_symmetric_interval() -> None:
    r"""``qKq`` for unit RBF on ``[-1, 1]``:

    With ``d = (b-a)/(√2 ℓ) = √2``,
    ``qKq = 2 √π ℓ² [(exp(-d²) - 1)/√π + erf(d) · d]
          = 2(exp(-2) - 1) + 2 √π · √2 · erf(√2)``.
    """
    result = qkq_rbf_lebesgue(
        lower=jnp.array([-1.0]),
        upper=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
        density=jnp.asarray(1.0),
    )
    sqrt2 = jnp.sqrt(2.0)
    expected = 2.0 * (jnp.exp(-2.0) - 1.0) + 2.0 * jnp.sqrt(jnp.pi) * sqrt2 * erf(sqrt2)
    assert jnp.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Matern-1/2 × Lebesgue (Exponential kernel)
# ---------------------------------------------------------------------------


def test_qk_matern12_lebesgue_at_interval_centre() -> None:
    r"""For unit Matern-1/2 (``k(x,x') = exp(-|x-x'|/ℓ)``) on ``[-1, 1]`` at ``x'=0``:

    ``qK(0) = σ² · ℓ · (2 - exp(-1/ℓ) - exp(-1/ℓ)) = 2 - 2/e``.
    """
    result = qk_matern12_product_lebesgue(
        points=jnp.array([[0.0]]),
        lower=jnp.array([-1.0]),
        upper=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
    )
    expected = 2.0 - 2.0 * jnp.exp(-1.0)
    assert jnp.allclose(result, expected, atol=1e-6)


def test_qkq_matern12_lebesgue_closed_form() -> None:
    r"""``qKq`` for unit Matern-1/2 on ``[-1, 1]``:

    ``qKq = 2 σ² ℓ ((b-a) + ℓ (exp(-(b-a)/ℓ) - 1))
          = 2 (2 + exp(-2) - 1) = 2 (1 + exp(-2))``.
    """
    result = qkq_matern12_product_lebesgue(
        lower=jnp.array([-1.0]),
        upper=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
    )
    expected = 2.0 * (1.0 + jnp.exp(-2.0))
    assert jnp.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Matern-3/2 × Lebesgue
# ---------------------------------------------------------------------------


def test_qk_matern32_lebesgue_at_interval_centre() -> None:
    r"""For unit Matern-3/2 on ``[-1, 1]`` at ``x'=0``:

    ``qK(0) = 4ℓ/√3 - exp(√3(x'-b)/ℓ)(b + 2ℓ/√3 - x')
              - exp(√3(a-x')/ℓ)(x' + 2ℓ/√3 - a)``.

    At ``x'=0``, both exp tails contribute the same value.
    """
    s3 = jnp.sqrt(3.0)
    result = qk_matern32_product_lebesgue(
        points=jnp.array([[0.0]]),
        lower=jnp.array([-1.0]),
        upper=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
    )
    expected = 4.0 / s3 - 2.0 * jnp.exp(-s3) * (1.0 + 2.0 / s3)
    assert jnp.allclose(result, expected, atol=1e-6)


def test_qkq_matern32_lebesgue_closed_form() -> None:
    r"""``qKq`` for unit Matern-3/2 on ``[-1, 1]``:

    ``c = √3 (b - a)``,
    ``qKq = 2ℓ/3 · (2c - 3ℓ + exp(-c/ℓ)(c + 3ℓ))``.
    """
    c = jnp.sqrt(3.0) * 2.0
    result = qkq_matern32_product_lebesgue(
        lower=jnp.array([-1.0]),
        upper=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
    )
    expected = (2.0 / 3.0) * (2.0 * c - 3.0 + jnp.exp(-c) * (c + 3.0))
    assert jnp.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Matern-5/2 × Lebesgue
# ---------------------------------------------------------------------------


def test_qkq_matern52_lebesgue_closed_form() -> None:
    r"""``qKq`` for unit Matern-5/2 on ``[-1, 1]``:

    ``c = √5 (b - a)``,
    ``qKq = (2ℓ (8c - 15ℓ) + 2 exp(-c/ℓ) (5a² - 10ab + 5b² + 7cℓ + 15ℓ²)) / 15``.
    """
    a = -1.0
    b = 1.0
    c = jnp.sqrt(5.0) * (b - a)
    bracket = 5 * a**2 - 10 * a * b + 5 * b**2 + 7 * c * 1.0 + 15 * 1.0**2
    expected = (2.0 * 1.0 * (8.0 * c - 15.0 * 1.0) + 2.0 * jnp.exp(-c / 1.0) * bracket) / 15.0
    result = qkq_matern52_product_lebesgue(
        lower=jnp.array([-1.0]),
        upper=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
    )
    assert jnp.allclose(result, expected, atol=1e-5)


def test_qk_matern52_lebesgue_is_finite_and_positive() -> None:
    """``qK`` for Matern-5/2 returns finite positive values on the domain."""
    result = qk_matern52_product_lebesgue(
        points=jnp.array([[-0.5], [0.0], [0.5]]),
        lower=jnp.array([-1.0]),
        upper=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
    )
    assert jnp.all(jnp.isfinite(result))
    assert jnp.all(result > 0.0)


# ---------------------------------------------------------------------------
# Brownian × Lebesgue
# ---------------------------------------------------------------------------


def test_qk_brownian_lebesgue_closed_form() -> None:
    r"""For Brownian motion (``k(x,x') = σ² min(x,x')``) on ``[0.5, 2.0]`` at ``x'=1``:

    ``qK(x') = σ² density (b x' - 0.5 x'² - 0.5 a²)
            = 2·1 - 0.5·1 - 0.5·0.25 = 1.375``.
    """
    result = qk_brownian_lebesgue(
        points=jnp.array([[1.0]]),
        lower=jnp.asarray(0.5),
        upper=jnp.asarray(2.0),
        amplitude=jnp.asarray(1.0),
        density=jnp.asarray(1.0),
    )
    assert jnp.allclose(result, 1.375, atol=1e-6)


def test_qkq_brownian_lebesgue_closed_form() -> None:
    r"""``qKq`` for Brownian on ``[0.5, 2.0]``:

    ``qKq = 0.5 b (b² - a²) - (b³ - a³)/6 - 0.5 a² (b - a)
          = 3.75 - 1.3125 - 0.1875 = 2.25``.
    """
    result = qkq_brownian_lebesgue(
        lower=jnp.asarray(0.5),
        upper=jnp.asarray(2.0),
        amplitude=jnp.asarray(1.0),
        density=jnp.asarray(1.0),
    )
    assert jnp.allclose(result, 2.25, atol=1e-6)


def test_qk_brownian_lebesgue_is_zero_at_zero_lower_bound() -> None:
    """At ``x' = a`` (interval start), ``qK(a) = σ² (a(b-a) + 0.5(a²-a²) - 0.5 a²)``.

    For ``a=0``, ``qK(0) = σ² density · 0`` — Brownian motion at the
    origin has zero kernel mean.
    """
    result = qk_brownian_lebesgue(
        points=jnp.array([[0.0]]),
        lower=jnp.asarray(0.0),
        upper=jnp.asarray(1.0),
        amplitude=jnp.asarray(1.0),
        density=jnp.asarray(1.0),
    )
    assert jnp.allclose(result, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Multi-dim product structure tests
# ---------------------------------------------------------------------------


def test_qk_rbf_lebesgue_multi_dim_factorises_over_dimensions() -> None:
    """RBF × Lebesgue factorises: ``qK(x'_1, x'_2) = qK_1(x'_1) · qK_2(x'_2)``."""
    qk_2d = qk_rbf_lebesgue(
        points=jnp.array([[0.0, 0.0]]),
        lower=jnp.array([-1.0, -1.0]),
        upper=jnp.array([1.0, 1.0]),
        lengthscales=jnp.array([1.0, 1.0]),
        amplitude=jnp.asarray(1.0),
        density=jnp.asarray(1.0),
    )
    qk_1d = qk_rbf_lebesgue(
        points=jnp.array([[0.0]]),
        lower=jnp.array([-1.0]),
        upper=jnp.array([1.0]),
        lengthscales=jnp.array([1.0]),
        amplitude=jnp.asarray(1.0),
        density=jnp.asarray(1.0),
    )
    assert jnp.allclose(qk_2d, qk_1d * qk_1d, atol=1e-6)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_all_six_kernel_means_compile_under_jit() -> None:
    """All six ``qK`` functions must compile under ``jax.jit``."""
    points = jnp.array([[0.5]])
    lower = jnp.array([-1.0])
    upper = jnp.array([1.0])
    lengthscales = jnp.array([1.0])
    amplitude = jnp.asarray(1.0)

    jax.jit(qk_rbf_gaussian)(
        points=points,
        measure_mean=jnp.array([0.0]),
        measure_variance=jnp.array([1.0]),
        lengthscales=lengthscales,
        amplitude=amplitude,
    )
    jax.jit(qk_rbf_lebesgue)(
        points=points,
        lower=lower,
        upper=upper,
        lengthscales=lengthscales,
        amplitude=amplitude,
        density=jnp.asarray(1.0),
    )
    jax.jit(qk_matern12_product_lebesgue)(
        points=points,
        lower=lower,
        upper=upper,
        lengthscales=lengthscales,
        amplitude=amplitude,
    )
    jax.jit(qk_matern32_product_lebesgue)(
        points=points,
        lower=lower,
        upper=upper,
        lengthscales=lengthscales,
        amplitude=amplitude,
    )
    jax.jit(qk_matern52_product_lebesgue)(
        points=points,
        lower=lower,
        upper=upper,
        lengthscales=lengthscales,
        amplitude=amplitude,
    )
    jax.jit(qk_brownian_lebesgue)(
        points=jnp.array([[0.7]]),
        lower=jnp.asarray(0.5),
        upper=jnp.asarray(2.0),
        amplitude=amplitude,
        density=jnp.asarray(1.0),
    )
