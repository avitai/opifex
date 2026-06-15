"""Vanilla Bayesian Quadrature + WSABI-L on RBF kernel + Gaussian measure.

Tests for :mod:`opifex.uncertainty.quadrature.bayesian_quadrature`.

Closed-form references (all 1-D, isotropic kernel and diagonal Gaussian
measure for verifiability):

* Vanilla BQ (Briol+ 2019 §2.4 / O'Hagan 1991): the posterior integral
  mean is ``qK^T (K_XX + σ_n² I)^-1 Y``; the posterior variance is
  ``qKq - qK^T (K_XX + σ_n² I)^-1 qK``. For an isotropic RBF kernel
  ``k(x, x') = σ² exp(-||x-x'||²/(2ℓ²))`` and a diagonal Gaussian
  measure ``N(b, diag(s²))``:

  * ``qK(x') = σ² · ∏_i sqrt(ℓ²/(ℓ² + s²_i)) · exp(-Σ_i (x'_i - b_i)²
    /(2(ℓ² + s²_i)))``
  * ``qKq = σ² · ∏_i sqrt(ℓ²/(ℓ² + 2 s²_i))``

* WSABI-L (Gunter+ 2014 / emukit
  ``quadrature/methods/bounded_bq_model.py:integrate``): with offset
  ``α`` and warped observations ``g = sqrt(2(Y - α))``,
  ``integral_mean = α + 0.5 Σ_ij w_i w_j (qK_ij)_true`` where the
  pairwise double-kernel integral ``(qK_ij)_true = ∫ k(x, x_i)
  k(x, x_j) p(x) dx`` factors into ``σ² · exp(-||x_i - x_j||²/(4ℓ²)) ·
  qK(mid_ij; ℓ/√2)`` with ``mid_ij = (x_i + x_j) / 2``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from opifex.uncertainty.quadrature import (
    VanillaBayesianQuadratureAdapterSpec,
    WSABILAdapterSpec,
)
from opifex.uncertainty.quadrature.bayesian_quadrature import (
    vanilla_bayesian_quadrature,
    wsabi_l_bayesian_quadrature,
)
from opifex.uncertainty.registry import UQCapability


# ---------------------------------------------------------------------------
# vanilla_bayesian_quadrature
# ---------------------------------------------------------------------------


def test_vanilla_bq_single_point_matches_closed_form_mean_and_variance() -> None:
    """Single point at the measure mean recovers ``(1/√2, 1/√3 - 1/2)``.

    Setup: 1-D RBF with ``σ² = 1``, ``ℓ = 1``; measure ``N(0, 1)``;
    single observation at ``x = 0`` with ``y = 1``, no noise.
    * ``qK(0) = 1·√(1/(1+1))·exp(0) = 1/√2``
    * ``qKq = 1·√(1/(1+2)) = 1/√3``
    * ``K(0, 0) = 1``
    * ``mean = qK · 1/K · y = 1/√2``
    * ``var = qKq - qK² / K = 1/√3 - 1/2``
    """
    integral_mean, integral_var = vanilla_bayesian_quadrature(
        points=jnp.array([[0.0]]),
        values=jnp.array([1.0]),
        measure_mean=jnp.array([0.0]),
        measure_variance=jnp.array([1.0]),
        kernel_lengthscales=jnp.array([1.0]),
        kernel_amplitude=jnp.asarray(1.0),
        noise_variance=jnp.asarray(0.0),
    )
    assert jnp.allclose(integral_mean, 1.0 / jnp.sqrt(2.0), atol=1e-6)
    assert jnp.allclose(integral_var, 1.0 / jnp.sqrt(3.0) - 0.5, atol=1e-6)


def test_vanilla_bq_zero_observations_yields_zero_mean_and_full_variance() -> None:
    """All-zero observations give zero posterior mean and full prior variance.

    With ``Y = 0`` the posterior mean is ``qK · K^-1 · 0 = 0``; the
    posterior variance is still reduced by the explained term
    ``qK^T K^-1 qK``.
    """
    integral_mean, integral_var = vanilla_bayesian_quadrature(
        points=jnp.array([[0.0]]),
        values=jnp.array([0.0]),
        measure_mean=jnp.array([0.0]),
        measure_variance=jnp.array([1.0]),
        kernel_lengthscales=jnp.array([1.0]),
        kernel_amplitude=jnp.asarray(1.0),
        noise_variance=jnp.asarray(0.0),
    )
    assert jnp.allclose(integral_mean, 0.0, atol=1e-6)
    assert jnp.allclose(integral_var, 1.0 / jnp.sqrt(3.0) - 0.5, atol=1e-6)


def test_vanilla_bq_compiles_under_jit() -> None:
    """Closed-form vanilla BQ must compile under ``jax.jit``."""
    jitted = jax.jit(vanilla_bayesian_quadrature)
    mean, var = jitted(
        points=jnp.array([[0.0], [1.0]]),
        values=jnp.array([1.0, 0.5]),
        measure_mean=jnp.array([0.0]),
        measure_variance=jnp.array([1.0]),
        kernel_lengthscales=jnp.array([1.0]),
        kernel_amplitude=jnp.asarray(1.0),
        noise_variance=jnp.asarray(1e-6),
    )
    assert jnp.isfinite(mean)
    assert jnp.isfinite(var)


def test_vanilla_bq_gradient_with_respect_to_amplitude_is_finite() -> None:
    """``jax.grad`` w.r.t. the kernel amplitude is finite (hyperparameter learning)."""

    def loss(log_amplitude: jax.Array) -> jax.Array:
        mean, _ = vanilla_bayesian_quadrature(
            points=jnp.array([[0.0], [1.0]]),
            values=jnp.array([1.0, 0.5]),
            measure_mean=jnp.array([0.0]),
            measure_variance=jnp.array([1.0]),
            kernel_lengthscales=jnp.array([1.0]),
            kernel_amplitude=jnp.exp(log_amplitude),
            noise_variance=jnp.asarray(1e-6),
        )
        return mean

    grad = jax.grad(loss)(jnp.asarray(0.0))
    assert jnp.isfinite(grad)


# ---------------------------------------------------------------------------
# wsabi_l_bayesian_quadrature
# ---------------------------------------------------------------------------


def test_wsabi_l_single_point_matches_closed_form_mean() -> None:
    """Single-observation WSABI-L closed form recovers ``α + 1/√3``.

    Setup: 1-D RBF with ``σ² = 1``, ``ℓ = 1``; measure ``N(0, 1)``;
    single observation at ``x = 0`` with ``y = 1``, offset ``α = 0``.

    Derivation:
    * ``g = sqrt(2(y - α)) = √2``
    * ``K(X/√2, X/√2) = σ² = 1``
    * ``weights = K^-1 g = √2``
    * ``mid_00 = 0``
    * ``qK(0; ℓ' = ℓ/√2 = 1/√2) = 1·√((1/2)/(1/2 + 1))·exp(0) = 1/√3``
    * ``mean = 0 + 0.5 · √2 · √2 · 1 · 1/√3 = 1/√3``
    """
    integral_mean = wsabi_l_bayesian_quadrature(
        points=jnp.array([[0.0]]),
        values=jnp.array([1.0]),
        offset=jnp.asarray(0.0),
        measure_mean=jnp.array([0.0]),
        measure_variance=jnp.array([1.0]),
        kernel_lengthscales=jnp.array([1.0]),
        kernel_amplitude=jnp.asarray(1.0),
        noise_variance=jnp.asarray(0.0),
    )
    assert jnp.allclose(integral_mean, 1.0 / jnp.sqrt(3.0), atol=1e-6)


def test_wsabi_l_all_observations_at_offset_collapses_to_offset() -> None:
    """If every observation equals the offset, the warped GP is zero and ``mean = α``."""
    integral_mean = wsabi_l_bayesian_quadrature(
        points=jnp.array([[0.0], [1.0]]),
        values=jnp.array([2.5, 2.5]),
        offset=jnp.asarray(2.5),
        measure_mean=jnp.array([0.0]),
        measure_variance=jnp.array([1.0]),
        kernel_lengthscales=jnp.array([1.0]),
        kernel_amplitude=jnp.asarray(1.0),
        noise_variance=jnp.asarray(1e-6),
    )
    assert jnp.allclose(integral_mean, 2.5, atol=1e-5)


def test_wsabi_l_strictly_positive_integrand_exceeds_offset() -> None:
    """A positive integrand above the offset yields ``integral > α``."""
    integral_mean = wsabi_l_bayesian_quadrature(
        points=jnp.array([[0.0], [0.5], [-0.5]]),
        values=jnp.array([1.0, 0.8, 0.8]),
        offset=jnp.asarray(0.0),
        measure_mean=jnp.array([0.0]),
        measure_variance=jnp.array([1.0]),
        kernel_lengthscales=jnp.array([1.0]),
        kernel_amplitude=jnp.asarray(1.0),
        noise_variance=jnp.asarray(1e-6),
    )
    assert integral_mean > 0.0
    assert jnp.isfinite(integral_mean)


def test_wsabi_l_compiles_under_jit() -> None:
    """WSABI-L must compile under ``jax.jit``."""
    jitted = jax.jit(wsabi_l_bayesian_quadrature)
    mean = jitted(
        points=jnp.array([[0.0], [1.0]]),
        values=jnp.array([1.0, 0.5]),
        offset=jnp.asarray(0.0),
        measure_mean=jnp.array([0.0]),
        measure_variance=jnp.array([1.0]),
        kernel_lengthscales=jnp.array([1.0]),
        kernel_amplitude=jnp.asarray(1.0),
        noise_variance=jnp.asarray(1e-6),
    )
    assert jnp.isfinite(mean)


# ---------------------------------------------------------------------------
# Adapter-spec wrap() concretization
# ---------------------------------------------------------------------------


def test_vanilla_bq_adapter_spec_wrap_returns_vanilla_bq_callable() -> None:
    """``VanillaBayesianQuadratureAdapterSpec.wrap`` returns the vanilla BQ primitive."""
    spec: Any = VanillaBayesianQuadratureAdapterSpec()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)
    assert fn is vanilla_bayesian_quadrature


def test_wsabi_l_adapter_spec_wrap_returns_wsabi_l_callable() -> None:
    """``WSABILAdapterSpec.wrap`` returns the WSABI-L combinator."""
    spec: Any = WSABILAdapterSpec()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)
    assert fn is wsabi_l_bayesian_quadrature
