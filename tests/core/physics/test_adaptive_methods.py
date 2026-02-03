"""Tests for adaptive method utilities.

This test suite covers:
1. Error estimation functions (9 tests)
   - compute_residual_error (3 tests)
   - compute_gradient_error (3 tests including JIT)
   - compute_hessian_error (3 tests including JIT)
2. Mesh refinement indicators (3 tests)
   - identify_refinement_zones (3 tests)

Total: 12 tests

Following TDD principles: These tests are written FIRST before implementation.

All methods are based on well-established literature:
- Residual-based error: Direct PDE residual magnitude
- Gradient-based error: Gradient magnitude (Zienkiewicz-Zhu approach)
- Hessian-based error: Frobenius norm of Hessian (curvature indicator)
"""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from opifex.core.physics.adaptive_methods import (
    compute_gradient_error,
    compute_hessian_error,
    compute_residual_error,
    identify_refinement_zones,
)
from opifex.core.physics.autodiff_engine import AutoDiffEngine


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_points_2d():
    """2D test points."""
    return jnp.array([[0.5, 0.5], [1.0, 1.0], [0.0, 0.0], [0.25, 0.75]])


@pytest.fixture
def constant_function():
    """Constant function u(x) = 5."""

    def u(x: Float[Array, "... dim"]) -> Float[Array, "..."]:
        return 5.0 * jnp.ones(x.shape[:-1])

    return u


@pytest.fixture
def linear_function():
    """Linear function u(x) = 2x + 3y."""

    def u(x: Float[Array, "... dim"]) -> Float[Array, "..."]:
        return 2.0 * x[..., 0] + 3.0 * x[..., 1]

    return u


@pytest.fixture
def quadratic_function():
    """Quadratic function u(x) = x² + y²."""

    def u(x: Float[Array, "... dim"]) -> Float[Array, "..."]:
        return jnp.sum(x**2, axis=-1)

    return u


@pytest.fixture
def exponential_function():
    """Exponential decay: u(x) = exp(-||x||²)."""

    def u(x: Float[Array, "... dim"]) -> Float[Array, "..."]:
        r_squared = jnp.sum(x**2, axis=-1)
        return jnp.exp(-r_squared)

    return u


@pytest.fixture
def periodic_function():
    """Periodic function: u(x) = sin(2πx)."""

    def u(x: Float[Array, "... dim"]) -> Float[Array, "..."]:
        return jnp.sin(2 * jnp.pi * x[..., 0])

    return u


# =============================================================================
# Error Estimation Tests - compute_residual_error (3 tests)
# =============================================================================


class TestResidualError:
    """Test suite for residual-based error estimation."""

    def test_zero_residual_zero_error(self, quadratic_function, test_points_2d):
        """Test that zero residual gives zero error.

        For a perfect solution, residual = 0 → error = 0.
        """
        zero_residual = jnp.zeros(test_points_2d.shape[0])

        error = compute_residual_error(
            quadratic_function,
            test_points_2d,
            zero_residual,
        )

        assert error.shape == (test_points_2d.shape[0],)
        assert jnp.allclose(error, 0.0, atol=1e-10)

    def test_large_residual_large_error(self, quadratic_function, test_points_2d):
        """Test that large residual gives large error.

        Error should scale with residual magnitude.
        """
        large_residual = jnp.full(test_points_2d.shape[0], 100.0)

        error = compute_residual_error(
            quadratic_function,
            test_points_2d,
            large_residual,
        )

        assert error.shape == (test_points_2d.shape[0],)
        assert jnp.all(error > 0.0)
        # Error should be proportional to residual
        assert jnp.allclose(error, jnp.abs(large_residual), atol=1e-5)

    def test_residual_error_scaling(self, quadratic_function, test_points_2d):
        """Test scaling properties of residual error.

        Doubling residual should double error.
        """
        residual_1x = jnp.array([1.0, 2.0, 3.0, 4.0])
        residual_2x = 2.0 * residual_1x

        error_1x = compute_residual_error(
            quadratic_function,
            test_points_2d,
            residual_1x,
        )

        error_2x = compute_residual_error(
            quadratic_function,
            test_points_2d,
            residual_2x,
        )

        assert jnp.allclose(error_2x, 2.0 * error_1x, atol=1e-5)


# =============================================================================
# Error Estimation Tests - compute_gradient_error (2 tests)
# =============================================================================


class TestGradientError:
    """Test suite for gradient-based error estimation."""

    def test_constant_function_zero_gradient(self, constant_function, test_points_2d):
        """Test that constant function has zero gradient error.

        For u = const, ∇u = 0 → gradient error = 0.
        """
        error = compute_gradient_error(
            constant_function,
            test_points_2d,
            AutoDiffEngine,
        )

        assert error.shape == (test_points_2d.shape[0],)
        assert jnp.allclose(error, 0.0, atol=1e-5)

    def test_linear_function_constant_gradient(self, linear_function, test_points_2d):
        """Test that linear function has constant gradient error.

        For u = 2x + 3y, ∇u = [2, 3] → ||∇u|| = √13 everywhere.
        """
        error = compute_gradient_error(
            linear_function,
            test_points_2d,
            AutoDiffEngine,
        )

        expected_norm = jnp.sqrt(2.0**2 + 3.0**2)  # √13 ≈ 3.606

        assert error.shape == (test_points_2d.shape[0],)
        assert jnp.allclose(error, expected_norm, atol=1e-5)

    def test_gradient_error_jit_compatible(self, quadratic_function, test_points_2d):
        """Test that gradient error computation is JIT-compatible."""

        @jax.jit
        def compute_error(x):
            return compute_gradient_error(quadratic_function, x, AutoDiffEngine)

        error = compute_error(test_points_2d)
        assert error.shape == (test_points_2d.shape[0],)
        assert jnp.all(jnp.isfinite(error))


# =============================================================================
# Error Estimation Tests - compute_hessian_error (2 tests)
# =============================================================================


class TestHessianError:
    """Test suite for curvature-based error estimation."""

    def test_quadratic_function_constant_hessian(
        self, quadratic_function, test_points_2d
    ):
        """Test that quadratic function has constant Hessian.

        For u = x² + y², H = [[2, 0], [0, 2]] everywhere.
        ||H||_F = √(4 + 4) = √8 ≈ 2.828
        """
        error = compute_hessian_error(
            quadratic_function,
            test_points_2d,
            AutoDiffEngine,
        )

        # Frobenius norm of [[2, 0], [0, 2]] is sqrt(2^2 + 2^2) = sqrt(8)
        expected_norm = jnp.sqrt(8.0)

        assert error.shape == (test_points_2d.shape[0],)
        assert jnp.allclose(error, expected_norm, atol=1e-4)

    def test_hessian_error_varying_curvature(
        self, exponential_function, test_points_2d
    ):
        """Test Hessian error for function with varying curvature.

        Exponential function has higher curvature near origin.
        """
        error = compute_hessian_error(
            exponential_function,
            test_points_2d,
            AutoDiffEngine,
        )

        assert error.shape == (test_points_2d.shape[0],)
        assert jnp.all(error >= 0.0)

        # Error should be highest near origin [0, 0]
        origin_idx = 2
        far_idx = 1  # [1, 1] is far from origin
        assert error[origin_idx] > error[far_idx]

    def test_hessian_error_jit_compatible(self, quadratic_function, test_points_2d):
        """Test that Hessian error computation is JIT-compatible."""

        @jax.jit
        def compute_error(x):
            return compute_hessian_error(quadratic_function, x, AutoDiffEngine)

        error = compute_error(test_points_2d)
        assert error.shape == (test_points_2d.shape[0],)
        assert jnp.all(jnp.isfinite(error))


# =============================================================================
# Mesh Refinement Tests - identify_refinement_zones (2 tests)
# =============================================================================


class TestRefinementZones:
    """Test suite for mesh refinement zone identification."""

    def test_no_refinement_below_threshold(self):
        """Test that no points are refined when all errors are below threshold."""
        error_indicator = jnp.array([0.05, 0.08, 0.03, 0.09])
        threshold = 0.1

        needs_refinement = identify_refinement_zones(
            error_indicator,
            threshold=threshold,
        )

        assert needs_refinement.shape == error_indicator.shape
        assert needs_refinement.dtype == bool
        assert not jnp.any(needs_refinement)  # All False

    def test_correct_mask_above_threshold(self):
        """Test that correct points are identified for refinement."""
        error_indicator = jnp.array([0.05, 0.15, 0.03, 0.25])
        threshold = 0.1

        needs_refinement = identify_refinement_zones(
            error_indicator,
            threshold=threshold,
        )

        expected_mask = jnp.array([False, True, False, True])

        assert needs_refinement.shape == error_indicator.shape
        assert jnp.array_equal(needs_refinement, expected_mask)

    def test_percentile_based_selection(self):
        """Test percentile-based refinement selection.

        With percentile=75, top 25% of points should be refined.
        """
        error_indicator = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        # Top 25% means 2 out of 8 points
        needs_refinement = identify_refinement_zones(
            error_indicator,
            threshold=0.0,  # Use percentile instead
            percentile=75.0,
        )

        # Should refine points with errors > 75th percentile (0.625)
        # That's the last two points: 0.7, 0.8
        assert jnp.sum(needs_refinement) == 2
        assert needs_refinement[-2]  # 0.7 should be refined
        assert needs_refinement[-1]  # 0.8 should be refined
