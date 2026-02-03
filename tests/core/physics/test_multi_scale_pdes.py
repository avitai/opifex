"""Tests for multi-scale PDE types in the registry.

This test suite covers:
1. Homogenization PDE (5 tests)
2. Two-Scale Expansion PDE (5 tests)
3. AMR Poisson PDE (5 tests)
4. Integration tests (3 tests)
5. Performance tests (2 tests)

Total: 20 tests

Following TDD principles: These tests are written FIRST before implementation.
"""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from opifex.core.physics.autodiff_engine import AutoDiffEngine
from opifex.core.physics.pde_registry import PDEResidualRegistry


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_quadratic():
    """Simple quadratic function u(x) = x² + y²."""

    def u(x: Float[Array, "... dim"]) -> Float[Array, "..."]:
        return jnp.sum(x**2, axis=-1)

    return u


@pytest.fixture
def linear_function():
    """Linear function u(x) = x + y."""

    def u(x: Float[Array, "... dim"]) -> Float[Array, "..."]:
        return jnp.sum(x, axis=-1)

    return u


@pytest.fixture
def constant_function():
    """Constant function u(x) = 1."""

    def u(x: Float[Array, "... dim"]) -> Float[Array, "..."]:
        return jnp.ones(x.shape[:-1])

    return u


@pytest.fixture
def test_points_2d():
    """2D test points."""
    return jnp.array([[0.5, 0.5], [1.0, 1.0], [0.0, 0.0]])


# =============================================================================
# Homogenization PDE Tests (5 tests)
# =============================================================================


class TestHomogenizationPDE:
    """Test suite for homogenization PDE: -∇·(a_ε(x)∇u) = f."""

    def test_homogenization_registered(self):
        """Test that homogenization PDE is registered."""
        assert PDEResidualRegistry.contains("homogenization")
        pde_fn = PDEResidualRegistry.get("homogenization")
        assert callable(pde_fn)

    def test_homogenization_homogeneous_coefficient(
        self, simple_quadratic, test_points_2d
    ):
        """Test homogenization with constant coefficient (reduces to Poisson).

        For u = x² + y², ∇²u = 4
        With constant a_ε = 1, -∇·(a_ε∇u) = -∇²u = -4
        Source term f = -4 → residual should be ~0
        """
        pde_fn = PDEResidualRegistry.get("homogenization")

        # Constant coefficient function
        def coeff(x):
            return jnp.ones(x.shape[0])

        source = jnp.full(test_points_2d.shape[0], -4.0)
        residual = pde_fn(
            simple_quadratic,
            test_points_2d,
            AutoDiffEngine,
            coefficient_fn=coeff,
            source_term=source,
        )

        assert residual.shape == (test_points_2d.shape[0],)
        assert jnp.allclose(residual, 0.0, atol=1e-5)

    def test_homogenization_periodic_coefficient(
        self, simple_quadratic, test_points_2d
    ):
        """Test homogenization with periodic coefficient.

        Coefficient: a_ε(x) = 1 + 0.5*cos(2πx)
        This represents a periodic microstructure.
        """
        pde_fn = PDEResidualRegistry.get("homogenization")

        # Periodic coefficient
        def coeff(x):
            return 1.0 + 0.5 * jnp.cos(2 * jnp.pi * x[:, 0])

        residual = pde_fn(
            simple_quadratic,
            test_points_2d,
            AutoDiffEngine,
            coefficient_fn=coeff,
        )

        assert residual.shape == (test_points_2d.shape[0],)
        assert jnp.all(jnp.isfinite(residual))

    def test_homogenization_discontinuous_coefficient(
        self, linear_function, test_points_2d
    ):
        """Test homogenization with discontinuous coefficient (material interface).

        Coefficient: a_ε(x) = 1 if x < 0.5 else 2
        Represents interface between two materials.
        """
        pde_fn = PDEResidualRegistry.get("homogenization")

        # Discontinuous coefficient
        def coeff(x):
            return jnp.where(x[:, 0] < 0.5, 1.0, 2.0)

        residual = pde_fn(
            linear_function,
            test_points_2d,
            AutoDiffEngine,
            coefficient_fn=coeff,
        )

        assert residual.shape == (test_points_2d.shape[0],)
        assert jnp.all(jnp.isfinite(residual))

    def test_homogenization_default_parameters(self, simple_quadratic, test_points_2d):
        """Test homogenization with default parameters (coefficient=1, source=0)."""
        pde_fn = PDEResidualRegistry.get("homogenization")

        residual = pde_fn(
            simple_quadratic,
            test_points_2d,
            AutoDiffEngine,
        )

        # Should equal negative Laplacian
        laplacian = AutoDiffEngine.compute_laplacian(simple_quadratic, test_points_2d)
        expected = -laplacian

        assert residual.shape == (test_points_2d.shape[0],)
        assert jnp.allclose(residual, expected, atol=1e-5)

    def test_homogenization_jit_compatibility(self, simple_quadratic, test_points_2d):
        """Test that homogenization PDE is JIT-compatible."""
        pde_fn = PDEResidualRegistry.get("homogenization")

        @jax.jit
        def compute_residual(x):
            return pde_fn(simple_quadratic, x, AutoDiffEngine)

        residual = compute_residual(test_points_2d)
        assert residual.shape == (test_points_2d.shape[0],)
        assert jnp.all(jnp.isfinite(residual))


# =============================================================================
# Two-Scale Expansion PDE Tests (5 tests)
# =============================================================================


class TestTwoScalePDE:
    """Test suite for two-scale expansion PDE."""

    def test_two_scale_registered(self):
        """Test that two-scale PDE is registered."""
        assert PDEResidualRegistry.contains("two_scale")
        pde_fn = PDEResidualRegistry.get("two_scale")
        assert callable(pde_fn)

    def test_two_scale_decoupled_scales(self, test_points_2d):
        """Test two-scale PDE with decoupled scales (ε=0).

        With ε=0, only macroscale operator L₀ should be active.
        """
        pde_fn = PDEResidualRegistry.get("two_scale")

        # Macroscale and microscale functions
        def u_macro(x):
            return jnp.sum(x**2, axis=-1)

        def u_micro(x):
            return jnp.sin(10 * jnp.sum(x, axis=-1))

        macro_res, micro_res = pde_fn(
            u_macro,
            u_micro,
            test_points_2d,
            AutoDiffEngine,
            epsilon=0.0,
        )

        assert macro_res.shape == (test_points_2d.shape[0],)
        assert micro_res.shape == (test_points_2d.shape[0],)
        assert jnp.all(jnp.isfinite(macro_res))
        assert jnp.all(jnp.isfinite(micro_res))

    def test_two_scale_weak_coupling(self, test_points_2d):
        """Test two-scale PDE with weak coupling (small ε)."""
        pde_fn = PDEResidualRegistry.get("two_scale")

        def u_macro(x):
            return jnp.sum(x**2, axis=-1)

        def u_micro(x):
            return jnp.sin(10 * jnp.sum(x, axis=-1))

        # Test with small epsilon
        macro_res, micro_res = pde_fn(
            u_macro,
            u_micro,
            test_points_2d,
            AutoDiffEngine,
            epsilon=0.01,
        )

        assert macro_res.shape == (test_points_2d.shape[0],)
        assert micro_res.shape == (test_points_2d.shape[0],)

    def test_two_scale_scale_separation(self, test_points_2d):
        """Test that microscale varies faster than macroscale."""
        pde_fn = PDEResidualRegistry.get("two_scale")

        # Slowly varying macroscale
        def u_macro(x):
            return jnp.sum(x, axis=-1)

        # Rapidly varying microscale
        def u_micro(x):
            return jnp.sin(20 * jnp.pi * jnp.sum(x, axis=-1))

        macro_res_small, _ = pde_fn(
            u_macro, u_micro, test_points_2d, AutoDiffEngine, epsilon=0.01
        )

        macro_res_large, _ = pde_fn(
            u_macro, u_micro, test_points_2d, AutoDiffEngine, epsilon=0.1
        )

        # Residuals should differ based on scale parameter
        assert not jnp.allclose(macro_res_small, macro_res_large, atol=1e-10)

    def test_two_scale_return_tuple(self, test_points_2d):
        """Test that two-scale PDE returns tuple of (macro, micro) residuals."""
        pde_fn = PDEResidualRegistry.get("two_scale")

        def u_macro(x):
            return jnp.sum(x**2, axis=-1)

        def u_micro(x):
            return jnp.zeros(x.shape[0])

        result = pde_fn(
            u_macro,
            u_micro,
            test_points_2d,
            AutoDiffEngine,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        macro_res, micro_res = result
        assert macro_res.shape == (test_points_2d.shape[0],)
        assert micro_res.shape == (test_points_2d.shape[0],)

    def test_two_scale_jit_compatibility(self, test_points_2d):
        """Test that two-scale PDE is JIT-compatible."""
        pde_fn = PDEResidualRegistry.get("two_scale")

        def u_macro(x):
            return jnp.sum(x**2, axis=-1)

        def u_micro(x):
            return jnp.sin(10 * jnp.sum(x, axis=-1))

        @jax.jit
        def compute_residuals(x):
            return pde_fn(u_macro, u_micro, x, AutoDiffEngine)

        macro_res, micro_res = compute_residuals(test_points_2d)
        assert jnp.all(jnp.isfinite(macro_res))
        assert jnp.all(jnp.isfinite(micro_res))


# =============================================================================
# AMR Poisson PDE Tests (5 tests)
# =============================================================================


class TestAMRPoissonPDE:
    """Test suite for Poisson PDE with adaptive mesh refinement indicators."""

    def test_amr_poisson_registered(self):
        """Test that AMR Poisson PDE is registered."""
        assert PDEResidualRegistry.contains("amr_poisson")
        pde_fn = PDEResidualRegistry.get("amr_poisson")
        assert callable(pde_fn)

    def test_amr_poisson_smooth_solution(self, simple_quadratic, test_points_2d):
        """Test AMR Poisson with smooth solution (uniform error).

        For smooth u = x² + y², error indicator should be relatively uniform.
        """
        pde_fn = PDEResidualRegistry.get("amr_poisson")

        source = jnp.full(test_points_2d.shape[0], 4.0)
        residual, error_indicator = pde_fn(
            simple_quadratic,
            test_points_2d,
            AutoDiffEngine,
            source_term=source,
        )

        assert residual.shape == (test_points_2d.shape[0],)
        assert error_indicator.shape == (test_points_2d.shape[0],)

        # Residual should be ~0 for correct source term
        assert jnp.allclose(residual, 0.0, atol=1e-5)

        # Error indicator should be non-negative
        assert jnp.all(error_indicator >= 0.0)

    def test_amr_poisson_singular_solution(self, test_points_2d):
        """Test AMR Poisson with singular solution (high error at singularity).

        Function with singularity: u(x) = 1/||x||
        Should have high error indicator near origin.
        """
        pde_fn = PDEResidualRegistry.get("amr_poisson")

        def singular_u(x):
            r = jnp.sqrt(jnp.sum(x**2, axis=-1) + 1e-10)
            return 1.0 / r

        _, error_indicator = pde_fn(
            singular_u,
            test_points_2d,
            AutoDiffEngine,
        )

        # Error indicator should be highest near origin
        origin_idx = 2  # [0.0, 0.0] is at index 2
        assert error_indicator[origin_idx] > error_indicator[1]  # Higher than [1, 1]

    def test_amr_poisson_error_indicator_computation(
        self, simple_quadratic, test_points_2d
    ):
        """Test that error indicator is computed correctly.

        Error indicator = ||∇u|| + ||H||_F
        where H is Hessian matrix.
        """
        pde_fn = PDEResidualRegistry.get("amr_poisson")

        _, error_indicator = pde_fn(
            simple_quadratic,
            test_points_2d,
            AutoDiffEngine,
        )

        # Manually compute expected error indicator
        grad = AutoDiffEngine.compute_gradient(simple_quadratic, test_points_2d)
        grad_norm = jnp.linalg.norm(grad, axis=-1)

        hess = AutoDiffEngine.compute_hessian(simple_quadratic, test_points_2d)
        hess_norm = jnp.linalg.norm(hess.reshape(hess.shape[0], -1), axis=-1)

        expected_indicator = grad_norm + hess_norm

        assert jnp.allclose(error_indicator, expected_indicator, atol=1e-5)

    def test_amr_poisson_return_tuple(self, simple_quadratic, test_points_2d):
        """Test that AMR Poisson returns tuple of (residual, error_indicator)."""
        pde_fn = PDEResidualRegistry.get("amr_poisson")

        result = pde_fn(
            simple_quadratic,
            test_points_2d,
            AutoDiffEngine,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        residual, error_indicator = result
        assert residual.shape == (test_points_2d.shape[0],)
        assert error_indicator.shape == (test_points_2d.shape[0],)

    def test_amr_poisson_jit_compatibility(self, simple_quadratic, test_points_2d):
        """Test that AMR Poisson is JIT-compatible."""
        pde_fn = PDEResidualRegistry.get("amr_poisson")

        @jax.jit
        def compute_with_indicator(x):
            return pde_fn(simple_quadratic, x, AutoDiffEngine)

        residual, error_indicator = compute_with_indicator(test_points_2d)
        assert jnp.all(jnp.isfinite(residual))
        assert jnp.all(jnp.isfinite(error_indicator))


# =============================================================================
# Integration Tests (3 tests)
# =============================================================================


class TestMultiScaleIntegration:
    """Integration tests for multi-scale PDE workflows."""

    def test_homogenization_workflow(self, test_points_2d):
        """End-to-end test of homogenization workflow.

        1. Define problem with periodic microstructure
        2. Compute residuals
        3. Verify physically meaningful results
        """
        pde_fn = PDEResidualRegistry.get("homogenization")

        # Define problem
        def u(x):
            return jnp.sum(x**2, axis=-1)

        def periodic_coeff(x):
            return 1.0 + 0.5 * jnp.cos(4 * jnp.pi * x[:, 0])

        # Compute
        residual = pde_fn(
            u, test_points_2d, AutoDiffEngine, coefficient_fn=periodic_coeff
        )

        # Verify
        assert residual.shape == (test_points_2d.shape[0],)
        assert jnp.all(jnp.isfinite(residual))

    def test_two_scale_coupling_verification(self, test_points_2d):
        """Verify coupling between macro and micro scales."""
        pde_fn = PDEResidualRegistry.get("two_scale")

        def u_macro(x):
            return jnp.sum(x, axis=-1)

        def u_micro(x):
            return jnp.sin(20 * jnp.pi * x[:, 0])

        # Compare different epsilon values
        eps_small = 0.01
        eps_large = 0.1

        macro_small, _ = pde_fn(
            u_macro, u_micro, test_points_2d, AutoDiffEngine, epsilon=eps_small
        )

        macro_large, _ = pde_fn(
            u_macro, u_micro, test_points_2d, AutoDiffEngine, epsilon=eps_large
        )

        # Verify coupling effect
        assert not jnp.allclose(macro_small, macro_large)

    def test_amr_refinement_cycle(self, test_points_2d):
        """Test complete AMR cycle: compute → identify → refine."""
        pde_fn = PDEResidualRegistry.get("amr_poisson")

        def u(x):
            # Non-uniform function
            return jnp.exp(-jnp.sum(x**2, axis=-1))

        # Compute residual and error
        _, error_indicator = pde_fn(u, test_points_2d, AutoDiffEngine)

        # Identify refinement zones (top 50%)
        threshold = jnp.percentile(error_indicator, 50)
        needs_refinement = error_indicator > threshold

        # Verify at least some points need refinement
        assert jnp.any(needs_refinement)
        assert not jnp.all(needs_refinement)


# =============================================================================
# Performance Tests (2 tests)
# =============================================================================


class TestMultiScalePerformance:
    """Performance tests for multi-scale PDEs."""

    def test_jit_compilation_all_pdes(self, test_points_2d):
        """Test JIT compilation for all 3 multi-scale PDEs."""

        # Homogenization
        hom_fn = PDEResidualRegistry.get("homogenization")

        @jax.jit
        def jit_hom(x):
            def u(x_):
                return jnp.sum(x_**2, axis=-1)

            return hom_fn(u, x, AutoDiffEngine)

        res_hom = jit_hom(test_points_2d)
        assert jnp.all(jnp.isfinite(res_hom))

        # Two-scale
        ts_fn = PDEResidualRegistry.get("two_scale")

        @jax.jit
        def jit_ts(x):
            def u_macro(x_):
                return jnp.sum(x_**2, axis=-1)

            def u_micro(x_):
                return jnp.zeros(x_.shape[0])

            return ts_fn(u_macro, u_micro, x, AutoDiffEngine)

        macro_res, micro_res = jit_ts(test_points_2d)
        assert jnp.all(jnp.isfinite(macro_res))
        assert jnp.all(jnp.isfinite(micro_res))

        # AMR Poisson
        amr_fn = PDEResidualRegistry.get("amr_poisson")

        @jax.jit
        def jit_amr(x):
            def u(x_):
                return jnp.sum(x_**2, axis=-1)

            return amr_fn(u, x, AutoDiffEngine)

        residual, error = jit_amr(test_points_2d)
        assert jnp.all(jnp.isfinite(residual))
        assert jnp.all(jnp.isfinite(error))

    def test_vmap_compatibility(self):
        """Test VMAP compatibility for batched computations."""
        hom_fn = PDEResidualRegistry.get("homogenization")

        def u(x):
            return jnp.sum(x**2, axis=-1)

        # Create batch of different point sets
        batch_points = jnp.array(
            [
                [[0.5, 0.5], [1.0, 1.0]],
                [[0.0, 0.0], [0.25, 0.25]],
            ]
        )

        # VMAP over batch dimension
        batch_residuals = jax.vmap(lambda x: hom_fn(u, x, AutoDiffEngine))(batch_points)

        assert batch_residuals.shape == (2, 2)  # (batch, points)
        assert jnp.all(jnp.isfinite(batch_residuals))
