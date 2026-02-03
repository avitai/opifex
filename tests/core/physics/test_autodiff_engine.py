"""Tests for AutoDiffEngine - centralized autodiff utilities.

Following strict TDD - tests written FIRST to define expected behavior.

The AutoDiffEngine provides a DRY (Don't Repeat Yourself) utility for
computing spatial derivatives using JAX autodiff. This eliminates code
duplication across different PDE types and provides a clean, extensible
API for users to build custom physics constraints.

Key principles:
- All methods are static (no state)
- All methods are JIT-compatible
- Full type hints for IDE support
- Works with any JAX-compatible model
"""

import jax
import jax.numpy as jnp

from opifex.core.physics import autodiff_engine


class TestBasicDerivatives:
    """Test basic derivative computations."""

    def test_gradient_linear_function(self):
        """Gradient of ax + by should be [a, b]."""

        def f(x):
            return 3.0 * x[..., 0] + 4.0 * x[..., 1]

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        grad = autodiff_engine.compute_gradient(f, x)
        expected = jnp.array([[3.0, 4.0], [3.0, 4.0]])
        assert jnp.allclose(grad, expected, atol=1e-6)

    def test_gradient_quadratic_function(self):
        """Gradient of x² + y² should be [2x, 2y]."""

        def f(x):
            return x[..., 0] ** 2 + x[..., 1] ** 2

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        grad = autodiff_engine.compute_gradient(f, x)
        expected = jnp.array([[2.0, 4.0], [6.0, 8.0]])
        assert jnp.allclose(grad, expected, atol=1e-6)

    def test_gradient_multidimensional(self):
        """Gradient should work for 3D inputs."""

        def f(x):
            return x[..., 0] ** 2 + x[..., 1] ** 2 + x[..., 2] ** 2

        x = jnp.array([[1.0, 2.0, 3.0]])

        grad = autodiff_engine.compute_gradient(f, x)
        expected = jnp.array([[2.0, 4.0, 6.0]])
        assert jnp.allclose(grad, expected, atol=1e-6)


class TestLaplacianOperator:
    """Test Laplacian (∇²) computation."""

    def test_laplacian_quadratic_2d(self):
        """Laplacian of x² + y² should be 4 (constant)."""

        def f(x):
            return x[..., 0] ** 2 + x[..., 1] ** 2

        x = jnp.array([[1.0, 1.0], [0.5, 0.5], [2.0, 3.0]])

        laplacian = autodiff_engine.compute_laplacian(f, x)
        expected = jnp.full((3,), 4.0)  # Constant value
        assert jnp.allclose(laplacian, expected, atol=1e-6)

    def test_laplacian_quadratic_3d(self):
        """Laplacian of x² + y² + z² should be 6."""

        def f(x):
            return x[..., 0] ** 2 + x[..., 1] ** 2 + x[..., 2] ** 2

        x = jnp.array([[1.0, 1.0, 1.0]])

        laplacian = autodiff_engine.compute_laplacian(f, x)
        expected = jnp.array([6.0])
        assert jnp.allclose(laplacian, expected, atol=1e-6)

    def test_laplacian_gaussian(self):
        """Laplacian of Gaussian exp(-r²) should match analytical."""

        def f(x):
            r_squared = jnp.sum(x**2, axis=-1)
            return jnp.exp(-r_squared)

        x = jnp.array([[0.5, 0.5]])

        # Analytical: ∇²(exp(-r²)) = exp(-r²)(4r² - 2n) where n is dimension
        r_squared = jnp.sum(x**2, axis=-1)
        n_dim = x.shape[-1]
        expected = jnp.exp(-r_squared) * (4 * r_squared - 2 * n_dim)

        laplacian = autodiff_engine.compute_laplacian(f, x)
        assert jnp.allclose(laplacian, expected, atol=1e-5)

    def test_laplacian_harmonic_function(self):
        """Laplacian of harmonic function should be zero."""

        # Harmonic function in 2D: f(x,y) = x² - y²
        def f(x):
            return x[..., 0] ** 2 - x[..., 1] ** 2

        x = jnp.array([[1.0, 1.0], [2.0, 3.0]])

        laplacian = autodiff_engine.compute_laplacian(f, x)
        expected = jnp.zeros(2)  # Should be zero everywhere
        assert jnp.allclose(laplacian, expected, atol=1e-6)


class TestDivergenceOperator:
    """Test divergence (∇·F) computation for vector fields."""

    def test_divergence_constant_field(self):
        """Divergence of constant field should be zero."""

        def vector_field(x):
            # F = [1, 1]
            return jnp.ones_like(x)

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        div = autodiff_engine.compute_divergence(vector_field, x)
        expected = jnp.zeros(2)
        assert jnp.allclose(div, expected, atol=1e-6)

    def test_divergence_linear_field(self):
        """Divergence of F = [x, y] should be 2."""

        def vector_field(x):
            return x  # F = [x, y]

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        div = autodiff_engine.compute_divergence(vector_field, x)
        expected = jnp.full(2, 2.0)  # ∂x/∂x + ∂y/∂y = 1 + 1 = 2
        assert jnp.allclose(div, expected, atol=1e-6)

    def test_divergence_radial_field(self):
        """Divergence of radial field F = r should match analytical."""

        def vector_field(x):
            # F = [x, y] (radial outward)
            return x

        x = jnp.array([[2.0, 3.0]])

        # For F = [x, y]: ∇·F = 2 (in 2D)
        div = autodiff_engine.compute_divergence(vector_field, x)
        expected = jnp.array([2.0])
        assert jnp.allclose(div, expected, atol=1e-6)


class TestHessianOperator:
    """Test Hessian (matrix of second derivatives) computation."""

    def test_hessian_quadratic(self):
        """Hessian of x² + y² should be 2I (identity matrix)."""

        def f(x):
            return x[..., 0] ** 2 + x[..., 1] ** 2

        x = jnp.array([[1.0, 2.0]])

        hessian = autodiff_engine.compute_hessian(f, x)
        expected = jnp.array([[[2.0, 0.0], [0.0, 2.0]]])  # 2*I
        assert jnp.allclose(hessian, expected, atol=1e-6)

    def test_hessian_mixed_derivatives(self):
        """Hessian should capture mixed derivatives correctly."""

        def f(x):
            return x[..., 0] * x[..., 1]  # f = xy

        x = jnp.array([[1.0, 2.0]])

        # Analytical Hessian: [[0, 1], [1, 0]]
        hessian = autodiff_engine.compute_hessian(f, x)
        expected = jnp.array([[[0.0, 1.0], [1.0, 0.0]]])
        assert jnp.allclose(hessian, expected, atol=1e-6)


class TestJITCompatibility:
    """Test that all autodiff operations are JIT-compatible."""

    def test_gradient_jit(self):
        """Gradient is JIT-compiled and works correctly."""

        def f(x):
            return jnp.sum(x**2, axis=-1)

        # Functions are JIT-compiled with static_argnums
        x = jnp.array([[1.0, 2.0]])
        grad = autodiff_engine.compute_gradient(f, x)
        assert jnp.allclose(grad, jnp.array([[2.0, 4.0]]), atol=1e-6)

        # Should work on different data (JIT cache reuses compilation)
        x2 = jnp.array([[3.0, 4.0]])
        grad2 = autodiff_engine.compute_gradient(f, x2)
        assert jnp.allclose(grad2, jnp.array([[6.0, 8.0]]), atol=1e-6)

    def test_laplacian_jit(self):
        """Laplacian is JIT-compiled and works correctly."""

        def f(x):
            return jnp.sum(x**2, axis=-1)

        # Functions are JIT-compiled with static_argnums
        x = jnp.array([[1.0, 2.0]])
        laplacian = autodiff_engine.compute_laplacian(f, x)
        assert jnp.allclose(laplacian, jnp.array([4.0]), atol=1e-6)

        # Should work on different data (JIT cache reuses compilation)
        x2 = jnp.array([[2.0, 3.0]])
        laplacian2 = autodiff_engine.compute_laplacian(f, x2)
        assert jnp.allclose(laplacian2, jnp.array([4.0]), atol=1e-6)


class TestBatchedOperations:
    """Test autodiff operations on batched inputs."""

    def test_gradient_batched(self):
        """Gradient should work on batched inputs."""

        def f(x):
            return x[..., 0] ** 2 + x[..., 1] ** 2

        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        grad = autodiff_engine.compute_gradient(f, x)
        expected = jnp.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
        assert grad.shape == (3, 2)
        assert jnp.allclose(grad, expected, atol=1e-6)

    def test_laplacian_batched(self):
        """Laplacian should work on batched inputs."""

        def f(x):
            return x[..., 0] ** 2 + x[..., 1] ** 2

        x = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        laplacian = autodiff_engine.compute_laplacian(f, x)
        expected = jnp.array([4.0, 4.0, 4.0])
        assert laplacian.shape == (3,)
        assert jnp.allclose(laplacian, expected, atol=1e-6)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_point_input(self):
        """AutoDiff should work with single point."""

        def f(x):
            return jnp.sum(x**2, axis=-1)

        x = jnp.array([[1.0, 2.0]])

        grad = autodiff_engine.compute_gradient(f, x)
        assert grad.shape == (1, 2)
        assert jnp.allclose(grad, jnp.array([[2.0, 4.0]]), atol=1e-6)

    def test_1d_input(self):
        """AutoDiff should handle 1D inputs."""

        def f(x):
            return x[..., 0] ** 2

        x = jnp.array([[2.0]])

        laplacian = autodiff_engine.compute_laplacian(f, x)
        expected = jnp.array([2.0])  # d²/dx²(x²) = 2
        assert jnp.allclose(laplacian, expected, atol=1e-6)

    def test_high_dimensional(self):
        """AutoDiff should work in high dimensions."""

        def f(x):
            return jnp.sum(x**2, axis=-1)

        x = jax.random.normal(jax.random.PRNGKey(0), (5, 10))  # 10D

        laplacian = autodiff_engine.compute_laplacian(f, x)
        expected = jnp.full(5, 20.0)  # Sum of 10 terms, each = 2
        assert laplacian.shape == (5,)
        assert jnp.allclose(laplacian, expected, atol=1e-5)


class TestNumericalAccuracy:
    """Test numerical accuracy of autodiff operations."""

    def test_gradient_accuracy(self):
        """Gradient should be accurate to machine precision."""

        def f(x):
            return x[..., 0] ** 3 + 2 * x[..., 1] ** 2

        x = jnp.array([[1.5, 2.5]])

        grad = autodiff_engine.compute_gradient(f, x)
        # Analytical: [3x², 4y] = [3*1.5², 4*2.5] = [6.75, 10.0]
        expected = jnp.array([[6.75, 10.0]])
        assert jnp.allclose(grad, expected, atol=1e-10)

    def test_laplacian_accuracy(self):
        """Laplacian should be accurate for smooth functions."""

        def f(x):
            return jnp.sin(x[..., 0]) * jnp.cos(x[..., 1])

        x = jnp.array([[jnp.pi / 4, jnp.pi / 3]])

        laplacian = autodiff_engine.compute_laplacian(f, x)
        # Analytical: -sin(x)cos(y) - sin(x)cos(y) = -2sin(x)cos(y)
        expected = -2 * jnp.sin(jnp.pi / 4) * jnp.cos(jnp.pi / 3)
        assert jnp.allclose(laplacian, expected, atol=1e-8)


class TestUsagePatterns:
    """Test realistic usage patterns for PDE residuals."""

    def test_poisson_equation_pattern(self):
        """Test pattern for Poisson equation: ∇²u = f."""

        def u(x):
            # Solution: u = x² + y²
            return x[..., 0] ** 2 + x[..., 1] ** 2

        def source_term(x):
            # Source: f = 4 (since ∇²u = 4)
            return jnp.full(x.shape[0], 4.0)

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        laplacian_u = autodiff_engine.compute_laplacian(u, x)
        f = source_term(x)
        residual = laplacian_u - f
        assert jnp.allclose(residual, 0.0, atol=1e-6)  # Should be zero

    def test_heat_equation_pattern(self):
        """Test pattern for heat equation: ∂u/∂t = α∇²u."""

        # This tests the Laplacian part (time derivative would be separate)
        def u(x):
            return jnp.exp(-(x[..., 0] ** 2) - x[..., 1] ** 2)

        x = jnp.array([[0.5, 0.5]])

        laplacian_u = autodiff_engine.compute_laplacian(u, x)
        # For Gaussian, Laplacian is non-zero
        assert laplacian_u.shape == (1,)
        assert jnp.isfinite(laplacian_u).all()

    def test_custom_pde_pattern(self):
        """Test pattern for custom PDE with multiple derivatives."""

        def u(x):
            return x[..., 0] ** 2 * x[..., 1]

        x = jnp.array([[1.0, 2.0]])

        grad_u = autodiff_engine.compute_gradient(u, x)
        laplacian_u = autodiff_engine.compute_laplacian(u, x)

        # Custom PDE: ∇²u + |∇u|² = some constraint
        grad_norm_sq = jnp.sum(grad_u**2, axis=-1)
        custom_residual = laplacian_u + grad_norm_sq

        assert jnp.isfinite(custom_residual).all()
