"""
Comprehensive tests for optimized hyperbolic manifold operations.

Tests JAX compatibility (JIT, VMAP, GRAD) for all hyperbolic manifold operations
after optimization with vectorized operations.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.geometry.manifolds.hyperbolic import HyperbolicManifold


class TestOptimizedHyperbolicManifold:
    """Test optimized hyperbolic manifold operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = HyperbolicManifold(curvature=-1.0, dimension=2)
        self.key = jax.random.PRNGKey(42)

    def test_manifold_creation(self):
        """Test hyperbolic manifold creation."""
        assert self.manifold.curvature == -1.0
        assert self.manifold.dimension == 2
        assert self.manifold.radius == 1.0

    def test_point_validation_jit_compatibility(self):
        """Test that point validation works with JIT compilation."""

        @jax.jit
        def validate_point(manifold, point):
            return manifold._validate_point(point)

        # Test point inside disk
        point_inside = jnp.array([0.3, 0.4])
        result_normal = self.manifold._validate_point(point_inside)
        result_jit = validate_point(self.manifold, point_inside)

        assert jnp.allclose(result_normal, result_jit)

        # Test point outside disk (should be projected)
        point_outside = jnp.array([1.5, 1.2])
        result_normal = self.manifold._validate_point(point_outside)
        result_jit = validate_point(self.manifold, point_outside)

        assert jnp.allclose(result_normal, result_jit)
        assert jnp.linalg.norm(result_jit) < 1.0


class TestOptimizedGyrovectorOperations:
    """Test optimized gyrovector operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

    def test_gyroaddition_jit_compatibility(self):
        """Test that gyroaddition works with JIT compilation."""

        @jax.jit
        def compute_gyroaddition(manifold, u, v):
            return manifold._gyroaddition(u, v)

        u = jnp.array([0.2, 0.3])
        v = jnp.array([0.1, 0.4])

        result_normal = self.manifold._gyroaddition(u, v)
        result_jit = compute_gyroaddition(self.manifold, u, v)

        assert jnp.allclose(result_normal, result_jit)

    def test_gyroaddition_vmap_compatibility(self):
        """Test that gyroaddition works with vmap."""

        def compute_gyroaddition_single(u, v):
            return self.manifold._gyroaddition(u, v)

        vmap_gyroaddition = jax.vmap(compute_gyroaddition_single)

        # Batch of vector pairs
        u_batch = jnp.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.4]])
        v_batch = jnp.array([[0.2, 0.1], [0.1, 0.3], [0.3, 0.2]])

        results = vmap_gyroaddition(u_batch, v_batch)

        assert results.shape == (3, 2)
        assert jnp.all(jnp.isfinite(results))
        # All results should be in Poincaré disk
        assert jnp.all(jnp.linalg.norm(results, axis=1) < 1.0)

    def test_gyroaddition_grad_compatibility(self):
        """Test that gyroaddition works with grad."""

        def gyroaddition_norm(u, v):
            result = self.manifold._gyroaddition(u, v)
            return jnp.linalg.norm(result)

        grad_fn = jax.grad(gyroaddition_norm, argnums=(0, 1))

        u = jnp.array([0.1, 0.2])
        v = jnp.array([0.2, 0.1])

        grad_u, grad_v = grad_fn(u, v)

        assert grad_u.shape == (2,)
        assert grad_v.shape == (2,)
        assert jnp.all(jnp.isfinite(grad_u))
        assert jnp.all(jnp.isfinite(grad_v))


class TestOptimizedExpLogMaps:
    """Test optimized exponential and logarithmic maps."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

    def test_exp_map_jit_compatibility(self):
        """Test that exponential map works with JIT compilation."""

        @jax.jit
        def compute_exp_map(manifold, base, tangent):
            return manifold.exp_map(base, tangent)

        base = jnp.array([0.1, 0.2])
        tangent = jnp.array([0.3, 0.1])

        result_normal = self.manifold.exp_map(base, tangent)
        result_jit = compute_exp_map(self.manifold, base, tangent)

        assert jnp.allclose(result_normal, result_jit, atol=1e-6)

    def test_log_map_jit_compatibility(self):
        """Test that logarithmic map works with JIT compilation."""

        @jax.jit
        def compute_log_map(manifold, base, point):
            return manifold.log_map(base, point)

        base = jnp.array([0.1, 0.2])
        point = jnp.array([0.3, 0.4])

        result_normal = self.manifold.log_map(base, point)
        result_jit = compute_log_map(self.manifold, base, point)

        assert jnp.allclose(result_normal, result_jit, atol=1e-6)

    def test_exp_log_consistency_vmap(self):
        """Test exp/log map consistency with vmap."""

        def test_consistency(base, tangent):
            # exp_map followed by log_map should return original tangent
            point = self.manifold.exp_map(base, tangent)
            recovered_tangent = self.manifold.log_map(base, point)
            return jnp.linalg.norm(tangent - recovered_tangent)

        vmap_test = jax.vmap(test_consistency)

        # Batch of base points and tangent vectors
        bases = jnp.array([[0.1, 0.1], [0.2, 0.3], [0.0, 0.4]])
        tangents = jnp.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.2]])

        errors = vmap_test(bases, tangents)

        assert errors.shape == (3,)
        assert jnp.all(errors < 1e-4)  # Should be very small errors

    def test_geodesic_distance_grad_compatibility(self):
        """Test that geodesic distance works with grad."""

        def distance_function(base, point):
            return self.manifold.geodesic_distance(base, point)

        grad_fn = jax.grad(distance_function, argnums=(0, 1))

        base = jnp.array([0.1, 0.2])
        point = jnp.array([0.3, 0.4])

        grad_base, grad_point = grad_fn(base, point)

        assert grad_base.shape == (2,)
        assert grad_point.shape == (2,)
        assert jnp.all(jnp.isfinite(grad_base))
        assert jnp.all(jnp.isfinite(grad_point))


class TestOptimizedChristoffelSymbols:
    """Test optimized Christoffel symbol computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

    def test_christoffel_symbols_jit_compatibility(self):
        """Test that Christoffel symbols computation works with JIT compilation."""

        @jax.jit
        def compute_christoffel(manifold, point):
            return manifold.christoffel_symbols(point)

        point = jnp.array([0.2, 0.3])

        result_normal = self.manifold.christoffel_symbols(point)
        result_jit = compute_christoffel(self.manifold, point)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (2, 2, 2)

    def test_christoffel_symbols_vmap_compatibility(self):
        """Test that Christoffel symbols computation works with vmap."""

        def compute_christoffel_single(point):
            return self.manifold.christoffel_symbols(point)

        vmap_christoffel = jax.vmap(compute_christoffel_single)

        points = jnp.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.4]])
        results = vmap_christoffel(points)

        assert results.shape == (3, 2, 2, 2)
        assert jnp.all(jnp.isfinite(results))

    def test_christoffel_symbols_grad_compatibility(self):
        """Test that Christoffel symbols computation works with grad."""

        def christoffel_trace(point):
            christoffel = self.manifold.christoffel_symbols(point)
            # Sum over all components for a scalar output
            return jnp.sum(christoffel)

        grad_fn = jax.grad(christoffel_trace)

        point = jnp.array([0.1, 0.3])
        gradient = grad_fn(point)

        assert gradient.shape == (2,)
        assert jnp.all(jnp.isfinite(gradient))


class TestOptimizedMetricTensor:
    """Test optimized metric tensor operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

    def test_metric_tensor_jit_compatibility(self):
        """Test that metric tensor computation works with JIT compilation."""

        @jax.jit
        def compute_metric(manifold, point):
            return manifold.metric_tensor(point)

        point = jnp.array([0.2, 0.3])

        result_normal = self.manifold.metric_tensor(point)
        result_jit = compute_metric(self.manifold, point)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (2, 2)

    def test_metric_tensor_properties(self):
        """Test that metric tensor has correct properties."""
        point = jnp.array([0.1, 0.2])
        metric = self.manifold.metric_tensor(point)

        # Should be symmetric
        assert jnp.allclose(metric, metric.T)

        # Should be positive definite
        eigenvals = jnp.linalg.eigvals(metric)
        assert jnp.all(eigenvals > 0)

    def test_metric_tensor_batch_vmap(self):
        """Test metric tensor computation with batch processing."""

        def compute_metric_single(point):
            return self.manifold.metric_tensor(point)

        vmap_metric = jax.vmap(compute_metric_single)

        points = jnp.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.4]])
        metrics = vmap_metric(points)

        assert metrics.shape == (3, 2, 2)
        assert jnp.all(jnp.isfinite(metrics))

        # All should be positive definite
        for i in range(3):
            eigenvals = jnp.linalg.eigvals(metrics[i])
            assert jnp.all(eigenvals > 0)


class TestOptimizedParallelTransport:
    """Test optimized parallel transport operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

    def test_parallel_transport_jit_compatibility(self):
        """Test that parallel transport works with JIT compilation."""

        @jax.jit
        def compute_parallel_transport(manifold, tangent, start, end):
            return manifold.parallel_transport(tangent, start, end)

        tangent = jnp.array([0.1, 0.2])
        start = jnp.array([0.1, 0.1])
        end = jnp.array([0.3, 0.2])

        result_normal = self.manifold.parallel_transport(tangent, start, end)
        result_jit = compute_parallel_transport(self.manifold, tangent, start, end)

        assert jnp.allclose(result_normal, result_jit, atol=1e-6)

    def test_parallel_transport_preserves_norm(self):
        """Test that parallel transport preserves tangent vector norms."""
        tangent = jnp.array([0.2, 0.1])
        start = jnp.array([0.1, 0.2])
        end = jnp.array([0.3, 0.3])

        # Compute norms using metric tensor
        start_metric = self.manifold.metric_tensor(start)
        end_metric = self.manifold.metric_tensor(end)

        original_norm = jnp.sqrt(jnp.einsum("i,ij,j->", tangent, start_metric, tangent))

        transported = self.manifold.parallel_transport(tangent, start, end)
        transported_norm = jnp.sqrt(
            jnp.einsum("i,ij,j->", transported, end_metric, transported)
        )

        assert jnp.allclose(original_norm, transported_norm, atol=1e-5)


class TestCombinedTransformations:
    """Test combined JAX transformations on optimized hyperbolic operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = HyperbolicManifold(curvature=-1.0, dimension=2)

    def test_jit_vmap_combination(self):
        """Test JIT compilation of vmap functions."""

        def compute_distance(base, point):
            return self.manifold.geodesic_distance(base, point)

        vmap_compute = jax.vmap(compute_distance, in_axes=(None, 0))
        jit_vmap_compute = jax.jit(vmap_compute)

        base = jnp.array([0.1, 0.2])
        points = jnp.array([[0.2, 0.3], [0.3, 0.1], [0.1, 0.4]])

        result_vmap = vmap_compute(base, points)
        result_jit_vmap = jit_vmap_compute(base, points)

        assert jnp.allclose(result_vmap, result_jit_vmap)

    def test_jit_grad_combination(self):
        """Test JIT compilation of gradient functions."""

        def distance_squared(base, point):
            dist = self.manifold.geodesic_distance(base, point)
            return dist**2

        grad_fn = jax.grad(distance_squared, argnums=1)
        jit_grad_fn = jax.jit(grad_fn)

        base = jnp.array([0.1, 0.2])
        point = jnp.array([0.3, 0.4])

        result_grad = grad_fn(base, point)
        result_jit_grad = jit_grad_fn(base, point)

        assert jnp.allclose(result_grad, result_jit_grad)

    def test_vmap_grad_combination(self):
        """Test vmap of gradient functions."""

        def exp_map_norm(base, tangent):
            point = self.manifold.exp_map(base, tangent)
            return jnp.linalg.norm(point)

        grad_fn = jax.grad(exp_map_norm, argnums=1)
        vmap_grad_fn = jax.vmap(grad_fn, in_axes=(0, 0))

        bases = jnp.array([[0.1, 0.1], [0.2, 0.2], [0.1, 0.3]])
        tangents = jnp.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.1]])

        gradients = vmap_grad_fn(bases, tangents)

        assert gradients.shape == (3, 2)
        assert jnp.all(jnp.isfinite(gradients))


class TestPerformanceBenchmarks:
    """Performance benchmarks for optimized hyperbolic operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = HyperbolicManifold(curvature=-1.0, dimension=2)
        self.key = jax.random.PRNGKey(42)

    def test_batch_christoffel_computation_performance(self):
        """Test performance of batch Christoffel symbol computations."""

        @jax.jit
        def batch_christoffel_computation(points):
            def compute_christoffel(point):
                return self.manifold.christoffel_symbols(point)

            return jax.vmap(compute_christoffel)(points)

        # Large batch of points
        batch_size = 100
        points = jax.random.uniform(self.key, (batch_size, 2)) * 0.8  # Stay in disk

        # Warm up JIT compilation
        _ = batch_christoffel_computation(points)

        # Test that computation completes successfully
        results = batch_christoffel_computation(points)
        assert results.shape == (batch_size, 2, 2, 2)
        assert jnp.all(jnp.isfinite(results))

    def test_batch_geodesic_distance_performance(self):
        """Test performance of batch geodesic distance computations."""

        @jax.jit
        def batch_distance_computation(base_points, target_points):
            def compute_distance(base, target):
                return self.manifold.geodesic_distance(base, target)

            return jax.vmap(compute_distance)(base_points, target_points)

        batch_size = 100
        base_points = jax.random.uniform(self.key, (batch_size, 2)) * 0.6
        target_points = (
            jax.random.uniform(jax.random.split(self.key)[0], (batch_size, 2)) * 0.6
        )

        # Warm up JIT compilation
        _ = batch_distance_computation(base_points, target_points)

        # Test that computation completes successfully
        results = batch_distance_computation(base_points, target_points)
        assert results.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(results))
        assert jnp.all(results >= 0)  # Distances should be non-negative


if __name__ == "__main__":
    pytest.main([__file__])
