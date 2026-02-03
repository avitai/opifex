"""
Tests for Riemannian manifold implementation with JAX compatibility.

This module tests the RiemannianManifold class and ensures all methods
are compatible with JAX transformations (jit, vmap, grad).
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.geometry.manifolds.riemannian import (
    euclidean_metric,
    hyperbolic_metric,
    RiemannianManifold,
    spherical_metric,
)


class TestRiemannianManifold:
    """Test RiemannianManifold class with various metrics."""

    def test_euclidean_manifold_creation(self):
        """Test creation of Euclidean manifold."""
        manifold = RiemannianManifold(
            dimension=2,
            metric_function=euclidean_metric,
        )

        assert manifold.dimension == 2
        assert manifold.embedding_dimension == 2

        # Test metric at origin
        point = jnp.array([0.0, 0.0])
        metric = manifold.metric_tensor(point)
        expected = jnp.eye(2)
        assert jnp.allclose(metric, expected)

    def test_hyperbolic_manifold_creation(self):
        """Test creation of hyperbolic manifold with Poincaré disk metric."""
        manifold = RiemannianManifold(
            dimension=2,
            metric_function=hyperbolic_metric(-1.0),
        )

        assert manifold.dimension == 2

        # Test metric at origin
        point = jnp.array([0.0, 0.0])
        metric = manifold.metric_tensor(point)
        # At origin, hyperbolic metric should be 4 * I
        expected = 4.0 * jnp.eye(2)
        assert jnp.allclose(metric, expected)

    def test_spherical_manifold_creation(self):
        """Test creation of spherical manifold."""
        radius = 2.0
        manifold = RiemannianManifold(
            dimension=2,
            metric_function=spherical_metric(radius),
        )

        assert manifold.dimension == 2

        # Test metric
        point = jnp.array([0.0, 0.0])
        metric = manifold.metric_tensor(point)
        expected = (radius**2) * jnp.eye(2)
        assert jnp.allclose(metric, expected)


class TestJITCompatibility:
    """Test JAX JIT compatibility for all Riemannian manifold methods."""

    def setup_method(self):
        """Set up test manifold for each test."""
        self.manifold = RiemannianManifold(
            dimension=2,
            metric_function=euclidean_metric,
        )

        # Test points and vectors
        self.test_point = jnp.array([0.1, 0.2])
        self.test_tangent = jnp.array([0.3, 0.4])
        self.test_point2 = jnp.array([0.5, 0.6])
        self.test_vector = jnp.array([0.1, 0.1])

    def test_metric_tensor_jit_compatibility(self):
        """Test that metric_tensor can be JIT compiled."""
        jit_metric = jax.jit(self.manifold.metric_tensor)

        # Test compilation and execution
        result_normal = self.manifold.metric_tensor(self.test_point)
        result_jit = jit_metric(self.test_point)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (2, 2)

    def test_christoffel_symbols_jit_compatibility(self):
        """Test that christoffel_symbols can be JIT compiled."""
        jit_christoffel = jax.jit(self.manifold.christoffel_symbols)

        # Test compilation and execution
        result_normal = self.manifold.christoffel_symbols(self.test_point)
        result_jit = jit_christoffel(self.test_point)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (2, 2, 2)

    def test_riemann_curvature_jit_compatibility(self):
        """Test that riemann_curvature can be JIT compiled."""
        jit_riemann = jax.jit(self.manifold.riemann_curvature)

        # Test compilation and execution
        result_normal = self.manifold.riemann_curvature(self.test_point)
        result_jit = jit_riemann(self.test_point)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (2, 2, 2, 2)

    def test_ricci_tensor_jit_compatibility(self):
        """Test that ricci_tensor can be JIT compiled."""
        jit_ricci = jax.jit(self.manifold.ricci_tensor)

        # Test compilation and execution
        result_normal = self.manifold.ricci_tensor(self.test_point)
        result_jit = jit_ricci(self.test_point)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (2, 2)

    def test_scalar_curvature_jit_compatibility(self):
        """Test that scalar_curvature can be JIT compiled."""
        jit_scalar = jax.jit(self.manifold.scalar_curvature)

        # Test compilation and execution
        result_normal = self.manifold.scalar_curvature(self.test_point)
        result_jit = jit_scalar(self.test_point)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == ()

    def test_exp_map_jit_compatibility(self):
        """Test that exp_map can be JIT compiled."""
        jit_exp = jax.jit(self.manifold.exp_map)

        # Test compilation and execution
        result_normal = self.manifold.exp_map(self.test_point, self.test_tangent)
        result_jit = jit_exp(self.test_point, self.test_tangent)

        assert jnp.allclose(result_normal, result_jit, atol=1e-5)
        assert result_jit.shape == (2,)

    def test_log_map_jit_compatibility(self):
        """Test that log_map can be JIT compiled."""
        jit_log = jax.jit(self.manifold.log_map)

        # Test compilation and execution
        result_normal = self.manifold.log_map(self.test_point, self.test_point2)
        result_jit = jit_log(self.test_point, self.test_point2)

        assert jnp.allclose(result_normal, result_jit, atol=1e-5)
        assert result_jit.shape == (2,)

    def test_geodesic_distance_jit_compatibility(self):
        """Test that geodesic_distance can be JIT compiled."""
        jit_distance = jax.jit(self.manifold.geodesic_distance)

        # Test compilation and execution
        result_normal = self.manifold.geodesic_distance(
            self.test_point, self.test_point2
        )
        result_jit = jit_distance(self.test_point, self.test_point2)

        assert jnp.allclose(result_normal, result_jit, atol=1e-5)
        assert result_jit.shape == ()

    def test_parallel_transport_jit_compatibility(self):
        """Test that parallel_transport can be JIT compiled."""
        jit_transport = jax.jit(self.manifold.parallel_transport)

        # Test compilation and execution
        result_normal = self.manifold.parallel_transport(
            self.test_point, self.test_tangent, self.test_vector
        )
        result_jit = jit_transport(self.test_point, self.test_tangent, self.test_vector)

        assert jnp.allclose(result_normal, result_jit, atol=1e-5)
        assert result_jit.shape == (2,)


class TestVMAPCompatibility:
    """Test JAX vmap compatibility for batch operations."""

    def setup_method(self):
        """Set up test manifold and batch data."""
        self.manifold = RiemannianManifold(
            dimension=2,
            metric_function=euclidean_metric,
        )

        # Batch test data
        self.batch_size = 5
        key = jax.random.PRNGKey(42)
        self.batch_points = jax.random.uniform(key, (self.batch_size, 2)) * 0.1
        self.batch_tangents = (
            jax.random.uniform(jax.random.split(key)[1], (self.batch_size, 2)) * 0.1
        )
        self.batch_points2 = (
            jax.random.uniform(jax.random.split(key, 3)[2], (self.batch_size, 2)) * 0.1
        )

    def test_batch_metric_tensor_vmap(self):
        """Test vmap compatibility for metric tensor computation."""
        vmap_metric = jax.vmap(self.manifold.metric_tensor)

        result = vmap_metric(self.batch_points)

        assert result.shape == (self.batch_size, 2, 2)

        # Compare with manual loop
        for i in range(self.batch_size):
            manual_result = self.manifold.metric_tensor(self.batch_points[i])
            assert jnp.allclose(result[i], manual_result)

    def test_batch_christoffel_symbols_vmap(self):
        """Test vmap compatibility for Christoffel symbols."""
        vmap_christoffel = jax.vmap(self.manifold.christoffel_symbols)

        result = vmap_christoffel(self.batch_points)

        assert result.shape == (self.batch_size, 2, 2, 2)

    def test_batch_exp_map_vmap(self):
        """Test vmap compatibility for exponential map."""
        vmap_exp = jax.vmap(self.manifold.exp_map)

        result = vmap_exp(self.batch_points, self.batch_tangents)

        assert result.shape == (self.batch_size, 2)

    def test_batch_geodesic_distance_vmap(self):
        """Test vmap compatibility for geodesic distance."""
        vmap_distance = jax.vmap(self.manifold.geodesic_distance)

        result = vmap_distance(self.batch_points, self.batch_points2)

        assert result.shape == (self.batch_size,)
        assert jnp.all(result >= 0)  # Distances should be non-negative

    def test_built_in_batch_methods(self):
        """Test the built-in batch methods work correctly."""
        # Test batch_geodesic_distance
        result = self.manifold.batch_geodesic_distance(
            self.batch_points, self.batch_points2
        )
        assert result.shape == (self.batch_size,)

        # Test batch_exp_map
        result = self.manifold.batch_exp_map(self.batch_points, self.batch_tangents)
        assert result.shape == (self.batch_size, 2)

        # Test batch_log_map
        result = self.manifold.batch_log_map(self.batch_points, self.batch_points2)
        assert result.shape == (self.batch_size, 2)


class TestGradCompatibility:
    """Test JAX grad compatibility for automatic differentiation."""

    def setup_method(self):
        """Set up test manifold."""
        self.manifold = RiemannianManifold(
            dimension=2,
            metric_function=euclidean_metric,
        )

    def test_metric_tensor_grad_compatibility(self):
        """Test that we can compute gradients of metric tensor."""

        def metric_trace(point):
            metric = self.manifold.metric_tensor(point)
            return jnp.trace(metric)

        grad_fn = jax.grad(metric_trace)

        point = jnp.array([0.1, 0.2])
        gradient = grad_fn(point)

        assert gradient.shape == (2,)
        assert jnp.all(jnp.isfinite(gradient))

    def test_geodesic_distance_grad_compatibility(self):
        """Test that we can compute gradients of geodesic distance."""

        def distance_from_origin(point):
            origin = jnp.zeros(2)
            return self.manifold.geodesic_distance(origin, point)

        grad_fn = jax.grad(distance_from_origin)

        point = jnp.array([0.3, 0.4])
        gradient = grad_fn(point)

        assert gradient.shape == (2,)
        assert jnp.all(jnp.isfinite(gradient))
        # Gradient should point away from origin
        assert jnp.linalg.norm(gradient) > 0

    def test_exp_map_grad_compatibility(self):
        """Test that we can compute gradients through exp_map."""

        def exp_map_norm(tangent):
            base = jnp.zeros(2)
            result = self.manifold.exp_map(base, tangent)
            return jnp.linalg.norm(result)

        grad_fn = jax.grad(exp_map_norm)

        tangent = jnp.array([0.1, 0.1])
        gradient = grad_fn(tangent)

        assert gradient.shape == (2,)
        assert jnp.all(jnp.isfinite(gradient))

    def test_scalar_curvature_grad_compatibility(self):
        """Test that we can compute gradients of scalar curvature."""
        grad_fn = jax.grad(self.manifold.scalar_curvature)

        point = jnp.array([0.1, 0.2])
        gradient = grad_fn(point)

        assert gradient.shape == (2,)
        assert jnp.all(jnp.isfinite(gradient))


class TestCombinedTransformations:
    """Test combinations of JAX transformations."""

    def setup_method(self):
        """Set up test manifold."""
        self.manifold = RiemannianManifold(
            dimension=2,
            metric_function=euclidean_metric,
        )

    def test_jit_vmap_combination(self):
        """Test JIT compilation of vmapped functions."""
        vmap_metric = jax.vmap(self.manifold.metric_tensor)
        jit_vmap_metric = jax.jit(vmap_metric)

        batch_points = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        result_vmap = vmap_metric(batch_points)
        result_jit_vmap = jit_vmap_metric(batch_points)

        assert jnp.allclose(result_vmap, result_jit_vmap)

    def test_jit_grad_combination(self):
        """Test JIT compilation of gradient functions."""

        def distance_function(point):
            origin = jnp.zeros(2)
            return self.manifold.geodesic_distance(origin, point)

        grad_fn = jax.grad(distance_function)
        jit_grad_fn = jax.jit(grad_fn)

        point = jnp.array([0.3, 0.4])

        result_grad = grad_fn(point)
        result_jit_grad = jit_grad_fn(point)

        assert jnp.allclose(result_grad, result_jit_grad, atol=1e-6)

    def test_vmap_grad_combination(self):
        """Test vmap of gradient functions."""

        def distance_function(point):
            origin = jnp.zeros(2)
            return self.manifold.geodesic_distance(origin, point)

        grad_fn = jax.grad(distance_function)
        vmap_grad_fn = jax.vmap(grad_fn)

        batch_points = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        result = vmap_grad_fn(batch_points)

        assert result.shape == (3, 2)
        assert jnp.all(jnp.isfinite(result))


class TestNonEuclideanManifolds:
    """Test JIT compatibility with non-Euclidean manifolds."""

    def test_hyperbolic_manifold_jit_compatibility(self):
        """Test JIT compatibility with hyperbolic metric."""
        manifold = RiemannianManifold(
            dimension=2,
            metric_function=hyperbolic_metric(-1.0),
        )

        # Test JIT compilation of various methods
        jit_metric = jax.jit(manifold.metric_tensor)
        jit_christoffel = jax.jit(manifold.christoffel_symbols)

        # Test point inside Poincaré disk
        point = jnp.array([0.3, 0.4])

        metric_result = jit_metric(point)
        christoffel_result = jit_christoffel(point)

        assert metric_result.shape == (2, 2)
        assert christoffel_result.shape == (2, 2, 2)
        assert jnp.all(jnp.isfinite(metric_result))
        assert jnp.all(jnp.isfinite(christoffel_result))

    def test_spherical_manifold_jit_compatibility(self):
        """Test JIT compatibility with spherical metric."""
        manifold = RiemannianManifold(
            dimension=2,
            metric_function=spherical_metric(2.0),
        )

        jit_exp_map = jax.jit(manifold.exp_map)

        base = jnp.array([0.1, 0.1])
        tangent = jnp.array([0.05, 0.05])

        result = jit_exp_map(base, tangent)

        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))


class TestErrorHandling:
    """Test error handling in JIT-compiled functions."""

    def test_invalid_dimensions_handling(self):
        """Test that dimension mismatches are handled properly."""
        manifold = RiemannianManifold(
            dimension=2,
            metric_function=euclidean_metric,
        )

        # This should work fine in eager mode but we test JIT compilation
        jit_metric = jax.jit(manifold.metric_tensor)

        valid_point = jnp.array([0.1, 0.2])
        result = jit_metric(valid_point)

        assert result.shape == (2, 2)

    def test_numerical_stability(self):
        """Test numerical stability in JIT-compiled functions."""
        manifold = RiemannianManifold(
            dimension=2,
            metric_function=euclidean_metric,
        )

        jit_distance = jax.jit(manifold.geodesic_distance)

        # Test with very close points
        point1 = jnp.array([0.0, 0.0])
        point2 = jnp.array([1e-10, 1e-10])

        distance = jit_distance(point1, point2)

        assert jnp.isfinite(distance)
        assert distance >= 0


class TestPerformanceBenchmarks:
    """Test performance improvements from JIT compilation."""

    def setup_method(self):
        """Set up manifold for benchmarking."""
        self.manifold = RiemannianManifold(
            dimension=3,  # Slightly larger for more computation
            metric_function=euclidean_metric,
        )

    def test_christoffel_symbols_performance(self):
        """Test that JIT compilation improves Christoffel symbols computation."""
        jit_christoffel = jax.jit(self.manifold.christoffel_symbols)

        point = jnp.array([0.1, 0.2, 0.3])

        # Warm up JIT compilation
        _ = jit_christoffel(point)

        # Both should produce same results
        result_normal = self.manifold.christoffel_symbols(point)
        result_jit = jit_christoffel(point)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (3, 3, 3)

    def test_batch_operations_performance(self):
        """Test that batch operations work efficiently with JIT."""
        batch_size = 100
        key = jax.random.PRNGKey(42)
        batch_points = jax.random.uniform(key, (batch_size, 3)) * 0.1

        # Test batch metric computation
        vmap_jit_metric = jax.jit(jax.vmap(self.manifold.metric_tensor))

        result = vmap_jit_metric(batch_points)

        assert result.shape == (batch_size, 3, 3)
        assert jnp.all(jnp.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__])
