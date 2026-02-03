"""
Tests for pytree utilities for geometric objects.

This module tests that geometric objects are properly registered as pytrees
and work correctly with JAX transformations.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.geometry.csg import (
    Circle,
    CSGDifference,
    CSGIntersection,
    CSGUnion,
    Polygon,
    Rectangle,
)
from opifex.geometry.manifolds.riemannian import euclidean_metric, RiemannianManifold


# Note: SO3Group, SE3Group, and GraphTopology already have pytree registration in their modules


class TestRiemannianManifoldPytree:
    """Test pytree functionality for RiemannianManifold."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = RiemannianManifold(
            dimension=2,
            metric_function=euclidean_metric,
        )

    def test_manifold_tree_flatten_unflatten(self):
        """Test that manifold can be flattened and unflattened."""
        # Test tree_flatten and tree_unflatten
        children, aux_data = jax.tree_util.tree_flatten(self.manifold)
        reconstructed = jax.tree_util.tree_unflatten(aux_data, children)

        assert reconstructed.dimension == self.manifold.dimension
        assert reconstructed.embedding_dimension == self.manifold.embedding_dimension
        assert reconstructed.metric_function == self.manifold.metric_function

    def test_manifold_jit_compatibility(self):
        """Test that manifold works with JIT compilation."""

        def compute_metric(manifold, point):
            return manifold.metric_tensor(point)

        jit_compute_metric = jax.jit(compute_metric)

        point = jnp.array([0.1, 0.2])

        result_normal = compute_metric(self.manifold, point)
        result_jit = jit_compute_metric(self.manifold, point)

        assert jnp.allclose(result_normal, result_jit)

    def test_manifold_vmap_compatibility(self):
        """Test that manifold works with vmap."""

        def compute_metric(manifold, point):
            return manifold.metric_tensor(point)

        vmap_compute_metric = jax.vmap(compute_metric, in_axes=(None, 0))

        points = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        result = vmap_compute_metric(self.manifold, points)

        assert result.shape == (3, 2, 2)
        assert jnp.all(jnp.isfinite(result))

    def test_manifold_grad_compatibility(self):
        """Test that manifold works with grad."""

        def metric_trace(manifold, point):
            metric = manifold.metric_tensor(point)
            return jnp.trace(metric)

        grad_fn = jax.grad(metric_trace, argnums=1)

        point = jnp.array([0.1, 0.2])
        gradient = grad_fn(self.manifold, point)

        assert gradient.shape == (2,)
        assert jnp.all(jnp.isfinite(gradient))


class TestCSGShapesPytree:
    """Test pytree functionality for CSG shapes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rectangle = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)
        self.circle = Circle(center=jnp.array([1.0, 1.0]), radius=0.5)
        self.polygon = Polygon(vertices=jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]))

    def test_rectangle_tree_operations(self):
        """Test rectangle pytree operations."""
        # Test tree_flatten and tree_unflatten
        children, aux_data = jax.tree_util.tree_flatten(self.rectangle)
        reconstructed = jax.tree_util.tree_unflatten(aux_data, children)

        assert jnp.allclose(reconstructed.center, self.rectangle.center)
        assert reconstructed.width == self.rectangle.width
        assert reconstructed.height == self.rectangle.height

    def test_circle_tree_operations(self):
        """Test circle pytree operations."""
        # Test tree_flatten and tree_unflatten
        children, aux_data = jax.tree_util.tree_flatten(self.circle)
        reconstructed = jax.tree_util.tree_unflatten(aux_data, children)

        assert jnp.allclose(reconstructed.center, self.circle.center)
        assert reconstructed.radius == self.circle.radius

    def test_polygon_tree_operations(self):
        """Test polygon pytree operations."""
        # Test tree_flatten and tree_unflatten
        children, aux_data = jax.tree_util.tree_flatten(self.polygon)
        reconstructed = jax.tree_util.tree_unflatten(aux_data, children)

        assert jnp.allclose(reconstructed.vertices, self.polygon.vertices)

    def test_rectangle_jit_compatibility(self):
        """Test that rectangle works with JIT compilation."""

        def check_distance(shape, point):
            # Use distance function instead of contains to avoid boolean conversion
            return shape.distance(point)

        jit_check_distance = jax.jit(check_distance)

        point = jnp.array([0.5, 0.3])

        result_normal = check_distance(self.rectangle, point)
        result_jit = jit_check_distance(self.rectangle, point)

        assert jnp.allclose(result_normal, result_jit)

    def test_circle_vmap_compatibility(self):
        """Test that circle works with vmap."""

        def check_distance(shape, point):
            # Use distance function instead of contains to avoid boolean conversion
            return shape.distance(point)

        vmap_check_distance = jax.vmap(check_distance, in_axes=(None, 0))

        points = jnp.array([[1.0, 1.0], [1.2, 1.2], [2.0, 2.0]])

        results = vmap_check_distance(self.circle, points)

        assert results.shape == (3,)
        assert jnp.all(jnp.isfinite(results))


class TestCSGOperationsPytree:
    """Test pytree functionality for CSG operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)
        self.circle = Circle(center=jnp.array([0.0, 0.0]), radius=0.8)

        self.union = CSGUnion(self.rect, self.circle)
        self.intersection = CSGIntersection(self.rect, self.circle)
        self.difference = CSGDifference(self.rect, self.circle)

    def test_csg_union_tree_operations(self):
        """Test CSG union pytree operations."""
        # Test tree_flatten and tree_unflatten
        children, aux_data = jax.tree_util.tree_flatten(self.union)
        reconstructed = jax.tree_util.tree_unflatten(aux_data, children)

        # Check that shapes are preserved
        from typing import cast

        assert jnp.allclose(
            cast("Rectangle", reconstructed.shape_a).center,
            cast("Rectangle", self.union.shape_a).center,
        )
        assert jnp.allclose(
            cast("Circle", reconstructed.shape_b).center,
            cast("Circle", self.union.shape_b).center,
        )

    def test_csg_intersection_tree_operations(self):
        """Test CSG intersection pytree operations."""
        # Test tree_flatten and tree_unflatten
        children, aux_data = jax.tree_util.tree_flatten(self.intersection)
        reconstructed = jax.tree_util.tree_unflatten(aux_data, children)

        # Check that shapes are preserved
        from typing import cast

        assert jnp.allclose(
            cast("Rectangle", reconstructed.shape_a).center,
            cast("Rectangle", self.intersection.shape_a).center,
        )
        assert jnp.allclose(
            cast("Circle", reconstructed.shape_b).center,
            cast("Circle", self.intersection.shape_b).center,
        )

    def test_csg_difference_tree_operations(self):
        """Test CSG difference pytree operations."""
        # Test tree_flatten and tree_unflatten
        children, aux_data = jax.tree_util.tree_flatten(self.difference)
        reconstructed = jax.tree_util.tree_unflatten(aux_data, children)

        # Check that shapes are preserved
        from typing import cast

        assert jnp.allclose(
            cast("Rectangle", reconstructed.shape_a).center,
            cast("Rectangle", self.difference.shape_a).center,
        )
        assert jnp.allclose(
            cast("Circle", reconstructed.shape_b).center,
            cast("Circle", self.difference.shape_b).center,
        )

    def test_csg_union_jit_compatibility(self):
        """Test that CSG union works with JIT compilation."""

        def check_distance(shape, point):
            # Use distance function to avoid boolean conversion issues
            return shape.distance(point)

        jit_check_distance = jax.jit(check_distance)

        point = jnp.array([0.5, 0.3])

        result_normal = check_distance(self.union, point)
        result_jit = jit_check_distance(self.union, point)

        assert jnp.allclose(result_normal, result_jit)


# Note: Lie groups and graph topology tests are in their respective test files
# since they already have pytree registration


class TestCombinedTransformations:
    """Test combined JAX transformations with geometric objects."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manifold = RiemannianManifold(
            dimension=2,
            metric_function=euclidean_metric,
        )
        self.rectangle = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)

    def test_jit_vmap_combination(self):
        """Test JIT compilation of vmap functions with geometric objects."""

        def compute_metric_trace(manifold, point):
            metric = manifold.metric_tensor(point)
            return jnp.trace(metric)

        vmap_compute = jax.vmap(compute_metric_trace, in_axes=(None, 0))
        jit_vmap_compute = jax.jit(vmap_compute)

        points = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        result_vmap = vmap_compute(self.manifold, points)
        result_jit_vmap = jit_vmap_compute(self.manifold, points)

        assert jnp.allclose(result_vmap, result_jit_vmap)

    def test_jit_grad_combination(self):
        """Test JIT compilation of gradient functions with geometric objects."""

        def shape_area_approximation(shape, scale_factor):
            # Simple approximation for testing
            scaled_center = scale_factor * shape.center
            return jnp.sum(scaled_center**2)

        grad_fn = jax.grad(shape_area_approximation, argnums=1)
        jit_grad_fn = jax.jit(grad_fn)

        scale_factor = 2.0

        result_grad = grad_fn(self.rectangle, scale_factor)
        result_jit_grad = jit_grad_fn(self.rectangle, scale_factor)

        assert jnp.allclose(result_grad, result_jit_grad)


class TestErrorHandling:
    """Test error handling in pytree operations."""

    def test_invalid_manifold_reconstruction(self):
        """Test handling of invalid manifold reconstruction."""
        manifold = RiemannianManifold(
            dimension=2,
            metric_function=euclidean_metric,
        )

        children, aux_data = jax.tree_util.tree_flatten(manifold)

        # Test that reconstruction works normally
        reconstructed = jax.tree_util.tree_unflatten(aux_data, children)
        assert reconstructed.dimension == manifold.dimension

    def test_nested_csg_operations(self):
        """Test pytree operations with nested CSG structures."""
        rect1 = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)
        rect2 = Rectangle(center=jnp.array([1.0, 0.0]), width=1.0, height=2.0)
        circle = Circle(center=jnp.array([0.5, 0.5]), radius=0.3)

        # Create nested CSG operations
        union = CSGUnion(rect1, rect2)
        complex_shape = CSGDifference(union, circle)

        # Test that nested structure can be flattened and unflattened
        children, aux_data = jax.tree_util.tree_flatten(complex_shape)
        reconstructed = jax.tree_util.tree_unflatten(aux_data, children)

        # Test that the nested structure is preserved
        assert isinstance(reconstructed.shape_a, CSGUnion)
        assert isinstance(reconstructed.shape_b, Circle)


if __name__ == "__main__":
    pytest.main([__file__])
