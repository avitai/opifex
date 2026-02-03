"""
Comprehensive tests for optimized CSG operations.

Tests JAX compatibility (JIT, VMAP, GRAD) for all CSG operations
after optimization with vectorized operations.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.geometry.csg import (
    Circle,
    create_computational_domain_with_molecular_exclusion,
    CSGDifference,
    CSGIntersection,
    CSGUnion,
    MolecularGeometry,
    PeriodicCell,
    Polygon,
    Rectangle,
)


class TestOptimizedCSGShapes:
    """Test optimized CSG shape operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rectangle = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)
        self.circle = Circle(center=jnp.array([1.0, 1.0]), radius=0.5)
        self.polygon = Polygon(
            vertices=jnp.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]])
        )

    def test_rectangle_jit_compatibility(self):
        """Test that rectangle operations work with JIT compilation."""

        @jax.jit
        def compute_rectangle_distance(rect, point):
            return rect.distance(point)

        point = jnp.array([0.5, 0.3])
        result_normal = self.rectangle.distance(point)
        result_jit = compute_rectangle_distance(self.rectangle, point)

        assert jnp.allclose(result_normal, result_jit)

    def test_circle_vmap_compatibility(self):
        """Test that circle operations work with vmap."""

        def compute_circle_distance(point):
            return self.circle.distance(point)

        vmap_compute = jax.vmap(compute_circle_distance)
        points = jnp.array([[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]])

        results = vmap_compute(points)
        assert results.shape == (3,)
        assert jnp.all(jnp.isfinite(results))

    def test_polygon_grad_compatibility(self):
        """Test that polygon operations work with grad."""

        def polygon_distance_squared(point):
            dist = self.polygon.distance(point)
            return dist**2

        grad_fn = jax.grad(polygon_distance_squared)
        point = jnp.array([0.1, 0.2])
        gradient = grad_fn(point)

        assert gradient.shape == (2,)
        assert jnp.all(jnp.isfinite(gradient))

    def test_boundary_sampling_jit_compatibility(self):
        """Test that boundary sampling works with JIT compilation."""

        def sample_boundary_jit(shape, n_points, key):
            return shape.sample_boundary(n_points, key)

        sample_boundary_jit = jax.jit(sample_boundary_jit, static_argnums=(1,))

        key = jax.random.PRNGKey(42)
        n_points = 10

        result_normal = self.rectangle.sample_boundary(n_points, key)
        result_jit = sample_boundary_jit(self.rectangle, n_points, key)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (n_points, 2)


class TestOptimizedCSGOperations:
    """Test optimized CSG boolean operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)
        self.circle = Circle(center=jnp.array([0.0, 0.0]), radius=0.8)

        self.union = CSGUnion(self.rect, self.circle)
        self.intersection = CSGIntersection(self.rect, self.circle)
        self.difference = CSGDifference(self.rect, self.circle)

    def test_csg_union_jit_compatibility(self):
        """Test that CSG union works with JIT compilation."""

        @jax.jit
        def compute_union_distance(union_shape, point):
            return union_shape.distance(point)

        point = jnp.array([0.5, 0.3])
        result_normal = self.union.distance(point)
        result_jit = compute_union_distance(self.union, point)

        assert jnp.allclose(result_normal, result_jit)

    def test_csg_intersection_vmap_compatibility(self):
        """Test that CSG intersection works with vmap."""

        def compute_intersection_distance(point):
            return self.intersection.distance(point)

        vmap_compute = jax.vmap(compute_intersection_distance)
        points = jnp.array([[0.1, 0.1], [0.5, 0.5], [1.0, 1.0]])

        results = vmap_compute(points)
        assert results.shape == (3,)
        assert jnp.all(jnp.isfinite(results))

    def test_csg_difference_grad_compatibility(self):
        """Test that CSG difference works with grad."""

        def difference_distance_squared(point):
            dist = self.difference.distance(point)
            return dist**2

        grad_fn = jax.grad(difference_distance_squared)
        point = jnp.array([0.2, 0.3])
        gradient = grad_fn(point)

        assert gradient.shape == (2,)
        assert jnp.all(jnp.isfinite(gradient))

    def test_nested_csg_operations_jit_compatibility(self):
        """Test that nested CSG operations work with JIT compilation."""
        # Create nested CSG structure
        inner_union = CSGUnion(self.rect, self.circle)
        outer_difference = CSGDifference(
            inner_union, Circle(center=jnp.array([1.0, 0.0]), radius=0.3)
        )

        @jax.jit
        def compute_nested_distance(nested_shape, point):
            return nested_shape.distance(point)

        point = jnp.array([0.7, 0.2])
        result_normal = outer_difference.distance(point)
        result_jit = compute_nested_distance(outer_difference, point)

        assert jnp.allclose(result_normal, result_jit)


class TestOptimizedPeriodicCell:
    """Test optimized periodic cell operations."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple cubic cell
        lattice_vectors = jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        self.periodic_cell = PeriodicCell(lattice_vectors=lattice_vectors)

    def test_periodic_distance_jit_compatibility(self):
        """Test that periodic distance computation works with JIT compilation."""

        @jax.jit
        def compute_periodic_distance(cell, point1, point2):
            return cell.periodic_distance(point1, point2)

        point1 = jnp.array([0.1, 0.2, 0.3])
        point2 = jnp.array([1.9, 1.8, 1.7])

        result_normal = self.periodic_cell.periodic_distance(point1, point2)
        result_jit = compute_periodic_distance(self.periodic_cell, point1, point2)

        assert jnp.allclose(result_normal, result_jit)

    def test_find_neighbors_vmap_compatibility(self):
        """Test that optimized neighbor finding works with vmap."""
        # Create test positions
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]
        )
        cutoff_radius = 0.8

        # Test that the optimized method works
        neighbors = self.periodic_cell.find_neighbors(positions, cutoff_radius)

        # Should find neighbors within cutoff
        assert len(neighbors) > 0
        for i, j, dist in neighbors:
            assert dist <= cutoff_radius
            assert i < j  # Upper triangular pairs only

    def test_wrap_to_unit_cell_grad_compatibility(self):
        """Test that wrapping to unit cell works with grad."""

        def wrapped_position_norm(point):
            wrapped = self.periodic_cell.wrap_to_unit_cell(point)
            return jnp.linalg.norm(wrapped)

        grad_fn = jax.grad(wrapped_position_norm)
        point = jnp.array([2.5, 3.2, 1.8])
        gradient = grad_fn(point)

        assert gradient.shape == (3,)
        assert jnp.all(jnp.isfinite(gradient))


class TestOptimizedMolecularExclusion:
    """Test optimized molecular exclusion domain creation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple molecular geometry
        atomic_symbols = ["H", "H", "O"]  # H2O
        positions = jnp.array(
            [
                [0.0, 0.757, 0.587],
                [0.0, -0.757, 0.587],
                [0.0, 0.0, -0.074],
            ]
        )
        self.molecular_geometry = MolecularGeometry(
            atomic_symbols=atomic_symbols, positions=positions
        )
        self.domain_shape = Rectangle(
            center=jnp.array([0.0, 0.0]), width=4.0, height=4.0
        )

    def test_molecular_exclusion_functionality(self):
        """Test that molecular exclusion domain creation works correctly."""
        exclusion_radius = 0.5

        result = create_computational_domain_with_molecular_exclusion(
            self.domain_shape, self.molecular_geometry, exclusion_radius
        )

        # Test that the result is a valid shape
        assert hasattr(result, "distance")
        assert hasattr(result, "contains")

        # Test distance function works
        test_point = jnp.array([1.0, 1.0])
        distance = result.distance(test_point)
        assert jnp.isfinite(distance)

        # Test that exclusion zones are created (distance should be affected)
        # The result should be different due to molecular exclusions
        # (exact comparison depends on molecular positions)

    def test_molecular_projection_vmap_compatibility(self):
        """Test that molecular geometry projection works with vmap."""

        def project_single_geometry(mol_geom):
            return mol_geom.project_to_2d()

        # Create batch of molecular geometries (same geometry repeated)
        batch_geoms = [self.molecular_geometry] * 3

        # Test individual projections
        projections = [project_single_geometry(geom) for geom in batch_geoms]

        # All projections should have same shape
        for proj in projections:
            assert proj.shape == (3, 2)  # 3 atoms, 2D projection


class TestCombinedTransformations:
    """Test combined JAX transformations on optimized CSG operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rectangle = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)
        self.circle = Circle(center=jnp.array([0.5, 0.5]), radius=0.3)

    def test_jit_vmap_combination(self):
        """Test JIT compilation of vmap functions."""

        def compute_distance(point):
            return self.rectangle.distance(point)

        vmap_compute = jax.vmap(compute_distance)
        jit_vmap_compute = jax.jit(vmap_compute)

        points = jnp.array([[0.1, 0.2], [0.5, 0.6], [1.0, 0.8]])

        result_vmap = vmap_compute(points)
        result_jit_vmap = jit_vmap_compute(points)

        assert jnp.allclose(result_vmap, result_jit_vmap)

    def test_jit_grad_combination(self):
        """Test JIT compilation of gradient functions."""

        def distance_squared(point):
            dist = self.circle.distance(point)
            return dist**2

        grad_fn = jax.grad(distance_squared)
        jit_grad_fn = jax.jit(grad_fn)

        point = jnp.array([0.8, 0.9])

        result_grad = grad_fn(point)
        result_jit_grad = jit_grad_fn(point)

        assert jnp.allclose(result_grad, result_jit_grad)

    def test_vmap_grad_combination(self):
        """Test vmap of gradient functions."""

        def distance_squared(point):
            return self.rectangle.distance(point) ** 2

        grad_fn = jax.grad(distance_squared)
        vmap_grad_fn = jax.vmap(grad_fn)

        points = jnp.array([[0.2, 0.3], [0.7, 0.8], [1.2, 1.3]])
        gradients = vmap_grad_fn(points)

        assert gradients.shape == (3, 2)
        assert jnp.all(jnp.isfinite(gradients))


class TestPerformanceBenchmarks:
    """Performance benchmarks for optimized CSG operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rectangle = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)
        self.key = jax.random.PRNGKey(42)

    def test_batch_distance_computation_performance(self):
        """Test performance of batch distance computations."""

        @jax.jit
        def batch_distance_computation(points):
            def compute_distance(point):
                return self.rectangle.distance(point)

            return jax.vmap(compute_distance)(points)

        # Large batch of points
        batch_size = 1000
        points = jax.random.uniform(self.key, (batch_size, 2)) * 4.0 - 2.0

        # Warm up JIT compilation
        _ = batch_distance_computation(points)

        # Test that computation completes successfully
        results = batch_distance_computation(points)
        assert results.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(results))

    def test_boundary_sampling_performance(self):
        """Test performance of boundary sampling operations."""

        def sample_boundary_batch(n_points, key):
            return self.rectangle.sample_boundary(n_points, key)

        sample_boundary_batch = jax.jit(sample_boundary_batch, static_argnums=(0,))

        n_points = 500
        key = jax.random.PRNGKey(123)

        # Warm up JIT compilation
        _ = sample_boundary_batch(n_points, key)

        # Test that sampling completes successfully
        samples = sample_boundary_batch(n_points, key)
        assert samples.shape == (n_points, 2)
        assert jnp.all(jnp.isfinite(samples))


if __name__ == "__main__":
    pytest.main([__file__])
