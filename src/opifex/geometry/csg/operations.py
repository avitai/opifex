"""CSG set operations (Union, Intersection, Difference) and helpers.

Hosts the internal SDF arithmetic (:class:`_SDFOperations`) used by every
CSG node, plus the convenience constructors (:func:`union`,
:func:`intersection`, :func:`difference`), boundary-analysis helpers
(:func:`compute_boundary_normals`, :func:`sample_boundary_points`), and the
smooth-blended :func:`smooth_union`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from opifex.geometry.csg.types import _EnhancedShapeBase


if TYPE_CHECKING:
    from opifex.geometry.csg.types import Point2D, Points2D, Shape2D


class _SDFOperations:
    """Internal SDF operations for robust CSG."""

    @staticmethod
    def union_sdf(
        dist_a: Float[jax.Array, ""], dist_b: Float[jax.Array, ""]
    ) -> Float[jax.Array, ""]:
        """SDF union operation with smooth approximation for differentiability."""
        # Use polynomial smooth minimum for better numerical stability
        k = 0.1  # smoothing parameter
        h = jnp.maximum(k - jnp.abs(dist_a - dist_b), 0.0)
        return jnp.minimum(dist_a, dist_b) - h * h / (4.0 * k)

    @staticmethod
    def intersection_sdf(
        dist_a: Float[jax.Array, ""], dist_b: Float[jax.Array, ""]
    ) -> Float[jax.Array, ""]:
        """SDF intersection with smooth approximation for differentiability."""
        # Use polynomial smooth maximum for better numerical stability
        k = 0.1  # smoothing parameter
        h = jnp.maximum(k - jnp.abs(dist_a - dist_b), 0.0)
        return jnp.maximum(dist_a, dist_b) + h * h / (4.0 * k)

    @staticmethod
    def difference_sdf(
        dist_a: Float[jax.Array, ""], dist_b: Float[jax.Array, ""]
    ) -> Float[jax.Array, ""]:
        """SDF difference operation with smooth approximation for differentiability."""
        # Use polynomial smooth maximum for better numerical stability
        k = 0.1  # smoothing parameter
        a = dist_a
        b = -dist_b
        h = jnp.maximum(k - jnp.abs(a - b), 0.0)
        return jnp.maximum(a, b) + h * h / (4.0 * k)


class CSGUnion(_EnhancedShapeBase):
    """Union of two shapes (A ∪ B) with enhanced algorithms."""

    def __init__(self, shape_a: Shape2D, shape_b: Shape2D) -> None:
        self.shape_a = shape_a
        self.shape_b = shape_b
        # Check if shapes support distance fields for enhanced operations
        self._has_sdf = hasattr(shape_a, "distance") and hasattr(shape_b, "distance")

    def contains(self, point: Point2D) -> bool:
        """Point is in union if it's in either shape."""
        if self._has_sdf:
            # Use SDF-based robust evaluation
            dist_a = self.shape_a.distance(point)
            dist_b = self.shape_b.distance(point)
            union_dist = _SDFOperations.union_sdf(dist_a, dist_b)
            return bool(union_dist <= 0)
        # Fallback to original method
        return self.shape_a.contains(point) or self.shape_b.contains(point)

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to union boundary."""
        dist_a = self.shape_a.distance(point)
        dist_b = self.shape_b.distance(point)
        # Union SDF: minimum of distances
        result = _SDFOperations.union_sdf(dist_a, dist_b)
        return jnp.array(result)

    def sample_boundary(self, n: int, key: jax.Array) -> Points2D:
        """Sample boundary points using enhanced filtering."""
        if self._has_sdf:
            # Enhanced sampling using distance-based filtering
            key1, key2, key3 = jax.random.split(key, 3)

            # Oversample from both shapes
            oversample_factor = 2
            points_a = self.shape_a.sample_boundary(n * oversample_factor, key1)
            points_b = self.shape_b.sample_boundary(n * oversample_factor, key2)

            # Filter points near the true boundary
            def is_boundary_point(point):
                dist_a = self.shape_a.distance(point)
                dist_b = self.shape_b.distance(point)
                union_dist = _SDFOperations.union_sdf(dist_a, dist_b)
                return jnp.abs(union_dist) < 1e-3

            all_points = jnp.concatenate([points_a, points_b], axis=0)
            boundary_mask = jax.vmap(is_boundary_point)(all_points)
            boundary_points = all_points[boundary_mask]

            # Sample n if we have more than needed
            if len(boundary_points) >= n:
                indices = jax.random.choice(key3, len(boundary_points), (n,), replace=False)
                return boundary_points[indices]
            # If not enough boundary points, fill with regular sampling
            remaining = n - len(boundary_points)
            if remaining > 0:
                extra_a = self.shape_a.sample_boundary(remaining // 2, key1)
                extra_b = self.shape_b.sample_boundary(remaining - remaining // 2, key2)
                return jnp.concatenate([boundary_points, extra_a, extra_b], axis=0)
            return boundary_points
        # Fallback to original method
        key1, key2 = jax.random.split(key)
        points_a = self.shape_a.sample_boundary(n // 2, key1)
        points_b = self.shape_b.sample_boundary(n - n // 2, key2)
        return jnp.concatenate([points_a, points_b], axis=0)

    def sample_interior(self, n: int, key: jax.Array) -> Points2D:
        """Sample points from union interior."""
        # Simple approach: sample from A and B proportionally
        key1, key2 = jax.random.split(key)
        points_a = self.shape_a.sample_interior(n // 2, key1)
        points_b = self.shape_b.sample_interior(n - n // 2, key2)
        return jnp.concatenate([points_a, points_b], axis=0)

    def compute_normal(self, point: Point2D) -> Point2D:
        """Compute normal (enhanced approach)."""
        if self._has_sdf:

            def union_distance(p):
                dist_a = self.shape_a.distance(p)
                dist_b = self.shape_b.distance(p)
                return _SDFOperations.union_sdf(dist_a, dist_b)

            gradient_fn = jax.grad(union_distance)
            normal = gradient_fn(point)
            norm = jnp.linalg.norm(normal)
            result = jnp.where(norm > 1e-10, normal / norm, jnp.array([1.0, 0.0]))
            return jnp.asarray(result).reshape(2)
        # Fallback to original method
        if self.shape_a.contains(point):
            return self.shape_a.compute_normal(point)
        return self.shape_b.compute_normal(point)


class CSGIntersection(_EnhancedShapeBase):
    """Intersection of two shapes (A ∩ B) with enhanced algorithms."""

    def __init__(self, shape_a: Shape2D, shape_b: Shape2D) -> None:
        self.shape_a = shape_a
        self.shape_b = shape_b
        self._has_sdf = hasattr(shape_a, "distance") and hasattr(shape_b, "distance")

    def contains(self, point: Point2D) -> bool:
        """Point is in intersection if it's in both shapes."""
        if self._has_sdf:
            dist_a = self.shape_a.distance(point)
            dist_b = self.shape_b.distance(point)
            intersection_dist = _SDFOperations.intersection_sdf(dist_a, dist_b)
            return bool(intersection_dist <= 0)
        return self.shape_a.contains(point) and self.shape_b.contains(point)

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to intersection boundary."""
        dist_a = self.shape_a.distance(point)
        dist_b = self.shape_b.distance(point)
        # Intersection SDF: maximum of distances
        result = _SDFOperations.intersection_sdf(dist_a, dist_b)
        return jnp.array(result)

    def sample_boundary(self, n: int, key: jax.Array) -> Points2D:
        """Sample boundary points (enhanced approach)."""
        if self._has_sdf:
            # Similar enhanced sampling as union
            key1, key2, key3 = jax.random.split(key, 3)

            oversample_factor = 3
            points_a = self.shape_a.sample_boundary(n * oversample_factor, key1)
            points_b = self.shape_b.sample_boundary(n * oversample_factor, key2)

            def is_intersection_boundary(point):
                dist_a = self.shape_a.distance(point)
                dist_b = self.shape_b.distance(point)
                intersection_dist = _SDFOperations.intersection_sdf(dist_a, dist_b)
                return jnp.abs(intersection_dist) < 1e-3

            all_points = jnp.concatenate([points_a, points_b], axis=0)
            boundary_mask = jax.vmap(is_intersection_boundary)(all_points)
            boundary_points = all_points[boundary_mask]

            if len(boundary_points) >= n:
                indices = jax.random.choice(key3, len(boundary_points), (n,), replace=False)
                return boundary_points[indices]
            if len(boundary_points) > 0:
                return boundary_points
            # Fallback if no intersection boundary found
            return jnp.zeros((1, 2))
        # Simplified implementation for non-SDF shapes
        key1, _ = jax.random.split(key)
        points_a = self.shape_a.sample_boundary(n, key1)
        mask = jax.vmap(self.shape_b.contains)(points_a)
        valid_points = points_a[mask]
        return valid_points[:n] if len(valid_points) >= n else points_a[:1]

    def sample_interior(self, n: int, key: jax.Array) -> Points2D:
        """Sample points from intersection interior."""
        # Rejection sampling from Shape A
        # Since intersection is subset of A, this is efficient if overlap is high
        candidates = self.shape_a.sample_interior(n * 2, key)
        mask = jax.vmap(self.shape_b.contains)(candidates)
        valid = candidates[mask]

        if valid.shape[0] >= n:
            return valid[:n]
        # Pad with last valid or zeros
        if valid.shape[0] > 0:
            padding = jnp.repeat(valid[-1:], n - valid.shape[0], axis=0)
            return jnp.concatenate([valid, padding], axis=0)
        return jnp.zeros((n, 2))

    def compute_normal(self, point: Point2D) -> Point2D:
        """Compute normal (enhanced approach)."""
        if self._has_sdf:

            def intersection_distance(p):
                dist_a = self.shape_a.distance(p)
                dist_b = self.shape_b.distance(p)
                return _SDFOperations.intersection_sdf(dist_a, dist_b)

            gradient_fn = jax.grad(intersection_distance)
            normal = gradient_fn(point)
            norm = jnp.linalg.norm(normal)
            result = jnp.where(norm > 1e-10, normal / norm, jnp.array([1.0, 0.0]))
            return jnp.asarray(result).reshape(2)
        # Use normal from first shape as approximation
        return self.shape_a.compute_normal(point)


class CSGDifference(_EnhancedShapeBase):
    """Difference of two shapes (A - B) with enhanced algorithms."""

    def __init__(self, shape_a: Shape2D, shape_b: Shape2D) -> None:
        self.shape_a = shape_a
        self.shape_b = shape_b
        self._has_sdf = hasattr(shape_a, "distance") and hasattr(shape_b, "distance")

    def contains(self, point: Point2D) -> bool:
        """Point is in difference if it's in A but not in B."""
        if self._has_sdf:
            dist_a = self.shape_a.distance(point)
            dist_b = self.shape_b.distance(point)
            difference_dist = _SDFOperations.difference_sdf(dist_a, dist_b)
            return bool(difference_dist <= 0)
        return self.shape_a.contains(point) and not self.shape_b.contains(point)

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to difference boundary."""
        dist_a = self.shape_a.distance(point)
        dist_b = self.shape_b.distance(point)
        # Difference SDF: maximum of first shape and negative of second
        result = _SDFOperations.difference_sdf(dist_a, dist_b)
        return jnp.array(result)

    def sample_boundary(self, n: int, key: jax.Array) -> Points2D:
        """Sample points on difference boundary."""
        # Sample candidates from shape_a boundary and filter
        candidates = self.shape_a.sample_boundary(n * 2, key)

        def is_difference_boundary(point):
            """Check if point is on difference boundary."""
            # Point is on boundary if it's on shape_a and outside shape_b
            on_a = jnp.isclose(self.shape_a.distance(point), 0.0, atol=1e-6)
            outside_b = self.shape_b.distance(point) > 1e-6
            return on_a & outside_b

        boundary_mask = jax.vmap(is_difference_boundary)(candidates)
        boundary_points = candidates[boundary_mask]

        if len(boundary_points) >= n:
            indices = jax.random.choice(key, len(boundary_points), (n,), replace=False)
            return boundary_points[indices]
        if len(boundary_points) > 0:
            return boundary_points
        # Fallback
        return self.shape_a.sample_boundary(n, key)

    def sample_interior(self, n: int, key: jax.Array) -> Points2D:
        """Sample points from difference interior (A - B)."""
        # Rejection sampling from Shape A: accept if NOT in B
        candidates = self.shape_a.sample_interior(n * 2, key)
        mask = jax.vmap(lambda p: not self.shape_b.contains(p))(candidates)
        valid = candidates[mask]

        if valid.shape[0] >= n:
            return valid[:n]
        if valid.shape[0] > 0:
            padding = jnp.repeat(valid[-1:], n - valid.shape[0], axis=0)
            return jnp.concatenate([valid, padding], axis=0)
        return jnp.zeros((n, 2))

    def compute_normal(self, point: Point2D) -> Point2D:
        """Compute normal from shape A."""
        if self._has_sdf:

            def difference_distance(p):
                dist_a = self.shape_a.distance(p)
                dist_b = self.shape_b.distance(p)
                return _SDFOperations.difference_sdf(dist_a, dist_b)

            gradient_fn = jax.grad(difference_distance)
            normal = gradient_fn(point)
            norm = jnp.linalg.norm(normal)
            result = jnp.where(norm > 1e-10, normal / norm, jnp.array([1.0, 0.0]))
            return jnp.asarray(result).reshape(2)
        return self.shape_a.compute_normal(point)


# CSG operation functions (preserving exact API)
def union(shape_a: Shape2D, shape_b: Shape2D) -> CSGUnion:
    """Create union of two shapes."""
    return CSGUnion(shape_a, shape_b)


def intersection(shape_a: Shape2D, shape_b: Shape2D) -> CSGIntersection:
    """Create intersection of two shapes."""
    return CSGIntersection(shape_a, shape_b)


def difference(shape_a: Shape2D, shape_b: Shape2D) -> CSGDifference:
    """Create difference of two shapes."""
    return CSGDifference(shape_a, shape_b)


# Boundary analysis functions (preserving exact API)
def compute_boundary_normals(shape: Shape2D, point: Point2D) -> Point2D:
    """Compute boundary normal at a point."""
    return shape.compute_normal(point)


def sample_boundary_points(shape: Shape2D, n_points: int, key: jax.Array | None = None) -> Points2D:
    """Sample points on shape boundary."""
    if key is None:
        key = jax.random.PRNGKey(42)
    return shape.sample_boundary(n=n_points, key=key)


def smooth_union(shape_a: Shape2D, shape_b: Shape2D, smoothness: float = 0.1):
    """Create smooth union with controllable blending (enhanced feature)."""

    class SmoothCSGUnion(_EnhancedShapeBase):
        def __init__(self, shape_a, shape_b, smoothness) -> None:
            self.shape_a = shape_a
            self.shape_b = shape_b
            self.smoothness = smoothness
            self._has_sdf = hasattr(shape_a, "distance") and hasattr(shape_b, "distance")

        def contains(self, point: Point2D) -> bool:
            if self._has_sdf:
                return bool(self.distance(point) <= 0)
            return self.shape_a.contains(point) or self.shape_b.contains(point)

        def distance(self, point: Point2D) -> Float[jax.Array, ""]:
            if self._has_sdf:
                dist_a = self.shape_a.distance(point)
                dist_b = self.shape_b.distance(point)
                # Smooth minimum operation
                h = jnp.maximum(self.smoothness - jnp.abs(dist_a - dist_b), 0.0) / self.smoothness
                return jnp.minimum(dist_a, dist_b) - h * h * self.smoothness * 0.25
            return jnp.array(0.0)

        def sample_boundary(self, n_points: int, key: jax.Array) -> Points2D:
            # Use combined sampling from both shapes
            key1, key2 = jax.random.split(key)
            points_a = self.shape_a.sample_boundary(n_points // 2, key1)
            points_b = self.shape_b.sample_boundary(n_points - n_points // 2, key2)
            return jnp.concatenate([points_a, points_b], axis=0)

        def compute_normal(self, point: Point2D) -> Point2D:
            if self._has_sdf:
                gradient_fn = jax.grad(self.distance)
                normal = gradient_fn(point)
                norm = jnp.linalg.norm(normal)
                result = jnp.where(norm > 1e-10, normal / norm, jnp.array([1.0, 0.0]))
                return jnp.asarray(result).reshape(2)
            # Fallback
            if self.shape_a.contains(point):
                return self.shape_a.compute_normal(point)
            return self.shape_b.compute_normal(point)

    return SmoothCSGUnion(shape_a, shape_b, smoothness)
