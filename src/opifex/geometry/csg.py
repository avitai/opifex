"""
Enhanced Constructive Solid Geometry (CSG) operations for Opifex geometry system.

This module provides CSG operations for 2D shapes with support for:
- Basic shapes (Rectangle, Circle, Polygon)
- Set operations (Union, Intersection, Difference)
- Boundary analysis and point sampling
- 3D molecular geometry support

Enhanced with modern techniques and optimal design patterns.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Float


# Type aliases using proper jaxtyping annotations
Point2D = Float[jax.Array, "2"]  # 2D point
Point3D = Float[jax.Array, "3"]  # 3D point
Points2D = Float[jax.Array, "n 2"]  # N x 2 array of 2D points
Points3D = Float[jax.Array, "n 3"]  # N x 3 array of 3D points


from opifex.geometry.base import Geometry


@runtime_checkable
class Shape2D(Geometry, Protocol):
    """Protocol for 2D geometric shapes."""

    @abstractmethod
    def contains(self, point: Point2D) -> bool:
        """Check if a point is contained within the shape."""
        ...

    @abstractmethod
    def compute_normal(self, point: Point2D) -> Point2D:
        """Compute outward normal at a boundary point."""
        ...

    @abstractmethod
    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to shape boundary."""
        ...

    @abstractmethod
    def sample_boundary(self, n: int, key: jax.Array) -> Points2D:
        """Sample points on the shape boundary."""
        ...


# Enhanced base functionality for shapes
class _EnhancedShapeBase:
    """Base class providing enhanced functionality to all shapes."""

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to shape boundary (enhanced feature)."""
        # Default implementation - subclasses should override for efficiency
        # This is used internally for robust CSG operations
        warnings.warn("Default distance implementation is less efficient", stacklevel=2)
        return jnp.array(0.0)

    def boundary_sdf(self, points: Float[jax.Array, "... d"]) -> Float[jax.Array, ...]:
        """Compute Signed Distance Function (SDF) to the boundary.

        Implements the Geometry protocol by delegating to distance().
        """
        # Handle batching via vmap if single point logic provided
        if points.ndim > 1:
            return jax.vmap(self.distance)(points)
        return self.distance(points)

    def sample_interior(self, n: int, key: jax.Array) -> Points2D:
        """Sample points from the interior of the geometry."""
        raise NotImplementedError("Subclasses must implement sample_interior")


class Rectangle(_EnhancedShapeBase):
    """2D rectangle shape for computational domains."""

    def __init__(self, center: Point2D, width: float, height: float):
        """
        Initialize rectangle.

        Args:
            center: Center point of the rectangle
            width: Width of the rectangle (must be positive)
            height: Height of the rectangle (must be positive)
        """
        self.center = jnp.asarray(center)
        # Keep width/height as scalars to avoid tracer leaks, but handle both types
        if hasattr(width, "shape") or hasattr(height, "shape"):  # JAX arrays
            self.width = width
            self.height = height
        else:  # Python scalars
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive")
            self.width = float(width)
            self.height = float(height)

        # Precompute bounds for efficiency
        self.x_min = self.center[0] - self.width / 2
        self.x_max = self.center[0] + self.width / 2
        self.y_min = self.center[1] - self.height / 2
        self.y_max = self.center[1] + self.height / 2

    def contains(self, point: Point2D) -> bool:
        """Check if point is inside rectangle (inclusive of boundary)."""
        point = jnp.asarray(point)
        return bool(
            (self.x_min <= point[0] <= self.x_max)
            and (self.y_min <= point[1] <= self.y_max)
        )

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to rectangle boundary (smooth and differentiable)."""
        point = jnp.asarray(point)

        # Use smooth absolute value: |x| ≈ sqrt(x^2 + ε^2) - ε
        eps = 1e-8

        def smooth_abs(x):
            return jnp.sqrt(x * x + eps * eps) - eps

        # Distance to each edge using smooth operations
        d_x = smooth_abs(point[0] - self.center[0]) - self.width / 2
        d_y = smooth_abs(point[1] - self.center[1]) - self.height / 2

        # Smooth maximum using logsumexp for better numerical stability
        def smooth_max(a, b, k=10.0):
            return jnp.logaddexp(k * a, k * b) / k

        # Combine distances for SDF using smooth operations
        zero = jnp.array(0.0)
        outside_dist = jnp.sqrt(smooth_max(d_x, zero) ** 2 + smooth_max(d_y, zero) ** 2)
        inside_dist = smooth_max(d_x, d_y)

        # Use smooth minimum to blend inside and outside distances
        # When both d_x <= 0 and d_y <= 0, we want inside_dist
        # Otherwise, we want outside_dist
        condition_value = smooth_max(-d_x, -d_y)  # positive when inside
        blend_factor = jnp.tanh(10.0 * condition_value)
        result = blend_factor * inside_dist + (1 - blend_factor) * outside_dist

        return jnp.asarray(result)

    def sample_boundary(self, n: int, key: jax.Array) -> Points2D:
        """Sample points uniformly on rectangle boundary."""
        # Total perimeter
        perimeter = 2 * (self.width + self.height)

        # Generate random parameters along perimeter
        t = jax.random.uniform(key, (n,)) * perimeter

        def point_on_boundary(param):
            """Map parameter to boundary point."""
            # Bottom edge
            cond1 = param < self.width
            p1 = jnp.array([self.x_min + param, self.y_min])

            # Right edge
            param2 = param - self.width
            cond2 = (param >= self.width) & (param < self.width + self.height)
            p2 = jnp.array([self.x_max, self.y_min + param2])

            # Top edge
            param3 = param - self.width - self.height
            cond3 = (param >= self.width + self.height) & (
                param < 2 * self.width + self.height
            )
            p3 = jnp.array([self.x_max - param3, self.y_max])

            # Left edge
            param4 = param - 2 * self.width - self.height
            p4 = jnp.array([self.x_min, self.y_max - param4])

            # Use scalar conditions with jnp.where for JAX compatibility
            result = jnp.where(
                cond1,
                p1,
                jnp.where(cond2, p2, jnp.where(cond3, p3, p4)),
            )
            return jnp.asarray(result)

        points = jax.vmap(point_on_boundary)(t)
        return jnp.asarray(points).reshape(n, 2)

    def sample_interior(self, n: int, key: jax.Array) -> Points2D:
        """Sample points uniformly from rectangle interior."""
        key1, key2 = jax.random.split(key)
        x = jax.random.uniform(key1, (n,), minval=self.x_min, maxval=self.x_max)
        y = jax.random.uniform(key2, (n,), minval=self.y_min, maxval=self.y_max)
        return jnp.stack([x, y], axis=1)

    def compute_normal(self, point: Point2D) -> Point2D:
        """Compute outward normal at boundary point."""
        point = jnp.asarray(point)

        # Determine which edge the point is on
        on_left = jnp.isclose(point[0], self.x_min, atol=1e-6)
        on_right = jnp.isclose(point[0], self.x_max, atol=1e-6)
        on_bottom = jnp.isclose(point[1], self.y_min, atol=1e-6)
        on_top = jnp.isclose(point[1], self.y_max, atol=1e-6)

        result = jnp.where(
            on_left,
            jnp.array([-1.0, 0.0]),
            jnp.where(
                on_right,
                jnp.array([1.0, 0.0]),
                jnp.where(
                    on_bottom,
                    jnp.array([0.0, -1.0]),
                    jnp.where(
                        on_top,
                        jnp.array([0.0, 1.0]),
                        jnp.array([0.0, 0.0]),  # Default for points not on boundary
                    ),
                ),
            ),
        )
        return jnp.asarray(result).reshape(2)


class Circle(_EnhancedShapeBase):
    """2D circle shape for computational domains."""

    def __init__(self, center: Point2D, radius: float):
        """
        Initialize circle.

        Args:
            center: Center point of the circle
            radius: Radius of the circle (must be positive)
        """
        self.center = jnp.asarray(center)
        # Keep radius as scalar to avoid tracer leaks, but handle both types
        if hasattr(radius, "shape"):  # JAX array
            self.radius = radius
        else:  # Python scalar
            if radius <= 0:
                raise ValueError("Radius must be positive")
            self.radius = float(radius)

    def contains(self, point: Point2D) -> bool:
        """Check if point is inside circle (inclusive of boundary)."""
        point = jnp.asarray(point)
        distance_squared = jnp.sum((point - self.center) ** 2)
        return bool(distance_squared <= self.radius**2)

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to circle boundary (smooth and differentiable)."""
        point = jnp.asarray(point)
        # Use smooth norm: ||x|| ≈ sqrt(x^2 + ε^2) - ε for differentiability at origin
        eps = 1e-8
        diff = point - self.center
        dist_to_center = jnp.sqrt(jnp.sum(diff * diff) + eps * eps) - eps
        return dist_to_center - self.radius

    def sample_boundary(self, n: int, key: jax.Array) -> Points2D:
        """Sample points uniformly on circle boundary."""
        # Generate random angles
        angles = jax.random.uniform(key, (n,)) * 2 * jnp.pi

        # Convert to Cartesian coordinates
        x = self.center[0] + self.radius * jnp.cos(angles)
        y = self.center[1] + self.radius * jnp.sin(angles)

        return jnp.stack([x, y], axis=1)

    def sample_interior(self, n: int, key: jax.Array) -> Points2D:
        """Sample points uniformly from circle interior."""
        key1, key2 = jax.random.split(key)
        # Rejection sampling or polar coordinates with sqrt(r)
        theta = jax.random.uniform(key1, (n,), maxval=2 * jnp.pi)
        r = jnp.sqrt(jax.random.uniform(key2, (n,))) * self.radius

        x = self.center[0] + r * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(theta)
        return jnp.stack([x, y], axis=1)

    def compute_normal(self, point: Point2D) -> Point2D:
        """Compute outward normal at boundary point."""
        point = jnp.asarray(point)

        # Normal is the direction from center to point
        direction = point - self.center
        # Normalize to unit vector
        norm = jnp.linalg.norm(direction)

        # Handle the case where point is at center
        normal = jnp.where(
            norm > 1e-12,
            direction / norm,
            jnp.array([1.0, 0.0]),  # Default direction if at center
        )

        return jnp.asarray(normal)


class Polygon(_EnhancedShapeBase):
    """2D polygon shape defined by vertices."""

    def __init__(self, vertices: Points2D):
        """
        Initialize polygon from vertices.

        Args:
            vertices: Array of vertex coordinates, shape (N, 2) where N >= 3

        Raises:
            ValueError: If fewer than 3 vertices provided
        """
        vertices = jnp.asarray(vertices)
        if vertices.shape[0] < 3:
            raise ValueError("Polygon must have at least 3 vertices")

        self.vertices = jnp.asarray(vertices)
        self.n_vertices = vertices.shape[0]

    def contains(self, point: Point2D) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        point = jnp.asarray(point)

        def ray_intersects_edge(i):
            """Check if horizontal ray from point intersects edge i."""
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % self.n_vertices]

            # Check if ray can intersect (y-coordinate conditions)
            y_check = (v1[1] > point[1]) != (v2[1] > point[1])

            # Compute x-intersection point
            x_intersect = v1[0] + (point[1] - v1[1]) / (v2[1] - v1[1]) * (v2[0] - v1[0])

            # Ray intersects if intersection is to the right of the point
            return y_check & (point[0] < x_intersect)

        # Count intersections
        intersections = jnp.sum(
            jax.vmap(ray_intersects_edge)(jnp.arange(self.n_vertices))
        )

        # Point is inside if odd number of intersections
        return bool(intersections % 2 == 1)

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to polygon boundary (enhanced)."""
        point = jnp.asarray(point)

        # Find minimum distance to all edges
        def distance_to_edge(i):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % self.n_vertices]

            # Vector from v1 to v2
            edge_vec = v2 - v1
            # Vector from v1 to point
            point_vec = point - v1

            # Project point onto edge line
            edge_length_sq = jnp.sum(edge_vec**2)
            t = jnp.clip(jnp.dot(point_vec, edge_vec) / edge_length_sq, 0.0, 1.0)

            # Closest point on edge
            closest = v1 + t * edge_vec
            return jnp.linalg.norm(point - closest)

        distances = jax.vmap(distance_to_edge)(jnp.arange(self.n_vertices))
        min_dist = jnp.min(distances)

        # Determine sign based on containment
        inside = self.contains(point)
        return jnp.where(inside, -min_dist, min_dist)

    def sample_boundary(self, n: int, key: jax.Array) -> Points2D:
        """Sample points uniformly on polygon boundary."""
        # Compute edge lengths
        edges = jnp.roll(self.vertices, -1, axis=0) - self.vertices
        edge_lengths = jnp.linalg.norm(edges, axis=1)
        total_perimeter = jnp.sum(edge_lengths)

        # Generate random parameters along perimeter
        t = jax.random.uniform(key, (n,)) * total_perimeter

        def point_on_boundary(param):
            """Map parameter to boundary point."""
            cumulative_lengths = jnp.cumsum(
                jnp.concatenate([jnp.array([0.0]), edge_lengths])
            )

            # Find which edge the parameter corresponds to
            edge_idx = jnp.searchsorted(cumulative_lengths[1:], param, side="right")
            edge_idx = jnp.clip(edge_idx, 0, self.n_vertices - 1)

            # Parameter along the specific edge
            edge_param = (param - cumulative_lengths[edge_idx]) / edge_lengths[edge_idx]
            edge_param = jnp.clip(edge_param, 0.0, 1.0)

            # Interpolate along edge
            v1 = self.vertices[edge_idx]
            v2 = self.vertices[(edge_idx + 1) % self.n_vertices]

            return v1 + edge_param * (v2 - v1)

        result = jax.vmap(point_on_boundary)(t)
        return jnp.asarray(result)

    def sample_interior(self, n: int, key: jax.Array) -> Points2D:
        """Sample points from polygon interior using rejection sampling."""
        # Find bounding box
        min_vals = jnp.min(self.vertices, axis=0)
        max_vals = jnp.max(self.vertices, axis=0)

        # Simple rejection sampling
        # Note: For complex polygons, ear clipping triangulation is better
        # but more complex
        def rejection_sample(current_key, num_needed):
            # Generate proposals
            key1, key2 = jax.random.split(current_key)
            proposals_x = jax.random.uniform(
                key1, (num_needed * 2,), minval=min_vals[0], maxval=max_vals[0]
            )
            proposals_y = jax.random.uniform(
                key2, (num_needed * 2,), minval=min_vals[1], maxval=max_vals[1]
            )
            proposals = jnp.stack([proposals_x, proposals_y], axis=1)

            # Check containment
            mask = jax.vmap(self.contains)(proposals)
            return proposals[mask]

        # Initial batch
        valid_points = rejection_sample(key, n)

        # Pad or slice to get exactly n
        # This is a naive implementation; production code might iterate or use dynamic
        # shapes if allowed. For fixed shape JAX, we typically oversample and then
        # mask/pad.
        if valid_points.shape[0] >= n:
            return valid_points[:n]

        # If not enough, pad with last point (not ideal but safe for array shapes)
        padding = jnp.repeat(valid_points[-1:], n - valid_points.shape[0], axis=0)
        return jnp.concatenate([valid_points, padding], axis=0)

    def compute_normal(self, point: Point2D) -> Point2D:
        """Compute outward normal at boundary point."""
        point = jnp.asarray(point)

        # Find closest edge
        def distance_to_edge(i):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % self.n_vertices]

            # Project point onto edge
            edge_vec = v2 - v1
            edge_length_sq = jnp.sum(edge_vec**2)

            t = jnp.clip(jnp.dot(point - v1, edge_vec) / edge_length_sq, 0.0, 1.0)
            closest_point = v1 + t * edge_vec

            return jnp.linalg.norm(point - closest_point)

        distances = jax.vmap(distance_to_edge)(jnp.arange(self.n_vertices))
        closest_edge = jnp.argmin(distances)

        # Compute normal for closest edge
        v1 = self.vertices[closest_edge]
        v2 = self.vertices[(closest_edge + 1) % self.n_vertices]
        edge_vec = v2 - v1

        # Perpendicular vector (rotated 90 degrees)
        normal = jnp.array([-edge_vec[1], edge_vec[0]])
        return normal / jnp.linalg.norm(normal)


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

    def __init__(self, shape_a: Shape2D, shape_b: Shape2D):
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
                indices = jax.random.choice(
                    key3, len(boundary_points), (n,), replace=False
                )
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

    def __init__(self, shape_a: Shape2D, shape_b: Shape2D):
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
                indices = jax.random.choice(
                    key3, len(boundary_points), (n,), replace=False
                )
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

    def __init__(self, shape_a: Shape2D, shape_b: Shape2D):
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


def sample_boundary_points(
    shape: Shape2D, n_points: int, key: jax.Array | None = None
) -> Points2D:
    """Sample points on shape boundary."""
    if key is None:
        key = jax.random.PRNGKey(42)
    return shape.sample_boundary(n=n_points, key=key)


# 3D Molecular Geometry Support (preserving exact API)
class MolecularGeometry:
    """3D molecular geometry with atomic coordinates."""

    def __init__(self, atomic_symbols: list[str], positions: jax.Array):
        """Initialize molecular geometry.

        Args:
            atomic_symbols: List of atomic symbols (e.g., ['H', 'H', 'O'])
            positions: Atomic positions in Bohr, shape (N, 3)

        Raises:
            ValueError: If number of symbols doesn't match number of positions
        """
        positions = jnp.asarray(positions)

        if len(atomic_symbols) != positions.shape[0]:
            raise ValueError("Number of atomic symbols must match number of positions")

        self.atomic_symbols = atomic_symbols
        self.positions = positions
        self.n_atoms = len(atomic_symbols)

    def compute_distances(self) -> jax.Array:
        """Compute all pairwise interatomic distances."""
        # Compute pairwise distance matrix
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        return jnp.linalg.norm(diff, axis=2)

    def project_to_2d(self, plane: str = "xy") -> jax.Array:
        """Project 3D coordinates to 2D plane."""
        if plane == "xy":
            return self.positions[:, :2]
        if plane == "xz":
            return self.positions[:, [0, 2]]
        if plane == "yz":
            return self.positions[:, [1, 2]]
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'")

    @classmethod
    def from_molecular_system(cls, molecular_system) -> MolecularGeometry:
        """Create molecular geometry from MolecularSystem."""
        # Extract atomic symbols
        atomic_symbols = cls._extract_atomic_symbols(molecular_system)

        # Extract positions
        positions = cls._extract_positions(molecular_system)

        if atomic_symbols is None or positions is None:
            # Fallback: inspect the molecular system object for debugging
            available_attrs = [
                attr for attr in dir(molecular_system) if not attr.startswith("_")
            ]
            raise ValueError(
                f"Molecular system must have atomic symbols and positions. "
                f"Available attributes: {available_attrs}. "
                f"Found atomic_symbols: {atomic_symbols is not None}, "
                f"Found positions: {positions is not None}"
            )

        return cls(atomic_symbols, positions)

    @classmethod
    def _extract_atomic_symbols(cls, molecular_system):
        """Extract atomic symbols from molecular system."""
        # Check for atomic_symbols attribute
        if hasattr(molecular_system, "atomic_symbols"):
            return molecular_system.atomic_symbols
        if hasattr(molecular_system, "symbols"):
            return molecular_system.symbols

        # Check atoms attribute - fix nested if statements
        if (
            hasattr(molecular_system, "atoms")
            and isinstance(molecular_system.atoms, list)
            and len(molecular_system.atoms) > 0
            and isinstance(molecular_system.atoms[0], tuple)
        ):
            return [atom[0] for atom in molecular_system.atoms]

        if hasattr(molecular_system, "species"):
            return molecular_system.species

        # Convert atomic numbers to symbols (fallback)
        if hasattr(molecular_system, "atomic_numbers"):
            atomic_number_to_symbol = {
                1: "H",
                2: "He",
                3: "Li",
                4: "Be",
                5: "B",
                6: "C",
                7: "N",
                8: "O",
                9: "F",
                10: "Ne",
                11: "Na",
                12: "Mg",
                13: "Al",
                14: "Si",
                15: "P",
                16: "S",
                17: "Cl",
                18: "Ar",
            }
            return [
                atomic_number_to_symbol.get(num, f"X{num}")
                for num in molecular_system.atomic_numbers
            ]
        return None

    @classmethod
    def _extract_positions(cls, molecular_system):
        """Extract positions from molecular system."""
        if hasattr(molecular_system, "positions"):
            return molecular_system.positions
        if hasattr(molecular_system, "coords"):
            return molecular_system.coords
        if hasattr(molecular_system, "geometry"):
            return molecular_system.geometry

        # Check atoms attribute - fix nested if statements
        if (
            hasattr(molecular_system, "atoms")
            and isinstance(molecular_system.atoms, list)
            and len(molecular_system.atoms) > 0
            and isinstance(molecular_system.atoms[0], tuple)
        ):
            return jnp.array([atom[1] for atom in molecular_system.atoms])
        return None


class PeriodicCell:
    """Periodic boundary conditions for materials systems."""

    def __init__(self, lattice_vectors: jax.Array):
        """Initialize periodic cell.

        Args:
            lattice_vectors: 3x3 array of lattice vectors in Bohr
        """
        self.lattice_vectors = jnp.asarray(lattice_vectors)
        if self.lattice_vectors.shape != (3, 3):
            raise ValueError("Lattice vectors must be 3x3 array")

        # Precompute reciprocal lattice vectors for efficiency
        self.reciprocal_vectors = jnp.linalg.inv(self.lattice_vectors).T

    @property
    def volume(self) -> float:
        """Compute volume of the unit cell."""
        return float(jnp.abs(jnp.linalg.det(self.lattice_vectors)))

    def wrap_coordinates(self, positions: jax.Array) -> jax.Array:
        """Wrap coordinates into unit cell."""
        # Convert to fractional coordinates
        fractional = jnp.linalg.solve(self.lattice_vectors.T, positions.T).T
        # Wrap to [0, 1)
        fractional_wrapped = fractional % 1.0
        # Convert back to Cartesian
        return fractional_wrapped @ self.lattice_vectors

    def wrap_to_unit_cell(self, point: Point3D) -> Point3D:
        """Wrap a single point to unit cell [0, 1)³."""
        # Convert to fractional coordinates
        fractional = jnp.dot(point, self.reciprocal_vectors)
        # Wrap to [0, 1)
        wrapped_fractional = fractional - jnp.floor(fractional)
        # Convert back to Cartesian coordinates
        return jnp.dot(wrapped_fractional, self.lattice_vectors)

    def periodic_distance(self, point1: Point3D, point2: Point3D) -> jax.Array:
        """Compute minimum distance between points considering periodicity."""
        # Convert to fractional coordinates
        frac1 = jnp.dot(point1, self.reciprocal_vectors)
        frac2 = jnp.dot(point2, self.reciprocal_vectors)

        # Compute minimum image difference
        diff_frac = frac2 - frac1
        diff_frac = diff_frac - jnp.round(diff_frac)  # Wrap to [-0.5, 0.5)

        # Convert back to Cartesian and compute distance
        diff_cart = jnp.dot(diff_frac, self.lattice_vectors)
        return jnp.linalg.norm(diff_cart)

    def minimum_image_distance(self, pos1: jax.Array, pos2: jax.Array) -> jax.Array:
        """Compute minimum image distance between two positions."""
        # Convert to fractional coordinates
        frac1 = jnp.linalg.solve(self.lattice_vectors.T, pos1.T).T
        frac2 = jnp.linalg.solve(self.lattice_vectors.T, pos2.T).T

        # Compute fractional displacement
        frac_disp = frac2 - frac1
        # Apply minimum image convention
        frac_disp = frac_disp - jnp.round(frac_disp)

        # Convert back to Cartesian
        cart_disp = frac_disp @ self.lattice_vectors
        return jnp.linalg.norm(cart_disp)

    def find_neighbors(
        self, positions: Points3D, cutoff_radius: float
    ) -> list[tuple[int, int, float]]:
        """Find neighboring atoms within cutoff radius considering periodicity.

        Args:
            positions: Atomic positions, shape (N, 3)
            cutoff_radius: Cutoff distance for neighbors

        Returns:
            List of (atom1_idx, atom2_idx, distance) tuples
        """
        n_atoms = positions.shape[0]

        # Create all pairwise combinations using JAX vectorized operations
        i_indices, j_indices = jnp.meshgrid(
            jnp.arange(n_atoms), jnp.arange(n_atoms), indexing="ij"
        )

        # Only consider upper triangular pairs (i < j)
        upper_tri_mask = i_indices < j_indices

        # Get valid pairs
        valid_i = i_indices[upper_tri_mask]
        valid_j = j_indices[upper_tri_mask]

        # Compute distances for all valid pairs using vmap
        def compute_pair_distance(i, j):
            return self.periodic_distance(positions[i], positions[j])

        distances = jax.vmap(compute_pair_distance)(valid_i, valid_j)

        # Filter by cutoff radius
        within_cutoff = distances <= cutoff_radius

        # Extract results
        neighbor_i = valid_i[within_cutoff]
        neighbor_j = valid_j[within_cutoff]
        neighbor_distances = distances[within_cutoff]

        # Convert to list of tuples for compatibility
        neighbors = []
        for idx in range(neighbor_i.shape[0]):
            neighbors.append(
                (
                    int(neighbor_i[idx]),
                    int(neighbor_j[idx]),
                    float(neighbor_distances[idx]),
                )
            )

        return neighbors


# Additional utility functions with optimal design patterns
def create_computational_domain_with_molecular_exclusion(
    domain_shape: Shape2D,
    molecular_geometry: MolecularGeometry,
    exclusion_radius: float = 1.0,
) -> Shape2D:
    """Create computational domain with molecular exclusion zones."""
    # Project molecular geometry to 2D
    projected_positions = molecular_geometry.project_to_2d()

    # Create exclusion zones around atoms using simple loop (more compatible)
    result_domain = domain_shape

    # Apply exclusions one by one
    for i in range(projected_positions.shape[0]):
        pos = projected_positions[i]
        exclusion_zone = Circle(center=pos, radius=exclusion_radius)
        result_domain = difference(result_domain, exclusion_zone)

    return result_domain


def create_molecular_geometry_from_dft_problem(dft_problem) -> MolecularGeometry:
    """Create molecular geometry from DFT problem specification."""
    # Extract molecular information from DFT problem
    if hasattr(dft_problem, "molecular_system"):
        mol_sys = dft_problem.molecular_system
        return MolecularGeometry.from_molecular_system(mol_sys)
    raise ValueError("DFT problem must have molecular_system attribute")


# Enhanced utilities (new features with optimal design patterns)
def ensure_safe_jax_environment():
    """Ensure JAX environment is properly configured for reliable operation."""
    try:
        # Import from the core testing infrastructure if available
        from opifex.core.testing_infrastructure import (
            ensure_safe_jax_environment as core_ensure_safe,
        )

        return core_ensure_safe()
    except ImportError:
        # Fallback implementation for standalone use - test basic JAX operation
        try:
            test_array = jnp.array([1.0, 2.0, 3.0])
            result = jnp.sum(test_array)
            jax.block_until_ready(result)  # Force computation
        except Exception as e:
            warnings.warn(
                f"JAX environment issue detected: {e}. Check JAX installation.",
                stacklevel=2,
            )


def compute_shape_area(
    shape: Shape2D, bbox: jax.Array | None = None, resolution: int = 1000
) -> float:
    """Compute area of shape using Monte Carlo integration (enhanced feature)."""
    if bbox is None:
        # Estimate bounding box by sampling boundary
        key = jax.random.PRNGKey(42)
        boundary_points = shape.sample_boundary(100, key)
        bbox_min = jnp.min(boundary_points, axis=0)
        bbox_max = jnp.max(boundary_points, axis=0)
        # Add some padding
        padding = 0.1 * (bbox_max - bbox_min)
        bbox = jnp.array([bbox_min - padding, bbox_max + padding])

    # Generate random points in bounding box
    key = jax.random.PRNGKey(123)
    size = bbox[1] - bbox[0]
    points = jax.random.uniform(key, (resolution, 2)) * size + bbox[0]

    # Count points inside shape
    inside_points = jax.vmap(shape.contains)(points)
    area_fraction = jnp.mean(inside_points)

    # Scale by bounding box area
    bbox_area = jnp.prod(size)
    return float(area_fraction * bbox_area)


def smooth_union(shape_a: Shape2D, shape_b: Shape2D, smoothness: float = 0.1):
    """Create smooth union with controllable blending (enhanced feature)."""

    class SmoothCSGUnion(_EnhancedShapeBase):
        def __init__(self, shape_a, shape_b, smoothness):
            self.shape_a = shape_a
            self.shape_b = shape_b
            self.smoothness = smoothness
            self._has_sdf = hasattr(shape_a, "distance") and hasattr(
                shape_b, "distance"
            )

        def contains(self, point: Point2D) -> bool:
            if self._has_sdf:
                return bool(self.distance(point) <= 0)
            return self.shape_a.contains(point) or self.shape_b.contains(point)

        def distance(self, point: Point2D) -> Float[jax.Array, ""]:
            if self._has_sdf:
                dist_a = self.shape_a.distance(point)
                dist_b = self.shape_b.distance(point)
                # Smooth minimum operation
                h = (
                    jnp.maximum(self.smoothness - jnp.abs(dist_a - dist_b), 0.0)
                    / self.smoothness
                )
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


# Validation and testing support
def validate_implementation():
    """Validate that the enhanced implementation works correctly."""
    import logging

    logging.info("Validating enhanced CSG implementation...")

    # Test basic shapes
    rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.0)
    circle = Circle(center=jnp.array([0.0, 0.0]), radius=1.0)

    # Test contains - replace asserts with proper validation
    if not rect.contains(jnp.array([0.5, 0.25])):
        raise ValueError("Rectangle contains test failed")
    if not circle.contains(jnp.array([0.5, 0.0])):
        raise ValueError("Circle contains test failed")

    # Test CSG operations
    union_shape = union(rect, circle)
    intersection_shape = intersection(rect, circle)
    difference_shape = difference(rect, circle)

    if not union_shape.contains(jnp.array([0.0, 0.0])):
        raise ValueError("Union shape test failed")
    if not intersection_shape.contains(jnp.array([0.0, 0.0])):
        raise ValueError("Intersection shape test failed")
    if not difference_shape.contains(jnp.array([0.8, 0.0])):
        raise ValueError("Difference shape test failed")

    # Test enhanced features
    if hasattr(rect, "distance"):
        dist = rect.distance(jnp.array([1.5, 0.0]))
        if not (dist > 0):  # Outside rectangle
            raise ValueError("Distance test failed")

    # Test boundary sampling
    key = jax.random.PRNGKey(42)
    boundary_points = rect.sample_boundary(50, key)
    if boundary_points.shape != (50, 2):
        raise ValueError("Boundary sampling test failed")

    logging.info("All validation tests passed!")


if __name__ == "__main__":
    validate_implementation()
