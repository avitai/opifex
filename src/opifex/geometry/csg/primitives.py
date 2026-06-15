"""Base 2D shape primitives: Interval, Rectangle, Circle, Polygon."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from opifex.geometry.base import Geometry
from opifex.geometry.csg.types import _EnhancedShapeBase


if TYPE_CHECKING:
    from opifex.geometry.csg.types import Point1D, Point2D, Points1D, Points2D


class Interval(Geometry):
    """1D interval [a, b] for computational domains.

    This is the 1D analog of Rectangle for use with PINNs on 1D PDEs.
    """

    def __init__(self, a: float, b: float) -> None:
        """Initialize 1D interval.

        Args:
            a: Left endpoint of the interval
            b: Right endpoint of the interval (must be > a)
        """
        if b <= a:
            raise ValueError(f"Right endpoint must be greater than left: {b} <= {a}")
        self.a = float(a)
        self.b = float(b)
        self.length = self.b - self.a

    def contains(self, point: Point1D) -> bool:
        """Check if point is inside interval (inclusive of boundary)."""
        point = jnp.asarray(point)
        x = point[0] if point.ndim > 0 else point
        return bool(self.a <= x <= self.b)

    def distance(self, point: Point1D) -> Float[jax.Array, ""]:
        """Compute signed distance to interval boundary.

        Returns negative if inside, positive if outside.
        """
        point = jnp.asarray(point)
        x = point[0] if point.ndim > 0 else point

        # Distance to left and right boundaries
        dist_left = x - self.a
        dist_right = self.b - x

        # If inside, return negative of minimum distance to boundary
        # If outside, return positive distance to nearest boundary
        inside = (dist_left >= 0) & (dist_right >= 0)
        min_inside_dist = jnp.minimum(dist_left, dist_right)

        outside_left = x < self.a
        outside_dist = jnp.where(outside_left, -dist_left, dist_right)

        return jnp.where(inside, -min_inside_dist, jnp.abs(outside_dist))

    def sample_interior(self, n: int, key: jax.Array) -> Points1D:
        """Sample points uniformly from interval interior."""
        x = jax.random.uniform(key, (n,), minval=self.a, maxval=self.b)
        return x.reshape(n, 1)

    def sample_boundary(self, n: int, key: jax.Array) -> Points1D:  # noqa: ARG002 - boundary-sampler interface receives an rng key
        """Sample points from interval boundary (endpoints).

        For 1D intervals, the boundary consists of just two points: {a, b}.
        If n <= 2, returns exactly the endpoints. Otherwise, alternates
        between a and b to fill the requested count.

        Args:
            n: Number of boundary points to return
            key: Random key (unused for 1D intervals, kept for API compatibility)

        Returns:
            Array of shape (n, 1) containing boundary points
        """
        # For 1D, boundary is just the two endpoints
        # Always include both endpoints for proper boundary conditions
        if n == 1:
            return jnp.array([[self.a]])
        if n == 2:
            return jnp.array([[self.a], [self.b]])

        # For n > 2, alternate between endpoints
        indices = jnp.arange(n) % 2
        x = jnp.where(indices == 0, self.a, self.b)
        return x.reshape(n, 1)

    def boundary_sdf(self, points: Float[jax.Array, "... d"]) -> Float[jax.Array, ...]:
        """Compute Signed Distance Function (SDF) to the boundary."""
        if points.ndim > 1:
            return jax.vmap(self.distance)(points)
        return self.distance(points)

    def get_boundary_points(self) -> Points1D:
        """Get the boundary points (both endpoints).

        Returns:
            Array of shape (2, 1) containing [a, b]
        """
        return jnp.array([[self.a], [self.b]])


class Rectangle(_EnhancedShapeBase):
    """2D rectangle shape for computational domains."""

    def __init__(self, center: Point2D, width: float, height: float) -> None:
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
            (self.x_min <= point[0] <= self.x_max) and (self.y_min <= point[1] <= self.y_max)
        )

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to rectangle boundary (smooth and differentiable)."""
        point = jnp.asarray(point)

        # Use smooth absolute value: |x| ≈ sqrt(x^2 + ε^2) - ε
        eps = 1e-8

        def smooth_abs(x):
            """Return a differentiable approximation of the absolute value."""
            return jnp.sqrt(x * x + eps * eps) - eps

        # Distance to each edge using smooth operations
        d_x = smooth_abs(point[0] - self.center[0]) - self.width / 2
        d_y = smooth_abs(point[1] - self.center[1]) - self.height / 2

        # Smooth maximum using logsumexp for better numerical stability
        def smooth_max(a, b, k=10.0):
            """Return a differentiable log-sum-exp approximation of ``max(a, b)``."""
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
            cond3 = (param >= self.width + self.height) & (param < 2 * self.width + self.height)
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

    def __init__(self, center: Point2D, radius: float) -> None:
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

    def __init__(self, vertices: Points2D) -> None:
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
        intersections = jnp.sum(jax.vmap(ray_intersects_edge)(jnp.arange(self.n_vertices)))

        # Point is inside if odd number of intersections
        return bool(intersections % 2 == 1)

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to polygon boundary (enhanced)."""
        point = jnp.asarray(point)

        # Find minimum distance to all edges
        def distance_to_edge(i):
            """Return the distance from the query point to polygon edge ``i``."""
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
            cumulative_lengths = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), edge_lengths]))

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
            """Draw interior samples by rejection sampling within the bounding box."""
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
            """Return the distance from the query point to polygon edge ``i``."""
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
