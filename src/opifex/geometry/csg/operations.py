"""CSG set operations (Union, Intersection, Difference) and helpers.

Hosts the internal SDF arithmetic (:class:`_SDFOperations`) used by every
CSG node, plus the convenience constructors (:func:`union`,
:func:`intersection`, :func:`difference`), boundary-analysis helpers
(:func:`compute_boundary_normals`, :func:`sample_boundary_points`), and the
smooth-blended :func:`smooth_union`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Float  # noqa: TC002

from opifex.geometry.csg._sampling import draw_until
from opifex.geometry.csg.types import (
    _EnhancedShapeBase,
    BoundingBox,
    Membership,
    Point2D,
    PointBatch2D,
    Points2D,
    Shape2D,
)


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
        """Store the two operand shapes of the CSG operation."""
        self.shape_a = shape_a
        self.shape_b = shape_b

    def contains(self, points: PointBatch2D) -> Membership:
        """Return whether each point lies in A or in B (DeepXDE ``CSGUnion.inside``)."""
        return self.shape_a.contains(points) | self.shape_b.contains(points)

    @property
    def bbox(self) -> BoundingBox:
        """Return the box holding both operands' boxes."""
        a, b = self.shape_a.bbox, self.shape_b.bbox
        return jnp.stack([jnp.minimum(a[0], b[0]), jnp.maximum(a[1], b[1])])

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to union boundary."""
        dist_a = self.shape_a.distance(point)
        dist_b = self.shape_b.distance(point)
        # Union SDF: minimum of distances
        result = _SDFOperations.union_sdf(dist_a, dist_b)
        return jnp.array(result)

    def sample_boundary(self, n: int, key: jax.Array) -> Points2D:
        """Sample the union's boundary: A's boundary outside B and B's outside A (DeepXDE)."""
        return draw_until(
            n,
            key,
            lambda k: _boundary_round(
                k,
                n,
                self.shape_a,
                self.shape_b,
                keep_a_inside_b=False,
                keep_b_inside_a=False,
            ),
        )

    def sample_interior(self, n: int, key: jax.Array) -> Points2D:
        """Sample the union uniformly: uniform points of its bounding box that it contains."""
        return draw_until(n, key, lambda k: _box_round(k, n, self))

    def compute_normal(self, point: Point2D) -> Point2D:
        """Compute normal (enhanced approach)."""

        def union_distance(p):
            """Return the signed distance to the union (minimum of the operands)."""
            dist_a = self.shape_a.distance(p)
            dist_b = self.shape_b.distance(p)
            return _SDFOperations.union_sdf(dist_a, dist_b)

        gradient_fn = jax.grad(union_distance)
        normal = gradient_fn(point)
        norm = jnp.linalg.norm(normal)
        result = jnp.where(norm > 1e-10, normal / norm, jnp.array([1.0, 0.0]))
        return jnp.asarray(result).reshape(2)


class CSGIntersection(_EnhancedShapeBase):
    """Intersection of two shapes (A ∩ B) with enhanced algorithms."""

    def __init__(self, shape_a: Shape2D, shape_b: Shape2D) -> None:
        """Store the two operand shapes of the CSG operation."""
        self.shape_a = shape_a
        self.shape_b = shape_b

    def contains(self, points: PointBatch2D) -> Membership:
        """Return whether each point lies in A and in B (DeepXDE ``CSGIntersection.inside``)."""
        return self.shape_a.contains(points) & self.shape_b.contains(points)

    @property
    def bbox(self) -> BoundingBox:
        """Return the overlap of the operands' boxes."""
        a, b = self.shape_a.bbox, self.shape_b.bbox
        return jnp.stack([jnp.maximum(a[0], b[0]), jnp.minimum(a[1], b[1])])

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to intersection boundary."""
        dist_a = self.shape_a.distance(point)
        dist_b = self.shape_b.distance(point)
        # Intersection SDF: maximum of distances
        result = _SDFOperations.intersection_sdf(dist_a, dist_b)
        return jnp.array(result)

    def sample_boundary(self, n: int, key: jax.Array) -> Points2D:
        """Sample the intersection's boundary: A's boundary in B and B's in A (DeepXDE)."""
        return draw_until(
            n,
            key,
            lambda k: _boundary_round(
                k,
                n,
                self.shape_a,
                self.shape_b,
                keep_a_inside_b=True,
                keep_b_inside_a=True,
            ),
        )

    def sample_interior(self, n: int, key: jax.Array) -> Points2D:
        """Sample the intersection: A's interior points that B contains (DeepXDE)."""

        def round_(round_key: jax.Array) -> Points2D:
            candidates = self.shape_a.sample_interior(n, round_key)
            return candidates[self.shape_b.contains(candidates)]

        return draw_until(n, key, round_)

    def compute_normal(self, point: Point2D) -> Point2D:
        """Compute normal (enhanced approach)."""

        def intersection_distance(p):
            """Return the signed distance to the intersection (maximum of the operands)."""
            dist_a = self.shape_a.distance(p)
            dist_b = self.shape_b.distance(p)
            return _SDFOperations.intersection_sdf(dist_a, dist_b)

        gradient_fn = jax.grad(intersection_distance)
        normal = gradient_fn(point)
        norm = jnp.linalg.norm(normal)
        result = jnp.where(norm > 1e-10, normal / norm, jnp.array([1.0, 0.0]))
        return jnp.asarray(result).reshape(2)


class CSGDifference(_EnhancedShapeBase):
    """Difference of two shapes (A - B) with enhanced algorithms."""

    def __init__(self, shape_a: Shape2D, shape_b: Shape2D) -> None:
        """Store the two operand shapes of the CSG operation."""
        self.shape_a = shape_a
        self.shape_b = shape_b

    def contains(self, points: PointBatch2D) -> Membership:
        """Return whether each point lies in A and not in B (DeepXDE ``CSGDifference.inside``).

        B's boundary belongs to B, so it is not in the difference.
        """
        return self.shape_a.contains(points) & ~self.shape_b.contains(points)

    @property
    def bbox(self) -> BoundingBox:
        """Return A's box, which holds the difference."""
        return self.shape_a.bbox

    def distance(self, point: Point2D) -> Float[jax.Array, ""]:
        """Compute signed distance to difference boundary."""
        dist_a = self.shape_a.distance(point)
        dist_b = self.shape_b.distance(point)
        # Difference SDF: maximum of first shape and negative of second
        result = _SDFOperations.difference_sdf(dist_a, dist_b)
        return jnp.array(result)

    def sample_boundary(self, n: int, key: jax.Array) -> Points2D:
        """Sample the difference's boundary: A's outside B and B's inside A, the hole (DeepXDE)."""
        return draw_until(
            n,
            key,
            lambda k: _boundary_round(
                k,
                n,
                self.shape_a,
                self.shape_b,
                keep_a_inside_b=False,
                keep_b_inside_a=True,
            ),
        )

    def sample_interior(self, n: int, key: jax.Array) -> Points2D:
        """Sample the difference: A's interior points that B does not contain (DeepXDE)."""

        def round_(round_key: jax.Array) -> Points2D:
            candidates = self.shape_a.sample_interior(n, round_key)
            return candidates[~self.shape_b.contains(candidates)]

        return draw_until(n, key, round_)

    def compute_normal(self, point: Point2D) -> Point2D:
        """Compute normal from shape A."""

        def difference_distance(p):
            """Return the signed distance to the difference of the two operands."""
            dist_a = self.shape_a.distance(p)
            dist_b = self.shape_b.distance(p)
            return _SDFOperations.difference_sdf(dist_a, dist_b)

        gradient_fn = jax.grad(difference_distance)
        normal = gradient_fn(point)
        norm = jnp.linalg.norm(normal)
        result = jnp.where(norm > 1e-10, normal / norm, jnp.array([1.0, 0.0]))
        return jnp.asarray(result).reshape(2)


def _box_round(key: jax.Array, n: int, shape: Shape2D) -> Points2D:
    """Draw ``n`` uniform points of ``shape.bbox`` and keep those ``shape`` contains."""
    bbox = shape.bbox
    candidates = jax.random.uniform(key, (n, 2)) * (bbox[1] - bbox[0]) + bbox[0]
    return candidates[shape.contains(candidates)]


def _boundary_round(
    key: jax.Array,
    n: int,
    shape_a: Shape2D,
    shape_b: Shape2D,
    *,
    keep_a_inside_b: bool,
    keep_b_inside_a: bool,
) -> Points2D:
    """Draw ``n`` boundary points of each operand, keep those on the result, and shuffle them.

    Args:
        key: Key the two boundary draws and the shuffle come from.
        n: Number of boundary points drawn from each operand.
        shape_a: First operand.
        shape_b: Second operand.
        keep_a_inside_b: Keep A's points that B contains (else those it does not).
        keep_b_inside_a: Keep B's points that A contains (else those it does not).

    Returns:
        The kept points in random order.
    """
    key_a, key_b, key_order = jax.random.split(key, 3)
    points_a = shape_a.sample_boundary(n, key_a)
    points_b = shape_b.sample_boundary(n, key_b)
    kept_a = points_a[shape_b.contains(points_a) == keep_a_inside_b]
    kept_b = points_b[shape_a.contains(points_b) == keep_b_inside_a]
    return jax.random.permutation(key_order, jnp.concatenate([kept_a, kept_b]))


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


def sample_boundary_points(shape: Shape2D, n_points: int, *, key: jax.Array) -> Points2D:
    """Sample ``n_points`` points on ``shape``'s boundary, drawn from ``key``."""
    return shape.sample_boundary(n=n_points, key=key)


def smooth_union(shape_a: Shape2D, shape_b: Shape2D, smoothness: float = 0.1):
    """Create smooth union with controllable blending (enhanced feature)."""

    class SmoothCSGUnion(_EnhancedShapeBase):
        """Union of two shapes with a smooth-minimum blend at the seam."""

        def __init__(self, shape_a, shape_b, smoothness) -> None:
            """Store the two operand shapes and the blend smoothness."""
            self.shape_a = shape_a
            self.shape_b = shape_b
            self.smoothness = smoothness

        def contains(self, points: PointBatch2D) -> Membership:
            """Return whether each point lies where the smooth-minimum distance is <= 0.

            The blend defines this shape, so its membership is its signed distance's sign.
            """
            points = jnp.asarray(points)
            flat = points.reshape(-1, 2)
            return (jax.vmap(self.distance)(flat) <= 0).reshape(points.shape[:-1])

        @property
        def bbox(self) -> BoundingBox:
            """Return the operands' joint box, widened by the blend's reach.

            The smooth minimum lowers the distance by at most ``smoothness / 4``, so the shape
            reaches at most that far beyond the plain union.
            """
            a, b = self.shape_a.bbox, self.shape_b.bbox
            reach = self.smoothness / 4
            return jnp.stack([jnp.minimum(a[0], b[0]) - reach, jnp.maximum(a[1], b[1]) + reach])

        def distance(self, point: Point2D) -> Float[jax.Array, ""]:
            """Return the smooth-minimum signed distance to the union at ``point``."""
            dist_a = self.shape_a.distance(point)
            dist_b = self.shape_b.distance(point)
            # Smooth minimum operation
            h = jnp.maximum(self.smoothness - jnp.abs(dist_a - dist_b), 0.0) / self.smoothness
            return jnp.minimum(dist_a, dist_b) - h * h * self.smoothness * 0.25

        def sample_boundary(self, n_points: int, key: jax.Array) -> Points2D:
            """Sample boundary points by combining draws from both operand shapes."""
            # Use combined sampling from both shapes
            key1, key2 = jax.random.split(key)
            points_a = self.shape_a.sample_boundary(n_points // 2, key1)
            points_b = self.shape_b.sample_boundary(n_points - n_points // 2, key2)
            return jnp.concatenate([points_a, points_b], axis=0)

        def compute_normal(self, point: Point2D) -> Point2D:
            """Return the unit outward normal as the normalised distance gradient at ``point``."""
            gradient_fn = jax.grad(self.distance)
            normal = gradient_fn(point)
            norm = jnp.linalg.norm(normal)
            result = jnp.where(norm > 1e-10, normal / norm, jnp.array([1.0, 0.0]))
            return jnp.asarray(result).reshape(2)

    return SmoothCSGUnion(shape_a, shape_b, smoothness)
