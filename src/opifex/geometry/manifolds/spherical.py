"""Spherical manifold implementation with geodesic operations.

This module implements spherical manifolds S^n with JAX-native operations
for scientific machine learning applications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from jaxtyping import Float

    from opifex.geometry.manifolds.base import (
        ManifoldPoint,
        MetricTensor,
        TangentVector,
    )


class SphericalManifold:
    """Spherical manifold S^n implementation with geodesic operations.

    Implements n-dimensional sphere embedded in (n+1)-dimensional Euclidean space.
    All operations are JAX-compatible and autodifferentiable.
    """

    def __init__(self, radius: float = 1.0, dimension: int | None = None):
        """Initialize spherical manifold.

        Args:
            radius: Radius of the sphere (default: 1.0)
            dimension: Intrinsic dimension (inferred from data if None)
        """
        self.radius = radius
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of the sphere."""
        if self._dimension is None:
            raise ValueError("Dimension must be set or inferred from data")
        return self._dimension

    @property
    def embedding_dimension(self) -> int:
        """Dimension of ambient Euclidean space."""
        return self.dimension + 1

    def _validate_point(self, point: ManifoldPoint) -> ManifoldPoint:
        """Validate and project point onto sphere."""
        # Project onto sphere surface
        norm = jnp.linalg.norm(point, axis=-1, keepdims=True)
        return self.radius * point / norm

    def exp_map(self, base: ManifoldPoint, tangent: TangentVector) -> ManifoldPoint:
        """Exponential map: tangent space → sphere via geodesics.

        Args:
            base: Base point on sphere
            tangent: Tangent vector at base point

        Returns:
            Point on sphere reached by geodesic
        """
        # Ensure base point is on sphere
        base = self._validate_point(base)

        # Compute tangent vector norm
        tangent_norm = jnp.linalg.norm(tangent, axis=-1, keepdims=True)

        # Handle zero tangent vector
        safe_norm = jnp.where(tangent_norm > 1e-8, tangent_norm, 1.0)
        unit_tangent = tangent / jnp.asarray(safe_norm)

        # Geodesic on sphere: great circle parametrization
        result = base * jnp.cos(tangent_norm / self.radius) + unit_tangent * jnp.sin(
            tangent_norm / self.radius
        )

        # Handle zero tangent case
        return jnp.where(tangent_norm > 1e-8, result, base)

    def log_map(self, base: ManifoldPoint, point: ManifoldPoint) -> TangentVector:
        """Logarithmic map: sphere → tangent space.

        Args:
            base: Base point for tangent space
            point: Target point on sphere

        Returns:
            Tangent vector pointing from base toward point
        """
        base = self._validate_point(base)
        point = self._validate_point(point)

        # Compute angle between points
        dot_product = jnp.sum(base * point, axis=-1, keepdims=True) / (self.radius**2)
        # Clamp to handle numerical errors
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        angle = jnp.arccos(dot_product)

        # Compute tangent direction
        tangent_direction = point - base * dot_product
        tangent_norm = jnp.linalg.norm(tangent_direction, axis=-1, keepdims=True)

        # Handle antipodal points (non-unique geodesic)
        safe_norm = jnp.where(tangent_norm > 1e-8, tangent_norm, 1.0)
        unit_direction = tangent_direction / jnp.asarray(safe_norm)

        # Scale by geodesic distance
        result = self.radius * angle * unit_direction

        # Handle same point case
        return jnp.where(tangent_norm > 1e-8, result, jnp.zeros_like(point))

    def geodesic_distance(
        self, point1: ManifoldPoint, point2: ManifoldPoint
    ) -> Float[jax.Array, ...]:
        """Compute geodesic distance between points on sphere.

        Args:
            point1: First point on sphere
            point2: Second point on sphere

        Returns:
            Geodesic distance (great circle distance)
        """
        point1 = self._validate_point(point1)
        point2 = self._validate_point(point2)

        # Compute angle via dot product
        dot_product = jnp.sum(point1 * point2, axis=-1) / (self.radius**2)
        dot_product = jnp.clip(dot_product, -1.0, 1.0)

        return self.radius * jnp.arccos(dot_product)

    def metric_tensor(self, point: ManifoldPoint) -> MetricTensor:
        """Riemannian metric tensor for sphere (induced from Euclidean).

        Args:
            point: Point on sphere

        Returns:
            Metric tensor matrix at point
        """
        point = self._validate_point(point)

        # For sphere, metric is identity in tangent space coordinates
        # This is a simplified implementation - full implementation would
        # require coordinate chart specification
        dim = point.shape[-1] - 1  # Intrinsic dimension
        return jnp.eye(dim) * (self.radius**2)

    def random_point(self, key: jax.Array, shape: tuple = ()) -> ManifoldPoint:
        """Generate random point on sphere.

        Args:
            key: JAX random key
            shape: Shape prefix for batch dimensions

        Returns:
            Random point uniformly distributed on sphere
        """
        # Generate Gaussian random vector and normalize
        if self._dimension is None:
            raise ValueError("Dimension must be set to generate random points")

        full_shape = (*shape, self.embedding_dimension)
        gaussian = jax.random.normal(key, full_shape)

        # Normalize to sphere
        norm = jnp.linalg.norm(gaussian, axis=-1, keepdims=True)
        return self.radius * gaussian / norm

    # Geometry Protocol Implementation
    def sample_interior(self, n: int, key: jax.Array) -> ManifoldPoint:
        """Sample n points from the interior of the geometry (the sphere itself)."""
        return self.random_point(key, shape=(n,))

    def sample_boundary(self, n: int, key: jax.Array) -> ManifoldPoint:
        """Sample n points from the boundary. Sphere has no boundary."""
        # Return empty array with correct shape (0, embedding_dimension)
        # We ignore n because there are no points to sample
        return jnp.zeros((0, self.embedding_dimension))

    def boundary_sdf(self, points: ManifoldPoint) -> Float[jax.Array, ...]:
        """Compute SDF to the boundary. Sphere has no boundary."""
        # Distance to empty boundary is infinity
        shape = points.shape[:-1]
        return jnp.full(shape, jnp.inf)


# JAX pytree registration
def _spherical_manifold_tree_flatten(manifold):
    children = (manifold.radius,)
    aux_data = (manifold._dimension,)
    return children, aux_data


def _spherical_manifold_tree_unflatten(aux_data, children):
    (radius,) = children
    (dimension,) = aux_data
    return SphericalManifold(radius=radius, dimension=dimension)


jax.tree_util.register_pytree_node(
    SphericalManifold,
    _spherical_manifold_tree_flatten,
    _spherical_manifold_tree_unflatten,
)
