"""Hyperbolic manifold implementation with Poincaré disk model.

This module implements hyperbolic manifolds H^n using the Poincaré disk model
with gyrovector space formalism for scientific machine learning applications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from jaxtyping import Float

    from opifex.geometry.manifolds.base import (
        ConnectionForm,
        ManifoldPoint,
        MetricTensor,
        TangentVector,
    )


class HyperbolicManifold:
    """Hyperbolic manifold H^n using Poincaré disk model.

    Implements n-dimensional hyperbolic space in the Poincaré disk model
    with gyrovector space operations. All operations are JAX-compatible.
    """

    def __init__(self, curvature: float = -1.0, dimension: int | None = None):
        """Initialize hyperbolic manifold.

        Args:
            curvature: Sectional curvature (negative, default: -1.0)
            dimension: Intrinsic dimension (inferred from data if None)
        """
        # Use JAX-compatible validation
        if not isinstance(curvature, (jax.Array, jnp.ndarray)) and curvature >= 0:
            raise ValueError("Hyperbolic manifold requires negative curvature")
        self.curvature = curvature
        self._dimension = dimension
        # Radius of curvature
        self.radius = jnp.sqrt(-1.0 / curvature)

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of hyperbolic space."""
        if self._dimension is None:
            raise ValueError("Dimension must be set or inferred from data")
        return self._dimension

    @property
    def embedding_dimension(self) -> int:
        """Dimension of Poincaré disk (same as intrinsic dimension)."""
        return self.dimension

    def _validate_point(self, point: ManifoldPoint) -> ManifoldPoint:
        """Validate and project point into Poincaré disk."""
        # Ensure point is inside unit disk
        norm = jnp.linalg.norm(point, axis=-1, keepdims=True)
        # Project points outside disk to boundary (with small epsilon)
        max_norm = 1.0 - 1e-6
        # Use jnp.where and ensure we return a jax.Array
        return jnp.asarray(jnp.where(norm >= 1.0, max_norm * point / norm, point))

    def _gyroaddition(self, u: ManifoldPoint, v: ManifoldPoint) -> ManifoldPoint:
        """Gyrovector addition in Poincaré disk model.

        Implements the Möbius addition formula for hyperbolic geometry.
        """
        u = self._validate_point(u)
        v = self._validate_point(v)

        # Möbius addition formula
        u_dot_v = jnp.sum(u * v, axis=-1, keepdims=True)
        u_norm_sq = jnp.sum(u * u, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(v * v, axis=-1, keepdims=True)

        numerator = (1 + 2 * u_dot_v + v_norm_sq) * u + (1 - u_norm_sq) * v
        denominator = 1 + 2 * u_dot_v + u_norm_sq * v_norm_sq

        result = numerator / denominator
        return self._validate_point(result)

    def exp_map(self, base: ManifoldPoint, tangent: TangentVector) -> ManifoldPoint:
        """Exponential map: tangent space → hyperbolic space via geodesics.

        Args:
            base: Base point in Poincaré disk
            tangent: Tangent vector at base point

        Returns:
            Point on hyperbolic space reached by geodesic
        """
        base = self._validate_point(base)

        # Lambda factor for gyrovector operations
        base_norm_sq = jnp.sum(base * base, axis=-1, keepdims=True)
        lambda_base = 2.0 / (1.0 - base_norm_sq)

        # Scaled tangent vector
        scaled_tangent = tangent / lambda_base

        # Tangent vector norm
        tangent_norm = jnp.linalg.norm(scaled_tangent, axis=-1, keepdims=True)

        # Handle zero tangent vector
        safe_norm = jnp.where(tangent_norm > 1e-8, tangent_norm, 1.0)
        unit_tangent = scaled_tangent / jnp.asarray(safe_norm)

        # Hyperbolic distance factor
        tanh_factor = jnp.tanh(tangent_norm / self.radius)
        scaled_unit = tanh_factor * unit_tangent

        # Gyroaddition of base point with scaled tangent
        result = self._gyroaddition(base, scaled_unit)

        # Handle zero tangent case
        return jnp.where(tangent_norm > 1e-8, result, base)

    def log_map(self, base: ManifoldPoint, point: ManifoldPoint) -> TangentVector:
        """Logarithmic map: hyperbolic space → tangent space.

        Args:
            base: Base point for tangent space
            point: Target point on hyperbolic space

        Returns:
            Tangent vector pointing from base toward point
        """
        base = self._validate_point(base)
        point = self._validate_point(point)

        # Gyrosubtraction: -base ⊕ point
        neg_base = -base
        diff = self._gyroaddition(neg_base, point)

        # Lambda factor at base
        base_norm_sq = jnp.sum(base * base, axis=-1, keepdims=True)
        lambda_base = 2.0 / (1.0 - base_norm_sq)

        # Hyperbolic distance and direction
        diff_norm = jnp.linalg.norm(diff, axis=-1, keepdims=True)

        # Handle same point case
        safe_norm = jnp.where(diff_norm > 1e-8, diff_norm, 1.0)
        unit_diff = diff / jnp.asarray(safe_norm)

        # Inverse hyperbolic tangent for distance
        clamped_norm = jnp.clip(diff_norm, 0.0, 1.0 - 1e-6)
        artanh_factor = jnp.arctanh(clamped_norm)

        # Scale by lambda factor and radius
        result = lambda_base * self.radius * artanh_factor * unit_diff

        # Handle same point case
        return jnp.where(diff_norm > 1e-8, result, jnp.zeros_like(point))

    def geodesic_distance(
        self, point1: ManifoldPoint, point2: ManifoldPoint
    ) -> Float[jax.Array, ...]:
        """Compute geodesic distance between points in hyperbolic space.

        Args:
            point1: First point in Poincaré disk
            point2: Second point in Poincaré disk

        Returns:
            Hyperbolic geodesic distance
        """
        point1 = self._validate_point(point1)
        point2 = self._validate_point(point2)

        # Poincaré distance formula
        diff = point1 - point2
        diff_norm_sq = jnp.sum(diff * diff, axis=-1)

        p1_norm_sq = jnp.sum(point1 * point1, axis=-1)
        p2_norm_sq = jnp.sum(point2 * point2, axis=-1)

        # Avoid division by zero
        denominator = (1 - p1_norm_sq) * (1 - p2_norm_sq)
        safe_denominator = jnp.where(denominator > 1e-8, denominator, 1e-8)

        # Hyperbolic distance formula
        ratio = 1 + 2 * diff_norm_sq / safe_denominator
        clamped_ratio = jnp.clip(ratio, 1.0, jnp.inf)

        return self.radius * jnp.arccosh(clamped_ratio)

    def metric_tensor(self, point: ManifoldPoint) -> MetricTensor:
        """Hyperbolic metric tensor in Poincaré disk coordinates.

        Args:
            point: Point in Poincaré disk

        Returns:
            Metric tensor matrix (conformal factor * Euclidean metric)
        """
        point = self._validate_point(point)

        # Conformal factor for Poincaré disk
        point_norm_sq = jnp.sum(point * point, axis=-1)
        conformal_factor = 4.0 * (self.radius**2) / ((1.0 - point_norm_sq) ** 2)

        # Metric is conformal factor times Euclidean metric
        dim = point.shape[-1]
        euclidean_metric = jnp.eye(dim)

        # Expand conformal factor to match metric shape
        if point.ndim > 1:
            # Batch case
            batch_shape = point.shape[:-1]
            conformal_factor = conformal_factor.reshape(*batch_shape, 1, 1)
            euclidean_metric = jnp.broadcast_to(
                euclidean_metric, (*batch_shape, dim, dim)
            )

        return conformal_factor * euclidean_metric

    def random_point(self, key: jax.Array, shape: tuple = ()) -> ManifoldPoint:
        """Generate random point uniformly in Poincaré disk.

        Args:
            key: JAX random key
            shape: Shape prefix for batch dimensions

        Returns:
            Random point uniformly distributed in hyperbolic space
        """
        if self._dimension is None:
            raise ValueError("Dimension must be set to generate random points")

        # Generate uniform random point in disk
        # Use rejection sampling or polar coordinates
        key1, key2 = jax.random.split(key)

        # Generate radius and angle
        full_shape = (*shape, self.dimension)

        if self.dimension == 2:
            # 2D case: use polar coordinates
            r = jax.random.uniform(key1, shape) ** 0.5  # sqrt for uniform area
            theta = 2 * jnp.pi * jax.random.uniform(key2, shape)

            x = r * jnp.cos(theta)
            y = r * jnp.sin(theta)
            return jnp.stack([x, y], axis=-1)
        # General case: rejection sampling
        max_attempts = 100

        def attempt_sample(i, state):
            key_i, accepted, samples = state
            key_i, subkey = jax.random.split(key_i)

            candidate = jax.random.uniform(subkey, full_shape, minval=-1.0, maxval=1.0)
            norms = jnp.linalg.norm(candidate, axis=-1)
            valid = norms < 1.0

            new_samples = jnp.where(
                valid[..., None] & ~accepted[..., None], candidate, samples
            )
            new_accepted = accepted | valid

            return key_i, new_accepted, new_samples

        key1, _ = jax.random.split(key1)
        initial_accepted = jnp.zeros(shape)
        initial_samples = jnp.zeros(full_shape)

        _, _, final_samples = jax.lax.fori_loop(
            0, max_attempts, attempt_sample, (key1, initial_accepted, initial_samples)
        )

        return final_samples

    # Geometry Protocol Implementation
    def sample_interior(self, n: int, key: jax.Array) -> ManifoldPoint:
        """Sample n points from the interior of the Poincaré disk."""
        return self.random_point(key, shape=(n,))

    def sample_boundary(self, n: int, key: jax.Array) -> ManifoldPoint:
        """Sample n points from the boundary (unit sphere)."""
        if self._dimension is None:
            raise ValueError("Dimension implies by context or must be set")

        # Sample uniformly on S^{d-1}
        full_shape = (n, self.dimension)
        gaussian = jax.random.normal(key, full_shape)
        norm = jnp.linalg.norm(gaussian, axis=-1, keepdims=True)
        # Project to unit sphere
        return gaussian / norm

    def boundary_sdf(self, points: ManifoldPoint) -> Float[jax.Array, ...]:
        """Compute SDF to the boundary (unit sphere).

        Returns Euclidean signed distance: ||x|| - 1
        Negative inside the disk, positive outside.
        """
        norm = jnp.linalg.norm(points, axis=-1)
        return norm - 1.0

    def christoffel_symbols(self, point: ManifoldPoint) -> ConnectionForm:
        """Christoffel symbols for hyperbolic geometry in Poincaré disk.

        Args:
            point: Point in Poincaré disk

        Returns:
            Christoffel symbols Γ^i_{jk} for hyperbolic metric
        """
        point = self._validate_point(point)
        dim = point.shape[-1]

        # For Poincaré disk model, we can compute Christoffel symbols analytically
        point_norm_sq = jnp.sum(point * point, axis=-1, keepdims=True)
        factor = 2.0 / (1.0 - point_norm_sq)

        # Initialize Christoffel symbols tensor
        if point.ndim > 1:
            batch_shape = point.shape[:-1]
            christoffel = jnp.zeros((*batch_shape, dim, dim, dim))
        else:
            christoffel = jnp.zeros((dim, dim, dim))

        # For Poincaré disk, the non-zero Christoffel symbols are:
        # Γ^i_{jk} = (δ_j^i x_k + δ_k^i x_j - δ_{jk} x^i) / (1 - |x|²)
        # Vectorized computation using einsum and identity matrices

        # Create identity matrices for Kronecker deltas
        identity = jnp.eye(dim)

        if point.ndim > 1:
            # Batch computation
            # δ_j^i x_k term: identity[j,i] * point[...,k]
            term1 = jnp.einsum("ji,bk->bijk", identity, point)
            # δ_k^i x_j term: identity[k,i] * point[...,j]
            term2 = jnp.einsum("ki,bj->bijk", identity, point)
            # δ_{jk} x^i term: identity[j,k] * point[...,i]
            term3 = jnp.einsum("jk,bi->bijk", identity, point)

            # Combine terms with factor
            christoffel = jnp.einsum(
                "b,bijk->bijk", factor[..., 0], term1 + term2 - term3
            )
        else:
            # Single point computation
            # δ_j^i x_k term: identity[j,i] * point[k]
            term1 = jnp.einsum("ji,k->ijk", identity, point)
            # δ_k^i x_j term: identity[k,i] * point[j]
            term2 = jnp.einsum("ki,j->ijk", identity, point)
            # δ_{jk} x^i term: identity[j,k] * point[i]
            term3 = jnp.einsum("jk,i->ijk", identity, point)

            # Combine terms with factor
            christoffel = factor[0] * (term1 + term2 - term3)

        return christoffel

    def parallel_transport(
        self, tangent: TangentVector, path_start: ManifoldPoint, path_end: ManifoldPoint
    ) -> TangentVector:
        """Parallel transport tangent vector along geodesic in hyperbolic space.

        Args:
            tangent: Tangent vector at path_start
            path_start: Starting point on hyperbolic manifold
            path_end: End point on hyperbolic manifold

        Returns:
            Parallel transported tangent vector at path_end
        """
        path_start = self._validate_point(path_start)
        path_end = self._validate_point(path_end)

        # For hyperbolic space in Poincaré disk, parallel transport along
        # geodesics can be computed using gyrovector operations

        # If points are the same, no transport needed
        distance = self.geodesic_distance(path_start, path_end)
        same_point = distance < 1e-8

        # Use JAX-compatible conditional
        return jnp.where(
            same_point,
            tangent,
            self._compute_parallel_transport(tangent, path_start, path_end),
        )

    def _compute_parallel_transport(
        self, tangent: TangentVector, path_start: ManifoldPoint, path_end: ManifoldPoint
    ) -> TangentVector:
        """Compute parallel transport when points are different."""
        # Lambda factors for rescaling
        start_norm_sq = jnp.sum(path_start * path_start, axis=-1, keepdims=True)
        end_norm_sq = jnp.sum(path_end * path_end, axis=-1, keepdims=True)

        lambda_start = 2.0 / (1.0 - start_norm_sq)
        lambda_end = 2.0 / (1.0 - end_norm_sq)

        # Rescale tangent vector by lambda factors (this is the parallel transport)
        # This is an approximation that works well for hyperbolic geometry
        return tangent * (lambda_start / lambda_end)


# JAX pytree registration
def _hyperbolic_manifold_tree_flatten(manifold):
    children = (manifold.curvature,)
    aux_data = (manifold._dimension,)
    return children, aux_data


def _hyperbolic_manifold_tree_unflatten(aux_data, children):
    (curvature,) = children
    (dimension,) = aux_data
    return HyperbolicManifold(curvature=curvature, dimension=dimension)


jax.tree_util.register_pytree_node(
    HyperbolicManifold,
    _hyperbolic_manifold_tree_flatten,
    _hyperbolic_manifold_tree_unflatten,
)
