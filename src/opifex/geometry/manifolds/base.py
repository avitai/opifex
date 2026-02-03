"""Base protocols and interfaces for manifold geometries.

This module defines the fundamental abstractions for working with Riemannian
manifolds in scientific machine learning applications.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable, TYPE_CHECKING, TypeAlias

import jax


if TYPE_CHECKING:
    from jaxtyping import Float


# Type aliases for geometric objects
ManifoldPoint: TypeAlias = jax.Array  # Points on manifold
TangentVector: TypeAlias = jax.Array  # Tangent vectors at points
MetricTensor: TypeAlias = jax.Array  # Riemannian metric tensor
ConnectionForm: TypeAlias = jax.Array  # Christoffel symbols


@runtime_checkable
class Manifold(Protocol):
    """Protocol for Riemannian manifolds with JAX compatibility.

    All operations must be JAX-transformable (jit, grad, vmap) and preserve
    the geometric structure of the manifold.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Intrinsic dimension of the manifold."""
        ...

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Dimension of ambient embedding space."""
        ...

    @abstractmethod
    def exp_map(self, base: ManifoldPoint, tangent: TangentVector) -> ManifoldPoint:
        """Exponential map: tangent space → manifold.

        Maps tangent vectors at base point to points on manifold via geodesics.
        Must be JAX-compatible and preserve geometric constraints.

        Args:
            base: Base point on manifold
            tangent: Tangent vector at base point

        Returns:
            Point on manifold reached by geodesic from base in direction tangent
        """
        ...

    @abstractmethod
    def log_map(self, base: ManifoldPoint, point: ManifoldPoint) -> TangentVector:
        """Logarithmic map: manifold → tangent space.

        Inverse of exponential map. Maps points to tangent vectors.

        Args:
            base: Base point for tangent space
            point: Target point on manifold

        Returns:
            Tangent vector pointing from base toward point
        """
        ...

    @abstractmethod
    def metric_tensor(self, point: ManifoldPoint) -> MetricTensor:
        """Riemannian metric tensor at given point.

        Defines inner product structure on tangent spaces.
        Must be positive definite and smooth.

        Args:
            point: Point on manifold

        Returns:
            Metric tensor matrix at point
        """
        ...

    @abstractmethod
    def geodesic_distance(
        self, point1: ManifoldPoint, point2: ManifoldPoint
    ) -> Float[jax.Array, ...]:
        """Compute geodesic distance between points.

        Args:
            point1: First point on manifold
            point2: Second point on manifold

        Returns:
            Geodesic distance (shortest path along manifold)
        """
        ...

    def christoffel_symbols(self, point: ManifoldPoint) -> ConnectionForm:
        """Christoffel symbols of Levi-Civita connection.

        Default implementation uses metric tensor. Can be overridden
        for manifolds with known analytical expressions.

        Args:
            point: Point on manifold

        Returns:
            Christoffel symbols Γ^i_{jk} at point
        """
        # Default implementation via finite differences of metric
        # Numerical computation of Christoffel symbols
        # Implementation details depend on specific manifold
        raise NotImplementedError("Implement for specific manifold")

    def parallel_transport(
        self, tangent: TangentVector, path_start: ManifoldPoint, path_end: ManifoldPoint
    ) -> TangentVector:
        """Parallel transport tangent vector along geodesic.

        Transports tangent vector from path_start to path_end while
        preserving parallelism with respect to the connection.

        Args:
            tangent: Tangent vector at path_start
            path_start: Starting point of transport
            path_end: End point of transport

        Returns:
            Parallel transported tangent vector at path_end
        """
        # Default implementation via integration of connection
        # Can be overridden for manifolds with analytical solutions
        raise NotImplementedError("Implement for specific manifold")


@runtime_checkable
class TangentSpace(Protocol):
    """Protocol for tangent spaces to manifolds.

    Tangent spaces are vector spaces attached to each point of a manifold.
    They provide the setting for differential calculus on manifolds.
    """

    @property
    @abstractmethod
    def base_point(self) -> ManifoldPoint:
        """Base point on manifold where tangent space is attached."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of tangent space (equals manifold dimension)."""
        ...

    @abstractmethod
    def inner_product(
        self, u: TangentVector, v: TangentVector
    ) -> Float[jax.Array, ...]:
        """Inner product of tangent vectors using Riemannian metric.

        Args:
            u: First tangent vector at base_point
            v: Second tangent vector at base_point

        Returns:
            Inner product ⟨u,v⟩ at base_point
        """
        ...

    @abstractmethod
    def norm(self, tangent: TangentVector) -> Float[jax.Array, ...]:
        """Norm of tangent vector.

        Args:
            tangent: Tangent vector at base_point

        Returns:
            Norm ||tangent|| = √⟨tangent,tangent⟩
        """
        ...


# JAX pytree registration for geometric types
# Note: Cannot register type aliases directly, only concrete classes
# Individual manifold implementations should register their own pytrees
# when they define concrete manifold classes that need pytree support
