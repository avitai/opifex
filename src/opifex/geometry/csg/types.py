"""Shared geometric type aliases and shape base types for the CSG package.

Provides:

* Point / Points jaxtyping aliases for 1D / 2D / 3D coordinates.
* :class:`Shape2D` runtime-checkable protocol describing the operations
  every CSG-compatible 2D shape must implement.
* :class:`_EnhancedShapeBase` default-method base used by every concrete
  primitive and CSG operation in this package.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Float

from opifex.geometry.base import Geometry


# Type aliases using proper jaxtyping annotations
Point1D = Float[jax.Array, "1"]  # 1D point
Point2D = Float[jax.Array, "2"]  # 2D point
Point3D = Float[jax.Array, "3"]  # 3D point
Points1D = Float[jax.Array, "n 1"]  # N x 1 array of 1D points
Points2D = Float[jax.Array, "n 2"]  # N x 2 array of 2D points
Points3D = Float[jax.Array, "n 3"]  # N x 3 array of 3D points


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
class _EnhancedShapeBase:  # pyright: ignore[reportUnusedClass]
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
