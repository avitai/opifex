"""Unified Geometry Protocol for Opifex.

This module defines the shared interface for all geometric domains,
unifying Constructive Solid Geometry (CSG) and Manifolds under a single
composable type system for Physics-Informed Neural Operators.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import jax  # noqa: TC002
from jaxtyping import Float  # noqa: TC002


@runtime_checkable
class Geometry(Protocol):
    """Unified Geometry Protocol.

    All geometric domains in Opifex must implement this interface to ensure
    composability with the unified Trainer and PDE solvers.
    """

    @abstractmethod
    def sample_interior(self, n: int, key: jax.Array) -> Float[jax.Array, "n d"]:
        """Sample n points from the interior of the geometry.

        Args:
            n: Number of points to sample
            key: PRNG key

        Returns:
            Array of shape (n, d) containing sampled points
        """
        ...

    @abstractmethod
    def sample_boundary(self, n: int, key: jax.Array) -> Float[jax.Array, "n d"]:
        """Sample n points from the boundary of the geometry.

        Args:
            n: Number of points to sample
            key: PRNG key

        Returns:
            Array of shape (n, d) containing sampled points
        """
        ...

    @abstractmethod
    def boundary_sdf(self, points: Float[jax.Array, "... d"]) -> Float[jax.Array, ...]:
        """Compute Signed Distance Function (SDF) to the boundary.

        Args:
            points: Points to evaluate SDF at (batchable)

        Returns:
            Signed distance values (negative inside, positive outside)
        """
        ...
