from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from opifex.geometry.csg import Rectangle
from opifex.geometry.manifolds.spherical import SphericalManifold


# Define the Master Geometry Protocol expected by Phase 1
@runtime_checkable
class Geometry(Protocol):
    """Unified Geometry Protocol for Opifex."""

    def sample_interior(self, n: int, key: jax.Array) -> jax.Array:
        """Sample n points from the interior of the geometry."""
        ...

    def sample_boundary(self, n: int, key: jax.Array) -> jax.Array:
        """Sample n points from the boundary of the geometry."""
        ...

    def boundary_sdf(self, points: jax.Array) -> jax.Array:
        """Compute Signed Distance Function to the boundary."""
        ...


class TestGeometryProtocolCompliance:
    """TDD tests for Geometry Unification."""

    def test_rectangular_csg_compliance(self):
        """Test if Rectangle adheres to the new Geometry protocol."""
        rect = Rectangle(center=jnp.array([0.0, 0.0]), width=1.0, height=1.0)

        # This checks runtime compliance structure
        # Currently expected to FAIL because Rectangle lacks `sample_interior` and `boundary_sdf`
        # (It has `distance` matching boundary_sdf logic, but different name)
        assert isinstance(rect, Geometry), (
            "Rectangle does not satisfy Geometry protocol"
        )

    def test_spherical_manifold_compliance(self):
        """Test if SphericalManifold adheres to the new Geometry protocol."""
        sphere = SphericalManifold(radius=1.0, dimension=2)

        # Currently expected to FAIL because Manifold lacks all 3 strict methods
        assert isinstance(sphere, Geometry), (
            "SphericalManifold does not satisfy Geometry protocol"
        )
