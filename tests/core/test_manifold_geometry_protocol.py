"""Tests for Manifold compliance with Geometry Protocol.

Ensures that Manifold implementations (Spherical, Hyperbolic) correctly implement
the Geometry protocol methods (sample_interior, sample_boundary, boundary_sdf)
required by the Unified Solver.
"""

import jax
import jax.numpy as jnp

from opifex.geometry.base import Geometry
from opifex.geometry.manifolds.hyperbolic import HyperbolicManifold
from opifex.geometry.manifolds.spherical import SphericalManifold


def test_spherical_manifold_is_geometry():
    """Test SphericalManifold implements Geometry protocol."""
    manifold = SphericalManifold(radius=1.0, dimension=2)
    assert isinstance(manifold, Geometry)


def test_spherical_manifold_geometry_methods():
    """Test specific Geometry methods on Sphere."""
    manifold = SphericalManifold(radius=1.0, dimension=2)
    key = jax.random.PRNGKey(0)

    # 1. sample_interior (should correspond to random points on sphere surface)
    points = manifold.sample_interior(10, key)
    assert points.shape == (10, 3)  # Embedding dim is 2+1=3
    norms = jnp.linalg.norm(points, axis=1)
    assert jnp.allclose(norms, 1.0)

    # 2. sample_boundary (Sphere has no boundary)
    b_points = manifold.sample_boundary(10, key)
    assert b_points.shape[0] == 0  # Should return empty

    # 3. boundary_sdf (Sphere has no boundary -> infinite distance)
    # The sphere surface itself is the domain. Boundry of the manifold is empty.
    sdf = manifold.boundary_sdf(points)
    assert jnp.all(jnp.isinf(sdf))


def test_hyperbolic_manifold_is_geometry():
    """Test HyperbolicManifold implements Geometry protocol."""
    manifold = HyperbolicManifold(curvature=-1.0, dimension=2)
    assert isinstance(manifold, Geometry)


def test_hyperbolic_manifold_geometry_methods():
    """Test specific Geometry methods on Hyperbolic space (Poincare Disk)."""
    manifold = HyperbolicManifold(curvature=-1.0, dimension=2)
    key = jax.random.PRNGKey(0)

    # 1. sample_interior (Points inside disk)
    points = manifold.sample_interior(100, key)
    assert points.shape == (100, 2)
    norms = jnp.linalg.norm(points, axis=1)
    assert jnp.all(norms < 1.0)

    # 2. sample_boundary (Unit circle)
    b_points = manifold.sample_boundary(100, key)
    assert b_points.shape == (100, 2)
    b_norms = jnp.linalg.norm(b_points, axis=1)
    assert jnp.allclose(b_norms, 1.0, atol=1e-5)

    # 3. boundary_sdf (Distance to unit circle: norm - 1)
    # Inside points should have negative SDF
    sdf = manifold.boundary_sdf(points)
    assert jnp.all(sdf < 0)
    # Boundary points should have ~0 SDF
    b_sdf = manifold.boundary_sdf(b_points)
    assert jnp.allclose(b_sdf, 0.0, atol=1e-5)


def test_runtime_checkable_compliance():
    """Verify runtime checkability."""
    sphere = SphericalManifold(radius=1.0, dimension=2)
    hyperbolic = HyperbolicManifold(curvature=-1.0, dimension=2)

    assert isinstance(sphere, Geometry)
    assert isinstance(hyperbolic, Geometry)
