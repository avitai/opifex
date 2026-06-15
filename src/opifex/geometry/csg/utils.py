"""CSG utility helpers: area estimation and implementation self-validation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.geometry.csg.operations import difference, intersection, union
from opifex.geometry.csg.primitives import Circle, Rectangle
from opifex.geometry.csg.types import Shape2D  # noqa: TC001


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


# Validation and testing support
def validate_implementation() -> None:
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
