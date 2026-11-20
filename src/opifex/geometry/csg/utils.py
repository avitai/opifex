"""CSG utility helpers: Monte Carlo area estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.geometry.csg.types import Shape2D  # noqa: TC001


def compute_shape_area(shape: Shape2D, *, key: jax.Array, resolution: int = 1000) -> float:
    """Estimate ``shape``'s area by Monte Carlo integration over its bounding box.

    The fraction of ``resolution`` uniform points in ``shape.bbox`` that ``shape.contains``
    accepts, times the box's area (the acceptance estimate of DeepXDE's rejection sampler).
    Its standard error is ``sqrt(A (A_box - A) / resolution)`` for area ``A``.

    Args:
        shape: The shape.
        key: Key the uniform points are drawn from.
        resolution: Number of uniform points.

    Returns:
        The estimated area.
    """
    bbox = shape.bbox
    size = bbox[1] - bbox[0]
    points = jax.random.uniform(key, (resolution, size.shape[0])) * size + bbox[0]
    return float(jnp.mean(shape.contains(points)) * jnp.prod(size))
