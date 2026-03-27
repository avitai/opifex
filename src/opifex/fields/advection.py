"""Advection schemes for grid fields.

Semi-Lagrangian and MacCormack advection for transporting fields
through velocity fields on structured grids.

Reference:
    Stam (1999) "Stable Fluids"
    MacCormack (1969) "The Effect of Viscosity in Hypervelocity
    Impact Cratering"
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.fields.field import CenteredGrid, Extrapolation


def _sample_at(
    field: CenteredGrid,
    coords: jnp.ndarray,
) -> jnp.ndarray:
    """Sample field at arbitrary physical coordinates via bilinear interpolation.

    Args:
        field: Source field to sample.
        coords: Physical coordinates, shape (*spatial, ndim).

    Returns:
        Interpolated values at the given coordinates.
    """
    # Convert physical coords to grid-index coords
    lower = field.box.lower
    dx = field.dx
    idx_coords = (coords - lower) / dx - 0.5  # cell-center offset

    # Use map_coordinates for interpolation
    # Need to reshape for map_coordinates API: list of coordinate arrays per dim
    ndim = field.spatial_dim
    coord_arrays = [idx_coords[..., d] for d in range(ndim)]

    if field.extrapolation == Extrapolation.PERIODIC:
        # Wrap coordinates for periodic fields
        res = jnp.array(field.resolution, dtype=jnp.float32)
        coord_arrays = [c % res[d] for d, c in enumerate(coord_arrays)]

    return jax.scipy.ndimage.map_coordinates(field.values, coord_arrays, order=1, mode="nearest")


def semi_lagrangian(
    field: CenteredGrid,
    velocity: CenteredGrid,
    dt: float,
) -> CenteredGrid:
    """Advect a field using semi-Lagrangian method.

    Traces particles backward in time by -dt and samples the field
    at the departure points. Unconditionally stable for any dt.

    Args:
        field: Scalar field to advect, shape (*resolution).
        velocity: Velocity field, shape (*resolution, ndim).
        dt: Time step.

    Returns:
        Advected field at time t + dt.
    """
    # Cell center coordinates
    centers = field.cell_centers()  # (*resolution, ndim)

    # Backward trace: departure points = centers - velocity * dt
    vel_at_centers = velocity.values
    departure = centers - vel_at_centers * dt

    # Sample field at departure points
    advected = _sample_at(field, departure)

    return CenteredGrid(values=advected, box=field.box, extrapolation=field.extrapolation)


def maccormack(
    field: CenteredGrid,
    velocity: CenteredGrid,
    dt: float,
    correction_strength: float = 1.0,
) -> CenteredGrid:
    """Advect using MacCormack scheme (higher-order correction).

    Performs a semi-Lagrangian step, then traces forward to estimate
    the error, and applies a correction. Clamped to prevent overshoots.

    Args:
        field: Scalar field to advect.
        velocity: Velocity field.
        dt: Time step.
        correction_strength: Blending factor for error correction (0-1).

    Returns:
        Advected field with reduced numerical diffusion.
    """
    centers = field.cell_centers()
    vel = velocity.values

    # Step 1: Semi-Lagrangian (backward trace)
    departure = centers - vel * dt
    forward_advected = _sample_at(field, departure)
    fwd_grid = CenteredGrid(forward_advected, field.box, field.extrapolation)

    # Step 2: Trace forward to estimate error
    arrival = centers + vel * dt
    backward_advected = _sample_at(fwd_grid, arrival)

    # Step 3: Error correction
    error = field.values - backward_advected
    corrected = forward_advected + correction_strength * 0.5 * error

    # Step 4: Clamp to prevent overshoots (local min/max)
    # Use the departure-point neighborhood for bounds
    lo = forward_advected - jnp.abs(error)
    hi = forward_advected + jnp.abs(error)
    clamped = jnp.clip(corrected, lo, hi)

    return CenteredGrid(values=clamped, box=field.box, extrapolation=field.extrapolation)


__all__ = ["maccormack", "semi_lagrangian"]
