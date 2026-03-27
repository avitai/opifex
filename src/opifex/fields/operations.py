"""Differential operators for grid fields.

Finite-difference implementations of gradient, divergence, curl, and
Laplacian on CenteredGrid fields. Boundary conditions are handled via
padding according to the grid's extrapolation type.

These build on opifex's existing autodiff engine for consistency with
the PINN infrastructure, but operate on discrete grid data rather than
continuous neural network functions.
"""

from __future__ import annotations

import jax.numpy as jnp

from opifex.fields.field import CenteredGrid, Extrapolation


def _pad_field(values: jnp.ndarray, extrapolation: Extrapolation) -> jnp.ndarray:
    """Pad field values with one ghost cell on each side.

    Args:
        values: Field values, shape (*spatial_dims).
        extrapolation: Boundary condition type.

    Returns:
        Padded values with shape (*[dim+2 for dim in spatial_dims]).
    """
    ndim = values.ndim
    if extrapolation == Extrapolation.ZERO:
        return jnp.pad(values, [(1, 1)] * ndim, mode="constant", constant_values=0.0)
    if extrapolation == Extrapolation.PERIODIC:
        return jnp.pad(values, [(1, 1)] * ndim, mode="wrap")
    if extrapolation == Extrapolation.NEUMANN:
        return jnp.pad(values, [(1, 1)] * ndim, mode="edge")
    raise ValueError(f"Unknown extrapolation: {extrapolation}")


def gradient(field: CenteredGrid) -> CenteredGrid:
    """Compute spatial gradient using central finite differences.

    For a scalar field u, returns ∇u as a vector field with an extra
    trailing dimension for the gradient components.

    Args:
        field: Scalar CenteredGrid.

    Returns:
        CenteredGrid with values shape (*resolution, spatial_dim).
    """
    padded = _pad_field(field.values, field.extrapolation)
    dx = field.dx
    ndim = field.spatial_dim

    components = []
    for d in range(ndim):
        # Central difference: (u[i+1] - u[i-1]) / (2*dx)
        slc_fwd = tuple(slice(2, None) if i == d else slice(1, -1) for i in range(ndim))
        slc_bwd = tuple(slice(None, -2) if i == d else slice(1, -1) for i in range(ndim))
        grad_d = (padded[slc_fwd] - padded[slc_bwd]) / (2.0 * dx[d])
        components.append(grad_d)

    grad_values = jnp.stack(components, axis=-1)
    return CenteredGrid(values=grad_values, box=field.box, extrapolation=field.extrapolation)


def divergence(field: CenteredGrid) -> CenteredGrid:
    """Compute divergence of a vector field using central differences.

    For a vector field v with shape (*resolution, spatial_dim), returns
    the scalar divergence ∇·v.

    Args:
        field: Vector CenteredGrid with trailing vector dimension.

    Returns:
        Scalar CenteredGrid with divergence values.
    """
    ndim = field.spatial_dim
    dx = field.dx
    div_values = jnp.zeros(field.resolution)

    for d in range(ndim):
        component = field.values[..., d]
        padded = _pad_field(component, field.extrapolation)

        slc_fwd = tuple(slice(2, None) if i == d else slice(1, -1) for i in range(ndim))
        slc_bwd = tuple(slice(None, -2) if i == d else slice(1, -1) for i in range(ndim))
        div_d = (padded[slc_fwd] - padded[slc_bwd]) / (2.0 * dx[d])
        div_values = div_values + div_d

    return CenteredGrid(values=div_values, box=field.box, extrapolation=field.extrapolation)


def laplacian(field: CenteredGrid) -> CenteredGrid:
    """Compute Laplacian using second-order central differences.

    ∇²u = Σ_d (u[i+1] - 2u[i] + u[i-1]) / dx_d²

    Args:
        field: Scalar CenteredGrid.

    Returns:
        Scalar CenteredGrid with Laplacian values.
    """
    padded = _pad_field(field.values, field.extrapolation)
    dx = field.dx
    ndim = field.spatial_dim
    lap_values = jnp.zeros(field.resolution)

    interior = tuple(slice(1, -1) for _ in range(ndim))
    for d in range(ndim):
        slc_fwd = tuple(slice(2, None) if i == d else slice(1, -1) for i in range(ndim))
        slc_bwd = tuple(slice(None, -2) if i == d else slice(1, -1) for i in range(ndim))

        lap_d = (padded[slc_fwd] - 2.0 * padded[interior] + padded[slc_bwd]) / (dx[d] ** 2)
        lap_values = lap_values + lap_d

    return CenteredGrid(values=lap_values, box=field.box, extrapolation=field.extrapolation)


def curl_2d(field: CenteredGrid) -> CenteredGrid:
    """Compute 2D curl (vorticity) of a vector field.

    For v = (vx, vy), curl = ∂vy/∂x - ∂vx/∂y.

    Args:
        field: 2D vector CenteredGrid with shape (Nx, Ny, 2).

    Returns:
        Scalar CenteredGrid with vorticity values.
    """
    if field.spatial_dim != 2:
        raise ValueError(f"2D curl requires 2D field, got {field.spatial_dim}D")

    dx = field.dx
    vx = field.values[..., 0]
    vy = field.values[..., 1]

    vx_padded = _pad_field(vx, field.extrapolation)
    vy_padded = _pad_field(vy, field.extrapolation)

    # ∂vy/∂x
    dvy_dx = (vy_padded[2:, 1:-1] - vy_padded[:-2, 1:-1]) / (2.0 * float(dx[0]))
    # ∂vx/∂y
    dvx_dy = (vx_padded[1:-1, 2:] - vx_padded[1:-1, :-2]) / (2.0 * float(dx[1]))

    curl_values = dvy_dx - dvx_dy
    return CenteredGrid(values=curl_values, box=field.box, extrapolation=field.extrapolation)


__all__ = ["curl_2d", "divergence", "gradient", "laplacian"]
