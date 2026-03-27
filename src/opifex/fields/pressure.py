"""Pressure solve for incompressible flow projection.

Implements the Helmholtz-Hodge decomposition:
    v* = v - ∇p  where ∇²p = ∇·v

The pressure Poisson equation is solved using either iterative
(Jacobi) or spectral (FFT) methods.

Reference:
    Chorin (1968) "Numerical solution of the Navier-Stokes equations"
"""

from __future__ import annotations

import jax.numpy as jnp

from opifex.fields.field import CenteredGrid, Extrapolation
from opifex.fields.operations import divergence, gradient


def pressure_solve_spectral(
    velocity: CenteredGrid,
) -> tuple[CenteredGrid, CenteredGrid]:
    """Solve pressure Poisson equation using FFT (periodic boundaries).

    Computes p such that ∇²p = ∇·v, then returns the divergence-free
    velocity v* = v - ∇p and the pressure field p.

    Only works with periodic boundary conditions.

    Args:
        velocity: Vector velocity field, shape (*resolution, ndim).

    Returns:
        Tuple of (divergence_free_velocity, pressure).

    Raises:
        ValueError: If boundary conditions are not periodic.
    """
    if velocity.extrapolation != Extrapolation.PERIODIC:
        raise ValueError("Spectral pressure solve requires periodic boundaries")

    # Compute divergence
    div = divergence(velocity)

    # Solve ∇²p = div via FFT
    div_hat = jnp.fft.fftn(div.values)

    # Build wavenumber grid
    resolution = div.resolution
    dx = div.dx
    freq_arrays = []
    for d, n in enumerate(resolution):
        freq = jnp.fft.fftfreq(n, d=dx[d])
        freq_arrays.append(freq)

    grids = jnp.meshgrid(*freq_arrays, indexing="ij")
    k_squared = sum((2.0 * jnp.pi * k) ** 2 for k in grids)

    # Avoid division by zero at k=0 (mean pressure is arbitrary)
    k_squared = jnp.where(k_squared == 0, 1.0, k_squared)

    # p_hat = -div_hat / k²
    p_hat = -div_hat / k_squared
    p_hat = p_hat.at[tuple(0 for _ in resolution)].set(0.0)  # zero mean

    pressure_values = jnp.real(jnp.fft.ifftn(p_hat))
    pressure = CenteredGrid(pressure_values, velocity.box, velocity.extrapolation)

    # Project: v* = v - ∇p
    grad_p = gradient(pressure)
    projected = velocity - grad_p

    return projected, pressure


def pressure_solve_jacobi(
    velocity: CenteredGrid,
    n_iterations: int = 100,
    omega: float = 1.0,
) -> tuple[CenteredGrid, CenteredGrid]:
    """Solve pressure Poisson equation using Jacobi iteration.

    Works with any boundary condition. Slower than spectral but
    more general.

    Args:
        velocity: Vector velocity field.
        n_iterations: Number of Jacobi iterations.
        omega: Relaxation parameter (1.0 = standard, >1 = SOR).

    Returns:
        Tuple of (divergence_free_velocity, pressure).
    """
    div = divergence(velocity)
    dx = velocity.dx
    ndim = velocity.spatial_dim

    # Initialize pressure to zero
    p = jnp.zeros(div.resolution)

    # Jacobi iteration: p_new = (1/2d) * (Σ neighbors(p) - dx² * div)
    for _ in range(n_iterations):
        p_padded = jnp.pad(
            p,
            [(1, 1)] * ndim,
            mode="constant" if velocity.extrapolation != Extrapolation.PERIODIC else "wrap",
        )

        neighbor_sum = jnp.zeros_like(p)
        for d in range(ndim):
            slc_fwd = tuple(slice(2, None) if i == d else slice(1, -1) for i in range(ndim))
            slc_bwd = tuple(slice(None, -2) if i == d else slice(1, -1) for i in range(ndim))
            neighbor_sum = neighbor_sum + p_padded[slc_fwd] + p_padded[slc_bwd]

        dx_sq = dx[0] ** 2  # assume uniform dx
        p_new = (neighbor_sum - dx_sq * div.values) / (2.0 * ndim)
        p = omega * p_new + (1.0 - omega) * p

    pressure = CenteredGrid(p, velocity.box, velocity.extrapolation)
    grad_p = gradient(pressure)
    projected = velocity - grad_p

    return projected, pressure


__all__ = ["pressure_solve_jacobi", "pressure_solve_spectral"]
