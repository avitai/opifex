"""Incompressible projection over the staggered layout.

On a staggered grid ``divergence`` is minus the adjoint of ``gradient``, so the pressure
Poisson operator ``divergence(gradient(.))`` is symmetric by construction, and on a uniform
Cartesian box it is separable: it acts on each axis independently, so a transform that
diagonalises one axis diagonalises the whole operator and the solve is a division rather
than an iteration.

Which transform is not a choice -- it is the boundary. The staggered gradient writes only
onto interior faces, so at a wall the pressure has a vanishing normal derivative, and that
homogeneous-Neumann condition is exactly the boundary condition the DCT-II diagonalises.
A periodic axis is the DFT's. So a walled box is solved directly, exactly and without
iteration, at the same cost as a periodic one:

    periodic axis of n cells   eigenvalues  -4 sin^2(pi k / n)   / h^2
    walled axis of n cells     eigenvalues  -4 sin^2(pi k / 2n)  / h^2

Both leave the constant, and only the constant, in the null space of a closed box, which is
the pressure's gauge freedom and carries no velocity. It is dropped rather than solved for.

Because the solve is a transform, a division and an inverse transform, it is ordinary
straight-line JAX: it traces once, maps under ``vmap`` and differentiates in reverse mode
with no custom rule. That is the reason to prefer it over an iterative solve wherever the
geometry separates, quite apart from the cost.

References:
    * Harlow, Welch 1965 -- *Numerical calculation of time-dependent viscous incompressible
      flow of fluid with free surface*, Phys. Fluids 8(12), 2182.
    * Lynch, Rice, Thomas 1964 -- *Direct solution of partial difference equations by tensor
      product methods*, Numer. Math. 6(1), 185.
    * Chorin 1968 -- *Numerical solution of the Navier-Stokes equations*,
      Math. Comp. 22(104), 745.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.fft import dct, idct

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import divergence, gradient, require_boundary, StaggeredGrid


def _axis_eigenvalues(cells: int, spacing: jax.Array, extrapolation: Extrapolation) -> jax.Array:
    """The eigenvalues the pressure operator has along one axis.

    Args:
        cells: Number of cells along the axis.
        spacing: Cell size along the axis.
        extrapolation: Boundary condition on that axis.

    Returns:
        One eigenvalue per mode along the axis.
    """
    require_boundary(extrapolation, "the pressure projection")
    index = jnp.arange(cells)
    periodic = extrapolation == Extrapolation.PERIODIC
    angle = jnp.pi * index / (cells if periodic else 2 * cells)
    return -4.0 * jnp.sin(angle) ** 2 / spacing**2


def _poisson_eigenvalues(
    resolution: tuple[int, ...], extrapolation: Extrapolation, box: Box
) -> jax.Array:
    """The eigenvalues of ``divergence(gradient(.))``, separable over the axes.

    Args:
        resolution: Number of cells along each axis.
        extrapolation: Boundary condition type.
        box: Physical domain bounds.

    Returns:
        An array shaped like the pressure, holding one eigenvalue per mode.
    """
    spacing = box.size / jnp.asarray(resolution, dtype=jnp.float32)
    per_axis = [
        _axis_eigenvalues(cells, spacing[axis], extrapolation)
        for axis, cells in enumerate(resolution)
    ]
    grids = jnp.meshgrid(*per_axis, indexing="ij")
    return jnp.sum(jnp.stack(grids), axis=0)


def _forward(values: jax.Array, extrapolation: Extrapolation) -> jax.Array:
    """Take the pressure into the basis its boundary diagonalises."""
    if extrapolation == Extrapolation.PERIODIC:
        return jnp.fft.fftn(values)
    transformed = values
    for axis in range(values.ndim):
        transformed = dct(transformed, type=2, norm="ortho", axis=axis)
    return transformed


def _inverse(coefficients: jax.Array, extrapolation: Extrapolation) -> jax.Array:
    """Bring the pressure back from that basis."""
    if extrapolation == Extrapolation.PERIODIC:
        return jnp.real(jnp.fft.ifftn(coefficients))
    transformed = coefficients
    for axis in range(coefficients.ndim):
        transformed = idct(transformed, type=2, norm="ortho", axis=axis)
    return transformed


def project(velocity: StaggeredGrid) -> tuple[StaggeredGrid, jax.Array]:
    """Remove the gradient part of a face velocity.

    Solves ``divergence(gradient(p)) = divergence(v)`` exactly by diagonalising the
    operator, then returns ``v - gradient(p)``. The result is divergence free under the
    same ``divergence`` that measured it, to rounding rather than to a tolerance.

    Args:
        velocity: Velocity on cell faces.

    Returns:
        Tuple of (divergence-free velocity, pressure at cell centres).
    """
    target = divergence(velocity)
    eigenvalues = _poisson_eigenvalues(velocity.resolution, velocity.extrapolation, velocity.box)

    # The constant carries no velocity, so the gauge is fixed by dropping it rather than
    # by solving for it. Substituted inside the division as well as outside, so the branch
    # not taken holds no division by zero for reverse mode to differentiate.
    singular = eigenvalues == 0
    coefficients = _forward(target, velocity.extrapolation)
    pressure = _inverse(
        jnp.where(singular, 0.0, coefficients / jnp.where(singular, 1.0, eigenvalues)),
        velocity.extrapolation,
    )

    correction = gradient(pressure, velocity)
    projected = StaggeredGrid(
        tuple(
            component - update
            for component, update in zip(velocity.components, correction.components, strict=True)
        ),
        velocity.box,
        velocity.extrapolation,
        velocity.resolution,
    )
    return projected, pressure


__all__ = ["project"]
