"""Convection on the staggered layout, in the form that produces no energy.

Written in divergence form with arithmetic-mean interpolation, the convective operator on
a staggered grid is **skew-symmetric about a discretely divergence-free transporting
velocity** (Harlow, Welch 1965; Verstappen, Veldman 2003). A skew operator satisfies
``<w, C w> = 0``, so it moves energy between modes without creating or destroying any: an
inviscid scheme neither damps nor amplifies, and whatever energy a simulation loses is the
time integrator's error, vanishing as the step shrinks, rather than a spatial error that
refinement cannot remove. First-order upwind is the opposite -- almost pure dissipation,
with a numerical viscosity of order ``|u| dx / 2`` that swamps a small physical one.

The qualifier carries the whole property. The proof holds about a divergence-free field
**and only there**; fed a transporting velocity that is not discretely solenoidal, the same
operator produces energy at order one. That is why a projection method must project at
every Runge-Kutta stage rather than once per step: a stage starting from an unprojected
field is outside the regime the skew-symmetry covers.

Each flux is formed where the two factors meet. For the flux of component ``d`` in its own
direction that is the cell centre, where both the transported and the transporting
component average from the faces either side; for the flux in any other direction ``e`` it
is the ``(d, e)`` corner, where the transported component averages along ``e`` and the
transporting one along ``d``. Differencing each flux back onto the ``d`` faces is what
makes the result telescope, and is where the discrete conservation comes from.

At a wall the same two fluxes are formed from the prescribed boundary velocity rather
than from a periodic neighbour, which is the treatment Sanderse, Verstappen and Koren
2014 sets out: the normal component is not a degree of freedom there, so the flux sits on
the boundary and takes its value directly. The condition the skew-symmetry needs at a
wall is **impermeability**, ``u.n = 0``, not no-slip -- the tangential wall value enters
only through a corner flux whose other factor is the wall-normal velocity, which is
identically zero. Measured, tangential wall values of 0, 7.3 and -100 give bit-identical
output while the same probe resolves a normal-wall leak of 0.01, so a moving wall would
need nothing here. Measured as a matrix, the wall operator is skew to rounding, and it
converges to the analytic ``div(u (x) u)`` at second order.

References:
    * Harlow, Welch 1965 -- *Numerical calculation of time-dependent viscous incompressible
      flow of fluid with free surface*, Phys. Fluids 8(12), 2182.
    * Verstappen, Veldman 2003 -- *Symmetry-preserving discretization of turbulent flow*,
      J. Comput. Phys. 187(1), 343.
    * Sanderse, Verstappen, Koren 2014 -- *Boundary treatment for fourth-order staggered
      mesh discretizations of the incompressible Navier-Stokes equations*,
      J. Comput. Phys. 257, 1472.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.fields.field import Extrapolation
from opifex.fields.staggered import require_boundary, StaggeredGrid


def _average_forward(values: jax.Array, axis: int) -> jax.Array:
    """The mean of each entry and its successor along ``axis``, wrapping."""
    return 0.5 * (values + jnp.roll(values, -1, axis=axis))


def _average_backward(values: jax.Array, axis: int) -> jax.Array:
    """The mean of each entry and its predecessor along ``axis``, wrapping."""
    return 0.5 * (values + jnp.roll(values, 1, axis=axis))


def _slice(values: jax.Array, axis: int, start: int | None, stop: int | None) -> jax.Array:
    """``values[..., start:stop, ...]`` along ``axis``."""
    taken = tuple(slice(start, stop) if i == axis else slice(None) for i in range(values.ndim))
    return values[taken]


def _pair_mean(values: jax.Array, axis: int) -> jax.Array:
    """The mean of neighbouring entries along ``axis``, without wrapping."""
    return 0.5 * (_slice(values, axis, None, -1) + _slice(values, axis, 1, None))


def _extend(values: jax.Array, axis: int, boundary: jax.Array) -> jax.Array:
    """``values`` with ``boundary`` placed at each end of ``axis``."""
    return jnp.concatenate([boundary, values, boundary], axis=axis)


def _wall_value(values: jax.Array, axis: int) -> jax.Array:
    """A slab of the prescribed wall velocity, which is zero for a no-slip wall."""
    return jnp.zeros_like(jnp.take(values, jnp.array([0]), axis=axis))


def _difference(flux: jax.Array, axis: int, spacing: jax.Array) -> jax.Array:
    """The difference of neighbouring fluxes along ``axis``, one shorter than the flux."""
    return (_slice(flux, axis, 1, None) - _slice(flux, axis, None, -1)) / spacing


def convect(transported: StaggeredGrid, velocity: StaggeredGrid) -> StaggeredGrid:
    """Transport ``transported`` by ``velocity``, in divergence form.

    Skew-symmetric in ``transported`` whenever ``velocity`` is discretely divergence free
    and every boundary is impermeable, which is the property the scheme is chosen for; see
    the module docstring for what each condition buys.

    Args:
        transported: The field being carried, on cell faces.
        velocity: The field carrying it, on the same faces.

    A boundary this does not carry is refused by ``require_boundary``, which names what
    every part of the layer carries.

    Returns:
        The convective term, on the same faces as ``transported``.
    """
    require_boundary(transported.extrapolation, "convection")
    require_boundary(velocity.extrapolation, "convection")
    wraps = transported.extrapolation == Extrapolation.PERIODIC

    spacing = transported.dx
    components = []
    for normal, carried in enumerate(transported.components):
        total = jnp.zeros_like(carried)
        for direction, carrier in enumerate(velocity.components):
            if direction == normal:
                # Both factors meet at the cell centre. At a wall the face carries the
                # prescribed velocity, which enters both factors alike: writing it into
                # the carrier is what makes the diagonal telescope into the divergence of
                # the two pressure cells either side, and so vanish.
                if wraps:
                    flux = _average_backward(carried, normal) * _average_backward(carrier, normal)
                else:
                    wall = _wall_value(carried, normal)
                    flux = _pair_mean(_extend(carried, normal, wall), normal) * _pair_mean(
                        _extend(carrier, normal, wall), normal
                    )
                total = total + (
                    (jnp.roll(flux, -1, axis=normal) - flux) / spacing[normal]
                    if wraps
                    else _difference(flux, normal, spacing[normal])
                )
            elif wraps:
                # They meet at the corner: the carried component averages along the flux
                # direction, the carrier across it.
                flux = _average_forward(carried, direction) * _average_forward(carrier, normal)
                total = total + (flux - jnp.roll(flux, 1, axis=direction)) / spacing[direction]
            else:
                # The corner sits on the wall at either end of the flux direction, where
                # the carried component takes the prescribed tangential value. That value
                # is unobservable -- it multiplies a wall-normal carrier that is
                # identically zero -- but the corner must exist for the shapes to meet.
                wall = _wall_value(carried, direction)
                carried_corner = _extend(_pair_mean(carried, direction), direction, wall)
                carrier_corner = _pair_mean(
                    _extend(carrier, direction, _wall_value(carrier, direction)), normal
                )
                flux = carried_corner * carrier_corner
                total = total + _difference(flux, direction, spacing[direction])
        components.append(total)

    return StaggeredGrid(
        tuple(components), transported.box, transported.extrapolation, transported.resolution
    )


__all__ = ["convect"]
