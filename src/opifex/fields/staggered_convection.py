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

from opifex.fields.staggered import require_boundary, StaggeredGrid


def _average_forward(values: jax.Array, axis: int) -> jax.Array:
    """The mean of each entry and its successor along ``axis``."""
    return 0.5 * (values + jnp.roll(values, -1, axis=axis))


def _average_backward(values: jax.Array, axis: int) -> jax.Array:
    """The mean of each entry and its predecessor along ``axis``."""
    return 0.5 * (values + jnp.roll(values, 1, axis=axis))


def convect(transported: StaggeredGrid, velocity: StaggeredGrid) -> StaggeredGrid:
    """Transport ``transported`` by ``velocity``, in divergence form.

    Skew-symmetric in ``transported`` whenever ``velocity`` is discretely divergence free,
    which is the property the scheme is chosen for; see the module docstring for what
    happens when it is not.

    Args:
        transported: The field being carried, on cell faces.
        velocity: The field carrying it, on the same faces.

    A boundary this does not carry is refused by ``require_boundary``, which names what
    every part of the layer carries: a wall needs a one-sided interpolation whose form
    decides whether energy conservation survives it, a separate question from the
    interior scheme.

    Returns:
        The convective term, on the same faces as ``transported``.
    """
    require_boundary(transported.extrapolation, "convection")
    require_boundary(velocity.extrapolation, "convection")

    spacing = transported.dx
    components = []
    for normal, carried in enumerate(transported.components):
        total = jnp.zeros_like(carried)
        for direction, carrier in enumerate(velocity.components):
            if direction == normal:
                # Both factors meet at the cell centre, and the flux differences forward
                # back onto the faces.
                flux = _average_backward(carried, normal) * _average_backward(carrier, normal)
                total = total + (jnp.roll(flux, -1, axis=normal) - flux) / spacing[normal]
            else:
                # They meet at the corner: the carried component averages along the flux
                # direction, the carrier across it.
                flux = _average_forward(carried, direction) * _average_forward(carrier, normal)
                total = total + (flux - jnp.roll(flux, 1, axis=direction)) / spacing[direction]
        components.append(total)

    return StaggeredGrid(
        tuple(components), transported.box, transported.extrapolation, transported.resolution
    )


__all__ = ["convect"]
