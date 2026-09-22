"""Incompressible Navier-Stokes in time, on the staggered layout.

    du/dt + div(u u) = -grad(p) + nu * lap(u),    div(u) = 0

Each spatial operator is chosen so that it contributes no energy error of its own:
``divergence`` is minus the adjoint of ``gradient``, so the projection is an exact discrete
Helmholtz decomposition; the convective term is skew-symmetric, so it produces no energy;
and the viscous term is symmetric and negative semi-definite, so it only ever removes some.
What remains is the time integrator, which is explicit Runge-Kutta here. Its energy error
follows the step and not the grid: refining at a fixed Courant number barely moves it, while
holding the step fixed and refining the grid doubles the Courant number and costs about a
factor of fifty a time. And it falls faster than the scheme's own order -- measured, each
halving of the step divides it by 34 to 59 rather than the sixteen a fourth-order method
would give. That follows from the operator being skew: a Runge-Kutta stability function
matches the exponential to its order, so on an imaginary spectrum the amplitude error
appears two orders later than the solution error, and classical RK4's
``|R(iy)| = 1 - y^6/144`` accumulates over a fixed interval as the fifth power of the step.

**The projection is applied at every stage, and that is not an implementation detail.** The
convective term's skew-symmetry is a statement about a discretely divergence-free
transporting field; a stage beginning from an unprojected one is outside the regime the
property covers. Projecting once per step instead costs three orders of convergence, which
the tests measure. Both reference implementations of this scheme project per stage.

The trajectory is a ``lax.scan`` of fixed length rather than an adaptive loop, so the whole
integration differentiates in reverse mode. An adaptive step is a ``lax.while_loop`` with a
data-dependent trip count, which has no reverse-mode rule at all -- the defect that makes
``opifex.physics.solvers.navier_stokes.solve_navier_stokes_2d`` non-differentiable.

References:
    * Sanderse 2013 -- *Energy-conserving Runge-Kutta methods for the incompressible
      Navier-Stokes equations*, J. Comput. Phys. 233, 100.
    * Sanderse, Koren 2012 -- *Accuracy analysis of explicit Runge-Kutta methods applied to
      the incompressible Navier-Stokes equations*, J. Comput. Phys. 231(8), 3041.
    * Harlow, Welch 1965 -- *Numerical calculation of time-dependent viscous incompressible
      flow of fluid with free surface*, Phys. Fluids 8(12), 2182.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.fields.field import Extrapolation
from opifex.fields.staggered import StaggeredGrid
from opifex.fields.staggered_convection import convect
from opifex.fields.staggered_pressure import project


def laplacian(field: StaggeredGrid) -> StaggeredGrid:
    """The viscous operator, component-wise on the faces each component lives on.

    Symmetric and negative semi-definite, so it removes energy and never adds any.

    Args:
        field: Velocity on cell faces.

    Returns:
        The Laplacian of each component, on the same faces.

    Raises:
        ValueError: If the boundary is not periodic. The stencil below wraps, so on a
            walled grid it would diffuse momentum out through one wall and back in
            through the opposite one -- measured at 8 cells, a spike on the first row of
            the tangential component reaches the far wall at 64.0 where it should reach
            zero. A wall needs the velocity on it inside the stencil, which is the same
            boundary question ``convect`` defers.
    """
    if field.extrapolation != Extrapolation.PERIODIC:
        msg = (
            "Staggered diffusion currently carries periodic boundaries only; the stencil "
            "wraps, so on a walled grid it would diffuse momentum through the boundary"
        )
        raise ValueError(msg)

    spacing = field.dx
    components = []
    for component in field.components:
        total = jnp.zeros_like(component)
        for axis in range(field.spatial_dim):
            neighbours = jnp.roll(component, 1, axis=axis) + jnp.roll(component, -1, axis=axis)
            total = total + (neighbours - 2.0 * component) / spacing[axis] ** 2
        components.append(total)
    return StaggeredGrid(tuple(components), field.box, field.extrapolation, field.resolution)


def tendency(velocity: StaggeredGrid, viscosity: float) -> StaggeredGrid:
    """The rate of change of the velocity, before the pressure is applied.

    Args:
        velocity: Velocity on cell faces, divergence free.
        viscosity: Kinematic viscosity.

    Returns:
        ``-div(u u) + nu * lap(u)``, on the same faces.
    """
    carried = convect(velocity, velocity)
    diffused = laplacian(velocity)
    return jax.tree.map(
        lambda advection, diffusion: -advection + viscosity * diffusion, carried, diffused
    )


def _combine(*weighted: tuple[float, StaggeredGrid]) -> StaggeredGrid:
    """A weighted sum of fields sharing a layout."""
    (_, first), *rest = weighted
    total = jax.tree.map(lambda value: weighted[0][0] * value, first)
    for weight, field in rest:
        total = jax.tree.map(lambda running, value, w=weight: running + w * value, total, field)
    return total


def step(velocity: StaggeredGrid, dt: float, viscosity: float) -> StaggeredGrid:
    """Advance one classical fourth-order Runge-Kutta step, projecting at every stage.

    Args:
        velocity: Velocity on cell faces, divergence free.
        dt: Time step.
        viscosity: Kinematic viscosity.

    Returns:
        The velocity one step later, divergence free.
    """
    first = tendency(velocity, viscosity)
    stage = project(_combine((1.0, velocity), (0.5 * dt, first)))[0]

    second = tendency(stage, viscosity)
    stage = project(_combine((1.0, velocity), (0.5 * dt, second)))[0]

    third = tendency(stage, viscosity)
    stage = project(_combine((1.0, velocity), (dt, third)))[0]

    fourth = tendency(stage, viscosity)
    advanced = _combine(
        (1.0, velocity),
        (dt / 6.0, first),
        (dt / 3.0, second),
        (dt / 3.0, third),
        (dt / 6.0, fourth),
    )
    return project(advanced)[0]


def stable_step_count(
    velocity: StaggeredGrid, total_time: float, viscosity: float, safety: float = 0.5
) -> int:
    """The smallest step count whose step is stable for both limits, with margin.

    An explicit scheme is bounded twice, and ``integrate`` guards neither because the step
    count must be static while the viscosity and the velocity need not be. Passing a step
    that violates either limit returns NaN rather than an error, so size the count here
    when the parameters are known on the host.

    The advective limit is the Courant condition. Measured on this scheme at 64 cells: it
    stays finite to a Courant number of 3.79, holds energy to 0.990 at 1.89 and to 0.9996
    at 0.95, so the useful ceiling is near one rather than near the stability edge.

    The diffusive limit is ``dt <= 0.348 * dx^2 / nu``: classical RK4 is stable on the
    negative real axis to 2.785, and the most negative eigenvalue of the MAC Laplacian is
    ``2 * ndim * 2 * nu / dx^2``, which is ``8 nu / dx^2`` in two dimensions. Measured by
    bisection at 64 cells, the limit falls at 0.358 to 0.393 of ``dx^2 / nu``.

    Args:
        velocity: The field to be integrated; its cell size and speed set both limits.
        total_time: The interval to integrate over.
        viscosity: Kinematic viscosity.
        safety: Fraction of the stricter limit to take as the step.

    Returns:
        A step count for ``integrate``, at least one.
    """
    spacing = float(jnp.min(velocity.dx))
    speed = float(max(jnp.max(jnp.abs(component)) for component in velocity.components))
    advective = spacing / speed if speed > 0.0 else jnp.inf
    diffusive = 0.348 * spacing**2 / viscosity if viscosity > 0.0 else jnp.inf
    step = safety * float(min(advective, diffusive))
    return max(1, int(jnp.ceil(total_time / step))) if step > 0.0 else 1


def integrate(
    velocity: StaggeredGrid, total_time: float, num_steps: int, viscosity: float
) -> StaggeredGrid:
    """Advance the velocity over ``total_time`` in ``num_steps`` equal steps.

    The step count is static and the trajectory is a ``lax.scan``, so the whole integration
    traces once and differentiates in reverse mode. The inner step is rematerialised rather
    than taped, which trades one extra forward evaluation per step for a backward pass whose
    memory does not grow with the trajectory.

    Args:
        velocity: Velocity on cell faces, divergence free.
        total_time: The interval to integrate over.
        num_steps: Number of equal steps (static).
        viscosity: Kinematic viscosity.

    Returns:
        The velocity at ``total_time``.
    """
    dt = total_time / num_steps

    @jax.checkpoint
    def advance(state: StaggeredGrid, _: None) -> tuple[StaggeredGrid, None]:
        return step(state, dt, viscosity), None

    final, _ = jax.lax.scan(advance, velocity, None, length=num_steps)
    return final


__all__ = ["integrate", "laplacian", "stable_step_count", "step", "tendency"]
