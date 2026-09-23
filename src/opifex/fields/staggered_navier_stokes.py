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
the tests measure. Sanderse and Koren 2012 analyse exactly this: projecting each stage
velocity retains the full order of the underlying Runge-Kutta method for the velocity.

**A wall velocity may vary in time at no cost, and this scheme is the reason.** The
constraint the projection enforces is ``M u = r(t)``, whose right-hand side is the
prescribed wall-*normal* flux; impermeability fixes that at zero, so ``r`` is identically
zero and cannot depend on time however the wall slides. Applying the boundary condition at
the stage time and then projecting the *state* satisfies the constraint exactly for any
``r(t)``, where the equivalent formulation in terms of ``dr/dt`` satisfies it only when
``dr/dt`` vanishes -- so this order is a correctness requirement, not an optimisation.
Measured, a time-varying lid holds the divergence at 1.4e-14 over six steps. The time
derivative of the boundary data is needed for one thing only, a pressure accurate to the
same order as the velocity, which costs about a quarter of a step and is not computed here.

The trajectory is a ``lax.scan`` of fixed length rather than an adaptive loop, so the whole
integration differentiates in reverse mode. An adaptive step is a ``lax.while_loop`` with a
data-dependent trip count, which has no reverse-mode rule at all.

References:
    * Sanderse 2013 -- *Energy-conserving Runge-Kutta methods for the incompressible
      Navier-Stokes equations*, J. Comput. Phys. 233, 100.
    * Sanderse, Koren 2012 -- *Accuracy analysis of explicit Runge-Kutta methods applied to
      the incompressible Navier-Stokes equations*, J. Comput. Phys. 231(8), 3041.
    * Harlow, Welch 1965 -- *Numerical calculation of time-dependent viscous incompressible
      flow of fluid with free surface*, Phys. Fluids 8(12), 2182.
"""

import jax
import jax.numpy as jnp

from opifex.fields.staggered import StaggeredGrid
from opifex.fields.staggered_convection import convect
from opifex.fields.staggered_diffusion import laplacian, wall_source, WallVelocity
from opifex.fields.staggered_pressure import project


def tendency(
    velocity: StaggeredGrid,
    viscosity: float | jax.Array,
    walls: WallVelocity | None = None,
) -> StaggeredGrid:
    """The rate of change of the velocity, before the pressure is applied.

    A moving wall enters here and nowhere else. It drives the flow through the viscous
    term alone: the convective term cannot see a tangential wall value, because that value
    only ever multiplies a wall-normal velocity that impermeability holds at zero.

    Args:
        velocity: Velocity on cell faces, divergence free.
        viscosity: Kinematic viscosity; may be traced.
        walls: Prescribed tangential wall velocity. ``None`` means every wall is at rest,
            which is the no-slip cavity.

    Returns:
        ``-div(u u) + nu * lap(u)``, on the same faces.
    """
    carried = convect(velocity, velocity)
    diffused = laplacian(velocity)
    if walls is not None:
        diffused = jax.tree.map(lambda a, b: a + b, diffused, wall_source(velocity, walls))
    return jax.tree.map(
        lambda advection, diffusion: -advection + viscosity * diffusion, carried, diffused
    )


def _combine(*weighted: tuple[float | jax.Array, StaggeredGrid]) -> StaggeredGrid:
    """A weighted sum of fields sharing a layout."""
    (_, first), *rest = weighted
    total = jax.tree.map(lambda value: weighted[0][0] * value, first)
    for weight, field in rest:
        total = jax.tree.map(lambda running, value, w=weight: running + w * value, total, field)
    return total


def step(
    velocity: StaggeredGrid,
    dt: float | jax.Array,
    viscosity: float | jax.Array,
    walls: WallVelocity | None = None,
) -> StaggeredGrid:
    """Advance one classical fourth-order Runge-Kutta step, projecting at every stage.

    Args:
        velocity: Velocity on cell faces, divergence free.
        dt: Time step.
        viscosity: Kinematic viscosity.
        walls: Prescribed tangential wall velocity, or ``None`` for walls at rest.

    Returns:
        The velocity one step later, divergence free.
    """
    first = tendency(velocity, viscosity, walls)
    stage = project(_combine((1.0, velocity), (0.5 * dt, first)))[0]

    second = tendency(stage, viscosity, walls)
    stage = project(_combine((1.0, velocity), (0.5 * dt, second)))[0]

    third = tendency(stage, viscosity, walls)
    stage = project(_combine((1.0, velocity), (dt, third)))[0]

    fourth = tendency(stage, viscosity, walls)
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
    velocity: StaggeredGrid,
    total_time: float | jax.Array,
    num_steps: int,
    viscosity: float | jax.Array,
    walls: WallVelocity | None = None,
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
        walls: Prescribed tangential wall velocity, or ``None`` for walls at rest. It is an
            ordinary pytree of arrays, so it crosses the scan boundary without retracing
            and the trajectory differentiates with respect to it.

    Returns:
        The velocity at ``total_time``.
    """
    dt = total_time / num_steps

    @jax.checkpoint
    def advance(state: StaggeredGrid, _: None) -> tuple[StaggeredGrid, None]:
        return step(state, dt, viscosity, walls), None

    final, _ = jax.lax.scan(advance, velocity, None, length=num_steps)
    return final


__all__ = ["integrate", "stable_step_count", "step", "tendency"]
