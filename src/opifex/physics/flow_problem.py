"""An incompressible-flow problem, as one validated bundle.

The bundle holds the condition no single object can check. A ``StaggeredGrid`` knows its
own resolution and boundary and a ``WallVelocity`` knows its own; nothing verifies the
two agree. A mismatch is not a shape error at construction -- it surfaces later as a
broadcast failure inside a traced step, several frames from the code that caused it.

The split between data and metadata is the whole design. The physics -- viscosity, the
wall profiles, the interval, the initial field -- is pytree data, so it differentiates and
a sweep over it compiles once. The step count is ``pytree_node=False``, because a scan
length cannot be a tracer. That split also fixes what recompiles: a static field is part
of the jit cache key, so every distinct value of it is a separate program, which is why
only fields that set shapes or select a code path are static.

The grid's own box, boundary and resolution are not repeated here. They already ride in
its treedef, and a second copy would be a second thing to keep in step.

**What is deliberately not validated:** whether the pressure system is singular. Under
all-Neumann boundaries the operator has a constant null mode, and the projection fixes
the gauge by dropping it, guarded on both sides of the division so reverse mode never
meets a division by zero. A host-side check re-asking that question would be weaker than
the structure that already answers it.

Validation lives in ``create`` rather than ``__post_init__``: a ``struct.dataclass`` runs
``__post_init__`` on **every unflatten**, so checks placed there would re-run at each
transform boundary, including inside ``jit``.
"""

import jax
import jax.numpy as jnp
from flax import struct

from opifex.fields.staggered import require_boundary, StaggeredGrid
from opifex.fields.staggered_diffusion import WallVelocity


class IncompressibleFlowProblem(struct.PyTreeNode, kw_only=True):
    """A viscous incompressible flow to integrate over an interval.

    Build one through :meth:`create`, which checks the parts agree.

    Attributes:
        initial_velocity: Velocity on cell faces at the start, divergence free.
        viscosity: Kinematic viscosity; traced, so it differentiates.
        total_time: The interval to integrate over; traced.
        walls: Prescribed tangential wall velocity, or ``None`` for walls at rest.
        num_steps: Number of equal steps. Static: it sets a scan length.
    """

    initial_velocity: StaggeredGrid
    viscosity: jax.Array
    total_time: jax.Array
    walls: WallVelocity | None = None
    num_steps: int = struct.field(pytree_node=False, default=1)

    @classmethod
    def create(
        cls,
        *,
        initial_velocity: StaggeredGrid,
        viscosity: jax.Array,
        total_time: jax.Array,
        walls: WallVelocity | None = None,
        num_steps: int = 1,
    ) -> "IncompressibleFlowProblem":
        """Build a problem, refusing parts that do not agree.

        Args:
            initial_velocity: Velocity on cell faces at the start.
            viscosity: Kinematic viscosity.
            total_time: The interval to integrate over.
            walls: Prescribed tangential wall velocity, or ``None``.
            num_steps: Number of equal steps.

        Returns:
            The validated problem.

        Raises:
            ValueError: If the step count is below one, the viscosity is negative, or the
                walls were built for a different resolution or boundary than the velocity.
        """
        require_boundary(initial_velocity.extrapolation, "convection")
        require_boundary(initial_velocity.extrapolation, "diffusion")

        if num_steps < 1:
            msg = f"num_steps must be at least one step, got {num_steps}"
            raise ValueError(msg)

        # Anti-diffusion is unconditionally unstable and the scheme returns NaN rather
        # than raising, so a negative viscosity has to be refused here or not at all.
        if float(jnp.min(jnp.asarray(viscosity))) < 0.0:
            msg = f"viscosity must be non-negative, got {viscosity}"
            raise ValueError(msg)

        if walls is not None:
            if walls.resolution != initial_velocity.resolution:
                msg = (
                    f"the walls were built for resolution {walls.resolution} and the "
                    f"velocity for {initial_velocity.resolution}; a mismatch surfaces as a "
                    "broadcast failure inside a traced step rather than here"
                )
                raise ValueError(msg)
            if walls.extrapolation != initial_velocity.extrapolation:
                msg = (
                    f"the walls were built for a {walls.extrapolation.value} boundary and "
                    f"the velocity for {initial_velocity.extrapolation.value}; the two "
                    "imply different face counts"
                )
                raise ValueError(msg)

        return cls(
            initial_velocity=initial_velocity,
            viscosity=viscosity,
            total_time=total_time,
            walls=walls,
            num_steps=num_steps,
        )


__all__ = ["IncompressibleFlowProblem"]
