"""The viscous operator on the staggered layout, and the wall velocity that drives it.

The operator is built so that symmetry is structural rather than arranged: it is a
divergence of a gradient, ``D = -G^T Lambda G`` with ``Lambda`` positive, so any boundary
closure that merely drops a known column keeps it symmetric negative-definite
(Verstappen, Veldman 2003, Eqs. 25 and 37). That matters more than it sounds, because
symmetry is the *only* property that separates this stencil from a plausible wrong one:
the quadratic one-sided extrapolation has a smaller truncation error at the wall and a
smaller global error, and is rejected by nothing else.

The two components meet the wall differently, and the asymmetry is real rather than an
oversight:

* the **normal** component's own-axis faces lie *on* the wall, so the boundary value is
  not an unknown and no ghost is needed. Its wall row is the ordinary three-point
  difference with that value substituted -- diagonal ``-2/h^2``, second-order accurate;
* the **tangential** component sits half a cell inside, so the wall value is reached by
  the mirror ghost ``u_ghost = 2 u_wall - u_1`` (Sanderse, Verstappen, Koren 2014, Eq. 92).
  Its wall row is ``(-3 u_1 + u_2 + 2 u_wall)/h^2`` -- diagonal ``-3/h^2``.

The wall row's truncation error is ``-u''(wall)/4`` and does **not** shrink with the cell
size. That is admissible rather than a defect: for an operator whose principal part is of
order ``m`` the boundary closure may be ``m`` orders lower without reducing the global
order, and diffusion has ``m = 2`` (Svard, Nordstrom 2006). Gustafsson 1975 is the wrong
authority here -- it assumes the boundary is at most *one* order down.

Walls do not tighten the explicit diffusive step. The wall spectrum approaches the
periodic ``4 ndim / h^2`` from below: the tangential direction contributes exactly
``-4/h^2`` and the normal direction ``-4 cos^2(pi/2n)/h^2``, so ``stable_step_count``'s
existing bound holds unchanged.

**The prescribed wall velocity is an array in the residual path, never static metadata.**
It enters as an additive term and the operator itself stays homogeneous, which is what
keeps it differentiable and cheap: carried as a static attribute it forces one retrace
per distinct wall value and reverse mode cannot reach it at all. It is a field rather
than a number per wall because the two cases that matter both vary along the wall -- the
energy test of Sanderse et al. (Eq. 166) and the regularised lid of Shen 1991, which is
the only form of the driven cavity with a well-defined order of accuracy, the classic
one being singular at the corners.

Only the *tangential* wall velocity is prescribed. A non-zero *normal* wall velocity
would be injection through the boundary, which changes the pressure problem the
projection solves; no-penetration is what keeps its transform exact.

References:
    * Sanderse, Verstappen, Koren 2014 -- *Boundary treatment for fourth-order staggered
      mesh discretizations of the incompressible Navier-Stokes equations*,
      J. Comput. Phys. 257, 1472.
    * Verstappen, Veldman 2003 -- *Symmetry-preserving discretization of turbulent flow*,
      J. Comput. Phys. 187(1), 343.
    * Svard, Nordstrom 2006 -- *On the order of accuracy for difference approximations of
      initial-boundary value problems*, J. Comput. Phys. 218(1), 333.
    * Shen 1991 -- *Hopf bifurcation of the unsteady regularized driven cavity flow*,
      J. Comput. Phys. 95(1), 228.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.fields.field import Extrapolation
from opifex.fields.staggered import require_boundary, StaggeredGrid


def _slab(values: jax.Array, axis: int, index: int) -> jax.Array:
    """One index along ``axis``, keeping the rank."""
    return jnp.take(values, jnp.array([index]), axis=axis)


def _shift(values: jax.Array, axis: int, offset: int, ghost: jax.Array) -> jax.Array:
    """``values`` shifted by ``offset`` along ``axis``, with ``ghost`` entering the gap."""
    forward = offset < 0
    window = slice(1, None) if forward else slice(None, -1)
    taken = tuple(window if i == axis else slice(None) for i in range(values.ndim))
    pieces = [values[taken], ghost] if forward else [ghost, values[taken]]
    return jnp.concatenate(pieces, axis=axis)


@jax.tree_util.register_pytree_node_class
class WallVelocity:
    """The prescribed tangential velocity on every solid wall.

    ``values[normal][axis]`` holds the velocity of component ``normal`` on the two walls
    perpendicular to ``axis``, stacked as ``(lower, upper)`` along a leading axis of two.
    It is ``None`` where ``axis == normal``, because that wall carries the component's own
    normal velocity, which no-penetration fixes at zero.

    The arrays are pytree leaves, so a wall profile traces, batches and differentiates
    like any other array.

    Attributes:
        values: One array per (component, perpendicular axis), or ``None``.
        resolution: Number of cells along each axis.
        extrapolation: The boundary this was built for.
    """

    __slots__ = ("extrapolation", "resolution", "values")

    def __init__(
        self,
        values: tuple[tuple[jax.Array | None, ...], ...],
        resolution: tuple[int, ...],
        extrapolation: Extrapolation,
    ) -> None:
        """Store the wall arrays and the layout they belong to.

        Args:
            values: One array per (component, perpendicular axis), or ``None``.
            resolution: Number of cells along each axis.
            extrapolation: The boundary this was built for.
        """
        self.values = values
        self.resolution = resolution
        self.extrapolation = extrapolation

    @classmethod
    def _build(
        cls,
        resolution: tuple[int, ...],
        extrapolation: Extrapolation,
        fill: float,
    ) -> WallVelocity:
        """Every tangential wall set to one constant."""
        shapes = StaggeredGrid.component_shapes(resolution, extrapolation)
        values = tuple(
            tuple(
                None
                if axis == normal
                else jnp.full(
                    (2, *(length for other, length in enumerate(shape) if other != axis)), fill
                )
                for axis in range(len(resolution))
            )
            for normal, shape in enumerate(shapes)
        )
        return cls(values, resolution, extrapolation)

    @classmethod
    def zeros(cls, resolution: tuple[int, ...], extrapolation: Extrapolation) -> WallVelocity:
        """Walls at rest, which is the no-slip cavity.

        Args:
            resolution: Number of cells along each axis.
            extrapolation: The boundary to build for.

        Returns:
            A wall velocity that contributes nothing.
        """
        return cls._build(resolution, extrapolation, 0.0)

    @classmethod
    def uniform(
        cls, resolution: tuple[int, ...], extrapolation: Extrapolation, value: float
    ) -> WallVelocity:
        """Every tangential wall sliding at one speed.

        Args:
            resolution: Number of cells along each axis.
            extrapolation: The boundary to build for.
            value: The speed every wall takes.

        Returns:
            A wall velocity constant over every wall.
        """
        return cls._build(resolution, extrapolation, value)

    def set(self, *, normal: int, axis: int, upper: bool, value: jax.Array) -> WallVelocity:
        """A copy with one wall replaced.

        Args:
            normal: Which velocity component the wall carries.
            axis: The axis the wall is perpendicular to.
            upper: The far wall rather than the near one.
            value: The profile along that wall.

        Returns:
            A new wall velocity; this one is unchanged.

        Raises:
            ValueError: If ``axis`` is the component's own axis, whose wall velocity is
                fixed at zero by no-penetration.
        """
        if axis == normal:
            msg = (
                f"component {normal} has no prescribed velocity on the wall perpendicular to "
                f"axis {axis}: that is its own normal component, which no-penetration fixes at "
                "zero. Prescribing it would be injection through the boundary and would change "
                "the pressure problem the projection solves."
            )
            raise ValueError(msg)
        current = self.values[normal][axis]
        if current is None:  # pragma: no cover - the guard above already returned
            msg = f"no wall array for component {normal} on axis {axis}"
            raise ValueError(msg)
        replaced = current.at[1 if upper else 0].set(value)
        values = tuple(
            tuple(replaced if (n, a) == (normal, axis) else entry for a, entry in enumerate(row))
            for n, row in enumerate(self.values)
        )
        return WallVelocity(values, self.resolution, self.extrapolation)

    def tree_flatten(
        self,
    ) -> tuple[tuple[jax.Array | None, ...], tuple[tuple[int, ...], Extrapolation]]:
        """Split into the wall arrays and the static layout."""
        flat = tuple(entry for row in self.values for entry in row)
        return flat, (self.resolution, self.extrapolation)

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[tuple[int, ...], Extrapolation],
        flat: tuple[jax.Array | None, ...],
    ) -> WallVelocity:
        """Rebuild from the wall arrays and the static layout.

        Args:
            aux: The dimension, resolution and boundary.
            flat: The wall arrays in row-major order.

        Returns:
            The reconstructed wall velocity.
        """
        resolution, extrapolation = aux
        ndim = len(resolution)
        values = tuple(
            tuple(flat[normal * ndim + axis] for axis in range(ndim)) for normal in range(ndim)
        )
        return cls(values, resolution, extrapolation)


def laplacian(field: StaggeredGrid) -> StaggeredGrid:
    """The viscous operator, component-wise on the faces each component lives on.

    Symmetric and negative definite, so it removes energy and never adds any. At a wall
    the stencil is one-sided rather than wrapping; see the module docstring for which row
    each component takes and why they differ.

    This is the *homogeneous* operator. A moving wall contributes through ``wall_source``,
    which keeps the wall data out of the operator and in the residual.

    Args:
        field: Velocity on cell faces.

    A boundary this does not carry is refused by ``require_boundary``, which names what
    every part of the layer carries.

    Returns:
        The Laplacian of each component, on the same faces.
    """
    require_boundary(field.extrapolation, "diffusion")
    wraps = field.extrapolation == Extrapolation.PERIODIC

    spacing = field.dx
    components = []
    for normal, component in enumerate(field.components):
        total = jnp.zeros_like(component)
        for axis in range(field.spatial_dim):
            if wraps:
                neighbours = jnp.roll(component, 1, axis=axis) + jnp.roll(component, -1, axis=axis)
            else:
                # On its own axis the boundary face IS the wall and carries a known value,
                # which no-penetration makes zero; on any other axis the wall lies half a
                # cell outside and is reached by the mirror ghost -u_1.
                edge = 0.0 if axis == normal else -1.0
                lower = _shift(component, axis, 1, edge * _slab(component, axis, 0))
                upper = _shift(component, axis, -1, edge * _slab(component, axis, -1))
                neighbours = lower + upper
            total = total + (neighbours - 2.0 * component) / spacing[axis] ** 2
        components.append(total)
    return StaggeredGrid(tuple(components), field.box, field.extrapolation, field.resolution)


def wall_source(field: StaggeredGrid, walls: WallVelocity) -> StaggeredGrid:
    """The boundary term a moving wall adds to the viscous operator.

    The mirror ghost splits into a homogeneous part, which ``laplacian`` carries, and
    ``2 u_wall / h^2`` on the wall-adjacent row of each tangential component, which is
    this. Keeping them apart is what lets the operator stay a fixed symmetric matrix
    while the wall data remains an ordinary differentiable array.

    Args:
        field: Velocity on cell faces, giving the layout and cell size.
        walls: The prescribed tangential velocity on each wall.

    Returns:
        A field that is zero except on wall-adjacent rows.
    """
    require_boundary(field.extrapolation, "diffusion")
    if field.extrapolation == Extrapolation.PERIODIC:
        return StaggeredGrid(
            tuple(jnp.zeros_like(component) for component in field.components),
            field.box,
            field.extrapolation,
            field.resolution,
        )

    spacing = field.dx
    components = []
    for normal, component in enumerate(field.components):
        total = jnp.zeros_like(component)
        for axis in range(field.spatial_dim):
            prescribed = walls.values[normal][axis]
            if prescribed is None:
                continue
            scale = 2.0 / spacing[axis] ** 2
            total = total.at[
                tuple(slice(None) if i != axis else 0 for i in range(component.ndim))
            ].add(scale * prescribed[0])
            total = total.at[
                tuple(slice(None) if i != axis else -1 for i in range(component.ndim))
            ].add(scale * prescribed[1])
        components.append(total)
    return StaggeredGrid(tuple(components), field.box, field.extrapolation, field.resolution)


__all__ = ["WallVelocity", "laplacian", "wall_source"]
