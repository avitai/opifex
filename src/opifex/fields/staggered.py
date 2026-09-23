"""Velocity on cell faces: the staggered Arakawa-C / MAC layout.

A collocated grid stores every component at the cell centre, so ``divergence`` and
``gradient`` are both two-point central differences and their composition reaches
``i +/- 2``. That operator annihilates the three checkerboard modes along with the
constant -- the odd-even decoupling -- and under a zero-gradient boundary the ``edge``
padding puts the first cell into its own difference, which gives the difference operator a
diagonal entry. No skew operator has one, so no choice of ghost value repairs it.

Staggering removes the boundary stencil instead of repairing it. Each velocity component
lives on the faces normal to its own axis, so a boundary-normal face is a *boundary
condition rather than an unknown*: the gradient writes only onto interior faces and never
needs a ghost pressure. The consequences are exact rather than asymptotic, and hold under
every boundary here:

* ``divergence`` is minus the adjoint of ``gradient`` in the plain Euclidean inner product,
  so the pressure Poisson operator ``divergence(gradient(.))`` is symmetric by construction;
* that operator is negative semi-definite, and its null space is the constants alone.

How many faces a boundary leaves along an axis of ``n`` cells is the whole of the layout:
periodic wraps, so the last face is the first and there are ``n``; a prescribed (Dirichlet)
velocity owns both outer faces, leaving ``n - 1`` unknowns; a zero-gradient boundary
determines neither, so both are stored and there are ``n + 1``.

References:
    * Harlow, Welch 1965 -- *Numerical calculation of time-dependent viscous incompressible
      flow of fluid with free surface*, Phys. Fluids 8(12), 2182.
    * Verstappen, Veldman 2003 -- *Symmetry-preserving discretization of turbulent flow*,
      J. Comput. Phys. 187(1), 343.
    * Sanderse 2013 -- *Energy-conserving Runge-Kutta methods for the incompressible
      Navier-Stokes equations*, J. Comput. Phys. 233, 100.
"""

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp

from opifex.fields.field import Box, Extrapolation


def face_count(cells: int, extrapolation: Extrapolation) -> int:
    """The number of unknown velocity faces along an axis of ``cells`` cells.

    Args:
        cells: Number of cells along the axis.
        extrapolation: Boundary condition on that axis.

    Returns:
        The number of faces stored as degrees of freedom.

    Raises:
        ValueError: If the boundary condition is not one the layout defines.
    """
    if extrapolation == Extrapolation.PERIODIC:
        return cells
    if extrapolation == Extrapolation.ZERO:
        return cells - 1
    if extrapolation == Extrapolation.NEUMANN:
        return cells + 1
    raise ValueError(f"Unknown extrapolation: {extrapolation}")


# Which boundaries each part of this layer carries. They differ, and a caller assembling a
# simulation from the parts should learn the whole picture at the first refusal rather than
# one operation at a time.
CARRIED_BOUNDARIES: Final[dict[str, frozenset[Extrapolation]]] = {
    "the grid operators": frozenset(
        {Extrapolation.PERIODIC, Extrapolation.ZERO, Extrapolation.NEUMANN}
    ),
    "the pressure projection": frozenset({Extrapolation.PERIODIC, Extrapolation.ZERO}),
    "convection": frozenset({Extrapolation.PERIODIC, Extrapolation.ZERO}),
    "diffusion": frozenset({Extrapolation.PERIODIC, Extrapolation.ZERO}),
}


def require_boundary(extrapolation: Extrapolation, operation: str) -> None:
    """Refuse a boundary an operation does not carry, naming what the layer does carry.

    Args:
        extrapolation: The boundary asked for.
        operation: A key of ``CARRIED_BOUNDARIES``.

    Raises:
        ValueError: If ``operation`` does not carry ``extrapolation``.
    """
    if extrapolation in CARRIED_BOUNDARIES[operation]:
        return
    carried = "; ".join(
        f"{name}: {'/'.join(sorted(b.value for b in boundaries))}"
        for name, boundaries in CARRIED_BOUNDARIES.items()
    )
    msg = (
        f"{operation} does not carry a {extrapolation.value} boundary on a staggered grid. "
        f"Across this layer: {carried}. What is missing is not a matter of padding: a "
        "zero-gradient boundary adds a face that neither the projection's transform "
        "diagonalises nor the one-sided viscous closure covers, so it would be an "
        "approximation rather than the scheme."
    )
    raise ValueError(msg)


@jax.tree_util.register_pytree_node_class
class StaggeredGrid:
    """A vector field on the faces of a uniform Cartesian grid.

    Component ``d`` lives on the faces normal to axis ``d``, so the components have
    different shapes and are held as a tuple rather than one stacked array.

    Attributes:
        components: One array per axis, on that axis's faces.
        box: Physical domain bounds.
        extrapolation: Boundary condition type.
        resolution: Number of cells along each axis.
    """

    __slots__ = ("box", "components", "extrapolation", "resolution")

    def __init__(
        self,
        components: tuple[jax.Array, ...],
        box: Box,
        extrapolation: Extrapolation,
        resolution: tuple[int, ...],
    ) -> None:
        """Initialize a staggered grid.

        Args:
            components: One array per axis, on that axis's faces.
            box: Physical domain bounds.
            extrapolation: Boundary condition type.
            resolution: Number of cells along each axis.
        """
        self.components = components
        self.box = box
        self.extrapolation = extrapolation
        self.resolution = resolution

    @staticmethod
    def component_shapes(
        resolution: tuple[int, ...], extrapolation: Extrapolation
    ) -> tuple[tuple[int, ...], ...]:
        """The shape of each component, staggered along its own axis.

        Args:
            resolution: Number of cells along each axis.
            extrapolation: Boundary condition type.

        Returns:
            One shape per axis.
        """
        return tuple(
            tuple(
                face_count(cells, extrapolation) if axis == normal else cells
                for axis, cells in enumerate(resolution)
            )
            for normal in range(len(resolution))
        )

    @classmethod
    def zeros(
        cls, resolution: tuple[int, ...], box: Box, extrapolation: Extrapolation
    ) -> StaggeredGrid:
        """A grid of the right shapes holding zeros.

        Args:
            resolution: Number of cells along each axis.
            box: Physical domain bounds.
            extrapolation: Boundary condition type.

        Returns:
            A staggered grid of zeros.
        """
        shapes = cls.component_shapes(resolution, extrapolation)
        return cls(tuple(jnp.zeros(shape) for shape in shapes), box, extrapolation, resolution)

    @property
    def spatial_dim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.resolution)

    @property
    def dx(self) -> jax.Array:
        """Cell size in each dimension."""
        return self.box.size / jnp.asarray(self.resolution, dtype=jnp.float32)

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], tuple[object, ...]]:
        """Flatten for JAX pytree protocol."""
        return self.components, (self.box, self.extrapolation, self.resolution)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[object, ...], children: tuple[jax.Array, ...]
    ) -> StaggeredGrid:
        """Unflatten from JAX pytree protocol."""
        box, extrapolation, resolution = aux_data
        return cls(tuple(children), box, extrapolation, resolution)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        """Readable summary."""
        shapes = ", ".join(str(component.shape) for component in self.components)
        return (
            f"StaggeredGrid(components=({shapes}), resolution={self.resolution}, "
            f"extrapolation={self.extrapolation.value})"
        )


def divergence(field: StaggeredGrid) -> jax.Array:
    """The divergence of a face velocity, on cell centres.

    Each axis contributes the difference of the two faces bounding the cell, which is the
    exact flux balance of the finite volume. Faces the boundary prescribes carry no
    unknown, so they contribute nothing and are simply absent from the difference.

    Args:
        field: Velocity on cell faces.

    Returns:
        Divergence at cell centres, of shape ``field.resolution``.
    """
    total = jnp.zeros(field.resolution)
    for axis, component in enumerate(field.components):
        spacing = field.dx[axis]
        if field.extrapolation == Extrapolation.PERIODIC:
            total = total + (component - jnp.roll(component, 1, axis=axis)) / spacing
        else:
            padding = [(0, 0)] * field.spatial_dim
            if field.extrapolation == Extrapolation.ZERO:
                padding[axis] = (1, 1)
            padded = jnp.pad(component, padding, mode="constant", constant_values=0.0)
            upper = tuple(
                slice(1, None) if index == axis else slice(None)
                for index in range(field.spatial_dim)
            )
            lower = tuple(
                slice(None, -1) if index == axis else slice(None)
                for index in range(field.spatial_dim)
            )
            total = total + (padded[upper] - padded[lower]) / spacing
    return total


def gradient(values: jax.Array, like: StaggeredGrid) -> StaggeredGrid:
    """The gradient of a cell-centred scalar, on the faces ``like`` stores.

    Written onto interior faces only, which is what makes this minus the adjoint of
    ``divergence``: a face the boundary prescribes is not a degree of freedom, so no ghost
    value of ``values`` is ever required and none can break the transpose.

    Args:
        values: Scalar at cell centres, of shape ``like.resolution``.
        like: The grid whose face layout and boundary the result takes.

    Returns:
        The gradient on cell faces.
    """
    components = []
    for axis in range(like.spatial_dim):
        spacing = like.dx[axis]
        if like.extrapolation == Extrapolation.PERIODIC:
            component = (jnp.roll(values, -1, axis=axis) - values) / spacing
        else:
            upper = tuple(
                slice(1, None) if index == axis else slice(None)
                for index in range(like.spatial_dim)
            )
            lower = tuple(
                slice(None, -1) if index == axis else slice(None)
                for index in range(like.spatial_dim)
            )
            # The pressure boundary is derived from the velocity one, not chosen: where the
            # velocity is prescribed the pressure has a vanishing normal derivative and the
            # face carries no unknown, and where the velocity is free the pressure is
            # prescribed instead. So a zero-gradient velocity pads the pressure with zero
            # and differences across it, rather than zeroing the gradient on the outer
            # faces -- which would not be the adjoint of the divergence that counts them.
            scalar = values
            if like.extrapolation == Extrapolation.NEUMANN:
                padding = [(0, 0)] * like.spatial_dim
                padding[axis] = (1, 1)
                scalar = jnp.pad(values, padding, mode="constant", constant_values=0.0)
            component = (scalar[upper] - scalar[lower]) / spacing
        components.append(component)
    return StaggeredGrid(tuple(components), like.box, like.extrapolation, like.resolution)


__all__ = ["StaggeredGrid", "divergence", "face_count", "gradient"]
