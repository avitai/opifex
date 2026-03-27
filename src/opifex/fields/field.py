"""Field abstractions for scientific computing on structured grids.

Provides immutable JAX pytree field types for representing scalar and vector
fields on uniform grids with physical coordinate support. Inspired by PhiFlow's
field API, implemented in pure JAX for JIT/vmap/pmap compatibility.

Key types:
    - ``Box``: Rectangular physical domain with lower/upper bounds
    - ``Extrapolation``: Boundary condition type (zero, periodic, neumann)
    - ``CenteredGrid``: Scalar/vector field on cell-centered grid points
"""

from __future__ import annotations

from enum import Enum

import jax
import jax.numpy as jnp


class Extrapolation(Enum):
    """Boundary extrapolation types for grid fields.

    Determines how values outside the grid domain are handled.
    """

    ZERO = "zero"
    PERIODIC = "periodic"
    NEUMANN = "neumann"  # zero-gradient


class Box:
    """Rectangular physical domain.

    Stores the lower and upper bounds of an axis-aligned box in N dimensions.
    Immutable — all properties are derived from the bounds.

    Attributes:
        lower: Lower corner coordinates.
        upper: Upper corner coordinates.
    """

    __slots__ = ("_lower", "_upper")

    def __init__(self, lower: tuple[float, ...], upper: tuple[float, ...]) -> None:
        """Initialize box from lower and upper corners.

        Args:
            lower: Lower corner coordinates.
            upper: Upper corner coordinates.

        Raises:
            ValueError: If dimensions don't match or bounds are invalid.
        """
        if len(lower) != len(upper):
            raise ValueError(f"Dimension mismatch: lower={len(lower)}, upper={len(upper)}")
        self._lower = jnp.array(lower)
        self._upper = jnp.array(upper)

    @property
    def lower(self) -> jnp.ndarray:
        """Lower corner coordinates."""
        return self._lower

    @property
    def upper(self) -> jnp.ndarray:
        """Upper corner coordinates."""
        return self._upper

    @property
    def spatial_dim(self) -> int:
        """Number of spatial dimensions."""
        return len(self._lower)

    @property
    def size(self) -> jnp.ndarray:
        """Box extent in each dimension."""
        return self._upper - self._lower

    @property
    def center(self) -> jnp.ndarray:
        """Box center point."""
        return (self._lower + self._upper) / 2.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return NotImplemented
        lower_eq = jnp.allclose(self._lower, other._lower)
        upper_eq = jnp.allclose(self._upper, other._upper)
        return bool(lower_eq and upper_eq)

    def __hash__(self) -> int:
        return hash((tuple(float(x) for x in self._lower), tuple(float(x) for x in self._upper)))

    def __repr__(self) -> str:
        lo = tuple(float(x) for x in self._lower)
        hi = tuple(float(x) for x in self._upper)
        return f"Box(lower={lo}, upper={hi})"


@jax.tree_util.register_pytree_node_class
class CenteredGrid:
    """Scalar or vector field on a cell-centered uniform grid.

    Values are stored at cell centers. The grid carries domain metadata
    (physical bounds and boundary conditions) and supports element-wise
    arithmetic. Registered as a JAX pytree so it can be used with
    ``jax.jit``, ``jax.vmap``, and ``jax.grad``.

    Attributes:
        values: Field data array, shape matches grid resolution.
        box: Physical domain bounds.
        extrapolation: Boundary condition type.
    """

    __slots__ = ("box", "extrapolation", "values")

    def __init__(
        self,
        values: jnp.ndarray,
        box: Box,
        extrapolation: Extrapolation = Extrapolation.ZERO,
    ) -> None:
        """Initialize centered grid.

        Args:
            values: Field values at cell centers.
            box: Physical domain bounds.
            extrapolation: Boundary condition type.
        """
        self.values = values
        self.box = box
        self.extrapolation = extrapolation

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape."""
        return self.values.shape

    @property
    def spatial_dim(self) -> int:
        """Number of spatial dimensions."""
        return self.box.spatial_dim

    @property
    def resolution(self) -> tuple[int, ...]:
        """Grid resolution (number of cells per dimension)."""
        return self.values.shape[: self.spatial_dim]

    @property
    def dx(self) -> jnp.ndarray:
        """Cell size in each dimension."""
        res = jnp.array(self.resolution, dtype=jnp.float32)
        return self.box.size / res

    def cell_centers(self) -> jnp.ndarray:
        """Compute physical coordinates of cell centers.

        Returns:
            Meshgrid array of shape (*resolution, spatial_dim).
        """
        axes = []
        for d in range(self.spatial_dim):
            n = self.resolution[d]
            lo = self.box.lower[d]
            hi = self.box.upper[d]
            dx_d = (hi - lo) / n
            axes.append(jnp.linspace(lo + dx_d / 2, hi - dx_d / 2, n))

        grids = jnp.meshgrid(*axes, indexing="ij")
        return jnp.stack(grids, axis=-1)

    # --- Arithmetic ---

    def __add__(self, other: CenteredGrid | float) -> CenteredGrid:
        if isinstance(other, CenteredGrid):
            return CenteredGrid(self.values + other.values, self.box, self.extrapolation)
        return CenteredGrid(self.values + other, self.box, self.extrapolation)

    def __sub__(self, other: CenteredGrid | float) -> CenteredGrid:
        if isinstance(other, CenteredGrid):
            return CenteredGrid(self.values - other.values, self.box, self.extrapolation)
        return CenteredGrid(self.values - other, self.box, self.extrapolation)

    def __mul__(self, other: CenteredGrid | float) -> CenteredGrid:
        if isinstance(other, CenteredGrid):
            return CenteredGrid(self.values * other.values, self.box, self.extrapolation)
        return CenteredGrid(self.values * other, self.box, self.extrapolation)

    def __rmul__(self, other: float) -> CenteredGrid:
        return self.__mul__(other)

    def __neg__(self) -> CenteredGrid:
        return CenteredGrid(-self.values, self.box, self.extrapolation)

    # --- JAX Pytree ---

    def tree_flatten(self):
        """Flatten for JAX pytree protocol."""
        return (self.values,), (self.box, self.extrapolation)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from JAX pytree protocol."""
        box, extrapolation = aux_data
        return cls(values=children[0], box=box, extrapolation=extrapolation)

    def __repr__(self) -> str:
        return (
            f"CenteredGrid(shape={self.shape}, box={self.box}, "
            f"extrapolation={self.extrapolation.value})"
        )


__all__ = ["Box", "CenteredGrid", "Extrapolation"]
