"""Shared fixtures and measurement helpers for the staggered-layout test modules.

These are the pieces every staggered test needs and none of them owns: the domain, a
random field on a given boundary, the flatten/unflatten pair that lets an operator be
read as a matrix, and the two matrix diagnostics.

Two of them encode a measurement lesson rather than a convenience, and are the reason
this module exists rather than a copy per test file:

``operator_matrix`` -- structure claims about these operators (adjointness, skewness,
symmetry, definiteness) are statements about the matrix, not about any one field, so
every such test assembles one. Four modules had grown three different spellings.

``asymmetry`` -- the off-diagonal and diagonal parts of ``A + A^T`` fail for different
reasons and must be read apart; the combined ratio saturates and cannot tell them apart.
"""

import itertools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import StaggeredGrid


BOX = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
FLOAT32_EPS = float(np.finfo(np.float32).eps)

# The boundaries the staggered layer carries end to end. Parametrising over this rather
# than over a hand-written list is what keeps a test honest when the contract widens.
CARRIED = [Extrapolation.PERIODIC, Extrapolation.ZERO]


def random_field(
    resolution: tuple[int, ...],
    extrapolation: Extrapolation,
    seed: int = 0,
) -> StaggeredGrid:
    """A field of standard normal noise on the faces ``extrapolation`` implies.

    The boundary is a parameter here and in every caller: a helper that pins one while
    the test varies another measures the helper, not the operator.

    Args:
        resolution: Number of cells along each axis.
        extrapolation: Boundary condition, which sets the face counts.
        seed: PRNG seed.

    Returns:
        A staggered field with the shapes ``extrapolation`` implies.
    """
    shapes = StaggeredGrid.component_shapes(resolution, extrapolation)
    keys = jax.random.split(jax.random.key(seed), len(shapes))
    return StaggeredGrid(
        tuple(jax.random.normal(key, shape) for key, shape in zip(keys, shapes, strict=True)),
        BOX,
        extrapolation,
        resolution,
    )


def flatten(field: StaggeredGrid) -> jax.Array:
    """Every component's entries end to end, so an operator can be read as a matrix."""
    return jnp.concatenate([component.ravel() for component in field.components])


def unflatten(
    flat: jax.Array, resolution: tuple[int, ...], extrapolation: Extrapolation
) -> StaggeredGrid:
    """The inverse of ``flatten`` for the layout ``extrapolation`` implies.

    Args:
        flat: A vector of the size ``flatten`` produces.
        resolution: Number of cells along each axis.
        extrapolation: Boundary condition, which sets the face counts.

    Returns:
        The staggered field whose ``flatten`` is ``flat``.
    """
    pieces, offset = [], 0
    for shape in StaggeredGrid.component_shapes(resolution, extrapolation):
        size = int(np.prod(shape))
        pieces.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return StaggeredGrid(tuple(pieces), BOX, extrapolation, resolution)


def operator_matrix(
    operator: Callable[[StaggeredGrid], StaggeredGrid | jax.Array],
    resolution: tuple[int, ...],
    extrapolation: Extrapolation,
) -> np.ndarray:
    """The matrix of a linear operator on the staggered layout, column by column.

    The output may be another staggered field or a scalar field on the cells, so that
    ``divergence`` and ``laplacian`` can both be read as matrices; the result is then
    rectangular rather than square.

    Args:
        operator: A linear map from a staggered field to a field.
        resolution: Number of cells along each axis.
        extrapolation: Boundary condition, which sets the face counts.

    Returns:
        A dense matrix, ``operator`` applied to each basis vector in turn.
    """
    size = sum(
        int(np.prod(shape)) for shape in StaggeredGrid.component_shapes(resolution, extrapolation)
    )
    columns = []
    for index in range(size):
        basis = unflatten(jnp.zeros(size).at[index].set(1.0), resolution, extrapolation)
        produced = operator(basis)
        columns.append(
            np.asarray(
                flatten(produced) if isinstance(produced, StaggeredGrid) else produced.ravel()
            )
        )
    return np.stack(columns, axis=1)


def asymmetry(matrix: np.ndarray) -> tuple[float, float]:
    """The off-diagonal and diagonal parts of ``A + A^T``, relative to the matrix scale.

    Read them apart. ``max|A + A^T| / max|A|`` saturates at exactly 2 whenever the
    largest entry lies on the diagonal, which is the case for the convective operator, so
    a broken boundary stencil and a transporting field with divergence produce the
    identical number and the metric cannot distinguish them.

    Args:
        matrix: The operator to examine.

    Returns:
        ``(off_diagonal, diagonal)``, each relative to ``max|A|``.
    """
    asymmetric = np.abs(matrix + matrix.T)
    scale = np.abs(matrix).max()
    diagonal = np.abs(np.diag(asymmetric)).max()
    off_diagonal = (asymmetric - np.diag(np.diag(asymmetric))).max()
    return off_diagonal / scale, diagonal / scale


def on_grid(scalar: Callable[[jax.Array, jax.Array], jax.Array]) -> Callable[..., jax.Array]:
    """A two-argument scalar function lifted over a pair of 2D coordinate arrays."""
    return jax.vmap(jax.vmap(scalar))


def face_coordinates(
    resolution: tuple[int, ...], extrapolation: Extrapolation
) -> tuple[tuple[jax.Array, ...], ...]:
    """The physical coordinates of each component's faces, one meshgrid per component.

    The convention is the whole of it: a component is staggered on its own axis and
    centred on every other. Sampling an initial condition at the wrong one of those is a
    silent first-order error rather than a failure.

    Args:
        resolution: Number of cells along each axis, equal on every axis.
        extrapolation: Boundary condition, which sets the face counts.

    Returns:
        One tuple of coordinate arrays per component.
    """
    (cells,) = set(resolution)
    spacing = 1.0 / cells
    centres = (jnp.arange(cells) + 0.5) * spacing
    staggered = (
        jnp.arange(cells) * spacing
        if extrapolation == Extrapolation.PERIODIC
        else jnp.arange(1, cells) * spacing
    )
    return tuple(
        jnp.meshgrid(
            *(staggered if axis == normal else centres for axis in range(len(resolution))),
            indexing="ij",
        )
        for normal in range(len(resolution))
    )


def stream_field(
    psi: Callable[[jax.Array, jax.Array], jax.Array],
    resolution: tuple[int, ...],
    extrapolation: Extrapolation,
) -> tuple[StaggeredGrid, tuple[jax.Array, ...]]:
    """A divergence-free, impermeable field from a stream function, with its transport.

    ``u = d(psi)/dy``, ``v = -d(psi)/dx`` is divergence free identically, and a ``psi``
    vanishing on the boundary makes ``u.n = 0`` there. The target ``div(u (x) u)`` is
    taken by autodiff of the analytic products rather than written out by hand, so a
    further stream function costs one line and cannot carry an algebra slip.

    Args:
        psi: The stream function, of ``(x, y)``.
        resolution: Number of cells along each axis.
        extrapolation: Boundary condition, which sets the face counts.

    Returns:
        ``(field, exact)``, the sampled velocity and the analytic convective term.
    """

    def u(x: jax.Array, y: jax.Array) -> jax.Array:
        return jax.grad(psi, 1)(x, y)

    def v(x: jax.Array, y: jax.Array) -> jax.Array:
        return -jax.grad(psi, 0)(x, y)

    exact = (
        lambda x, y: (
            jax.grad(lambda a, b: u(a, b) * u(a, b), 0)(x, y)
            + jax.grad(lambda a, b: u(a, b) * v(a, b), 1)(x, y)
        ),
        lambda x, y: (
            jax.grad(lambda a, b: v(a, b) * u(a, b), 0)(x, y)
            + jax.grad(lambda a, b: v(a, b) * v(a, b), 1)(x, y)
        ),
    )
    points = face_coordinates(resolution, extrapolation)
    field = StaggeredGrid(
        tuple(on_grid(f)(*p) for f, p in zip((u, v), points, strict=True)),
        BOX,
        extrapolation,
        resolution,
    )
    return field, tuple(on_grid(f)(*p) for f, p in zip(exact, points, strict=True))


# Three stream functions, because one can be accidentally blind. A single field whose
# curvature vanishes at the wall reports second order for a boundary row that is not, so
# a property asserted from one field is not asserted.
STREAMS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "sin(pi x) sin(pi y)": lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y),
    "sin(2 pi x) sin(pi y)": lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.sin(jnp.pi * y),
    "sin(pi x)^2 sin(2 pi y)": lambda x, y: jnp.sin(jnp.pi * x) ** 2 * jnp.sin(2 * jnp.pi * y),
}


def observed_orders(errors: Sequence[float]) -> list[float]:
    """The order between each consecutive pair of a halving refinement sequence."""
    return [float(np.log2(coarse / fine)) for coarse, fine in itertools.pairwise(errors)]


__all__ = [
    "BOX",
    "CARRIED",
    "FLOAT32_EPS",
    "STREAMS",
    "asymmetry",
    "face_coordinates",
    "flatten",
    "observed_orders",
    "on_grid",
    "operator_matrix",
    "random_field",
    "stream_field",
    "unflatten",
]
