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


#: Budget for ``jit(grad)`` against eager ``grad``, in units in the last place of the
#: gradient's own largest entry. **Not zero, and the reason is a measured backend
#: difference.** On this machine's GPU all three staggered operators agree bit-exactly,
#: but on the CPU backend ``ci.yml`` pins they do not: convection differs by 0.63 ULP and
#: the five-step trajectory by 1.38. Asserting exactness would have been green here and
#: red in CI. Sixteen leaves an order of headroom for XLA to reassociate differently
#: again, while a wrong adjoint under ``jit`` misses by O(1).
JIT_GRADIENT_ULP = 16.0

#: Budget for ``vmap(grad)`` against a Python loop over the same inputs. Measured
#: *exactly* zero for all three operators on both the CPU and the GPU backend -- batching
#: maps the reduction rather than reordering it -- so this is headroom against a future
#: backend choosing a different batched kernel, not slack covering a defect. A genuine
#: batching error puts one member's gradient on another and misses by O(1).
VMAP_GRADIENT_ULP = 4.0

#: Perturbation sizes for the Taylor remainder test, following pyadjoint's
#: ``taylor_test``: ``[0.01 / 2 ** i for i in range(4)]``.
TAYLOR_PERTURBATIONS = tuple(0.01 / 2**index for index in range(4))

#: The rate the first-order Taylor remainder must reach. Taylor's theorem puts the
#: remainder at O(eps^2) when the gradient is right, so the rate is 2; when the gradient
#: is wrong the ``eps`` term fails to cancel and the remainder is O(eps), giving 1. The
#: bar sits at 1.9 to leave room for the roundoff in the smallest residual, which is the
#: same allowance ``observed_orders``' callers make for a second-order operator.
TAYLOR_RATE_FLOOR = 1.9

#: Residuals below this are machine noise rather than a Taylor remainder, and their
#: ratios say nothing. pyadjoint warns at the same value.
TAYLOR_RESIDUAL_FLOOR = 1e-15


def taylor_remainder_rates(
    loss: Callable[[tuple[jax.Array, ...]], jax.Array],
    components: tuple[jax.Array, ...],
    *,
    seed: int = 7,
) -> tuple[list[float], list[float]]:
    """Convergence rates of the first-order Taylor remainder, as pyadjoint's taylor_test.

    This is the standard adjoint verification of the PDE-constrained optimisation
    literature, and it is used here rather than a hand-chosen tolerance because **its
    pass criterion is a convergence rate rather than a threshold anybody picked**.
    Taylor's theorem gives

        J(m + eps h) - J(m) - eps <grad J, h>  =  O(eps^2),

    so halving ``eps`` must quarter the remainder: rate 2. If the gradient is wrong by
    any fixed amount the ``eps`` term does not cancel, the remainder is O(eps), and the
    rate falls to 1. The two outcomes are a factor apart in a dimensionless quantity, so
    no tolerance has to be invented to tell them apart.

    Reference: ``pyadjoint/verification.py``, ``taylor_test`` and ``convergence_rates``
    (dolfin-adjoint/pyadjoint). The perturbations, the remainder and the
    log-ratio rate are taken from it; the Hessian term is dropped because only the
    first-order test is wanted here, which is pyadjoint's own default (``Hm=0``).

    Call it under ``jax.enable_x64(True)``: in float32 the remainder reaches the
    rounding floor at the first perturbation and the rates are noise.

    Args:
        loss: Scalar functional of a component tuple.
        components: The point to differentiate at.
        seed: Seed for the perturbation direction, so a failure is reproducible.

    Returns:
        The residuals and the rates between consecutive residuals. The caller asserts on
        ``min(rates)``, as pyadjoint's ``taylor_test`` returns that.
    """
    directions = tuple(
        jax.random.normal(jax.random.key(seed + index), component.shape, component.dtype)
        for index, component in enumerate(components)
    )
    norm = jnp.sqrt(sum(jnp.sum(each**2) for each in directions))
    directions = tuple(each / norm for each in directions)

    base = loss(components)
    gradient = jax.grad(loss)(components)
    slope = sum(
        jnp.vdot(part, direction) for part, direction in zip(gradient, directions, strict=True)
    )

    residuals = []
    for size in TAYLOR_PERTURBATIONS:
        step = jnp.asarray(size, dtype=components[0].dtype)
        moved = tuple(c + step * d for c, d in zip(components, directions, strict=True))
        residuals.append(float(jnp.abs(loss(moved) - base - step * slope)))

    rates = [
        float(np.log(fine / coarse) / np.log(0.5)) for coarse, fine in itertools.pairwise(residuals)
    ]
    return residuals, rates


def gradient_transform_agreement(
    loss: Callable[[tuple[jax.Array, ...]], jax.Array],
    components: tuple[jax.Array, ...],
) -> tuple[float, float]:
    """How far ``jit(grad)`` and ``vmap(grad)`` sit from plain ``grad``, in ULP.

    These are three *different* XLA programs, not one program run three ways, so a
    reverse rule can work eager and fail composed -- which is the whole subject of the
    ``lax.while_loop`` to ``lax.scan`` change: ``while_loop`` has no transpose rule at
    all, and ``vmap`` over an adjoint is the composition a batched boundary-control or
    ensemble problem actually runs.

    Both figures are relative to the gradient's own largest entry, so they do not change
    meaning with the operator's scale.

    Args:
        loss: Scalar functional of a component tuple.
        components: The point to differentiate at.

    Returns:
        ``(jit_versus_eager, vmap_versus_loop)``, each in ULP of ``max |grad|``.
    """
    eager = jax.grad(loss)(components)
    scale = max(float(jnp.max(jnp.abs(part))) for part in eager)
    unit = scale * float(jnp.finfo(components[0].dtype).eps)

    jitted = jax.jit(jax.grad(loss))(components)
    jit_gap = max(
        float(jnp.max(jnp.abs(left - right))) for left, right in zip(eager, jitted, strict=True)
    )

    scaled = tuple(1.5 * component for component in components)
    batch = tuple(jnp.stack([one, two]) for one, two in zip(components, scaled, strict=True))
    mapped = jax.vmap(jax.grad(loss))(batch)
    looped = (eager, jax.grad(loss)(scaled))
    vmap_gap = max(
        float(jnp.max(jnp.abs(part[index] - looped[index][axis])))
        for axis, part in enumerate(mapped)
        for index in range(2)
    )

    return jit_gap / unit, vmap_gap / unit


def gradient_versus_central_difference(
    loss: Callable[[tuple[jax.Array, ...]], jax.Array],
    components: tuple[jax.Array, ...],
    *,
    step: float,
    seed: int = 7,
) -> float:
    """Relative gap between reverse-mode AD and a central difference, along one direction.

    The reference calls only the primal, so it is independent of the adjoint being
    checked -- which is the point: comparing a gradient against another gradient cannot
    detect a transpose rule that is wrong in both modes.

    One fixed random direction reduces the derivative to a scalar, as
    :func:`jax.test_util.check_grads` does, so a step sweep stays affordable on a field
    with thousands of entries.

    **Call this under ``jax.enable_x64(True)``.** The suite pins ``jax_enable_x64`` off,
    and a float32 central difference of a sum of squares cancels catastrophically:
    measured on the Burgers solver the float32 relative error is 4.6e-02 at
    ``step=1e-4`` and the two evaluations cancel to *exactly zero* below ``1e-6``, so no
    float32 tolerance worth asserting exists. The same comparison in float64 reaches
    6e-11. ``tests/deployment/test_ase_calculator.py`` and
    ``tests/neural/atomistic/backbones/_helpers.py`` guard their force checks this way.

    ``step`` has no default on purpose: its best value is a property of the operator, not
    of this helper. An operator whose loss is quadratic -- any linear operator under a
    squared norm, such as the pressure projection -- has **no truncation term at all**,
    because the third derivative vanishes, so its error is pure roundoff and falls as
    ``step`` grows (measured best at 1e-2). An operator with a genuine third derivative
    has the usual U-shaped curve, with its minimum near 1e-4. One number for both costs
    two orders of accuracy.

    Args:
        loss: Scalar functional of a component tuple.
        components: The point to differentiate at.
        step: Central-difference step, chosen for the operator under test.
        seed: Seed for the direction, so a failure is reproducible.

    Returns:
        ``|differenced - analytic| / |analytic|``, the relative gap.
    """
    directions = tuple(
        jax.random.normal(jax.random.key(seed + index), component.shape, component.dtype)
        for index, component in enumerate(components)
    )
    norm = jnp.sqrt(sum(jnp.sum(each**2) for each in directions))
    directions = tuple(each / norm for each in directions)

    gradient = jax.grad(loss)(components)
    analytic = sum(
        jnp.vdot(part, direction) for part, direction in zip(gradient, directions, strict=True)
    )

    size = jnp.asarray(step, dtype=components[0].dtype)
    ahead = tuple(c + size * d for c, d in zip(components, directions, strict=True))
    behind = tuple(c - size * d for c, d in zip(components, directions, strict=True))
    differenced = (loss(ahead) - loss(behind)) / (2.0 * size)

    return float(jnp.abs(differenced - analytic) / jnp.abs(analytic))


__all__ = [
    "BOX",
    "CARRIED",
    "FLOAT32_EPS",
    "JIT_GRADIENT_ULP",
    "STREAMS",
    "TAYLOR_RATE_FLOOR",
    "TAYLOR_RESIDUAL_FLOOR",
    "VMAP_GRADIENT_ULP",
    "asymmetry",
    "face_coordinates",
    "flatten",
    "gradient_transform_agreement",
    "gradient_versus_central_difference",
    "observed_orders",
    "on_grid",
    "operator_matrix",
    "random_field",
    "stream_field",
    "taylor_remainder_rates",
    "unflatten",
]
