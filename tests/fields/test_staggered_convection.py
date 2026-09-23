"""Convection on the staggered layout, and the one property it is chosen for.

Written in divergence form with arithmetic-mean interpolation (Harlow, Welch 1965), the
convective operator is **skew-symmetric about a discretely divergence-free transporting
velocity**. Skew means it produces no energy: the inviscid scheme neither damps nor
amplifies, so whatever energy drift a simulation shows is the time integrator's and
vanishes as the step shrinks, rather than a spatial error that no amount of refinement
removes. First-order upwind, by contrast, is almost pure dissipation.

A wall is carried as well as a periodic boundary. There the property needs one more
condition -- impermeability, ``u.n = 0``, rather than no-slip -- and the wall stencil is
checked twice over: exactly skew, and second-order consistent against a closed-form
transport, because skew-symmetry alone is satisfied by operators that compute the wrong
thing.

The qualifier is the whole of it, and it is asserted here with a negative control: the
proof holds *about a divergence-free field and only there*. Fed a transporting velocity
that is not discretely solenoidal, the same operator produces energy at order one. This
is why a projection method must project at every Runge-Kutta stage rather than once per
step -- a stage that starts from an unprojected field is outside the regime the property
covers.

References:
    * Harlow, Welch 1965 -- *Numerical calculation of time-dependent viscous incompressible
      flow of fluid with free surface*, Phys. Fluids 8(12), 2182.
    * Verstappen, Veldman 2003 -- *Symmetry-preserving discretization of turbulent flow*,
      J. Comput. Phys. 187(1), 343.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import divergence, StaggeredGrid
from opifex.fields.staggered_convection import convect
from opifex.fields.staggered_pressure import project


BOX = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
RESOLUTION = (8, 8)
FLOAT32_EPS = float(np.finfo(np.float32).eps)


CARRIED = [Extrapolation.PERIODIC, Extrapolation.ZERO]


def _random(resolution, seed, extrapolation=Extrapolation.PERIODIC):
    shapes = StaggeredGrid.component_shapes(resolution, extrapolation)
    keys = jax.random.split(jax.random.key(seed), len(shapes))
    return StaggeredGrid(
        tuple(jax.random.normal(key, shape) for key, shape in zip(keys, shapes, strict=True)),
        BOX,
        extrapolation,
        resolution,
    )


def _on_grid(scalar):
    """A two-argument scalar function evaluated over a pair of 2D coordinate arrays."""
    return jax.vmap(jax.vmap(scalar))


def _stream(psi, resolution, extrapolation):
    """A divergence-free, impermeable field from a stream function, and its exact transport.

    ``u = d(psi)/dy``, ``v = -d(psi)/dx`` is divergence free identically, and any ``psi``
    vanishing on the boundary gives ``u.n = 0`` there. The target ``div(u (x) u)`` is taken
    by autodiff of the analytic products rather than written out by hand, so a second
    stream function costs one line and cannot carry an algebra slip.
    """

    def u(x: jax.Array, y: jax.Array) -> jax.Array:
        return jax.grad(psi, 1)(x, y)

    def v(x: jax.Array, y: jax.Array) -> jax.Array:
        return -jax.grad(psi, 0)(x, y)

    velocity = (u, v)
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
    (cells,) = set(resolution)
    spacing = 1.0 / cells
    faces = jnp.arange(1, cells) * spacing
    centres = (jnp.arange(cells) + 0.5) * spacing
    points = (
        jnp.meshgrid(faces, centres, indexing="ij"),
        jnp.meshgrid(centres, faces, indexing="ij"),
    )
    field = StaggeredGrid(
        tuple(_on_grid(f)(*p) for f, p in zip(velocity, points, strict=True)),
        BOX,
        extrapolation,
        resolution,
    )
    return field, tuple(_on_grid(f)(*p) for f, p in zip(exact, points, strict=True))


# Three stream functions, because one can be accidentally blind. The first alone measured
# second order at the wall for the *diffusion* stencil whose wall row is provably O(1),
# purely because its curvature vanishes there; a property asserted from one field is not
# asserted. Each vanishes on the whole boundary, so every normal component does too.
STREAMS = {
    "sin(pi x) sin(pi y)": lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y),
    "sin(2 pi x) sin(pi y)": lambda x, y: jnp.sin(2 * jnp.pi * x) * jnp.sin(jnp.pi * y),
    "sin(pi x)^2 sin(2 pi y)": lambda x, y: jnp.sin(jnp.pi * x) ** 2 * jnp.sin(2 * jnp.pi * y),
}


def _relative_divergence(projected: StaggeredGrid, before: StaggeredGrid) -> float:
    """Residual divergence as a fraction of what the field came in with."""
    return float(jnp.linalg.norm(divergence(projected)) / jnp.linalg.norm(divergence(before)))


def _flatten(field: StaggeredGrid) -> jax.Array:
    return jnp.concatenate([component.ravel() for component in field.components])


def _unflatten(flat: jax.Array, resolution, extrapolation) -> StaggeredGrid:
    pieces, offset = [], 0
    for shape in StaggeredGrid.component_shapes(resolution, extrapolation):
        size = int(np.prod(shape))
        pieces.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return StaggeredGrid(tuple(pieces), BOX, extrapolation, resolution)


def _asymmetry(matrix: np.ndarray) -> tuple[float, float]:
    """The off-diagonal and diagonal parts of ``C + C^T``, relative to the matrix scale.

    They fail for different reasons and must be read apart. The off-diagonals of this
    operator are skew *identically*, for any transporting field, because they are
    ``-f_i/2`` and ``+f_{i+1}/2`` either side of the diagonal -- so a nonzero off-diagonal
    part means a stencil or indexing error, nothing else. The diagonal carries half the
    divergence of the transporting mass flux over each control volume, so it vanishes only
    when that field is discretely divergence free and every boundary is impermeable.

    Reading them together hides both: ``max|C + C^T| / max|C|`` saturates at exactly 2 when
    the largest entry lies on the diagonal, which is the case here, so a broken wall
    stencil and a carrier with divergence produce the identical number.
    """
    asymmetric = np.abs(matrix + matrix.T)
    scale = np.abs(matrix).max()
    diagonal = np.abs(np.diag(asymmetric)).max()
    off_diagonal = (asymmetric - np.diag(np.diag(asymmetric))).max()
    return off_diagonal / scale, diagonal / scale


def _convection_matrix(velocity: StaggeredGrid) -> np.ndarray:
    """The matrix of ``w -> convect(w, velocity)``, the transporting field held fixed."""
    size = int(_flatten(velocity).size)
    columns = []
    for index in range(size):
        basis = _unflatten(
            jnp.zeros(size).at[index].set(1.0), velocity.resolution, velocity.extrapolation
        )
        columns.append(np.asarray(_flatten(convect(basis, velocity))))
    return np.stack(columns, axis=1)


class TestSkewSymmetry:
    """No energy production -- but only about a divergence-free transporting velocity."""

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_is_skew_about_a_divergence_free_velocity(
        self, extrapolation: Extrapolation
    ) -> None:
        raw = _random(RESOLUTION, seed=0, extrapolation=extrapolation)
        solenoidal, _ = project(raw)
        # The precondition belongs in the units the projection guarantees: it leaves a
        # fixed fraction of the incoming divergence, not a fixed absolute norm, and the
        # incoming norm here is ~123 because differencing random noise divides by dx.
        assert _relative_divergence(solenoidal, raw) < 1e-6

        off_diagonal, diagonal = _asymmetry(_convection_matrix(solenoidal))

        assert off_diagonal <= 1e-12, "a nonzero off-diagonal part is a stencil error"
        assert diagonal <= 1e-5

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_is_not_skew_about_a_field_with_divergence(
        self, extrapolation: Extrapolation
    ) -> None:
        # The negative control. Without it, an operator that is skew for the trivial
        # reason of being zero, or a test fixture that is accidentally solenoidal, would
        # pass the assertion above and prove nothing.
        divergent = _random(RESOLUTION, seed=1, extrapolation=extrapolation)
        assert float(jnp.linalg.norm(divergence(divergent))) > 1.0

        off_diagonal, diagonal = _asymmetry(_convection_matrix(divergent))

        # The control has to be distinguishable from a broken stencil, not merely large:
        # divergence in the carrier shows up on the diagonal alone.
        assert off_diagonal <= 1e-12
        assert diagonal > 0.1

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_the_energy_it_produces_is_zero(self, extrapolation: Extrapolation) -> None:
        # The same statement as a quadratic form, which is what actually matters to a
        # simulation: <w, C w> is the rate energy enters through the convective term.
        raw = _random(RESOLUTION, seed=2, extrapolation=extrapolation)
        solenoidal, _ = project(raw)
        assert _relative_divergence(solenoidal, raw) < 1e-6
        transported = _random(RESOLUTION, seed=3, extrapolation=extrapolation)

        produced = _flatten(transported) @ _flatten(convect(transported, solenoidal))
        scale = jnp.sum(jnp.abs(_flatten(transported) * _flatten(convect(transported, solenoidal))))

        assert abs(float(produced) / float(scale)) <= 1e-5


class TestTransforms:
    """jit, vmap and reverse-mode grad, all three, on both branches.

    The wall branch is not the periodic one with different numbers in it: it slices and
    concatenates where the periodic branch rolls, and produces arrays of a different
    shape. Covering only the periodic path would leave that code untraced.
    """

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_traces_once_under_jit(self, extrapolation: Extrapolation) -> None:
        velocity = _random(RESOLUTION, seed=4, extrapolation=extrapolation)
        traces = {"count": 0}

        def counted(field: StaggeredGrid) -> StaggeredGrid:
            traces["count"] += 1
            return convect(field, field)

        compiled = jax.jit(counted)
        for _ in range(4):
            compiled(velocity)

        assert traces["count"] == 1

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_differentiates_in_reverse_mode(self, extrapolation: Extrapolation) -> None:
        velocity = _random(RESOLUTION, seed=5, extrapolation=extrapolation)

        def loss(components: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(components, BOX, extrapolation, RESOLUTION)
            return jnp.sum(_flatten(convect(field, field)) ** 2)

        grads = jax.grad(loss)(velocity.components)

        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_maps_over_a_batch(self, extrapolation: Extrapolation) -> None:
        velocity = _random(RESOLUTION, seed=6, extrapolation=extrapolation)
        batch = tuple(jnp.stack([c, c]) for c in velocity.components)

        def run(components: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(components, BOX, extrapolation, RESOLUTION)
            return convect(field, field).components[0]

        mapped = jax.vmap(run)(batch)

        assert mapped.shape == (2, *velocity.components[0].shape)


class TestWalls:
    """The wall stencil, which has to be consistent as well as skew.

    Skew-symmetry alone is satisfied by operators that compute the wrong thing -- the zero
    operator is perfectly skew. Consistency against a closed-form transport is what says
    the stencil is also correct, and the two together are what the wall treatment claims.

    The wall rows are measured *apart from* the interior. Reading one maximum over the
    whole field lets a second-order interior hide a boundary row of any order, since the
    interior is the larger of the two on two of the three fields below.
    """

    @pytest.mark.parametrize("cells", [16, 64])
    def test_the_sampled_field_is_discretely_divergence_free(self, cells: int) -> None:
        # A positive control on the sampling positions, not on the operator: read at cell
        # centres rather than on their own faces, this would be first order in the cell
        # size rather than at rounding.
        #
        # The bound is the rounding model rather than a recorded measurement, which is
        # why it is expressed in the units the model predicts. Differencing terms of size
        # ``max|u| / dx`` over ``N`` entries accumulates about ``eps * max|u| / dx *
        # sqrt(N)``; measured, the ratio sits at 0.48 to 0.61 eps and is flat from 16
        # cells to 128, which is what says it is rounding and not a discretisation error
        # hiding at this one size.
        field, _ = _stream(STREAMS["sin(pi x) sin(pi y)"], (cells, cells), Extrapolation.ZERO)
        residual = divergence(field)
        speed = max(float(jnp.max(jnp.abs(component))) for component in field.components)
        rounding = speed * cells * np.sqrt(residual.size)

        assert float(jnp.linalg.norm(residual)) < 2.0 * FLOAT32_EPS * rounding

    @pytest.mark.parametrize("stream", list(STREAMS))
    def test_the_wall_rows_converge_at_second_order(self, stream: str) -> None:
        # 128 cells and beyond is where float32 rounding starts to show in the ratio, so
        # the window stops where truncation still dominates it.
        walls, interiors = [], []
        for cells in (16, 32, 64):
            field, exact = _stream(STREAMS[stream], (cells, cells), Extrapolation.ZERO)
            residual = np.asarray(convect(field, field).components[0] - exact[0])
            # The x-component's tangential walls are the first and last columns in y.
            walls.append(max(np.abs(residual[:, 0]).max(), np.abs(residual[:, -1]).max()))
            interiors.append(np.abs(residual[:, 1:-1]).max())

        for label, errors in (("wall", walls), ("interior", interiors)):
            for coarse, fine in itertools.pairwise(errors):
                assert coarse / fine > 3.6, f"{label} is not second order: {errors}"


class TestBoundariesNotYetCarried:
    """What the operator refuses, and why it refuses rather than approximating."""

    def test_a_boundary_outside_the_contract_is_refused(self) -> None:
        field = StaggeredGrid.zeros(RESOLUTION, BOX, Extrapolation.NEUMANN)

        with pytest.raises(ValueError, match="does not carry"):
            convect(field, field)
