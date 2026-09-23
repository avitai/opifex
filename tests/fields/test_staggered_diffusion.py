"""The viscous operator at a solid wall, and the wall velocity that drives it.

The wall treatment is the mirror ghost ``u_ghost = 2 u_wall - u_1`` of Sanderse,
Verstappen and Koren 2014 (Eq. 92), which makes the wall-adjacent row of the
**tangential** component ``(-3 u_1 + u_2 + 2 u_wall)/h^2``. The **normal** component's
own-axis faces lie on the wall and are not unknowns, so its row is the ordinary
three-point difference with the boundary value substituted, diagonal ``-2/h^2``.

What is asserted here, and why each assertion is present, comes from measuring which
candidate defect each one actually catches. Four plausible wrong stencils were built and
run against every candidate assertion:

* ``-2/h^2`` instead of ``-3/h^2`` -- caught by the row, by constant preservation, and by
  the order study;
* the quadratic extrapolation ``u_ghost = (8/3) u_wall - 2 u_1 + (1/3) u_2``, diagonal
  ``-4/h^2`` -- **caught by symmetry alone**. Its wall truncation error is *better* than
  the correct stencil's and its global error is *smaller* (1.28e-05 against 2.46e-05), so
  it passes an order study at 1.94. A suite without a symmetry assertion prefers it;
* a stencil that wraps to the opposite wall -- caught by the row and by locality;
* a sign error in the wall source -- caught by constant preservation, and **only when the
  wall value is non-zero**. With ``u_wall = 0`` a sign error is bit-identical to correct,
  which is why every assertion here that can carry a non-zero wall value does.

Two things are deliberately *not* asserted. The wall row's pointwise truncation error is
``-u''(wall)/4`` and does **not** vanish with the cell size; that is correct behaviour,
not a defect, and the global order survives it because the operator's principal part is
second order (Svard, Nordstrom 2006: an m-th order principal part tolerates a boundary
closure m orders lower). Gustafsson 1975 does not cover it -- that theorem assumes the
boundary is at most *one* order down. And the order is measured in the maximum norm,
because a boundary defect is the one thing an averaged norm hides.

References:
    * Sanderse, Verstappen, Koren 2014 -- *Boundary treatment for fourth-order staggered
      mesh discretizations of the incompressible Navier-Stokes equations*,
      J. Comput. Phys. 257, 1472. Eqs. (87)-(92).
    * Verstappen, Veldman 2003 -- *Symmetry-preserving discretization of turbulent flow*,
      J. Comput. Phys. 187(1), 343. Eqs. (25), (37).
    * Svard, Nordstrom 2006 -- *On the order of accuracy for difference approximations of
      initial-boundary value problems*, J. Comput. Phys. 218(1), 333.
    * Shen 1991 -- *Hopf bifurcation of the unsteady regularized driven cavity flow*,
      J. Comput. Phys. 95(1), 228. The regularised lid ``16 x^2 (1-x)^2``.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.fields.field import Extrapolation
from opifex.fields.staggered import StaggeredGrid
from opifex.fields.staggered_diffusion import laplacian, wall_source, WallVelocity
from tests.fields.staggered_support import (
    BOX,
    CARRIED,
    face_coordinates,
    flatten,
    observed_orders,
    operator_matrix,
    random_field,
)


RESOLUTION = (8, 8)


def _regularised_lid(x: jax.Array) -> jax.Array:
    """Shen 1991's lid, whose value and slope both vanish at the corners."""
    return 16.0 * x**2 * (1.0 - x) ** 2


def _lid_profile(resolution) -> jax.Array:
    """The stored upper-wall profile for the x-component, narrowed off ``None``."""
    stored = _moving_lid(resolution, Extrapolation.ZERO).values[0][1]
    assert stored is not None, "axis 1 is tangential to component 0, so it carries a wall"
    return stored[1]


def _moving_lid(resolution, extrapolation) -> WallVelocity:
    """A cavity whose top wall slides with the regularised profile."""
    coordinates = face_coordinates(resolution, extrapolation)
    along_the_lid = coordinates[0][0][:, 0]
    walls = WallVelocity.zeros(resolution, extrapolation)
    return walls.set(normal=0, axis=1, upper=True, value=_regularised_lid(along_the_lid))


class TestTheWallStencil:
    """The rows themselves, which is the only assertion that catches three of four defects."""

    def test_the_tangential_row_is_minus_three_over_h_squared(self) -> None:
        cells = 4
        spacing = 1.0 / cells
        matrix = operator_matrix(laplacian, (cells, cells), Extrapolation.ZERO)
        # The x-component occupies the first (cells-1)*cells rows, laid out (x, y) in C
        # order, so stepping along y is the fastest index and one column of y is a
        # contiguous block of length `cells`.
        block = matrix[:cells, :cells] * spacing**2

        expected = np.array(
            [
                [-3.0, 1.0, 0.0, 0.0],
                [1.0, -2.0, 1.0, 0.0],
                [0.0, 1.0, -2.0, 1.0],
                [0.0, 0.0, 1.0, -3.0],
            ]
        )
        # The y-direction contributes the wall rows; the x-direction adds -2/h^2 to every
        # diagonal, so compare only the y part.
        np.testing.assert_allclose(block - np.diag(np.full(cells, -2.0)), expected, atol=1e-4)

    def test_it_does_not_couple_the_two_walls(self) -> None:
        # Defect (c): a wrapping stencil. Measured on the periodic operator, a spike on
        # the first row reaches the far wall at 1/h^2.
        matrix = operator_matrix(laplacian, (8, 8), Extrapolation.ZERO)
        block = matrix[:8, :8]

        assert float(abs(block[0, -1])) == 0.0
        assert float(abs(block[-1, 0])) == 0.0

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_is_exactly_symmetric(self, extrapolation: Extrapolation) -> None:
        # Defect (b): the quadratic extrapolation. Symmetry is the ONLY assertion that
        # rejects it -- its global error is smaller than the correct stencil's and it
        # passes an order study at 1.94.
        matrix = operator_matrix(laplacian, (6, 6), Extrapolation.ZERO)

        assert float(np.abs(matrix - matrix.T).max()) / float(np.abs(matrix).max()) < 1e-6

    def test_it_is_negative_definite_with_the_expected_spectrum(self) -> None:
        # Walls do not tighten the diffusive step limit: the wall spectrum approaches the
        # periodic 4*ndim/h^2 from below. Normal direction (wall on a face, n-1
        # unknowns) contributes -4 cos^2(pi/2n)/h^2; tangential contributes -4/h^2.
        cells = 8
        spacing = 1.0 / cells
        matrix = operator_matrix(laplacian, (cells, cells), Extrapolation.ZERO)
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
        expected = -(4.0 * np.cos(np.pi / (2 * cells)) ** 2 + 4.0) / spacing**2

        assert float(eigenvalues.max()) < 0.0, "must be negative definite, not semi-definite"
        assert float(eigenvalues.min()) == pytest.approx(expected, rel=1e-4)
        assert abs(float(eigenvalues.min())) < 8.0 / spacing**2


class TestTheWallVelocity:
    """The prescribed wall field, which must be an array and must reach the residual."""

    def test_a_constant_field_leaves_only_its_own_impermeable_walls(self) -> None:
        # Defect (d): the sign of the wall source. This is the ONLY assertion that
        # catches it, and only because the wall value is non-zero -- with u_wall = 0 a
        # sign error is bit-identical to correct.
        #
        # A non-zero constant is not an admissible velocity: it puts u.n = c on the wall
        # its own component is normal to, where no-penetration requires zero. So the
        # tangential directions must cancel exactly while the own-axis wall rows show
        # precisely -c/h^2. That is sharper than "the residual is zero", because a
        # tangential sign error would put -/+2c/h^2 on the other two rows.
        constant = 3.7
        cells = 8
        spacing = 1.0 / cells
        resolution = (cells, cells)
        shapes = StaggeredGrid.component_shapes(resolution, Extrapolation.ZERO)
        field = StaggeredGrid(
            tuple(jnp.full(shape, constant) for shape in shapes),
            BOX,
            Extrapolation.ZERO,
            resolution,
        )
        walls = WallVelocity.uniform(resolution, Extrapolation.ZERO, constant)

        residual = jax.tree.map(lambda a, b: a + b, laplacian(field), wall_source(field, walls))

        for normal, produced in enumerate(residual.components):
            expected = np.zeros(produced.shape)
            near = tuple(slice(None) if i != normal else 0 for i in range(produced.ndim))
            far = tuple(slice(None) if i != normal else -1 for i in range(produced.ndim))
            expected[near] = -constant / spacing**2
            expected[far] = -constant / spacing**2
            np.testing.assert_allclose(np.asarray(produced), expected, rtol=1e-5, atol=1e-2)

    def test_the_lid_profile_is_carried_pointwise_not_as_one_number(self) -> None:
        # The reason this is a field: both the standard energy test (Sanderse Eq. 166)
        # and the only defensible order-verification case (Shen 1991) need a lid that
        # varies along the wall. A scalar-per-side contract cannot express either.
        values = _lid_profile((16, 16))

        assert values.ndim == 1 and values.size == 15
        assert float(values.max()) == pytest.approx(1.0, rel=1e-2), "peaks at 1 mid-wall"
        # It vanishes *quadratically* at the corner, so the value at the first face falls
        # by four when the cells halve. A fixed threshold would record only what one
        # resolution happens to produce -- 0.0549 at 16 cells is the profile being right.
        corners = [_lid_profile((cells, cells))[0] for cells in (16, 32, 64)]
        for coarse, fine in itertools.pairwise(corners):
            assert coarse / fine == pytest.approx(4.0, rel=0.15), f"not quadratic: {corners}"

    def test_a_wall_at_rest_contributes_nothing(self) -> None:
        field = random_field(RESOLUTION, Extrapolation.ZERO, seed=2)
        walls = WallVelocity.zeros(RESOLUTION, Extrapolation.ZERO)

        assert float(jnp.max(jnp.abs(flatten(wall_source(field, walls))))) == 0.0

    def test_a_moving_wall_reaches_only_the_wall_rows(self) -> None:
        field = random_field((8, 8), Extrapolation.ZERO, seed=3)
        source = wall_source(field, _moving_lid((8, 8), Extrapolation.ZERO))
        carried = source.components[0]

        assert float(jnp.max(jnp.abs(carried[:, -1]))) > 0.0, "the moving wall row"
        assert float(jnp.max(jnp.abs(carried[:, :-1]))) == 0.0, "every other row is untouched"


class TestConvergence:
    """Global second order in the maximum norm, which is what survives the O(1) row."""

    def test_it_converges_at_second_order_in_the_maximum_norm(self) -> None:
        # A manufactured field with NON-ZERO curvature at the wall, so the O(1) wall row
        # is actually exercised. u = sin(2 pi x) sin(pi y) has u''(wall) = 0 along y and
        # would report a flattering order; this one does not.
        def exact(x: jax.Array, y: jax.Array) -> jax.Array:
            return jnp.sin(2.0 * jnp.pi * x) * y * (1.0 - y)

        def exact_laplacian(x: jax.Array, y: jax.Array) -> jax.Array:
            return -4.0 * jnp.pi**2 * jnp.sin(2.0 * jnp.pi * x) * y * (1.0 - y) - 2.0 * jnp.sin(
                2.0 * jnp.pi * x
            )

        # The operator is SOLVED, not applied. Applying it and reading the pointwise
        # residual measures the wall row's truncation error, which is O(1) by design --
        # measured flat at [0.515, 0.502, 0.500] over these three grids, which is the
        # theory being confirmed rather than a defect. Global order is a statement about
        # the solution, and it is the one that survives an O(1) boundary row.
        errors = []
        for cells in (16, 32, 64):
            x, y = face_coordinates((cells, cells), Extrapolation.ZERO)[0]
            shapes = StaggeredGrid.component_shapes((cells, cells), Extrapolation.ZERO)
            walls = WallVelocity.zeros((cells, cells), Extrapolation.ZERO)

            def negated(component, shapes=shapes, cells=cells, walls=walls):
                field = StaggeredGrid(
                    (component, jnp.zeros(shapes[1])), BOX, Extrapolation.ZERO, (cells, cells)
                )
                combined = jax.tree.map(
                    lambda a, b: a + b, laplacian(field), wall_source(field, walls)
                )
                return -combined.components[0]

            solution, _ = jax.scipy.sparse.linalg.cg(
                negated, -exact_laplacian(x, y), tol=1e-12, maxiter=50000
            )
            errors.append(float(jnp.max(jnp.abs(solution - exact(x, y)))))

        orders = observed_orders(errors)
        assert min(orders) > 1.85, f"not second order: {errors} -> {orders}"
        assert max(orders) < 2.15, f"suspiciously fast, check the fixture: {orders}"


class TestTransforms:
    """jit, vmap and reverse-mode grad -- including grad through the wall field.

    The wall velocity is an ordinary array in the residual path rather than a value in
    static metadata, and that is the point: carried as static data it forces one retrace
    per distinct wall value and reverse mode cannot reach it at all.
    """

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_traces_once_under_jit(self, extrapolation: Extrapolation) -> None:
        field = random_field(RESOLUTION, extrapolation, seed=4)
        traces = {"count": 0}

        def counted(grid: StaggeredGrid) -> StaggeredGrid:
            traces["count"] += 1
            return laplacian(grid)

        compiled = jax.jit(counted)
        for _ in range(4):
            compiled(field)

        assert traces["count"] == 1

    def test_a_changing_wall_velocity_does_not_retrace(self) -> None:
        # The measured failure mode of the static-metadata architecture: three traces for
        # the wall values [0, 1, 2, 1]. As an array leaf, jit specialises on shape only.
        field = random_field(RESOLUTION, Extrapolation.ZERO, seed=5)
        traces = {"count": 0}

        def counted(grid: StaggeredGrid, walls: WallVelocity) -> StaggeredGrid:
            traces["count"] += 1
            return wall_source(grid, walls)

        compiled = jax.jit(counted)
        for value in (0.0, 1.0, 2.0, 1.0):
            compiled(field, WallVelocity.uniform(RESOLUTION, Extrapolation.ZERO, value))

        assert traces["count"] == 1

    def test_it_differentiates_with_respect_to_the_wall_velocity(self) -> None:
        field = random_field(RESOLUTION, Extrapolation.ZERO, seed=6)

        def loss(walls: WallVelocity) -> jax.Array:
            return jnp.sum(flatten(wall_source(field, walls)) ** 2)

        gradient = jax.grad(loss)(WallVelocity.uniform(RESOLUTION, Extrapolation.ZERO, 0.5))
        leaves = [leaf for leaf in jax.tree.leaves(gradient) if leaf is not None]

        assert leaves, "the wall velocity must be a differentiable leaf, not static data"
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
        assert max(float(jnp.max(jnp.abs(leaf))) for leaf in leaves) > 0.0

    def test_it_maps_over_a_batch_of_wall_velocities(self) -> None:
        field = random_field(RESOLUTION, Extrapolation.ZERO, seed=7)
        batch = jax.tree.map(
            lambda leaf: jnp.stack([leaf, 2.0 * leaf]),
            WallVelocity.uniform(RESOLUTION, Extrapolation.ZERO, 1.0),
        )

        mapped = jax.vmap(lambda walls: wall_source(field, walls).components[0])(batch)

        assert mapped.shape[0] == 2
        assert float(jnp.max(jnp.abs(mapped[1] - 2.0 * mapped[0]))) < 1e-5


class TestEnergy:
    """``u.D u`` is dissipation alone only while the wall is at rest.

    With a moving wall Sanderse Eq. (92) says the quadratic form picks up a boundary-work
    term. Asserting pure dissipation would encode a claim that is false for any moving
    wall, and would have to be rewritten exactly when a rewrite is least trustworthy.
    """

    def test_a_wall_at_rest_only_dissipates(self) -> None:
        field = random_field((8, 8), Extrapolation.ZERO, seed=8)
        walls = WallVelocity.zeros((8, 8), Extrapolation.ZERO)

        rate = float(
            flatten(field)
            @ flatten(jax.tree.map(lambda a, b: a + b, laplacian(field), wall_source(field, walls)))
        )

        assert rate < 0.0

    def test_a_moving_wall_adds_boundary_work(self) -> None:
        field = random_field((8, 8), Extrapolation.ZERO, seed=8)
        at_rest = WallVelocity.zeros((8, 8), Extrapolation.ZERO)
        moving = _moving_lid((8, 8), Extrapolation.ZERO)

        def rate(walls: WallVelocity) -> float:
            return float(
                flatten(field)
                @ flatten(
                    jax.tree.map(lambda a, b: a + b, laplacian(field), wall_source(field, walls))
                )
            )

        work = rate(moving) - rate(at_rest)

        # The dissipation is identical -- it depends only on the field -- so the whole
        # difference is the boundary-work term, which is linear in the wall velocity.
        assert work != 0.0
        doubled = jax.tree.map(lambda leaf: 2.0 * leaf, moving)
        assert rate(doubled) - rate(at_rest) == pytest.approx(2.0 * work, rel=1e-4)


class TestBoundariesNotYetCarried:
    """What the operator still refuses."""

    def test_a_boundary_outside_the_contract_is_refused(self) -> None:
        field = StaggeredGrid.zeros(RESOLUTION, BOX, Extrapolation.NEUMANN)

        with pytest.raises(ValueError, match="does not carry"):
            laplacian(field)
