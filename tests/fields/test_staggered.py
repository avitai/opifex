"""The staggered (Arakawa-C / MAC) layout, and the identities it is chosen for.

A collocated grid cannot hold these: its ``divergence`` and ``gradient`` are two-point
central differences, so their composition reaches ``i +/- 2``, it annihilates the three
checkerboard modes as well as the constant, and under a zero-gradient boundary ``edge``
padding puts the first cell into its own difference and creates a diagonal entry, which no
skew operator can have. Staggering removes the boundary stencil rather than repairing it:
a boundary-normal face velocity is not a degree of freedom, so the gradient never needs a
ghost pressure.

What that buys is checked here directly, as matrices, because these are statements about
the operators rather than about any particular field:

* ``divergence`` is minus the adjoint of ``gradient``, exactly, in the plain Euclidean
  inner product, under every boundary -- the property that makes the pressure Poisson
  operator symmetric and the projection an exact discrete Helmholtz decomposition;
* that operator is negative semi-definite with a null space of exactly the constants,
  where the collocated one has four null modes and, under a zero-gradient boundary,
  positive eigenvalues as well.

References:
    * Harlow, Welch 1965 -- *Numerical calculation of time-dependent viscous incompressible
      flow of fluid with free surface*, Phys. Fluids 8(12), 2182.
    * Verstappen, Veldman 2003 -- *Symmetry-preserving discretization of turbulent flow*,
      J. Comput. Phys. 187(1), 343.
    * Sanderse 2013 -- *Energy-conserving Runge-Kutta methods for the incompressible
      Navier-Stokes equations*, J. Comput. Phys. 233, 100.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import (
    CARRIED_BOUNDARIES,
    divergence,
    face_count,
    gradient,
    require_boundary,
    StaggeredGrid,
)
from opifex.fields.staggered_convection import convect
from opifex.fields.staggered_diffusion import laplacian
from opifex.fields.staggered_pressure import project


BOX = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))

# Every operation the contract names, so a row of the table can be checked against the
# operator it governs rather than against the table itself. A new row with no entry here
# fails ``test_every_operation_is_reachable``, which is what keeps the two in step.
OPERATIONS = {
    "the grid operators": divergence,
    "the pressure projection": project,
    "convection": lambda field: convect(field, field),
    "diffusion": laplacian,
}
BOUNDARIES = [Extrapolation.PERIODIC, Extrapolation.ZERO, Extrapolation.NEUMANN]


def _operator_matrix(resolution, extrapolation, forward):
    """The dense matrix of a linear operator, one column per basis vector."""
    probe = forward(jnp.zeros(resolution), extrapolation)
    columns = []
    for index in range(int(np.prod(resolution))):
        basis = jnp.zeros(int(np.prod(resolution))).at[index].set(1.0)
        columns.append(np.asarray(forward(basis.reshape(resolution), extrapolation)))
    return np.stack([column.ravel() for column in columns], axis=1), probe


def _gradient_matrix(resolution, extrapolation):
    def forward(values, boundary):
        field = StaggeredGrid.zeros(resolution, BOX, boundary)
        return jnp.concatenate(
            [component.ravel() for component in gradient(values, field).components]
        )

    return _operator_matrix(resolution, extrapolation, forward)[0]


def _divergence_matrix(resolution, extrapolation):
    sizes = [
        int(np.prod(shape)) for shape in StaggeredGrid.component_shapes(resolution, extrapolation)
    ]
    total = sum(sizes)
    columns = []
    for index in range(total):
        flat = jnp.zeros(total).at[index].set(1.0)
        pieces, offset = [], 0
        for shape, size in zip(
            StaggeredGrid.component_shapes(resolution, extrapolation), sizes, strict=True
        ):
            pieces.append(flat[offset : offset + size].reshape(shape))
            offset += size
        field = StaggeredGrid(tuple(pieces), BOX, extrapolation, resolution)
        columns.append(np.asarray(divergence(field)).ravel())
    return np.stack(columns, axis=1)


class TestFaceCounts:
    """How many velocity faces a boundary leaves along an axis."""

    @pytest.mark.parametrize(
        ("extrapolation", "expected"),
        [
            # Periodic: the last face is the first, so n faces carry n cells.
            (Extrapolation.PERIODIC, 8),
            # Dirichlet: both wall faces are prescribed, so they are not unknowns.
            (Extrapolation.ZERO, 7),
            # Zero-gradient: neither outer face is determined, so both are stored.
            (Extrapolation.NEUMANN, 9),
        ],
    )
    def test_the_boundary_sets_the_number_of_unknown_faces(
        self, extrapolation: Extrapolation, expected: int
    ) -> None:
        assert face_count(8, extrapolation) == expected


class TestDiscreteAdjointness:
    """``divergence`` is minus the adjoint of ``gradient``, exactly."""

    @pytest.mark.parametrize("extrapolation", BOUNDARIES)
    def test_divergence_is_minus_the_transpose_of_gradient(
        self, extrapolation: Extrapolation
    ) -> None:
        resolution = (6, 6)

        gradient_matrix = _gradient_matrix(resolution, extrapolation)
        divergence_matrix = _divergence_matrix(resolution, extrapolation)

        mismatch = np.abs(divergence_matrix + gradient_matrix.T).max()
        assert mismatch <= 1e-12 * np.abs(gradient_matrix).max()

    @pytest.mark.parametrize("extrapolation", BOUNDARIES)
    def test_the_poisson_operator_is_symmetric_negative_semi_definite(
        self, extrapolation: Extrapolation
    ) -> None:
        resolution = (6, 6)
        gradient_matrix = _gradient_matrix(resolution, extrapolation)
        poisson = _divergence_matrix(resolution, extrapolation) @ gradient_matrix

        scale = np.abs(poisson).max()
        assert np.abs(poisson - poisson.T).max() <= 1e-12 * scale
        assert np.linalg.eigvalsh((poisson + poisson.T) / 2).max() <= 1e-10 * scale

    @pytest.mark.parametrize("extrapolation", BOUNDARIES)
    def test_at_most_the_constants_are_annihilated(self, extrapolation: Extrapolation) -> None:
        # The collocated operator annihilates three checkerboard modes as well, which is
        # the odd-even decoupling staggering exists to avoid. What is left is whether the
        # constant is pinned, and that follows from the derived pressure boundary: a closed
        # box prescribes the velocity everywhere, leaving the pressure pure Neumann and
        # undetermined up to a constant, while an open boundary prescribes the pressure and
        # pins it. Whether the total divergence may float is the compatibility question a
        # solver has to answer before it solves anything.
        resolution = (6, 6)
        poisson = _divergence_matrix(resolution, extrapolation) @ _gradient_matrix(
            resolution, extrapolation
        )

        nullity = poisson.shape[0] - np.linalg.matrix_rank(poisson)

        closed = extrapolation in {Extrapolation.PERIODIC, Extrapolation.ZERO}
        assert nullity == (1 if closed else 0)


class TestTransforms:
    """The operators must carry jit, vmap and grad, all three."""

    def test_they_trace_under_jit_without_retracing(self) -> None:
        resolution = (8, 8)
        field = StaggeredGrid.zeros(resolution, BOX, Extrapolation.PERIODIC)
        traces = {"count": 0}

        def counted(grid: StaggeredGrid) -> jax.Array:
            traces["count"] += 1
            return divergence(grid)

        compiled = jax.jit(counted)
        for _ in range(4):
            compiled(field)

        assert traces["count"] == 1

    def test_the_divergence_differentiates_in_reverse_mode(self) -> None:
        resolution = (8, 8)
        field = StaggeredGrid.zeros(resolution, BOX, Extrapolation.NEUMANN)
        components = tuple(
            jax.random.normal(jax.random.key(axis), component.shape)
            for axis, component in enumerate(field.components)
        )

        def loss(values: tuple[jax.Array, ...]) -> jax.Array:
            grid = StaggeredGrid(values, BOX, Extrapolation.NEUMANN, resolution)
            return jnp.sum(divergence(grid) ** 2)

        grads = jax.grad(loss)(components)

        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    def test_the_divergence_maps_over_a_batch(self) -> None:
        resolution = (8, 8)
        field = StaggeredGrid.zeros(resolution, BOX, Extrapolation.PERIODIC)
        batch = tuple(jnp.stack([c, c]) for c in field.components)

        def project(values: tuple[jax.Array, ...]) -> jax.Array:
            return divergence(StaggeredGrid(values, BOX, Extrapolation.PERIODIC, resolution))

        mapped = jax.vmap(project)(batch)

        assert mapped.shape == (2, *resolution)


class TestTheBoundaryContract:
    """One owner for which boundaries each part of the layer carries.

    The parts differ -- the grid operators take all three, the projection and convection
    take two, and diffusion takes one -- so a caller assembling a simulation from them
    would otherwise meet the limit one operation at a time, at whichever raised first.
    """

    def test_every_operation_declares_what_it_carries(self) -> None:
        assert set(CARRIED_BOUNDARIES) == {
            "the grid operators",
            "the pressure projection",
            "convection",
            "diffusion",
        }
        assert all(boundaries for boundaries in CARRIED_BOUNDARIES.values())

    def test_every_operation_is_reachable(self) -> None:
        # Without this the table below silently stops covering a new row.
        assert set(OPERATIONS) == set(CARRIED_BOUNDARIES)

    def test_a_carried_boundary_is_accepted(self) -> None:
        require_boundary(Extrapolation.PERIODIC, "convection")

    def test_a_refusal_names_the_whole_layer_not_just_the_caller(self) -> None:
        # Every part of the layer now carries a wall, so the only boundary left outside
        # the contract is the zero-gradient one.
        with pytest.raises(ValueError, match="does not carry") as refusal:
            require_boundary(Extrapolation.NEUMANN, "diffusion")

        message = str(refusal.value)
        assert "diffusion" in message
        # The point of the single owner: one refusal teaches the whole picture.
        for operation in CARRIED_BOUNDARIES:
            assert operation in message
        # And says what is actually missing, so the reader knows it is not padding.
        assert "diagonalise" in message

    @pytest.mark.parametrize(
        ("operation", "extrapolation"),
        [
            ("the pressure projection", Extrapolation.ZERO),
            ("the grid operators", Extrapolation.NEUMANN),
            ("convection", Extrapolation.ZERO),
            ("diffusion", Extrapolation.ZERO),
        ],
    )
    def test_the_declaration_matches_what_the_code_does(
        self, operation: str, extrapolation: Extrapolation
    ) -> None:
        # Run the operation itself, not ``require_boundary``: asking the guard whether the
        # table permits something only re-reads the table, and would still pass if the
        # operator it guards had never been taught the boundary at all.
        field = StaggeredGrid.zeros((8, 8), BOX, extrapolation)

        OPERATIONS[operation](field)
