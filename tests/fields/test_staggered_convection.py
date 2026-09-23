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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.fields.field import Extrapolation
from opifex.fields.staggered import divergence, StaggeredGrid
from opifex.fields.staggered_convection import convect
from opifex.fields.staggered_pressure import project
from tests.fields.staggered_support import (
    asymmetry,
    BOX,
    CARRIED,
    flatten,
    FLOAT32_EPS,
    observed_orders,
    operator_matrix,
    random_field,
    stream_field,
    STREAMS,
)


RESOLUTION = (8, 8)


def _relative_divergence(projected: StaggeredGrid, before: StaggeredGrid) -> float:
    """Residual divergence as a fraction of what the field came in with."""
    return float(jnp.linalg.norm(divergence(projected)) / jnp.linalg.norm(divergence(before)))


class TestSkewSymmetry:
    """No energy production -- but only about a divergence-free transporting velocity."""

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_is_skew_about_a_divergence_free_velocity(
        self, extrapolation: Extrapolation
    ) -> None:
        raw = random_field(RESOLUTION, seed=0, extrapolation=extrapolation)
        solenoidal, _ = project(raw)
        # The precondition belongs in the units the projection guarantees: it leaves a
        # fixed fraction of the incoming divergence, not a fixed absolute norm, and the
        # incoming norm here is ~123 because differencing random noise divides by dx.
        assert _relative_divergence(solenoidal, raw) < 1e-6

        off_diagonal, diagonal = asymmetry(
            operator_matrix(lambda w: convect(w, solenoidal), RESOLUTION, extrapolation)
        )

        assert off_diagonal <= 1e-12, "a nonzero off-diagonal part is a stencil error"
        assert diagonal <= 1e-5

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_is_not_skew_about_a_field_with_divergence(
        self, extrapolation: Extrapolation
    ) -> None:
        # The negative control. Without it, an operator that is skew for the trivial
        # reason of being zero, or a test fixture that is accidentally solenoidal, would
        # pass the assertion above and prove nothing.
        divergent = random_field(RESOLUTION, seed=1, extrapolation=extrapolation)
        assert float(jnp.linalg.norm(divergence(divergent))) > 1.0

        off_diagonal, diagonal = asymmetry(
            operator_matrix(lambda w: convect(w, divergent), RESOLUTION, extrapolation)
        )

        # The control has to be distinguishable from a broken stencil, not merely large:
        # divergence in the carrier shows up on the diagonal alone.
        assert off_diagonal <= 1e-12
        assert diagonal > 0.1

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_the_energy_it_produces_is_zero(self, extrapolation: Extrapolation) -> None:
        # The same statement as a quadratic form, which is what actually matters to a
        # simulation: <w, C w> is the rate energy enters through the convective term.
        raw = random_field(RESOLUTION, seed=2, extrapolation=extrapolation)
        solenoidal, _ = project(raw)
        assert _relative_divergence(solenoidal, raw) < 1e-6
        transported = random_field(RESOLUTION, seed=3, extrapolation=extrapolation)

        produced = flatten(transported) @ flatten(convect(transported, solenoidal))
        scale = jnp.sum(jnp.abs(flatten(transported) * flatten(convect(transported, solenoidal))))

        assert abs(float(produced) / float(scale)) <= 1e-5


class TestTransforms:
    """jit, vmap and reverse-mode grad, all three, on both branches.

    The wall branch is not the periodic one with different numbers in it: it slices and
    concatenates where the periodic branch rolls, and produces arrays of a different
    shape. Covering only the periodic path would leave that code untraced.
    """

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_traces_once_under_jit(self, extrapolation: Extrapolation) -> None:
        velocity = random_field(RESOLUTION, seed=4, extrapolation=extrapolation)
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
        velocity = random_field(RESOLUTION, seed=5, extrapolation=extrapolation)

        def loss(components: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(components, BOX, extrapolation, RESOLUTION)
            return jnp.sum(flatten(convect(field, field)) ** 2)

        grads = jax.grad(loss)(velocity.components)

        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    @pytest.mark.parametrize("extrapolation", CARRIED)
    def test_it_maps_over_a_batch(self, extrapolation: Extrapolation) -> None:
        velocity = random_field(RESOLUTION, seed=6, extrapolation=extrapolation)
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
        field, _ = stream_field(STREAMS["sin(pi x) sin(pi y)"], (cells, cells), Extrapolation.ZERO)
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
            field, exact = stream_field(STREAMS[stream], (cells, cells), Extrapolation.ZERO)
            residual = np.asarray(convect(field, field).components[0] - exact[0])
            # The x-component's tangential walls are the first and last columns in y.
            walls.append(max(np.abs(residual[:, 0]).max(), np.abs(residual[:, -1]).max()))
            interiors.append(np.abs(residual[:, 1:-1]).max())

        for label, errors in (("wall", walls), ("interior", interiors)):
            assert min(observed_orders(errors)) > 1.85, f"{label} is not 2nd order: {errors}"


class TestBoundariesNotYetCarried:
    """What the operator refuses, and why it refuses rather than approximating."""

    def test_a_boundary_outside_the_contract_is_refused(self) -> None:
        field = StaggeredGrid.zeros(RESOLUTION, BOX, Extrapolation.NEUMANN)

        with pytest.raises(ValueError, match="does not carry"):
            convect(field, field)
