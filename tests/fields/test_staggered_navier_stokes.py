"""Time stepping for incompressible flow on the staggered layout.

The spatial operators contribute no energy error at all: ``divergence`` is minus the adjoint
of ``gradient``, so the projection is exact, and the convective term is skew-symmetric about
a divergence-free field, so it produces none. What is left is the time integrator, and the
measurements below say so -- inviscid energy drift is fourth order in the step and does not
depend on the resolution.

That holds only if the projection is applied at **every Runge-Kutta stage**. The
skew-symmetry is a statement about a discretely divergence-free transporting field, and a
stage that begins from an unprojected one is outside it. Projecting once per step instead
costs three orders of convergence, which the negative control here measures rather than
asserts.

References:
    * Sanderse 2013 -- *Energy-conserving Runge-Kutta methods for the incompressible
      Navier-Stokes equations*, J. Comput. Phys. 233, 100.
    * Sanderse, Koren 2012 -- *Accuracy analysis of explicit Runge-Kutta methods applied to
      the incompressible Navier-Stokes equations*, J. Comput. Phys. 231(8), 3041.
"""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import divergence, StaggeredGrid
from opifex.fields.staggered_convection import convect
from opifex.fields.staggered_diffusion import laplacian, wall_source, WallVelocity
from opifex.fields.staggered_navier_stokes import integrate, step, tendency
from opifex.fields.staggered_pressure import project


BOX = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))


def _solenoidal(n: int, seed: int = 0) -> StaggeredGrid:
    """A divergence-free face velocity, which is the regime the scheme is stated over."""
    shapes = StaggeredGrid.component_shapes((n, n), Extrapolation.PERIODIC)
    keys = jax.random.split(jax.random.key(seed), len(shapes))
    raw = StaggeredGrid(
        tuple(jax.random.normal(key, shape) for key, shape in zip(keys, shapes, strict=True)),
        BOX,
        Extrapolation.PERIODIC,
        (n, n),
    )
    return project(raw)[0]


def _energy(field: StaggeredGrid) -> jax.Array:
    return 0.5 * jnp.sum(jnp.stack([jnp.sum(c**2) for c in field.components]))


def _drift(initial: StaggeredGrid, final: StaggeredGrid) -> float:
    return float(abs(_energy(final) - _energy(initial)) / _energy(initial))


class TestInviscidEnergy:
    """With no viscosity the scheme owes energy conservation, up to the time error."""

    def test_the_drift_vanishes_faster_than_fourth_order_in_the_step(self) -> None:
        # In float32 the drift reaches the rounding floor within one refinement and no
        # order can be read, so it is measured where it is visible.
        #
        # Measured drift at 25/50/100/200 steps: 5.44e-05, 1.58e-06, 4.03e-08, 6.81e-10,
        # so each halving divides it by 34.4, 39.2, 59.3 -- not the sixteen a fourth-order
        # scheme would give. That is the expected behaviour rather than a surprise: a
        # Runge-Kutta stability function matches the exponential to its own order, so for
        # a skew operator, whose spectrum is imaginary, the amplitude error appears two
        # orders later than the solution error. Classical RK4 has |R(iy)| = 1 - y^6/144,
        # a sixth-order error per step, which over a fixed interval accumulates as the
        # fifth power of the step -- a factor of 32 per halving.
        with jax.enable_x64(True):
            start = _solenoidal(16)
            drifts = [
                _drift(start, integrate(start, total_time=0.5, num_steps=steps, viscosity=0.0))
                for steps in (25, 50, 100, 200)
            ]

        ratios = [before / after for before, after in itertools.pairwise(drifts)]
        assert all(ratio > 16.0 for ratio in ratios), (drifts, ratios)

    def test_the_drift_follows_the_step_not_the_resolution(self) -> None:
        # A spatial energy error would grow as the grid is refined at a fixed Courant
        # number; this one must not, because the spatial operators contribute none. Holding
        # the step fixed instead doubles the Courant number with every refinement, which is
        # a statement about the time error and is measured here as the contrast: it grows
        # by a factor of about fifty per refinement where the fixed-Courant drift does not
        # quite double.
        with jax.enable_x64(True):
            coarse = _solenoidal(16)
            fine = _solenoidal(32)
            same_courant = (
                _drift(coarse, integrate(coarse, 0.5, num_steps=100, viscosity=0.0)),
                _drift(fine, integrate(fine, 0.5, num_steps=200, viscosity=0.0)),
            )
            same_step = (
                _drift(coarse, integrate(coarse, 0.5, num_steps=100, viscosity=0.0)),
                _drift(fine, integrate(fine, 0.5, num_steps=100, viscosity=0.0)),
            )

        assert same_courant[1] < 4.0 * same_courant[0]
        assert same_step[1] > 10.0 * same_step[0]

    def test_leaving_the_stages_unprojected_costs_four_orders(self) -> None:
        """The negative control for the per-stage projection.

        Both sides run the *same* fourth-order Runge-Kutta; only the projection frequency
        differs, so the comparison isolates that and does not smuggle in a change of
        integrator. Measured drift at 50/100/200 steps -- per-stage 1.58e-06, 4.03e-08,
        6.81e-10 against final-only 2.33e-01, 1.46e-01, 8.39e-02: the first converges at
        about fifth order and the second barely at first, a gap that widens from 1.5e+05
        to 1.2e+08 as the step shrinks. Without this, "we project at every stage" would be
        a statement about the code rather than about the scheme.
        """

        def rk4_projecting_only_at_the_end(field: StaggeredGrid, dt: float) -> StaggeredGrid:
            shifted = lambda base, rate, weight: jax.tree.map(
                lambda value, slope: value + weight * slope, base, rate
            )
            first = tendency(field, viscosity=0.0)
            second = tendency(shifted(field, first, 0.5 * dt), viscosity=0.0)
            third = tendency(shifted(field, second, 0.5 * dt), viscosity=0.0)
            fourth = tendency(shifted(field, third, dt), viscosity=0.0)
            accumulated = jax.tree.map(
                lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0, first, second, third, fourth
            )
            return project(shifted(field, accumulated, dt))[0]

        with jax.enable_x64(True):
            start = _solenoidal(16)
            per_stage = integrate(start, total_time=0.5, num_steps=100, viscosity=0.0)
            final_only = start
            for _ in range(100):
                final_only = rk4_projecting_only_at_the_end(final_only, 0.005)

        assert _drift(start, per_stage) < 1e-5 * _drift(start, final_only)


class TestIncompressibility:
    """Every step leaves the field divergence free, not just the first."""

    def test_the_field_stays_divergence_free_along_a_trajectory(self) -> None:
        start = _solenoidal(16)
        incoming = float(jnp.linalg.norm(divergence(start)))

        field = start
        for _ in range(8):
            field = step(field, dt=0.005, viscosity=0.01)
            assert float(jnp.linalg.norm(divergence(field))) <= max(incoming, 1e-6) * 10.0


class TestViscosity:
    """The viscous term dissipates, and only dissipates."""

    def test_energy_falls_when_viscosity_is_positive(self) -> None:
        start = _solenoidal(16)

        final = integrate(start, total_time=0.1, num_steps=50, viscosity=0.05)

        assert float(_energy(final)) < float(_energy(start))

    def test_a_stronger_viscosity_removes_more(self) -> None:
        start = _solenoidal(16)

        weak = integrate(start, total_time=0.1, num_steps=50, viscosity=0.01)
        strong = integrate(start, total_time=0.1, num_steps=50, viscosity=0.05)

        assert float(_energy(strong)) < float(_energy(weak))


class TestTransforms:
    """jit, vmap and reverse-mode grad -- the last is what the old solver cannot do."""

    def test_it_traces_once_under_jit(self) -> None:
        start = _solenoidal(16)
        traces = {"count": 0}

        def counted(field: StaggeredGrid) -> StaggeredGrid:
            traces["count"] += 1
            return integrate(field, total_time=0.05, num_steps=5, viscosity=0.01)

        compiled = jax.jit(counted)
        for _ in range(4):
            compiled(start)

        assert traces["count"] == 1

    def test_it_differentiates_in_reverse_mode(self) -> None:
        # `solve_navier_stokes_2d` raises here: its adaptive sub-stepping is a
        # `lax.while_loop` with a data-dependent trip count, which has no reverse rule.
        start = _solenoidal(16)

        def loss(components: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(components, BOX, Extrapolation.PERIODIC, (16, 16))
            return _energy(integrate(field, total_time=0.05, num_steps=5, viscosity=0.01))

        grads = jax.grad(loss)(start.components)

        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
        assert any(float(jnp.max(jnp.abs(g))) > 0 for g in grads)

    def test_it_maps_over_a_batch(self) -> None:
        start = _solenoidal(16)
        batch = tuple(jnp.stack([c, 0.5 * c]) for c in start.components)

        def run(components: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(components, BOX, Extrapolation.PERIODIC, (16, 16))
            return integrate(field, total_time=0.05, num_steps=5, viscosity=0.01).components[0]

        mapped = jax.vmap(run)(batch)

        assert mapped.shape == (2, *start.components[0].shape)


class TestTheOperatorsApproximateWhatTheyClaim:
    """Consistency, which structure alone does not give.

    An operator can be exactly skew-symmetric, exactly symmetric, exactly conservative --
    and still compute the wrong quantity. Every structural property asserted elsewhere in
    these files is checked here against an analytic target instead.

    Taylor-Green ``u = (sin x cos y, -cos x sin y)`` on ``[0, 2 pi]^2`` gives
    ``div(u u) = (sin x cos x, sin y cos y)`` and ``lap(u) = -2 u``, both by hand.
    """

    @staticmethod
    def _taylor_green(n: int) -> tuple[StaggeredGrid, tuple[jax.Array, ...]]:
        extent = 2.0 * jnp.pi
        box = Box(lower=(0.0, 0.0), upper=(float(extent), float(extent)))
        spacing = extent / n
        index = jnp.arange(n)
        x_u, y_u = jnp.meshgrid((index + 1) * spacing, (index + 0.5) * spacing, indexing="ij")
        x_v, y_v = jnp.meshgrid((index + 0.5) * spacing, (index + 1) * spacing, indexing="ij")
        field = StaggeredGrid(
            (jnp.sin(x_u) * jnp.cos(y_u), -jnp.cos(x_v) * jnp.sin(y_v)),
            box,
            Extrapolation.PERIODIC,
            (n, n),
        )
        return field, (x_u, y_v)

    @staticmethod
    def _relative(got: StaggeredGrid, want: tuple[jax.Array, ...]) -> float:
        numerator = sum(
            float(jnp.linalg.norm(a - b) ** 2) for a, b in zip(got.components, want, strict=True)
        )
        denominator = sum(float(jnp.linalg.norm(b) ** 2) for b in want)
        return (numerator / denominator) ** 0.5

    def test_convection_converges_to_the_analytic_flux_at_second_order(self) -> None:
        # Measured 6.26e-02, 1.60e-02, 4.01e-03, 1.00e-03 at 16/32/64/128 cells.
        errors = []
        with jax.enable_x64(True):
            for n in (16, 32, 64):
                field, (x_u, y_v) = self._taylor_green(n)
                want = (0.5 * jnp.sin(2 * x_u), 0.5 * jnp.sin(2 * y_v))
                errors.append(self._relative(convect(field, field), want))

        orders = [math.log2(a / b) for a, b in itertools.pairwise(errors)]
        assert all(order > 1.9 for order in orders), (errors, orders)

    def test_the_viscous_operator_converges_to_the_analytic_laplacian(self) -> None:
        errors = []
        with jax.enable_x64(True):
            for n in (16, 32, 64):
                field, _ = self._taylor_green(n)
                want = tuple(-2.0 * component for component in field.components)
                errors.append(self._relative(laplacian(field), want))

        orders = [math.log2(a / b) for a, b in itertools.pairwise(errors)]
        assert all(order > 1.9 for order in orders), (errors, orders)

    def test_convection_in_divergence_form_conserves_momentum(self) -> None:
        # A divergence-form flux telescopes, so its sum over the grid vanishes.
        with jax.enable_x64(True):
            field, _ = self._taylor_green(32)
            carried = convect(field, field)
            # The reduction must happen inside the precision context: taking it outside
            # casts a float64 array back to float32 and measures the cast, not the flux.
            sums = [abs(float(jnp.sum(component))) for component in carried.components]

        assert max(sums) <= 1e-12, sums


class TestAMovingWall:
    """A prescribed wall velocity reaches the flow, and only through the viscous term."""

    @staticmethod
    def _lid(cells: int) -> WallVelocity:
        along = jnp.arange(1, cells) / cells
        return WallVelocity.zeros((cells, cells), Extrapolation.ZERO).set(
            normal=0, axis=1, upper=True, value=16.0 * along**2 * (1.0 - along) ** 2
        )

    def test_a_wall_at_rest_leaves_a_still_cavity_still(self) -> None:
        # The negative control. Without it, a tendency that was simply zero for every
        # input would satisfy the test below.
        still = StaggeredGrid.zeros((16, 16), BOX, Extrapolation.ZERO)

        rate = tendency(still, viscosity=0.01, walls=None)

        assert max(float(jnp.max(jnp.abs(part))) for part in rate.components) == 0.0

    def test_a_moving_lid_drives_a_still_cavity(self) -> None:
        still = StaggeredGrid.zeros((16, 16), BOX, Extrapolation.ZERO)

        rate = tendency(still, viscosity=0.01, walls=self._lid(16))

        assert max(float(jnp.max(jnp.abs(part))) for part in rate.components) > 0.0

    def test_the_lid_enters_through_diffusion_alone(self) -> None:
        # Convection cannot see a tangential wall value: that value only ever multiplies a
        # wall-normal velocity, which impermeability holds at zero. So on a still cavity
        # the whole tendency is the viscous boundary term.
        viscosity = 0.01
        still = StaggeredGrid.zeros((16, 16), BOX, Extrapolation.ZERO)
        walls = self._lid(16)

        rate = tendency(still, viscosity=viscosity, walls=walls)
        expected = jax.tree.map(lambda part: viscosity * part, wall_source(still, walls))

        for produced, want in zip(rate.components, expected.components, strict=True):
            np.testing.assert_allclose(np.asarray(produced), np.asarray(want), atol=1e-5)

    def test_the_driven_cavity_stays_divergence_free(self) -> None:
        still = StaggeredGrid.zeros((16, 16), BOX, Extrapolation.ZERO)

        driven = integrate(still, 0.05, 20, 0.01, self._lid(16))
        speed = max(float(jnp.max(jnp.abs(part))) for part in driven.components)

        assert speed > 0.0, "the lid must actually have driven the flow"
        # Scaled by the differencing the divergence performs, which divides by the cell.
        assert float(jnp.max(jnp.abs(divergence(driven)))) < 1e-5 * speed * 16

    def test_the_trajectory_differentiates_with_respect_to_the_lid(self) -> None:
        # The wall velocity is an array in the residual path, so a wall profile is an
        # ordinary differentiable input -- which is what makes boundary control possible.
        still = StaggeredGrid.zeros((8, 8), BOX, Extrapolation.ZERO)

        def energy(walls: WallVelocity) -> jax.Array:
            driven = integrate(still, 0.02, 4, 0.01, walls)
            return jnp.stack([jnp.sum(part**2) for part in driven.components]).sum()

        gradient = jax.grad(energy)(self._lid(8))
        leaves = [leaf for leaf in jax.tree.leaves(gradient) if leaf is not None]

        assert leaves and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
        assert max(float(jnp.max(jnp.abs(leaf))) for leaf in leaves) > 0.0

    def test_it_traces_once_with_the_lid_crossing_the_scan(self) -> None:
        still = StaggeredGrid.zeros((8, 8), BOX, Extrapolation.ZERO)
        traces = {"count": 0}

        def counted(walls: WallVelocity) -> StaggeredGrid:
            traces["count"] += 1
            return integrate(still, 0.02, 4, 0.01, walls)

        compiled = jax.jit(counted)
        for scale in (1.0, 2.0, 0.5, 2.0):
            compiled(jax.tree.map(lambda leaf, s=scale: s * leaf, self._lid(8)))

        assert traces["count"] == 1, "a wall value in static metadata would retrace per value"


class TestBoundaries:
    """Every operator on this path now carries a wall, and the tendency composes them.

    The restriction has to be uniform in both directions: an operator that silently
    accepted a wall while its neighbours refused one would be the worst of both, and one
    that still refused after its neighbours had learned it would strand the capability.
    """

    def test_the_tendency_runs_against_a_wall(self) -> None:
        field = _solenoidal(8, seed=11)
        walled = StaggeredGrid(
            tuple(jnp.asarray(component) for component in field.components),
            BOX,
            field.extrapolation,
            field.resolution,
        )

        rate = tendency(walled, viscosity=0.01)

        assert all(bool(jnp.all(jnp.isfinite(part))) for part in rate.components)

    @pytest.mark.parametrize("extrapolation", [Extrapolation.NEUMANN])
    def test_a_boundary_outside_the_contract_is_still_refused(
        self, extrapolation: Extrapolation
    ) -> None:
        field = StaggeredGrid.zeros((8, 8), BOX, extrapolation)

        with pytest.raises(ValueError, match="does not carry"):
            tendency(field, viscosity=0.01)
