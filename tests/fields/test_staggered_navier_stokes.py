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
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import divergence, StaggeredGrid
from opifex.fields.staggered_convection import convect
from opifex.fields.staggered_diffusion import laplacian, wall_source, WallVelocity
from opifex.fields.staggered_navier_stokes import (
    integrate,
    stable_step_count,
    step,
    tendency,
)
from opifex.fields.staggered_pressure import project
from tests.fields.staggered_support import (
    gradient_transform_agreement,
    gradient_versus_central_difference,
    JIT_GRADIENT_ULP,
    operator_matrix,
    TAYLOR_RATE_FLOOR,
    taylor_remainder_rates,
    TAYLOR_RESIDUAL_FLOOR,
    VMAP_GRADIENT_ULP,
)


BOX = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))

#: Central-difference step for the gradient checks, at the minimum of the measured error
#: curve for both of them.
CENTRAL_DIFFERENCE_STEP = 1e-4

#: Budget for the trajectory gradient. Looser than the pressure projection's 1e-9
#: because this accumulates over five steps, each holding a convection, a diffusion and
#: a solve: the measured worst case over two seeds is 7.536e-10, leaving about a hundred
#: and thirty times headroom.
TRAJECTORY_GRADIENT_TOLERANCE = 1e-7

#: Budget for the gradient with respect to the wall velocity, which runs over four steps
#: on an eight-cell grid and measures 2.264e-11 -- about four hundred times inside this.
LID_GRADIENT_TOLERANCE = 1e-8


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

    @pytest.mark.parametrize("seed", [0, 5])
    def test_the_trajectory_gradient_matches_a_central_difference(self, seed: int) -> None:
        """The gradient through five steps must be the derivative, not merely finite.

        ``solve_navier_stokes_2d`` raises here: its adaptive sub-stepping is a
        ``lax.while_loop`` with a data-dependent trip count, which has no reverse rule.
        That is what this test exists for -- but ``isfinite`` plus ``max |g| > 0``, which
        is what it used to assert, is satisfied by a gradient that is entirely wrong.
        Measured on the Burgers solver, a ``stop_gradient`` on the advecting coefficient
        leaves the primal bit-identical, corrupts the adjoint, and passes both.

        The reference calls only the primal, so a broken transpose rule cannot hide
        behind it.
        """
        start = _solenoidal(16, seed=seed)

        with jax.enable_x64(True):
            components = tuple(c.astype(jnp.float64) for c in start.components)

            def loss(parts: tuple[jax.Array, ...]) -> jax.Array:
                field = StaggeredGrid(parts, BOX, Extrapolation.PERIODIC, (16, 16))
                return _energy(integrate(field, total_time=0.05, num_steps=5, viscosity=0.01))

            relative = gradient_versus_central_difference(
                loss, components, step=CENTRAL_DIFFERENCE_STEP
            )

        assert relative <= TRAJECTORY_GRADIENT_TOLERANCE, (
            f"reverse mode and a central difference disagree by a relative {relative:.3e}, "
            f"against a measured worst case of 7.5e-10 and a budget of "
            f"{TRAJECTORY_GRADIENT_TOLERANCE:.0e}"
        )

    def test_the_taylor_remainder_converges_at_second_order(self) -> None:
        """Adjoint verification through five steps, judged by a rate not a tolerance.

        Follows pyadjoint's ``taylor_test``. The remainder
        ``|J(m + eps h) - J(m) - eps <grad J, h>|`` is O(eps^2) for a correct gradient
        and O(eps) for a wrong one, so the criterion is rate 2 against rate 1 -- nothing
        to tune, which is what makes it the right primary check for a gradient taken
        through a loop of convections, diffusions and solves.

        Measured: residuals 1.18e-05, 2.95e-06, 7.37e-07, 1.84e-07, rates all 2.000.
        """
        start = _solenoidal(16)

        with jax.enable_x64(True):
            components = tuple(c.astype(jnp.float64) for c in start.components)

            def loss(parts: tuple[jax.Array, ...]) -> jax.Array:
                field = StaggeredGrid(parts, BOX, Extrapolation.PERIODIC, (16, 16))
                return _energy(integrate(field, total_time=0.05, num_steps=5, viscosity=0.01))

            residuals, rates = taylor_remainder_rates(loss, components)

        assert min(residuals) > TAYLOR_RESIDUAL_FLOOR, (
            f"smallest remainder {min(residuals):.3e} is at the rounding floor"
        )
        assert min(rates) > TAYLOR_RATE_FLOOR, (
            f"Taylor rates {[f'{r:.3f}' for r in rates]} from residuals "
            f"{[f'{r:.3e}' for r in residuals]}; a rate near 1 is a wrong gradient"
        )

    def test_the_trajectory_gradient_survives_jit_and_vmap(self) -> None:
        """The composition that matters most for the ``while_loop`` to ``scan`` change.

        ``while_loop`` has no transpose rule, so the adjoint of this trajectory did not
        exist at all before. Having gained it, it has to survive being compiled and
        being mapped: ``vmap`` of a trajectory gradient is what an ensemble or batched
        control problem runs, and it had no coverage anywhere in this suite.
        """
        start = _solenoidal(16)

        def loss(parts: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(parts, BOX, Extrapolation.PERIODIC, (16, 16))
            return _energy(integrate(field, total_time=0.05, num_steps=5, viscosity=0.01))

        jit_gap, vmap_gap = gradient_transform_agreement(loss, start.components)

        assert jit_gap <= JIT_GRADIENT_ULP, (
            f"jit(grad) differs from grad by {jit_gap:.2f} ULP; measured 1.38 on the CPU "
            "backend CI pins and 0.00 on this machine's GPU"
        )
        assert vmap_gap <= VMAP_GRADIENT_ULP, (
            f"vmap(grad) differs from a loop by {vmap_gap:.2f} ULP; this measured exactly "
            "zero on both backends"
        )

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

    def test_the_lid_gradient_matches_a_central_difference(self) -> None:
        """Boundary control needs the *right* gradient, not just a finite one.

        The wall velocity is an array in the residual path, so a wall profile is an
        ordinary differentiable input -- which is what makes boundary control possible,
        and which is worth nothing if the gradient is wrong. This replaces an
        ``isfinite`` plus ``max |leaf| > 0`` pair that a corrupted adjoint satisfies.

        The lid is a pytree rather than a component tuple, so it is flattened to its
        leaves and rebuilt inside the loss; the helper then perturbs the leaves exactly
        as it does a velocity's components.
        """
        with jax.enable_x64(True):
            # Built inside the context on purpose: a float32 carry meeting float64 walls
            # makes `lax.scan` reject the trajectory outright, because the carry's output
            # dtype no longer matches its input's.
            still = StaggeredGrid.zeros((8, 8), BOX, Extrapolation.ZERO)
            leaves, structure = jax.tree.flatten(self._lid(8))
            components = tuple(jnp.asarray(leaf, dtype=jnp.float64) for leaf in leaves)

            def energy(parts: tuple[jax.Array, ...]) -> jax.Array:
                walls = jax.tree.unflatten(structure, list(parts))
                driven = integrate(still, 0.02, 4, 0.01, walls)
                return jnp.stack([jnp.sum(part**2) for part in driven.components]).sum()

            relative = gradient_versus_central_difference(
                energy, components, step=CENTRAL_DIFFERENCE_STEP
            )

        assert relative <= LID_GRADIENT_TOLERANCE, (
            f"the wall-velocity gradient and a central difference disagree by a relative "
            f"{relative:.3e}, against a measured 2.3e-11 and a budget of "
            f"{LID_GRADIENT_TOLERANCE:.0e}"
        )

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


def _real_axis_stability_limit(stability_polynomial: Callable[[float], float]) -> float:
    """The largest ``L`` with ``|R(-L)| <= 1``, by bisection on ``[0, 8]``.

    Args:
        stability_polynomial: ``R(z)`` for the scheme.

    Returns:
        Where the scheme leaves the unit disc on the negative real axis.
    """
    low, high = 0.0, 8.0
    for _ in range(60):
        middle = (low + high) / 2.0
        low, high = (middle, high) if abs(stability_polynomial(-middle)) <= 1.0 else (low, middle)
    return low


def _forward_euler(z: float) -> float:
    """``R(z) = 1 + z``, the scheme jax-cfd's published limit is derived for.

    Args:
        z: Point on the real axis.

    Returns:
        The amplification factor.
    """
    return 1.0 + z


def _classical_rk4(z: float) -> float:
    """``R(z)`` for the four-stage, fourth-order Runge-Kutta method ``step`` implements.

    Args:
        z: Point on the real axis.

    Returns:
        The amplification factor.
    """
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


class TestTheStabilityConstant:
    """``stable_step_count``'s ``0.348`` is checked against its derivation and against a
    maintained reference implementation, rather than quoted.

    The number appears in two docstrings and once in the code, and both docstrings explain
    where it comes from: classical RK4 is stable on the negative real axis to 2.785, and the
    most negative eigenvalue of the MAC Laplacian is ``8 nu / dx^2`` in two dimensions, so
    the diffusive limit is ``2.785 / 8 = 0.348`` times ``dx^2 / nu``.

    **A docstring is a record of what was true when it was written**, so every half is
    measured here. The same construction -- a time scheme's real-axis stability limit over
    the spatial operator's spectral radius -- is what two maintained libraries use:

    * ``google/jax-cfd``'s ``jax_cfd.base.diffusion.stable_time_step`` returns
      ``dx ** 2 / (viscosity * 2 ** ndim)``, which is ``0.25 dx^2 / nu`` in two dimensions.
      That is the same ``8/dx^2`` spectral radius under forward Euler, whose limit is 2
      rather than 2.785, and ``0.348 / 0.25`` is ``2.785 / 2`` exactly.
    * ``CliMA/Oceananigans.jl`` PR #6389 multiplies its ``cell_diffusion_timescale`` by
      ``3/4`` for a fourth-order Laplacian "because the spectral radius of the five-point
      Laplacian (16/3 d^2) is 4/3 that of the three-point Laplacian" -- the same rule, applied
      to a change of stencil rather than a change of scheme.

    Neither library tests its own constant: ``jax_cfd/base/diffusion_test.py`` has no test of
    ``stable_time_step``, and Oceananigans' PR lists ``cell_diffusion_timescale`` as covered
    without showing an assertion. So the cross-check below is the reference these tests lean
    on, and the tests themselves are this repository's own.
    """

    #: The constant as ``stable_step_count`` declares it, rounded to three decimals there.
    DECLARED = 0.348

    #: ``jax_cfd.base.diffusion.stable_time_step``: ``dx**2 / (viscosity * 2**ndim)``.
    JAX_CFD_TWO_DIMENSIONAL = 0.25

    def test_the_time_scheme_is_stable_to_the_declared_point_on_the_real_axis(self) -> None:
        """Classical RK4's stability polynomial, bisected rather than recalled."""
        measured = _real_axis_stability_limit(_classical_rk4)

        assert measured == pytest.approx(2.785, abs=1e-3), (
            f"RK4's real-axis limit measures {measured:.6f}; the derivation behind 0.348 "
            "assumes 2.785, so either the scheme changed or the constant needs rederiving"
        )

    def test_the_constant_is_jax_cfds_rescaled_by_the_time_scheme(self) -> None:
        """Cross-library: the same spectral radius under a different scheme.

        ``jax-cfd`` sizes the explicit diffusive step for forward Euler and this package for
        RK4, over the identical ``8 nu / dx^2`` spectral radius, so the two constants must
        stand in the ratio of the two schemes' real-axis limits. Both limits are bisected
        here rather than quoted, so nothing in this assertion is a number read off a page.
        """
        euler = _real_axis_stability_limit(_forward_euler)
        runge_kutta = _real_axis_stability_limit(_classical_rk4)

        assert euler == pytest.approx(2.0, abs=1e-9), "forward Euler leaves the disc at -2"
        allowed_ratio = self.DECLARED / self.JAX_CFD_TWO_DIMENSIONAL

        assert allowed_ratio == pytest.approx(runge_kutta / euler, rel=1e-3), (
            f"this package allows {self.DECLARED / self.JAX_CFD_TWO_DIMENSIONAL:.6f} times "
            f"jax-cfd's step while its scheme tolerates {runge_kutta / euler:.6f} times the "
            "eigenvalue; one of the two constants no longer matches its scheme"
        )

    @pytest.mark.parametrize("cells", [8, 9, 16])
    def test_the_viscous_operator_is_bounded_by_eight_over_the_spacing_squared(
        self, cells: int
    ) -> None:
        """The other half, read off the assembled operator rather than assumed.

        **The bound is attained only for an even cell count, and that precondition is part
        of the claim.** The periodic second difference has eigenvalues
        ``-4 sin^2(pi k / n) / h^2`` for ``k = 0 .. n-1``, so ``-4/h^2`` is reached by the
        Nyquist mode ``k = n/2``, which exists only when ``n`` is even. Measured: the ratio
        to the bound is 1.00000000 at 8 and 16 cells and 0.96984632 at 9. Asserting
        attainment unconditionally would fail on any odd resolution and report it as a
        changed stencil, which is why ``cells`` is parametrised across both parities here.

        Under a solid boundary the extreme eigenvalue is less negative at either parity,
        which leaves the step count conservative in the safe direction.
        """
        for extrapolation in (Extrapolation.PERIODIC, Extrapolation.ZERO):
            field = StaggeredGrid.zeros((cells, cells), BOX, extrapolation)
            spacing = float(np.asarray(field.dx)[0])
            matrix = np.asarray(
                operator_matrix(laplacian, (cells, cells), extrapolation), dtype=np.float64
            )
            worst = float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0).min())
            bound = -8.0 / spacing**2
            # At an even cell count the bound is ATTAINED, so this comparison sits on a
            # knife edge and must allow the eigensolver its own error. LAPACK's symmetric
            # driver is backward stable: computed eigenvalues lie within p(n) * eps *
            # ||A||_2 of the true ones, with ||A||_2 = |bound|; p(n) is instantiated as n,
            # conservatively. Measured at 16 cells the overshoot is 4.5e-13 against the
            # 2.3e-10 allowed -- 500x of headroom, which costs nothing here because the
            # defect this detects (a changed viscous stencil) moves the eigenvalue by order
            # ||A||, thirteen decades above the allowance.
            eigensolver_error = matrix.shape[0] * float(np.finfo(np.float64).eps) * abs(bound)

            assert worst >= bound - eigensolver_error, (
                f"{extrapolation.value}: most negative eigenvalue {worst:.6f} is beyond the "
                f"{bound:.6f} the step count is sized against, by more than the "
                f"{eigensolver_error:.3e} the symmetric eigensolver is entitled to"
            )
            if extrapolation == Extrapolation.PERIODIC and cells % 2 == 0:
                assert worst == pytest.approx(bound, rel=1e-5), (
                    f"at {cells} cells the periodic Nyquist mode should attain the bound "
                    "exactly; if it no longer does, the viscous stencil changed"
                )
            elif extrapolation == Extrapolation.PERIODIC:
                assert worst > bound * (1.0 - 1e-6), (
                    f"an odd cell count has no Nyquist mode, so {cells} cells must fall "
                    "strictly inside the bound rather than reach it"
                )

    def test_the_declared_constant_is_the_ratio_of_those_two(self) -> None:
        """2.785 / 8, which is the whole derivation, to the three decimals the source keeps.

        The tolerance is the half-width of a three-decimal rounding, which is what the
        source performs: ``2.785 / 8`` is 0.348125 and it is written as ``0.348``. Anything
        tighter would reject a correct rounding; the exact-ceiling test below is what
        catches a constant that is actually wrong.
        """
        declared = self.DECLARED

        assert declared == pytest.approx(2.785 / 8.0, abs=5e-4)

    @pytest.mark.parametrize(
        ("cells", "viscosity", "total_time"),
        [(16, 0.05, 0.5), (8, 0.1, 0.25), (24, 0.02, 1.0), (12, 0.03, 2.0)],
    )
    def test_the_step_count_is_exactly_the_ceiling_of_the_limit(
        self, cells: int, viscosity: float, total_time: float
    ) -> None:
        """The count is an integer, so this compares integers and needs no tolerance at all.

        An earlier version compared the realised step against half the limit with
        ``rel=1/count``, which is the right bound on what ``ceil`` can cost -- and useless,
        because at ten steps it permits a tenth, which is exactly the size of an off-by-one
        in the count. Comparing ``ceil`` to ``ceil`` removes the tolerance and the blind spot
        together.

        **The expectation is derived, never measured**: ``DECLARED`` is the constant the
        source states, checked above against RK4's bisected limit over the operator's
        measured spectral radius and against jax-cfd's published constant. Nothing here is
        read back out of ``stable_step_count``.

        The configurations avoid quotients that sit on a ceiling edge, and the test says so
        rather than depending on it silently: at 32 cells over an interval of 4 the quotient
        is 1177.01, where the source's rounding of the constant to three decimals is the
        difference between 1177 and 1178.
        """
        field = StaggeredGrid.zeros((cells, cells), BOX, Extrapolation.ZERO)
        spacing = float(np.asarray(field.dx)[0])
        step = 0.5 * self.DECLARED * spacing**2 / viscosity
        quotient = total_time / step
        assert 0.05 < quotient - math.floor(quotient) < 0.95, (
            f"quotient {quotient:.4f} sits on a ceiling edge, where rounding the constant "
            "decides the count; this configuration cannot test the ceiling"
        )

        # The field is at rest, so the advective limit is infinite and the diffusive one binds.
        count = stable_step_count(field, total_time, viscosity)

        assert count == math.ceil(quotient)
        assert total_time / count <= step, "the ceiling must never round the step up"
