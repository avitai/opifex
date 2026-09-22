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

import jax
import jax.numpy as jnp

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import divergence, StaggeredGrid
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
