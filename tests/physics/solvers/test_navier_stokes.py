"""
TDD Tests for Navier-Stokes Solver.

Following TDD principles: Tests written FIRST, then implementation.
The 2D incompressible Navier-Stokes equations:
    du/dt + (u·∇)u = -∇p/ρ + ν∇²u
    ∇·u = 0 (incompressibility)
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestNavierStokesTransforms:
    """The solver must be jit/vmap-compatible (opifex is JAX-native).

    Would have caught the ``float()``-in-``while`` defect that made the
    Navier-Stokes solver un-jittable and forced slow per-sample data generation.
    """

    def test_solve_is_jit_compatible(self) -> None:
        """solve_navier_stokes_2d traces under jit and preserves the IC."""
        from opifex.physics.solvers.navier_stokes import (
            create_taylor_green_vortex,
            solve_navier_stokes_2d,
        )

        u0, v0 = create_taylor_green_vortex(16)
        u_traj, v_traj = jax.jit(
            lambda a, b: solve_navier_stokes_2d(a, b, 0.05, (0.0, 0.5), 3, 16)
        )(u0, v0)
        assert u_traj.shape == (4, 16, 16)
        assert jnp.allclose(u_traj[0], u0) and jnp.allclose(v_traj[0], v0)
        assert jnp.all(jnp.isfinite(u_traj)) and jnp.all(jnp.isfinite(v_traj))

    def test_solve_is_vmap_compatible(self) -> None:
        """The solver vmaps over a batch of initial fields and viscosities."""
        from opifex.physics.solvers.navier_stokes import (
            create_taylor_green_vortex,
            solve_navier_stokes_2d,
        )

        u0, v0 = create_taylor_green_vortex(16)
        us, vs = jnp.stack([u0, u0]), jnp.stack([v0, v0])
        nus = jnp.array([0.05, 0.1])
        batched = jax.jit(
            jax.vmap(lambda a, b, n: solve_navier_stokes_2d(a, b, n, (0.0, 0.5), 3, 16))
        )
        u_traj, v_traj = batched(us, vs, nus)
        assert u_traj.shape == (2, 4, 16, 16) and v_traj.shape == (2, 4, 16, 16)
        assert jnp.all(jnp.isfinite(u_traj)) and jnp.all(jnp.isfinite(v_traj))


class TestIncompressibility:
    """The equation the solver is named for: the evolved field stays divergence free.

    The projection step exists to enforce this, and nothing else in the suite measures
    whether it does. The divergence is read with the same central difference the solver
    projects against, so this states the solver's own contract rather than an independent
    discretisation's opinion of it.

    What is left after an exact projection is the round-off of that central difference,
    float32 eps over dx, which is 6e-07 at the resolution used here. The limit below is
    just above it; the measured values are 2.4e-07 across an evolved trajectory and
    4.7e-10 for a single step from a divergent start.
    """

    # What is left after an exact projection is the round-off of the central difference
    # that measures it, which divides by dx and so grows as the grid is refined. Measured
    # max|div| over an evolved trajectory, as a multiple of eps/dx: 2.36, 2.09, 1.87, 1.81
    # at 16, 32, 64 and 128 cells -- flat, which makes eps/dx the model and a single
    # number the wrong shape. The limit is four times the largest of those.
    @staticmethod
    def _roundoff(n: int) -> float:
        return 8.0 * float(jnp.finfo(jnp.float32).eps) / (2.0 * jnp.pi / n)

    @staticmethod
    def _max_divergence(u: jax.Array, v: jax.Array) -> float:
        dx = 2 * jnp.pi / u.shape[0]
        du_dx = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * dx)
        dv_dy = (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * dx)
        return float(jnp.max(jnp.abs(du_dx + dv_dy)))

    def test_an_evolved_field_stays_divergence_free(self) -> None:
        from opifex.physics.solvers.navier_stokes import (
            create_taylor_green_vortex,
            solve_navier_stokes_2d,
        )

        u0, v0 = create_taylor_green_vortex(32)
        u_traj, v_traj = solve_navier_stokes_2d(u0, v0, 0.05, (0.0, 0.5), 4, 32)

        for step in range(1, u_traj.shape[0]):
            assert self._max_divergence(u_traj[step], v_traj[step]) <= self._roundoff(32)

    @pytest.mark.parametrize(("final_time", "resolution"), [(1.0, 32), (1.0, 64), (5.0, 32)])
    def test_the_vortex_decays_at_the_analytic_rate(
        self, final_time: float, resolution: int
    ) -> None:
        """The Taylor-Green vortex decays as ``exp(-2 nu t)``, and now it does.

        The limit is not a fixed tolerance. Measured, the relative error times ``n^2`` is
        flat -- 0.0655, 0.0656, 0.0663, 0.0658 at 16, 32, 64 and 128 cells for ``t = 1``
        -- and scales linearly with the interval, giving ``0.066 * t / n^2``. So the
        second order is itself the assertion, and a fixed tolerance wide enough for the
        coarsest grid would pass a scheme a hundred times worse on the finest.

        An advection term dissipating faster than the equations it solves fails this: a
        numerical viscosity of ``|u| dx / 2`` at 32 cells stands an order of magnitude
        above a physical ``nu`` of 0.01, which costs the vortex 8% of its decay at
        ``t = 1`` and 29% at ``t = 5`` and converges only at first order.
        """
        from opifex.physics.solvers.navier_stokes import (
            create_taylor_green_vortex,
            solve_navier_stokes_2d,
        )

        viscosity = 0.01
        u0, v0 = create_taylor_green_vortex(resolution)

        u_traj, _ = solve_navier_stokes_2d(u0, v0, viscosity, (0.0, final_time), 5, resolution)

        decay = float(jnp.max(jnp.abs(u_traj[-1]))) / float(jnp.max(jnp.abs(u0)))
        analytic = float(jnp.exp(-2 * viscosity * final_time))
        limit = 0.2 * final_time / resolution**2
        assert abs(decay - analytic) / analytic <= limit

    def test_a_divergent_start_is_projected_within_one_step(self) -> None:
        # The Taylor-Green vortex starts divergence free, so it cannot show whether the
        # projection removes divergence or merely fails to introduce it.
        from opifex.physics.solvers.navier_stokes import solve_navier_stokes_2d

        resolution = 32
        axis = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
        x, y = jnp.meshgrid(axis, axis, indexing="ij")
        u0 = jnp.sin(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x)
        v0 = jnp.cos(x) * jnp.sin(y)

        before = self._max_divergence(u0, v0)
        u_traj, v_traj = solve_navier_stokes_2d(u0, v0, 0.05, (0.0, 0.1), 1, resolution)

        assert before > 0.1
        assert self._max_divergence(u_traj[-1], v_traj[-1]) <= self._roundoff(resolution)


class TestNavierStokesSolver:
    """Test suite for Navier-Stokes solver following TDD principles."""

    def test_import_solver(self):
        """Test that we can import the solver."""
        from opifex.physics.solvers.navier_stokes import solve_navier_stokes_2d

        assert solve_navier_stokes_2d is not None

    def test_solve_returns_correct_shape(self):
        """Test that solver returns correct output shape."""
        from opifex.physics.solvers.navier_stokes import solve_navier_stokes_2d

        resolution = 32
        time_steps = 5

        # Initial velocity field (u, v)
        u0 = jnp.zeros((resolution, resolution))
        v0 = jnp.zeros((resolution, resolution))

        u_traj, v_traj = solve_navier_stokes_2d(
            u0=u0,
            v0=v0,
            nu=0.01,
            time_range=(0.0, 1.0),
            time_steps=time_steps,
            resolution=resolution,
        )

        # Should return (time_steps+1, resolution, resolution) for each component
        # (includes initial condition)
        assert u_traj.shape == (time_steps + 1, resolution, resolution)
        assert v_traj.shape == (time_steps + 1, resolution, resolution)

    def test_solve_preserves_initial_condition(self):
        """Test that first time step matches initial condition."""
        from opifex.physics.solvers.navier_stokes import solve_navier_stokes_2d

        resolution = 32

        # Create non-trivial initial condition (Taylor-Green vortex)
        x = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        u0 = jnp.sin(X) * jnp.cos(Y)
        v0 = -jnp.cos(X) * jnp.sin(Y)

        u_traj, v_traj = solve_navier_stokes_2d(
            u0=u0,
            v0=v0,
            nu=0.01,
            time_range=(0.0, 0.1),
            time_steps=3,
            resolution=resolution,
        )

        # The first frame is the initial condition itself, concatenated onto the saved
        # states rather than computed, so it is equal rather than close.
        assert jnp.array_equal(u_traj[0], u0)
        assert jnp.array_equal(v_traj[0], v0)

    def test_solve_evolves_in_time(self):
        """Test that solution evolves over time (not static)."""
        from opifex.physics.solvers.navier_stokes import solve_navier_stokes_2d

        resolution = 32

        # Create non-trivial initial condition
        x = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        u0 = jnp.sin(X) * jnp.cos(Y)
        v0 = -jnp.cos(X) * jnp.sin(Y)

        u_traj, _ = solve_navier_stokes_2d(
            u0=u0,
            v0=v0,
            nu=0.01,
            time_range=(0.0, 1.0),
            time_steps=5,
            resolution=resolution,
        )

        # The Taylor-Green vortex decays as exp(-2 nu t), so at nu=0.01 over one time unit
        # the amplitude falls by 2%. A margin of 0.1 was wider than the whole physical
        # change and read spurious drift as evolution.
        #
        # The one-sided form this replaces -- decays at least as fast as the analytic rate
        # -- was written for the upwind scheme on the reasoning that a discretisation adds
        # dissipation and never removes it. That is false for a scheme whose convective
        # term produces no energy: its error is dispersive and lands on either side. It
        # now sits 6.3e-05 above the analytic amplitude, which the old assertion read as a
        # failure. Two-sided, against the same second-order bound the decay-rate test
        # uses.
        amplitudes = [float(jnp.max(jnp.abs(frame))) for frame in u_traj]
        analytic = float(jnp.exp(-2 * 0.01 * 1.0))

        assert all(later < earlier for earlier, later in itertools.pairwise(amplitudes))
        assert abs(amplitudes[-1] / amplitudes[0] - analytic) / analytic <= 0.2 / resolution**2

    def test_solve_with_different_viscosities(self):
        """Test that higher viscosity leads to more diffusion."""
        from opifex.physics.solvers.navier_stokes import solve_navier_stokes_2d

        resolution = 32

        # Create initial condition with sharp gradients
        x = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        u0 = jnp.sin(2 * X) * jnp.cos(2 * Y)  # Higher frequency
        v0 = -jnp.cos(2 * X) * jnp.sin(2 * Y)

        # Solve with low viscosity
        u_low, _ = solve_navier_stokes_2d(
            u0=u0,
            v0=v0,
            nu=0.001,  # Low viscosity
            time_range=(0.0, 0.5),
            time_steps=5,
            resolution=resolution,
        )

        # Solve with high viscosity
        u_high, _ = solve_navier_stokes_2d(
            u0=u0,
            v0=v0,
            nu=0.1,  # High viscosity
            time_range=(0.0, 0.5),
            time_steps=5,
            resolution=resolution,
        )

        # Higher viscosity should result in more diffused solution (lower variance)
        var_low = jnp.var(u_low[-1])
        var_high = jnp.var(u_high[-1])
        assert var_high < var_low

    def test_solve_deterministic(self):
        """Test that solver is deterministic."""
        from opifex.physics.solvers.navier_stokes import solve_navier_stokes_2d

        resolution = 32

        u0 = jnp.ones((resolution, resolution)) * 0.1
        v0 = jnp.zeros((resolution, resolution))

        u1, v1 = solve_navier_stokes_2d(
            u0=u0,
            v0=v0,
            nu=0.01,
            time_range=(0.0, 0.5),
            time_steps=5,
            resolution=resolution,
        )

        u2, v2 = solve_navier_stokes_2d(
            u0=u0,
            v0=v0,
            nu=0.01,
            time_range=(0.0, 0.5),
            time_steps=5,
            resolution=resolution,
        )

        assert jnp.allclose(u1, u2)
        assert jnp.allclose(v1, v2)

    def test_solve_returns_finite_values(self):
        """Test that solver returns finite (non-NaN, non-Inf) values."""
        from opifex.physics.solvers.navier_stokes import solve_navier_stokes_2d

        resolution = 32

        # Random-ish initial condition
        x = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        u0 = jnp.sin(X) * jnp.cos(Y)
        v0 = -jnp.cos(X) * jnp.sin(Y)

        u_traj, v_traj = solve_navier_stokes_2d(
            u0=u0,
            v0=v0,
            nu=0.01,
            time_range=(0.0, 1.0),
            time_steps=10,
            resolution=resolution,
        )

        assert jnp.all(jnp.isfinite(u_traj))
        assert jnp.all(jnp.isfinite(v_traj))


class TestNavierStokesVortexInitialConditions:
    """Tests for Navier-Stokes initial condition generators."""

    def test_taylor_green_vortex_import(self):
        """Test that Taylor-Green vortex generator can be imported."""
        from opifex.physics.solvers.navier_stokes import create_taylor_green_vortex

        assert create_taylor_green_vortex is not None

    def test_taylor_green_vortex_shape(self):
        """Test that Taylor-Green vortex has correct shape."""
        from opifex.physics.solvers.navier_stokes import create_taylor_green_vortex

        resolution = 64
        u0, v0 = create_taylor_green_vortex(resolution)

        assert u0.shape == (resolution, resolution)
        assert v0.shape == (resolution, resolution)

    def test_taylor_green_vortex_incompressible(self):
        """Test that Taylor-Green vortex satisfies incompressibility.

        The Taylor-Green vortex is analytically divergence-free:
        u = A * sin(x) * cos(y), v = -A * cos(x) * sin(y)
        ∂u/∂x + ∂v/∂y = A*cos(x)*cos(y) - A*cos(x)*cos(y) = 0

        Note on numerical precision:
        - With float64, the finite difference divergence is ~10^-15 (machine epsilon)
        - With float32 (JAX default), we observe ~10^-6 due to catastrophic
          cancellation when summing two nearly equal and opposite derivatives.
        - The O(h²) truncation error is smaller than float32 roundoff error,
          so we cannot verify O(h²) convergence without float64.

        This test verifies:
        1. The divergence is small (within float32 tolerance)
        2. The implementation is correct (verified analytically)
        """
        from opifex.physics.solvers.navier_stokes import create_taylor_green_vortex

        def compute_max_divergence(resolution):
            u0, v0 = create_taylor_green_vortex(resolution)
            dx = 2 * jnp.pi / resolution
            du_dx = (jnp.roll(u0, -1, axis=0) - jnp.roll(u0, 1, axis=0)) / (2 * dx)
            dv_dy = (jnp.roll(v0, -1, axis=1) - jnp.roll(v0, 1, axis=1)) / (2 * dx)
            divergence = du_dx + dv_dy
            return float(jnp.max(jnp.abs(divergence)))

        # The floor is the cancellation itself: float32 eps over dx, which grows as the
        # grid is refined. Measured 2.4e-06 at 64 and 6.7e-06 at 128, against a floor of
        # 1.2e-06 and 2.4e-06 -- a factor of two to three. The limit is ten times the
        # floor, so it follows the resolution instead of being one number wide enough for
        # both, which at 1e-04 was forty times the value it was meant to catch.
        def roundoff_floor(resolution: int) -> float:
            return float(jnp.finfo(jnp.float32).eps) / (2 * jnp.pi / resolution)

        div_64 = compute_max_divergence(64)
        assert div_64 < 10 * roundoff_floor(64), f"Divergence too large at res=64: {div_64}"

        div_128 = compute_max_divergence(128)
        assert div_128 < 10 * roundoff_floor(128), f"Divergence too large at res=128: {div_128}"

        # Note: We cannot test O(h²) convergence with float32 because roundoff
        # error dominates. The analytical function is divergence-free, which
        # can be verified with float64 where divergence is ~10^-15.


class TestLidDrivenCavity:
    """The cavity is a boundary-value problem, so it is tested as one.

    It is driven by a wall held in motion for all time, not by an initial condition. The
    previous approximation put a smooth profile inside a periodic box, which has no wall to
    drag anything and so decays instead of recirculating; it also returned a peak of 0.679
    when asked for a lid velocity of 1.0.
    """

    def test_a_lid_at_rest_leaves_the_cavity_at_rest(self):
        """The negative control: without it, a solver returning zeros would pass below."""
        from opifex.physics.solvers.navier_stokes import solve_lid_driven_cavity

        still = solve_lid_driven_cavity(16, nu=0.05, lid_velocity=0.0, total_time=1.0)

        assert max(float(jnp.max(jnp.abs(c))) for c in still.components) == 0.0

    def test_the_lid_drives_a_recirculating_vortex(self):
        """The cavity's defining feature, and the one the old approximation could not make.

        A lid-driven cavity is a boundary-value problem: the flow is produced by a wall
        held in motion, not by an initial blob. The signature is a sign reversal up the
        centreline -- fluid dragged along under the lid, returning beneath it. A periodic
        box with a smooth initial profile has no wall to drag anything and simply decays.
        """
        from opifex.physics.solvers.navier_stokes import solve_lid_driven_cavity

        cavity = solve_lid_driven_cavity(32, nu=0.05, lid_velocity=1.0, total_time=4.0)
        centreline = np.asarray(cavity.components[0])[16]

        assert float(centreline.max()) > 0.3, "fluid must be dragged along by the lid"
        assert float(centreline.min()) < -0.05, "and must return beneath it"

    def test_the_driven_cavity_stays_divergence_free(self):
        """Incompressibility is what the projection is for, so it is asserted directly."""
        from opifex.fields.staggered import divergence
        from opifex.physics.solvers.navier_stokes import solve_lid_driven_cavity

        cavity = solve_lid_driven_cavity(32, nu=0.05, lid_velocity=1.0, total_time=4.0)
        speed = max(float(jnp.max(jnp.abs(c))) for c in cavity.components)

        assert float(jnp.max(jnp.abs(divergence(cavity)))) < 1e-4 * speed * 32

    def test_the_default_step_count_is_stable(self):
        """A step above the viscous limit returns NaN rather than raising."""
        from opifex.physics.solvers.navier_stokes import solve_lid_driven_cavity

        cavity = solve_lid_driven_cavity(32, nu=0.05, lid_velocity=1.0, total_time=4.0)

        assert all(bool(jnp.all(jnp.isfinite(c))) for c in cavity.components)
