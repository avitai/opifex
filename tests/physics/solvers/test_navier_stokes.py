"""
TDD Tests for Navier-Stokes Solver.

Following TDD principles: Tests written FIRST, then implementation.
The 2D incompressible Navier-Stokes equations:
    du/dt + (u·∇)u = -∇p/ρ + ν∇²u
    ∇·u = 0 (incompressibility)
"""

import jax.numpy as jnp


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

        # First time step should be initial condition
        assert jnp.allclose(u_traj[0], u0, atol=1e-5)
        assert jnp.allclose(v_traj[0], v0, atol=1e-5)

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

        # Solution should change over time
        assert not jnp.allclose(u_traj[0], u_traj[-1], atol=0.1)

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

        # Test at resolution 64 - allow for float32 roundoff error
        # Float32 epsilon is ~1e-7, divided by dx (~0.1) gives ~1e-6
        div_64 = compute_max_divergence(64)
        assert div_64 < 1e-4, f"Divergence too large at res=64: {div_64}"

        # Test at resolution 128 - should also be small
        div_128 = compute_max_divergence(128)
        assert div_128 < 1e-4, f"Divergence too large at res=128: {div_128}"

        # Note: We cannot test O(h²) convergence with float32 because roundoff
        # error dominates. The analytical function is divergence-free, which
        # can be verified with float64 where divergence is ~10^-15.

    def test_lid_driven_cavity_import(self):
        """Test that lid-driven cavity IC can be imported."""
        from opifex.physics.solvers.navier_stokes import create_lid_driven_cavity_ic

        assert create_lid_driven_cavity_ic is not None

    def test_lid_driven_cavity_shape(self):
        """Test that lid-driven cavity IC has correct shape."""
        from opifex.physics.solvers.navier_stokes import create_lid_driven_cavity_ic

        resolution = 64
        u0, v0 = create_lid_driven_cavity_ic(resolution, lid_velocity=1.0)

        assert u0.shape == (resolution, resolution)
        assert v0.shape == (resolution, resolution)
