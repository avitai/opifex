"""Tests for Burgers2DSolver implementation.

Following TDD principles and JAX/Flax NNX guidelines from critical_technical_guidelines.md
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.physics.solvers.burgers import (
    Burgers2DSolver,
    solve_burgers_1d,
    solve_burgers_2d,
)
from opifex.physics.solvers.diffusion_advection import solve_diffusion_advection_2d
from opifex.physics.solvers.shallow_water import solve_shallow_water_2d


class TestBurgers2DSolver:
    """Test suite for Burgers2DSolver with comprehensive coverage."""

    def test_solver_initialization_default_params(self):
        """Test solver initializes with default parameters correctly."""
        solver = Burgers2DSolver()

        assert solver.resolution == 64
        assert solver.domain_size == (2.0 * jnp.pi, 2.0 * jnp.pi)
        assert solver.viscosity == 0.01
        assert solver.dt_max == 0.001

    def test_solver_initialization_custom_params(self):
        """Test solver initializes with custom parameters correctly."""
        solver = Burgers2DSolver(
            resolution=32, domain_size=(4.0, 6.0), viscosity=0.05, dt_max=0.005
        )

        assert solver.resolution == 32
        assert solver.domain_size == (4.0, 6.0)
        assert solver.viscosity == 0.05
        assert solver.dt_max == 0.005

    def test_grid_spacing_computation(self):
        """Test grid spacing is computed correctly."""
        solver = Burgers2DSolver(resolution=32, domain_size=(4.0, 6.0))

        expected_dx = 4.0 / 32
        expected_dy = 6.0 / 32

        assert jnp.isclose(solver.dx, expected_dx)
        assert jnp.isclose(solver.dy, expected_dy)

    def test_coordinate_grid_creation(self):
        """Test coordinate grid creation."""
        solver = Burgers2DSolver(resolution=16, domain_size=(2.0, 3.0))

        assert solver.X.shape == (16, 16)
        assert solver.Y.shape == (16, 16)

        # Check that grids cover the domain properly
        assert jnp.min(solver.X) >= 0
        assert jnp.max(solver.X) < 2.0
        assert jnp.min(solver.Y) >= 0
        assert jnp.max(solver.Y) < 3.0

    def test_derivatives_computation(self):
        """Test spatial derivatives computation."""
        solver = Burgers2DSolver(resolution=16)

        # Simple test fields
        u = jnp.sin(solver.X) * jnp.cos(solver.Y)
        v = jnp.cos(solver.X) * jnp.sin(solver.Y)

        derivatives = solver._compute_derivatives(u, v)

        # Should return 8 derivative arrays
        assert len(derivatives) == 8

        # All should have correct shape
        for deriv in derivatives:
            assert deriv.shape == (16, 16)
            assert jnp.all(jnp.isfinite(deriv))

    def test_adaptive_time_step_computation(self):
        """Test adaptive time step computation."""
        solver = Burgers2DSolver(resolution=16, dt_max=0.01)

        # Small velocity field
        u = jnp.ones((16, 16)) * 0.1
        v = jnp.ones((16, 16)) * 0.1

        dt = solver._adaptive_time_step(u, v)

        assert dt > 0
        assert dt <= solver.dt_max
        assert jnp.isfinite(dt)

    def test_rk4_step(self):
        """Test RK4 time integration step."""
        solver = Burgers2DSolver(resolution=16)

        # Simple initial state
        u = jnp.sin(solver.X) * jnp.cos(solver.Y) * 0.1
        v = jnp.cos(solver.X) * jnp.sin(solver.Y) * 0.1

        dt = 0.001
        u_new, v_new = solver._rk4_step((u, v), dt)

        assert u_new.shape == u.shape
        assert v_new.shape == v.shape
        assert jnp.all(jnp.isfinite(u_new))
        assert jnp.all(jnp.isfinite(v_new))

    def test_vortex_initial_condition_creation(self):
        """Test vortex initial condition creation."""
        solver = Burgers2DSolver(resolution=16)

        u, v = solver.create_vortex_initial_condition(strength=1.0)

        assert u.shape == (16, 16)
        assert v.shape == (16, 16)
        assert jnp.all(jnp.isfinite(u))
        assert jnp.all(jnp.isfinite(v))

    def test_vortex_initial_condition_with_custom_center(self):
        """Test vortex initial condition with custom center."""
        solver = Burgers2DSolver(resolution=16, domain_size=(4.0, 4.0))

        center = (1.0, 2.0)
        u, v = solver.create_vortex_initial_condition(strength=0.5, center=center)

        assert u.shape == (16, 16)
        assert v.shape == (16, 16)
        assert jnp.all(jnp.isfinite(u))
        assert jnp.all(jnp.isfinite(v))

    def test_shear_layer_initial_condition_creation(self):
        """Test shear layer initial condition creation."""
        solver = Burgers2DSolver(resolution=16)

        u, v = solver.create_shear_layer_initial_condition(shear_strength=1.0)

        assert u.shape == (16, 16)
        assert v.shape == (16, 16)
        assert jnp.all(jnp.isfinite(u))
        assert jnp.all(jnp.isfinite(v))

    def test_complete_solve_small_problem(self):
        """Test complete solve for a small problem."""
        solver = Burgers2DSolver(
            resolution=16, viscosity=0.1
        )  # High viscosity for stability

        # Create simple initial condition
        u, v = solver.create_vortex_initial_condition(strength=0.1)  # Small strength

        time_final = 0.01  # Short time
        times, u_traj, v_traj = solver.solve((u, v), time_final, save_every=1)

        # Check output shapes
        n_steps = len(times)
        assert u_traj.shape == (n_steps, 16, 16)
        assert v_traj.shape == (n_steps, 16, 16)

        # Check that all values are finite
        assert jnp.all(jnp.isfinite(u_traj))
        assert jnp.all(jnp.isfinite(v_traj))

        # Check that initial condition is preserved
        assert jnp.allclose(u_traj[0], u, rtol=1e-10)
        assert jnp.allclose(v_traj[0], v, rtol=1e-10)

    def test_solve_without_save_every(self):
        """Test solve without intermediate saving."""
        solver = Burgers2DSolver(resolution=16, viscosity=0.1)

        u, v = solver.create_vortex_initial_condition(strength=0.1)

        time_final = 0.005
        times, u_traj, v_traj = solver.solve((u, v), time_final)  # No save_every

        # Should only have initial and final states
        assert len(times) == 2
        assert u_traj.shape[0] == 2
        assert v_traj.shape[0] == 2

    def test_input_validation(self):
        """Test input validation in solve method."""
        solver = Burgers2DSolver(resolution=16)

        # Wrong shape inputs
        u_wrong = jnp.ones((10, 10))  # Wrong size
        v_correct = jnp.ones((16, 16))

        with pytest.raises(ValueError, match=r"u shape .* != expected .*"):
            solver.solve((u_wrong, v_correct), 0.01)

    def test_energy_behavior(self):
        """Test that energy behaves reasonably."""
        solver = Burgers2DSolver(resolution=16, viscosity=0.05)

        # Create initial condition with some energy
        u, v = solver.create_vortex_initial_condition(strength=0.2)

        time_final = 0.01
        times, u_traj, v_traj = solver.solve((u, v), time_final, save_every=2)

        # Compute energy at each time step
        energies = []
        for i in range(len(times)):
            energy = jnp.sum(u_traj[i] ** 2 + v_traj[i] ** 2)
            energies.append(energy)

        # Energy should generally decrease due to viscosity
        # (Allow some tolerance for numerical effects)
        assert energies[-1] <= energies[0] + 0.1 * energies[0]

    def test_jax_transformations_compatibility(self):
        """Test JAX transformations compatibility."""
        solver = Burgers2DSolver(resolution=8)  # Small for testing

        # Test that vmap works with initial condition creation
        @jax.vmap
        def create_multiple_vortices(strength):
            return solver.create_vortex_initial_condition(strength=strength)

        strengths = jnp.array([0.5, 1.0, 1.5])
        u_batch, v_batch = create_multiple_vortices(strengths)

        assert u_batch.shape == (3, 8, 8)
        assert v_batch.shape == (3, 8, 8)
        assert jnp.all(jnp.isfinite(u_batch))
        assert jnp.all(jnp.isfinite(v_batch))

    def test_deterministic_behavior(self):
        """Test that solver produces deterministic results."""
        solver1 = Burgers2DSolver(resolution=16, viscosity=0.1)
        solver2 = Burgers2DSolver(resolution=16, viscosity=0.1)

        # Same initial condition
        u, v = solver1.create_vortex_initial_condition(strength=0.1)

        time_final = 0.005
        times1, u_traj1, v_traj1 = solver1.solve((u, v), time_final)
        times2, u_traj2, v_traj2 = solver2.solve((u, v), time_final)

        assert jnp.allclose(times1, times2, rtol=1e-10)
        assert jnp.allclose(u_traj1, u_traj2, rtol=1e-10)
        assert jnp.allclose(v_traj1, v_traj2, rtol=1e-10)

    def test_error_handling_invalid_params(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError, match="resolution must be a positive integer"):
            Burgers2DSolver(resolution=0)

        with pytest.raises(ValueError, match="viscosity must be a positive number"):
            Burgers2DSolver(viscosity=-0.1)

        with pytest.raises(ValueError, match="dt_max must be a positive number"):
            Burgers2DSolver(dt_max=-0.001)


class TestSolveBurgers1d:
    """Test suite for the standalone solve_burgers_1d function."""

    def test_output_shape(self):
        """Output shape should be (time_steps+1, resolution)."""
        resolution = 32
        time_steps = 5
        ic = jnp.sin(jnp.pi * jnp.linspace(-1, 1, resolution))
        result = solve_burgers_1d(
            ic, viscosity=0.1, time_steps=time_steps, resolution=resolution
        )
        assert result.shape == (time_steps + 1, resolution)

    def test_initial_condition_preserved(self):
        """First time step should match the initial condition."""
        resolution = 32
        ic = jnp.sin(jnp.pi * jnp.linspace(-1, 1, resolution))
        result = solve_burgers_1d(
            ic, viscosity=0.1, time_steps=3, resolution=resolution
        )
        assert jnp.allclose(result[0], ic, atol=1e-6)

    def test_numerical_stability_low_viscosity(self):
        """Solution must stay bounded for low viscosity over long time."""
        resolution = 64
        x = jnp.linspace(-1, 1, resolution)
        ic = jnp.sin(jnp.pi * x)
        result = solve_burgers_1d(
            ic,
            viscosity=0.01,
            time_range=(0.0, 1.0),
            time_steps=5,
            resolution=resolution,
        )
        # All values must be finite and bounded
        assert jnp.all(jnp.isfinite(result))
        assert float(jnp.max(jnp.abs(result))) < 10.0

    def test_numerical_stability_shock_initial_condition(self):
        """Solution must stay bounded for step-function (shock) initial conditions."""
        resolution = 64
        x = jnp.linspace(-1, 1, resolution)
        ic = jnp.tanh(x / 0.1)  # Sharp step function
        result = solve_burgers_1d(
            ic,
            viscosity=0.01,
            time_range=(0.0, 1.0),
            time_steps=5,
            resolution=resolution,
        )
        assert jnp.all(jnp.isfinite(result))
        assert float(jnp.max(jnp.abs(result))) < 10.0

    def test_diffusion_smooths_solution(self):
        """High viscosity should smooth the solution over time."""
        resolution = 64
        x = jnp.linspace(-1, 1, resolution)
        ic = jnp.sin(3 * jnp.pi * x)
        result = solve_burgers_1d(
            ic,
            viscosity=0.5,
            time_range=(0.0, 1.0),
            time_steps=5,
            resolution=resolution,
        )
        # Standard deviation should decrease over time (diffusion smooths)
        stds = [float(jnp.std(result[t])) for t in range(result.shape[0])]
        assert stds[-1] < stds[0]

    def test_deterministic(self):
        """Same inputs should produce same outputs."""
        resolution = 32
        ic = jnp.sin(jnp.pi * jnp.linspace(-1, 1, resolution))
        r1 = solve_burgers_1d(ic, viscosity=0.1, time_steps=3, resolution=resolution)
        r2 = solve_burgers_1d(ic, viscosity=0.1, time_steps=3, resolution=resolution)
        assert jnp.allclose(r1, r2, atol=1e-10)


class TestSolveBurgers2d:
    """Test suite for the standalone solve_burgers_2d function."""

    def test_output_shape(self):
        """Output shape should be (time_steps+1, resolution, resolution)."""
        resolution = 16
        time_steps = 3
        ic = jnp.zeros((resolution, resolution))
        result = solve_burgers_2d(
            ic, viscosity=0.1, time_steps=time_steps, resolution=resolution
        )
        assert result.shape == (time_steps + 1, resolution, resolution)

    def test_initial_condition_preserved(self):
        """First time step should match the initial condition."""
        resolution = 16
        x = jnp.linspace(-1, 1, resolution)
        X, Y = jnp.meshgrid(x, x, indexing="ij")
        ic = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        result = solve_burgers_2d(
            ic, viscosity=0.1, time_steps=3, resolution=resolution
        )
        assert jnp.allclose(result[0], ic, atol=1e-6)

    def test_numerical_stability_low_viscosity(self):
        """Solution must stay bounded for low viscosity over long time."""
        resolution = 32
        x = jnp.linspace(-1, 1, resolution)
        X, Y = jnp.meshgrid(x, x, indexing="ij")
        ic = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        result = solve_burgers_2d(
            ic,
            viscosity=0.01,
            time_range=(0.0, 1.0),
            time_steps=3,
            resolution=resolution,
        )
        assert jnp.all(jnp.isfinite(result))
        assert float(jnp.max(jnp.abs(result))) < 10.0


def test_diffusion_advection_validation():
    arr = jnp.ones((8, 8))
    # initial_condition not 2D
    with pytest.raises(ValueError, match="initial_condition must be a 2D array"):
        solve_diffusion_advection_2d(jnp.ones(8), 1.0, (1.0, 1.0))
    # diffusion_coeff <= 0
    with pytest.raises(ValueError, match="diffusion_coeff must be a positive number"):
        solve_diffusion_advection_2d(arr, 0.0, (1.0, 1.0))
    # dt <= 0
    with pytest.raises(ValueError, match="dt must be a positive number"):
        solve_diffusion_advection_2d(arr, 1.0, (1.0, 1.0), dt=0)
    # n_steps <= 0
    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        solve_diffusion_advection_2d(arr, 1.0, (1.0, 1.0), n_steps=0)
    # grid_spacing <= 0
    with pytest.raises(ValueError, match="grid_spacing must be a positive number"):
        solve_diffusion_advection_2d(arr, 1.0, (1.0, 1.0), grid_spacing=0)
    # advection_vel wrong type
    with pytest.raises(
        ValueError, match="advection_vel must be a tuple of two numbers"
    ):
        solve_diffusion_advection_2d(arr, 1.0, (1.0, 2.0, 3.0))  # type: ignore[arg-type]


def test_shallow_water_validation():
    arr = jnp.ones((8, 8))
    # h_initial not 2D
    with pytest.raises(ValueError, match="h_initial must be a 2D array"):
        solve_shallow_water_2d(jnp.ones(8), arr, arr)
    # u_initial not 2D
    with pytest.raises(ValueError, match="u_initial must be a 2D array"):
        solve_shallow_water_2d(arr, jnp.ones(8), arr)
    # v_initial not 2D
    with pytest.raises(ValueError, match="v_initial must be a 2D array"):
        solve_shallow_water_2d(arr, arr, jnp.ones(8))
    # shapes mismatch
    with pytest.raises(ValueError, match="All input fields must have the same shape"):
        solve_shallow_water_2d(arr, jnp.ones((7, 8)), arr)
    # g <= 0
    with pytest.raises(ValueError, match="g must be a positive number"):
        solve_shallow_water_2d(arr, arr, arr, g=0)
    # dt <= 0
    with pytest.raises(ValueError, match="dt must be a positive number"):
        solve_shallow_water_2d(arr, arr, arr, dt=0)
    # n_steps <= 0
    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        solve_shallow_water_2d(arr, arr, arr, n_steps=0)
    # grid_spacing <= 0
    with pytest.raises(ValueError, match="grid_spacing must be a positive number"):
        solve_shallow_water_2d(arr, arr, arr, grid_spacing=0)
