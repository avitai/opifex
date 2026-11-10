"""Tests for Burgers2DSolver implementation.

Following TDD principles and JAX/Flax NNX guidelines from critical_technical_guidelines.md
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.physics.solvers.burgers import (
    Burgers2DSolver,
    solve_burgers_2d,
    stable_substep_count,
)
from opifex.physics.solvers.diffusion_advection import solve_diffusion_advection_2d
from opifex.physics.solvers.shallow_water import solve_shallow_water_2d


class TestBurgersSolverTransforms:
    """The PDE solvers must be jit/vmap-compatible (opifex is JAX-native).

    These would have caught the ``float()``-in-``while``-loop defect that made
    ``solve_burgers_2d`` un-jittable and forced slow per-sample data generation
    (vmapped on-device generation is ~10^5x faster). The 1D Burgers operator data
    is now produced by the pseudo-spectral solver
    (:mod:`opifex.physics.spectral.steppers`), tested under ``tests/physics/spectral``.
    """

    def test_solve_2d_is_jit_and_vmap_compatible(self) -> None:
        """solve_burgers_2d is also jit/vmap-compatible."""
        ics = jax.random.normal(jax.random.PRNGKey(1), (4, 16, 16))
        viscs = jnp.full((4,), 0.1)
        batched = jax.jit(jax.vmap(lambda u, v: solve_burgers_2d(u, v, (0.0, 1.0), 3, 16)))
        out = batched(ics, viscs)
        assert out.shape == (4, 4, 16, 16)
        assert jnp.all(jnp.isfinite(out))


class TestBurgers2DSolver:
    """Test suite for Burgers2DSolver with full coverage."""

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
        solver = Burgers2DSolver(resolution=16, viscosity=0.1)  # High viscosity for stability

        # Create simple initial condition
        u, v = solver.create_vortex_initial_condition(strength=0.1)  # Small strength

        times, u_traj, v_traj = solver.solve((u, v), time_final=0.01, num_saves=4)

        assert times.shape == (5,)
        assert u_traj.shape == (5, 16, 16)
        assert v_traj.shape == (5, 16, 16)
        assert jnp.all(jnp.isfinite(u_traj))
        assert jnp.all(jnp.isfinite(v_traj))
        assert jnp.allclose(u_traj[0], u, rtol=1e-10)
        assert jnp.allclose(v_traj[0], v, rtol=1e-10)
        np.testing.assert_allclose(times, jnp.linspace(0.0, 0.01, 5), rtol=1e-6)

    def test_solve_saves_only_the_final_state_by_default(self):
        solver = Burgers2DSolver(resolution=16, viscosity=0.1)
        u, v = solver.create_vortex_initial_condition(strength=0.1)

        times, u_traj, v_traj = solver.solve((u, v), time_final=0.005)

        assert times.shape == (2,)
        assert u_traj.shape[0] == 2
        assert v_traj.shape[0] == 2

    def test_solve_is_jit_and_vmap_compatible(self):
        """The class solves on device, as the module-level function does."""
        solver = Burgers2DSolver(resolution=16, viscosity=0.1)
        u, v = solver.create_vortex_initial_condition(strength=0.1)
        batch = (jnp.stack([u, 0.5 * u]), jnp.stack([v, 0.5 * v]))

        batched = jax.jit(jax.vmap(lambda a, b: solver.solve((a, b), time_final=0.005)[1]))(*batch)

        assert batched.shape == (2, 2, 16, 16)
        assert jnp.all(jnp.isfinite(batched))

    def test_solving_in_two_legs_matches_one(self):
        """Integrating to t, then on to 2t, lands where integrating to 2t does."""
        solver = Burgers2DSolver(resolution=16, viscosity=0.1)
        u, v = solver.create_vortex_initial_condition(strength=0.1)

        _, first, second = solver.solve((u, v), time_final=0.004)
        _, again, and_again = solver.solve((first[-1], second[-1]), time_final=0.004)
        _, straight_u, straight_v = solver.solve((u, v), time_final=0.008)

        np.testing.assert_allclose(again[-1], straight_u[-1], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(and_again[-1], straight_v[-1], rtol=1e-4, atol=1e-6)

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
        times, u_traj, v_traj = solver.solve((u, v), time_final, num_saves=3)

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


class TestSolveBurgers2d:
    """Test suite for the standalone solve_burgers_2d function."""

    def test_output_shape(self):
        """Output shape should be (time_steps+1, resolution, resolution)."""
        resolution = 16
        time_steps = 3
        ic = jnp.zeros((resolution, resolution))
        result = solve_burgers_2d(ic, viscosity=0.1, time_steps=time_steps, resolution=resolution)
        assert result.shape == (time_steps + 1, resolution, resolution)

    def test_initial_condition_preserved(self):
        """First time step should match the initial condition."""
        resolution = 16
        x = jnp.linspace(-1, 1, resolution)
        X, Y = jnp.meshgrid(x, x, indexing="ij")
        ic = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        result = solve_burgers_2d(ic, viscosity=0.1, time_steps=3, resolution=resolution)
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
    with pytest.raises(ValueError, match="advection_vel must be a tuple of two numbers"):
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


class TestReverseModeDifferentiation:
    """Both solvers must carry reverse mode, which a ``lax.while_loop`` cannot.

    The sub-steps between save times used to step adaptively to the CFL limit inside a
    ``lax.while_loop``. That traces and maps, so `jit` and `vmap` tests passed and the
    defect stayed hidden -- but `while_loop` has no transpose rule, so `jax.grad` of
    anything reaching either solver raised outright. A fixed sub-step count under
    ``lax.scan`` costs the adaptivity and buys the gradient.
    """

    @staticmethod
    def _scalar_field(n: int) -> jax.Array:
        axis = jnp.linspace(-1.0, 1.0, n)
        x, y = jnp.meshgrid(axis, axis, indexing="ij")
        return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)

    def test_solve_burgers_2d_differentiates_in_reverse_mode(self) -> None:
        start = self._scalar_field(16)

        gradient = jax.grad(
            lambda u: jnp.sum(solve_burgers_2d(u, 0.05, (0.0, 0.2), 2, 16)[-1] ** 2)
        )(start)

        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(jnp.max(jnp.abs(gradient))) > 0.0

    def test_the_solver_class_differentiates_in_reverse_mode(self) -> None:
        n = 16
        axis = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        x, y = jnp.meshgrid(axis, axis, indexing="ij")
        u_start, v_start = jnp.sin(x) * jnp.cos(y), -jnp.cos(x) * jnp.sin(y)
        solver = Burgers2DSolver(viscosity=0.05, resolution=n)

        gradient = jax.grad(
            lambda u: jnp.sum(solver.solve((u, v_start), 0.2, num_saves=1)[1][-1] ** 2)
        )(u_start)

        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(jnp.max(jnp.abs(gradient))) > 0.0

    def test_the_substep_count_follows_the_courant_limit(self) -> None:
        # Below the limit the scheme does not degrade gracefully: measured at 32 cells
        # with nu = 0.05 over T = 0.5, against a 4096-step reference, 32 sub-steps (0.75
        # of the limit) gives 1.6e-02, 16 gives 2.7, and 8 gives 4.6e+06. So the count is
        # derived rather than guessed.
        start = self._scalar_field(32)

        coarse = stable_substep_count(start, 0.05, 0.25, 2.0 / 31)
        fine = stable_substep_count(start, 0.05, 0.25, 2.0 / 63)

        assert coarse >= 1
        # Halving the cell halves the advective limit, so the count at least doubles.
        assert fine >= 2 * coarse


#: Reverse mode must reproduce forward mode to within the solve's own summation depth.
#: Both come from JAX's autodiff but lower to *different* XLA programs, so agreement is
#: evidence about the transpose rule -- which is exactly what the ``lax.while_loop`` to
#: ``lax.scan`` change above alters.
#:
#: The budget is the number of sequential sub-steps the gradient accumulates over,
#: ``time_steps * substeps = 2 * 32``, because two orderings of one value can differ by
#: O(depth) roundings and no more. It is deliberately *not* the measured agreement: on
#: the shipped solver that is 2.35 eps in float32 and 1.59 eps in float64, and an
#: earlier version of this constant took 4.0 from those figures. A gradient-only
#: mutation (``stop_gradient`` on the advecting coefficient, which leaves the primal
#: bit-identical) then moved float32 agreement to 10.05 eps purely by changing the XLA
#: graph, and the test failed for a reason that was not the defect. Two orderings of the
#: *same* value cannot drift beyond the depth; a genuinely wrong transpose rule returns a
#: different value altogether, which this still refuses by orders of magnitude.
MODE_AGREEMENT_ULP = 64.0

#: Step for the float64 central difference, at the measured minimum of its own error
#: curve. Sweeping h over 1e-1 .. 1e-8 on a strictly-signed field puts the relative
#: error at 1.4e-04, 9.1e-08, 6.0e-11, 5.6e-09, 1.1e-05 -- the classic U, truncation
#: falling as h^2 then roundoff rising as 1/h, with its floor at 1e-4.
CENTRAL_DIFFERENCE_STEP = 1e-4

#: Tolerance for the float64 central difference, from that sweep: the worst relative
#: error over the cases exercised here is 4.3e-10 (inviscid) and 6.0e-11 (viscous), so
#: 1e-8 keeps at least 23x headroom while still refusing anything meaningfully wrong.
#:
#: It is emphatically *not* JAX's own ``default_gradient_tolerance`` of 2e-3 for
#: float32. That number is the float32 finite-difference floor -- with
#: ``check_grads``'s ``EPS = 1e-4``, roundoff alone is eps/h = 1.2e-03 -- and measured
#: here float32 differencing of this solver is worse still: 4.6e-02 relative at h=1e-4,
#: and exactly zero below h=1e-6, where the two evaluations cancel completely. A
#: float32 finite difference cannot certify this gradient at any tolerance worth
#: asserting, which is why the comparison runs in float64.
CENTRAL_DIFFERENCE_TOLERANCE = 1e-8


class TestGradientAccuracy:
    """The gradient must *be* the derivative, not merely be finite and non-zero.

    ``TestReverseModeDifferentiation`` above establishes that reverse mode runs. Its two
    assertions -- ``isfinite`` and ``max |g| > 0`` -- are both satisfied by a gradient
    that is entirely wrong, so neither measures accuracy. These do.

    **The field matters more than the tolerance, and that is the whole lesson here.**
    The upwind advection is a ``where(u > 0, ...)`` switch, so the loss is piecewise
    smooth with a kink wherever ``u`` vanishes.
    ``TestReverseModeDifferentiation._scalar_field`` is ``sin(pi x) cos(pi y)`` on
    ``linspace(-1, 1, n)``, which is exactly zero on the ``x = +-1`` rows -- it sits *on*
    the switching surface. Measured in float64, a central difference there is wrong by a
    relative 2.13e-04 and **stays wrong at that value for every h from 1e-2 to 1e-8**,
    because at a kink a central difference converges to the mean of the two one-sided
    derivatives while autodiff returns one branch. Adding a constant offset to lift the
    field off zero drops the same comparison to 6.0e-11.

    So a finite-difference check added to that field would have had to carry a tolerance
    near 5e-04 to pass, and would then have reported this gradient as "accurate to 5e-04"
    when on a smooth field it is accurate to 6e-11 -- a threshold chosen to make a test
    pass, hiding a four-order-of-magnitude-better result and asserting nothing about the
    adjoint. The smooth case below is therefore the accuracy test, and the kink is
    covered separately, by the assertion that actually holds there.
    """

    @staticmethod
    def _direction(shape: tuple[int, ...], dtype) -> jax.Array:
        """One fixed unit vector, so the derivative is a scalar and h sweeps are cheap.

        Args:
            shape: Field shape to draw for.
            dtype: Dtype of the returned direction.

        Returns:
            A unit-norm array of `shape`.
        """
        drawn = jax.random.normal(jax.random.key(0), shape, dtype=dtype)
        return drawn / jnp.linalg.norm(drawn)

    @staticmethod
    def _loss(values: jax.Array, viscosity: float) -> jax.Array:
        """Scalar functional of the final state, which is what a caller differentiates.

        Args:
            values: Initial condition.
            viscosity: Kinematic viscosity.

        Returns:
            The sum of squares of the final state.
        """
        return jnp.sum(solve_burgers_2d(values, viscosity, (0.0, 0.2), 2, 16)[-1] ** 2)

    def _smooth_start(self, offset: float, dtype) -> jax.Array:
        """A field held strictly away from zero, so upwind takes one branch throughout.

        Args:
            offset: Constant added to lift the field off the switching surface.
            dtype: Dtype of the returned field.

        Returns:
            A 16x16 field whose values all share one sign.
        """
        axis = jnp.linspace(-1.0, 1.0, 16, dtype=dtype)
        x, y = jnp.meshgrid(axis, axis, indexing="ij")
        return (offset + jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)).astype(dtype)

    @pytest.mark.parametrize("viscosity", [0.05, 0.0])
    @pytest.mark.parametrize("offset", [2.0, -2.0])
    def test_the_gradient_matches_a_central_difference_on_a_smooth_field(
        self, viscosity: float, offset: float
    ) -> None:
        """The reference is independent of the adjoint: only the primal is called.

        Run under ``jax.enable_x64(True)``, following
        ``tests/deployment/test_ase_calculator.py``'s force check and
        ``tests/neural/atomistic/backbones/_helpers.py``: the conftest pins
        ``jax_enable_x64`` off, and a float32 difference of a sum-of-squares over a
        256-cell field cancels catastrophically.

        ``viscosity=0.0`` is the inviscid edge case, where the diffusive term vanishes
        and advection alone carries the gradient; both signs of ``offset`` are taken
        because upwind's branch -- and so the entire adjoint stencil -- depends on the
        sign of ``u``, and a field of one sign exercises only one of the two branches.
        """
        with jax.enable_x64(True):
            start = self._smooth_start(offset, jnp.float64)
            assert float(jnp.min(jnp.abs(start))) > 0.5, (
                "this case is only valid off the upwind switching surface; a field "
                "reaching zero makes the central difference straddle a kink"
            )
            direction = self._direction(start.shape, jnp.float64)

            analytic = float(jnp.vdot(jax.grad(self._loss)(start, viscosity), direction))
            step = jnp.asarray(CENTRAL_DIFFERENCE_STEP, dtype=jnp.float64)
            differenced = float(
                (
                    self._loss(start + step * direction, viscosity)
                    - self._loss(start - step * direction, viscosity)
                )
                / (2.0 * step)
            )

            np.testing.assert_allclose(
                analytic, differenced, rtol=CENTRAL_DIFFERENCE_TOLERANCE, atol=0.0
            )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_forward_and_reverse_mode_agree_to_a_few_ulp(self, dtype) -> None:
        """Checks the transpose rule, which is what ``lax.scan`` replaced.

        ``lax.while_loop`` has no transpose rule at all, so this is the assertion that
        would have gone from *raising* to *passing* across that change -- and unlike the
        finite difference it is valid on any field, kinks included, because both modes
        differentiate the same branch.
        """
        with jax.enable_x64(dtype is jnp.float64):
            start = self._smooth_start(2.0, dtype)
            direction = self._direction(start.shape, dtype)

            reverse = float(jnp.vdot(jax.grad(self._loss)(start, 0.05), direction))
            _, forward = jax.jvp(lambda values: self._loss(values, 0.05), (start,), (direction,))

            budget = MODE_AGREEMENT_ULP * float(jnp.finfo(dtype).eps)
            np.testing.assert_allclose(float(forward), reverse, rtol=budget, atol=0.0)

    def test_at_an_upwind_kink_the_two_modes_still_agree(self) -> None:
        """The edge case: on the switching surface, this is the assertion that holds.

        ``sin(pi x) cos(pi y)`` is exactly zero on the ``x = +-1`` rows, so the loss has
        a kink at the evaluation point. The gradient is a valid one-sided derivative
        there and both modes return it, but a central difference does not converge to
        it -- which is asserted explicitly in the companion case below, so that nobody
        later "repairs" this by loosening a tolerance.
        """
        with jax.enable_x64(True):
            axis = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float64)
            x, y = jnp.meshgrid(axis, axis, indexing="ij")
            start = (jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)).astype(jnp.float64)
            assert float(jnp.min(jnp.abs(start))) < 1e-12, (
                "this case is about the switching surface; a field that never reaches "
                "zero would make it a duplicate of the smooth one"
            )
            direction = self._direction(start.shape, jnp.float64)

            reverse = float(jnp.vdot(jax.grad(self._loss)(start, 0.05), direction))
            _, forward = jax.jvp(lambda values: self._loss(values, 0.05), (start,), (direction,))

            budget = MODE_AGREEMENT_ULP * float(jnp.finfo(jnp.float64).eps)
            np.testing.assert_allclose(float(forward), reverse, rtol=budget, atol=0.0)

    def test_a_central_difference_does_not_converge_at_the_kink(self) -> None:
        """Pins the mechanism, so the smooth case's field choice cannot be undone.

        Without this, a later reader sees an oddly specific ``offset=2.0`` and is free to
        drop it, re-introduce the zero-crossing field, and widen the tolerance until the
        test passes again. Refining h by four orders of magnitude must leave the
        discrepancy essentially unchanged -- that is the signature of a kink rather than
        of truncation (which falls as h^2) or roundoff (which grows as 1/h).
        """
        with jax.enable_x64(True):
            axis = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float64)
            x, y = jnp.meshgrid(axis, axis, indexing="ij")
            start = (jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)).astype(jnp.float64)
            direction = self._direction(start.shape, jnp.float64)
            analytic = float(jnp.vdot(jax.grad(self._loss)(start, 0.05), direction))

            errors = []
            for step_size in (1e-4, 1e-8):
                step = jnp.asarray(step_size, dtype=jnp.float64)
                differenced = float(
                    (
                        self._loss(start + step * direction, 0.05)
                        - self._loss(start - step * direction, 0.05)
                    )
                    / (2.0 * step)
                )
                errors.append(abs(differenced - analytic) / abs(analytic))

            coarse, fine = errors
            assert coarse > 1e-5, (
                "the central difference is expected to disagree at the kink; if it now "
                "agrees, the scheme's upwind switch has changed and the smooth case's "
                f"offset may no longer be needed (relative error {coarse:.3e})"
            )
            # A plateau, not a trend: four orders of refinement move it under 2x.
            assert 0.5 < fine / coarse < 2.0, (
                "refining h changed the discrepancy, so it is truncation or roundoff "
                f"rather than a kink: {coarse:.3e} -> {fine:.3e}"
            )


class TestValueAccuracy:
    """Exact invariants of the scheme, asserted exactly.

    Each of these is a property of the discretisation rather than of this
    implementation, and each was checked against the solver before being asserted --
    all three hold bit-exactly, so none carries a tolerance. A tolerance here would be
    pure slack, and slack is what lets a defect through.
    """

    @staticmethod
    def _solve(values: jax.Array, viscosity: float = 0.05) -> jax.Array:
        """Advance a field with the module-level solver.

        Args:
            values: Initial condition.
            viscosity: Kinematic viscosity.

        Returns:
            The saved trajectory.
        """
        return solve_burgers_2d(values, viscosity, (0.0, 0.2), 2, 16)

    def test_the_zero_field_stays_exactly_zero(self) -> None:
        """Every term is homogeneous in ``u``, so zero is a fixed point exactly."""
        trajectory = self._solve(jnp.zeros((16, 16)))

        assert bool(jnp.all(trajectory == 0.0))

    @pytest.mark.parametrize("constant", [1.0, 3.7, -2.5])
    def test_a_constant_field_stays_exactly_constant(self, constant: float) -> None:
        """Advection and diffusion of a constant both vanish, to the last bit.

        A negative constant is included because upwind takes the other branch there,
        and a branch that differenced the wrong neighbour would still preserve a
        *positive* constant.
        """
        trajectory = self._solve(jnp.full((16, 16), constant))

        assert bool(jnp.all(trajectory == constant))

    def test_the_trajectory_opens_with_the_initial_condition_unchanged(self) -> None:
        """The first saved state is stored, not computed, so it is bit-identical.

        ``TestBurgers2DSolver`` asserts this with ``rtol=1e-10`` and ``atol=1e-6``.
        Both are slack on a value that is copied: measured, ``jnp.array_equal`` holds.
        Worse, ``jnp.allclose(a, b, rtol=1e-10)`` carries a default ``atol=1e-8``, which
        dominates wherever the field passes through zero -- so the stated ``1e-10`` is
        not the tolerance actually in force on much of the field.
        """
        axis = jnp.linspace(-1.0, 1.0, 16)
        x, y = jnp.meshgrid(axis, axis, indexing="ij")
        start = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)

        trajectory = self._solve(start)

        assert bool(jnp.array_equal(trajectory[0], start))

    def test_the_same_inputs_give_bit_identical_results(self) -> None:
        """Determinism is exact or it is not determinism.

        ``TestBurgers2DSolver.test_deterministic_behavior`` asserts this with
        ``rtol=1e-10``, which admits a difference where there can be none: the same
        program on the same device is reproducible to the bit. A tolerance there cannot
        fail for any reason that would not also be a far larger bug.
        """
        axis = jnp.linspace(-1.0, 1.0, 16)
        x, y = jnp.meshgrid(axis, axis, indexing="ij")
        start = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)

        assert bool(jnp.array_equal(self._solve(start), self._solve(start)))
