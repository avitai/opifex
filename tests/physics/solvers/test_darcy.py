"""Test Darcy Flow Solver.

Tests for solve_darcy_flow function following TDD principles.
The solver should correctly handle the variable coefficient Darcy equation:
    ∇·(a(x)∇u(x)) = f(x)
"""

import jax.numpy as jnp
import pytest


class TestSolveDarcyFlow:
    """Tests for the Darcy flow solver."""

    def test_different_coefficients_produce_different_solutions(self):
        """Different permeability fields should produce different pressure solutions.

        This is the fundamental property of the Darcy equation - the permeability
        field a(x) directly affects the solution u(x). If this test fails, it means
        the solver is ignoring the coefficient field.
        """
        from opifex.physics.solvers.darcy import solve_darcy_flow

        resolution = 32

        # Two different permeability fields
        coeff_field_1 = jnp.ones((resolution, resolution))
        coeff_field_2 = jnp.ones((resolution, resolution)) * 2.0

        # Solve for both
        u1 = solve_darcy_flow(coeff_field_1, resolution, max_iter=100)
        u2 = solve_darcy_flow(coeff_field_2, resolution, max_iter=100)

        # Solutions must be different
        diff_norm = jnp.linalg.norm(u1 - u2)
        assert diff_norm > 1e-10, (
            f"Solutions should differ for different permeability fields. "
            f"Difference norm: {diff_norm}"
        )

    def test_spatially_varying_coefficient_changes_solution(self):
        """Spatially varying permeability should produce different solution than constant.

        This tests that the solver correctly handles spatially varying coefficients,
        not just different uniform values.
        """
        from opifex.physics.solvers.darcy import solve_darcy_flow

        resolution = 32

        # Constant permeability
        coeff_constant = jnp.ones((resolution, resolution))

        # Spatially varying permeability (checkerboard pattern)
        x = jnp.linspace(0, 1, resolution)
        y = jnp.linspace(0, 1, resolution)
        X, Y = jnp.meshgrid(x, y)
        coeff_varying = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)

        # Solve both
        u_constant = solve_darcy_flow(coeff_constant, resolution, max_iter=100)
        u_varying = solve_darcy_flow(coeff_varying, resolution, max_iter=100)

        # Solutions must be different
        diff_norm = jnp.linalg.norm(u_constant - u_varying)
        assert diff_norm > 1e-6, (
            f"Spatially varying coefficient should produce different solution. "
            f"Difference norm: {diff_norm}"
        )

    def test_boundary_conditions_satisfied(self):
        """Solution should have zero Dirichlet boundary conditions."""
        from opifex.physics.solvers.darcy import solve_darcy_flow

        resolution = 32
        coeff_field = jnp.ones((resolution, resolution))
        u = solve_darcy_flow(coeff_field, resolution, max_iter=100)

        # Check all boundaries are zero
        assert jnp.allclose(u[0, :], 0.0), "Top boundary should be zero"
        assert jnp.allclose(u[-1, :], 0.0), "Bottom boundary should be zero"
        assert jnp.allclose(u[:, 0], 0.0), "Left boundary should be zero"
        assert jnp.allclose(u[:, -1], 0.0), "Right boundary should be zero"

    def test_solution_non_trivial(self):
        """Solution should be non-trivial (not all zeros) in the interior."""
        from opifex.physics.solvers.darcy import solve_darcy_flow

        resolution = 32
        coeff_field = jnp.ones((resolution, resolution))
        u = solve_darcy_flow(coeff_field, resolution, max_iter=100)

        # Interior should have non-zero values
        interior = u[1:-1, 1:-1]
        max_val = jnp.max(jnp.abs(interior))
        assert max_val > 1e-6, f"Interior solution should be non-trivial, max={max_val}"

    def test_solution_positive_for_positive_source(self):
        """With positive source term and zero BCs, solution should be positive."""
        from opifex.physics.solvers.darcy import solve_darcy_flow

        resolution = 32
        coeff_field = jnp.ones((resolution, resolution))
        u = solve_darcy_flow(coeff_field, resolution, max_iter=200)

        # Interior should be positive (for standard Poisson with positive RHS)
        interior = u[1:-1, 1:-1]
        assert jnp.all(interior >= 0), "Solution should be non-negative"

    def test_higher_permeability_yields_lower_pressure_gradient(self):
        """Higher permeability should result in lower maximum pressure.

        For Darcy's law: q = -a∇u, with constant flux, higher a means lower ∇u.
        With zero BCs and constant source, higher permeability = lower max pressure.
        """
        from opifex.physics.solvers.darcy import solve_darcy_flow

        resolution = 32

        coeff_low = jnp.ones((resolution, resolution)) * 1.0
        coeff_high = jnp.ones((resolution, resolution)) * 10.0

        u_low = solve_darcy_flow(coeff_low, resolution, max_iter=200)
        u_high = solve_darcy_flow(coeff_high, resolution, max_iter=200)

        max_u_low = jnp.max(u_low)
        max_u_high = jnp.max(u_high)

        assert max_u_high < max_u_low, (
            f"Higher permeability should yield lower max pressure. "
            f"max_u_low={max_u_low}, max_u_high={max_u_high}"
        )

    def test_symmetric_coefficient_yields_symmetric_solution(self):
        """Symmetric permeability field should produce symmetric solution."""
        from opifex.physics.solvers.darcy import solve_darcy_flow

        resolution = 32

        # Create symmetric coefficient field
        x = jnp.linspace(0, 1, resolution)
        y = jnp.linspace(0, 1, resolution)
        X, Y = jnp.meshgrid(x, y)
        # Symmetric about center
        coeff = 1.0 + jnp.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.1)

        u = solve_darcy_flow(coeff, resolution, max_iter=200)

        # Check symmetry about the center (x and y swap)
        u_transposed = u.T
        symmetry_error = jnp.max(jnp.abs(u - u_transposed))
        assert symmetry_error < 1e-6, (
            f"Solution should be symmetric, error={symmetry_error}"
        )

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        from opifex.physics.solvers.darcy import solve_darcy_flow

        # Wrong dimensions
        with pytest.raises(ValueError, match="2D array"):
            solve_darcy_flow(jnp.ones((32,)), 32)

        # Wrong shape
        with pytest.raises(ValueError, match="shape"):
            solve_darcy_flow(jnp.ones((32, 64)), 32)

        # Invalid resolution
        with pytest.raises(ValueError, match="positive integer"):
            solve_darcy_flow(jnp.ones((32, 32)), -1)

    def test_jit_compatible(self):
        """Solver should work correctly under JIT compilation."""
        from opifex.physics.solvers.darcy import solve_darcy_flow

        resolution = 16
        coeff = jnp.ones((resolution, resolution))

        # Call directly (already JIT compiled via decorator)
        u1 = solve_darcy_flow(coeff, resolution, max_iter=50)

        # Should produce same result when called again
        u2 = solve_darcy_flow(coeff, resolution, max_iter=50)

        assert jnp.allclose(u1, u2), "JIT compilation should be deterministic"

    def test_convergence_with_iterations(self):
        """Solution should converge as iterations increase."""
        from opifex.physics.solvers.darcy import solve_darcy_flow

        resolution = 32
        coeff = jnp.ones((resolution, resolution))

        # Jacobi converges slowly - need many iterations to see convergence
        u_500 = solve_darcy_flow(coeff, resolution, max_iter=500)
        u_1000 = solve_darcy_flow(coeff, resolution, max_iter=1000)
        u_2000 = solve_darcy_flow(coeff, resolution, max_iter=2000)

        # Difference between successive solution snapshots should decrease
        diff_1 = jnp.linalg.norm(u_1000 - u_500)
        diff_2 = jnp.linalg.norm(u_2000 - u_1000)

        # As we approach convergence, changes should get smaller
        assert diff_2 < diff_1, (
            f"Solution should converge. diff_500_1000={diff_1}, diff_1000_2000={diff_2}"
        )

        # Also verify the solutions are approaching a stable value
        # (both differences should be small compared to the solution norm)
        rel_diff = diff_2 / (jnp.linalg.norm(u_2000) + 1e-10)
        assert rel_diff < 0.1, f"Solution should stabilize. Relative diff: {rel_diff}"
