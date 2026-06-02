"""Tests for boundary condition application functions.

This module tests the single source of truth for boundary condition
application logic across the Opifex framework.
"""

import jax.numpy as jnp
import pytest

from opifex.core.physics.boundaries import (
    apply_boundary_condition,
    apply_dirichlet,
    apply_mixed,
    apply_neumann,
    apply_periodic,
    apply_robin,
    BoundaryType,
)


class TestBoundaryTypeEnum:
    """Test BoundaryType enum definition."""

    def test_enum_members(self):
        """Test that all expected boundary types are defined."""
        expected_types = {"DIRICHLET", "NEUMANN", "ROBIN", "PERIODIC", "MIXED"}
        actual_types = {bt.name for bt in BoundaryType}
        assert actual_types == expected_types

    def test_enum_values(self):
        """Test enum string values."""
        assert BoundaryType.DIRICHLET.value == "dirichlet"
        assert BoundaryType.NEUMANN.value == "neumann"
        assert BoundaryType.ROBIN.value == "robin"
        assert BoundaryType.PERIODIC.value == "periodic"
        assert BoundaryType.MIXED.value == "mixed"


class TestApplyDirichlet:
    """Test Dirichlet boundary condition application."""

    def test_dirichlet_zero_boundary(self):
        """Test Dirichlet BC with zero boundary values."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_dirichlet(params, boundary_value=0.0)

        # First and last elements should be zero
        assert result[0] == 0.0
        assert result[-1] == 0.0
        # Interior values should be unchanged
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_dirichlet_nonzero_boundary(self):
        """Test Dirichlet BC with non-zero boundary values."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        boundary_value = 10.0
        result = apply_dirichlet(params, boundary_value=boundary_value)

        assert result[0] == boundary_value
        assert result[-1] == boundary_value
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_dirichlet_different_left_right(self):
        """Test Dirichlet BC with different left and right boundary values."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        left_value = 0.0
        right_value = 10.0
        result = apply_dirichlet(params, left_boundary=left_value, right_boundary=right_value)

        assert result[0] == left_value
        assert result[-1] == right_value
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_dirichlet_single_element(self):
        """Test Dirichlet BC on single element array."""
        params = jnp.array([5.0])
        result = apply_dirichlet(params, boundary_value=0.0)

        # Single element should be set to boundary value
        assert result[0] == 0.0

    def test_dirichlet_two_elements(self):
        """Test Dirichlet BC on two element array."""
        params = jnp.array([1.0, 2.0])
        result = apply_dirichlet(params, boundary_value=0.0)

        # Both elements are boundaries
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_dirichlet_batched(self):
        """Test Dirichlet BC on batched parameters."""
        params = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = apply_dirichlet(params, boundary_value=0.0)

        # Each batch should have boundaries set
        assert jnp.allclose(result[:, 0], 0.0)
        assert jnp.allclose(result[:, -1], 0.0)


class TestApplyNeumann:
    """Test Neumann boundary condition application."""

    def test_neumann_zero_derivative(self):
        """Test Neumann BC with zero derivative at boundaries."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_neumann(params)

        # First element should equal second (zero derivative)
        assert result[0] == params[1]
        # Last element should equal second-to-last (zero derivative)
        assert result[-1] == params[-2]
        # Interior values should be unchanged
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_neumann_gradient_continuity(self):
        """Test that Neumann BC ensures gradient continuity."""
        params = jnp.array([10.0, 2.0, 3.0, 4.0, 20.0])
        result = apply_neumann(params)

        # Boundaries should match interior neighbors
        assert result[0] == params[1]
        assert result[-1] == params[-2]

    def test_neumann_two_elements(self):
        """Test Neumann BC on two element array."""
        params = jnp.array([1.0, 2.0])
        result = apply_neumann(params)

        # Should return unchanged for insufficient elements
        assert jnp.allclose(result, params)

    def test_neumann_single_element(self):
        """Test Neumann BC on single element array."""
        params = jnp.array([5.0])
        result = apply_neumann(params)

        # Should return unchanged
        assert jnp.allclose(result, params)

    def test_neumann_batched(self):
        """Test Neumann BC on batched parameters."""
        params = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = apply_neumann(params)

        # Each batch should have Neumann boundaries
        assert jnp.allclose(result[:, 0], params[:, 1])
        assert jnp.allclose(result[:, -1], params[:, -2])


class TestApplyPeriodic:
    """Test periodic boundary condition application."""

    def test_periodic_simple(self):
        """Test periodic BC makes first and last values equal."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_periodic(params)

        # First and last should be equal (average of original values)
        avg_boundary = (params[0] + params[-1]) / 2
        assert result[0] == avg_boundary
        assert result[-1] == avg_boundary
        # Interior values should be unchanged
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_periodic_already_periodic(self):
        """Test periodic BC on already periodic values."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 1.0])
        result = apply_periodic(params)

        # Should remain at the same value (average of 1.0 and 1.0)
        assert result[0] == 1.0
        assert result[-1] == 1.0

    def test_periodic_single_element(self):
        """Test periodic BC on single element array."""
        params = jnp.array([5.0])
        result = apply_periodic(params)

        # Should return unchanged
        assert jnp.allclose(result, params)

    def test_periodic_two_elements(self):
        """Test periodic BC on two element array."""
        params = jnp.array([1.0, 5.0])
        result = apply_periodic(params)

        # Should average the two values
        avg = (params[0] + params[-1]) / 2
        assert result[0] == avg
        assert result[-1] == avg

    def test_periodic_batched(self):
        """Test periodic BC on batched parameters."""
        params = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = apply_periodic(params)

        # Each batch should have periodic boundaries
        for i in range(params.shape[0]):
            avg = (params[i, 0] + params[i, -1]) / 2
            assert result[i, 0] == avg
            assert result[i, -1] == avg


class TestApplyRobin:
    """Test Robin (mixed) boundary condition application."""

    def test_robin_equal_weights(self):
        """Test Robin BC with equal alpha and beta weights."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha = 1.0
        beta = 1.0
        gamma = 0.0
        result = apply_robin(params, alpha=alpha, beta=beta, gamma=gamma)

        # Robin BC: alpha*u + beta*du/dn = gamma
        # With alpha=beta=1, gamma=0: u + du/dn = 0
        assert result.shape == params.shape
        assert jnp.isfinite(result).all()

    def test_robin_dirichlet_limit(self):
        """Test Robin BC approaches Dirichlet when beta=0."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha = 1.0
        beta = 0.0
        gamma = 0.0

        result = apply_robin(params, alpha=alpha, beta=beta, gamma=gamma)

        # Should behave like Dirichlet with value = gamma/alpha = 0
        assert jnp.allclose(result[0], 0.0, atol=1e-6)
        assert jnp.allclose(result[-1], 0.0, atol=1e-6)

    def test_robin_neumann_limit(self):
        """Test Robin BC approaches Neumann when alpha=0."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha = 0.0
        beta = 1.0
        gamma = 0.0

        result = apply_robin(params, alpha=alpha, beta=beta, gamma=gamma)

        # Should behave like Neumann (derivative = 0)
        # Implementation may vary, just ensure it's finite and reasonable
        assert jnp.isfinite(result).all()


class TestApplyBoundaryCondition:
    """Test unified boundary condition application function."""

    def test_apply_dirichlet_via_enum(self):
        """Test applying Dirichlet via unified function."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_boundary_condition(params, BoundaryType.DIRICHLET, boundary_value=0.0)

        assert result[0] == 0.0
        assert result[-1] == 0.0

    def test_apply_dirichlet_via_string(self):
        """Test applying Dirichlet via string."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_boundary_condition(params, "dirichlet", boundary_value=0.0)

        assert result[0] == 0.0
        assert result[-1] == 0.0

    def test_apply_neumann_via_enum(self):
        """Test applying Neumann via unified function."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_boundary_condition(params, BoundaryType.NEUMANN)

        assert result[0] == params[1]
        assert result[-1] == params[-2]

    def test_apply_neumann_via_string(self):
        """Test applying Neumann via string."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_boundary_condition(params, "neumann")

        assert result[0] == params[1]
        assert result[-1] == params[-2]

    def test_apply_periodic_via_enum(self):
        """Test applying periodic via unified function."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_boundary_condition(params, BoundaryType.PERIODIC)

        avg = (params[0] + params[-1]) / 2
        assert result[0] == avg
        assert result[-1] == avg

    def test_apply_periodic_via_string(self):
        """Test applying periodic via string."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_boundary_condition(params, "periodic")

        avg = (params[0] + params[-1]) / 2
        assert result[0] == avg
        assert result[-1] == avg

    def test_invalid_boundary_type(self):
        """Test that invalid boundary type raises error."""
        params = jnp.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown boundary type"):
            apply_boundary_condition(params, "invalid_type")


class TestWeightedBoundaryApplication:
    """Test weighted boundary condition application."""

    def test_dirichlet_with_weight(self):
        """Test Dirichlet BC with partial weight."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = 0.5
        boundary_value = 0.0

        result = apply_dirichlet(params, boundary_value=boundary_value, weight=weight)

        # With weight=0.5, should be halfway between original and constrained
        expected_first = weight * boundary_value + (1 - weight) * params[0]
        expected_last = weight * boundary_value + (1 - weight) * params[-1]

        assert jnp.isclose(result[0], expected_first)
        assert jnp.isclose(result[-1], expected_last)

    def test_neumann_with_weight(self):
        """Test Neumann BC with partial weight."""
        params = jnp.array([10.0, 2.0, 3.0, 4.0, 20.0])
        weight = 0.5

        result = apply_neumann(params, weight=weight)

        # With weight=0.5, should be halfway between original and constrained
        expected_first = weight * params[1] + (1 - weight) * params[0]
        expected_last = weight * params[-2] + (1 - weight) * params[-1]

        assert jnp.isclose(result[0], expected_first)
        assert jnp.isclose(result[-1], expected_last)

    def test_periodic_with_weight(self):
        """Test periodic BC with partial weight."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = 0.5

        result = apply_periodic(params, weight=weight)

        # Constrained value is the average
        avg = (params[0] + params[-1]) / 2
        expected_first = weight * avg + (1 - weight) * params[0]
        expected_last = weight * avg + (1 - weight) * params[-1]

        assert jnp.isclose(result[0], expected_first)
        assert jnp.isclose(result[-1], expected_last)

    def test_zero_weight_returns_original(self):
        """Test that weight=0 returns original parameters."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = 0.0

        result = apply_dirichlet(params, boundary_value=0.0, weight=weight)

        assert jnp.allclose(result, params)

    def test_full_weight_applies_constraint(self):
        """Test that weight=1.0 fully applies constraint."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weight = 1.0

        result = apply_dirichlet(params, boundary_value=0.0, weight=weight)

        assert result[0] == 0.0
        assert result[-1] == 0.0


class TestJAXCompatibility:
    """Test JAX compatibility of boundary functions."""

    def test_dirichlet_jit_compatible(self):
        """Test that apply_dirichlet is JIT-compatible."""
        import jax

        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        jitted_fn = jax.jit(lambda x: apply_dirichlet(x, boundary_value=0.0))
        result = jitted_fn(params)

        assert result[0] == 0.0
        assert result[-1] == 0.0

    def test_neumann_jit_compatible(self):
        """Test that apply_neumann is JIT-compatible."""
        import jax

        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        jitted_fn = jax.jit(apply_neumann)
        result = jitted_fn(params)

        assert result[0] == params[1]
        assert result[-1] == params[-2]

    def test_periodic_jit_compatible(self):
        """Test that apply_periodic is JIT-compatible."""
        import jax

        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        jitted_fn = jax.jit(apply_periodic)
        result = jitted_fn(params)

        avg = (params[0] + params[-1]) / 2
        assert result[0] == avg
        assert result[-1] == avg

    def test_vmap_compatible(self):
        """Test that boundary functions are vmap-compatible."""
        import jax

        # Batch of parameter arrays
        params_batch = jnp.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        )

        vmapped_fn = jax.vmap(lambda x: apply_dirichlet(x, boundary_value=0.0))
        result = vmapped_fn(params_batch)

        # All batches should have zero boundaries
        assert jnp.allclose(result[:, 0], 0.0)
        assert jnp.allclose(result[:, -1], 0.0)

    def test_grad_compatible(self):
        """Test that boundary functions are differentiable."""
        import jax

        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def loss_fn(x):
            constrained = apply_dirichlet(x, boundary_value=0.0)
            return jnp.sum(constrained**2)

        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(params)

        assert gradients.shape == params.shape
        assert jnp.isfinite(gradients).all()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array(self):
        """Test boundary conditions on empty array."""
        params = jnp.array([])

        result = apply_dirichlet(params, boundary_value=0.0)
        assert result.shape == params.shape

    def test_multidimensional_array(self):
        """Test boundary conditions on 2D array."""
        params = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Should work on last dimension by default
        result = apply_dirichlet(params, boundary_value=0.0)

        assert result.shape == params.shape
        assert jnp.allclose(result[:, 0], 0.0)
        assert jnp.allclose(result[:, -1], 0.0)

    def test_negative_boundary_values(self):
        """Test boundary conditions with negative values."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_dirichlet(params, boundary_value=-10.0)

        assert result[0] == -10.0
        assert result[-1] == -10.0

    def test_large_values(self):
        """Test boundary conditions with large values."""
        params = jnp.array([1e6, 2e6, 3e6, 4e6, 5e6])
        result = apply_dirichlet(params, boundary_value=0.0)

        assert result[0] == 0.0
        assert result[-1] == 0.0
        assert jnp.isfinite(result).all()

    def test_nan_handling(self):
        """Test that functions handle NaN values appropriately."""
        params = jnp.array([1.0, 2.0, jnp.nan, 4.0, 5.0])
        result = apply_dirichlet(params, boundary_value=0.0)

        # Boundaries should be set regardless of NaN in interior
        assert result[0] == 0.0
        assert result[-1] == 0.0


class TestNeumannNonzeroDerivative:
    """Test Neumann BC with a non-zero prescribed outward normal derivative.

    Reference: deepxde NeumannBC residual ``du/dn - g`` with the outward normal
    convention (left normal points in -x, right normal points in +x).
    """

    def test_neumann_nonzero_left_outward_normal(self):
        """Left outward-normal derivative -(u1-u0)/dx must equal target g."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target = 0.5
        result = apply_neumann(params, normal_derivative=target, dx=1.0)

        # Outward normal on the left edge points in -x:
        # du/dn|_left = -(u[1] - u[0]) / dx == target
        achieved = -(result[1] - result[0]) / 1.0
        assert jnp.isclose(achieved, target)

    def test_neumann_nonzero_right_outward_normal(self):
        """Right outward-normal derivative (u_-1 - u_-2)/dx must equal target g."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target = 0.5
        result = apply_neumann(params, normal_derivative=target, dx=1.0)

        # Outward normal on the right edge points in +x:
        # du/dn|_right = (u[-1] - u[-2]) / dx == target
        achieved = (result[-1] - result[-2]) / 1.0
        assert jnp.isclose(achieved, target)

    def test_neumann_nonzero_respects_dx(self):
        """The prescribed flux must be scaled by the grid spacing dx."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target = 2.0
        dx = 0.25
        result = apply_neumann(params, normal_derivative=target, dx=dx)

        # u[0] = u[1] + g * dx (since du/dn|_left = -(u1-u0)/dx = g)
        assert jnp.isclose(result[0], result[1] + target * dx)

    def test_neumann_zero_derivative_unchanged_by_new_param(self):
        """Default g=0 reproduces the historical zero-derivative behaviour."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_neumann(params)

        assert result[0] == params[1]
        assert result[-1] == params[-2]


class TestRobinNeumannLimitNonzeroFlux:
    """Robin BC with alpha=0 and non-zero gamma must enforce du/dn = gamma/beta.

    Previously this path silently returned the input unchanged (no constraint).
    """

    def test_robin_neumann_limit_applies_nonzero_flux(self):
        """alpha=0, beta!=0, gamma!=0 enforces the prescribed normal derivative."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        beta = 2.0
        gamma = 1.0
        result = apply_robin(params, alpha=0.0, beta=beta, gamma=gamma)

        # Equivalent to Neumann with du/dn = gamma / beta on both edges.
        expected = gamma / beta
        achieved_left = -(result[1] - result[0])
        achieved_right = result[-1] - result[-2]
        assert jnp.isclose(achieved_left, expected)
        assert jnp.isclose(achieved_right, expected)

    def test_robin_neumann_limit_not_silent_noop(self):
        """The Neumann-limit must actually modify the boundaries (not a no-op)."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_robin(params, alpha=0.0, beta=1.0, gamma=3.0)

        assert not jnp.allclose(result, params)


class TestApplyMixed:
    """Test genuine per-boundary mixed boundary conditions.

    Reference: deepxde dispatches each boundary segment by its declared BC type
    (DirichletBC / NeumannBC / RobinBC ``error`` methods). The mixed handler must
    apply the proper per-side constraint, never silently substitute Dirichlet.
    """

    def test_mixed_dirichlet_left_neumann_right(self):
        """Dirichlet on the left edge, Neumann on the right edge."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_mixed(
            params,
            left_type=BoundaryType.DIRICHLET,
            right_type=BoundaryType.NEUMANN,
            left_kwargs={"boundary_value": 0.0},
        )

        # Left: Dirichlet value fixed to 0.0
        assert jnp.isclose(result[0], 0.0)
        # Right: zero-derivative Neumann -> right equals its neighbour
        assert jnp.isclose(result[-1], result[-2])
        # Interior untouched
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_mixed_neumann_left_dirichlet_right(self):
        """Neumann on the left edge, Dirichlet on the right edge."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_mixed(
            params,
            left_type=BoundaryType.NEUMANN,
            right_type=BoundaryType.DIRICHLET,
            right_kwargs={"boundary_value": 7.0},
        )

        assert jnp.isclose(result[0], result[1])  # zero-derivative Neumann left
        assert jnp.isclose(result[-1], 7.0)  # Dirichlet right
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_mixed_does_not_fall_back_to_dirichlet(self):
        """A Neumann edge must use the derivative rule, not a Dirichlet value."""
        params = jnp.array([10.0, 2.0, 3.0, 4.0, 20.0])
        result = apply_mixed(
            params,
            left_type=BoundaryType.NEUMANN,
            right_type=BoundaryType.NEUMANN,
        )

        # If this silently fell back to Dirichlet it would set 0.0 at the edges.
        assert result[0] != 0.0
        assert result[-1] != 0.0
        assert jnp.isclose(result[0], params[1])
        assert jnp.isclose(result[-1], params[-2])

    def test_mixed_robin_left_dirichlet_right(self):
        """Robin on the left edge, Dirichlet on the right edge."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_mixed(
            params,
            left_type=BoundaryType.ROBIN,
            right_type=BoundaryType.DIRICHLET,
            left_kwargs={"alpha": 1.0, "beta": 0.0, "gamma": 4.0},
            right_kwargs={"boundary_value": 9.0},
        )

        # Robin with beta=0 -> Dirichlet limit u = gamma/alpha = 4.0 on the left.
        assert jnp.isclose(result[0], 4.0)
        assert jnp.isclose(result[-1], 9.0)

    def test_mixed_accepts_string_types(self):
        """Per-side types may be given as strings."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_mixed(
            params,
            left_type="dirichlet",
            right_type="neumann",
            left_kwargs={"boundary_value": 2.0},
        )

        assert jnp.isclose(result[0], 2.0)
        assert jnp.isclose(result[-1], result[-2])

    def test_mixed_unknown_per_side_type_raises(self):
        """An unknown per-side BC type must raise (fail-fast, no fallback)."""
        params = jnp.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown boundary type"):
            apply_mixed(params, left_type="bogus", right_type="dirichlet")

    def test_mixed_unsupported_per_side_type_raises(self):
        """Per-side MIXED/PERIODIC are not valid single-edge constraints."""
        params = jnp.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="not a valid per-boundary"):
            apply_mixed(params, left_type=BoundaryType.MIXED, right_type=BoundaryType.DIRICHLET)
        with pytest.raises(ValueError, match="not a valid per-boundary"):
            apply_mixed(params, left_type=BoundaryType.DIRICHLET, right_type=BoundaryType.PERIODIC)


class TestApplyBoundaryConditionMixedDispatch:
    """The unified dispatcher must route MIXED to genuine per-boundary handling."""

    def test_mixed_via_enum_dispatches_to_apply_mixed(self):
        """MIXED via the unified API applies per-side constraints, not Dirichlet."""
        params = jnp.array([10.0, 2.0, 3.0, 4.0, 20.0])
        result = apply_boundary_condition(
            params,
            BoundaryType.MIXED,
            left_type=BoundaryType.DIRICHLET,
            right_type=BoundaryType.NEUMANN,
            left_kwargs={"boundary_value": 0.0},
        )

        assert jnp.isclose(result[0], 0.0)
        assert jnp.isclose(result[-1], result[-2])

    def test_mixed_via_string_dispatches_to_apply_mixed(self):
        """MIXED via string applies per-side constraints."""
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_boundary_condition(
            params,
            "mixed",
            left_type="neumann",
            right_type="dirichlet",
            right_kwargs={"boundary_value": 1.0},
        )

        assert jnp.isclose(result[0], result[1])
        assert jnp.isclose(result[-1], 1.0)

    def test_unknown_type_still_raises(self):
        """Unknown top-level type still fails fast."""
        params = jnp.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown boundary type"):
            apply_boundary_condition(params, "not_a_type")


class TestMixedJAXCompatibility:
    """jit/grad/vmap smoke tests for the mixed boundary residual function."""

    def test_mixed_jit_compatible(self):
        """apply_mixed is JIT-compatible (types are static, arrays traced)."""
        import jax

        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        @jax.jit
        def constrain(x):
            return apply_mixed(
                x,
                left_type=BoundaryType.DIRICHLET,
                right_type=BoundaryType.NEUMANN,
                left_kwargs={"boundary_value": 0.0},
            )

        result = constrain(params)
        assert jnp.isclose(result[0], 0.0)
        assert jnp.isclose(result[-1], result[-2])

    def test_mixed_vmap_compatible(self):
        """apply_mixed vmaps over a batch of parameter arrays."""
        import jax

        batch = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        def constrain(x):
            return apply_mixed(
                x,
                left_type=BoundaryType.DIRICHLET,
                right_type=BoundaryType.NEUMANN,
                left_kwargs={"boundary_value": 0.0},
            )

        result = jax.vmap(constrain)(batch)
        assert jnp.allclose(result[:, 0], 0.0)
        assert jnp.allclose(result[:, -1], result[:, -2])

    def test_mixed_grad_compatible(self):
        """apply_mixed is differentiable through the boundary projection."""
        import jax

        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def loss_fn(x):
            constrained = apply_mixed(
                x,
                left_type=BoundaryType.NEUMANN,
                right_type=BoundaryType.DIRICHLET,
                right_kwargs={"boundary_value": 0.0},
            )
            return jnp.sum(constrained**2)

        gradients = jax.grad(loss_fn)(params)
        assert gradients.shape == params.shape
        assert jnp.isfinite(gradients).all()
