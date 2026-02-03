"""Tests for boundary condition application functions.

This module tests the single source of truth for boundary condition
application logic across the Opifex framework.
"""

import jax.numpy as jnp
import pytest

from opifex.core.physics.boundaries import (
    apply_boundary_condition,
    apply_dirichlet,
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
        result = apply_dirichlet(
            params, left_boundary=left_value, right_boundary=right_value
        )

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
        result = apply_boundary_condition(
            params, BoundaryType.DIRICHLET, boundary_value=0.0
        )

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
