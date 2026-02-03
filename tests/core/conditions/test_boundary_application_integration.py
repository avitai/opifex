"""Integration tests for boundary condition application.

Tests that OOP boundary condition classes can apply themselves
using the functional boundary application system internally.
"""

import jax.numpy as jnp

from opifex.core.conditions import (
    BoundaryConditionCollection,
    DirichletBC,
    NeumannBC,
    RobinBC,
)


class TestDirichletBCApplication:
    """Test Dirichlet BC can apply itself to parameters."""

    def test_apply_constant_value(self):
        """Test applying Dirichlet BC with constant value to left boundary."""
        bc = DirichletBC(boundary="left", value=0.0)
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = bc.apply(params)

        # Should have zero only at LEFT boundary
        assert result[0] == 0.0
        # Right boundary should be unchanged
        assert result[-1] == params[-1]
        # Interior should be unchanged
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_apply_nonzero_constant(self):
        """Test applying Dirichlet BC with non-zero constant value."""
        bc = DirichletBC(boundary="left", value=10.0)
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = bc.apply(params, left_boundary=10.0, right_boundary=10.0)

        # Boundaries should be set to 10.0
        assert result[0] == 10.0
        assert result[-1] == 10.0
        # Interior should be unchanged
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_apply_function_value(self):
        """Test applying Dirichlet BC with function value."""

        def boundary_func(x):
            return 10.0 * jnp.ones_like(x)

        bc = DirichletBC(boundary="left", value=boundary_func)
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = jnp.linspace(0, 1, 5)  # Spatial coordinates

        result = bc.apply(params, x=x)

        # Only LEFT boundary should be set to function value
        assert jnp.isclose(result[0], 10.0)
        # Right boundary should be unchanged
        assert jnp.isclose(result[-1], params[-1])

    def test_apply_with_weight(self):
        """Test applying Dirichlet BC with custom weight."""
        bc = DirichletBC(boundary="left", value=0.0)
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = bc.apply(params, weight=0.5)

        # Partial application: weight * boundary + (1 - weight) * original
        # Only LEFT boundary should be affected
        expected_first = 0.5 * 0.0 + 0.5 * params[0]

        assert jnp.isclose(result[0], expected_first)
        # Right boundary should be unchanged
        assert jnp.isclose(result[-1], params[-1])

    def test_apply_left_right_different(self):
        """Test applying Dirichlet BC with different left/right values."""
        bc = DirichletBC(boundary="left", value=0.0)
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Apply should support left_boundary and right_boundary kwargs
        result = bc.apply(params, left_boundary=0.0, right_boundary=10.0)

        assert result[0] == 0.0
        assert result[-1] == 10.0
        # Interior unchanged
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_apply_time_dependent(self):
        """Test applying time-dependent Dirichlet BC."""

        def time_dependent_func(x, t):
            return t * jnp.ones_like(x)

        bc = DirichletBC(
            boundary="left", value=time_dependent_func, time_dependent=True
        )
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = jnp.linspace(0, 1, 5)

        result = bc.apply(params, x=x, t=5.0)

        # At t=5.0, LEFT boundary value should be 5.0
        assert jnp.isclose(result[0], 5.0)
        # Right boundary should be unchanged
        assert jnp.isclose(result[-1], params[-1])


class TestNeumannBCApplication:
    """Test Neumann BC can apply itself to parameters."""

    def test_apply_zero_derivative(self):
        """Test applying Neumann BC with zero derivative."""
        bc = NeumannBC(boundary="wall", value=0.0)
        params = jnp.array([10.0, 2.0, 3.0, 4.0, 20.0])

        result = bc.apply(params)

        # Zero derivative: boundaries equal neighbors
        assert result[0] == params[1]
        assert result[-1] == params[-2]
        # Interior unchanged
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_apply_with_weight(self):
        """Test applying Neumann BC with custom weight."""
        bc = NeumannBC(boundary="wall", value=0.0)
        params = jnp.array([10.0, 2.0, 3.0, 4.0, 20.0])

        result = bc.apply(params, weight=0.5)

        # Partial application of zero-derivative condition
        # result[0] = weight * params[1] + (1 - weight) * params[0]
        expected_first = 0.5 * params[1] + 0.5 * params[0]
        expected_last = 0.5 * params[-2] + 0.5 * params[-1]

        assert jnp.isclose(result[0], expected_first)
        assert jnp.isclose(result[-1], expected_last)


class TestRobinBCApplication:
    """Test Robin BC can apply itself to parameters."""

    def test_apply_dirichlet_limit(self):
        """Test Robin BC reduces to Dirichlet when beta=0."""
        bc = RobinBC(boundary="left", alpha=1.0, beta=0.0, gamma=0.0)
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = bc.apply(params)

        # Should behave like Dirichlet with value=gamma/alpha=0, only on LEFT
        assert jnp.isclose(result[0], 0.0, atol=1e-6)
        # Right boundary should be unchanged
        assert jnp.isclose(result[-1], params[-1])

    def test_apply_with_coefficients(self):
        """Test Robin BC with various coefficients."""
        bc = RobinBC(boundary="left", alpha=2.0, beta=1.0, gamma=6.0)
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = bc.apply(params)

        # Robin condition: alpha*u + beta*du/dn = gamma
        # Should modify boundary values
        assert jnp.isfinite(result).all()
        # Interior should be unchanged
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_apply_with_weight(self):
        """Test Robin BC with custom weight."""
        bc = RobinBC(boundary="left", alpha=1.0, beta=0.0, gamma=0.0)
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = bc.apply(params, weight=0.5)

        # Partial application
        assert jnp.isfinite(result).all()


class TestBoundaryConditionCollectionApplication:
    """Test applying multiple boundary conditions."""

    def test_apply_all_conditions(self):
        """Test applying all boundary conditions in a collection."""
        # Use two Dirichlet BCs that won't conflict
        bc1 = DirichletBC(boundary="left", value=0.0)
        bc2 = DirichletBC(boundary="right", value=10.0)

        collection = BoundaryConditionCollection([bc1, bc2])
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Apply all BCs using the collection's method
        result = collection.apply_all(params)

        # Should have both BCs applied
        assert jnp.isfinite(result).all()
        # Both Dirichlet BCs should be applied
        assert result[0] == 0.0  # Left boundary
        assert result[-1] == 10.0  # Right boundary
        # Interior should be unchanged
        assert jnp.allclose(result[1:-1], params[1:-1])

    def test_apply_all_empty_collection(self):
        """Test applying empty boundary condition collection."""
        collection = BoundaryConditionCollection([])
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = collection.apply_all(params)

        # Should return unchanged parameters
        assert jnp.allclose(result, params)

    def test_apply_all_multiple_dirichlet(self):
        """Test applying multiple Dirichlet BCs sequentially."""
        bc1 = DirichletBC(boundary="left", value=0.0)
        bc2 = DirichletBC(boundary="right", value=10.0)

        collection = BoundaryConditionCollection([bc1, bc2])
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = collection.apply_all(params)

        # Both Dirichlet BCs should be applied
        # Note: Since both modify boundaries, last one wins or they compose
        assert jnp.isfinite(result).all()

    def test_apply_all_with_global_weight(self):
        """Test applying all BCs with global weight parameter."""
        bc1 = DirichletBC(boundary="left", value=0.0)
        bc2 = NeumannBC(boundary="right", value=0.0)

        collection = BoundaryConditionCollection([bc1, bc2])
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = collection.apply_all(params, weight=0.5)

        # Should apply both BCs with 50% strength
        assert jnp.isfinite(result).all()


class TestIntegrationWithPhysicsLosses:
    """Test integration with physics-informed loss computation."""

    def test_boundary_enforcement_in_optimization_loop(self):
        """Test that BCs can be applied in optimization loop."""
        bc = DirichletBC(boundary="left", value=0.0)

        # Simulate optimization loop
        params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        for _ in range(3):
            # Apply BC after each gradient step (simulated)
            params = bc.apply(params)

        # Only LEFT BC should be maintained
        assert params[0] == 0.0
        # Right boundary should be unchanged from original
        assert params[-1] == 5.0

    def test_multiple_bcs_in_training(self):
        """Test applying multiple BCs during training."""
        # Use Dirichlet BCs for both boundaries to avoid conflicts
        bcs = BoundaryConditionCollection(
            [
                DirichletBC(boundary="left", value=0.0),
                DirichletBC(boundary="right", value=0.0),
            ]
        )

        params = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])

        # Apply BCs (as would happen in training)
        params = bcs.apply_all(params)

        # Both Dirichlet BCs applied
        assert params[0] == 0.0
        assert params[-1] == 0.0
        # Interior should be unchanged
        assert jnp.allclose(params[1:-1], jnp.array([4.0, 3.0, 2.0]))
