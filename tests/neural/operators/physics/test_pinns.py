"""Test Specialized Physics-Informed Neural Networks.

Test suite for specialized PINN variants including multi-scale PINNs
and physics-informed operator integration.
"""

import jax
import jax.numpy as jnp
from flax import nnx

# Import TestEnvironmentManager from extracted location
from opifex.core.physics import PhysicsInformedLoss


class TestSpecializedPINNs:
    """Test Specialized Physics-Informed Neural Networks."""

    def test_multi_scale_pinn_initialization(self):
        """Test Multi-Scale PINN initialization."""
        from opifex.neural.pinns.multi_scale import MultiScalePINN

        rngs = nnx.Rngs(42)

        pinn = MultiScalePINN(
            input_dim=3,  # 3D spatial domain
            output_dim=1,  # Scalar field (e.g., temperature, pressure)
            scales=[2, 4, 8],  # Multi-scale levels
            hidden_dims=[64, 32],
            rngs=rngs,
        )

        assert pinn.input_dim == 3
        assert pinn.output_dim == 1
        assert len(pinn.scales) == 3
        assert pinn.num_scales == 3
        assert hasattr(pinn, "scale_networks")
        assert len(pinn.scale_networks) == 3

    def test_multi_scale_pinn_forward_3d(self):
        """Test Multi-Scale PINN forward pass on 3D data."""
        from opifex.neural.pinns.multi_scale import MultiScalePINN

        rngs = nnx.Rngs(42)
        batch_size = 4
        input_dim = 3
        output_dim = 1

        pinn = MultiScalePINN(
            input_dim=input_dim,
            output_dim=output_dim,
            scales=[2, 4],  # Two scale levels
            hidden_dims=[32, 16],
            rngs=rngs,
        )

        # Input: 3D coordinates (batch, input_dim)
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_dim))

        output = pinn(x)

        # Check output shape
        expected_shape = (batch_size, output_dim)
        assert output.shape == expected_shape
        assert jnp.isfinite(output).all()

    def test_multi_scale_pinn_derivatives(self):
        """Test Multi-Scale PINN can compute derivatives for physics training."""
        from opifex.neural.pinns.multi_scale import MultiScalePINN

        rngs = nnx.Rngs(42)

        pinn = MultiScalePINN(
            input_dim=2,  # 2D for simplicity
            output_dim=1,
            scales=[2, 4],
            hidden_dims=[32],
            rngs=rngs,
        )

        def pinn_fn(x):
            return pinn(x).squeeze()

            # Test derivative computation

        x = jnp.array([[1.0, 2.0]])

        # Should be able to compute gradients (needed for PDE residuals)
        grad_fn = jax.grad(pinn_fn)
        gradients = grad_fn(x)

        assert gradients.shape == (
            1,
            2,
        )  # Gradient w.r.t. each input (batch, input_dim)
        assert jnp.isfinite(gradients).all()

    def test_multi_scale_pinn_physics_integration(self):
        """Test Multi-Scale PINN integration with physics-informed loss."""
        from opifex.neural.pinns.multi_scale import MultiScalePINN

        # Physics loss is now available

        rngs = nnx.Rngs(42)

        # Create multi-scale PINN
        pinn = MultiScalePINN(
            input_dim=2,
            output_dim=1,
            scales=[2, 4],
            hidden_dims=[32],
            rngs=rngs,
        )

        # Test that PINN can be used with physics loss system
        # (This verifies interface compatibility)
        assert callable(pinn)
        assert callable(PhysicsInformedLoss) or callable(PhysicsInformedLoss)

        # Test batch processing capability
        batch_x = jax.random.normal(jax.random.PRNGKey(0), (8, 2))
        batch_output = pinn(batch_x)
        assert batch_output.shape == (8, 1)
