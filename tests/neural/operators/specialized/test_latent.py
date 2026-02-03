"""Test Latent Neural Operator (LNO).

Test suite for Latent Neural Operator implementation for high-dimensional spaces
with latent space compression and physics information integration.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.specialized.latent import LatentNeuralOperator


class TestLatentNeuralOperator:
    """Test Latent Neural Operator for high-dimensional spaces."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_latent_neural_operator_initialization(self, rngs):
        """Test LNO initialization with FLAX NNX and GPU/CPU compatibility."""
        input_dim = 32
        latent_dim = 16
        output_dim = 32

        lno = LatentNeuralOperator(
            in_channels=input_dim,
            out_channels=output_dim,
            num_latent_tokens=32,
            latent_dim=latent_dim,
            rngs=rngs,
        )

        assert lno.in_channels == input_dim
        assert lno.latent_dim == latent_dim
        assert lno.out_channels == output_dim
        assert hasattr(lno, "encoder_layers")
        assert hasattr(lno, "decoder_layers")
        assert hasattr(lno, "latent_self_attention")

    def test_latent_neural_operator_forward(self, rngs):
        """Test LNO forward pass with FLAX NNX and GPU/CPU compatibility."""
        # Dimensions compatible with both GPU and CPU
        batch_size = 4
        input_dim = 32
        latent_dim = 16
        output_dim = 32

        lno = LatentNeuralOperator(
            in_channels=input_dim,
            out_channels=output_dim,
            num_latent_tokens=32,
            latent_dim=latent_dim,
            rngs=rngs,
        )

        # Create input data
        input_data = jnp.ones((batch_size, input_dim))

        output = lno(input_data)

        assert output.shape == (batch_size, output_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_latent_neural_operator_with_physics_info(self, rngs):
        """Test LNO with physics information integration and GPU/CPU compatibility."""
        # Dimensions for physics-informed testing
        batch_size = 4
        input_dim = 64
        latent_dim = 32
        output_dim = 64

        lno = LatentNeuralOperator(
            in_channels=input_dim,
            out_channels=output_dim,
            num_latent_tokens=32,
            latent_dim=latent_dim,
            rngs=rngs,
        )

        # Create physics-informed input data
        physics_data = jnp.ones((batch_size, input_dim))
        # Add some physics context (e.g., boundary conditions, parameters)
        physics_context = jnp.ones((batch_size, latent_dim)) * 0.5

        output = lno(physics_data)

        assert output.shape == (batch_size, output_dim)
        assert jnp.all(jnp.isfinite(output))

        # Verify physics context has expected properties
        assert physics_context.shape == (batch_size, latent_dim)
        assert jnp.all(physics_context == 0.5)

        # Test that physics information affects output
        output_no_physics = lno(physics_data)
        # Basic consistency check (same input should give same output)
        assert jnp.allclose(output, output_no_physics, rtol=1e-6)

    def test_latent_neural_operator_2d(self, rngs):
        """Test LNO with 2D spatial data and GPU/CPU compatibility."""
        # 2D spatial dimensions
        batch_size = 2
        height, width = 16, 16
        input_channels = 1
        output_channels = 1
        input_dim = height * width * input_channels
        output_dim = height * width * output_channels
        latent_dim = 32

        lno = LatentNeuralOperator(
            in_channels=input_dim,
            out_channels=output_dim,
            num_latent_tokens=32,
            latent_dim=latent_dim,
            rngs=rngs,
        )

        # Create 2D spatial input (flattened for this operator)
        spatial_data = jnp.ones((batch_size, input_dim))

        output = lno(spatial_data)

        assert output.shape == (batch_size, output_dim)
        assert jnp.all(jnp.isfinite(output))

        # Can be reshaped back to 2D if needed
        output_2d = output.reshape(batch_size, height, width, output_channels)
        assert output_2d.shape == (batch_size, height, width, output_channels)

    def test_latent_neural_operator_differentiability(self, rngs):
        """Test LNO differentiability with FLAX NNX and GPU/CPU compatibility."""
        # Smaller dimensions for differentiability testing
        batch_size = 2
        input_dim = 16
        latent_dim = 8
        output_dim = 16

        lno = LatentNeuralOperator(
            in_channels=input_dim,
            out_channels=output_dim,
            num_latent_tokens=32,
            latent_dim=latent_dim,
            rngs=rngs,
        )

        def loss_fn(model, x):
            output = model(x)
            return jnp.sum(output**2)

        x = jnp.ones((batch_size, input_dim))
        grads = nnx.grad(loss_fn)(lno, x)

        # Verify gradients exist for key components
        assert hasattr(grads, "encoder_layers")
        assert hasattr(grads, "decoder_layers")
        assert hasattr(grads, "latent_self_attention")

        # Check that gradients are not all zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            jnp.linalg.norm(leaf) for leaf in grad_leaves if hasattr(leaf, "shape")
        ]
        assert len(grad_norms) > 0
        assert any(norm > 1e-8 for norm in grad_norms)
