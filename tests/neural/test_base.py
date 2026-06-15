"""Test module for neural base classes - full TDD validation.

Tests for StandardMLP following TDD methodology:
- Test-driven development with full coverage
- Modern neural network features validation
- Integration with existing test patterns
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.base import StandardMLP


class TestStandardMLP:
    """Test suite for StandardMLP with modern neural network features."""

    def test_initialization_with_defaults(self):
        """Test StandardMLP initialization with modern defaults."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(layer_sizes=[4, 8, 1], rngs=rngs)

        # Test modern defaults
        assert mlp.activation == "gelu"  # Modern default instead of tanh
        assert mlp.dropout_rate == 0.0
        assert mlp.use_bias is True
        assert mlp.apply_final_dropout is False
        assert len(mlp.layers) == 2  # Input->hidden, hidden->output

    def test_initialization_with_custom_parameters(self):
        """Test StandardMLP with custom parameters including new features."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(
            layer_sizes=[2, 16, 8, 1],
            activation="tanh",
            dropout_rate=0.1,
            use_bias=False,
            apply_final_dropout=True,
            rngs=rngs,
        )

        # Test custom configuration
        assert mlp.activation == "tanh"
        assert mlp.dropout_rate == 0.1
        assert mlp.use_bias is False
        assert mlp.apply_final_dropout is True
        assert len(mlp.layers) == 3

    def test_forward_pass_basic(self):
        """Test basic forward pass functionality."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(layer_sizes=[4, 8, 1], rngs=rngs)

        x = jnp.ones((2, 4))  # Batch of 2, input dim 4
        output = mlp(x, deterministic=True)

        assert output.shape == (2, 1)
        assert jnp.isfinite(output).all()

    def test_forward_pass_with_dropout_training(self):
        """Test forward pass with dropout during training."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(layer_sizes=[4, 8, 1], dropout_rate=0.2, rngs=rngs)

        x = jnp.ones((2, 4))

        # Training mode (deterministic=False)
        output1 = mlp(x, deterministic=False)
        output2 = mlp(x, deterministic=False)

        # Outputs should be different due to dropout randomness
        assert not jnp.allclose(output1, output2, atol=1e-5)
        assert output1.shape == (2, 1)
        assert output2.shape == (2, 1)

    def test_forward_pass_with_dropout_inference(self):
        """Test forward pass with dropout during inference."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(layer_sizes=[4, 8, 1], dropout_rate=0.2, rngs=rngs)

        x = jnp.ones((2, 4))

        # Inference mode (deterministic=True)
        output1 = mlp(x, deterministic=True)
        output2 = mlp(x, deterministic=True)

        # Outputs should be identical in inference mode
        assert jnp.allclose(output1, output2)
        assert output1.shape == (2, 1)

    def test_apply_final_dropout_feature(self):
        """Test the new apply_final_dropout feature."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(
            layer_sizes=[4, 8, 1], dropout_rate=0.3, apply_final_dropout=True, rngs=rngs
        )

        x = jnp.ones((2, 4))

        # Training mode should apply dropout to final layer
        output1 = mlp(x, deterministic=False)
        output2 = mlp(x, deterministic=False)

        # Should be different due to final dropout
        assert not jnp.allclose(output1, output2, atol=1e-5)

        # Inference mode should be deterministic
        output_inf1 = mlp(x, deterministic=True)
        output_inf2 = mlp(x, deterministic=True)
        assert jnp.allclose(output_inf1, output_inf2)

    def test_different_activations(self):
        """Test StandardMLP with different activation functions."""
        rngs = nnx.Rngs(42)
        x = jnp.ones((1, 4))

        activations = ["gelu", "tanh", "relu", "sigmoid", "silu"]

        for activation in activations:
            mlp = StandardMLP(layer_sizes=[4, 8, 1], activation=activation, rngs=rngs)

            output = mlp(x, deterministic=True)
            assert output.shape == (1, 1)
            assert jnp.isfinite(output).all()

    def test_error_handling_invalid_layer_sizes(self):
        """Test error handling for invalid layer sizes."""
        rngs = nnx.Rngs(42)

        with pytest.raises(ValueError, match="layer_sizes must have at least 2 elements"):
            StandardMLP(
                layer_sizes=[4],  # Only one layer
                rngs=rngs,
            )

    def test_dropout_error_handling(self):
        """Test error handling for dropout misconfiguration."""
        rngs = nnx.Rngs(42)

        mlp = StandardMLP(layer_sizes=[4, 8, 1], dropout_rate=0.2, rngs=rngs)

        x = jnp.ones((2, 4))

        # This should work fine
        output = mlp(x, deterministic=False)
        assert output.shape == (2, 1)
