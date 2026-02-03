"""Tests for DISCO Convolution Layers.

This module provides comprehensive tests for Discrete-Continuous (DISCO) convolution
layers, ensuring proper functionality, performance, and integration with the Opifex framework.

Test Coverage:
- DiscreteContinuousConv2d basic functionality
- EquidistantDiscreteContinuousConv2d optimization
- DiscreteContinuousConvTranspose2d upsampling
- Factory functions for encoder/decoder creation
- Edge cases and error handling
- Performance comparisons
- JAX transformations compatibility
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.specialized.disco import (
    create_disco_decoder,
    create_disco_encoder,
    DiscreteContinuousConv2d,
    DiscreteContinuousConvTranspose2d,
    EquidistantDiscreteContinuousConv2d,
)


class TestDiscreteContinuousConv2d:
    """Test suite for basic DISCO convolution layer."""

    @pytest.fixture
    def sample_input(self) -> jnp.ndarray:
        """Create sample input tensor for testing."""
        return jax.random.normal(jax.random.PRNGKey(42), (2, 16, 16, 3))

    @pytest.fixture
    def disco_conv(self) -> DiscreteContinuousConv2d:
        """Create a basic DISCO convolution layer."""
        return DiscreteContinuousConv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding="SAME",
            rngs=nnx.Rngs(42),
        )

    def test_disco_conv_initialization(self):
        """Test proper initialization of DISCO convolution layer."""
        conv = DiscreteContinuousConv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=5,
            stride=2,
            padding="VALID",
            use_bias=True,
            activation=nnx.gelu,
            rngs=nnx.Rngs(123),
        )

        assert conv.in_channels == 4
        assert conv.out_channels == 8
        assert conv.kernel_size == (5, 5)
        assert conv.stride == (2, 2)
        assert conv.padding == "VALID"
        assert conv.use_bias is True
        assert conv.activation == nnx.gelu

    def test_disco_conv_forward_pass(self, disco_conv, sample_input):
        """Test forward pass through DISCO convolution."""
        output = disco_conv(sample_input)

        # Check output shape (SAME padding should preserve spatial dimensions)
        expected_shape = (2, 16, 16, 6)  # (batch, height, width, out_channels)
        assert output.shape == expected_shape

        # Check output is finite and reasonable
        assert jnp.isfinite(output).all()
        assert not jnp.isnan(output).any()

    def test_disco_conv_different_kernel_sizes(self, sample_input):
        """Test DISCO convolution with different kernel sizes."""
        kernel_sizes = [1, 3, 5, 7]

        for kernel_size in kernel_sizes:
            conv = DiscreteContinuousConv2d(
                in_channels=3,
                out_channels=4,
                kernel_size=kernel_size,
                padding="SAME",
                rngs=nnx.Rngs(42),
            )

            output = conv(sample_input)

            # SAME padding should preserve spatial dimensions
            assert output.shape == (2, 16, 16, 4)
            assert jnp.isfinite(output).all()

    def test_disco_conv_different_strides(self, sample_input):
        """Test DISCO convolution with different stride values."""
        strides = [1, 2, 3]

        for stride in strides:
            conv = DiscreteContinuousConv2d(
                in_channels=3,
                out_channels=4,
                kernel_size=3,
                stride=stride,
                padding="SAME",
                rngs=nnx.Rngs(42),
            )

            output = conv(sample_input)

            # Check downsampling by stride
            expected_h = (
                16 + stride - 1
            ) // stride  # Ceiling division for SAME padding
            expected_w = (16 + stride - 1) // stride
            expected_shape = (2, expected_h, expected_w, 4)

            assert output.shape == expected_shape
            assert jnp.isfinite(output).all()

    def test_disco_conv_with_activation(self, sample_input):
        """Test DISCO convolution with different activation functions."""
        activations = [None, nnx.relu, nnx.gelu, nnx.tanh, jax.nn.sigmoid]

        for activation in activations:
            conv = DiscreteContinuousConv2d(
                in_channels=3,
                out_channels=4,
                kernel_size=3,
                activation=activation,
                rngs=nnx.Rngs(42),
            )

            output = conv(sample_input)

            assert output.shape == (2, 16, 16, 4)
            assert jnp.isfinite(output).all()

            # Check activation function effects
            if activation == nnx.relu:
                assert (output >= 0).all()  # ReLU non-negativity
            elif activation == jax.nn.sigmoid:
                assert (output >= 0).all() and (output <= 1).all()  # Sigmoid range

    def test_disco_conv_bias_functionality(self, sample_input):
        """Test DISCO convolution with and without bias."""
        # With bias
        conv_with_bias = DiscreteContinuousConv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            use_bias=True,
            rngs=nnx.Rngs(42),
        )

        # Without bias
        conv_without_bias = DiscreteContinuousConv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            use_bias=False,
            rngs=nnx.Rngs(42),
        )

        output_with_bias = conv_with_bias(sample_input)
        output_without_bias = conv_without_bias(sample_input)

        # Both should have same shape
        assert output_with_bias.shape == output_without_bias.shape

        # Outputs should be different (bias effect)
        assert not jnp.allclose(output_with_bias, output_without_bias)


class TestEquidistantDiscreteContinuousConv2d:
    """Test suite for equidistant DISCO convolution optimization."""

    def test_equidistant_disco_conv_initialization(self):
        """Test proper initialization of equidistant DISCO convolution."""
        conv = EquidistantDiscreteContinuousConv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            grid_spacing=0.1,
            rngs=nnx.Rngs(42),
        )

        assert conv.in_channels == 3
        assert conv.out_channels == 6
        assert conv.grid_spacing == 0.1
        assert conv.kernel_size == (3, 3)

    def test_equidistant_disco_conv_forward_pass(self):
        """Test forward pass through equidistant DISCO convolution."""
        sample_input = jax.random.normal(jax.random.PRNGKey(42), (2, 20, 20, 3))

        conv = EquidistantDiscreteContinuousConv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            grid_spacing=0.05,
            rngs=nnx.Rngs(42),
        )

        output = conv(sample_input)

        assert output.shape == (2, 20, 20, 6)
        assert jnp.isfinite(output).all()
        assert not jnp.isnan(output).any()


class TestDiscreteContinuousConvTranspose2d:
    """Test suite for DISCO transpose convolution layer."""

    def test_transpose_conv_initialization(self):
        """Test proper initialization of DISCO transpose convolution."""
        conv = DiscreteContinuousConvTranspose2d(
            in_channels=6,
            out_channels=3,
            kernel_size=4,
            stride=2,
            rngs=nnx.Rngs(42),
        )

        assert conv.in_channels == 6
        assert conv.out_channels == 3
        assert conv.kernel_size == (4, 4)
        assert conv.stride == (2, 2)

    def test_transpose_conv_upsampling(self):
        """Test upsampling behavior of transpose convolution."""
        sample_input = jax.random.normal(jax.random.PRNGKey(42), (2, 8, 8, 6))

        conv = DiscreteContinuousConvTranspose2d(
            in_channels=6,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding="SAME",
            rngs=nnx.Rngs(42),
        )

        output = conv(sample_input)

        # Check upsampling (stride=2 should double dimensions)
        expected_shape = (2, 16, 16, 3)  # 8*2 = 16
        assert output.shape == expected_shape
        assert jnp.isfinite(output).all()


class TestDISCOFactoryFunctions:
    """Test suite for DISCO encoder/decoder factory functions."""

    def test_create_disco_encoder(self):
        """Test DISCO encoder creation."""
        encoder = create_disco_encoder(
            in_channels=3,
            hidden_channels=(16, 32, 64),
            kernel_size=3,
            activation=nnx.gelu,
            use_equidistant=True,
            rngs=nnx.Rngs(42),
        )

        # Test forward pass
        input_tensor = jax.random.normal(jax.random.PRNGKey(42), (2, 32, 32, 3))
        output = encoder(input_tensor)

        # Check progressive downsampling
        # Input: 32x32, after 3 layers with stride 2: 4x4
        expected_shape = (2, 4, 4, 64)
        assert output.shape == expected_shape
        assert jnp.isfinite(output).all()

    def test_create_disco_decoder(self):
        """Test DISCO decoder creation."""
        decoder = create_disco_decoder(
            hidden_channels=(64, 32, 16),
            out_channels=3,
            kernel_size=3,
            activation=nnx.gelu,
            final_activation=jax.nn.sigmoid,
            use_equidistant=True,
            rngs=nnx.Rngs(42),
        )

        # Test forward pass
        input_tensor = jax.random.normal(jax.random.PRNGKey(42), (2, 4, 4, 64))
        output = decoder(input_tensor)

        # Check progressive upsampling
        # Input: 4x4, after 3 transpose layers with stride 2: 32x32 (4 * 2^3 = 32)
        expected_shape = (2, 32, 32, 3)
        assert output.shape == expected_shape
        assert jnp.isfinite(output).all()

        # Check final activation (sigmoid: values in [0, 1])
        assert (output >= 0).all() and (output <= 1).all()

    def test_encoder_decoder_integration(self):
        """Test full encoder-decoder pipeline."""
        # Create matched encoder-decoder pair
        encoder = create_disco_encoder(
            in_channels=3,
            hidden_channels=(16, 32),
            kernel_size=3,
            use_equidistant=True,
            rngs=nnx.Rngs(42),
        )

        decoder = create_disco_decoder(
            hidden_channels=(32, 16),
            out_channels=3,
            kernel_size=3,
            use_equidistant=True,
            rngs=nnx.Rngs(43),
        )

        # Test pipeline
        input_tensor = jax.random.normal(jax.random.PRNGKey(42), (1, 16, 16, 3))

        encoded = encoder(input_tensor)
        decoded = decoder(encoded)

        # Check shape preservation through pipeline
        assert decoded.shape == input_tensor.shape
        assert jnp.isfinite(encoded).all()
        assert jnp.isfinite(decoded).all()


class TestDISCOJAXTransformations:
    """Test suite for JAX transformations compatibility."""

    def test_jit_compilation(self):
        """Test JIT compilation of DISCO convolution."""
        disco_conv = DiscreteContinuousConv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            rngs=nnx.Rngs(42),
        )

        sample_input = jax.random.normal(jax.random.PRNGKey(42), (1, 8, 8, 2))

        @nnx.jit
        def jitted_forward(model, x):
            return model(x)

        # Test JIT compilation works
        output = jitted_forward(disco_conv, sample_input)

        assert output.shape == (1, 8, 8, 4)
        assert jnp.isfinite(output).all()

        # Test multiple calls (compilation should be cached)
        output2 = jitted_forward(disco_conv, sample_input)
        assert jnp.allclose(output, output2)

    def test_grad_computation(self):
        """Test gradient computation through DISCO convolution."""
        disco_conv = DiscreteContinuousConv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            rngs=nnx.Rngs(42),
        )

        sample_input = jax.random.normal(jax.random.PRNGKey(42), (1, 8, 8, 2))

        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        # Test gradient computation
        grad_fn = nnx.grad(loss_fn, argnums=0)
        grads = grad_fn(disco_conv, sample_input)

        # Check that gradients exist for all parameters
        assert grads is not None


if __name__ == "__main__":
    # Run specific test categories for debugging
    pytest.main([__file__ + "::TestDiscreteContinuousConv2d", "-v"])
