"""Test U-Net style Fourier Neural Operator implementation.

Modern tests for UFourierNeuralOperator aligned with current API.
Focuses on proper U-Net architecture testing without legacy compatibility.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.ufno import UFourierNeuralOperator


class TestUFourierNeuralOperator:
    """Test suite for UFourierNeuralOperator with modern API."""

    @pytest.fixture
    def sample_data_2d(self):
        """Create 2D sample data for testing."""
        batch_size = 4
        height, width = 64, 64
        in_channels = 3

        return jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, in_channels, height, width)
        )

    @pytest.fixture
    def sample_data_3d(self):
        """Create 3D sample data for testing."""
        batch_size = 2
        depth, height, width = 32, 32, 32
        in_channels = 2

        return jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, in_channels, depth, height, width)
        )

    def test_ufno_initialization(self):
        """Test UFNO initialization with modern parameters."""
        model = UFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            modes=(16, 16),
            num_levels=3,
            rngs=nnx.Rngs(0),
        )

        assert hasattr(model, "lifting")
        assert hasattr(model, "projection")
        assert hasattr(model, "bottleneck_spectral")
        assert hasattr(model, "encoder_0")
        assert hasattr(model, "encoder_1")
        assert hasattr(model, "decoder_0")
        assert hasattr(model, "decoder_1")

    def test_ufno_forward_pass(self, sample_data_2d):
        """Test UFNO forward pass."""
        model = UFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            modes=(16, 16),
            num_levels=3,
            rngs=nnx.Rngs(0),
        )

        output = model(sample_data_2d)

        expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_ufno_different_levels(self, sample_data_2d):
        """Test UFNO with different numbers of levels."""
        for num_levels in [2, 3, 4]:
            model = UFourierNeuralOperator(
                in_channels=3,
                out_channels=2,
                hidden_channels=32,
                modes=(16, 16),
                num_levels=num_levels,
                rngs=nnx.Rngs(0),
            )

            output = model(sample_data_2d)

            expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_ufno_different_modes(self, sample_data_2d):
        """Test UFNO with different Fourier modes."""
        for modes in [(8, 8), (16, 16), (32, 32)]:
            model = UFourierNeuralOperator(
                in_channels=3,
                out_channels=2,
                hidden_channels=32,
                modes=modes,
                num_levels=3,
                rngs=nnx.Rngs(0),
            )

            output = model(sample_data_2d)

            expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_ufno_downsample_factors(self, sample_data_2d):
        """Test UFNO with different downsampling factors."""
        for downsample_factor in [2, 4]:
            model = UFourierNeuralOperator(
                in_channels=3,
                out_channels=2,
                hidden_channels=32,
                modes=(16, 16),
                num_levels=2,  # Use fewer levels for larger downsample factors
                downsample_factor=downsample_factor,
                rngs=nnx.Rngs(0),
            )

            output = model(sample_data_2d)

            expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_ufno_hidden_channels(self, sample_data_2d):
        """Test UFNO with different hidden channel sizes."""
        for hidden_channels in [16, 32, 64]:
            model = UFourierNeuralOperator(
                in_channels=3,
                out_channels=2,
                hidden_channels=hidden_channels,
                modes=(16, 16),
                num_levels=3,
                rngs=nnx.Rngs(0),
            )

            output = model(sample_data_2d)

            expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_ufno_activation_functions(self, sample_data_2d):
        """Test UFNO with different activation functions."""
        for activation in [nnx.gelu, nnx.relu, nnx.tanh]:
            model = UFourierNeuralOperator(
                in_channels=3,
                out_channels=2,
                hidden_channels=32,
                modes=(16, 16),
                num_levels=3,
                activation=activation,
                rngs=nnx.Rngs(0),
            )

            output = model(sample_data_2d)

            expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_ufno_3d_input(self, sample_data_3d):
        """Test UFNO with 3D input data."""
        model = UFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes=(8, 8, 8),
            num_levels=2,
            rngs=nnx.Rngs(0),
        )

        output = model(sample_data_3d)

        expected_shape = (sample_data_3d.shape[0], 1, *sample_data_3d.shape[2:])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_ufno_same_input_output_channels(self, sample_data_2d):
        """Test UFNO with same input and output channels."""
        model = UFourierNeuralOperator(
            in_channels=3,
            out_channels=3,
            hidden_channels=32,
            modes=(16, 16),
            num_levels=3,
            rngs=nnx.Rngs(0),
        )

        output = model(sample_data_2d)

        expected_shape = sample_data_2d.shape
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_ufno_single_level(self, sample_data_2d):
        """Test UFNO with single level (no actual U-Net structure)."""
        model = UFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            modes=(16, 16),
            num_levels=1,
            rngs=nnx.Rngs(0),
        )

        output = model(sample_data_2d)

        expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_ufno_gradient_computation(self, sample_data_2d):
        """Test gradient computation through UFNO."""
        model = UFourierNeuralOperator(
            in_channels=3,
            out_channels=1,
            hidden_channels=32,
            modes=(16, 16),
            num_levels=3,
            rngs=nnx.Rngs(0),
        )

        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        grads = nnx.grad(loss_fn)(model, sample_data_2d)

        # Check gradient properties
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)

    def test_ufno_jax_transformations(self, sample_data_2d):
        """Test UFNO compatibility with JAX transformations."""
        model = UFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            modes=(16, 16),
            num_levels=3,
            rngs=nnx.Rngs(0),
        )

        @jax.jit
        def jitted_forward(x):
            return model(x)

        output = jitted_forward(sample_data_2d)

        expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_ufno_memory_efficient_large_input(self):
        """Test UFNO with larger inputs for memory efficiency."""
        # Create larger input data
        batch_size = 2
        height, width = 128, 128
        in_channels = 4

        large_data = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, in_channels, height, width)
        )

        model = UFourierNeuralOperator(
            in_channels=4,
            out_channels=2,
            hidden_channels=32,
            modes=(32, 32),
            num_levels=4,
            rngs=nnx.Rngs(0),
        )

        output = model(large_data)

        expected_shape = (batch_size, 2, height, width)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_ufno_different_spatial_resolutions(self):
        """Test UFNO with different spatial resolutions."""
        resolutions = [(32, 32), (64, 64), (96, 96)]

        for height, width in resolutions:
            data = jax.random.normal(jax.random.PRNGKey(0), (2, 3, height, width))

            model = UFourierNeuralOperator(
                in_channels=3,
                out_channels=2,
                hidden_channels=32,
                modes=(16, 16),
                num_levels=3,
                rngs=nnx.Rngs(0),
            )

            output = model(data)

            expected_shape = (2, 2, height, width)
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_ufno_parameter_efficiency(self):
        """Test parameter efficiency compared to standard architectures."""
        # Create UFNO with moderate size
        model = UFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            num_levels=2,
            rngs=nnx.Rngs(0),
        )

        # Count parameters
        def count_parameters(model):
            return sum(
                jnp.prod(jnp.array(param.shape))
                for param in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
            )

        param_count = count_parameters(model)

        # Should have reasonable number of parameters
        assert param_count > 1000  # Not too small
        assert param_count < 1_000_000  # Not too large

        # Test functionality
        test_data = jax.random.normal(jax.random.PRNGKey(4), (2, 2, 32, 32))
        output = model(test_data)

        expected_shape = (2, 1, 32, 32)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))
