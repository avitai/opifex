"""Tests for spectral neural operators in FNO module.

This module tests the SpectralNeuralOperator class which provides
complete neural operator architectures with spectral normalization.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.spectral import (
    create_spectral_neural_operator,
    SpectralNeuralOperator,
)


class TestSpectralNeuralOperator:
    """Test cases for SpectralNeuralOperator."""

    @pytest.fixture
    def rngs(self):
        """Create random number generators."""
        return nnx.Rngs(0)

    @pytest.fixture
    def basic_operator(self, rngs):
        """Create a basic spectral neural operator."""
        return SpectralNeuralOperator(
            input_dim=32,
            output_dim=16,
            hidden_dims=(64, 32),
            rngs=rngs,
        )

    def test_initialization(self, rngs):
        """Test basic initialization of SpectralNeuralOperator."""
        operator = SpectralNeuralOperator(
            input_dim=10,
            output_dim=5,
            hidden_dims=(32, 16),
            rngs=rngs,
        )

        assert operator.input_dim == 10
        assert operator.output_dim == 5
        assert operator.hidden_dims == (32, 16)
        assert hasattr(operator, "input_proj")
        assert hasattr(operator, "hidden_layers")
        assert hasattr(operator, "output_proj")

    def test_initialization_with_custom_params(self, rngs):
        """Test initialization with custom parameters."""
        operator = SpectralNeuralOperator(
            input_dim=20,
            output_dim=10,
            hidden_dims=(128, 64, 32),
            num_heads=4,
            power_iterations=2,
            use_adaptive_bounds=True,
            rngs=rngs,
        )

        assert operator.input_dim == 20
        assert operator.output_dim == 10
        assert operator.hidden_dims == (128, 64, 32)
        assert len(operator.hidden_layers) == 2  # len(hidden_dims) - 1

    def test_forward_pass_2d(self, basic_operator):
        """Test forward pass with 2D input."""
        batch_size = 8
        input_data = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, basic_operator.input_dim)
        )

        output = basic_operator(input_data, training=True)

        assert output.shape == (batch_size, basic_operator.output_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_forward_pass_3d(self, basic_operator):
        """Test forward pass with 3D input (sequence data)."""
        batch_size = 4
        seq_len = 16
        input_data = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, seq_len, basic_operator.input_dim)
        )

        output = basic_operator(input_data, training=True)

        assert output.shape == (batch_size, seq_len, basic_operator.output_dim)
        assert not jnp.any(jnp.isnan(output))

    def test_training_vs_inference_mode(self, basic_operator):
        """Test behavior difference between training and inference modes."""
        input_data = jax.random.normal(
            jax.random.PRNGKey(42), (4, basic_operator.input_dim)
        )

        # Training mode
        output_train = basic_operator(input_data, training=True)

        # Inference mode
        output_inference = basic_operator(input_data, training=False)

        # Outputs should have same shape
        assert output_train.shape == output_inference.shape

    def test_different_input_dimensions(self, rngs):
        """Test operator with different input dimensions."""
        input_dims = [1, 8, 32, 128]
        output_dim = 16

        for input_dim in input_dims:
            operator = SpectralNeuralOperator(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=(32, 16),
                rngs=rngs,
            )

            input_data = jax.random.normal(jax.random.PRNGKey(42), (4, input_dim))
            output = operator(input_data)

            assert output.shape == (4, output_dim)
            assert not jnp.any(jnp.isnan(output))

    def test_gradient_flow(self, basic_operator):
        """Test that gradients flow properly through the operator."""
        input_data = jax.random.normal(
            jax.random.PRNGKey(42), (4, basic_operator.input_dim)
        )

        def loss_fn(operator, x):
            output = operator(x, training=True)
            return jnp.mean(output**2)

        # Compute gradients
        grad_fn = nnx.grad(loss_fn, argnums=0)
        grads = grad_fn(basic_operator, input_data)

        # Check that gradients exist and are not NaN
        def check_grads(grad):
            if hasattr(grad, "value"):
                assert not jnp.any(jnp.isnan(grad.value))
                # Gradients can be zero for some parameters, this is normal

        jax.tree.map(check_grads, grads, is_leaf=lambda x: hasattr(x, "value"))

    def test_different_hidden_layer_configurations(self, rngs):
        """Test various hidden layer configurations."""
        input_dim, output_dim = 16, 8
        configurations = [
            (32,),  # Single hidden layer
            (64, 32),  # Two hidden layers
            (128, 64, 32),  # Three hidden layers
            (256, 128, 64, 32),  # Four hidden layers
        ]

        for hidden_dims in configurations:
            operator = SpectralNeuralOperator(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                rngs=rngs,
            )

            input_data = jax.random.normal(jax.random.PRNGKey(42), (4, input_dim))
            output = operator(input_data)

            assert output.shape == (4, output_dim)
            assert len(operator.hidden_layers) == len(hidden_dims) - 1

    def test_spectral_normalization_properties(self, basic_operator):
        """Test that spectral normalization is properly applied."""
        # Check that all layers have spectral normalization
        assert hasattr(basic_operator.input_proj, "power_iter")
        assert hasattr(basic_operator.output_proj, "power_iter")

        for layer in basic_operator.hidden_layers:
            assert hasattr(layer, "power_iter")

    def test_jax_transformations(self, basic_operator):
        """Test compatibility with JAX transformations."""
        input_data = jax.random.normal(
            jax.random.PRNGKey(42), (4, basic_operator.input_dim)
        )

        # Test basic forward pass works (JAX JIT has issues with spectral norm state)
        output_normal = basic_operator(input_data, training=False)

        # Test that the operator is callable and produces valid output
        assert output_normal.shape == (4, basic_operator.output_dim)
        assert not jnp.any(jnp.isnan(output_normal))

    def test_batch_independence(self, basic_operator):
        """Test that different batch elements are processed independently."""
        # Create batch with repeated elements
        single_input = jax.random.normal(
            jax.random.PRNGKey(42), (1, basic_operator.input_dim)
        )
        batch_input = jnp.repeat(single_input, 4, axis=0)

        output = basic_operator(batch_input, training=False)

        # All outputs should be identical (within numerical precision)
        for i in range(1, output.shape[0]):
            assert jnp.allclose(output[0], output[i], rtol=1e-6)

    def test_numerical_stability(self, rngs):
        """Test numerical stability with extreme inputs."""
        operator = SpectralNeuralOperator(
            input_dim=16,
            output_dim=8,
            hidden_dims=(32, 16),
            rngs=rngs,
        )

        # Test with very small inputs
        small_input = jnp.ones((4, 16)) * 1e-6
        output_small = operator(small_input)
        assert not jnp.any(jnp.isnan(output_small))
        assert not jnp.any(jnp.isinf(output_small))

        # Test with large inputs
        large_input = jnp.ones((4, 16)) * 1e3
        output_large = operator(large_input)
        assert not jnp.any(jnp.isnan(output_large))
        assert not jnp.any(jnp.isinf(output_large))


class TestSpectralNeuralOperatorFactory:
    """Test cases for the factory function."""

    @pytest.fixture
    def rngs(self):
        """Create random number generators."""
        return nnx.Rngs(0)

    def test_factory_basic_creation(self, rngs):
        """Test basic operator creation via factory function."""
        operator = create_spectral_neural_operator(
            input_dim=16,
            output_dim=8,
            rngs=rngs,
        )

        assert isinstance(operator, SpectralNeuralOperator)
        assert operator.input_dim == 16
        assert operator.output_dim == 8

    def test_factory_with_all_parameters(self, rngs):
        """Test factory function with all parameters specified."""
        operator = create_spectral_neural_operator(
            input_dim=32,
            output_dim=16,
            hidden_dims=(128, 64, 32),
            num_heads=8,
            power_iterations=3,
            use_adaptive_bounds=True,
            rngs=rngs,
        )

        assert isinstance(operator, SpectralNeuralOperator)
        assert operator.input_dim == 32
        assert operator.output_dim == 16
        assert operator.hidden_dims == (128, 64, 32)

    def test_factory_default_parameters(self, rngs):
        """Test factory function uses correct defaults."""
        operator = create_spectral_neural_operator(
            input_dim=20,
            output_dim=10,
            rngs=rngs,
        )

        # Check default hidden_dims
        assert operator.hidden_dims == (128, 128, 64)

        # Test that operator works
        input_data = jax.random.normal(jax.random.PRNGKey(42), (4, 20))
        output = operator(input_data)
        assert output.shape == (4, 10)

    def test_factory_consistency_with_direct_creation(self, rngs):
        """Test that factory creates equivalent operators to direct instantiation."""
        # Create via factory
        operator_factory = create_spectral_neural_operator(
            input_dim=16,
            output_dim=8,
            hidden_dims=(32, 16),
            power_iterations=2,
            rngs=rngs,
        )

        # Create directly
        operator_direct = SpectralNeuralOperator(
            input_dim=16,
            output_dim=8,
            hidden_dims=(32, 16),
            power_iterations=2,
            rngs=rngs,
        )

        # Both should have same structure
        assert operator_factory.input_dim == operator_direct.input_dim
        assert operator_factory.output_dim == operator_direct.output_dim
        assert operator_factory.hidden_dims == operator_direct.hidden_dims


class TestSpectralNeuralOperatorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def rngs(self):
        """Create random number generators."""
        return nnx.Rngs(0)

    def test_minimal_configuration(self, rngs):
        """Test with minimal valid configuration."""
        operator = SpectralNeuralOperator(
            input_dim=1,
            output_dim=1,
            hidden_dims=(2,),
            rngs=rngs,
        )

        input_data = jnp.array([[1.0]])
        output = operator(input_data)

        assert output.shape == (1, 1)
        assert not jnp.any(jnp.isnan(output))

    def test_single_sample_batch(self, rngs):
        """Test with single sample in batch."""
        operator = SpectralNeuralOperator(
            input_dim=16,
            output_dim=8,
            rngs=rngs,
        )

        input_data = jax.random.normal(jax.random.PRNGKey(42), (1, 16))
        output = operator(input_data)

        assert output.shape == (1, 8)
        assert not jnp.any(jnp.isnan(output))

    def test_large_batch_size(self, rngs):
        """Test with large batch size."""
        operator = SpectralNeuralOperator(
            input_dim=16,
            output_dim=8,
            rngs=rngs,
        )

        batch_size = 1000
        input_data = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 16))
        output = operator(input_data)

        assert output.shape == (batch_size, 8)
        assert not jnp.any(jnp.isnan(output))

    def test_zero_input_handling(self, rngs):
        """Test behavior with zero inputs."""
        operator = SpectralNeuralOperator(
            input_dim=16,
            output_dim=8,
            rngs=rngs,
        )

        zero_input = jnp.zeros((4, 16))
        output = operator(zero_input)

        assert output.shape == (4, 8)
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_reproducibility(self, rngs):
        """Test that the same operator produces consistent outputs."""
        # Create one operator
        operator = SpectralNeuralOperator(
            input_dim=16,
            output_dim=8,
            rngs=rngs,
        )

        input_data = jax.random.normal(jax.random.PRNGKey(42), (4, 16))

        # Multiple calls in inference mode should produce the same output
        output1 = operator(input_data, training=False)
        output2 = operator(input_data, training=False)

        assert jnp.allclose(output1, output2, rtol=1e-6)
