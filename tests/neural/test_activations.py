"""Test activation function library.

Test suite for comprehensive activation function collection following FLAX NNX patterns.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from opifex.neural.activations import (
    get_activation,
    list_activations,
    register_activation,
)


class TestActivationRegistry:
    """Test activation function registry functionality."""

    def test_list_available_activations(self):
        """Test listing all available activation functions."""
        activations = list_activations()

        # Should include standard FLAX NNX activations
        expected_standard = {
            "relu",
            "gelu",
            "silu",
            "swish",
            "tanh",
            "sigmoid",
            "elu",
            "leaky_relu",
            "softplus",
        }

        # Should include custom scientific activations
        expected_custom = {
            "mish",
            "snake",
            "gaussian",
            "normalized_tanh",
            "soft_exponential",
        }

        all_expected = expected_standard | expected_custom

        for activation in all_expected:
            assert activation in activations, f"Missing activation: {activation}"

    def test_get_activation_function(self):
        """Test retrieving activation functions by name."""
        # Test FLAX NNX activations (used directly)
        relu_fn = get_activation("relu")
        assert relu_fn is nnx.relu

        gelu_fn = get_activation("gelu")
        assert gelu_fn is nnx.gelu

        # Test custom activations
        mish_fn = get_activation("mish")
        assert callable(mish_fn)

        snake_fn = get_activation("snake")
        assert callable(snake_fn)

    def test_unknown_activation_error(self):
        """Test error handling for unknown activation functions."""
        with pytest.raises(
            ValueError, match="Unknown activation function: unknown_func"
        ):
            get_activation("unknown_func")

    def test_register_custom_activation(self):
        """Test registering custom activation functions."""

        def custom_activation(x):
            return x * 2

        register_activation("custom_double", custom_activation)

        # Should now be available
        activations = list_activations()
        assert "custom_double" in activations

        # Should be retrievable
        fn = get_activation("custom_double")
        assert fn is custom_activation


class TestStandardActivations:
    """Test standard activation functions work correctly."""

    def test_relu_function(self):
        """Test ReLU activation."""
        relu_fn = get_activation("relu")
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0])
        result = relu_fn(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_tanh_function(self):
        """Test Tanh activation."""
        tanh_fn = get_activation("tanh")
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = jnp.tanh(x)
        result = tanh_fn(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_gelu_function(self):
        """Test GELU activation."""
        gelu_fn = get_activation("gelu")
        x = jnp.array([-1.0, 0.0, 1.0])
        result = gelu_fn(x)

        # GELU should be smooth and approximately linear near 0
        assert jnp.isfinite(result).all()
        assert result[1] == pytest.approx(0.0, abs=1e-6)  # GELU(0) ≈ 0

    def test_sigmoid_function(self):
        """Test Sigmoid activation."""
        sigmoid_fn = get_activation("sigmoid")
        x = jnp.array([-5.0, 0.0, 5.0])
        result = sigmoid_fn(x)

        # Sigmoid should be bounded [0, 1]
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()
        assert result[1] == pytest.approx(0.5, abs=1e-6)  # sigmoid(0) = 0.5


class TestScientificActivations:
    """Test scientific computing specific activation functions."""

    def test_mish_activation(self):
        """Test Mish activation function."""
        mish_fn = get_activation("mish")
        x = jnp.array([-2.0, 0.0, 2.0])
        result = mish_fn(x)

        # Should be smooth and finite
        assert jnp.isfinite(result).all()
        assert result[1] == pytest.approx(0.0, abs=1e-6)  # mish(0) ≈ 0

    def test_snake_activation(self):
        """Test Snake activation function."""
        snake_fn = get_activation("snake")
        x = jnp.array([-1.0, 0.0, 1.0])
        result = snake_fn(x)

        # Should be smooth and finite
        assert jnp.isfinite(result).all()
        assert result[1] == pytest.approx(0.0, abs=1e-6)  # snake(0) ≈ 0

    def test_gaussian_activation(self):
        """Test Gaussian activation function."""
        gaussian_fn = get_activation("gaussian")
        x = jnp.array([-2.0, 0.0, 2.0])
        result = gaussian_fn(x)

        # Should be bounded [0, 1] and smooth
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()
        assert jnp.isfinite(result).all()
        assert result[1] == pytest.approx(1.0, abs=1e-6)  # gaussian(0) = 1

    def test_normalized_tanh(self):
        """Test normalized tanh activation."""
        normalized_tanh_fn = get_activation("normalized_tanh")
        x = jnp.array([-2.0, 0.0, 2.0])
        result = normalized_tanh_fn(x)

        # Should be smooth and bounded
        assert jnp.isfinite(result).all()
        assert result[1] == pytest.approx(0.0, abs=1e-6)  # Should pass through origin

    def test_soft_exponential(self):
        """Test soft exponential activation."""
        soft_exp_fn = get_activation("soft_exponential")
        x = jnp.array([-1.0, 0.0, 1.0])
        result = soft_exp_fn(x)

        # Should be smooth and finite
        assert jnp.isfinite(result).all()


class TestActivationProperties:
    """Test mathematical properties of activation functions."""

    @pytest.mark.parametrize(
        "activation_name",
        ["relu", "tanh", "gelu", "silu", "swish", "mish"],
    )
    def test_activation_differentiability(self, activation_name):
        """Test that activations are differentiable."""
        activation_fn = get_activation(activation_name)

        def test_fn(x):
            return jnp.sum(activation_fn(x))

        x = jnp.array([1.0, 2.0, 3.0])
        grad_fn = jax.grad(test_fn)
        gradients = grad_fn(x)

        assert jnp.isfinite(gradients).all()

    @pytest.mark.parametrize("activation_name", ["gaussian", "normalized_tanh", "mish"])
    def test_activation_bounds(self, activation_name):
        """Test activation function bounds where applicable."""
        activation_fn = get_activation(activation_name)
        x = jnp.linspace(-10.0, 10.0, 100)
        result = activation_fn(x)

        # All should be finite
        assert jnp.isfinite(result).all()

        # Gaussian should be in [0, 1]
        if activation_name == "gaussian":
            assert (result >= 0.0).all()
            assert (result <= 1.0).all()


class TestActivationPerformance:
    """Test activation function performance characteristics."""

    def test_batch_processing(self):
        """Test activation functions work with batched inputs."""
        relu_fn = get_activation("relu")
        batch_size = 10
        feature_dim = 32

        x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, feature_dim))
        result = relu_fn(x)

        assert result.shape == (batch_size, feature_dim)
        assert jnp.isfinite(result).all()


class TestPhysicsInformedActivations:
    """Test physics-informed properties of activation functions."""

    def test_conservation_properties(self):
        """Test that certain activations preserve important properties."""
        # Test that tanh preserves antisymmetry
        tanh_fn = get_activation("tanh")
        x = jnp.array([1.0, 2.0, 3.0])
        pos_result = tanh_fn(x)
        neg_result = tanh_fn(-x)

        np.testing.assert_array_almost_equal(pos_result, -neg_result, decimal=6)

    def test_smoothness_properties(self):
        """Test smoothness of activation functions."""
        smooth_activations = ["tanh", "gelu", "silu", "mish"]

        for name in smooth_activations:
            activation_fn = get_activation(name)

            # Test second derivative exists (smoothness indicator)
            def second_derivative(x, fn=activation_fn):
                # Apply to each element separately to handle vector inputs
                return jax.vmap(jax.grad(jax.grad(fn)))(x)

            x = jnp.array([0.0, 1.0, -1.0])
            second_deriv = second_derivative(x)
            assert jnp.isfinite(second_deriv).all()
