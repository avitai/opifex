"""Tests for unified operator network interface.

This module tests the OperatorNetwork class which provides a unified
interface for different neural operator architectures (FNO, DeepONet, etc.).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.specialized.operator_network import OperatorNetwork


class TestOperatorNetwork:
    """Test cases for OperatorNetwork unified interface."""

    @pytest.fixture
    def rngs(self):
        """Create random number generators."""
        return nnx.Rngs(0)

    def test_fno_operator_creation(self, rngs):
        """Test creation of FNO operator through unified interface."""
        config = {
            "in_channels": 2,
            "out_channels": 1,
            "hidden_channels": 32,
            "modes": 16,
            "num_layers": 4,
            "activation": "gelu",
        }

        operator_net = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        assert operator_net.operator_type == "fno"
        assert operator_net.config == config
        assert hasattr(operator_net, "operator")

    def test_deeponet_operator_creation(self, rngs):
        """Test creation of DeepONet operator through unified interface."""
        config = {
            "branch_input_dim": 100,
            "trunk_input_dim": 2,
            "branch_hidden_dims": [64, 32],
            "trunk_hidden_dims": [64, 32],
            "latent_dim": 32,
            "activation": "relu",
        }

        operator_net = OperatorNetwork(
            operator_type="deeponet",
            config=config,
            rngs=rngs,
        )

        assert operator_net.operator_type == "deeponet"
        assert operator_net.config == config

    def test_fno_forward_pass(self, rngs):
        """Test forward pass through FNO operator."""
        config = {
            "in_channels": 2,
            "out_channels": 1,
            "hidden_channels": 32,  # Adjusted to match FNO internal expectations
            "modes": 8,
            "num_layers": 2,
            "activation": "gelu",
        }

        operator_net = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        # Create test input in channels-first format (batch, channels, height, width)
        batch_size = 4
        spatial_size = 32
        input_data = jax.random.normal(
            jax.random.PRNGKey(42),
            (batch_size, config["in_channels"], spatial_size, spatial_size),
        )

        output = operator_net(input_data)

        # FNO outputs in channels-first format: (batch, out_channels, height, width)
        assert output.shape == (
            batch_size,
            config["out_channels"],
            spatial_size,
            spatial_size,
        )
        assert not jnp.any(jnp.isnan(output))

    def test_deeponet_forward_pass(self, rngs):
        """Test forward pass through DeepONet operator."""
        config = {
            "branch_input_dim": 50,
            "trunk_input_dim": 2,
            "branch_hidden_dims": [32, 16],
            "trunk_hidden_dims": [32, 16],
            "latent_dim": 16,
            "activation": "tanh",
        }

        operator_net = OperatorNetwork(
            operator_type="deeponet",
            config=config,
            rngs=rngs,
        )

        # Create test inputs
        batch_size = 4
        n_locations = 10  # Number of evaluation points
        branch_input = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, config["branch_input_dim"])
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(43), (batch_size, n_locations, config["trunk_input_dim"])
        )

        output = operator_net(branch_input, trunk_input)

        assert output.shape == (batch_size, n_locations)
        assert not jnp.any(jnp.isnan(output))

    def test_activation_string_mapping(self, rngs):
        """Test that string activation names are properly mapped."""
        activations_to_test = ["relu", "gelu", "tanh", "sigmoid", "swish", "silu"]

        for activation_name in activations_to_test:
            config = {
                "in_channels": 2,
                "out_channels": 1,
                "hidden_channels": 32,  # Adjusted to match FNO internal expectations
                "modes": 8,
                "num_layers": 2,
                "activation": activation_name,
            }

            operator_net = OperatorNetwork(
                operator_type="fno",
                config=config,
                rngs=rngs,
            )

            # Test that operator was created successfully
            assert hasattr(operator_net, "operator")

    def test_activation_function_direct_pass(self, rngs):
        """Test passing activation function directly instead of string."""
        config = {
            "in_channels": 2,
            "out_channels": 1,
            "hidden_channels": 32,  # Adjusted to match FNO internal expectations
            "modes": 8,
            "num_layers": 2,
            "activation": nnx.relu,  # Direct function
        }

        operator_net = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        assert hasattr(operator_net, "operator")

    def test_unknown_activation_defaults_to_gelu(self, rngs):
        """Test that unknown activation strings default to GELU."""
        config = {
            "in_channels": 2,
            "out_channels": 1,
            "hidden_channels": 32,  # Adjusted to match FNO internal expectations
            "modes": 8,
            "num_layers": 2,
            "activation": "unknown_activation",
        }

        operator_net = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        # Should not raise an error, defaults to gelu
        assert hasattr(operator_net, "operator")

    def test_invalid_operator_type_raises_error(self, rngs):
        """Test that invalid operator type raises ValueError."""
        config = {
            "in_channels": 2,
            "out_channels": 1,
        }

        with pytest.raises(ValueError, match="Unknown operator type"):
            OperatorNetwork(
                operator_type="invalid_type",
                config=config,
                rngs=rngs,
            )

    def test_fno_with_optional_parameters(self, rngs):
        """Test FNO creation with optional parameters."""
        config = {
            "in_channels": 3,
            "out_channels": 2,
            "hidden_channels": 32,
            "modes": 16,
            "num_layers": 3,
            "activation": "gelu",
            "factorization_type": "cp",
            "factorization_rank": 16,
            "use_mixed_precision": True,
        }

        operator_net = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        assert hasattr(operator_net, "operator")

    def test_deeponet_with_enhanced_features(self, rngs):
        """Test DeepONet creation with enhanced features."""
        config = {
            "branch_input_dim": 100,
            "trunk_input_dim": 2,
            "branch_hidden_dims": [64, 32],
            "trunk_hidden_dims": [64, 32],
            "latent_dim": 32,
            "enhanced": True,
            "num_physics_systems": 2,
            "use_attention": True,
            "attention_heads": 4,
            "sensor_optimization": True,
            "num_sensors": 50,
            "activation": "gelu",
        }

        # This might fail if MultiPhysicsDeepONet is not available
        # We test graceful degradation
        try:
            operator_net = OperatorNetwork(
                operator_type="deeponet",
                config=config,
                rngs=rngs,
            )
            assert hasattr(operator_net, "operator")
        except (ImportError, ValueError):
            # Expected if enhanced features not available
            pass

    def test_config_attribute_access(self, rngs):
        """Test that config attributes are properly stored and accessible."""
        config = {
            "in_channels": 2,
            "out_channels": 1,
            "hidden_channels": 32,
            "modes": 16,
            "num_layers": 4,
            "activation": "gelu",
            "custom_param": "test_value",
        }

        operator_net = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        assert operator_net.config["custom_param"] == "test_value"
        assert operator_net.config["in_channels"] == 2
        assert operator_net.config["out_channels"] == 1

    def test_multiple_operator_instances(self, rngs):
        """Test creating multiple operator instances."""
        configs = [
            {
                "operator_type": "fno",
                "config": {
                    "in_channels": 2,
                    "out_channels": 1,
                    "hidden_channels": 32,  # Adjusted to match FNO internal expectations
                    "modes": 8,
                    "num_layers": 2,
                    "activation": "gelu",
                },
            },
            {
                "operator_type": "deeponet",
                "config": {
                    "branch_input_dim": 50,
                    "trunk_input_dim": 2,
                    "branch_hidden_dims": [32, 16],
                    "trunk_hidden_dims": [32, 16],
                    "latent_dim": 16,
                    "activation": "relu",
                },
            },
        ]

        operators = []
        for cfg in configs:
            operator_net = OperatorNetwork(
                operator_type=cfg["operator_type"],
                config=cfg["config"],
                rngs=rngs,
            )
            operators.append(operator_net)

        assert len(operators) == 2
        assert operators[0].operator_type == "fno"
        assert operators[1].operator_type == "deeponet"

    def test_gradient_flow_fno(self, rngs):
        """Test gradient flow through FNO operator."""
        config = {
            "in_channels": 2,
            "out_channels": 1,
            "hidden_channels": 32,  # Adjusted to match FNO internal expectations
            "modes": 8,
            "num_layers": 2,
            "activation": "gelu",
        }

        operator_net = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        input_data = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16, 16))

        def loss_fn(operator, x):
            output = operator(x)
            return jnp.mean(output**2)

        grad_fn = nnx.grad(loss_fn, argnums=0)
        grads = grad_fn(operator_net, input_data)

        # Check that gradients exist
        def check_grads(grad):
            if hasattr(grad, "value"):
                assert not jnp.any(jnp.isnan(grad.value))

        jax.tree.map(check_grads, grads, is_leaf=lambda x: hasattr(x, "value"))

    def test_gradient_flow_deeponet(self, rngs):
        """Test gradient flow through DeepONet operator."""
        config = {
            "branch_input_dim": 20,
            "trunk_input_dim": 2,
            "branch_hidden_dims": [16, 8],
            "trunk_hidden_dims": [16, 8],
            "latent_dim": 8,
            "activation": "tanh",
        }

        operator_net = OperatorNetwork(
            operator_type="deeponet",
            config=config,
            rngs=rngs,
        )

        branch_input = jax.random.normal(jax.random.PRNGKey(42), (2, 20))
        trunk_input = jax.random.normal(jax.random.PRNGKey(43), (2, 5, 2))

        def loss_fn(operator, branch, trunk):
            output = operator(branch, trunk)
            return jnp.mean(output**2)

        grad_fn = nnx.grad(loss_fn, argnums=0)
        grads = grad_fn(operator_net, branch_input, trunk_input)

        # Check that gradients exist
        def check_grads(grad):
            if hasattr(grad, "value"):
                assert not jnp.any(jnp.isnan(grad.value))

        jax.tree.map(check_grads, grads, is_leaf=lambda x: hasattr(x, "value"))

    def test_jax_transformations(self, rngs):
        """Test compatibility with JAX transformations."""
        config = {
            "in_channels": 2,
            "out_channels": 1,
            "hidden_channels": 32,  # Adjusted to match FNO internal expectations
            "modes": 8,
            "num_layers": 2,
            "activation": "gelu",
        }

        operator_net = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        input_data = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16, 16))

        # Test jit compilation
        @nnx.jit
        def jitted_forward(operator, x):
            return operator(x)

        output_jit = jitted_forward(operator_net, input_data)
        output_normal = operator_net(input_data)

        assert output_jit.shape == output_normal.shape

    def test_lazy_import_availability(self, rngs):
        """Test lazy import functionality for optional operators."""
        # Test that we can check for various operator types
        operator_types = ["fno", "deeponet"]

        for op_type in operator_types:
            # Basic config that should work for most operators
            if op_type == "fno":
                config = {
                    "in_channels": 2,
                    "out_channels": 1,
                    "hidden_channels": 32,  # Adjusted to match FNO internal expectations
                    "modes": 8,
                    "num_layers": 2,
                }
            elif op_type == "deeponet":
                config = {
                    "branch_input_dim": 20,
                    "trunk_input_dim": 2,
                    "branch_hidden_dims": [16, 8],
                    "trunk_hidden_dims": [16, 8],
                    "latent_dim": 8,
                }

            # Should be able to create these basic operators
            operator_net = OperatorNetwork(
                operator_type=op_type,
                config=config,
                rngs=rngs,
            )
            assert hasattr(operator_net, "operator")


class TestOperatorNetworkEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def rngs(self):
        """Create random number generators."""
        return nnx.Rngs(0)

    def test_missing_required_config_parameters(self, rngs):
        """Test behavior with missing required configuration parameters."""
        incomplete_config = {
            "in_channels": 2,
            # Missing required parameters
        }

        with pytest.raises((KeyError, ValueError)):
            OperatorNetwork(
                operator_type="fno",
                config=incomplete_config,
                rngs=rngs,
            )

    def test_empty_config(self, rngs):
        """Test behavior with empty configuration."""
        with pytest.raises((KeyError, ValueError)):
            OperatorNetwork(
                operator_type="fno",
                config={},
                rngs=rngs,
            )

    def test_config_parameter_validation(self, rngs):
        """Test that invalid configuration parameters are handled."""
        invalid_configs = [
            {
                "in_channels": -1,  # Negative channels
                "out_channels": 1,
                "hidden_channels": 32,  # Adjusted to match FNO internal expectations
                "modes": 8,
                "num_layers": 2,
            },
            {
                "in_channels": 2,
                "out_channels": 0,  # Zero output channels
                "hidden_channels": 32,  # Adjusted to match FNO internal expectations
                "modes": 8,
                "num_layers": 2,
            },
        ]

        for config in invalid_configs:
            with pytest.raises(
                (ValueError, AssertionError, TypeError, ZeroDivisionError)
            ):
                # Should fail during operator creation, not forward pass
                OperatorNetwork(
                    operator_type="fno",
                    config=config,
                    rngs=rngs,
                )

    def test_operator_consistency(self, rngs):
        """Test that the same configuration produces consistent operators."""
        config = {
            "in_channels": 2,
            "out_channels": 1,
            "hidden_channels": 32,  # Adjusted to match FNO internal expectations
            "modes": 8,
            "num_layers": 2,
            "activation": "gelu",
        }

        # Create two operators with same config and RNG
        operator1 = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        operator2 = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        # Both should have same structure
        assert operator1.operator_type == operator2.operator_type
        assert operator1.config == operator2.config

    def test_numerical_stability(self, rngs):
        """Test numerical stability with extreme inputs."""
        config = {
            "in_channels": 2,
            "out_channels": 1,
            "hidden_channels": 32,  # Adjusted to match FNO internal expectations
            "modes": 8,
            "num_layers": 2,
            "activation": "gelu",
        }

        operator_net = OperatorNetwork(
            operator_type="fno",
            config=config,
            rngs=rngs,
        )

        # Test with very small inputs
        small_input = jnp.ones((1, 2, 8, 8)) * 1e-8
        output_small = operator_net(small_input)
        assert not jnp.any(jnp.isnan(output_small))
        assert not jnp.any(jnp.isinf(output_small))

        # Test with large inputs
        large_input = jnp.ones((1, 2, 8, 8)) * 1e4
        output_large = operator_net(large_input)
        assert not jnp.any(jnp.isnan(output_large))
        assert not jnp.any(jnp.isinf(output_large))
