"""Test base DeepONet components.

Test suite for core Deep Operator Network building blocks including
DeepONet components and base functionality.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.deeponet.base import (
    AdaptiveDeepONet,
    DeepONet,
    MultiFidelityDeepONet,
)


class TestDeepONetComponents:
    """Test DeepONet component functionality."""

    def test_deeponet_branch_trunk_integration(self):
        """Test integration between branch and trunk networks in DeepONet."""
        # Updated to use rngs as keyword-only argument
        rngs = nnx.Rngs(42)

        # Create DeepONet with specific branch and trunk sizes
        deeponet = DeepONet(
            branch_sizes=[100, 128, 64, 32],  # Input: 100 sensors, output: 32
            trunk_sizes=[2, 64, 32],  # Input: 2D coordinates, output: 32
            rngs=rngs,
        )

        # Test branch and trunk outputs separately
        batch_size = 8
        branch_input = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 100))
        trunk_input = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 50, 2))

        # Test branch output extraction
        branch_output = deeponet.get_branch_output(branch_input)
        assert branch_output.shape == (batch_size, 32)
        assert jnp.isfinite(branch_output).all()

        # Test trunk output extraction
        trunk_output = deeponet.get_trunk_output(trunk_input)
        assert trunk_output.shape == (batch_size, 50, 32)
        assert jnp.isfinite(trunk_output).all()

    def test_deeponet_sizes_validation(self):
        """Test DeepONet validates branch and trunk output dimensions match."""
        # Updated to use rngs as keyword-only argument
        rngs = nnx.Rngs(42)

        # This should work - matching output dimensions
        deeponet_valid = DeepONet(
            branch_sizes=[50, 32, 16],  # Output: 16
            trunk_sizes=[2, 32, 16],  # Output: 16
            rngs=rngs,
        )
        assert deeponet_valid.output_dim == 16

        # This should raise an error - mismatched output dimensions
        with pytest.raises(
            ValueError, match=r"Branch output dim .* must match trunk output dim"
        ):
            DeepONet(
                branch_sizes=[50, 32, 16],  # Output: 16
                trunk_sizes=[2, 32, 32],  # Output: 32 (mismatch!)
                rngs=rngs,
            )

    def test_deeponet_output_activation(self):
        """Test DeepONet with output activation."""
        rngs = nnx.Rngs(42)

        # Create DeepONet with output activation
        deeponet = DeepONet(
            branch_sizes=[50, 32, 16],
            trunk_sizes=[2, 32, 16],
            output_activation="tanh",  # Add output activation
            rngs=rngs,
        )

        # Test forward pass
        branch_input = jax.random.normal(jax.random.PRNGKey(0), (4, 50))
        trunk_input = jax.random.normal(jax.random.PRNGKey(1), (4, 10, 2))

        output = deeponet(branch_input, trunk_input)
        assert output.shape == (4, 10)
        # Check that output is in tanh range [-1, 1]
        assert jnp.all(output >= -1.0) and jnp.all(output <= 1.0)

    def test_deeponet_single_batch_handling(self):
        """Test DeepONet with single batch trunk input (branch_input batch size = 1)."""
        rngs = nnx.Rngs(42)

        deeponet = DeepONet(
            branch_sizes=[30, 32, 16],
            trunk_sizes=[2, 32, 16],
            rngs=rngs,
        )

        # Single batch: branch_input shape (1, 30), trunk_input shape (20, 2)
        branch_input = jax.random.normal(jax.random.PRNGKey(0), (1, 30))
        trunk_input = jax.random.normal(jax.random.PRNGKey(1), (20, 2))  # Single batch

        output = deeponet(branch_input, trunk_input)
        assert output.shape == (20,)  # Single batch output
        assert jnp.isfinite(output).all()

    def test_deeponet_batch_size_validation(self):
        """Test DeepONet validates batch sizes match."""
        rngs = nnx.Rngs(42)

        deeponet = DeepONet(
            branch_sizes=[30, 32, 16],
            trunk_sizes=[2, 32, 16],
            rngs=rngs,
        )

        # Mismatched batch sizes
        branch_input = jax.random.normal(jax.random.PRNGKey(0), (4, 30))
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (6, 20, 2)
        )  # Different batch size

        with pytest.raises(ValueError, match=r"Branch input batch size.*must match"):
            deeponet(branch_input, trunk_input)

    def test_deeponet_trunk_output_reshaping(self):
        """Test trunk output reshaping for different input shapes."""
        rngs = nnx.Rngs(42)

        deeponet = DeepONet(
            branch_sizes=[20, 32, 16],
            trunk_sizes=[2, 32, 16],
            rngs=rngs,
        )

        # Test single batch trunk input
        trunk_input_single = jax.random.normal(jax.random.PRNGKey(0), (10, 2))
        trunk_output_single = deeponet.get_trunk_output(trunk_input_single)
        assert trunk_output_single.shape == (10, 16)

        # Test multi-batch trunk input
        trunk_input_multi = jax.random.normal(jax.random.PRNGKey(1), (3, 10, 2))
        trunk_output_multi = deeponet.get_trunk_output(trunk_input_multi)
        assert trunk_output_multi.shape == (3, 10, 16)


class TestDeepONet:
    """Test Deep Operator Network."""

    def setup_method(self):
        """Setup for each test method."""
        self.backend = jax.default_backend()
        print(f"Running DeepONet tests on {self.backend}")

    def test_deeponet_initialization(self):
        """Test DeepONet initialization with modern API."""
        rngs = nnx.Rngs(42)

        deeponet = DeepONet(
            branch_sizes=[100, 128, 64, 64],  # [n_sensors, hidden1, hidden2, output]
            trunk_sizes=[2, 64, 32, 64],  # [coord_dim, hidden1, hidden2, output]
            rngs=rngs,
        )

        # Note: The refactored DeepONet may not expose these attributes directly
        # Testing the existence of core components instead
        assert hasattr(deeponet, "branch_net")
        assert hasattr(deeponet, "trunk_net")

    def test_deeponet_forward(self):
        """Test DeepONet forward pass with modern API."""
        rngs = nnx.Rngs(42)
        batch_size = 8
        branch_input_size = 50  # Number of sensors/measurement points
        trunk_input_dim = 2  # Coordinate dimensions
        num_locations = 32

        deeponet = DeepONet(
            branch_sizes=[
                branch_input_size,
                64,
                32,
                32,
            ],  # [sensors, hidden1, hidden2, output]
            trunk_sizes=[
                trunk_input_dim,
                32,
                16,
                32,
            ],  # [coord_dim, hidden1, hidden2, output]
            rngs=rngs,
        )

        # Branch input: function values at sensor locations
        branch_input = jax.random.normal(
            jax.random.PRNGKey(0),
            (batch_size, branch_input_size),
        )

        # Trunk input: query locations
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1),
            (batch_size, num_locations, trunk_input_dim),
        )

        # Modern API: deterministic parameter instead of rngs
        output = deeponet(branch_input, trunk_input)

        # Output: function values at query locations
        expected_shape = (batch_size, num_locations)
        assert output.shape == expected_shape
        assert jnp.isfinite(output).all()

    def test_deeponet_different_locations(self):
        """Test DeepONet with different number of query locations."""
        rngs = nnx.Rngs(42)

        deeponet = DeepONet(
            branch_sizes=[25, 32, 16],  # [sensors, hidden, output]
            trunk_sizes=[1, 32, 16],  # [coord_dim, hidden, output]
            rngs=rngs,
        )

        branch_input = jax.random.normal(jax.random.PRNGKey(0), (4, 25))

        # Test with different numbers of query locations
        for num_locs in [10, 20, 50]:
            trunk_input = jax.random.normal(jax.random.PRNGKey(1), (4, num_locs, 1))
            output = deeponet(branch_input, trunk_input)
            assert output.shape == (4, num_locs)

    def test_deeponet_differentiability(self):
        """Test DeepONet is differentiable."""
        rngs = nnx.Rngs(42)

        deeponet = DeepONet(
            branch_sizes=[20, 32, 16],  # [sensors, hidden, output]
            trunk_sizes=[2, 32, 16],  # [coord_dim, hidden, output]
            rngs=rngs,
        )

        def loss_fn(model, branch_input, trunk_input):
            return jnp.mean(model(branch_input, trunk_input) ** 2)

        branch_input = jax.random.normal(jax.random.PRNGKey(0), (2, 20))
        trunk_input = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 2))

        # Should not raise error
        grads = nnx.grad(loss_fn)(deeponet, branch_input, trunk_input)
        assert grads is not None


class TestAdaptiveDeepONet:
    """Test Adaptive DeepONet with learned sensor placement."""

    def setup_method(self):
        """Setup for each test method."""
        self.backend = jax.default_backend()
        print(f"Running AdaptiveDeepONet tests on {self.backend}")

    def test_adaptive_deeponet_initialization_uniform(self):
        """Test AdaptiveDeepONet initialization with uniform sensor init."""
        rngs = nnx.Rngs(42)

        adaptive_deeponet = AdaptiveDeepONet(
            branch_sizes=[50, 64, 32],  # 50 sensors
            trunk_sizes=[2, 64, 32],  # 2D coordinates
            sensor_dim=2,  # 2D sensor locations
            sensor_init="uniform",
            rngs=rngs,
        )

        assert hasattr(adaptive_deeponet, "sensor_locations")
        assert hasattr(adaptive_deeponet, "deeponet")
        assert adaptive_deeponet.n_sensors == 50
        assert adaptive_deeponet.sensor_dim == 2

        # Check sensor locations are in [-1, 1] range for uniform init
        sensor_locs = adaptive_deeponet.get_sensor_locations()
        assert sensor_locs.shape == (50, 2)
        assert jnp.all(sensor_locs >= -1.0) and jnp.all(sensor_locs <= 1.0)

    def test_adaptive_deeponet_initialization_normal(self):
        """Test AdaptiveDeepONet initialization with normal sensor init."""
        rngs = nnx.Rngs(42)

        adaptive_deeponet = AdaptiveDeepONet(
            branch_sizes=[30, 64, 32],
            trunk_sizes=[2, 64, 32],
            sensor_dim=2,
            sensor_init="normal",
            rngs=rngs,
        )

        sensor_locs = adaptive_deeponet.get_sensor_locations()
        assert sensor_locs.shape == (30, 2)
        # Normal distribution should have reasonable values
        assert jnp.isfinite(sensor_locs).all()

    def test_adaptive_deeponet_invalid_sensor_init(self):
        """Test AdaptiveDeepONet raises error for invalid sensor_init."""
        rngs = nnx.Rngs(42)

        with pytest.raises(ValueError, match="Unknown sensor_init"):
            AdaptiveDeepONet(
                branch_sizes=[30, 64, 32],
                trunk_sizes=[2, 64, 32],
                sensor_dim=2,
                sensor_init="invalid",
                rngs=rngs,
            )

    def test_adaptive_deeponet_forward_single_batch(self):
        """Test AdaptiveDeepONet forward pass with single batch."""
        rngs = nnx.Rngs(42)

        adaptive_deeponet = AdaptiveDeepONet(
            branch_sizes=[20, 64, 32],
            trunk_sizes=[2, 64, 32],
            sensor_dim=2,
            rngs=rngs,
        )

        # Define a simple test function
        def test_function(x):
            return jnp.sin(x[0]) + jnp.cos(x[1])

        # Single batch trunk input
        trunk_input = jax.random.normal(jax.random.PRNGKey(0), (10, 2))

        output = adaptive_deeponet(test_function, trunk_input)
        assert output.shape == (10,)
        assert jnp.isfinite(output).all()

    def test_adaptive_deeponet_forward_multi_batch(self):
        """Test AdaptiveDeepONet forward pass with multiple batches."""
        rngs = nnx.Rngs(42)

        adaptive_deeponet = AdaptiveDeepONet(
            branch_sizes=[15, 64, 32],
            trunk_sizes=[2, 64, 32],
            sensor_dim=2,
            rngs=rngs,
        )

        # Define a simple test function
        def test_function(x):
            return jnp.sin(x[0]) + jnp.cos(x[1])

        # Multiple batch trunk input
        trunk_input = jax.random.normal(jax.random.PRNGKey(0), (4, 10, 2))

        output = adaptive_deeponet(test_function, trunk_input)
        assert output.shape == (4, 10)
        assert jnp.isfinite(output).all()

    def test_adaptive_deeponet_sensor_locations(self):
        """Test AdaptiveDeepONet sensor location access."""
        rngs = nnx.Rngs(42)

        adaptive_deeponet = AdaptiveDeepONet(
            branch_sizes=[25, 64, 32],
            trunk_sizes=[2, 64, 32],
            sensor_dim=2,
            rngs=rngs,
        )

        sensor_locs = adaptive_deeponet.get_sensor_locations()
        assert sensor_locs.shape == (25, 2)
        assert jnp.isfinite(sensor_locs).all()


class TestMultiFidelityDeepONet:
    """Test Multi-fidelity DeepONet."""

    def setup_method(self):
        """Setup for each test method."""
        self.backend = jax.default_backend()
        print(f"Running MultiFidelityDeepONet tests on {self.backend}")

    def test_multifidelity_deeponet_initialization_linear(self):
        """Test MultiFidelityDeepONet initialization with linear fusion."""
        rngs = nnx.Rngs(42)

        multifidelity_deeponet = MultiFidelityDeepONet(
            branch_sizes=[30, 64, 32],
            trunk_sizes=[2, 64, 32],
            n_fidelities=3,
            fusion_strategy="linear",
            rngs=rngs,
        )

        assert hasattr(multifidelity_deeponet, "fidelity_nets")
        assert hasattr(multifidelity_deeponet, "fusion_weights")
        assert len(multifidelity_deeponet.fidelity_nets) == 3
        assert multifidelity_deeponet.n_fidelities == 3
        assert multifidelity_deeponet.fusion_strategy == "linear"

        # Check fusion weights are initialized properly
        fusion_weights = multifidelity_deeponet.get_fusion_weights()
        assert fusion_weights is not None
        assert fusion_weights.shape == (3,)
        assert jnp.allclose(jnp.sum(fusion_weights), 1.0)

    def test_multifidelity_deeponet_initialization_nonlinear(self):
        """Test MultiFidelityDeepONet initialization with nonlinear fusion."""
        rngs = nnx.Rngs(42)

        multifidelity_deeponet = MultiFidelityDeepONet(
            branch_sizes=[30, 64, 32],
            trunk_sizes=[2, 64, 32],
            n_fidelities=2,
            fusion_strategy="nonlinear",
            rngs=rngs,
        )

        assert hasattr(multifidelity_deeponet, "fidelity_nets")
        assert hasattr(multifidelity_deeponet, "fusion_net")
        assert len(multifidelity_deeponet.fidelity_nets) == 2
        assert multifidelity_deeponet.fusion_strategy == "nonlinear"

        # Nonlinear fusion should return None for fusion weights
        fusion_weights = multifidelity_deeponet.get_fusion_weights()
        assert fusion_weights is None

    def test_multifidelity_deeponet_invalid_fusion_strategy(self):
        """Test MultiFidelityDeepONet raises error for invalid fusion strategy."""
        rngs = nnx.Rngs(42)

        with pytest.raises(ValueError, match="Unknown fusion_strategy"):
            MultiFidelityDeepONet(
                branch_sizes=[30, 64, 32],
                trunk_sizes=[2, 64, 32],
                n_fidelities=2,
                fusion_strategy="invalid",
                rngs=rngs,
            )

    def test_multifidelity_deeponet_forward_linear(self):
        """Test MultiFidelityDeepONet forward pass with linear fusion."""
        rngs = nnx.Rngs(42)

        multifidelity_deeponet = MultiFidelityDeepONet(
            branch_sizes=[20, 64, 32],
            trunk_sizes=[2, 64, 32],
            n_fidelities=2,
            fusion_strategy="linear",
            rngs=rngs,
        )

        # Create branch inputs for each fidelity
        branch_inputs = [
            jax.random.normal(jax.random.PRNGKey(0), (4, 20)),  # Low fidelity
            jax.random.normal(jax.random.PRNGKey(1), (4, 20)),  # High fidelity
        ]

        trunk_input = jax.random.normal(jax.random.PRNGKey(2), (4, 10, 2))

        output = multifidelity_deeponet(branch_inputs, trunk_input)
        assert output.shape == (4, 10)
        assert jnp.isfinite(output).all()

    def test_multifidelity_deeponet_forward_nonlinear(self):
        """Test MultiFidelityDeepONet forward pass with nonlinear fusion."""
        rngs = nnx.Rngs(42)

        multifidelity_deeponet = MultiFidelityDeepONet(
            branch_sizes=[20, 64, 32],
            trunk_sizes=[2, 64, 32],
            n_fidelities=2,
            fusion_strategy="nonlinear",
            rngs=rngs,
        )

        # Create branch inputs for each fidelity
        branch_inputs = [
            jax.random.normal(jax.random.PRNGKey(0), (4, 20)),  # Low fidelity
            jax.random.normal(jax.random.PRNGKey(1), (4, 20)),  # High fidelity
        ]

        trunk_input = jax.random.normal(jax.random.PRNGKey(2), (4, 10, 2))

        output = multifidelity_deeponet(branch_inputs, trunk_input)
        assert output.shape == (4, 10)
        assert jnp.isfinite(output).all()

    def test_multifidelity_deeponet_branch_input_validation(self):
        """Test MultiFidelityDeepONet validates correct number of branch inputs."""
        rngs = nnx.Rngs(42)

        multifidelity_deeponet = MultiFidelityDeepONet(
            branch_sizes=[20, 64, 32],
            trunk_sizes=[2, 64, 32],
            n_fidelities=3,
            fusion_strategy="linear",
            rngs=rngs,
        )

        # Wrong number of branch inputs
        branch_inputs = [
            jax.random.normal(jax.random.PRNGKey(0), (4, 20)),  # Only 2 inputs
            jax.random.normal(jax.random.PRNGKey(1), (4, 20)),  # Expected 3
        ]

        trunk_input = jax.random.normal(jax.random.PRNGKey(2), (4, 10, 2))

        with pytest.raises(ValueError, match="Expected 3 branch inputs, got 2"):
            multifidelity_deeponet(branch_inputs, trunk_input)

    def test_multifidelity_deeponet_fusion_weights_access(self):
        """Test MultiFidelityDeepONet fusion weights access."""
        rngs = nnx.Rngs(42)

        # Test linear fusion
        linear_deeponet = MultiFidelityDeepONet(
            branch_sizes=[20, 64, 32],
            trunk_sizes=[2, 64, 32],
            n_fidelities=2,
            fusion_strategy="linear",
            rngs=rngs,
        )

        fusion_weights = linear_deeponet.get_fusion_weights()
        assert fusion_weights is not None
        assert fusion_weights.shape == (2,)

        # Test nonlinear fusion
        nonlinear_deeponet = MultiFidelityDeepONet(
            branch_sizes=[20, 64, 32],
            trunk_sizes=[2, 64, 32],
            n_fidelities=2,
            fusion_strategy="nonlinear",
            rngs=rngs,
        )

        fusion_weights = nonlinear_deeponet.get_fusion_weights()
        assert fusion_weights is None
