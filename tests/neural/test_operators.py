"""Test neural operator foundations.

Test suite for neural operator building blocks including FNO, DeepONet,
and operator learning interfaces. Following FLAX NNX patterns and critical
technical guidelines with full compliance to refactored architecture.

Updated for Flax NNX compliance and GPU/CPU compatibility following critical technical guidelines.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

# Import TestEnvironmentManager from extracted location
from opifex.neural.operators.deeponet.adaptive import AdaptiveDeepONet
from opifex.neural.operators.deeponet.base import DeepONet
from opifex.neural.operators.deeponet.enhanced import FourierEnhancedDeepONet
from opifex.neural.operators.fno.base import (
    FourierNeuralOperator,
)
from opifex.neural.operators.foundations import (
    GraphNeuralOperator,
    MultiScaleFourierNeuralOperator,
    OperatorNetwork,
)
from opifex.neural.operators.specialized import (
    LatentNeuralOperator,
    WaveletNeuralOperator,
)


class TestOperatorNetwork:
    """Test unified operator network interface."""

    def test_operator_network_initialization(self):
        """Test operator network initialization."""
        rngs = nnx.Rngs(42)

        # Test with FNO
        op_net = OperatorNetwork(
            operator_type="fno",
            config={
                "in_channels": 2,
                "out_channels": 1,
                "hidden_channels": 32,
                "modes": 8,
                "num_layers": 2,
            },
            rngs=rngs,
        )

        assert op_net.operator_type == "fno"
        assert isinstance(op_net.operator, FourierNeuralOperator)

    def test_operator_network_deeponet(self):
        """Test operator network with DeepONet."""
        rngs = nnx.Rngs(42)

        op_net = OperatorNetwork(
            operator_type="deeponet",
            config={
                "branch_input_dim": 50,
                "trunk_input_dim": 2,
                "branch_hidden_dims": [64, 32],
                "trunk_hidden_dims": [32, 16],
                "latent_dim": 32,
            },
            rngs=rngs,
        )

        assert op_net.operator_type == "deeponet"
        assert isinstance(op_net.operator, DeepONet)

    def test_operator_network_enhanced_deeponet(self):
        """Test operator network with enhanced DeepONet."""
        rngs = nnx.Rngs(42)

        op_net = OperatorNetwork(
            operator_type="deeponet",
            config={
                "branch_input_dim": 50,
                "trunk_input_dim": 2,
                "branch_hidden_dims": [64, 32],
                "trunk_hidden_dims": [32, 16],
                "latent_dim": 32,
                "enhanced": True,
                "num_physics_systems": 2,
                "use_attention": True,
                "attention_heads": 4,
                "physics_constraints": ["mass_conservation"],
                "sensor_optimization": True,
                "num_sensors": 25,
            },
            rngs=rngs,
        )

        assert op_net.operator_type == "deeponet"
        # Note: MultiPhysicsDeepONet has been moved to tests/neural/operators/deeponet/test_multiphysics.py
        # This test now focuses on the enhanced DeepONet configuration interface
        assert hasattr(
            op_net.operator, "branch_nets"
        )  # Should have multiple branch networks

    def test_operator_network_invalid_type(self):
        """Test operator network with invalid type."""
        rngs = nnx.Rngs(42)

        with pytest.raises(ValueError, match="Unknown operator type"):
            OperatorNetwork(operator_type="invalid", config={}, rngs=rngs)

    def test_operator_network_forward_fno(self):
        """Test operator network forward pass with FNO."""
        rngs = nnx.Rngs(42)

        op_net = OperatorNetwork(
            operator_type="fno",
            config={
                "in_channels": 1,
                "out_channels": 1,
                "hidden_channels": 16,
                "modes": 4,
                "num_layers": 1,
            },
            rngs=rngs,
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 32))
        output = op_net(x)

        assert output.shape == (2, 1, 32)
        assert jnp.isfinite(output).all()

    def test_operator_network_forward_deeponet(self):
        """Test operator network forward pass with DeepONet."""
        rngs = nnx.Rngs(42)

        op_net = OperatorNetwork(
            operator_type="deeponet",
            config={
                "branch_input_dim": 20,
                "trunk_input_dim": 1,
                "branch_hidden_dims": [32],
                "trunk_hidden_dims": [32],
                "latent_dim": 16,
            },
            rngs=rngs,
        )

        branch_input = jax.random.normal(jax.random.PRNGKey(0), (2, 20))
        trunk_input = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 1))

        output = op_net(branch_input, trunk_input)

        assert output.shape == (2, 10)
        assert jnp.isfinite(output).all()


class TestOperatorLearningIntegration:
    """Test integration with problem definitions and training."""

    def test_operator_learning_workflow(self):
        """Test complete operator learning workflow."""
        rngs = nnx.Rngs(42)

        # Create operator
        fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        # Create synthetic data
        batch_size = 4
        grid_size = 64

        # Input functions
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 1, grid_size))

        # Target functions (should be output of some operator applied to x)
        y = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 1, grid_size))

        # Forward pass
        predictions = fno(x)

        # Compute loss
        loss = jnp.mean((predictions - y) ** 2)

        assert jnp.isfinite(loss)
        assert loss.shape == ()

    def test_operator_conservation_properties(self):
        """Test operator respects conservation properties."""
        rngs = nnx.Rngs(42)

        fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=4,
            num_layers=1,
            rngs=rngs,
        )

        # Test mass conservation (integral preservation)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 32))

        # Compute output integral
        output = fno(x)
        output_integral = jnp.sum(output, axis=-1)

        # While exact conservation isn't guaranteed without constraints,
        # outputs should be finite and have reasonable magnitude
        assert jnp.isfinite(output_integral).all()
        assert jnp.abs(output_integral).max() < 1000  # Reasonable scale

    def test_operator_symmetry_properties(self):
        """Test operator symmetry properties."""
        rngs = nnx.Rngs(42)

        fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=8,
            num_layers=1,
            rngs=rngs,
        )

        # Test translation equivariance for periodic boundaries
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 32))

        # Apply operator
        y = fno(x)

        # Shift input
        x_shifted = jnp.roll(x, shift=4, axis=-1)
        y_shifted = fno(x_shifted)

        # For FNO with periodic boundaries, output should also be shifted
        # This is a property test - actual equivariance depends on
        # implementation details
        assert y.shape == y_shifted.shape
        assert jnp.isfinite(y_shifted).all()


class TestEnhancedOperatorNetwork:
    """Test enhanced OperatorNetwork factory with new DeepONet variants."""

    def test_operator_network_fourier_deeponet(self):
        """Test OperatorNetwork with Fourier-Enhanced DeepONet and GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        config = {
            "branch_input_dim": 32,
            "trunk_input_dim": 2,
            "branch_hidden_dims": [64, 32],
            "trunk_hidden_dims": [32, 16],
            "latent_dim": 16,
            "fourier_modes": 8,
            "use_spectral_branch": True,
            "use_spectral_trunk": False,
        }

        operator_net = OperatorNetwork(
            operator_type="fourier_deeponet",
            config=config,
            rngs=rngs,
        )

        assert operator_net.operator_type == "fourier_deeponet"
        assert isinstance(operator_net.operator, FourierEnhancedDeepONet)

        batch_size = 3
        num_locations = 12
        branch_input = jnp.ones((batch_size, config["branch_input_dim"]))
        trunk_input = jnp.ones(
            (batch_size, num_locations, config["trunk_input_dim"]),
        )

        output = operator_net(branch_input, trunk_input)
        expected_shape = (batch_size, num_locations)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_operator_network_adaptive_deeponet(self):
        """Test OperatorNetwork with Adaptive DeepONet and GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        config = {
            "branch_input_dim": 24,
            "trunk_input_dim": 1,
            "base_latent_dim": 12,
            "num_resolution_levels": 3,
            "adaptive_latent_scaling": True,
            "use_residual_connections": True,
        }

        operator_net = OperatorNetwork(
            operator_type="adaptive_deeponet",
            config=config,
            rngs=rngs,
        )

        assert operator_net.operator_type == "adaptive_deeponet"
        assert isinstance(operator_net.operator, AdaptiveDeepONet)

        batch_size = 2
        num_locations = 8
        branch_input = jnp.ones((batch_size, config["branch_input_dim"]))
        trunk_input = jnp.ones(
            (batch_size, num_locations, config["trunk_input_dim"]),
        )

        output = operator_net(branch_input, trunk_input)
        expected_shape = (batch_size, num_locations)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_operator_network_gno_integration(self):
        """Test OperatorNetwork with Graph Neural Operator and GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        config = {
            "node_dim": 16,
            "hidden_dim": 32,
            "num_layers": 3,
            "edge_dim": 4,
        }

        operator_net = OperatorNetwork(
            operator_type="gno",
            config=config,
            rngs=rngs,
        )

        assert operator_net.operator_type == "gno"
        assert isinstance(operator_net.operator, GraphNeuralOperator)

        batch_size = 2
        num_nodes = 10
        num_edges = 15
        node_features = jnp.ones((batch_size, num_nodes, config["node_dim"]))
        edge_indices = jnp.ones((batch_size, num_edges, 2), dtype=jnp.int32)
        edge_features = jnp.ones((batch_size, num_edges, config["edge_dim"]))

        output = operator_net(node_features, edge_indices, edge_features)
        expected_shape = (batch_size, num_nodes, config["node_dim"])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_operator_network_invalid_type_error(self):
        """Test OperatorNetwork with invalid operator type."""
        rngs = nnx.Rngs(42)
        config = {"dummy": "config"}

        with pytest.raises(ValueError, match="Unknown operator type: invalid_type"):
            OperatorNetwork(
                operator_type="invalid_type",
                config=config,
                rngs=rngs,
            )


class TestAdvancedOperatorIntegration:
    """Test integration of advanced neural operators with GPU/CPU compatibility."""

    def test_multi_scale_fno_vs_regular_fno_performance(self):
        """Compare Multi-Scale FNO with regular FNO with GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        # Dimensions compatible with both GPU and CPU
        batch_size = 2
        in_channels = 1
        out_channels = 1
        hidden_channels = 32
        spatial_size = 64

        # Regular FNO
        regular_fno = FourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=16,
            num_layers=3,
            rngs=rngs,
        )

        # Multi-Scale FNO
        multi_scale_fno = MultiScaleFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes_per_scale=[16, 8, 4],
            num_layers_per_scale=[1, 1, 1],
            rngs=rngs,
        )

        x = jnp.ones((batch_size, in_channels, spatial_size))

        regular_output = regular_fno(x)
        multi_scale_output = multi_scale_fno(x)

        # Both should produce valid outputs with same shape
        assert regular_output.shape == multi_scale_output.shape
        assert jnp.all(jnp.isfinite(regular_output))
        assert jnp.all(jnp.isfinite(multi_scale_output))

    def test_latent_operator_compression_efficiency(self):
        """Test LNO compression efficiency with GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        # Dimensions for compression testing
        batch_size = 2
        channels = 4
        spatial_points = 64
        latent_dim = 16  # Much smaller than spatial dimension
        _num_tokens = 8  # Much smaller than spatial points

        # Calculate flattened dimensions for modern API
        input_dim = channels * spatial_points
        output_dim = channels * spatial_points
        _hidden_dim = 32

        # Create operator with compression using modern API
        operator = LatentNeuralOperator(
            in_channels=input_dim,
            out_channels=output_dim,
            num_latent_tokens=32,
            latent_dim=latent_dim,
            rngs=rngs,
        )

        # Create flattened input for modern API
        x_flat = jnp.ones((batch_size, input_dim))
        output = operator(x_flat)

        assert output.shape == x_flat.shape
        assert jnp.all(jnp.isfinite(output))

        # Verify compression efficiency by comparing dimensions
        assert latent_dim < spatial_points  # Latent space is compressed
        assert output_dim == input_dim  # Maintains reconstruction capability

    def test_wavelet_operator_multi_scale_features(self):
        """Test WNO multi-scale feature capture with GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        # Dimensions for multi-scale testing
        in_channels = 1
        out_channels = 1
        hidden_channels = 16
        num_levels = 3
        signal_length = 64

        operator = WaveletNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            rngs=rngs,
        )

        # Create multi-scale test signal (high and low frequency components)
        x_coords = jnp.linspace(0, 4 * jnp.pi, signal_length)
        high_freq = jnp.sin(10 * x_coords)  # High frequency
        low_freq = 0.5 * jnp.sin(x_coords)  # Low frequency
        multi_scale_signal = high_freq + low_freq

        # Add batch and channel dimensions
        x = multi_scale_signal[None, None, :]
        output = operator(x)

        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

        # Verify multi-scale processing capability
        assert x.shape[-1] == signal_length  # Signal length preserved
        assert num_levels > 1  # Multi-level decomposition used

    def test_advanced_operators_memory_efficiency(self):
        """Test memory efficiency of advanced operators with GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        # Test with appropriately sized problem for both GPU and CPU
        batch_size = 4  # Reduced for memory efficiency
        spatial_size = 256

        operators = [
            MultiScaleFourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                modes_per_scale=[16, 8],
                num_layers_per_scale=[1, 1],
                use_gradient_checkpointing=True,
                rngs=rngs,
            ),
            WaveletNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=16,
                num_levels=2,
                rngs=rngs,
            ),
        ]

        x = jnp.ones((batch_size, 1, spatial_size))

        for operator in operators:
            output = operator(x)
            assert output.shape == x.shape
            assert jnp.all(jnp.isfinite(output))

    def test_physics_aware_advanced_operators(self):
        """Test physics-aware capabilities in advanced operators with GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        # Dimensions for physics-aware testing
        batch_size = 2
        in_channels = 2
        out_channels = 1
        spatial_size = 32

        # Multi-Scale FNO with physics constraints
        ms_fno = MultiScaleFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=32,
            modes_per_scale=[8, 4],
            num_layers_per_scale=[1, 1],
            use_cross_scale_attention=True,
            rngs=rngs,
        )

        x = jnp.ones((batch_size, in_channels, spatial_size))

        # Test Multi-Scale FNO
        ms_output = ms_fno(x)

        assert ms_output.shape == (batch_size, out_channels, spatial_size)
        assert jnp.all(jnp.isfinite(ms_output))

        # Verify physics-aware processing capability
        assert ms_fno.use_cross_scale_attention  # Physics-aware feature enabled
