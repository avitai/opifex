"""Test multi-physics enhanced DeepONet implementation.

Modern tests aligned with the current MultiPhysicsDeepONet API.
Focuses on proper physics-aware neural operator testing without legacy compatibility.
"""

import jax
import pytest
from flax import nnx

from opifex.neural.operators.deeponet.multiphysics import MultiPhysicsDeepONet


class TestMultiPhysicsDeepONet:
    """Test suite for MultiPhysicsDeepONet with modern API."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size = 8
        branch_input_dim = 64
        trunk_input_dim = 2

        # Modern input format using Arrays
        branch_inputs = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, branch_input_dim)
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, trunk_input_dim)
        )

        return {
            "branch_inputs": branch_inputs,
            "trunk_input": trunk_input,
            "spatial_coords": jax.random.normal(
                jax.random.PRNGKey(2), (batch_size, trunk_input_dim)
            ),
        }

    def test_single_physics_initialization(self):
        """Test initialization with single physics system."""
        model = MultiPhysicsDeepONet(
            branch_input_dim=64,
            trunk_input_dim=2,
            branch_hidden_dims=[128, 128],
            trunk_hidden_dims=[128, 128],
            latent_dim=64,
            num_physics_systems=1,
            sensor_optimization=False,  # Explicitly disable sensor optimization
            rngs=nnx.Rngs(0),
        )

        assert len(model.physics_operators) == 1
        assert model.latent_dim == 64
        assert model.num_physics_systems == 1

    def test_multi_physics_initialization(self):
        """Test initialization with multiple physics systems."""
        model = MultiPhysicsDeepONet(
            branch_input_dim=64,
            trunk_input_dim=2,
            branch_hidden_dims=[128, 128],
            trunk_hidden_dims=[128, 128],
            latent_dim=64,
            num_physics_systems=3,
            physics_constraints=["conservation", "symmetry"],
            sensor_optimization=False,  # Explicitly disable sensor optimization
            rngs=nnx.Rngs(0),
        )

        assert len(model.physics_operators) == 3
        assert len(model.physics_constraints) == 2
        assert hasattr(model, "system_coupling")

    def test_forward_pass_single_system(self, sample_data):
        """Test forward pass with single physics system."""
        model = MultiPhysicsDeepONet(
            branch_input_dim=64,
            trunk_input_dim=2,
            branch_hidden_dims=[128, 128],
            trunk_hidden_dims=[128, 128],
            latent_dim=64,
            num_physics_systems=1,
            sensor_optimization=False,  # Explicitly disable sensor optimization
            rngs=nnx.Rngs(0),
        )

        output = model(
            sample_data["branch_inputs"],
            sample_data["trunk_input"],
        )

        expected_shape = sample_data["trunk_input"].shape[:-1]  # Remove last dim
        assert output.shape == expected_shape

    def test_forward_pass_multi_system(self, sample_data):
        """Test forward pass with multiple physics systems."""
        model = MultiPhysicsDeepONet(
            branch_input_dim=64,
            trunk_input_dim=2,
            branch_hidden_dims=[128, 128],
            trunk_hidden_dims=[128, 128],
            latent_dim=64,
            num_physics_systems=2,
            sensor_optimization=False,  # Explicitly disable sensor optimization
            rngs=nnx.Rngs(0),
        )

        # Prepare multi-system branch inputs
        branch_inputs = [
            sample_data["branch_inputs"],
            sample_data["branch_inputs"],
        ]

        output = model(
            branch_inputs,
            sample_data["trunk_input"],
        )

        expected_shape = sample_data["trunk_input"].shape[:-1]
        assert output.shape == expected_shape

    def test_physics_attention_integration(self, sample_data):
        """Test physics-aware attention mechanism."""
        model = MultiPhysicsDeepONet(
            branch_input_dim=64,
            trunk_input_dim=2,
            branch_hidden_dims=[128, 128],
            trunk_hidden_dims=[128, 128],
            latent_dim=64,
            use_attention=True,
            attention_heads=4,
            physics_constraints=["conservation"],
            sensor_optimization=False,  # Explicitly disable sensor optimization
            rngs=nnx.Rngs(0),
        )

        assert hasattr(model, "physics_attention")

        output = model(
            sample_data["branch_inputs"],
            sample_data["trunk_input"],
        )

        expected_shape = sample_data["trunk_input"].shape[:-1]
        assert output.shape == expected_shape

    def test_sensor_optimization_enabled(self, sample_data):
        """Test with sensor optimization enabled."""
        model = MultiPhysicsDeepONet(
            branch_input_dim=64,
            trunk_input_dim=2,
            branch_hidden_dims=[128, 128],
            trunk_hidden_dims=[128, 128],
            latent_dim=64,
            sensor_optimization=True,
            num_sensors=32,
            rngs=nnx.Rngs(0),
        )

        assert hasattr(model, "sensor_optimizer")
        assert model.sensor_optimization is True

        output = model(
            sample_data["branch_inputs"],
            sample_data["trunk_input"],
            spatial_coords=sample_data["spatial_coords"],
        )

        expected_shape = sample_data["trunk_input"].shape[:-1]
        assert output.shape == expected_shape

    def test_sensor_positions_access(self):
        """Test accessing sensor positions."""
        model = MultiPhysicsDeepONet(
            branch_input_dim=64,
            trunk_input_dim=2,
            branch_hidden_dims=[128, 128],
            trunk_hidden_dims=[128, 128],
            latent_dim=64,
            sensor_optimization=True,
            num_sensors=32,
            rngs=nnx.Rngs(0),
        )

        positions = model.get_sensor_positions()
        # Should return None or actual positions depending on implementation
        assert positions is None or isinstance(positions, jax.Array)

    def test_multi_system_coupling(self, sample_data):
        """Test multi-system coupling functionality."""
        model = MultiPhysicsDeepONet(
            branch_input_dim=64,
            trunk_input_dim=2,
            branch_hidden_dims=[128, 128],
            trunk_hidden_dims=[128, 128],
            latent_dim=64,
            num_physics_systems=3,
            sensor_optimization=False,  # Explicitly disable sensor optimization
            rngs=nnx.Rngs(0),
        )

        # Test with multiple branch inputs
        branch_inputs = [
            jax.random.normal(jax.random.PRNGKey(i), (8, 64)) for i in range(3)
        ]

        output = model(
            branch_inputs,
            sample_data["trunk_input"],
        )

        expected_shape = sample_data["trunk_input"].shape[:-1]
        assert output.shape == expected_shape

    def test_physics_constraints_setting(self):
        """Test setting physics constraints."""
        model = MultiPhysicsDeepONet(
            branch_input_dim=64,
            trunk_input_dim=2,
            branch_hidden_dims=[128, 128],
            trunk_hidden_dims=[128, 128],
            latent_dim=64,
            sensor_optimization=False,  # Explicitly disable sensor optimization
            rngs=nnx.Rngs(0),
        )

        new_constraints = ["momentum", "energy", "continuity"]
        model.set_physics_constraints(new_constraints)

        assert model.physics_constraints == new_constraints

    def test_training_mode_behavior(self, sample_data):
        """Test behavior in training mode."""
        model = MultiPhysicsDeepONet(
            branch_input_dim=64,
            trunk_input_dim=2,
            branch_hidden_dims=[128, 128],
            trunk_hidden_dims=[128, 128],
            latent_dim=64,
            use_attention=True,
            physics_constraints=["conservation"],
            sensor_optimization=False,  # Explicitly disable sensor optimization
            rngs=nnx.Rngs(0),
        )

        output = model(
            sample_data["branch_inputs"],
            sample_data["trunk_input"],
            training=True,
        )

        expected_shape = sample_data["trunk_input"].shape[:-1]
        assert output.shape == expected_shape

    def test_activation_function_compatibility(self):
        """Test different activation functions."""
        for activation in [nnx.relu, nnx.gelu, nnx.tanh]:
            model = MultiPhysicsDeepONet(
                branch_input_dim=64,
                trunk_input_dim=2,
                branch_hidden_dims=[128, 128],
                trunk_hidden_dims=[128, 128],
                latent_dim=64,
                activation=activation,
                sensor_optimization=False,  # Explicitly disable sensor optimization
                rngs=nnx.Rngs(0),
            )

            assert len(model.physics_operators) == 1
