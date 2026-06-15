"""Test multi-physics enhanced DeepONet implementation.

Modern tests aligned with the current MultiPhysicsDeepONet API.
Focuses on proper physics-aware neural operator testing without legacy compatibility.
"""

import jax
import jax.numpy as jnp
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
        branch_inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, branch_input_dim))
        trunk_input = jax.random.normal(jax.random.PRNGKey(1), (batch_size, trunk_input_dim))

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
        # Each physics system owns an independent trunk network.
        trunk_ids = {id(op.trunk_net) for op in model.physics_operators}
        assert len(trunk_ids) == 3

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

        # Multi-physics single-location output carries a per-physics axis:
        # (batch, num_physics_systems).
        batch_size = sample_data["trunk_input"].shape[0]
        assert output.shape == (batch_size, 2)

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
        branch_inputs = [jax.random.normal(jax.random.PRNGKey(i), (8, 64)) for i in range(3)]

        output = model(
            branch_inputs,
            sample_data["trunk_input"],
        )

        # Three coupled physics systems -> per-physics axis of size 3.
        batch_size = sample_data["trunk_input"].shape[0]
        assert output.shape == (batch_size, 3)

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


class TestPerPhysicsTrunks:
    """Tests for genuine per-physics (independent) trunk networks.

    Following the IndependentStrategy of Lu et al. 2022 (CMAME 393, 114778,
    section 3.1.6): each physics field is produced by its own branch and trunk
    network, then the fields are stacked along a trailing physics axis.
    """

    @staticmethod
    def _make_model(num_physics_systems: int, *, use_attention: bool = False):
        return MultiPhysicsDeepONet(
            branch_input_dim=16,
            trunk_input_dim=2,
            branch_hidden_dims=[32],
            trunk_hidden_dims=[32],
            latent_dim=24,
            num_physics_systems=num_physics_systems,
            use_attention=use_attention,
            sensor_optimization=False,
            rngs=nnx.Rngs(0),
        )

    def test_multi_location_output_shape_has_physics_axis(self):
        """Multi-location output is (batch, n_points, n_physics)."""
        model = self._make_model(3)
        batch_size, n_points = 4, 5
        branch = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 16))
        trunk = jax.random.normal(jax.random.PRNGKey(1), (batch_size, n_points, 2))

        output = model([branch, branch, branch], trunk)

        assert output.shape == (batch_size, n_points, 3)

    def test_distinct_trunks_yield_distinct_physics_fields(self):
        """Identical branch inputs must still give different physics fields.

        If every field shared one trunk (the old stub), feeding identical
        branch inputs would make all fields collapse onto each other. Distinct
        per-physics trunks (and branches) break that degeneracy.
        """
        model = self._make_model(2)
        batch_size, n_points = 3, 6
        branch = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 16))
        trunk = jax.random.normal(jax.random.PRNGKey(1), (batch_size, n_points, 2))

        output = model([branch, branch], trunk)

        field_0 = output[..., 0]
        field_1 = output[..., 1]
        assert not jnp.allclose(field_0, field_1)

    def test_per_physics_trunk_parameters_are_used(self):
        """Each physics output depends on its own trunk's parameters.

        Perturbing only physics operator 1's trunk must change field 1 while
        leaving field 0 untouched - proving field i is wired to trunk i.
        """
        model = self._make_model(2)
        batch_size, n_points = 2, 4
        branch = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 16))
        trunk = jax.random.normal(jax.random.PRNGKey(1), (batch_size, n_points, 2))

        before = model([branch, branch], trunk)

        # Perturb only the second physics operator's trunk first layer.
        kernel = model.physics_operators[1].trunk_net.layers[0].kernel
        kernel.value = kernel.value + 1.0

        after = model([branch, branch], trunk)

        # Field 0 (trunk 0) is unchanged; field 1 (trunk 1) changes.
        assert jnp.allclose(before[..., 0], after[..., 0])
        assert not jnp.allclose(before[..., 1], after[..., 1])

    def test_jit_grad_vmap_smoke(self):
        """jit/grad/vmap must all succeed on the per-physics forward pass."""
        model = self._make_model(2)
        graphdef, state = nnx.split(model)

        batch_size, n_points = 2, 4
        branch = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 16))
        trunk = jax.random.normal(jax.random.PRNGKey(1), (batch_size, n_points, 2))

        def forward(state, branch, trunk):
            merged = nnx.merge(graphdef, state)
            return merged([branch, branch], trunk)

        # jit
        jitted = jax.jit(forward)
        out = jitted(state, branch, trunk)
        assert out.shape == (batch_size, n_points, 2)

        # grad through a scalar loss
        def loss_fn(state):
            return jnp.sum(forward(state, branch, trunk) ** 2)

        grads = jax.grad(loss_fn)(state)
        leaves = jax.tree_util.tree_leaves(grads)
        assert leaves
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)

        # vmap over a leading ensemble axis of inputs
        branch_batched = jnp.stack([branch, branch + 1.0, branch - 1.0])
        trunk_batched = jnp.stack([trunk, trunk * 2.0, trunk * 0.5])
        vmapped = jax.vmap(forward, in_axes=(None, 0, 0))
        out_v = vmapped(state, branch_batched, trunk_batched)
        assert out_v.shape == (3, batch_size, n_points, 2)
