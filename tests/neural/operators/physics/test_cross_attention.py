"""Test Physics Cross Attention mechanism.

Modern tests for PhysicsCrossAttention aligned with current API.
Focuses on proper physics-informed attention testing without legacy compatibility.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.physics.attention import PhysicsCrossAttention


class TestPhysicsCrossAttention:
    """Test suite for PhysicsCrossAttention with modern API."""

    @pytest.fixture
    def sample_data_single_system(self):
        """Create sample data for single system testing."""
        batch_size = 4
        seq_len = 16
        embed_dim = 32

        return jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, embed_dim))

    @pytest.fixture
    def sample_data_multi_system(self):
        """Create sample data for multi-system testing."""
        batch_size = 4
        num_systems = 2
        seq_len = 16
        embed_dim = 32

        return jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_systems, seq_len, embed_dim)
        )

    @pytest.fixture
    def physics_info(self):
        """Create physics constraint data."""
        batch_size = 4
        seq_len = 16
        physics_dim = 3  # For energy, momentum, angular_momentum

        return jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, physics_dim))

    def test_physics_cross_attention_initialization(self):
        """Test physics cross attention initialization with modern parameters."""
        model = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy", "momentum"],
            num_physics_systems=2,
            rngs=nnx.Rngs(0),
        )

        assert hasattr(model, "cross_attention_layers")
        assert hasattr(model, "physics_projection")
        assert hasattr(model, "conservation_projection")
        assert len(model.cross_attention_layers) == 2

    def test_physics_cross_attention_single_system(self, sample_data_single_system, physics_info):
        """Test physics cross attention with single system."""
        model = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy", "momentum", "angular_momentum"],
            num_physics_systems=1,
            rngs=nnx.Rngs(0),
        )

        output = model(sample_data_single_system, physics_info=physics_info)

        assert output.shape == sample_data_single_system.shape
        assert jnp.all(jnp.isfinite(output))

    def test_physics_cross_attention_multi_system(self, sample_data_multi_system, physics_info):
        """Test physics cross attention with multi-system."""
        model = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy", "momentum"],
            num_physics_systems=2,
            rngs=nnx.Rngs(0),
        )

        # Adjust physics_info for multi-system (add system dimension)
        physics_info_multi = jnp.expand_dims(physics_info[:, :, :2], axis=1).repeat(2, axis=1)

        output = model(sample_data_multi_system, physics_info=physics_info_multi)

        assert output.shape == sample_data_multi_system.shape
        assert jnp.all(jnp.isfinite(output))

    def test_physics_cross_attention_different_num_heads(self, sample_data_single_system):
        """Test physics cross attention with different numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            model = PhysicsCrossAttention(
                embed_dim=32,
                num_heads=num_heads,
                physics_constraints=["energy"],
                num_physics_systems=1,
                rngs=nnx.Rngs(0),
            )

            output = model(sample_data_single_system)

            assert output.shape == sample_data_single_system.shape
            assert jnp.all(jnp.isfinite(output))

    def test_physics_cross_attention_different_constraints(self, sample_data_single_system):
        """Test physics cross attention with different constraint sets."""
        constraint_sets = [
            ["energy"],
            ["momentum"],
            ["energy", "momentum"],
            ["energy", "momentum", "angular_momentum"],
        ]

        for constraints in constraint_sets:
            model = PhysicsCrossAttention(
                embed_dim=32,
                num_heads=4,
                physics_constraints=constraints,
                num_physics_systems=1,
                rngs=nnx.Rngs(0),
            )

            output = model(sample_data_single_system)

            assert output.shape == sample_data_single_system.shape
            assert jnp.all(jnp.isfinite(output))

    def test_physics_cross_attention_conservation_enforcement(self, sample_data_single_system):
        """Test conservation law enforcement."""
        model = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy", "momentum"],
            num_physics_systems=1,
            conservation_weight=0.1,
            rngs=nnx.Rngs(0),
        )

        # Test forward with conservation
        output, conservation_loss = model.forward_with_conservation(sample_data_single_system)

        assert output.shape == sample_data_single_system.shape
        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(jnp.isfinite(conservation_loss))
        assert conservation_loss.shape == ()  # Scalar loss

    def test_physics_cross_attention_adaptive_weighting(self, sample_data_single_system):
        """Test adaptive constraint weighting."""
        # Test with adaptive weighting enabled
        model_adaptive = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy", "momentum"],
            num_physics_systems=1,
            adaptive_weighting=True,
            rngs=nnx.Rngs(0),
        )

        # Test with adaptive weighting disabled
        model_fixed = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy", "momentum"],
            num_physics_systems=1,
            adaptive_weighting=False,
            rngs=nnx.Rngs(0),
        )

        output_adaptive = model_adaptive(sample_data_single_system)
        output_fixed = model_fixed(sample_data_single_system)

        assert output_adaptive.shape == sample_data_single_system.shape
        assert output_fixed.shape == sample_data_single_system.shape
        assert jnp.all(jnp.isfinite(output_adaptive))
        assert jnp.all(jnp.isfinite(output_fixed))

    def test_physics_cross_attention_cross_system_coupling(self, sample_data_multi_system):
        """Test cross-system coupling mechanism."""
        # Test with cross-system coupling enabled
        model_coupled = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy"],
            num_physics_systems=2,
            cross_system_coupling=True,
            rngs=nnx.Rngs(0),
        )

        # Test with cross-system coupling disabled
        model_decoupled = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy"],
            num_physics_systems=2,
            cross_system_coupling=False,
            rngs=nnx.Rngs(0),
        )

        output_coupled = model_coupled(sample_data_multi_system)
        output_decoupled = model_decoupled(sample_data_multi_system)

        assert output_coupled.shape == sample_data_multi_system.shape
        assert output_decoupled.shape == sample_data_multi_system.shape
        assert jnp.all(jnp.isfinite(output_coupled))
        assert jnp.all(jnp.isfinite(output_decoupled))

    def test_physics_cross_attention_gradient_computation(self, sample_data_single_system):
        """Test gradient computation through physics cross attention."""
        model = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy", "momentum"],
            num_physics_systems=1,
            rngs=nnx.Rngs(0),
        )

        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        grads = nnx.grad(loss_fn)(model, sample_data_single_system)

        # Check gradient properties
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)

    def test_physics_cross_attention_jax_transformations(self, sample_data_single_system):
        """Test physics cross attention compatibility with JAX transformations."""
        model = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy"],
            num_physics_systems=1,
            rngs=nnx.Rngs(0),
        )

        @jax.jit
        def jitted_forward(x):
            return model(x)

        output = jitted_forward(sample_data_single_system)

        assert output.shape == sample_data_single_system.shape
        assert jnp.all(jnp.isfinite(output))

    def test_physics_cross_attention_dropout(self, sample_data_single_system):
        """Test physics cross attention with dropout."""
        model = PhysicsCrossAttention(
            embed_dim=32,
            num_heads=4,
            physics_constraints=["energy"],
            num_physics_systems=1,
            dropout_rate=0.1,
            rngs=nnx.Rngs(0),
        )

        # Test in training mode
        output_train = model(sample_data_single_system, training=True)

        # Test in evaluation mode
        output_eval = model(sample_data_single_system, training=False)

        assert output_train.shape == sample_data_single_system.shape
        assert output_eval.shape == sample_data_single_system.shape
        assert jnp.all(jnp.isfinite(output_train))
        assert jnp.all(jnp.isfinite(output_eval))

    def test_physics_cross_attention_large_systems(self):
        """Test physics cross attention with larger multi-system setup."""
        batch_size = 2
        num_systems = 4
        seq_len = 32
        embed_dim = 64

        large_data = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, num_systems, seq_len, embed_dim)
        )

        model = PhysicsCrossAttention(
            embed_dim=64,
            num_heads=8,
            physics_constraints=["energy", "momentum", "angular_momentum"],
            num_physics_systems=4,
            rngs=nnx.Rngs(0),
        )

        output = model(large_data)

        assert output.shape == large_data.shape
        assert jnp.all(jnp.isfinite(output))

    def test_physics_cross_attention_parameter_efficiency(self):
        """Test parameter efficiency for physics cross attention."""
        model = PhysicsCrossAttention(
            embed_dim=64,
            num_heads=8,
            physics_constraints=["energy", "momentum"],
            num_physics_systems=3,
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
        assert param_count < 500_000  # Not too large

        # Test functionality
        batch_size = 2
        seq_len = 16

        test_data = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, 64))
        output = model(test_data)

        assert output.shape == (batch_size, seq_len, 64)
        assert jnp.all(jnp.isfinite(output))


class TestConservationResidualLoss:
    """Test the genuine flux-divergence conservation loss in cross-attention."""

    @staticmethod
    def _model() -> PhysicsCrossAttention:
        """Build a small single-system cross-attention module."""
        return PhysicsCrossAttention(
            embed_dim=8,
            num_heads=2,
            physics_constraints=["energy", "momentum"],
            num_physics_systems=1,
            conservation_weight=0.5,
            rngs=nnx.Rngs(0),
        )

    def test_divergence_free_field_gives_zero_loss(self):
        """A field constant along the sequence axis is divergence-free -> ~0."""
        model = self._model()
        # Constant across seq_len -> zero spatial divergence.
        field = jnp.broadcast_to(jax.random.normal(jax.random.PRNGKey(1), (4, 1, 8)), (4, 16, 8))
        loss = model._compute_conservation_loss(field)
        assert loss < 1e-10

    def test_nonzero_divergence_gives_positive_loss(self):
        """A field varying along the sequence axis gives a positive loss."""
        model = self._model()
        seq = jnp.linspace(0.0, 1.0, 16)
        # Linearly varying along the sequence axis -> non-zero divergence.
        field = jnp.broadcast_to(seq[None, :, None], (4, 16, 8))
        loss = model._compute_conservation_loss(field)
        assert loss > 0.0
        assert jnp.isfinite(loss)

    def test_conservation_weight_scales_loss(self):
        """The loss scales linearly with conservation_weight."""
        seq = jnp.linspace(0.0, 1.0, 16)
        field = jnp.broadcast_to(seq[None, :, None], (4, 16, 8))

        model_a = PhysicsCrossAttention(
            embed_dim=8,
            num_heads=2,
            physics_constraints=["energy"],
            num_physics_systems=1,
            conservation_weight=0.2,
            rngs=nnx.Rngs(0),
        )
        model_b = PhysicsCrossAttention(
            embed_dim=8,
            num_heads=2,
            physics_constraints=["energy"],
            num_physics_systems=1,
            conservation_weight=0.4,
            rngs=nnx.Rngs(0),
        )
        loss_a = model_a._compute_conservation_loss(field)
        loss_b = model_b._compute_conservation_loss(field)
        assert jnp.isclose(loss_b, 2.0 * loss_a)

    def test_jit_grad_vmap(self):
        """Conservation loss is jit/grad/vmap compatible through the module."""
        model = self._model()
        x = jax.random.normal(jax.random.PRNGKey(2), (4, 16, 8))

        @nnx.jit
        def loss_fn(m, inputs):
            _, loss = m.forward_with_conservation(inputs)
            return loss

        jitted = loss_fn(model, x)
        assert jnp.isfinite(jitted)

        grads = nnx.grad(loss_fn)(model, x)
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert any(jnp.linalg.norm(leaf) > 0.0 for leaf in grad_leaves)

        fields = jax.random.normal(jax.random.PRNGKey(3), (5, 4, 16, 8))
        batched = jax.vmap(model._compute_conservation_loss)(fields)
        assert batched.shape == (5,)
        assert jnp.all(jnp.isfinite(batched))
