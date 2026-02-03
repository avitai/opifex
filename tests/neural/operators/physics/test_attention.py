"""Test Physics-Aware Attention Mechanisms.

Test suite for physics-informed attention components that incorporate
physical constraints and conservation laws.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.physics.attention import PhysicsAwareAttention


class TestPhysicsAwareAttention:
    """Test physics-aware attention mechanism."""

    def setup_method(self):
        """Setup for each test method with GPU/CPU backend detection."""
        self.backend = jax.default_backend()
        print(f"Running PhysicsAwareAttention tests on {self.backend}")

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_physics_attention_initialization(self, rngs):
        """Test physics attention initialization."""
        embed_dim = 128
        num_heads = 8
        physics_constraints = ["mass_conservation", "energy_conservation"]

        attention = PhysicsAwareAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            physics_constraints=physics_constraints,
            rngs=rngs,
        )

        assert attention.embed_dim == embed_dim
        assert attention.num_heads == num_heads
        assert attention.head_dim == embed_dim // num_heads
        assert attention.physics_constraints == physics_constraints
        assert hasattr(attention, "q_proj")
        assert hasattr(attention, "k_proj")
        assert hasattr(attention, "v_proj")
        assert hasattr(attention, "out_proj")
        assert hasattr(attention, "physics_proj")
        assert hasattr(attention, "constraint_weights")

    def test_physics_attention_forward(self, rngs):
        """Test physics attention forward pass."""
        batch_size = 4
        seq_len = 32
        embed_dim = 128
        num_heads = 8
        num_constraints = 3

        attention = PhysicsAwareAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            physics_constraints=["constraint1", "constraint2", "constraint3"],
            rngs=rngs,
        )

        x = jnp.ones((batch_size, seq_len, embed_dim))
        physics_info = jnp.ones((batch_size, num_constraints))

        output = attention(x, physics_info=physics_info, training=True)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_physics_attention_without_constraints(self, rngs):
        """Test physics attention without physics constraints."""
        batch_size = 2
        seq_len = 16
        embed_dim = 64

        attention = PhysicsAwareAttention(
            embed_dim=embed_dim,
            num_heads=4,
            rngs=rngs,
        )

        x = jnp.ones((batch_size, seq_len, embed_dim))
        output = attention(x, training=False)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert jnp.all(jnp.isfinite(output))
        # Should not have physics projection when no constraints are specified
        assert not hasattr(attention, "physics_proj")

    def test_physics_attention_differentiability(self, rngs):
        """Test physics attention differentiability."""
        attention = PhysicsAwareAttention(
            embed_dim=64,
            num_heads=4,
            physics_constraints=["test_constraint"],
            rngs=rngs,
        )

        def loss_fn(model, x, physics_info):
            return jnp.sum(model(x, physics_info=physics_info) ** 2)

        x = jnp.ones((2, 8, 64))
        physics_info = jnp.ones((2, 1))

        grads = nnx.grad(loss_fn)(attention, x, physics_info)
        assert hasattr(grads, "q_proj")
        assert hasattr(grads, "physics_proj")

        # Check that gradients are not all zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            jnp.linalg.norm(leaf) for leaf in grad_leaves if hasattr(leaf, "shape")
        ]
        assert len(grad_norms) > 0
        assert any(norm > 1e-8 for norm in grad_norms)

    def test_physics_attention_different_head_counts(self, rngs):
        """Test physics attention with different numbers of heads."""
        embed_dim = 96  # Divisible by various head counts
        batch_size = 2
        seq_len = 8

        for num_heads in [1, 2, 3, 4, 6, 8]:
            attention = PhysicsAwareAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                physics_constraints=["test"],
                rngs=rngs,
            )

            x = jnp.ones((batch_size, seq_len, embed_dim))
            physics_info = jnp.ones((batch_size, 1))

            output = attention(x, physics_info=physics_info, training=True)
            assert output.shape == (batch_size, seq_len, embed_dim)
            assert jnp.all(jnp.isfinite(output))
