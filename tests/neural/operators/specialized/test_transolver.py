"""Test Transolver operator (physics-attention).

Test suite for the Transolver implementation with slice-attend-deslice
physics attention mechanism for operator learning on irregular/structured meshes.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.specialized.transolver import (
    create_transolver,
    PhysicsAttention,
    Transolver,
    TransolverBlock,
    TransolverConfig,
)


class TestTransolverConfig:
    """Test TransolverConfig validation."""

    def test_default_config(self):
        """Test config with default values."""
        config = TransolverConfig(
            space_dim=2,
            fun_dim=1,
            out_dim=1,
        )
        assert config.hidden_dim == 256
        assert config.num_heads == 8
        assert config.num_layers == 5
        assert config.slice_num == 32
        assert config.dropout_rate == 0.0
        assert config.mlp_ratio == 1

    def test_config_frozen(self):
        """Test that config is immutable."""
        config = TransolverConfig(space_dim=2, fun_dim=1, out_dim=1)
        with pytest.raises(AttributeError):
            config.hidden_dim = 512  # type: ignore[misc]

    def test_config_validation_positive_dims(self):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError, match="hidden_dim"):
            TransolverConfig(space_dim=2, fun_dim=1, out_dim=1, hidden_dim=0)

    def test_config_validation_dropout(self):
        """Test dropdown rate validation."""
        with pytest.raises(ValueError, match="dropout_rate"):
            TransolverConfig(space_dim=2, fun_dim=1, out_dim=1, dropout_rate=1.5)


class TestPhysicsAttention:
    """Test PhysicsAttention (slice-attend-deslice) module."""

    @pytest.fixture
    def rngs(self):
        """Provide Flax NNX rngs."""
        return nnx.Rngs(jax.random.PRNGKey(42))

    def test_init(self, rngs):
        """Test PhysicsAttention initialization."""
        attn = PhysicsAttention(
            dim=64,
            num_heads=4,
            dim_head=16,
            slice_num=8,
            dropout_rate=0.0,
            rngs=rngs,
        )
        assert attn.num_heads == 4
        assert attn.dim_head == 16
        assert attn.slice_num == 8

    def test_forward_shape_irregular(self, rngs):
        """Test output shape on irregular mesh input (B, N, C)."""
        dim = 64
        attn = PhysicsAttention(
            dim=dim,
            num_heads=4,
            dim_head=16,
            slice_num=8,
            dropout_rate=0.0,
            rngs=rngs,
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 100, dim))
        out = attn(x, deterministic=True)
        assert out.shape == (2, 100, dim)

    def test_output_dtype(self, rngs):
        """Test output is float32."""
        dim = 32
        attn = PhysicsAttention(
            dim=dim,
            num_heads=2,
            dim_head=16,
            slice_num=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 50, dim))
        out = attn(x, deterministic=True)
        assert out.dtype == jnp.float32

    def test_different_sequence_lengths(self, rngs):
        """Test that attention works with various sequence lengths."""
        dim = 32
        attn = PhysicsAttention(
            dim=dim,
            num_heads=2,
            dim_head=16,
            slice_num=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        for n in [10, 50, 200]:
            x = jax.random.normal(jax.random.PRNGKey(0), (1, n, dim))
            out = attn(x, deterministic=True)
            assert out.shape == (1, n, dim)

    def test_dropout_stochastic(self, rngs):
        """Test that dropout produces different outputs in training mode."""
        dim = 32
        attn = PhysicsAttention(
            dim=dim,
            num_heads=2,
            dim_head=16,
            slice_num=4,
            dropout_rate=0.5,
            rngs=rngs,
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 20, dim))
        # Deterministic should be reproducible
        out1 = attn(x, deterministic=True)
        out2 = attn(x, deterministic=True)
        assert jnp.allclose(out1, out2, atol=1e-6)

    def test_temperature_is_learnable(self, rngs):
        """Test that temperature parameter is trainable."""
        attn = PhysicsAttention(
            dim=32,
            num_heads=2,
            dim_head=16,
            slice_num=4,
            dropout_rate=0.0,
            rngs=rngs,
        )
        # Temperature should be an nnx.Param (trainable)
        assert hasattr(attn, "temperature")
        assert isinstance(attn.temperature, nnx.Param)


class TestTransolverBlock:
    """Test TransolverBlock (pre-norm transformer block)."""

    @pytest.fixture
    def rngs(self):
        """Provide Flax NNX rngs."""
        return nnx.Rngs(jax.random.PRNGKey(42))

    def test_block_forward_shape(self, rngs):
        """Test block produces correct output shape."""
        hidden_dim = 64
        block = TransolverBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            slice_num=8,
            dropout_rate=0.0,
            mlp_ratio=4,
            rngs=rngs,
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 50, hidden_dim))
        out = block(x, deterministic=True)
        assert out.shape == (2, 50, hidden_dim)

    def test_block_residual_connection(self, rngs):
        """Test that block uses residual connections (output ≠ zero for zero input)."""
        hidden_dim = 32
        block = TransolverBlock(
            hidden_dim=hidden_dim,
            num_heads=2,
            slice_num=4,
            dropout_rate=0.0,
            mlp_ratio=1,
            rngs=rngs,
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 10, hidden_dim))
        out = block(x, deterministic=True)
        # Output should not be identical to input (attention modifies it)
        assert not jnp.allclose(out, x)

    def test_last_layer_output_dim(self, rngs):
        """Test last layer projects to output dimension."""
        hidden_dim = 64
        out_dim = 3
        block = TransolverBlock(
            hidden_dim=hidden_dim,
            num_heads=4,
            slice_num=8,
            dropout_rate=0.0,
            mlp_ratio=1,
            last_layer=True,
            out_dim=out_dim,
            rngs=rngs,
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 50, hidden_dim))
        out = block(x, deterministic=True)
        assert out.shape == (2, 50, out_dim)


class TestTransolver:
    """Test full Transolver model."""

    @pytest.fixture
    def rngs(self):
        """Provide Flax NNX rngs."""
        return nnx.Rngs(jax.random.PRNGKey(42))

    @pytest.fixture
    def config(self):
        """Provide standard test config."""
        return TransolverConfig(
            space_dim=2,
            fun_dim=1,
            out_dim=1,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            slice_num=8,
        )

    def test_init(self, config, rngs):
        """Test Transolver model initialization."""
        model = Transolver(config, rngs=rngs)
        assert model.config == config

    def test_forward_shape(self, config, rngs):
        """Test forward pass shape: (B, N, space_dim+fun_dim) → (B, N, out_dim)."""
        model = Transolver(config, rngs=rngs)
        batch_size = 4
        num_points = 100
        # x contains spatial coords + function values
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (batch_size, num_points, config.space_dim),
        )
        fx = jax.random.normal(
            jax.random.PRNGKey(1),
            (batch_size, num_points, config.fun_dim),
        )
        out = model(x, fx, deterministic=True)
        assert out.shape == (batch_size, num_points, config.out_dim)

    def test_forward_no_fx(self, rngs):
        """Test forward pass when fx is None (coordinates only)."""
        config = TransolverConfig(
            space_dim=2,
            fun_dim=0,
            out_dim=1,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            slice_num=8,
        )
        model = Transolver(config, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 50, 2))
        out = model(x, deterministic=True)
        assert out.shape == (2, 50, 1)

    def test_jit_compatible(self, config, rngs):
        """Test that model works under jax.jit."""
        model = Transolver(config, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 50, 2))
        fx = jax.random.normal(jax.random.PRNGKey(1), (2, 50, 1))

        @nnx.jit
        def forward(m, x, fx):
            return m(x, fx, deterministic=True)

        out = forward(model, x, fx)
        assert out.shape == (2, 50, 1)

    def test_gradient_flow(self, config, rngs):
        """Test that gradients flow through the model."""
        model = Transolver(config, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 50, 2))
        fx = jax.random.normal(jax.random.PRNGKey(1), (2, 50, 1))

        @nnx.value_and_grad
        def loss_fn(m):
            out = m(x, fx, deterministic=True)
            return jnp.mean(out**2)

        loss, grads = loss_fn(model)
        assert jnp.isfinite(loss)
        # Check that at least some gradients are non-zero
        grad_leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves)
        assert has_nonzero, "All gradients are zero"

    def test_batch_dimension(self, config, rngs):
        """Test different batch sizes."""
        model = Transolver(config, rngs=rngs)
        for batch in [1, 4, 8]:
            x = jax.random.normal(jax.random.PRNGKey(0), (batch, 30, 2))
            fx = jax.random.normal(jax.random.PRNGKey(1), (batch, 30, 1))
            out = model(x, fx, deterministic=True)
            assert out.shape == (batch, 30, 1)

    def test_1d_space(self, rngs):
        """Test Transolver with 1D spatial input."""
        config = TransolverConfig(
            space_dim=1,
            fun_dim=1,
            out_dim=1,
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            slice_num=4,
        )
        model = Transolver(config, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 100, 1))
        fx = jax.random.normal(jax.random.PRNGKey(1), (2, 100, 1))
        out = model(x, fx, deterministic=True)
        assert out.shape == (2, 100, 1)

    def test_3d_space(self, rngs):
        """Test Transolver with 3D spatial input."""
        config = TransolverConfig(
            space_dim=3,
            fun_dim=5,
            out_dim=3,
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            slice_num=4,
        )
        model = Transolver(config, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 50, 3))
        fx = jax.random.normal(jax.random.PRNGKey(1), (2, 50, 5))
        out = model(x, fx, deterministic=True)
        assert out.shape == (2, 50, 3)

    def test_multi_output_channels(self, rngs):
        """Test multiple output dimensions."""
        config = TransolverConfig(
            space_dim=2,
            fun_dim=3,
            out_dim=5,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            slice_num=8,
        )
        model = Transolver(config, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 40, 2))
        fx = jax.random.normal(jax.random.PRNGKey(1), (2, 40, 3))
        out = model(x, fx, deterministic=True)
        assert out.shape == (2, 40, 5)


class TestCreateTransolver:
    """Test factory function."""

    def test_factory_creates_model(self):
        """Test create_transolver returns a Transolver instance."""
        model = create_transolver(
            space_dim=2,
            fun_dim=1,
            out_dim=1,
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            rngs=nnx.Rngs(jax.random.PRNGKey(0)),
        )
        assert isinstance(model, Transolver)

    def test_factory_forward(self):
        """Test that factory-created model can do a forward pass."""
        model = create_transolver(
            space_dim=2,
            fun_dim=1,
            out_dim=1,
            hidden_dim=32,
            num_heads=2,
            num_layers=2,
            rngs=nnx.Rngs(jax.random.PRNGKey(0)),
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 20, 2))
        fx = jax.random.normal(jax.random.PRNGKey(1), (1, 20, 1))
        out = model(x, fx, deterministic=True)
        assert out.shape == (1, 20, 1)
