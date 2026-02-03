"""Tests for U-Net Neural Operator (UNO).

Tests the existing UNO implementation:
- UNetBlock and UNeuralOperator (encode → spectral → decode with skip connections)
- create_uno factory

Reference: Ashiqur Rahman et al. "U-NO: U-shaped Neural Operators" (2022)
    - U-Net arch for operators: 26–44% error reduction on Darcy/Navier-Stokes
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.specialized.uno import (
    create_uno,
    UNetBlock,
    UNeuralOperator,
)


@pytest.fixture
def rngs():
    return nnx.Rngs(0)


# =========================================================================
# UNetBlock Tests
# =========================================================================


class TestUNetBlock:
    """Test the U-Net convolutional block."""

    def test_init(self, rngs):
        """UNetBlock should initialize."""
        block = UNetBlock(in_channels=8, out_channels=16, rngs=rngs)
        assert block is not None

    def test_forward_shape(self, rngs):
        """Output shape should match out_channels."""
        block = UNetBlock(in_channels=8, out_channels=16, rngs=rngs)
        x = jnp.ones((2, 16, 16, 8))
        y = block(x)
        assert y.shape == (2, 16, 16, 16)

    def test_stride(self, rngs):
        """Stride should reduce spatial dims."""
        block = UNetBlock(in_channels=8, out_channels=16, stride=2, rngs=rngs)
        x = jnp.ones((2, 16, 16, 8))
        y = block(x)
        assert y.shape[1] == 8  # Halved
        assert y.shape[2] == 8

    def test_no_norm(self, rngs):
        """Should work without normalization."""
        block = UNetBlock(in_channels=8, out_channels=16, use_norm=False, rngs=rngs)
        x = jnp.ones((2, 16, 16, 8))
        y = block(x)
        assert y.shape == (2, 16, 16, 16)

    def test_finite_output(self, rngs):
        """Output should be finite."""
        block = UNetBlock(in_channels=4, out_channels=8, rngs=rngs)
        x = jnp.ones((1, 8, 8, 4))
        y = block(x)
        assert jnp.all(jnp.isfinite(y))


# =========================================================================
# UNeuralOperator Tests
# =========================================================================


class TestUNeuralOperator:
    """Test the main UNO model."""

    def test_init(self, rngs):
        """UNO should initialize with default params."""
        model = UNeuralOperator(
            input_channels=1,
            output_channels=1,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        assert model is not None

    def test_forward_shape(self, rngs):
        """Forward should map input → output with correct channels."""
        model = UNeuralOperator(
            input_channels=1,
            output_channels=3,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 32, 32, 1))
        y = model(x)
        assert y.shape == (2, 32, 32, 3)

    def test_same_channels(self, rngs):
        """Should work when input_channels == output_channels."""
        model = UNeuralOperator(
            input_channels=2,
            output_channels=2,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((1, 16, 16, 2))
        y = model(x)
        assert y.shape == (1, 16, 16, 2)

    def test_without_spectral(self, rngs):
        """Should work without Fourier layers at bottleneck."""
        model = UNeuralOperator(
            input_channels=1,
            output_channels=1,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            use_spectral=False,
            rngs=rngs,
        )
        x = jnp.ones((1, 16, 16, 1))
        y = model(x)
        assert y.shape == (1, 16, 16, 1)

    def test_finite_output(self, rngs):
        """Output should be finite for random input."""
        model = UNeuralOperator(
            input_channels=1,
            output_channels=1,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (2, 16, 16, 1))
        y = model(x)
        assert jnp.all(jnp.isfinite(y))

    def test_deterministic(self, rngs):
        """Same input → same output (no stochastic layers in eval)."""
        model = UNeuralOperator(
            input_channels=1,
            output_channels=1,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((1, 16, 16, 1))
        y1 = model(x, deterministic=True)
        y2 = model(x, deterministic=True)
        assert jnp.allclose(y1, y2)

    def test_batch_dimension(self, rngs):
        """Should handle different batch sizes."""
        model = UNeuralOperator(
            input_channels=1,
            output_channels=1,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        for batch in [1, 3]:
            x = jnp.ones((batch, 16, 16, 1))
            y = model(x)
            assert y.shape[0] == batch

    def test_jit_compatible(self, rngs):
        """UNO should work under JIT."""
        model = UNeuralOperator(
            input_channels=1,
            output_channels=1,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((1, 16, 16, 1))

        @nnx.jit
        def forward_jit(model, x):
            return model(x)

        y = forward_jit(model, x)
        assert y.shape == (1, 16, 16, 1)
        assert jnp.all(jnp.isfinite(y))

    def test_gradient_flow(self, rngs):
        """Gradients should flow through UNO."""
        model = UNeuralOperator(
            input_channels=1,
            output_channels=1,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 16, 16, 1))

        @nnx.jit
        def loss_fn(model):
            return jnp.mean(model(x) ** 2)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(model)
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, "__len__"))
        assert has_nonzero


# =========================================================================
# Factory Function Tests
# =========================================================================


class TestCreateUNO:
    """Test the create_uno factory function."""

    def test_factory_creates_model(self, rngs):
        """Factory should create a UNeuralOperator."""
        model = create_uno(
            input_channels=1,
            output_channels=1,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        assert isinstance(model, UNeuralOperator)

    def test_factory_forward(self, rngs):
        """Factory-created model should forward correctly."""
        model = create_uno(
            input_channels=2,
            output_channels=3,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((1, 16, 16, 2))
        y = model(x)
        assert y.shape == (1, 16, 16, 3)

    def test_factory_uses_spectral(self, rngs):
        """Factory should create model with spectral layers by default."""
        model = create_uno(
            input_channels=1,
            output_channels=1,
            hidden_channels=16,
            modes=4,
            n_layers=2,
            rngs=rngs,
        )
        assert model.use_spectral is True
