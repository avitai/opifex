"""Tests for operation fusion optimizer."""

import jax
import jax.numpy as jnp

from opifex.core.optimization.fusion_optimizer import (
    fused_elementwise_chain,
    fused_linear_activation,
    optimize_memory_layout_for_fusion,
)


class TestFusedLinearActivation:
    """Tests for fused linear + activation."""

    def test_output_shape(self):
        """Output has correct shape after linear + activation."""
        x = jnp.ones((4, 8))
        w = jnp.ones((8, 16))
        result = fused_linear_activation(x, w)
        assert result.shape == (4, 16)

    def test_with_bias(self):
        """Bias is added before activation."""
        x = jnp.zeros((2, 4))
        w = jnp.zeros((4, 3))
        bias = jnp.ones(3)
        result = fused_linear_activation(x, w, bias=bias, activation=jax.nn.relu)
        assert jnp.allclose(result, jnp.ones((2, 3)))

    def test_activation_applied(self):
        """Negative values are zeroed by relu activation."""
        x = jnp.array([[1.0, -1.0]])
        w = jnp.eye(2)
        result = fused_linear_activation(x, w, activation=jax.nn.relu)
        assert float(result[0, 0]) > 0
        assert float(result[0, 1]) == 0.0

    def test_without_bias(self):
        """Works correctly without bias."""
        x = jnp.ones((2, 3))
        w = jnp.ones((3, 2))
        result = fused_linear_activation(x, w, activation=lambda z: z)
        assert jnp.allclose(result, jnp.full((2, 2), 3.0))


class TestFusedElementwiseChain:
    """Tests for chained elementwise operations."""

    def test_identity_chain(self):
        """Empty operation chain returns input unchanged."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = fused_elementwise_chain(x, [])
        assert jnp.allclose(result, x)

    def test_single_op(self):
        """Single operation is applied correctly."""
        x = jnp.array([1.0, 4.0, 9.0])
        result = fused_elementwise_chain(x, [jnp.sqrt])
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))

    def test_chained_ops(self):
        """Multiple operations are composed in order."""
        x = jnp.array([-1.0, 0.0, 1.0])
        result = fused_elementwise_chain(x, [jnp.abs, lambda z: z + 1.0])
        assert jnp.allclose(result, jnp.array([2.0, 1.0, 2.0]))


class TestOptimizeMemoryLayout:
    """Tests for memory layout optimization."""

    def test_nchw_to_nhwc(self):
        """Converts NCHW tensor to NHWC layout."""
        x = jnp.ones((2, 3, 8, 8))  # NCHW
        result = optimize_memory_layout_for_fusion(x, target_layout="NHWC")
        assert result.shape == (2, 8, 8, 3)

    def test_same_layout_noop(self):
        """NCHW to NCHW is a no-op (function assumes NCHW input)."""
        x = jnp.ones((2, 3, 8, 8))  # NCHW
        result = optimize_memory_layout_for_fusion(x, target_layout="NCHW")
        assert result.shape == (2, 3, 8, 8)
        assert jnp.allclose(result, x)
