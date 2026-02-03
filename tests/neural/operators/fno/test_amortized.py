"""Test Amortized Fourier Neural Operator (AM-FNO).

Test suite for Amortized Fourier Neural Operator implementation that
uses amortized computation for efficient multi-resolution processing.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.amortized import AmortizedFourierNeuralOperator


class TestAmortizedFourierNeuralOperator:
    """Test Amortized FNO for efficient multi-resolution processing."""

    def setup_method(self):
        """Setup for each test method."""
        self.backend = jax.default_backend()
        print(f"Running AmortizedFourierNeuralOperator tests on {self.backend}")

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_amortized_fno_initialization(self, rngs):
        """Test AM-FNO initialization with amortized parameters."""
        in_channels = 1
        out_channels = 1
        hidden_channels = 32
        modes = (16, 16)
        num_layers = 2
        kernel_hidden_dim = 64

        am_fno = AmortizedFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            kernel_hidden_dim=kernel_hidden_dim,
            rngs=rngs,
        )

        # Basic initialization check
        assert am_fno is not None
        assert callable(am_fno)

    def test_amortized_fno_forward_multi_resolution(self, rngs, rng_key):
        """Test AM-FNO forward pass with multi-resolution data."""
        batch_size = 2
        in_channels = 1
        out_channels = 1

        am_fno = AmortizedFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=32,
            modes=(16, 16),
            num_layers=2,
            kernel_hidden_dim=64,
            rngs=rngs,
        )

        # Test with original configuration from OPERATOR_CONFIGS
        input_shape = (batch_size, in_channels, 32, 32)
        expected_output_shape = (batch_size, out_channels, 16, 16)

        # Create input data
        x = jax.random.normal(rng_key, input_shape)

        output = am_fno(x)

        # Verify output shape (may be downsampled)
        assert output.shape == expected_output_shape
        assert jnp.all(jnp.isfinite(output))

    def test_amortized_fno_kernel_network(self, rngs, rng_key):
        """Test AM-FNO kernel network functionality."""
        am_fno = AmortizedFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=24,
            modes=(8, 8),
            num_layers=1,
            kernel_hidden_dim=32,
            rngs=rngs,
        )

        # Test with small input
        batch_size = 1
        x = jax.random.normal(rng_key, (batch_size, 2, 16, 16))

        output = am_fno(x)

        # Check that output is reasonable
        assert len(output.shape) == 4  # Should be 4D tensor
        assert output.shape[0] == batch_size
        assert jnp.all(jnp.isfinite(output))

    def test_amortized_fno_efficiency_comparison(self, rngs, rng_key):
        """Test AM-FNO efficiency with different kernel hidden dimensions."""
        # Test with different kernel network sizes
        kernel_dims = [32, 64, 128]
        outputs = []

        for kernel_dim in kernel_dims:
            am_fno = AmortizedFourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=16,
                modes=(8, 8),
                num_layers=1,
                kernel_hidden_dim=kernel_dim,
                rngs=rngs,
            )

            x = jax.random.normal(rng_key, (1, 1, 16, 16))
            output = am_fno(x)
            outputs.append(output)

            # Check basic properties
            assert jnp.all(jnp.isfinite(output))

        # Different kernel dimensions should work consistently
        for output in outputs:
            assert len(output.shape) == 4

    def test_amortized_fno_batch_processing(self, rngs, rng_key):
        """Test AM-FNO batch processing capabilities."""
        am_fno = AmortizedFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=24,
            modes=(8, 8),
            num_layers=2,
            kernel_hidden_dim=32,
            rngs=rngs,
        )

        # Test with different batch sizes
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            x = jax.random.normal(rng_key, (batch_size, 1, 16, 16))
            output = am_fno(x)

            # Check that batch dimension is preserved
            assert output.shape[0] == batch_size
            assert jnp.all(jnp.isfinite(output))

    def test_amortized_fno_differentiability(self, rngs, rng_key):
        """Test AM-FNO differentiability."""
        am_fno = AmortizedFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            num_layers=2,
            kernel_hidden_dim=32,
            rngs=rngs,
        )

        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        x = jax.random.normal(rng_key, (1, 1, 16, 16))

        grads = nnx.grad(loss_fn)(am_fno, x)

        # Verify gradients exist
        assert grads is not None

        # Check that gradients are not all zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            jnp.linalg.norm(leaf) for leaf in grad_leaves if hasattr(leaf, "shape")
        ]
        assert len(grad_norms) > 0
        assert any(norm > 1e-8 for norm in grad_norms)

    def test_amortized_fno_memory_efficiency(self, rngs, rng_key):
        """Test AM-FNO memory efficiency for amortized computation."""
        am_fno = AmortizedFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes=(16, 16),
            num_layers=2,
            kernel_hidden_dim=64,
            rngs=rngs,
        )

        # Test with moderate size data
        batch_size = 1
        x = jax.random.normal(rng_key, (batch_size, 2, 32, 32))

        # Should handle computation efficiently
        output = am_fno(x)

        assert jnp.all(jnp.isfinite(output))
        assert output.ndim == 4  # Should be 4D tensor

    def test_amortized_fno_multi_layer_processing(self, rngs, rng_key):
        """Test AM-FNO with multiple layers."""
        num_layers_list = [1, 2, 3]

        for num_layers in num_layers_list:
            am_fno = AmortizedFourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=16,
                modes=(8, 8),
                num_layers=num_layers,
                kernel_hidden_dim=32,
                rngs=rngs,
            )

            x = jax.random.normal(rng_key, (1, 1, 16, 16))
            output = am_fno(x)

            # Check basic properties regardless of layer count
            assert jnp.all(jnp.isfinite(output))
            assert output.ndim == 4

    def test_amortized_fno_jax_transformations(self, rngs, rng_key):
        """Test AM-FNO with JAX transformations (jit)."""
        am_fno = AmortizedFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(8, 8),
            num_layers=2,
            kernel_hidden_dim=32,
            rngs=rngs,
        )

        x = jax.random.normal(rng_key, (1, 1, 16, 16))

        # Test JIT compilation
        @jax.jit
        def jitted_forward(x):
            return am_fno(x)

        output_jit = jitted_forward(x)
        output_regular = am_fno(x)

        # Outputs should be close (allowing for numerical differences from JIT compilation)
        assert jnp.allclose(output_jit, output_regular, rtol=1e-2, atol=1e-4)

    def test_amortized_fno_kernel_adaptation(self, rngs, rng_key):
        """Test AM-FNO kernel adaptation to different input patterns."""
        am_fno = AmortizedFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=24,
            modes=(8, 8),
            num_layers=2,
            kernel_hidden_dim=48,
            rngs=rngs,
        )

        # Test with different input patterns
        # Smooth pattern
        x_smooth = jnp.ones((1, 1, 16, 16))
        output_smooth = am_fno(x_smooth)

        # Random pattern
        x_random = jax.random.normal(rng_key, (1, 1, 16, 16))
        output_random = am_fno(x_random)

        # Both should produce finite outputs
        assert jnp.all(jnp.isfinite(output_smooth))
        assert jnp.all(jnp.isfinite(output_random))

        # Outputs should be different for different inputs
        assert not jnp.allclose(output_smooth, output_random, rtol=1e-3)
