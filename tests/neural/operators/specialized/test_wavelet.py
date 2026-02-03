"""Test Wavelet Neural Operator (WNO).

Test suite for Wavelet Neural Operator implementation for multi-scale feature
extraction with discrete wavelet transforms and learnable wavelets.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.specialized.wavelet import WaveletNeuralOperator


class TestWaveletNeuralOperator:
    """Test Wavelet Neural Operator for multi-scale feature extraction."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_wavelet_neural_operator_initialization(self, rngs):
        """Test WNO initialization with FLAX NNX and GPU/CPU compatibility."""
        in_channels = 2
        out_channels = 1
        hidden_channels = 32
        num_levels = 3

        operator = WaveletNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            wavelet_type="db4",
            rngs=rngs,
        )

        assert operator.in_channels == in_channels
        assert operator.out_channels == out_channels
        assert operator.hidden_channels == hidden_channels
        assert operator.num_levels == num_levels
        assert len(operator.wavelet_processors) == num_levels
        assert operator.low_pass_filter.shape == (8,)  # db4 has 8 coefficients

    def test_wavelet_neural_operator_learnable_wavelets(self, rngs):
        """Test WNO with learnable wavelets and GPU/CPU compatibility."""
        in_channels = 1
        out_channels = 1
        hidden_channels = 16
        num_levels = 2

        operator = WaveletNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            use_learnable_wavelets=True,
            rngs=rngs,
        )

        # Learnable wavelets should be initialized randomly
        assert operator.use_learnable_wavelets is True
        assert operator.low_pass_filter.shape == (8,)
        assert operator.high_pass_filter.shape == (8,)

    def test_wavelet_neural_operator_forward(self, rngs):
        """Test WNO forward pass with FLAX NNX and GPU/CPU compatibility."""
        # Dimensions compatible with both GPU and CPU
        batch_size = 4
        in_channels = 1
        out_channels = 1
        hidden_channels = 16
        num_levels = 2
        spatial_size = 64  # Must be power of 2 for clean wavelet decomposition

        operator = WaveletNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            rngs=rngs,
        )

        # Create input data compatible with wavelets (power of 2 size)
        x = jnp.ones((batch_size, in_channels, spatial_size))
        output = operator(x)

        expected_shape = (batch_size, out_channels, spatial_size)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_wavelet_dwt_idwt_reconstruction(self, rngs):
        """Test DWT/IDWT signal reconstruction with GPU/CPU compatibility."""
        in_channels = 1
        out_channels = 1
        hidden_channels = 8
        num_levels = 1

        operator = WaveletNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            rngs=rngs,
        )

        # Create a simple test signal
        signal = jnp.sin(jnp.linspace(0, 4 * jnp.pi, 32))

        # Apply DWT
        approx, detail = operator._dwt_1d(signal)

        # Apply IDWT
        reconstructed = operator._idwt_1d(approx, detail)

        # Should approximately reconstruct (allowing for boundary effects)
        # Check that most of the signal is preserved
        reconstruction_error = jnp.mean(jnp.abs(signal - reconstructed))
        assert reconstruction_error < 0.5
        assert jnp.all(jnp.isfinite(reconstructed))

    def test_wavelet_neural_operator_multi_level(self, rngs):
        """Test multi-level wavelet decomposition and reconstruction."""
        operator = WaveletNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_levels=3,
            rngs=rngs,
        )

        # Create test signal
        signal = jnp.sin(jnp.linspace(0, 2 * jnp.pi, 64))

        # Multi-level decomposition
        coefficients = operator._multi_level_dwt(signal)

        assert len(coefficients) == 3
        # Each level should have approximation and detail coefficients
        for level_coeffs in coefficients:
            assert len(level_coeffs) == 2  # (approx, detail)

    def test_wavelet_neural_operator_differentiability(self, rngs):
        """Test WNO differentiability with FLAX NNX and GPU/CPU compatibility."""
        in_channels = 1
        out_channels = 1
        hidden_channels = 8
        num_levels = 2

        operator = WaveletNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            rngs=rngs,
        )

        def loss_fn(model, x):
            output = model(x)
            return jnp.sum(output**2)

        # Power of 2 size for wavelet compatibility
        x = jnp.ones((2, in_channels, 32))
        grads = nnx.grad(loss_fn)(operator, x)

        # Verify gradients exist for key components
        assert hasattr(grads, "input_proj")
        assert hasattr(grads, "output_proj")
        assert hasattr(grads, "low_pass_filter")
        assert hasattr(grads, "high_pass_filter")

        # Check that gradients are not all zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            jnp.linalg.norm(leaf) for leaf in grad_leaves if hasattr(leaf, "shape")
        ]
        assert len(grad_norms) > 0
        assert any(norm > 1e-8 for norm in grad_norms)
