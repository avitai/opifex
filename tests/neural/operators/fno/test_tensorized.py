"""Test Tensorized Fourier Neural Operator implementation.

Clean tests for TensorizedFourierNeuralOperator with simplified API.
Focuses on basic functionality testing with the corrected implementation.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.tensorized import (
    CPDecomposition,
    TensorizedFourierNeuralOperator,
    TensorizedSpectralConvolution,
    TensorTrainDecomposition,
    TuckerDecomposition,
)


class TestTensorizedFourierNeuralOperator:
    """Test suite for TensorizedFourierNeuralOperator with clean API."""

    @pytest.fixture
    def sample_data_2d(self):
        """Create 2D sample data for testing."""
        batch_size = 4
        height, width = 32, 32
        in_channels = 3

        return jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, in_channels, height, width)
        )

    @pytest.fixture
    def sample_data_3d(self):
        """Create 3D sample data for testing."""
        batch_size = 2
        depth, height, width = 16, 16, 16
        in_channels = 2

        return jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, in_channels, depth, height, width)
        )

    def test_tucker_fno_initialization(self):
        """Test Tucker factorization FNO initialization."""
        model = TensorizedFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            modes=(8, 8),
            factorization="tucker",
            rank=0.1,
            rngs=nnx.Rngs(0),
        )

        # Test model was created successfully
        assert model.factorization == "tucker"
        assert hasattr(model, "tfno_layers")

    def test_tucker_fno_forward_pass(self, sample_data_2d):
        """Test Tucker FNO forward pass."""
        model = TensorizedFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=16,
            modes=(4, 4),  # Small modes to avoid dimension issues
            factorization="tucker",
            rank=0.1,
            rngs=nnx.Rngs(0),
        )

        output = model(sample_data_2d)

        expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_cp_fno_initialization(self):
        """Test CP factorization FNO initialization."""
        model = TensorizedFourierNeuralOperator(
            in_channels=4,
            out_channels=3,
            hidden_channels=64,
            modes=(16, 16),
            factorization="cp",
            rank=0.2,
            rngs=nnx.Rngs(1),
        )

        assert model.factorization == "cp"
        assert hasattr(model, "tfno_layers")

    def test_cp_fno_forward_pass(self, sample_data_2d):
        """Test CP FNO forward pass."""
        model = TensorizedFourierNeuralOperator(
            in_channels=3,
            out_channels=1,
            hidden_channels=32,
            modes=(8, 8),
            factorization="cp",
            rank=0.15,
            rngs=nnx.Rngs(1),
        )

        output = model(sample_data_2d)

        expected_shape = (sample_data_2d.shape[0], 1, *sample_data_2d.shape[2:])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_tt_fno_initialization(self):
        """Test Tensor Train factorization FNO initialization."""
        model = TensorizedFourierNeuralOperator(
            in_channels=2,
            out_channels=4,
            hidden_channels=48,
            modes=(12, 12),
            factorization="tt",
            rank=0.1,
            rngs=nnx.Rngs(2),
        )

        assert model.factorization == "tt"
        assert hasattr(model, "tfno_layers")

    def test_tt_fno_forward_pass(self, sample_data_2d):
        """Test Tensor Train FNO forward pass."""
        model = TensorizedFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            modes=(8, 8),
            factorization="tt",
            rank=0.1,
            rngs=nnx.Rngs(2),
        )

        output = model(sample_data_2d)

        expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_3d_tensorized_fno(self, sample_data_3d):
        """Test 3D Tensorized FNO."""
        model = TensorizedFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes=(4, 4, 4),
            factorization="tucker",
            rank=0.1,
            rngs=nnx.Rngs(3),
        )

        output = model(sample_data_3d)

        expected_shape = (sample_data_3d.shape[0], 1, *sample_data_3d.shape[2:])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_multi_layer_tensorized_fno(self, sample_data_2d):
        """Test multi-layer tensorized FNO."""
        model = TensorizedFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            modes=(8, 8),
            num_layers=6,
            factorization="tucker",
            rank=0.1,
            rngs=nnx.Rngs(9),
        )

        output = model(sample_data_2d)

        expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_gradient_computation(self, sample_data_2d):
        """Test gradient computation through tensorized layers."""
        model = TensorizedFourierNeuralOperator(
            in_channels=3,
            out_channels=1,
            hidden_channels=32,
            modes=(8, 8),
            factorization="tucker",
            rank=0.1,
            rngs=nnx.Rngs(10),
        )

        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        grads = nnx.grad(loss_fn)(model, sample_data_2d)

        # Check gradient properties
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)

    def test_jax_transformations(self, sample_data_2d):
        """Test compatibility with JAX transformations."""
        model = TensorizedFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            modes=(8, 8),
            factorization="tucker",
            rank=0.1,
            rngs=nnx.Rngs(11),
        )

        @jax.jit
        def jitted_forward(x):
            return model(x)

        output = jitted_forward(sample_data_2d)

        expected_shape = (sample_data_2d.shape[0], 2, *sample_data_2d.shape[2:])
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_compression_stats(self):
        """Test compression statistics computation."""
        model = TensorizedFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            modes=(8, 8),
            factorization="tucker",
            rank=0.1,
            rngs=nnx.Rngs(12),
        )

        # Get compression stats from one of the layers
        stats = model.tfno_layers[0].get_compression_stats()

        assert isinstance(stats, dict)
        assert "compression_ratio" in stats
        assert "parameter_reduction" in stats
        assert stats["compression_ratio"] > 0
        assert stats["parameter_reduction"] >= 0


class TestSpectralConvolutionNotIdentity:
    """Tests to verify spectral convolution actually transforms data.

    These tests exist to catch the bug where multiply_factorized() returns
    input unchanged (identity function).
    """

    def test_tucker_multiply_factorized_output_shape(self):
        """Tucker multiply_factorized should produce output with correct shape."""
        tensor_shape = (4, 3, 8, 8)  # out_ch, in_ch, mode1, mode2
        tucker = TuckerDecomposition(tensor_shape, rank=0.5, rngs=nnx.Rngs(42))

        # Create input matching the expected shape
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 3, 8, 8))  # batch, in_ch, m1, m2

        output = tucker.multiply_factorized(x)

        # Output should have out_channels (4) instead of in_channels (3)
        assert output.shape == (2, 4, 8, 8), (
            f"Expected (2, 4, 8, 8), got {output.shape}"
        )
        assert jnp.all(jnp.isfinite(output)), "Output should be finite"

    def test_cp_multiply_factorized_output_shape(self):
        """CP multiply_factorized should produce output with correct shape."""
        tensor_shape = (4, 3, 8, 8)
        cp = CPDecomposition(tensor_shape, rank=4, rngs=nnx.Rngs(42))

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 3, 8, 8))

        output = cp.multiply_factorized(x)

        assert output.shape == (2, 4, 8, 8), (
            f"Expected (2, 4, 8, 8), got {output.shape}"
        )
        assert jnp.all(jnp.isfinite(output)), "Output should be finite"

    def test_tt_multiply_factorized_output_shape(self):
        """TensorTrain multiply_factorized should produce output with correct shape."""
        tensor_shape = (4, 3, 8, 8)
        tt = TensorTrainDecomposition(tensor_shape, max_rank=4, rngs=nnx.Rngs(42))

        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 3, 8, 8))

        output = tt.multiply_factorized(x)

        assert output.shape == (2, 4, 8, 8), (
            f"Expected (2, 4, 8, 8), got {output.shape}"
        )
        assert jnp.all(jnp.isfinite(output)), "Output should be finite"

    def test_spectral_convolution_transforms_data(self):
        """TensorizedSpectralConvolution should transform Fourier coefficients."""
        conv = TensorizedSpectralConvolution(
            in_channels=3,
            out_channels=4,
            modes=(8, 8),
            decomposition_type="tucker",
            rank=0.5,
            rngs=nnx.Rngs(42),
        )

        key = jax.random.PRNGKey(0)
        # Simulate Fourier-transformed input
        x_ft = jax.random.normal(key, (2, 3, 16, 16)) + 1j * jax.random.normal(
            jax.random.PRNGKey(1), (2, 3, 16, 16)
        )

        output = conv(x_ft)

        # Output should have correct shape (out_channels=4) and same spatial dims
        assert output.shape == (2, 4, 16, 16), (
            f"Expected (2, 4, 16, 16), got {output.shape}"
        )
        assert jnp.all(jnp.isfinite(output)), "Output should be finite"

    def test_tfno_spectral_layers_contribute(self, sample_data_2d=None):
        """Verify TFNO spectral layers contribute to output beyond projections."""
        if sample_data_2d is None:
            sample_data_2d = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 32, 32))

        # Model with spectral layers
        model_with_layers = TensorizedFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=16,
            modes=(8, 8),
            num_layers=4,
            factorization="tucker",
            rank=0.5,
            rngs=nnx.Rngs(42),
        )

        # Model with 1 layer as baseline
        model_baseline = TensorizedFourierNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=16,
            modes=(8, 8),
            num_layers=1,
            factorization="tucker",
            rank=0.5,
            rngs=nnx.Rngs(42),  # Same seed for fair comparison
        )

        output_layers = model_with_layers(sample_data_2d)
        output_baseline = model_baseline(sample_data_2d)

        # Both should produce valid outputs with correct shape
        assert output_layers.shape == (2, 2, 32, 32)
        assert output_baseline.shape == (2, 2, 32, 32)
        assert jnp.all(jnp.isfinite(output_layers))
        assert jnp.all(jnp.isfinite(output_baseline))

        # Outputs should differ (more layers should produce different results)
        assert not jnp.allclose(output_layers, output_baseline), (
            "Adding spectral layers should change output"
        )
