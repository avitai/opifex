"""Test Tensorized Fourier Neural Operator implementation.

Clean tests for TensorizedFourierNeuralOperator with simplified API.
Focuses on basic functionality testing with the corrected implementation.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.tensorized import TensorizedFourierNeuralOperator


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
