"""Test Fourier Neural Operator base components.

Test suite for basic FNO components including spectral convolution,
Fourier layers, and the complete FNO architecture.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.base import (
    FourierLayer,
    FourierNeuralOperator,
    FourierSpectralConvolution,
)


class TestFourierSpectralConvolution:
    """Test spectral convolution layer for FNO."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_spectral_convolution_initialization(self, rngs):
        """Test Fourier spectral convolution initialization."""
        in_channels = 16
        out_channels = 32
        modes = 8

        layer = FourierSpectralConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            rngs=rngs,
        )

        assert layer.in_channels == in_channels
        assert layer.out_channels == out_channels
        assert layer.modes == modes
        assert hasattr(layer, "weights")

    def test_spectral_convolution_forward(self, rngs):
        """Test Fourier spectral convolution forward pass."""
        batch_size = 4
        in_channels = 8
        out_channels = 16
        modes = 6
        grid_size = 32

        layer = FourierSpectralConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            rngs=rngs,
        )

        # Create complex input (Fourier domain)
        x = jnp.ones((batch_size, in_channels, grid_size // 2 + 1))

        output = layer(x)

        # Check output shape and type
        expected_shape = (batch_size, out_channels, grid_size // 2 + 1)
        assert output.shape == expected_shape

    def test_spectral_convolution_gradient_computation(self, rngs):
        """Test Fourier spectral convolution gradient computation."""
        layer = FourierSpectralConvolution(
            in_channels=8, out_channels=16, modes=8, rngs=rngs
        )

        def loss_fn(layer, x):
            return jnp.sum(jnp.abs(layer(x)) ** 2)

        x = jnp.ones((4, 8, 17))

        # Should not raise error
        grads = nnx.grad(loss_fn)(layer, x)
        assert hasattr(grads, "weights")


class TestFourierLayer:
    """Test Fourier layer for FNO."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_fourier_layer_initialization(self, rngs):
        """Test Fourier layer initialization."""
        in_channels = 32
        out_channels = 64
        modes = 16

        layer = FourierLayer(
            in_channels=in_channels, out_channels=out_channels, modes=modes, rngs=rngs
        )

        assert layer.in_channels == in_channels
        assert layer.out_channels == out_channels
        assert layer.modes == modes
        assert hasattr(layer, "spectral_conv")
        assert hasattr(layer, "linear")

    def test_fourier_layer_forward(self, rngs):
        """Test Fourier layer forward pass."""
        batch_size = 4
        in_channels = 16
        out_channels = 32
        modes = 12
        grid_size = 64

        layer = FourierLayer(
            in_channels=in_channels, out_channels=out_channels, modes=modes, rngs=rngs
        )

        # Create input in spatial domain
        x = jnp.ones((batch_size, in_channels, grid_size))

        output = layer(x)

        # Check output shape
        expected_shape = (batch_size, out_channels, grid_size)
        assert output.shape == expected_shape

    def test_fourier_layer_activation(self, rngs):
        """Test Fourier layer with different activations."""
        rngs = nnx.Rngs(0)

        # FIXED: Use channels that match properly with linear layer
        in_channels = 8
        out_channels = 8

        layer_default = FourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=8,
            rngs=rngs,
        )

        layer_custom = FourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=8,
            activation=nnx.relu,
            rngs=rngs,
        )

        x = jax.random.normal(
            rngs.params(),
            (2, in_channels, 16, 16),
        )

        out_default = layer_default(x)
        out_custom = layer_custom(x)

        assert out_default.shape == out_custom.shape
        # ReLU should zero out negative values
        assert jnp.all(out_custom >= 0)


class TestFourierNeuralOperator:
    """Test complete Fourier Neural Operator."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_fno_initialization(self, rngs):
        """Test FNO initialization."""
        fno = FourierNeuralOperator(
            in_channels=3,
            out_channels=1,
            hidden_channels=64,
            modes=16,
            num_layers=4,
            rngs=rngs,
        )

        assert fno.out_channels == 1
        assert fno.hidden_channels == 64
        assert fno.modes == 16
        assert fno.num_layers == 4
        assert len(fno.fourier_layers) == 4

    def test_fno_tucker_factorization(self, rngs):
        """Test FNO with Tucker factorization for parameter reduction."""
        # Create FNO with Tucker factorization enabled
        fno_tucker = FourierNeuralOperator(
            in_channels=32,
            out_channels=16,
            hidden_channels=128,
            modes=32,
            num_layers=4,
            factorization_type="tucker",
            factorization_rank=16,
            rngs=rngs,
        )

        # Create regular FNO for comparison
        fno_regular = FourierNeuralOperator(
            in_channels=32,
            out_channels=16,
            hidden_channels=128,
            modes=32,
            num_layers=4,
            rngs=rngs,
        )

        # Function to count parameters
        def count_parameters(model):
            return sum(
                jnp.prod(jnp.array(param.shape))
                for param in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
            )

        tucker_params = count_parameters(fno_tucker)
        regular_params = count_parameters(fno_regular)

        # Tucker factorization should reduce parameter count
        assert tucker_params < regular_params

    def test_fno_cp_factorization(self, rngs):
        """Test FNO with CP factorization for parameter reduction."""
        # Create FNO with CP factorization
        fno_cp = FourierNeuralOperator(
            in_channels=16,
            out_channels=8,
            hidden_channels=64,
            modes=16,
            num_layers=3,
            factorization_type="cp",
            factorization_rank=8,
            rngs=rngs,
        )

        # Create regular FNO for comparison
        fno_regular = FourierNeuralOperator(
            in_channels=16,
            out_channels=8,
            hidden_channels=64,
            modes=16,
            num_layers=3,
            rngs=rngs,
        )

        # Function to count parameters
        def count_parameters(model):
            return sum(
                jnp.prod(jnp.array(param.shape))
                for param in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
            )

        cp_params = count_parameters(fno_cp)
        regular_params = count_parameters(fno_regular)

        # CP factorization should reduce parameter count
        assert cp_params < regular_params

    def test_fno_forward_pass(self, rngs, rng_key):
        """Test FNO forward pass."""
        batch_size = 2
        in_channels = 3
        out_channels = 1
        grid_size = 64

        fno = FourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=32,
            modes=16,
            num_layers=3,
            rngs=rngs,
        )

        x = jax.random.normal(rng_key, (batch_size, in_channels, grid_size))

        output = fno(x)

        expected_shape = (batch_size, out_channels, grid_size)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_fno_2d_forward_pass(self, rngs, rng_key):
        """Test FNO 2D forward pass."""
        batch_size = 2
        in_channels = 2
        out_channels = 1
        spatial_size = 32

        fno = FourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=32,
            modes=16,
            num_layers=2,
            rngs=rngs,
        )

        x = jax.random.normal(
            rng_key,
            (batch_size, in_channels, spatial_size, spatial_size),
        )

        output = fno(x)

        expected_shape = (batch_size, out_channels, spatial_size, spatial_size)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_fno_gradient_computation(self, rngs, rng_key):
        """Test FNO gradient computation."""
        fno = FourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        def loss_fn(model, x):
            pred = model(x)
            return jnp.mean(pred**2)

        x = jax.random.normal(rng_key, (1, 2, 32))

        # Compute gradients
        grads = nnx.grad(loss_fn)(fno, x)

        # Check gradient properties
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)

    def test_fno_jax_transformations(self, rngs, rng_key):
        """Test FNO compatibility with JAX transformations."""
        fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        x = jax.random.normal(rng_key, (2, 1, 16))

        # Test JIT compilation
        @jax.jit
        def jitted_forward(inputs):
            return fno(inputs)

        output_jit = jitted_forward(x)

        # Test vmap
        @jax.vmap
        def vmapped_forward(inputs):
            return fno(jnp.expand_dims(inputs, 0))

        x_unbatched = x  # Shape: (2, 1, 16)
        output_vmap = vmapped_forward(x_unbatched)
        output_vmap = jnp.squeeze(output_vmap, axis=1)  # Remove extra dimension

        # Check outputs
        assert jnp.all(jnp.isfinite(output_jit))
        assert jnp.all(jnp.isfinite(output_vmap))

    def test_fno_different_activations(self, rngs):
        """Test FNO with different activation functions."""
        activations = [nnx.relu, nnx.gelu, nnx.silu, nnx.tanh]
        activation_names = ["relu", "gelu", "silu", "tanh"]

        x = jax.random.normal(rngs.params(), (1, 2, 16))
        outputs = {}

        for activation, name in zip(activations, activation_names, strict=False):
            fno = FourierNeuralOperator(
                in_channels=2,
                out_channels=1,
                hidden_channels=16,
                modes=8,
                num_layers=2,
                activation=activation,
                rngs=rngs,
            )

            outputs[name] = fno(x)
            assert outputs[name].shape == (1, 1, 16)
            assert jnp.all(jnp.isfinite(outputs[name]))

        # Different activations should produce different outputs
        output_values = list(outputs.values())
        for i in range(len(output_values)):
            for j in range(i + 1, len(output_values)):
                assert not jnp.allclose(output_values[i], output_values[j], rtol=1e-4)

    def test_fno_parameter_scaling(self, rngs):
        """Test FNO parameter scaling with model size."""

        def count_parameters(model):
            return sum(
                jnp.prod(jnp.array(param.shape))
                for param in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
            )

        # Small model
        fno_small = FourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )

        # Large model
        fno_large = FourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=64,
            modes=16,
            num_layers=4,
            rngs=rngs,
        )

        small_params = count_parameters(fno_small)
        large_params = count_parameters(fno_large)

        # Large model should have more parameters
        assert large_params > small_params
        print(f"Small FNO: {small_params} parameters")
        print(f"Large FNO: {large_params} parameters")
        print(f"Parameter ratio: {large_params / small_params:.2f}")

    def test_fno_gelu_activation(self, rngs):
        """Test FNO with gelu activation."""
        fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=8,
            num_layers=3,
            activation=nnx.gelu,
            rngs=rngs,
        )

        x = jax.random.normal(rngs.params(), (1, 1, 32, 32))
        output = fno(x)

        assert output.shape == (1, 1, 32, 32)
        assert jnp.all(jnp.isfinite(output))
