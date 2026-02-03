"""Test Multi-Scale Fourier Neural Operator.

Test suite for Multi-Scale FNO implementation with cross-scale attention
and gradient checkpointing.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.foundations import MultiScaleFourierNeuralOperator


class TestMultiScaleFourierNeuralOperator:
    """Test Multi-Scale Fourier Neural Operator."""

    def setup_method(self):
        """Setup for each test method with GPU/CPU backend detection."""
        self.backend = jax.default_backend()
        print(f"Running MultiScaleFourierNeuralOperator tests on {self.backend}")

    def test_multi_scale_fno_initialization(self):
        """Test Multi-Scale FNO initialization with GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        operator = MultiScaleFourierNeuralOperator(
            in_channels=3,
            out_channels=1,
            hidden_channels=64,
            modes_per_scale=[16, 8, 4],
            num_layers_per_scale=[2, 2, 2],
            rngs=rngs,
        )

        assert operator.in_channels == 3
        assert operator.out_channels == 1
        assert operator.hidden_channels == 64
        assert operator.num_scales == 3
        assert len(operator.scale_layers) == 3
        assert len(operator.scale_layers[0]) == 2  # 2 layers for first scale

    def test_multi_scale_fno_forward_1d(self):
        """Test Multi-Scale FNO forward pass for 1D input with GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)
        batch_size = 4
        in_channels = 2
        out_channels = 1
        grid_size = 128

        operator = MultiScaleFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=32,
            modes_per_scale=[16, 8, 4],
            num_layers_per_scale=[1, 1, 1],
            use_cross_scale_attention=True,
            rngs=rngs,
        )

        x = jnp.ones((batch_size, in_channels, grid_size))
        output = operator(x)

        expected_shape = (batch_size, out_channels, grid_size)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_multi_scale_fno_forward_2d(self):
        """Test Multi-Scale FNO forward pass for 2D input with GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)
        batch_size = 2
        in_channels = 1
        out_channels = 1
        grid_size = 64

        operator = MultiScaleFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=32,
            modes_per_scale=[8, 4],
            num_layers_per_scale=[1, 1],
            use_cross_scale_attention=False,  # Test without attention
            rngs=rngs,
        )

        x = jnp.ones((batch_size, in_channels, grid_size, grid_size))
        output = operator(x)

        expected_shape = (batch_size, out_channels, grid_size, grid_size)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_multi_scale_fno_gradient_checkpointing(self):
        """Test Multi-Scale FNO with gradient checkpointing and GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        operator = MultiScaleFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes_per_scale=[8, 4],
            num_layers_per_scale=[2, 2],
            use_gradient_checkpointing=True,
            rngs=rngs,
        )

        x = jnp.ones((2, 2, 64))

        def loss_fn(model, x):
            return jnp.sum(model(x) ** 2)

        # Should not raise error with gradient checkpointing
        grads = nnx.grad(loss_fn)(operator, x)
        assert hasattr(grads, "input_proj")

    def test_multi_scale_fno_differentiability(self):
        """Test Multi-Scale FNO differentiability with GPU/CPU compatibility."""
        rngs = nnx.Rngs(42)

        operator = MultiScaleFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes_per_scale=[4, 2],
            num_layers_per_scale=[1, 1],
            rngs=rngs,
        )

        def loss_fn(model, x):
            return jnp.sum(model(x) ** 2)

        x = jnp.ones((2, 1, 32))
        grads = nnx.grad(loss_fn)(operator, x)

        assert hasattr(grads, "input_proj")
        assert hasattr(grads, "output_proj")

        # Verify output properties
        output = operator(x)
        assert output.shape == (2, 1, 32)
        assert jnp.all(jnp.isfinite(output))

    def test_multi_scale_fno_vs_regular_fno_performance(self):
        """Test performance comparison between multi-scale and regular FNO (migrated from original test suite)."""
        # Original missing test: test_multi_scale_fno_vs_regular_fno_performance
        rngs_ms = nnx.Rngs(42)
        rngs_regular = nnx.Rngs(42)

        # Create multi-scale FNO
        ms_fno = MultiScaleFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes_per_scale=[16, 8, 4],  # Multiple scales
            num_layers_per_scale=[2, 2, 2],
            rngs=rngs_ms,
        )

        # Create regular FNO for comparison
        regular_fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=16,  # Single scale
            num_layers=4,
            rngs=rngs_regular,
        )

        # Test on same input
        batch_size = 4
        grid_size = 64
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (batch_size, 1, grid_size),
        )

        # Forward pass timing and accuracy comparison
        ms_output = ms_fno(x)
        regular_output = regular_fno(x)

        # Both should produce valid outputs
        assert ms_output.shape == (batch_size, 1, grid_size)
        assert regular_output.shape == (batch_size, 1, grid_size)
        assert jnp.isfinite(ms_output).all()
        assert jnp.isfinite(regular_output).all()

        # Multi-scale should capture different frequency components
        # (In practice, this would be validated against known multi-scale problems)
        ms_spectrum = jnp.abs(jnp.fft.fft(ms_output[0, 0, :]))
        regular_spectrum = jnp.abs(jnp.fft.fft(regular_output[0, 0, :]))

        # Spectra should be different due to different frequency handling
        assert not jnp.allclose(ms_spectrum, regular_spectrum, atol=1e-6)

    def test_multi_scale_fno_forward(self):
        """Test MultiScaleFourierNeuralOperator forward pass."""
        rngs = nnx.Rngs(42)

        ms_fno = MultiScaleFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=64,
            modes_per_scale=[16, 8, 4],
            num_layers_per_scale=[2, 2, 2],
            rngs=rngs,
        )

        batch_size = 4
        grid_size = 128
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 1, grid_size))

        output = ms_fno(x)

        assert output.shape == (batch_size, 1, grid_size)
        assert jnp.isfinite(output).all()

    def test_multi_scale_fno_different_scales(self):
        """Test MultiScaleFourierNeuralOperator with different scale configurations."""
        rngs = nnx.Rngs(42)

        # Test with three scales
        ms_fno_3scale = MultiScaleFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes_per_scale=[32, 16, 8],
            num_layers_per_scale=[2, 2, 2],
            rngs=rngs,
        )

        # Test with two scales
        ms_fno_2scale = MultiScaleFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes_per_scale=[16, 8],
            num_layers_per_scale=[2, 2],
            rngs=rngs,
        )

        batch_size = 2
        grid_size = 64
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 2, grid_size))

        output_3scale = ms_fno_3scale(x)
        output_2scale = ms_fno_2scale(x)

        assert output_3scale.shape == (batch_size, 1, grid_size)
        assert output_2scale.shape == (batch_size, 1, grid_size)
        assert jnp.isfinite(output_3scale).all()
        assert jnp.isfinite(output_2scale).all()

    def test_multi_scale_fno_gradient_computation(self):
        """Test MultiScaleFourierNeuralOperator gradient computation."""
        rngs = nnx.Rngs(42)

        ms_fno = MultiScaleFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes_per_scale=[8, 4],
            num_layers_per_scale=[1, 1],
            rngs=rngs,
        )

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        x = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 32))

        # Should not raise error
        grads = nnx.grad(loss_fn)(ms_fno, x)
        assert grads is not None
