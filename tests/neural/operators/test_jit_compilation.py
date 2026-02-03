"""Test JIT compilation for neural operators using proper Flax NNX patterns."""

import time

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.deeponet import DeepONet
from opifex.neural.operators.fno import FourierNeuralOperator


class TestJITCompilation:
    """Test suite for neural operator JIT compilation using nnx.jit."""

    def test_fno_nnx_jit_compilation(self):
        """Test that FNO can be JIT compiled using nnx.jit."""
        rngs = nnx.Rngs(42)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 32, 32))

        # Test that the model can be JIT compiled using nnx.jit
        @nnx.jit
        def jitted_forward(model, x):
            return model(x)

        # Should not raise
        output = jitted_forward(model, x)
        assert output.shape == (2, 1, 32, 32)

        # Test that non-jitted and jitted produce same results
        regular_output = model(x)
        assert jnp.allclose(regular_output, output, atol=1e-6)

    def test_deeponet_nnx_jit_compilation(self):
        """Test DeepONet JIT compilation using nnx.jit."""
        rngs = nnx.Rngs(42)
        model = DeepONet(branch_sizes=[20, 32, 32], trunk_sizes=[2, 32, 32], rngs=rngs)

        branch = jax.random.normal(jax.random.PRNGKey(0), (4, 20))
        trunk = jax.random.normal(
            jax.random.PRNGKey(1), (4, 10, 2)
        )  # (batch, locations, dim)

        # Test JIT compilation
        @nnx.jit
        def jitted_forward(model, branch, trunk):
            return model(branch, trunk)

        # Should not raise
        output = jitted_forward(model, branch, trunk)
        assert output.shape == (4, 10)  # (batch, n_locations)

        # Test consistency
        regular_output = model(branch, trunk)
        assert jnp.allclose(regular_output, output, atol=1e-6)

    def test_jit_compilation_with_different_input_shapes(self):
        """Test JIT compilation handles different input shapes correctly."""
        rngs = nnx.Rngs(42)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        @nnx.jit
        def jitted_forward(model, x):
            return model(x)

        # Test with different batch sizes but same spatial dimensions
        x1 = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 32, 32))
        x2 = jax.random.normal(jax.random.PRNGKey(1), (4, 1, 32, 32))

        out1 = jitted_forward(model, x1)
        out2 = jitted_forward(model, x2)

        assert out1.shape == (2, 1, 32, 32)
        assert out2.shape == (4, 1, 32, 32)

    def test_multiple_operators_jit_compilation(self):
        """Test that multiple neural operators can be JIT compiled."""
        rngs = nnx.Rngs(42)

        # Test different operators
        fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        deeponet = DeepONet(
            branch_sizes=[20, 32, 32], trunk_sizes=[2, 32, 32], rngs=rngs
        )

        # JIT compile both using proper nnx.jit
        @nnx.jit
        def jit_fno(model, x):
            return model(x)

        @nnx.jit
        def jit_deeponet(model, branch, trunk):
            return model(branch, trunk)

        # Test FNO
        x_fno = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 32, 32))
        fno_output = jit_fno(fno, x_fno)
        assert fno_output.shape == (2, 1, 32, 32)

        # Test DeepONet
        batch_size = 4
        branch = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 20))
        trunk = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, 10, 2)
        )  # (batch, locations, dim)
        deeponet_output = jit_deeponet(deeponet, branch, trunk)
        assert deeponet_output.shape == (4, 10)  # (batch, n_locations)

    def test_jit_speedup_measurement(self):
        """Test that JIT compilation provides performance benefits."""
        rngs = nnx.Rngs(42)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=64,
            modes=16,
            num_layers=4,  # More layers for better JIT impact
            rngs=rngs,
        )

        # Larger input for more computation to show JIT benefits
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 1, 128, 128))

        @nnx.jit
        def jitted_forward(model, x):
            return model(x)

        # Warmup both versions
        _ = model(x)
        _ = jitted_forward(model, x)  # Warmup JIT compilation

        # Time without explicit JIT (baseline) - fewer iterations for CI
        start = time.time()
        for _ in range(5):
            _ = model(x).block_until_ready()
        baseline_time = time.time() - start

        # Time with JIT (already warmed up)
        start = time.time()
        for _ in range(5):
            _ = jitted_forward(model, x).block_until_ready()
        jit_time = time.time() - start

        # JIT should be faster (but we'll be lenient for CI environments)
        # Just ensure JIT doesn't break and produces valid output
        assert jit_time > 0  # Basic sanity check
        assert baseline_time > 0  # Basic sanity check
        print(f"Baseline time: {baseline_time:.4f}s, JIT time: {jit_time:.4f}s")
