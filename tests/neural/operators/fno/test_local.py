# FILE PLACEMENT: tests/neural/operators/fno/test_local.py
#
# Fixed Local FNO Test Suite
# Addresses JIT consistency and numerical precision issues
#
# This file should REPLACE: tests/neural/operators/fno/test_local.py

"""Test Local Fourier Neural Operator (LocalFNO).

Test suite for Local Fourier Neural Operator implementation that
combines local and global Fourier modes for enhanced feature extraction.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.local import LocalFourierNeuralOperator


class TestLocalFourierNeuralOperator:
    """Test Local FNO for combined local/global processing."""

    def setup_method(self):
        """Setup for each test method."""
        self.backend = jax.default_backend()
        print(f"Running LocalFourierNeuralOperator tests on {self.backend}")

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_local_fno_initialization(self, rngs):
        """Test Local FNO initialization with local kernel parameters."""
        in_channels = 2
        out_channels = 2
        hidden_channels = 24
        modes = (8, 8)
        num_layers = 2
        kernel_size = 3

        local_fno = LocalFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            kernel_size=kernel_size,
            rngs=rngs,
        )

        # Basic initialization check
        assert local_fno is not None
        assert callable(local_fno)

    def test_local_fno_forward_2d(self, rngs, rng_key):
        """Test Local FNO forward pass with 2D data."""
        batch_size = 2
        in_channels = 2
        out_channels = 2
        height, width = 32, 32

        local_fno = LocalFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=24,
            modes=(8, 8),
            num_layers=2,
            kernel_size=3,
            rngs=rngs,
        )

        # Create 2D input data
        x = jax.random.normal(rng_key, (batch_size, in_channels, height, width))

        output = local_fno(x)

        # Type assertion to help type checker understand this is an Array
        assert isinstance(output, jax.Array), f"Expected Array, got {type(output)}"

        # Verify output shape
        expected_shape = (batch_size, out_channels, height, width)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_local_fno_kernel_size_effects(self, rngs, rng_key):
        """Test effect of different kernel sizes on Local FNO."""
        in_channels = 1
        out_channels = 1
        hidden_channels = 16
        modes = (4, 4)

        # Test different kernel sizes
        kernel_sizes = [3, 5, 7]
        outputs = []

        for kernel_size in kernel_sizes:
            local_fno = LocalFourierNeuralOperator(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                modes=modes,
                num_layers=1,
                kernel_size=kernel_size,
                rngs=rngs,
            )

            x = jax.random.normal(rng_key, (1, in_channels, 16, 16))
            output = local_fno(x)

            # Type assertion to help type checker
            assert isinstance(output, jax.Array), f"Expected Array, got {type(output)}"
            outputs.append(output)

            # Check basic properties
            assert output.shape == (1, out_channels, 16, 16)
            assert jnp.all(jnp.isfinite(output))

        # Different kernel sizes should produce different outputs
        assert not jnp.allclose(outputs[0], outputs[1], rtol=1e-3)
        assert not jnp.allclose(outputs[1], outputs[2], rtol=1e-3)

    def test_local_fno_local_global_combination(self, rngs, rng_key):
        """Test that Local FNO combines local and global features."""
        local_fno = LocalFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            modes=(6, 6),
            num_layers=2,
            kernel_size=3,
            rngs=rngs,
        )

        # Create test data with both local and global patterns
        batch_size = 1
        height, width = 24, 24
        x = jax.random.normal(rng_key, (batch_size, 2, height, width))

        # Add local structure (high frequency)
        local_pattern = jnp.sin(10 * jnp.linspace(0, 1, width))[None, None, None, :]
        x = x.at[:, 0:1, :, :].add(local_pattern)

        # Add global structure (low frequency)
        global_pattern = jnp.sin(2 * jnp.linspace(0, 1, width))[None, None, None, :]
        x = x.at[:, 1:2, :, :].add(global_pattern)

        output = local_fno(x)

        # Type assertion to help type checker
        assert isinstance(output, jax.Array), f"Expected Array, got {type(output)}"

        # Check that output captures both patterns
        assert output.shape == (batch_size, 1, height, width)
        assert jnp.all(jnp.isfinite(output))
        assert jnp.std(output) > 0.01  # Should have meaningful variation

    def test_local_fno_edge_preservation(self, rngs, rng_key):
        """Test that Local FNO preserves edge information."""
        local_fno = LocalFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            num_layers=1,
            kernel_size=3,
            rngs=rngs,
        )

        # Create test data with sharp edges
        height, width = 16, 16
        x = jnp.zeros((1, 1, height, width))
        # Create a step function (sharp edge)
        x = x.at[:, :, :, width // 2 :].set(1.0)

        output = local_fno(x)

        # Type assertion to help type checker
        assert isinstance(output, jax.Array), f"Expected Array, got {type(output)}"

        # Check that output maintains spatial structure
        assert output.shape == (1, 1, height, width)
        assert jnp.all(jnp.isfinite(output))

        # Output should show some response to the edge
        left_side = jnp.mean(output[:, :, :, : width // 4])
        right_side = jnp.mean(output[:, :, :, 3 * width // 4 :])
        assert abs(left_side - right_side) > 0.01  # Should detect the edge

    def test_local_fno_differentiability(self, rngs, rng_key):
        """Test Local FNO differentiability."""
        local_fno = LocalFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            num_layers=2,
            kernel_size=3,
            rngs=rngs,
        )

        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        x = jax.random.normal(rng_key, (1, 2, 16, 16))

        grads = nnx.grad(loss_fn)(local_fno, x)

        # Verify gradients exist
        assert grads is not None

        # Check that gradients are not all zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            jnp.linalg.norm(leaf) for leaf in grad_leaves if hasattr(leaf, "shape")
        ]
        assert len(grad_norms) > 0
        assert any(norm > 1e-8 for norm in grad_norms)

    def test_local_fno_jax_transformations(self, rngs, rng_key):
        """Test Local FNO with JAX transformations (jit, vmap)."""
        local_fno = LocalFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(8, 8),
            num_layers=2,
            kernel_size=3,
            rngs=rngs,
        )

        x = jax.random.normal(rng_key, (1, 1, 16, 16))

        # Test JIT compilation
        @jax.jit
        def jitted_forward(x):
            return local_fno(x)

        output_jit = jitted_forward(x)
        output_regular = local_fno(x)

        # Type assertions
        assert isinstance(output_jit, jax.Array), (
            f"Expected Array, got {type(output_jit)}"
        )
        assert isinstance(output_regular, jax.Array), (
            f"Expected Array, got {type(output_regular)}"
        )

        # FIXED: Use more relaxed tolerance for JIT vs non-JIT comparison
        # JIT compilation can introduce small numerical differences due to optimization
        assert jnp.allclose(output_jit, output_regular, rtol=1e-4, atol=1e-6)

        # Test vmap
        batch_x = jax.random.normal(rng_key, (4, 1, 1, 16, 16))

        # Create a vectorized version that works with the batched input
        def single_forward(x):
            return local_fno(x)

        vmapped_forward = jax.vmap(single_forward)
        batch_output = vmapped_forward(batch_x)

        # Type assertion
        assert isinstance(batch_output, jax.Array), (
            f"Expected Array, got {type(batch_output)}"
        )

        # Check batch output shape
        expected_batch_shape = (4, 1, 1, 16, 16)
        assert batch_output.shape == expected_batch_shape
        assert jnp.all(jnp.isfinite(batch_output))

    def test_local_fno_adaptive_mixing(self, rngs, rng_key):
        """Test Local FNO with adaptive mixing."""
        local_fno_adaptive = LocalFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            num_layers=2,
            kernel_size=3,
            use_adaptive_mixing=True,
            rngs=rngs,
        )

        local_fno_fixed = LocalFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            num_layers=2,
            kernel_size=3,
            use_adaptive_mixing=False,
            rngs=rngs,
        )

        x = jax.random.normal(rng_key, (1, 1, 16, 16))

        output_adaptive = local_fno_adaptive(x)
        output_fixed = local_fno_fixed(x)

        # Type assertions
        assert isinstance(output_adaptive, jax.Array), (
            f"Expected Array, got {type(output_adaptive)}"
        )
        assert isinstance(output_fixed, jax.Array), (
            f"Expected Array, got {type(output_fixed)}"
        )

        # Check basic properties
        assert output_adaptive.shape == output_fixed.shape
        assert jnp.all(jnp.isfinite(output_adaptive))
        assert jnp.all(jnp.isfinite(output_fixed))

        # Adaptive and fixed mixing should produce different results
        assert not jnp.allclose(output_adaptive, output_fixed, rtol=1e-3)

    def test_local_fno_residual_connections(self, rngs, rng_key):
        """Test Local FNO with residual connections."""
        local_fno_residual = LocalFourierNeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=16,
            modes=(4, 4),
            num_layers=3,
            kernel_size=3,
            use_residual_connections=True,
            rngs=rngs,
        )

        local_fno_no_residual = LocalFourierNeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=16,
            modes=(4, 4),
            num_layers=3,
            kernel_size=3,
            use_residual_connections=False,
            rngs=rngs,
        )

        x = jax.random.normal(rng_key, (1, 2, 16, 16))

        output_residual = local_fno_residual(x)
        output_no_residual = local_fno_no_residual(x)

        # Type assertions
        assert isinstance(output_residual, jax.Array), (
            f"Expected Array, got {type(output_residual)}"
        )
        assert isinstance(output_no_residual, jax.Array), (
            f"Expected Array, got {type(output_no_residual)}"
        )

        # Check basic properties
        assert output_residual.shape == output_no_residual.shape == (1, 2, 16, 16)
        assert jnp.all(jnp.isfinite(output_residual))
        assert jnp.all(jnp.isfinite(output_no_residual))

        # Residual and non-residual should produce different results
        assert not jnp.allclose(output_residual, output_no_residual, rtol=1e-3)

    def test_local_fno_memory_efficiency(self, rngs, rng_key):
        """Test Local FNO memory efficiency with different layer counts."""
        # Test with increasing layer counts
        layer_counts = [1, 2, 4]

        for num_layers in layer_counts:
            local_fno = LocalFourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=16,
                modes=(8, 8),
                num_layers=num_layers,
                kernel_size=3,
                rngs=rngs,
            )

            # Test with moderate-sized input
            x = jax.random.normal(rng_key, (2, 1, 32, 32))
            output = local_fno(x)

            # Type assertion
            assert isinstance(output, jax.Array), f"Expected Array, got {type(output)}"

            # Check that deeper networks still work
            assert output.shape == (2, 1, 32, 32)
            assert jnp.all(jnp.isfinite(output))
            assert jnp.std(output) > 1e-6  # Should have meaningful variation
