"""Test generic performance characteristics across neural operators.

Common performance testing utilities for all neural operator implementations
including memory profiling, computational efficiency, and JAX compatibility.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.base import FourierNeuralOperator


class TestGenericPerformance:
    """Test generic performance characteristics across operators."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_operator_memory_efficiency(self, rngs):
        """Test operator memory efficiency with various input sizes."""
        fno = FourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        # Test different input sizes
        sizes = [16, 32, 64]
        for size in sizes:
            x = jax.random.normal(rngs.params(), (1, 2, size, size))
            output = fno(x)

            assert output.shape == (1, 1, size, size)
            assert jnp.all(jnp.isfinite(output))

    def test_batch_processing_efficiency(self, rngs):
        """Test operator efficiency with different batch sizes."""
        fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=4,
            num_layers=1,
            rngs=rngs,
        )

        # Test different batch sizes
        batch_sizes = [1, 2, 4]
        spatial_size = 16

        for batch_size in batch_sizes:
            x = jax.random.normal(
                rngs.params(),
                (batch_size, 1, spatial_size, spatial_size),
            )
            output = fno(x)

            assert output.shape == (batch_size, 1, spatial_size, spatial_size)
            assert jnp.all(jnp.isfinite(output))

    def test_jax_transformations(self, rngs, rng_key):
        """Test operator compatibility with JAX transformations."""
        fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=4,
            num_layers=1,
            rngs=rngs,
        )

        x = jax.random.normal(rng_key, (2, 1, 8, 8))

        # Test regular forward pass
        output_regular = fno(x)

        # Test JIT compilation - fix the tracer issue
        @jax.jit
        def jit_forward(inputs):
            # Call the model directly on inputs, not passing the model as argument
            return fno(inputs)

        output_jit = jit_forward(x)

        # Test that outputs are equivalent (allowing for compilation-induced precision differences)
        assert jnp.allclose(output_regular, output_jit, rtol=1e-4, atol=1e-6)
        assert jnp.all(jnp.isfinite(output_jit))

        # Test vmap
        @jax.vmap
        def vmap_forward(inputs):
            # Remove batch dimension for vmap
            return fno(jnp.expand_dims(inputs, 0))

        # Apply vmap to individual samples
        x_unbatched = x  # Shape: (2, 1, 8, 8)
        output_vmap = vmap_forward(x_unbatched)  # Shape: (2, 1, 1, 8, 8)
        output_vmap = jnp.squeeze(output_vmap, axis=2)  # Remove extra dim: (2, 1, 8, 8)

        assert output_vmap.shape == output_regular.shape
        assert jnp.all(jnp.isfinite(output_vmap))

    def test_gradient_computation_efficiency(self, rngs):
        """Test efficient gradient computation across operators."""
        fno = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )

        def loss_fn(model, x):
            pred = model(x)
            return jnp.mean(pred**2)

        x = jax.random.normal(rngs.params(), (2, 1, 16, 16))

        # Compute gradients
        grads = nnx.grad(loss_fn)(fno, x)

        # Check gradient properties
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0

        # Check all gradients are finite
        all_finite_leaves = [jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves]
        assert all(jnp.asarray(all_finite_leaves))

        # Check gradients are not all zero (proper gradient flow)
        grad_norms = [jnp.linalg.norm(leaf) for leaf in grad_leaves]
        assert any(norm > 1e-6 for norm in grad_norms)

    def test_operator_scalability(self, rngs):
        """Test operator scalability with increasing complexity."""
        # Test small operator
        fno_small = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            modes=2,
            num_layers=1,
            rngs=rngs,
        )

        # Test medium operator
        fno_medium = FourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )

        # Test both with appropriate inputs
        x_small = jax.random.normal(rngs.params(), (1, 1, 8, 8))
        x_medium = jax.random.normal(rngs.params(), (1, 2, 16, 16))

        output_small = fno_small(x_small)
        output_medium = fno_medium(x_medium)

        # Both should produce valid finite outputs
        assert jnp.all(jnp.isfinite(output_small))
        assert jnp.all(jnp.isfinite(output_medium))

    def test_parameter_count_efficiency(self, rngs):
        """Test parameter count scaling with operator complexity."""

        def count_parameters(model):
            return sum(
                jnp.prod(jnp.array(param.shape))
                for param in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
            )

        # Small operator
        fno_small = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            modes=2,
            num_layers=1,
            rngs=rngs,
        )

        # Large operator
        fno_large = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        params_small = count_parameters(fno_small)
        params_large = count_parameters(fno_large)

        # Large operator should have more parameters
        assert params_large > params_small

        # Both should have reasonable parameter counts
        assert params_small > 0
        assert params_large > 0

        print(f"Small FNO parameters: {params_small}")
        print(f"Large FNO parameters: {params_large}")
        print(f"Parameter ratio: {params_large / params_small:.2f}")
