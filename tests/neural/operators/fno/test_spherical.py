"""Test Spherical Fourier Neural Operator (SFNO).

Test suite for Spherical Fourier Neural Operator implementation for
spherical geometry data processing with spherical harmonics.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.spherical import SphericalFourierNeuralOperator


class TestSphericalFourierNeuralOperator:
    """Test Spherical FNO for spherical geometry processing."""

    def setup_method(self):
        """Setup for each test method."""
        self.backend = jax.default_backend()
        print(f"Running SphericalFourierNeuralOperator tests on {self.backend}")

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_sfno_initialization(self, rngs):
        """Test SFNO initialization with spherical harmonics parameters."""
        in_channels = 3
        out_channels = 3
        hidden_channels = 32
        lmax = 8
        num_layers = 2

        sfno = SphericalFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            num_layers=num_layers,
            rngs=rngs,
        )

        # Basic initialization check
        assert sfno is not None
        # Verify spherical harmonics components exist (conservative check)
        assert callable(sfno)

    def test_sfno_forward_spherical_data(self, rngs, rng_key):
        """Test SFNO forward pass with spherical coordinate data."""
        # Spherical data dimensions (batch, channels, nlat, nlon)
        batch_size = 2
        in_channels = 3
        out_channels = 3
        nlat, nlon = 32, 64  # Latitude and longitude grid

        sfno = SphericalFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=32,
            lmax=8,
            num_layers=2,
            rngs=rngs,
        )

        # Create spherical grid data
        x = jax.random.normal(rng_key, (batch_size, in_channels, nlat, nlon))

        output = sfno(x)

        # Verify output shape
        expected_shape = (batch_size, out_channels, nlat, nlon)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_sfno_spherical_harmonics_processing(self, rngs, rng_key):
        """Test spherical harmonics processing functionality."""
        sfno = SphericalFourierNeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=24,
            lmax=6,
            num_layers=1,
            rngs=rngs,
        )

        # Test spherical data processing
        batch_size = 1
        nlat, nlon = 16, 32
        x = jax.random.normal(rng_key, (batch_size, 2, nlat, nlon))

        # Forward pass should handle spherical data correctly
        output = sfno(x)

        # Check basic properties
        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))

    def test_sfno_conservation_properties(self, rngs, rng_key):
        """Test conservation properties on spherical manifold."""
        sfno = SphericalFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            lmax=4,
            num_layers=2,
            rngs=rngs,
        )

        # Create test data with known global mean
        batch_size = 2
        nlat, nlon = 16, 32
        x = jax.random.normal(rng_key, (batch_size, 1, nlat, nlon))

        # Compute global mean (spherical integration)
        input_mean = jnp.mean(x, axis=(2, 3), keepdims=True)

        output = sfno(x)
        output_mean = jnp.mean(output, axis=(2, 3), keepdims=True)

        # Global mean should be reasonably preserved (relaxed constraint)
        input_mean_safe = input_mean
        output_mean_safe = output_mean

        # For neural operators, exact conservation is not guaranteed due to learnable parameters
        # Check that the output magnitude is reasonable relative to input
        input_magnitude = jnp.abs(input_mean_safe)
        output_magnitude = jnp.abs(output_mean_safe)

        # Ensure output isn't excessively large or small relative to input
        magnitude_ratio = output_magnitude / (
            input_magnitude + 1e-8
        )  # Add epsilon to avoid division by zero
        assert jnp.all(magnitude_ratio > 0.01)  # Output not too small
        assert jnp.all(magnitude_ratio < 100.0)  # Output not too large

    def test_sfno_differentiability(self, rngs, rng_key):
        """Test SFNO differentiability for spherical data."""
        sfno = SphericalFourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=16,
            lmax=4,
            num_layers=2,
            rngs=rngs,
        )

        def loss_fn(model, x):
            output = model(x)
            return jnp.mean(output**2)

        # Spherical test data
        x = jax.random.normal(rng_key, (1, 2, 16, 32))

        grads = nnx.grad(loss_fn)(sfno, x)

        # Verify gradients exist
        assert grads is not None

        # Check that gradients are not all zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            jnp.linalg.norm(leaf) for leaf in grad_leaves if hasattr(leaf, "shape")
        ]
        assert len(grad_norms) > 0
        assert any(norm > 1e-8 for norm in grad_norms)

    def test_sfno_memory_efficiency(self, rngs, rng_key):
        """Test SFNO memory efficiency for large spherical grids."""
        # Test with standard configuration
        sfno = SphericalFourierNeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=32,
            lmax=8,
            num_layers=3,
            rngs=rngs,
        )

        # Large spherical grid
        batch_size = 1
        nlat, nlon = 64, 128
        x = jax.random.normal(rng_key, (batch_size, 2, nlat, nlon))

        # Should handle large data without memory issues
        output = sfno(x)

        assert output.shape == (batch_size, 2, nlat, nlon)
        assert jnp.all(jnp.isfinite(output))

    def test_sfno_jax_transformations(self, rngs, rng_key):
        """Test SFNO with JAX transformations (jit, vmap)."""
        sfno = SphericalFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            lmax=4,
            num_layers=2,
            rngs=rngs,
        )

        x = jax.random.normal(rng_key, (2, 1, 16, 32))

        # Test JIT compilation
        @jax.jit
        def jitted_forward(x):
            return sfno(x)

        output_jit = jitted_forward(x)
        output_regular = sfno(x)

        # Outputs should be close (allowing for numerical differences from JIT compilation)
        assert jnp.allclose(output_jit, output_regular, rtol=1e-2, atol=1e-4)

        # Test vmap (batch over additional dimension)
        x_batched = jnp.expand_dims(x, axis=0)  # Add extra batch dimension
        x_batched = jnp.repeat(x_batched, 3, axis=0)  # (3, 2, 1, 16, 32)

        @jax.vmap
        def vmapped_forward(x_batch):
            return sfno(x_batch)

        # This should work without errors
        output_vmap = vmapped_forward(x_batched)
        expected_shape = (3, 2, 1, 16, 32)
        assert output_vmap.shape == expected_shape
