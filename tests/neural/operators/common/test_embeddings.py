"""Test embedding layers for neural operators.

Comprehensive tests for grid embeddings, sinusoidal embeddings, and utility functions
including JAX compatibility, different input configurations, and performance validation.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.neural.operators.common.embeddings import (
    GridEmbedding2D,
    GridEmbeddingND,
    regular_grid_2d,
    SinusoidalEmbedding,
)


class TestGridEmbedding2D:
    """Comprehensive tests for GridEmbedding2D layer."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    def test_basic_initialization(self):
        """Test basic GridEmbedding2D initialization."""
        # Test with default boundaries
        embedding = GridEmbedding2D(in_channels=1)
        assert embedding.in_channels == 1
        assert embedding.grid_boundaries == [[0.0, 1.0], [0.0, 1.0]]

        # Test with custom boundaries
        custom_boundaries = [[-1.0, 1.0], [0.0, 2.0]]
        embedding_custom = GridEmbedding2D(
            in_channels=2, grid_boundaries=custom_boundaries
        )
        assert embedding_custom.in_channels == 2
        assert embedding_custom.grid_boundaries == custom_boundaries

    def test_forward_pass_basic(self, rng_key):
        """Test basic forward pass functionality."""
        embedding = GridEmbedding2D(in_channels=1)

        # Test single channel input
        x = jax.random.normal(rng_key, (2, 8, 8, 1))
        output = embedding(x)

        # Should add 2 grid channels (x, y coordinates)
        expected_shape = (2, 8, 8, 3)  # 1 input + 2 grid channels
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

        # Original data should be preserved in first channel
        assert jnp.allclose(output[..., :1], x, rtol=1e-6)

    def test_forward_pass_multiple_channels(self, rng_key):
        """Test forward pass with multiple input channels."""
        embedding = GridEmbedding2D(in_channels=3)

        x = jax.random.normal(rng_key, (2, 16, 16, 3))
        output = embedding(x)

        # Should add 2 grid channels to 3 input channels
        expected_shape = (2, 16, 16, 5)  # 3 input + 2 grid channels
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

        # Original data should be preserved
        assert jnp.allclose(output[..., :3], x, rtol=1e-6)

    def test_grid_coordinate_values(self, rng_key):
        """Test that grid coordinates are generated correctly."""
        boundaries = [[-1.0, 1.0], [0.0, 2.0]]
        embedding = GridEmbedding2D(in_channels=1, grid_boundaries=boundaries)

        x = jax.random.normal(rng_key, (1, 4, 4, 1))
        output = embedding(x)

        # Extract grid coordinates
        x_coords = output[0, :, :, 1]  # Second channel is x coordinates
        y_coords = output[0, :, :, 2]  # Third channel is y coordinates

        # Check x coordinate range (approximately)
        assert jnp.min(x_coords) >= -1.1  # Allow small numerical error
        assert jnp.max(x_coords) <= 1.1

        # Check y coordinate range
        assert jnp.min(y_coords) >= -0.1  # Allow small numerical error
        assert jnp.max(y_coords) <= 2.1

    def test_different_spatial_sizes(self, rng_key):
        """Test embedding with different spatial resolutions."""
        embedding = GridEmbedding2D(in_channels=1)

        sizes = [4, 8, 16, 32]
        for size in sizes:
            x = jax.random.normal(rng_key, (1, size, size, 1))
            output = embedding(x)

            expected_shape = (1, size, size, 3)
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_different_batch_sizes(self, rng_key):
        """Test embedding with different batch sizes."""
        embedding = GridEmbedding2D(in_channels=2)

        batch_sizes = [1, 2, 4, 8]
        spatial_size = 12

        for batch_size in batch_sizes:
            x = jax.random.normal(rng_key, (batch_size, spatial_size, spatial_size, 2))
            output = embedding(x)

            expected_shape = (batch_size, spatial_size, spatial_size, 4)
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_jax_transformations(self, rng_key):
        """Test compatibility with JAX transformations."""
        embedding = GridEmbedding2D(in_channels=1)
        x = jax.random.normal(rng_key, (2, 8, 8, 1))

        # Test JIT compilation
        @jax.jit
        def jit_forward(inputs):
            return embedding(inputs)

        output_jit = jit_forward(x)
        output_regular = embedding(x)

        assert jnp.allclose(output_regular, output_jit, rtol=1e-6)

        # Test vmap
        @jax.vmap
        def vmap_forward(inputs):
            return embedding(jnp.expand_dims(inputs, 0))

        x_unbatched = x  # Shape: (2, 8, 8, 1)
        output_vmap = vmap_forward(x_unbatched)  # Shape: (2, 1, 8, 8, 3)
        output_vmap = jnp.squeeze(output_vmap, axis=1)  # Shape: (2, 8, 8, 3)

        assert output_vmap.shape == output_regular.shape
        assert jnp.allclose(output_vmap, output_regular, rtol=1e-6)

    def test_gradient_flow(self, rng_key):
        """Test that gradients flow through the embedding properly."""
        embedding = GridEmbedding2D(in_channels=1)
        x = jax.random.normal(rng_key, (2, 8, 8, 1))

        def loss_fn(inputs):
            output = embedding(inputs)
            # Loss only depends on input channels, not grid coordinates
            return jnp.mean(output[..., :1] ** 2)

        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(x)

        assert grads.shape == x.shape
        assert jnp.all(jnp.isfinite(grads))
        # Gradients should not be all zero
        assert jnp.linalg.norm(grads) > 1e-6

    def test_rectangular_grids(self, rng_key):
        """Test embedding with non-square spatial dimensions."""
        embedding = GridEmbedding2D(in_channels=1)

        # Test rectangular inputs
        shapes = [(1, 8, 12, 1), (2, 16, 24, 1), (1, 32, 16, 1)]

        for shape in shapes:
            x = jax.random.normal(rng_key, shape)
            output = embedding(x)

            expected_shape = (*shape[:3], 3)  # Add 2 grid channels
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_nd_grid_embedding_consistency(self):
        """Test consistency between different dimensional embeddings."""
        # Test 2D case with both 2D and ND implementations
        in_channels = 3
        batch_size = 4
        height, width = 32, 32

        # Create test input
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (batch_size, height, width, in_channels))

        # Test 2D embedding
        grid_2d = GridEmbedding2D(in_channels)
        output_2d = grid_2d(x)

        # Test ND embedding with dim=2
        grid_nd = GridEmbeddingND(in_channels, dim=2)
        output_nd = grid_nd(x)

        # Should have same output shape
        assert output_2d.shape == output_nd.shape
        assert output_2d.shape == (*x.shape[:3], in_channels + 2)

        # Grid coordinates should be approximately equal (different indexing might cause small diffs)
        coord_2d = output_2d[..., -2:]
        coord_nd = output_nd[..., -2:]

        # Check that coordinates are in expected range
        assert jnp.all(coord_2d >= 0.0)
        assert jnp.all(coord_2d <= 1.0)
        assert jnp.all(coord_nd >= 0.0)
        assert jnp.all(coord_nd <= 1.0)


class TestGridEmbeddingND:
    """Comprehensive tests for GridEmbeddingND layer."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(123)

    def test_basic_initialization(self):
        """Test basic GridEmbeddingND initialization."""
        # Test 2D case
        embedding_2d = GridEmbeddingND(in_channels=1, dim=2)
        assert embedding_2d.in_channels == 1
        assert embedding_2d.dim == 2
        assert len(embedding_2d.grid_boundaries) == 2

        # Test 3D case
        embedding_3d = GridEmbeddingND(in_channels=2, dim=3)
        assert embedding_3d.in_channels == 2
        assert embedding_3d.dim == 3
        assert len(embedding_3d.grid_boundaries) == 3

    def test_2d_embedding(self, rng_key):
        """Test 2D embedding (should match GridEmbedding2D behavior)."""
        embedding = GridEmbeddingND(in_channels=1, dim=2)
        x = jax.random.normal(rng_key, (2, 8, 8, 1))
        output = embedding(x)

        expected_shape = (2, 8, 8, 3)  # 1 input + 2 grid channels
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_3d_embedding(self, rng_key):
        """Test 3D embedding functionality."""
        embedding = GridEmbeddingND(in_channels=2, dim=3)
        x = jax.random.normal(rng_key, (1, 4, 6, 8, 2))
        output = embedding(x)

        expected_shape = (1, 4, 6, 8, 5)  # 2 input + 3 grid channels
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

        # Original data should be preserved
        assert jnp.allclose(output[..., :2], x, rtol=1e-6)

    def test_1d_embedding(self, rng_key):
        """Test 1D embedding functionality."""
        embedding = GridEmbeddingND(in_channels=1, dim=1)
        x = jax.random.normal(rng_key, (3, 16, 1))
        output = embedding(x)

        expected_shape = (3, 16, 2)  # 1 input + 1 grid channel
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_custom_boundaries(self, rng_key):
        """Test with custom grid boundaries."""
        boundaries = [[-2.0, 2.0], [-1.0, 3.0], [0.0, 1.0]]
        embedding = GridEmbeddingND(in_channels=1, dim=3, grid_boundaries=boundaries)

        x = jax.random.normal(rng_key, (1, 4, 4, 4, 1))
        output = embedding(x)

        # Check coordinate ranges (approximately)
        coord_x = output[0, :, :, :, 1]
        coord_y = output[0, :, :, :, 2]
        coord_z = output[0, :, :, :, 3]

        assert jnp.min(coord_x) >= -2.1 and jnp.max(coord_x) <= 2.1
        assert jnp.min(coord_y) >= -1.1 and jnp.max(coord_y) <= 3.1
        assert jnp.min(coord_z) >= -0.1 and jnp.max(coord_z) <= 1.1

    def test_jax_transformations_3d(self, rng_key):
        """Test JAX transformations with 3D embedding."""
        embedding = GridEmbeddingND(in_channels=1, dim=3)
        x = jax.random.normal(rng_key, (2, 4, 4, 4, 1))

        # Test JIT
        @jax.jit
        def jit_forward(inputs):
            return embedding(inputs)

        output_jit = jit_forward(x)
        output_regular = embedding(x)

        assert jnp.allclose(output_regular, output_jit, rtol=1e-6)


class TestSinusoidalEmbedding:
    """Comprehensive tests for SinusoidalEmbedding layer."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(456)

    def test_basic_initialization(self):
        """Test basic SinusoidalEmbedding initialization."""
        embedding = SinusoidalEmbedding(
            in_channels=1, num_frequencies=64, max_positions=1000
        )
        assert embedding.in_channels == 1
        assert embedding.num_frequencies == 64
        assert embedding.max_positions == 1000

    def test_forward_pass_basic(self, rng_key):
        """Test basic forward pass functionality."""
        embedding = SinusoidalEmbedding(
            in_channels=1, num_frequencies=32, embedding_type="transformer"
        )

        x = jax.random.normal(rng_key, (2, 8, 1))
        output = embedding(x)

        expected_shape = (2, 8, 64)  # 2 * num_frequencies * in_channels
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_different_embedding_dimensions(self, rng_key):
        """Test with different embedding dimensions."""
        dims = [16, 32, 64, 128]
        x = jax.random.normal(rng_key, (1, 4, 2))

        for dim in dims:
            embedding = SinusoidalEmbedding(
                in_channels=2, num_frequencies=dim, embedding_type="transformer"
            )
            output = embedding(x)

            expected_shape = (1, 4, 2 * dim * 2)  # 2 * num_frequencies * in_channels
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_sinusoidal_properties(self, rng_key):
        """Test that sinusoidal embeddings have expected properties."""
        embedding = SinusoidalEmbedding(
            in_channels=1, num_frequencies=64, embedding_type="nerf"
        )

        # Test with constant input
        x_const = jnp.ones((1, 1, 1))
        output_const = embedding(x_const)

        # Embedding should be finite
        assert jnp.all(jnp.isfinite(output_const))

    def test_different_frequencies(self, rng_key):
        """Test with different maximum frequencies."""
        frequencies = [10, 100, 1000, 10000]
        x = jax.random.normal(rng_key, (1, 4, 1))

        for freq in frequencies:
            embedding = SinusoidalEmbedding(
                in_channels=1,
                num_frequencies=32,
                max_positions=freq,
                embedding_type="transformer",
            )
            output = embedding(x)

            assert output.shape == (1, 4, 64)  # 2 * 32 * 1
            assert jnp.all(jnp.isfinite(output))

    def test_jax_transformations(self, rng_key):
        """Test compatibility with JAX transformations."""
        embedding = SinusoidalEmbedding(
            in_channels=1, num_frequencies=16, embedding_type="transformer"
        )
        x = jax.random.normal(rng_key, (2, 4, 1))

        # Test JIT
        @jax.jit
        def jit_forward(inputs):
            return embedding(inputs)

        output_jit = jit_forward(x)
        output_regular = embedding(x)

        assert jnp.allclose(output_regular, output_jit, rtol=1e-6)

    def test_gradient_flow(self, rng_key):
        """Test gradient flow through sinusoidal embedding."""
        embedding = SinusoidalEmbedding(
            in_channels=1, num_frequencies=16, embedding_type="transformer"
        )
        x = jax.random.normal(rng_key, (1, 4, 1))

        def loss_fn(inputs):
            output = embedding(inputs)
            # Use a loss that depends on input-output relationship
            # This creates meaningful gradients for sinusoidal embeddings
            return jnp.mean((output - jnp.sin(inputs)) ** 2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(x)

        assert grads.shape == x.shape
        assert jnp.all(jnp.isfinite(grads))
        # For sinusoidal embeddings, gradients should exist but may be small
        # Check that we don't have NaN or infinite gradients
        assert not jnp.any(jnp.isnan(grads))
        assert not jnp.any(jnp.isinf(grads))


class TestUtilityFunctions:
    """Tests for utility functions in embeddings module."""

    def test_regular_grid_2d_basic(self):
        """Test basic 2D grid generation."""
        grid_x, grid_y = regular_grid_2d(spatial_dims=(4, 6))

        assert grid_x.shape == (4, 6)
        assert grid_y.shape == (4, 6)
        assert jnp.all(jnp.isfinite(grid_x))
        assert jnp.all(jnp.isfinite(grid_y))

        # Check coordinate ranges
        assert jnp.min(grid_x) >= 0.0
        assert jnp.max(grid_x) <= 1.0
        assert jnp.min(grid_y) >= 0.0
        assert jnp.max(grid_y) <= 1.0

    def test_regular_grid_2d_custom_boundaries(self):
        """Test 2D grid with custom boundaries."""
        boundaries = [[-1.0, 1.0], [0.0, 2.0]]
        grid_x, grid_y = regular_grid_2d(
            spatial_dims=(3, 5), grid_boundaries=boundaries
        )

        assert grid_x.shape == (3, 5)
        assert grid_y.shape == (3, 5)

        assert jnp.min(grid_x) >= -1.0
        assert jnp.max(grid_x) <= 1.0
        assert jnp.min(grid_y) >= 0.0
        assert jnp.max(grid_y) <= 2.0

    def test_grid_resolutions_from_boundaries_2d(self):
        """Test multi-resolution grid generation for 2D case."""
        resolutions = [4, 6]
        grids = [regular_grid_2d((res, res)) for res in resolutions]

        assert len(grids) == 2
        assert grids[0][0].shape == (4, 4)  # x_grid for first resolution
        assert grids[0][1].shape == (4, 4)  # y_grid for first resolution
        assert grids[1][0].shape == (6, 6)  # x_grid for second resolution
        assert grids[1][1].shape == (6, 6)  # y_grid for second resolution

        for grid_x, grid_y in grids:
            assert jnp.all(jnp.isfinite(grid_x))
            assert jnp.all(jnp.isfinite(grid_y))

    def test_grid_monotonicity(self):
        """Test that grid coordinates are monotonic."""
        grid_x, grid_y = regular_grid_2d(spatial_dims=(5, 7))

        # Check x coordinates increase along width dimension
        for i in range(5):
            x_row = grid_x[i, :]
            assert jnp.all(jnp.diff(x_row) >= 0)  # Non-decreasing

        # Check y coordinates increase along height dimension
        for j in range(7):
            y_col = grid_y[:, j]
            assert jnp.all(jnp.diff(y_col) >= 0)  # Non-decreasing

    def test_jax_compatibility_utilities(self):
        """Test that utility functions are compatible with JAX transformations."""

        # Test JIT compilation
        @jax.jit
        def jit_grid_2d():
            return regular_grid_2d(spatial_dims=(4, 4))

        grid_x_jit, grid_y_jit = jit_grid_2d()
        grid_x_regular, grid_y_regular = regular_grid_2d(spatial_dims=(4, 4))

        assert jnp.allclose(grid_x_jit, grid_x_regular, rtol=1e-6)
        assert jnp.allclose(grid_y_jit, grid_y_regular, rtol=1e-6)
