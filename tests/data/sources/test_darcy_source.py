"""
TDD Tests for DarcyDataSource (Grain-compliant).

Following TDD principles: Tests written FIRST, then implementation.
"""

import jax.numpy as jnp
import pytest


class TestDarcyDataSource:
    """Test suite for DarcyDataSource following TDD principles."""

    def test_import_darcy_source(self):
        """Test that we can import DarcyDataSource."""
        from opifex.data.sources import DarcyDataSource

        _ = DarcyDataSource

    def test_source_initialization_basic(self):
        """Test basic initialization of DarcyDataSource."""
        from opifex.data.sources import DarcyDataSource

        source = DarcyDataSource(
            n_samples=10,
            resolution=16,
            seed=42,
        )

        assert source is not None
        assert len(source) == 10

    def test_source_len_returns_n_samples(self):
        """Test that __len__ returns correct number of samples."""
        from opifex.data.sources import DarcyDataSource

        for n_samples in [10, 50, 100]:
            source = DarcyDataSource(n_samples=n_samples, resolution=16)
            assert len(source) == n_samples

    def test_source_getitem_returns_dict(self):
        """Test that __getitem__ returns a dictionary with required keys."""
        from opifex.data.sources import DarcyDataSource

        source = DarcyDataSource(n_samples=10, resolution=16, seed=42)
        sample = source[0]

        assert isinstance(sample, dict)
        assert "input" in sample  # Permeability field
        assert "output" in sample  # Pressure solution

    def test_source_deterministic_generation(self):
        """Test that same index returns same sample (deterministic)."""
        from opifex.data.sources import DarcyDataSource

        source1 = DarcyDataSource(n_samples=10, resolution=16, seed=42)
        source2 = DarcyDataSource(n_samples=10, resolution=16, seed=42)

        sample1_a = source1[0]
        sample1_b = source2[0]

        assert jnp.allclose(sample1_a["input"], sample1_b["input"])
        assert jnp.allclose(sample1_a["output"], sample1_b["output"])

    def test_source_different_indices_different_samples(self):
        """Test that different indices return different samples."""
        from opifex.data.sources import DarcyDataSource

        source = DarcyDataSource(n_samples=10, resolution=16, seed=42)

        sample0 = source[0]
        sample1 = source[1]

        assert not jnp.allclose(sample0["input"], sample1["input"])

    def test_source_correct_shapes(self):
        """Test that Darcy returns correct shapes."""
        from opifex.data.sources import DarcyDataSource

        resolution = 32
        source = DarcyDataSource(
            n_samples=5,
            resolution=resolution,
            seed=42,
        )

        sample = source[0]

        # Input (permeability) should be 2D: (resolution, resolution)
        assert sample["input"].shape == (resolution, resolution)

        # Output (pressure) should be 2D: (resolution, resolution)
        assert sample["output"].shape == (resolution, resolution)

    def test_source_metadata_present(self):
        """Test that metadata is included in sample."""
        from opifex.data.sources import DarcyDataSource

        source = DarcyDataSource(
            n_samples=5,
            resolution=16,
            seed=42,
        )

        sample = source[3]

        assert "metadata" in sample
        assert sample["metadata"]["index"] == 3

    def test_source_different_seeds_different_data(self):
        """Test that different seeds produce different data."""
        from opifex.data.sources import DarcyDataSource

        source1 = DarcyDataSource(n_samples=10, resolution=16, seed=42)
        source2 = DarcyDataSource(n_samples=10, resolution=16, seed=43)

        sample1 = source1[0]
        sample2 = source2[0]

        assert not jnp.allclose(sample1["input"], sample2["input"])

    def test_source_reproducible_with_same_seed(self):
        """Test full reproducibility with same seed and index."""
        from opifex.data.sources import DarcyDataSource

        seed = 123

        source1 = DarcyDataSource(n_samples=20, resolution=32, seed=seed)
        source2 = DarcyDataSource(n_samples=20, resolution=32, seed=seed)

        for idx in [0, 5, 10, 15]:
            sample1 = source1[idx]
            sample2 = source2[idx]

            assert jnp.allclose(sample1["input"], sample2["input"])
            assert jnp.allclose(sample1["output"], sample2["output"])

    def test_source_index_out_of_bounds(self):
        """Test that indexing out of bounds raises appropriate error."""
        from opifex.data.sources import DarcyDataSource

        source = DarcyDataSource(n_samples=10, resolution=16)

        _ = source[9]  # Valid

        with pytest.raises(IndexError):
            _ = source[10]  # Out of bounds

    def test_source_resolution_parameter(self):
        """Test that resolution parameter affects output shape."""
        from opifex.data.sources import DarcyDataSource

        for resolution in [16, 32, 64]:
            source = DarcyDataSource(n_samples=3, resolution=resolution, seed=42)

            sample = source[0]
            assert sample["input"].shape == (resolution, resolution)
            assert sample["output"].shape == (resolution, resolution)
