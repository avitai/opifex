"""
TDD Tests for ShallowWaterDataSource (Grain-compliant).

Following TDD principles: Tests written FIRST, then implementation.
"""

import jax.numpy as jnp
import pytest


class TestShallowWaterDataSource:
    """Test suite for ShallowWaterDataSource following TDD principles."""

    def test_import_shallow_water_source(self):
        """Test that we can import ShallowWaterDataSource."""
        from opifex.data.sources import ShallowWaterDataSource

        _ = ShallowWaterDataSource

    def test_source_initialization_basic(self):
        """Test basic initialization."""
        from opifex.data.sources import ShallowWaterDataSource

        source = ShallowWaterDataSource(n_samples=10, resolution=16, seed=42)

        assert source is not None
        assert len(source) == 10

    def test_source_getitem_returns_dict(self):
        """Test that __getitem__ returns required keys."""
        from opifex.data.sources import ShallowWaterDataSource

        source = ShallowWaterDataSource(n_samples=10, resolution=16, seed=42)
        sample = source[0]

        assert isinstance(sample, dict)
        assert "input" in sample
        assert "output" in sample

    def test_source_deterministic_generation(self):
        """Test deterministic generation."""
        from opifex.data.sources import ShallowWaterDataSource

        source1 = ShallowWaterDataSource(n_samples=10, resolution=16, seed=42)
        source2 = ShallowWaterDataSource(n_samples=10, resolution=16, seed=42)

        sample1 = source1[0]
        sample2 = source2[0]

        assert jnp.allclose(sample1["input"], sample2["input"])
        assert jnp.allclose(sample1["output"], sample2["output"])

    def test_source_correct_shapes(self):
        """Test that shapes are correct."""
        from opifex.data.sources import ShallowWaterDataSource

        resolution = 32
        source = ShallowWaterDataSource(
            n_samples=5,
            resolution=resolution,
            seed=42,
        )

        sample = source[0]
        # Input should be 3-tuple: (h, u, v) each (resolution, resolution)
        assert sample["input"].shape == (3, resolution, resolution)
        # Output should be 3-tuple: (h, u, v) each (resolution, resolution)
        assert sample["output"].shape == (3, resolution, resolution)

    def test_source_index_out_of_bounds(self):
        """Test index bounds checking."""
        from opifex.data.sources import ShallowWaterDataSource

        source = ShallowWaterDataSource(n_samples=10, resolution=16)

        _ = source[9]  # Valid

        with pytest.raises(IndexError):
            _ = source[10]
