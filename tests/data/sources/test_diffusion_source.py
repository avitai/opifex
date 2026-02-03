"""
TDD Tests for DiffusionDataSource (Grain-compliant).

Following TDD principles: Tests written FIRST, then implementation.
"""

import jax.numpy as jnp
import pytest


class TestDiffusionDataSource:
    """Test suite for DiffusionDataSource following TDD principles."""

    def test_import_diffusion_source(self):
        """Test that we can import DiffusionDataSource."""
        from opifex.data.sources import DiffusionDataSource

        _ = DiffusionDataSource

    def test_source_initialization_basic(self):
        """Test basic initialization of DiffusionDataSource."""
        from opifex.data.sources import DiffusionDataSource

        source = DiffusionDataSource(n_samples=10, resolution=16, seed=42)

        assert source is not None
        assert len(source) == 10

    def test_source_len_returns_n_samples(self):
        """Test that __len__ returns correct number of samples."""
        from opifex.data.sources import DiffusionDataSource

        for n_samples in [10, 50, 100]:
            source = DiffusionDataSource(n_samples=n_samples, resolution=16)
            assert len(source) == n_samples

    def test_source_getitem_returns_dict(self):
        """Test that __getitem__ returns a dictionary with required keys."""
        from opifex.data.sources import DiffusionDataSource

        source = DiffusionDataSource(n_samples=10, resolution=16, seed=42)
        sample = source[0]

        assert isinstance(sample, dict)
        assert "input" in sample
        assert "output" in sample

    def test_source_deterministic_generation(self):
        """Test that same index returns same sample (deterministic)."""
        from opifex.data.sources import DiffusionDataSource

        source1 = DiffusionDataSource(n_samples=10, resolution=16, seed=42)
        source2 = DiffusionDataSource(n_samples=10, resolution=16, seed=42)

        sample1_a = source1[0]
        sample1_b = source2[0]

        assert jnp.allclose(sample1_a["input"], sample1_b["input"])
        assert jnp.allclose(sample1_a["output"], sample1_b["output"])

    def test_source_correct_shapes_2d(self):
        """Test that 2D diffusion returns correct shapes."""
        from opifex.data.sources import DiffusionDataSource

        resolution = 32
        time_steps = 5
        source = DiffusionDataSource(
            n_samples=5,
            resolution=resolution,
            time_steps=time_steps,
            dimension="2d",
            seed=42,
        )

        sample = source[0]
        assert sample["input"].shape == (resolution, resolution)
        assert sample["output"].shape == (time_steps, resolution, resolution)

    def test_source_correct_shapes_1d(self):
        """Test that 1D diffusion returns correct shapes."""
        from opifex.data.sources import DiffusionDataSource

        resolution = 64
        time_steps = 10
        source = DiffusionDataSource(
            n_samples=5,
            resolution=resolution,
            time_steps=time_steps,
            dimension="1d",
            seed=42,
        )

        sample = source[0]
        assert sample["input"].shape == (resolution,)
        assert sample["output"].shape == (time_steps, resolution)

    def test_source_metadata_present(self):
        """Test that metadata is included in sample."""
        from opifex.data.sources import DiffusionDataSource

        source = DiffusionDataSource(n_samples=5, resolution=16, seed=42)

        sample = source[3]

        assert "metadata" in sample
        assert sample["metadata"]["index"] == 3

    def test_source_different_seeds_different_data(self):
        """Test that different seeds produce different data."""
        from opifex.data.sources import DiffusionDataSource

        source1 = DiffusionDataSource(n_samples=10, resolution=16, seed=42)
        source2 = DiffusionDataSource(n_samples=10, resolution=16, seed=43)

        sample1 = source1[0]
        sample2 = source2[0]

        assert not jnp.allclose(sample1["input"], sample2["input"])

    def test_source_index_out_of_bounds(self):
        """Test that indexing out of bounds raises appropriate error."""
        from opifex.data.sources import DiffusionDataSource

        source = DiffusionDataSource(n_samples=10, resolution=16)

        _ = source[9]  # Valid

        with pytest.raises(IndexError):
            _ = source[10]  # Out of bounds
