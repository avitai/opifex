"""
TDD Tests for BurgersDataSource (Grain-compliant).

Following TDD principles: Tests written FIRST, then implementation.
"""

import jax.numpy as jnp
import pytest


# Tests will fail until implementation is complete


class TestBurgersDataSource:
    """Test suite for BurgersDataSource following TDD principles."""

    def test_import_grain_source(self):
        """Test that we can import BurgersDataSource."""
        from opifex.data.sources import BurgersDataSource

        _ = BurgersDataSource  # Verify import works

    def test_source_initialization_basic(self):
        """Test basic initialization of BurgersDataSource."""
        from opifex.data.sources import BurgersDataSource

        source = BurgersDataSource(
            n_samples=10,
            resolution=16,
            seed=42,
        )

        assert source is not None
        assert len(source) == 10

    def test_source_len_returns_n_samples(self):
        """Test that __len__ returns correct number of samples."""
        from opifex.data.sources import BurgersDataSource

        for n_samples in [10, 50, 100]:
            source = BurgersDataSource(n_samples=n_samples, resolution=16)
            assert len(source) == n_samples

    def test_source_getitem_returns_dict(self):
        """Test that __getitem__ returns a dictionary with required keys."""
        from opifex.data.sources import BurgersDataSource

        source = BurgersDataSource(n_samples=10, resolution=16, seed=42)
        sample = source[0]

        # Must return dictionary
        assert isinstance(sample, dict)

        # Must have required keys
        assert "input" in sample
        assert "output" in sample

    def test_source_deterministic_generation(self):
        """Test that same index returns same sample (deterministic)."""
        from opifex.data.sources import BurgersDataSource

        source1 = BurgersDataSource(n_samples=10, resolution=16, seed=42)
        source2 = BurgersDataSource(n_samples=10, resolution=16, seed=42)

        # Same index should give identical results
        sample1_a = source1[0]
        sample1_b = source2[0]

        assert jnp.allclose(sample1_a["input"], sample1_b["input"])
        assert jnp.allclose(sample1_a["output"], sample1_b["output"])

    def test_source_different_indices_different_samples(self):
        """Test that different indices return different samples."""
        from opifex.data.sources import BurgersDataSource

        source = BurgersDataSource(n_samples=10, resolution=16, seed=42)

        sample0 = source[0]
        sample1 = source[1]

        # Different indices should give different results
        assert not jnp.allclose(sample0["input"], sample1["input"])

    def test_source_correct_shapes_2d(self):
        """Test that 2D Burgers returns correct shapes."""
        from opifex.data.sources import BurgersDataSource

        resolution = 32
        time_steps = 5
        source = BurgersDataSource(
            n_samples=5,
            resolution=resolution,
            time_steps=time_steps,
            dimension="2d",
            seed=42,
        )

        sample = source[0]

        # 2D: input should be (resolution, resolution)
        assert sample["input"].shape == (resolution, resolution)

        # output should be (time_steps, resolution, resolution)
        assert sample["output"].shape == (time_steps, resolution, resolution)

    def test_source_correct_shapes_1d(self):
        """Test that 1D Burgers returns correct shapes."""
        from opifex.data.sources import BurgersDataSource

        resolution = 64
        time_steps = 10
        source = BurgersDataSource(
            n_samples=5,
            resolution=resolution,
            time_steps=time_steps,
            dimension="1d",
            seed=42,
        )

        sample = source[0]

        # 1D: input should be (resolution,)
        assert sample["input"].shape == (resolution,)

        # output should be (time_steps, resolution)
        assert sample["output"].shape == (time_steps, resolution)

    def test_source_index_out_of_bounds(self):
        """Test that indexing out of bounds raises appropriate error."""
        from opifex.data.sources import BurgersDataSource

        source = BurgersDataSource(n_samples=10, resolution=16)

        # Valid index
        _ = source[9]

        # Out of bounds should raise IndexError
        with pytest.raises(IndexError):
            _ = source[10]

    def test_source_metadata_present(self):
        """Test that metadata is included in sample."""
        from opifex.data.sources import BurgersDataSource

        source = BurgersDataSource(
            n_samples=5,
            resolution=16,
            dimension="2d",
            seed=42,
        )

        sample = source[3]

        # Should have metadata
        assert "metadata" in sample
        assert sample["metadata"]["index"] == 3
        assert sample["metadata"]["dimension"] == "2d"

    def test_source_viscosity_parameter_present(self):
        """Test that viscosity parameter is included."""
        from opifex.data.sources import BurgersDataSource

        source = BurgersDataSource(
            n_samples=5,
            resolution=16,
            viscosity_range=(0.01, 0.1),
            seed=42,
        )

        sample = source[0]

        assert "viscosity" in sample
        # Viscosity should be in specified range
        assert 0.01 <= sample["viscosity"] <= 0.1

    def test_source_time_points_present(self):
        """Test that time points are included."""
        from opifex.data.sources import BurgersDataSource

        time_range = (0.0, 2.0)
        time_steps = 5

        source = BurgersDataSource(
            n_samples=3,
            resolution=16,
            time_range=time_range,
            time_steps=time_steps,
            seed=42,
        )

        sample = source[0]

        assert "time_points" in sample
        assert len(sample["time_points"]) == time_steps
        assert sample["time_points"][0] == pytest.approx(time_range[0])
        assert sample["time_points"][-1] == pytest.approx(time_range[1])

    def test_source_different_seeds_different_data(self):
        """Test that different seeds produce different data."""
        from opifex.data.sources import BurgersDataSource

        source1 = BurgersDataSource(n_samples=10, resolution=16, seed=42)
        source2 = BurgersDataSource(n_samples=10, resolution=16, seed=43)

        sample1 = source1[0]
        sample2 = source2[0]

        # Different seeds should give different results
        assert not jnp.allclose(sample1["input"], sample2["input"])

    def test_source_same_index_same_seed_reproducible(self):
        """Test full reproducibility with same seed and index."""
        from opifex.data.sources import BurgersDataSource

        # Create source twice with same seed
        source1 = BurgersDataSource(n_samples=20, resolution=32, seed=123)
        source2 = BurgersDataSource(n_samples=20, resolution=32, seed=123)

        # Test multiple indices
        for idx in [0, 5, 10, 15]:
            sample1 = source1[idx]
            sample2 = source2[idx]

            assert jnp.allclose(sample1["input"], sample2["input"])
            assert jnp.allclose(sample1["output"], sample2["output"])
            assert sample1["viscosity"] == pytest.approx(sample2["viscosity"])

    def test_source_iterable(self):
        """Test that source can be iterated (for Grain compatibility)."""
        from opifex.data.sources import BurgersDataSource

        source = BurgersDataSource(n_samples=5, resolution=16, seed=42)

        # Should be able to iterate
        samples = [source[i] for i in range(len(source))]

        assert len(samples) == 5
        for sample in samples:
            assert "input" in sample
            assert "output" in sample

    def test_source_resolution_parameter(self):
        """Test that resolution parameter affects output shape."""
        from opifex.data.sources import BurgersDataSource

        for resolution in [16, 32, 64]:
            source = BurgersDataSource(
                n_samples=3, resolution=resolution, dimension="2d", seed=42
            )

            sample = source[0]
            assert sample["input"].shape == (resolution, resolution)

    def test_source_time_steps_parameter(self):
        """Test that time_steps parameter affects output shape."""
        from opifex.data.sources import BurgersDataSource

        for time_steps in [5, 10, 20]:
            source = BurgersDataSource(
                n_samples=3,
                resolution=16,
                time_steps=time_steps,
                dimension="1d",
                seed=42,
            )

            sample = source[0]
            assert sample["output"].shape[0] == time_steps
