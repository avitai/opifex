"""
TDD Tests for NavierStokesDataSource (Grain-compliant).

Following TDD principles: Tests written FIRST, then implementation.
"""

import jax.numpy as jnp
import pytest


class TestNavierStokesDataSource:
    """Test suite for NavierStokesDataSource following TDD principles."""

    def test_import_grain_source(self):
        """Test that we can import NavierStokesDataSource."""
        from opifex.data.sources import NavierStokesDataSource

        _ = NavierStokesDataSource  # Verify import works

    def test_source_initialization_basic(self):
        """Test basic initialization of NavierStokesDataSource."""
        from opifex.data.sources import NavierStokesDataSource

        source = NavierStokesDataSource(
            n_samples=10,
            resolution=32,
            seed=42,
        )

        assert source is not None
        assert len(source) == 10

    def test_source_len_returns_n_samples(self):
        """Test that __len__ returns correct number of samples."""
        from opifex.data.sources import NavierStokesDataSource

        for n_samples in [10, 50, 100]:
            source = NavierStokesDataSource(n_samples=n_samples, resolution=32)
            assert len(source) == n_samples

    def test_source_getitem_returns_dict(self):
        """Test that __getitem__ returns a dictionary with required keys."""
        from opifex.data.sources import NavierStokesDataSource

        source = NavierStokesDataSource(n_samples=10, resolution=32, seed=42)
        sample = source[0]

        # Must return dictionary
        assert isinstance(sample, dict)

        # Must have required keys
        assert "input" in sample
        assert "output" in sample

    def test_source_deterministic_generation(self):
        """Test that same index returns same sample (deterministic)."""
        from opifex.data.sources import NavierStokesDataSource

        source1 = NavierStokesDataSource(n_samples=10, resolution=32, seed=42)
        source2 = NavierStokesDataSource(n_samples=10, resolution=32, seed=42)

        # Same index should give identical results
        sample1_a = source1[0]
        sample1_b = source2[0]

        assert jnp.allclose(sample1_a["input"], sample1_b["input"])
        assert jnp.allclose(sample1_a["output"], sample1_b["output"])

    def test_source_different_indices_different_samples(self):
        """Test that different indices return different samples."""
        from opifex.data.sources import NavierStokesDataSource

        source = NavierStokesDataSource(n_samples=10, resolution=32, seed=42)

        sample0 = source[0]
        sample1 = source[1]

        # Different indices should give different results
        assert not jnp.allclose(sample0["input"], sample1["input"])

    def test_source_correct_shapes(self):
        """Test that NavierStokes returns correct shapes."""
        from opifex.data.sources import NavierStokesDataSource

        resolution = 32
        time_steps = 5
        source = NavierStokesDataSource(
            n_samples=5,
            resolution=resolution,
            time_steps=time_steps,
            seed=42,
        )

        sample = source[0]

        # Input: initial velocity field (u, v) stacked -> (2, resolution, resolution)
        assert sample["input"].shape == (2, resolution, resolution)

        # Output: velocity trajectories (time_steps, 2, resolution, resolution)
        assert sample["output"].shape == (time_steps, 2, resolution, resolution)

    def test_source_index_out_of_bounds(self):
        """Test that indexing out of bounds raises appropriate error."""
        from opifex.data.sources import NavierStokesDataSource

        source = NavierStokesDataSource(n_samples=10, resolution=32)

        # Valid index
        _ = source[9]

        # Out of bounds should raise IndexError
        with pytest.raises(IndexError):
            _ = source[10]

    def test_source_metadata_present(self):
        """Test that metadata is included in sample."""
        from opifex.data.sources import NavierStokesDataSource

        source = NavierStokesDataSource(
            n_samples=5,
            resolution=32,
            seed=42,
        )

        sample = source[3]

        # Should have metadata
        assert "metadata" in sample
        assert sample["metadata"]["index"] == 3
        assert sample["metadata"]["resolution"] == 32

    def test_source_reynolds_number_present(self):
        """Test that Reynolds number is included."""
        from opifex.data.sources import NavierStokesDataSource

        source = NavierStokesDataSource(
            n_samples=5,
            resolution=32,
            reynolds_range=(100.0, 1000.0),
            seed=42,
        )

        sample = source[0]

        assert "reynolds" in sample
        # Reynolds number should be in specified range
        assert 100.0 <= sample["reynolds"] <= 1000.0

    def test_source_time_points_present(self):
        """Test that time points are included."""
        from opifex.data.sources import NavierStokesDataSource

        time_range = (0.0, 2.0)
        time_steps = 5

        source = NavierStokesDataSource(
            n_samples=3,
            resolution=32,
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
        from opifex.data.sources import NavierStokesDataSource

        source1 = NavierStokesDataSource(n_samples=10, resolution=32, seed=42)
        source2 = NavierStokesDataSource(n_samples=10, resolution=32, seed=43)

        sample1 = source1[0]
        sample2 = source2[0]

        # Different seeds should give different results
        assert not jnp.allclose(sample1["input"], sample2["input"])

    def test_source_same_index_same_seed_reproducible(self):
        """Test full reproducibility with same seed and index."""
        from opifex.data.sources import NavierStokesDataSource

        # Create source twice with same seed
        source1 = NavierStokesDataSource(n_samples=20, resolution=32, seed=123)
        source2 = NavierStokesDataSource(n_samples=20, resolution=32, seed=123)

        # Test multiple indices
        for idx in [0, 5, 10, 15]:
            sample1 = source1[idx]
            sample2 = source2[idx]

            assert jnp.allclose(sample1["input"], sample2["input"])
            assert jnp.allclose(sample1["output"], sample2["output"])
            assert sample1["reynolds"] == pytest.approx(sample2["reynolds"])

    def test_source_iterable(self):
        """Test that source can be iterated (for Grain compatibility)."""
        from opifex.data.sources import NavierStokesDataSource

        source = NavierStokesDataSource(n_samples=5, resolution=32, seed=42)

        # Should be able to iterate
        samples = [source[i] for i in range(len(source))]

        assert len(samples) == 5
        for sample in samples:
            assert "input" in sample
            assert "output" in sample

    def test_source_resolution_parameter(self):
        """Test that resolution parameter affects output shape."""
        from opifex.data.sources import NavierStokesDataSource

        for resolution in [32, 64]:
            source = NavierStokesDataSource(n_samples=3, resolution=resolution, seed=42)

            sample = source[0]
            assert sample["input"].shape == (2, resolution, resolution)

    def test_source_time_steps_parameter(self):
        """Test that time_steps parameter affects output shape."""
        from opifex.data.sources import NavierStokesDataSource

        for time_steps in [5, 10, 20]:
            source = NavierStokesDataSource(
                n_samples=3,
                resolution=32,
                time_steps=time_steps,
                seed=42,
            )

            sample = source[0]
            assert sample["output"].shape[0] == time_steps


class TestNavierStokesDataLoader:
    """Test suite for create_navier_stokes_loader factory."""

    def test_import_loader_factory(self):
        """Test that loader factory can be imported."""
        from opifex.data.loaders import create_navier_stokes_loader

        assert create_navier_stokes_loader is not None

    def test_loader_returns_grain_dataloader(self):
        """Test that factory returns a Grain DataLoader."""
        import grain.python as grain

        from opifex.data.loaders import create_navier_stokes_loader

        loader = create_navier_stokes_loader(
            n_samples=10,
            batch_size=2,
            resolution=32,
            seed=42,
            worker_count=0,
        )

        assert isinstance(loader, grain.DataLoader)

    def test_loader_yields_batches(self):
        """Test that loader yields batches with correct structure."""
        from opifex.data.loaders import create_navier_stokes_loader

        batch_size = 4
        resolution = 32
        time_steps = 5

        loader = create_navier_stokes_loader(
            n_samples=12,
            batch_size=batch_size,
            resolution=resolution,
            time_steps=time_steps,
            seed=42,
            worker_count=0,
        )

        batch = next(iter(loader))

        # Batch should be a dictionary
        assert isinstance(batch, dict)
        assert "input" in batch
        assert "output" in batch

        # Check batch shapes
        assert batch["input"].shape == (batch_size, 2, resolution, resolution)
        assert batch["output"].shape == (
            batch_size,
            time_steps,
            2,
            resolution,
            resolution,
        )

    def test_loader_multiple_iterations(self):
        """Test that loader can be iterated multiple times."""
        from opifex.data.loaders import create_navier_stokes_loader

        loader = create_navier_stokes_loader(
            n_samples=16,
            batch_size=4,
            resolution=32,
            seed=42,
            worker_count=0,
        )

        # Should yield multiple batches
        batches = list(loader)
        assert len(batches) >= 1
