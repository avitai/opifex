"""
Integration tests for complete Grain pipeline.

Tests the full data loading pipeline: source → transforms → loader → iteration.
"""

import jax.numpy as jnp


class TestGrainIntegration:
    """Integration tests for Grain data loading pipeline."""

    def test_create_burgers_loader_basic(self):
        """Test creating a basic Burgers loader."""
        from opifex.data.loaders import create_burgers_loader

        loader = create_burgers_loader(
            n_samples=10,
            batch_size=2,
            resolution=16,
            shuffle=False,
            worker_count=0,  # No multiprocessing for testing
            enable_normalization=False,
            enable_spectral=False,
            enable_augmentation=False,
        )

        assert loader is not None

    def test_loader_iteration(self):
        """Test iterating through a Grain DataLoader."""
        from opifex.data.loaders import create_burgers_loader

        loader = create_burgers_loader(
            n_samples=10,
            batch_size=2,
            resolution=16,
            shuffle=False,
            worker_count=0,
            enable_normalization=False,
        )

        batches = list(loader)

        # With 10 samples and batch_size=2, should get 5 batches
        assert len(batches) == 5

    def test_batch_structure(self):
        """Test that batches have correct structure."""
        from opifex.data.loaders import create_burgers_loader

        loader = create_burgers_loader(
            n_samples=4,
            batch_size=2,
            resolution=16,
            time_steps=5,
            dimension="2d",
            shuffle=False,
            worker_count=0,
            enable_normalization=False,
        )

        batch = next(iter(loader))

        # Should have required keys
        assert "input" in batch
        assert "output" in batch
        assert "viscosity" in batch
        assert "time_points" in batch
        assert "metadata" in batch

    def test_batch_shapes_2d(self):
        """Test that 2D batches have correct shapes."""
        from opifex.data.loaders import create_burgers_loader

        batch_size = 4
        resolution = 32
        time_steps = 5

        loader = create_burgers_loader(
            n_samples=8,
            batch_size=batch_size,
            resolution=resolution,
            time_steps=time_steps,
            dimension="2d",
            shuffle=False,
            worker_count=0,
            enable_normalization=False,
        )

        batch = next(iter(loader))

        # Check shapes
        assert batch["input"].shape == (batch_size, resolution, resolution)
        assert batch["output"].shape == (batch_size, time_steps, resolution, resolution)

    def test_with_normalization(self):
        """Test loader with normalization enabled."""
        from opifex.data.loaders import create_burgers_loader

        loader = create_burgers_loader(
            n_samples=4,
            batch_size=2,
            resolution=16,
            shuffle=False,
            worker_count=0,
            enable_normalization=True,
            normalization_mean=0.0,
            normalization_std=1.0,
        )

        batch = next(iter(loader))

        # Data should be normalized (no specific value check due to random generation)
        assert "input" in batch
        assert "output" in batch

    def test_with_spectral_features(self):
        """Test loader with spectral features enabled."""
        from opifex.data.loaders import create_burgers_loader

        loader = create_burgers_loader(
            n_samples=4,
            batch_size=2,
            resolution=16,
            shuffle=False,
            worker_count=0,
            enable_normalization=False,
            enable_spectral=True,
        )

        batch = next(iter(loader))

        # Should have FFT features
        assert "input_fft" in batch
        assert "input" in batch  # Original should still be present

    def test_with_augmentation(self):
        """Test loader with augmentation enabled."""
        from opifex.data.loaders import create_burgers_loader

        loader = create_burgers_loader(
            n_samples=4,
            batch_size=2,
            resolution=16,
            shuffle=False,
            worker_count=0,
            enable_normalization=False,
            enable_augmentation=True,
            augmentation_noise_level=0.01,
        )

        batch = next(iter(loader))

        # Should have input (with noise added)
        assert "input" in batch

    def test_with_all_transforms(self):
        """Test loader with all transforms enabled."""
        from opifex.data.loaders import create_burgers_loader

        loader = create_burgers_loader(
            n_samples=4,
            batch_size=2,
            resolution=16,
            shuffle=False,
            worker_count=0,
            enable_normalization=True,
            enable_spectral=True,
            enable_augmentation=True,
        )

        batch = next(iter(loader))

        # Should have all features
        assert "input" in batch
        assert "output" in batch
        assert "input_fft" in batch

    def test_shuffle_produces_different_order(self):
        """Test that shuffle=True produces different batch orders."""
        from opifex.data.loaders import create_burgers_loader

        # Create two loaders with different seeds
        loader1 = create_burgers_loader(
            n_samples=10,
            batch_size=2,
            resolution=16,
            shuffle=True,
            seed=42,
            worker_count=0,
            enable_normalization=False,
        )

        loader2 = create_burgers_loader(
            n_samples=10,
            batch_size=2,
            resolution=16,
            shuffle=True,
            seed=43,  # Different seed
            worker_count=0,
            enable_normalization=False,
        )

        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # With different seeds and shuffle, batches should differ
        # (This might occasionally fail due to random chance, but very unlikely)
        assert not jnp.allclose(batch1["input"], batch2["input"])

    def test_reproducible_with_same_seed(self):
        """Test that same seed produces reproducible results."""
        from opifex.data.loaders import create_burgers_loader

        seed = 123

        loader1 = create_burgers_loader(
            n_samples=4,
            batch_size=2,
            resolution=16,
            shuffle=False,
            seed=seed,
            worker_count=0,
            enable_normalization=False,
        )

        loader2 = create_burgers_loader(
            n_samples=4,
            batch_size=2,
            resolution=16,
            shuffle=False,
            seed=seed,
            worker_count=0,
            enable_normalization=False,
        )

        batch1 = next(iter(loader1))
        batch2 = next(iter(loader2))

        # Same seed should produce identical results
        assert jnp.allclose(batch1["input"], batch2["input"])
        assert jnp.allclose(batch1["output"], batch2["output"])
