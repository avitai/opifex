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

    def test_binary_field_type_is_high_contrast(self):
        """field_type='binary' yields the benchmark binary high-contrast field (Li et al. 2020).

        The coefficient must take exactly the two configured values, giving a
        contrast that produces strong, learnable input-dependent solution
        variation (a smooth low-contrast field is nearly a constant operator).
        """
        from opifex.data.sources import DarcyDataSource

        low, high = 3.0, 12.0
        source = DarcyDataSource(
            n_samples=5, resolution=32, viscosity_range=(low, high), field_type="binary", seed=42
        )
        a = jnp.asarray(source[0]["input"])

        unique_values = jnp.unique(a)
        assert unique_values.size == 2, "Permeability must be binary (two values)"
        assert jnp.allclose(jnp.sort(unique_values), jnp.array([low, high]))
        # Both phases must actually appear (a non-degenerate threshold).
        assert jnp.any(a == low) and jnp.any(a == high)

    def test_smooth_field_type_is_default_and_continuous(self):
        """The default field_type='smooth' produces a continuous (many-valued) field."""
        from opifex.data.sources import DarcyDataSource

        source = DarcyDataSource(n_samples=3, resolution=32, seed=42)
        assert source.field_type == "smooth"
        a = jnp.asarray(source[0]["input"])
        assert jnp.unique(a).size > 2, "Smooth permeability must be continuous, not binary"

    def test_invalid_field_type_raises(self):
        """An unknown field_type is rejected at construction."""
        from opifex.data.sources import DarcyDataSource

        with pytest.raises(ValueError, match="field_type"):
            DarcyDataSource(n_samples=2, resolution=16, field_type="nonsense")

    def test_binary_solution_has_learnable_signal(self):
        """The binary a->u map varies across samples far more than a constant predictor.

        If samples were near-identical (low-contrast data), a constant mean field
        would already be ~accurate and there would be nothing to learn.
        """
        from opifex.data.sources import DarcyDataSource

        source = DarcyDataSource(n_samples=16, resolution=32, field_type="binary", seed=7)
        outputs = jnp.stack([jnp.asarray(source[i]["output"]) for i in range(16)])
        mean_field = outputs.mean(axis=0, keepdims=True)
        flat_dev = (outputs - mean_field).reshape(16, -1)
        flat = outputs.reshape(16, -1)
        mean_baseline = jnp.mean(jnp.linalg.norm(flat_dev, axis=1) / jnp.linalg.norm(flat, axis=1))
        # Binary high-contrast data gives ~0.13 here; the old smooth low-contrast
        # field gave ~0.09. 0.10 separates a learnable task from a near-constant one.
        assert mean_baseline > 0.10, (
            f"Samples too self-similar (mean-predictor rel-L2={mean_baseline:.3f}); "
            "the operator-learning task has too little input-dependent signal."
        )
