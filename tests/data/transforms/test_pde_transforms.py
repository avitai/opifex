"""
TDD Tests for Grain transforms (PDE data processing).

Following TDD principles: Tests written FIRST, then implementation.
"""

import jax.numpy as jnp


# Tests will fail until implementation is complete


class TestNormalizeTransform:
    """Test suite for NormalizeTransform following TDD principles."""

    def test_import_normalize_transform(self):
        """Test that we can import NormalizeTransform."""
        from opifex.data.transforms import (
            NormalizeTransform,  # noqa: F401 # pyright: ignore[reportUnusedImport]
        )

    def test_transform_initialization(self):
        """Test basic initialization of NormalizeTransform."""
        from opifex.data.transforms import NormalizeTransform

        transform = NormalizeTransform(mean=0.5, std=0.2)
        assert transform is not None

    def test_transform_normalizes_data(self):
        """Test that transform applies normalization correctly."""
        from opifex.data.transforms import NormalizeTransform

        transform = NormalizeTransform(mean=0.0, std=1.0)

        # Create sample with known statistics
        sample = {
            "input": jnp.array([1.0, 2.0, 3.0]),
            "output": jnp.array([4.0, 5.0, 6.0]),
        }

        result = transform.map(sample)

        # Should normalize input and output
        assert "input" in result
        assert "output" in result

    def test_transform_with_custom_mean_std(self):
        """Test normalization with custom mean and std."""
        from opifex.data.transforms import NormalizeTransform

        mean = 2.0
        std = 0.5
        transform = NormalizeTransform(mean=mean, std=std)

        sample = {"input": jnp.array([2.0, 2.5, 3.0])}

        result = transform.map(sample)

        # Manual calculation: (x - mean) / std
        expected = (jnp.array([2.0, 2.5, 3.0]) - mean) / std

        assert jnp.allclose(result["input"], expected)

    def test_transform_preserves_other_keys(self):
        """Test that transform preserves keys it doesn't modify."""
        from opifex.data.transforms import NormalizeTransform

        transform = NormalizeTransform(mean=0.0, std=1.0)

        sample = {
            "input": jnp.array([1.0, 2.0]),
            "output": jnp.array([3.0, 4.0]),
            "metadata": {"index": 0},
            "viscosity": 0.05,
        }

        result = transform.map(sample)

        # Metadata and viscosity should be preserved
        assert "metadata" in result
        assert "viscosity" in result
        assert result["metadata"]["index"] == 0
        assert result["viscosity"] == 0.05

    def test_transform_handles_zero_std(self):
        """Test that transform handles zero std gracefully."""
        from opifex.data.transforms import NormalizeTransform

        transform = NormalizeTransform(mean=0.0, std=0.0)

        sample = {"input": jnp.array([1.0, 2.0, 3.0])}

        # Should not raise error (implementation should add epsilon)
        result = transform.map(sample)
        assert result is not None


class TestSpectralTransform:
    """Test suite for SpectralTransform following TDD principles."""

    def test_import_spectral_transform(self):
        """Test that we can import SpectralTransform."""
        from opifex.data.transforms import (
            SpectralTransform,  # noqa: F401 # pyright: ignore[reportUnusedImport]
        )

    def test_transform_adds_fft(self):
        """Test that transform adds FFT features."""
        from opifex.data.transforms import SpectralTransform

        transform = SpectralTransform()

        sample = {"input": jnp.array([1.0, 2.0, 3.0, 4.0])}

        result = transform.map(sample)

        # Should add FFT features
        assert "input_fft" in result
        assert "input" in result  # Original should be preserved

    def test_transform_fft_correct(self):
        """Test that FFT computation is correct."""
        from opifex.data.transforms import SpectralTransform

        transform = SpectralTransform()

        data = jnp.array([1.0, 2.0, 3.0, 4.0])
        sample = {"input": data}

        result = transform.map(sample)

        # Verify FFT is correct
        expected_fft = jnp.fft.rfft(data)
        assert jnp.allclose(result["input_fft"], expected_fft)

    def test_transform_works_with_2d(self):
        """Test that transform works with 2D data."""
        from opifex.data.transforms import SpectralTransform

        transform = SpectralTransform()

        data_2d = jnp.ones((8, 8))
        sample = {"input": data_2d}

        result = transform.map(sample)

        # Should have FFT features
        assert "input_fft" in result
        # FFT along last axis
        expected_fft = jnp.fft.rfft(data_2d, axis=-1)
        assert result["input_fft"].shape == expected_fft.shape


class TestAddNoiseAugmentation:
    """Test suite for AddNoiseAugmentation following TDD principles."""

    def test_import_add_noise(self):
        """Test that we can import AddNoiseAugmentation."""
        from opifex.data.transforms import (
            AddNoiseAugmentation,  # noqa: F401 # pyright: ignore[reportUnusedImport]
        )

    def test_augmentation_initialization(self):
        """Test basic initialization."""
        from opifex.data.transforms import AddNoiseAugmentation

        aug = AddNoiseAugmentation(noise_level=0.01)
        assert aug is not None

    def test_augmentation_adds_noise(self):
        """Test that augmentation modifies input."""
        from opifex.data.transforms import AddNoiseAugmentation

        aug = AddNoiseAugmentation(noise_level=0.1)

        original_input = jnp.array([1.0, 2.0, 3.0, 4.0])
        sample = {"input": original_input.copy()}

        result = aug.map(sample)

        # Input should be modified (with high probability for noise_level=0.1)
        # We can't test exact values due to randomness, but shape should match
        assert result["input"].shape == original_input.shape

    def test_augmentation_configurable_noise_level(self):
        """Test that noise level is configurable."""
        from opifex.data.transforms import AddNoiseAugmentation

        # Very small noise
        aug_small = AddNoiseAugmentation(noise_level=0.001)
        aug_large = AddNoiseAugmentation(noise_level=1.0)

        assert aug_small.noise_level == 0.001
        assert aug_large.noise_level == 1.0

    def test_augmentation_preserves_shape(self):
        """Test that augmentation preserves array shape."""
        from opifex.data.transforms import AddNoiseAugmentation

        aug = AddNoiseAugmentation(noise_level=0.1)

        # Test with different shapes
        for shape in [(10,), (8, 8), (4, 16, 16)]:
            sample = {"input": jnp.ones(shape)}
            result = aug.map(sample)
            assert result["input"].shape == shape

    def test_augmentation_does_not_modify_output(self):
        """Test that augmentation only affects input, not output."""
        from opifex.data.transforms import AddNoiseAugmentation

        aug = AddNoiseAugmentation(noise_level=0.1)

        output_data = jnp.array([5.0, 6.0, 7.0, 8.0])
        sample = {
            "input": jnp.array([1.0, 2.0, 3.0, 4.0]),
            "output": output_data.copy(),
        }

        result = aug.map(sample)

        # Output should be unchanged
        assert jnp.allclose(result["output"], output_data)


class TestTransformComposition:
    """Test that multiple transforms can be composed."""

    def test_normalize_then_spectral(self):
        """Test composing normalization and spectral transform."""
        from opifex.data.transforms import NormalizeTransform, SpectralTransform

        normalize = NormalizeTransform(mean=0.0, std=1.0)
        spectral = SpectralTransform()

        sample = {"input": jnp.array([1.0, 2.0, 3.0, 4.0])}

        # Apply transforms in sequence
        result = normalize.map(sample)
        result = spectral.map(result)

        # Should have both normalized input and FFT
        assert "input" in result
        assert "input_fft" in result

    def test_augmentation_then_normalize(self):
        """Test composing augmentation and normalization."""
        from opifex.data.transforms import AddNoiseAugmentation, NormalizeTransform

        augment = AddNoiseAugmentation(noise_level=0.05)
        normalize = NormalizeTransform(mean=0.0, std=1.0)

        sample = {"input": jnp.array([1.0, 2.0, 3.0, 4.0])}

        # Apply transforms in sequence
        result = augment.map(sample)
        result = normalize.map(result)

        # Should have modified and normalized input
        assert "input" in result
        assert result["input"].shape == (4,)
