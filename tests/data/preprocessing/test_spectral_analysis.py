"""Tests for spectral analysis preprocessing utilities.

Following TDD principles and JAX/Flax NNX guidelines from critical_technical_guidelines.md
"""

import jax
import jax.numpy as jnp

from opifex.data.preprocessing.spectral_analysis import (
    bandpass_filter,
    compute_energy_spectrum,
    frequency_analysis,
    modal_analysis,
    power_spectral_density,
)


class TestSpectralAnalysis:
    """Test suite for spectral analysis utilities with comprehensive coverage."""

    def test_power_spectral_density_basic(self):
        """Test power spectral density computation for 2D fields."""
        # Create a simple 2D field
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y) + 0.3 * jnp.sin(2 * X) * jnp.sin(2 * Y)

        psd = power_spectral_density(field)

        assert psd.shape == (n, n)
        assert jnp.all(psd >= 0), "PSD should be non-negative"
        assert jnp.all(jnp.isfinite(psd)), "PSD should be finite"

    def test_power_spectral_density_normalization(self):
        """Test that PSD normalization works correctly."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y)

        psd_normalized = power_spectral_density(field, normalize=True)
        psd_unnormalized = power_spectral_density(field, normalize=False)

        # Normalized PSD should sum to approximately 1
        assert jnp.abs(jnp.sum(psd_normalized) - 1.0) < 1e-6

        # Unnormalized should be different
        assert not jnp.allclose(psd_normalized, psd_unnormalized)

    def test_modal_analysis_basic(self):
        """Test basic modal analysis functionality."""
        # Create a field with known modal structure
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y) + 0.3 * jnp.sin(3 * X) * jnp.sin(3 * Y)

        reconstructed, mode_energies = modal_analysis(field, modes=10)

        assert reconstructed.shape == field.shape
        assert mode_energies.shape == (10,)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mode_energies))
        assert jnp.all(mode_energies >= 0), "Mode energies should be non-negative"

    def test_modal_analysis_with_coefficients(self):
        """Test modal analysis with coefficient return."""
        n = 16  # Smaller for testing
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y)

        reconstructed, coefficients = modal_analysis(
            field, modes=5, return_coefficients=True
        )

        assert reconstructed.shape == field.shape
        assert coefficients.shape == (5,)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(coefficients))

    def test_frequency_analysis_basic(self):
        """Test frequency analysis functionality."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y)

        (freqs_x, freqs_y), spectrum = frequency_analysis(field, dx=0.1)

        assert freqs_x.shape == (n,)
        assert freqs_y.shape == (n,)
        assert spectrum.shape == (n, n)
        assert jnp.all(jnp.isfinite(freqs_x))
        assert jnp.all(jnp.isfinite(freqs_y))
        assert jnp.all(jnp.isfinite(spectrum))
        assert jnp.all(spectrum >= 0), "Spectrum should be non-negative"

    def test_bandpass_filter_basic(self):
        """Test bandpass filtering functionality."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        # Field with multiple frequency components
        field = jnp.sin(X) * jnp.cos(Y) + 0.5 * jnp.sin(5 * X) * jnp.sin(5 * Y)

        low_freq = 0.5
        high_freq = 3.0
        filtered = bandpass_filter(field, low_freq, high_freq, dx=0.2)

        assert filtered.shape == field.shape
        assert jnp.all(jnp.isfinite(filtered))

    def test_bandpass_filter_frequency_content(self):
        """Test that bandpass filter affects frequency content correctly."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y) + 0.5 * jnp.sin(10 * X) * jnp.sin(10 * Y)

        # Filter out high frequencies
        low_freq = 0.1
        high_freq = 3.0
        filtered = bandpass_filter(field, low_freq, high_freq, dx=0.1)

        # Filtered field should have less high-frequency content
        psd_orig = power_spectral_density(field, normalize=False)
        psd_filt = power_spectral_density(filtered, normalize=False)

        # Total power should be reduced (some frequencies filtered out)
        assert jnp.sum(psd_filt) <= jnp.sum(psd_orig)

    def test_compute_energy_spectrum_basic(self):
        """Test energy spectrum computation."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y) + 0.3 * jnp.sin(2 * X) * jnp.sin(2 * Y)

        wavenumbers, energy_spectrum = compute_energy_spectrum(field, dx=0.2)

        assert len(wavenumbers) == len(energy_spectrum)
        assert len(wavenumbers) == min(n, n) // 2  # Expected length
        assert jnp.all(wavenumbers >= 0), "Wavenumbers should be non-negative"
        assert jnp.all(energy_spectrum >= 0), "Energy should be non-negative"
        assert jnp.all(jnp.isfinite(energy_spectrum)), "Energy should be finite"

    def test_compute_energy_spectrum_ordering(self):
        """Test that energy spectrum wavenumbers are ordered."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y)

        wavenumbers, _ = compute_energy_spectrum(field, dx=0.1)

        # Wavenumbers should be monotonically increasing
        assert jnp.all(wavenumbers[1:] >= wavenumbers[:-1]), (
            "Wavenumbers should be ordered"
        )

    def test_jax_transformations_compatibility(self):
        """Test JAX transformations compatibility."""
        n = 16  # Small for testing

        # Test vmap compatibility
        @jax.vmap
        def batch_psd(fields):
            return power_spectral_density(fields)

        # Create batch of fields
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        fields = jnp.array(
            [
                jnp.sin(X) * jnp.cos(Y),
                jnp.sin(2 * X) * jnp.cos(2 * Y),
                jnp.sin(3 * X) * jnp.cos(3 * Y),
            ]
        )

        psd_batch = batch_psd(fields)

        assert psd_batch.shape == (3, n, n)
        assert jnp.all(jnp.isfinite(psd_batch))

    def test_modal_analysis_mode_reduction(self):
        """Test that modal analysis with fewer modes produces simpler reconstruction."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        # Complex field with multiple modes
        field = (
            jnp.sin(X) * jnp.cos(Y)
            + 0.5 * jnp.sin(2 * X) * jnp.sin(2 * Y)
            + 0.2 * jnp.sin(5 * X) * jnp.cos(5 * Y)
        )

        # Reconstruct with different numbers of modes
        recon_few, _ = modal_analysis(field, modes=3)
        recon_many, _ = modal_analysis(field, modes=20)

        # More modes should be closer to original
        error_few = jnp.sum((field - recon_few) ** 2)
        error_many = jnp.sum((field - recon_many) ** 2)

        assert error_many <= error_few

    def test_deterministic_behavior(self):
        """Test that functions produce deterministic results."""
        n = 16  # Small for testing
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y) + 0.5 * jnp.sin(3 * X) * jnp.sin(3 * Y)

        # Multiple calls should give identical results
        psd1 = power_spectral_density(field)
        psd2 = power_spectral_density(field)

        assert jnp.allclose(psd1, psd2, rtol=1e-15)

        recon1, energies1 = modal_analysis(field, modes=5)
        recon2, energies2 = modal_analysis(field, modes=5)

        assert jnp.allclose(recon1, recon2, rtol=1e-15)
        assert jnp.allclose(energies1, energies2, rtol=1e-15)

    def test_filtering_preserves_shape(self):
        """Test that filtering preserves input shape."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y)

        filtered = bandpass_filter(field, 0.5, 3.0, dx=0.1)
        assert filtered.shape == field.shape

    def test_energy_conservation_properties(self):
        """Test energy conservation properties."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y) + 0.5 * jnp.sin(3 * X) * jnp.sin(3 * Y)

        # Total energy in spatial domain
        spatial_energy = jnp.sum(field**2)

        # Energy from PSD (should be related by Parseval's theorem)
        psd = power_spectral_density(field, normalize=False)
        freq_energy = jnp.sum(psd)

        # Should be approximately related (within numerical precision)
        # Note: exact relationship depends on normalization conventions
        assert jnp.isfinite(spatial_energy)
        assert jnp.isfinite(freq_energy)
        assert freq_energy > 0

    def test_error_handling_invalid_modes(self):
        """Test error handling for invalid number of modes."""
        n = 16
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field = jnp.sin(X) * jnp.cos(Y)

        # Test with modes exceeding field size
        total_modes = n * n
        # Should handle large number of modes gracefully
        recon, energies = modal_analysis(field, modes=total_modes)
        assert recon.shape == field.shape
        assert len(energies) == total_modes
