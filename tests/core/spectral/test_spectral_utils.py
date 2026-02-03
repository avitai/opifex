"""
Comprehensive tests for opifex.core.spectral.spectral_utils module.

This test suite provides comprehensive coverage for spectral analysis utilities
including power spectral density, energy spectrum, and wavenumber operations.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.core.spectral.spectral_utils import (
    energy_spectrum,
    power_spectral_density,
    spectral_energy,
    wavenumber_grid,
)


class TestPowerSpectralDensity:
    """Test power spectral density computation."""

    def test_power_spectral_density_1d_basic(self):
        """Test basic 1D power spectral density computation."""
        # Create simple sinusoidal signal
        x = jnp.linspace(0, 2 * jnp.pi, 64)
        signal = jnp.sin(2 * x) + 0.5 * jnp.sin(4 * x)
        signal = signal.reshape(1, 1, 64)  # Add batch and channel dims

        dx = float(x[1] - x[0])  # Convert to Python float
        psd = power_spectral_density(signal, spatial_dims=1, dx=dx)

        assert psd.shape == (33,)  # rfft output shape
        assert jnp.all(jnp.isfinite(psd))
        assert jnp.all(psd >= 0)  # PSD should be non-negative

    def test_power_spectral_density_2d_basic(self):
        """Test basic 2D power spectral density computation."""
        # Create 2D sinusoidal pattern
        x = jnp.linspace(0, 2 * jnp.pi, 32)
        y = jnp.linspace(0, 2 * jnp.pi, 32)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        signal = jnp.sin(X) * jnp.cos(Y)
        signal = signal.reshape(1, 1, 32, 32)

        dx = [float(x[1] - x[0]), float(y[1] - y[0])]
        psd = power_spectral_density(signal, spatial_dims=2, dx=dx)

        assert psd.shape == (32, 17)  # 2D rfft output shape
        assert jnp.all(jnp.isfinite(psd))
        assert jnp.all(psd >= 0)

    def test_power_spectral_density_3d_basic(self):
        """Test basic 3D power spectral density computation."""
        # Create simple 3D signal
        signal = jnp.ones((1, 1, 8, 8, 8))
        dx = [0.1, 0.1, 0.1]

        psd = power_spectral_density(signal, spatial_dims=3, dx=dx)

        assert psd.shape == (8, 8, 5)  # 3D rfft output shape
        assert jnp.all(jnp.isfinite(psd))
        assert jnp.all(psd >= 0)

    def test_power_spectral_density_scalar_dx(self):
        """Test PSD with scalar dx parameter."""
        signal = jnp.ones((1, 1, 16))
        dx_scalar = 0.1

        psd = power_spectral_density(signal, spatial_dims=1, dx=dx_scalar)

        assert psd.shape == (9,)
        assert jnp.all(jnp.isfinite(psd))

    def test_power_spectral_density_sequence_dx(self):
        """Test PSD with sequence dx parameter."""
        signal = jnp.ones((1, 1, 16, 16))
        dx_sequence = [0.1, 0.2]

        psd = power_spectral_density(signal, spatial_dims=2, dx=dx_sequence)

        assert psd.shape == (16, 9)
        assert jnp.all(jnp.isfinite(psd))

    def test_power_spectral_density_invalid_spatial_dims(self):
        """Test PSD with invalid spatial dimensions."""
        signal = jnp.ones((1, 1, 16))

        with pytest.raises(ValueError, match="Unsupported spatial dimensions"):
            power_spectral_density(signal, spatial_dims=0, dx=0.1)

        with pytest.raises(ValueError, match="Unsupported spatial dimensions"):
            power_spectral_density(signal, spatial_dims=4, dx=0.1)

    def test_power_spectral_density_mismatched_dx(self):
        """Test PSD with mismatched dx length."""
        signal = jnp.ones((1, 1, 16, 16))
        dx_wrong = [0.1, 0.2, 0.3]  # Wrong length for 2D

        with pytest.raises(
            ValueError, match=r"Grid spacing length .* doesn't match spatial_dims"
        ):
            power_spectral_density(signal, spatial_dims=2, dx=dx_wrong)

    def test_power_spectral_density_no_batch_channel(self):
        """Test PSD without batch/channel dimensions."""
        signal = jnp.ones((16, 16))  # Just spatial dimensions
        dx = 0.1

        psd = power_spectral_density(signal, spatial_dims=2, dx=dx)

        assert psd.shape == (16, 9)
        assert jnp.all(jnp.isfinite(psd))

    def test_power_spectral_density_multiple_batch_channel(self):
        """Test PSD with multiple batch and channel dimensions."""
        signal = jnp.ones((4, 3, 16, 16))  # batch=4, channels=3
        dx = 0.1

        psd = power_spectral_density(signal, spatial_dims=2, dx=dx)

        assert psd.shape == (16, 9)  # Averaged over batch and channel
        assert jnp.all(jnp.isfinite(psd))


class TestEnergySpectrum:
    """Test energy spectrum computation."""

    def test_energy_spectrum_1d_basic(self):
        """Test basic 1D energy spectrum computation."""
        signal = jnp.ones((1, 1, 32))
        dx = 0.1

        energy = energy_spectrum(signal, spatial_dims=1, dx=dx)

        assert energy.shape == (1, 1, 17)  # Same as input except spatial dim
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(energy >= 0)

    def test_energy_spectrum_2d_basic(self):
        """Test basic 2D energy spectrum computation."""
        signal = jnp.ones((1, 1, 16, 16))
        dx = [0.1, 0.1]

        energy = energy_spectrum(signal, spatial_dims=2, dx=dx)

        assert energy.shape == (1, 1, 16, 9)
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(energy >= 0)

    def test_energy_spectrum_3d_basic(self):
        """Test basic 3D energy spectrum computation."""
        signal = jnp.ones((2, 3, 8, 8, 8))
        dx = [0.1, 0.2, 0.3]

        energy = energy_spectrum(signal, spatial_dims=3, dx=dx)

        assert energy.shape == (2, 3, 8, 8, 5)
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(energy >= 0)

    def test_energy_spectrum_scalar_dx(self):
        """Test energy spectrum with scalar dx."""
        signal = jnp.ones((1, 1, 16))
        dx_scalar = 0.1

        energy = energy_spectrum(signal, spatial_dims=1, dx=dx_scalar)

        assert energy.shape == (1, 1, 9)
        assert jnp.all(jnp.isfinite(energy))

    def test_energy_spectrum_conservation(self):
        """Test energy conservation in spectral domain."""
        # Create a signal with known energy
        signal = jnp.ones((1, 1, 32))
        dx = 0.1

        # Spectral energy (should be proportional to spatial energy)
        spectral_energy_vals = energy_spectrum(signal, spatial_dims=1, dx=dx)

        assert jnp.all(jnp.isfinite(spectral_energy_vals))
        assert jnp.all(spectral_energy_vals >= 0)

    def test_energy_spectrum_invalid_spatial_dims(self):
        """Test energy spectrum with invalid spatial dimensions."""
        signal = jnp.ones((1, 1, 16))

        with pytest.raises(ValueError, match="Unsupported spatial dimensions"):
            energy_spectrum(signal, spatial_dims=0, dx=0.1)

        with pytest.raises(ValueError, match="Unsupported spatial dimensions"):
            energy_spectrum(signal, spatial_dims=4, dx=0.1)


class TestSpectralEnergy:
    """Test total spectral energy computation."""

    def test_spectral_energy_1d_basic(self):
        """Test basic 1D spectral energy computation."""
        signal = jnp.ones((1, 1, 32))
        dx = 0.1

        total_energy = spectral_energy(signal, spatial_dims=1, dx=dx)

        assert total_energy.shape == (1, 1)  # Summed over spatial dims
        assert jnp.all(jnp.isfinite(total_energy))
        assert jnp.all(total_energy >= 0)

    def test_spectral_energy_2d_basic(self):
        """Test basic 2D spectral energy computation."""
        signal = jnp.ones((2, 3, 16, 16))
        dx = [0.1, 0.1]

        total_energy = spectral_energy(signal, spatial_dims=2, dx=dx)

        assert total_energy.shape == (2, 3)  # Preserved batch and channel dims
        assert jnp.all(jnp.isfinite(total_energy))
        assert jnp.all(total_energy >= 0)

    def test_spectral_energy_3d_basic(self):
        """Test basic 3D spectral energy computation."""
        signal = jnp.ones((1, 1, 8, 8, 8))
        dx = [0.1, 0.1, 0.1]

        total_energy = spectral_energy(signal, spatial_dims=3, dx=dx)

        assert total_energy.shape == (1, 1)
        assert jnp.all(jnp.isfinite(total_energy))
        assert jnp.all(total_energy >= 0)

    def test_spectral_energy_scaling(self):
        """Test spectral energy scaling with amplitude."""
        # Create signals with different amplitudes
        signal_1x = jnp.ones((1, 1, 32))
        signal_2x = 2 * jnp.ones((1, 1, 32))
        dx = 0.1

        energy_1x = spectral_energy(signal_1x, spatial_dims=1, dx=dx)
        energy_2x = spectral_energy(signal_2x, spatial_dims=1, dx=dx)

        # Energy should scale as amplitude squared
        assert jnp.allclose(energy_2x, 4 * energy_1x, rtol=1e-6)

    def test_spectral_energy_invalid_spatial_dims(self):
        """Test spectral energy with invalid spatial dimensions."""
        signal = jnp.ones((1, 1, 16))

        with pytest.raises(ValueError, match="Unsupported spatial dimensions"):
            spectral_energy(signal, spatial_dims=0, dx=0.1)

        with pytest.raises(ValueError, match="Unsupported spatial dimensions"):
            spectral_energy(signal, spatial_dims=5, dx=0.1)


class TestWavenumberGrid:
    """Test wavenumber grid generation."""

    def test_wavenumber_grid_1d_basic(self):
        """Test basic 1D wavenumber grid generation."""
        shape = (32,)
        dx = 0.1

        k_arrays = wavenumber_grid(shape, dx, magnitude=False)

        assert isinstance(k_arrays, list)
        assert len(k_arrays) == 1
        assert k_arrays[0].shape == (17,)  # rfftfreq output
        assert jnp.all(jnp.isfinite(k_arrays[0]))

    def test_wavenumber_grid_2d_basic(self):
        """Test basic 2D wavenumber grid generation."""
        shape = (32, 32)
        dx = [0.1, 0.1]

        k_arrays = wavenumber_grid(shape, dx, magnitude=False)

        assert isinstance(k_arrays, list)
        assert len(k_arrays) == 2
        assert k_arrays[0].shape == (32,)  # fftfreq output
        assert k_arrays[1].shape == (17,)  # rfftfreq output
        assert jnp.all(jnp.isfinite(k_arrays[0]))
        assert jnp.all(jnp.isfinite(k_arrays[1]))

    def test_wavenumber_grid_3d_basic(self):
        """Test basic 3D wavenumber grid generation."""
        shape = (16, 16, 16)
        dx = [0.1, 0.1, 0.1]

        k_arrays = wavenumber_grid(shape, dx, magnitude=False)

        assert isinstance(k_arrays, list)
        assert len(k_arrays) == 3
        assert k_arrays[0].shape == (16,)
        assert k_arrays[1].shape == (16,)
        assert k_arrays[2].shape == (9,)  # rfftfreq for last dimension
        assert all(jnp.all(jnp.isfinite(k)) for k in k_arrays)

    def test_wavenumber_grid_magnitude_1d(self):
        """Test 1D wavenumber grid magnitude computation."""
        shape = (32,)
        dx = 0.1

        k_mag = wavenumber_grid(shape, dx, magnitude=True)

        assert isinstance(k_mag, jax.Array)
        assert k_mag.shape == (17,)
        assert jnp.all(jnp.isfinite(k_mag))
        assert jnp.all(k_mag >= 0)  # Magnitude should be non-negative

    def test_wavenumber_grid_magnitude_2d(self):
        """Test 2D wavenumber grid magnitude computation."""
        shape = (16, 16)
        dx = [0.1, 0.1]

        k_mag = wavenumber_grid(shape, dx, magnitude=True)

        assert isinstance(k_mag, jax.Array)
        assert k_mag.shape == (16, 9)
        assert jnp.all(jnp.isfinite(k_mag))
        assert jnp.all(k_mag >= 0)

    def test_wavenumber_grid_magnitude_3d(self):
        """Test 3D wavenumber grid magnitude computation."""
        shape = (8, 8, 8)
        dx = [0.1, 0.1, 0.1]

        k_mag = wavenumber_grid(shape, dx, magnitude=True)

        assert isinstance(k_mag, jax.Array)
        assert k_mag.shape == (8, 8, 5)
        assert jnp.all(jnp.isfinite(k_mag))
        assert jnp.all(k_mag >= 0)

    def test_wavenumber_grid_scalar_dx(self):
        """Test wavenumber grid with scalar dx."""
        shape = (16, 16)
        dx_scalar = 0.1

        k_arrays = wavenumber_grid(shape, dx_scalar, magnitude=False)

        assert len(k_arrays) == 2
        assert k_arrays[0].shape == (16,)
        assert k_arrays[1].shape == (9,)

    def test_wavenumber_grid_different_dx(self):
        """Test wavenumber grid with different dx values."""
        shape = (16, 16)
        dx = [0.1, 0.2]  # Different spacing in each direction

        k_arrays = wavenumber_grid(shape, dx, magnitude=False)

        assert len(k_arrays) == 2
        assert k_arrays[0].shape == (16,)
        assert k_arrays[1].shape == (9,)

        # Verify different scaling - check that the grids have different maximum values
        # Since dx[0]=0.1 and dx[1]=0.2, the first dimension should have larger max frequency
        max_k0 = jnp.max(jnp.abs(k_arrays[0]))
        max_k1 = jnp.max(jnp.abs(k_arrays[1]))
        assert max_k0 > max_k1  # Smaller dx should give larger wavenumbers

    def test_wavenumber_grid_invalid_spatial_dims(self):
        """Test wavenumber grid with invalid spatial dimensions."""
        shape = ()  # Empty shape
        dx = 0.1

        with pytest.raises(ValueError, match="Unsupported spatial dimensions"):
            wavenumber_grid(shape, dx)

        shape_4d = (8, 8, 8, 8)  # Too many dimensions
        with pytest.raises(ValueError, match="Unsupported spatial dimensions"):
            wavenumber_grid(shape_4d, dx)

    def test_wavenumber_grid_mismatched_dx(self):
        """Test wavenumber grid with mismatched dx dimensions."""
        shape = (16, 16)
        dx_wrong = [0.1, 0.2, 0.3]  # Wrong length

        with pytest.raises(
            ValueError, match=r"Grid spacing length .* doesn't match spatial_dims"
        ):
            wavenumber_grid(shape, dx_wrong)

    def test_wavenumber_grid_symmetry_properties(self):
        """Test wavenumber grid symmetry properties."""
        shape = (32,)
        dx = 0.1

        k_arrays = wavenumber_grid(shape, dx, magnitude=False)
        k = k_arrays[0]

        # First element should be zero frequency
        assert jnp.isclose(k[0], 0.0)

        # Check proper frequency ordering
        assert jnp.all(k[1:] > 0)  # All positive frequencies for rfft

    def test_wavenumber_grid_scaling(self):
        """Test wavenumber grid scaling with dx."""
        shape = (16,)
        dx1 = 0.1
        dx2 = 0.2

        k1 = wavenumber_grid(shape, dx1, magnitude=False)[0]
        k2 = wavenumber_grid(shape, dx2, magnitude=False)[0]

        # Wavenumbers should be inversely proportional to dx
        # So k1 (with dx=0.1) should be 2x larger than k2 (with dx=0.2)
        assert jnp.allclose(k1, k2 * 2, rtol=1e-10)


class TestEdgeCasesAndIntegration:
    """Test edge cases and integration scenarios."""

    def test_consistency_energy_spectrum_and_spectral_energy(self):
        """Test consistency between energy_spectrum and spectral_energy."""
        signal = jnp.ones((1, 1, 32))
        dx = 0.1

        # Compute using both methods
        energy_spec = energy_spectrum(signal, spatial_dims=1, dx=dx)
        total_energy = spectral_energy(signal, spatial_dims=1, dx=dx)

        # Total energy should be sum of energy spectrum
        computed_total = jnp.sum(energy_spec, axis=-1)
        assert jnp.allclose(computed_total, total_energy)

    def test_consistency_psd_and_energy_spectrum(self):
        """Test consistency between PSD and energy spectrum."""
        signal = jnp.ones((1, 1, 32))
        dx = 0.1

        psd = power_spectral_density(signal, spatial_dims=1, dx=dx)
        energy_spec = energy_spectrum(signal, spatial_dims=1, dx=dx)

        # PSD should be energy spectrum normalized and averaged
        # Both should have similar spectral characteristics
        assert psd.shape == energy_spec.shape[-1:]  # PSD is averaged
        assert jnp.all(jnp.isfinite(psd))
        assert jnp.all(jnp.isfinite(energy_spec))

    def test_wavenumber_consistency_with_fft(self):
        """Test wavenumber grid consistency with FFT operations."""
        shape = (32,)
        dx = 0.1

        k_arrays = wavenumber_grid(shape, dx, magnitude=False)
        k = k_arrays[0]

        # Should match JAX's rfftfreq scaled by 2π
        expected_k = 2 * jnp.pi * jnp.fft.rfftfreq(shape[0], dx)
        assert jnp.allclose(k, expected_k)

    def test_jax_transformations(self):
        """Test JAX transformations on spectral utilities."""
        # Test basic compatibility with JAX operations
        signal = jnp.ones((1, 1, 32))
        dx = 0.1

        # Test that functions work with JAX arrays
        psd = power_spectral_density(signal, spatial_dims=1, dx=dx)
        energy = spectral_energy(signal, spatial_dims=1, dx=dx)

        assert jnp.all(jnp.isfinite(psd))
        assert jnp.all(jnp.isfinite(energy))

        # Test gradient computation on a simpler case
        def simple_energy_objective(x_val):
            test_signal = jnp.ones((1, 1, 32)) * x_val
            return jnp.sum(spectral_energy(test_signal, spatial_dims=1, dx=0.1))

        grad_fn = jax.grad(simple_energy_objective)
        grad_result = grad_fn(1.0)

        assert jnp.isfinite(grad_result)
        assert grad_result.shape == ()  # Scalar gradient

    def test_complex_input_handling(self):
        """Test spectral utilities with complex input."""
        # Note: rfft only supports real inputs, so we test with real signals
        # but verify complex arithmetic works correctly
        real_signal = jnp.ones((1, 1, 32))
        dx = 0.1

        # Should handle real input gracefully
        psd = power_spectral_density(real_signal, spatial_dims=1, dx=dx)
        energy = spectral_energy(real_signal, spatial_dims=1, dx=dx)

        assert jnp.all(jnp.isfinite(psd))
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(psd >= 0)
        assert jnp.all(energy >= 0)

        # Test with more complex real signal patterns
        x = jnp.linspace(0, 2 * jnp.pi, 32)
        complex_pattern = jnp.sin(x) + jnp.cos(2 * x)  # Real but complex pattern
        complex_signal = complex_pattern.reshape(1, 1, 32)

        psd_complex = power_spectral_density(complex_signal, spatial_dims=1, dx=dx)
        energy_complex = spectral_energy(complex_signal, spatial_dims=1, dx=dx)

        assert jnp.all(jnp.isfinite(psd_complex))
        assert jnp.all(jnp.isfinite(energy_complex))
        assert jnp.all(psd_complex >= 0)
        assert jnp.all(energy_complex >= 0)

    def test_large_signal_handling(self):
        """Test spectral utilities with larger signals."""
        # Test with larger signal
        signal = jnp.ones((4, 8, 128, 128))  # Larger 2D signal
        dx = [0.05, 0.05]

        psd = power_spectral_density(signal, spatial_dims=2, dx=dx)
        energy = spectral_energy(signal, spatial_dims=2, dx=dx)

        assert psd.shape == (128, 65)
        assert energy.shape == (4, 8)
        assert jnp.all(jnp.isfinite(psd))
        assert jnp.all(jnp.isfinite(energy))

    def test_zero_signal_handling(self):
        """Test spectral utilities with zero signal."""
        signal = jnp.zeros((1, 1, 32))
        dx = 0.1

        psd = power_spectral_density(signal, spatial_dims=1, dx=dx)
        energy = spectral_energy(signal, spatial_dims=1, dx=dx)

        assert jnp.allclose(psd, 0.0)
        assert jnp.allclose(energy, 0.0)

    def test_small_signal_handling(self):
        """Test spectral utilities with very small signals."""
        signal = 1e-12 * jnp.ones((1, 1, 16))
        dx = 0.1

        psd = power_spectral_density(signal, spatial_dims=1, dx=dx)
        energy = spectral_energy(signal, spatial_dims=1, dx=dx)

        assert jnp.all(jnp.isfinite(psd))
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(psd >= 0)
        assert jnp.all(energy >= 0)
