"""Tests for core spectral FFT operations.

This module tests the fundamental FFT utilities that will be used throughout
the spectral framework. Following test-driven development, these tests define
the expected behavior before implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.spectral.fft_operations import (
    fft_frequency_grid,
    get_spectral_frequencies,
    spectral_derivative,
    spectral_filter,
    standardized_fft,
    standardized_ifft,
)


class TestStandardizedFFT:
    """Test standardized FFT operations with consistent API."""

    def test_fft_1d_basic(self):
        """Test 1D FFT with basic functionality."""
        # Setup
        batch_size, channels, spatial_size = 4, 3, 32
        x = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, channels, spatial_size)
        )

        # Execute
        x_ft = standardized_fft(x, spatial_dims=1)

        # Verify
        expected_shape = (batch_size, channels, spatial_size // 2 + 1)
        assert x_ft.shape == expected_shape

        # Check dtype based on JAX precision setting
        expected_dtype = (
            jnp.complex128 if jax.config.read("jax_enable_x64") else jnp.complex64
        )
        assert x_ft.dtype == expected_dtype

    def test_fft_2d_basic(self):
        """Test 2D FFT with basic functionality."""
        # Setup
        batch_size, channels, h, w = 4, 3, 16, 20
        x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, channels, h, w))

        # Execute
        x_ft = standardized_fft(x, spatial_dims=2)

        # Verify
        expected_shape = (batch_size, channels, h, w // 2 + 1)
        assert x_ft.shape == expected_shape

        # Check dtype based on JAX precision setting
        expected_dtype = (
            jnp.complex128 if jax.config.read("jax_enable_x64") else jnp.complex64
        )
        assert x_ft.dtype == expected_dtype

    def test_fft_3d_basic(self):
        """Test 3D FFT with basic functionality."""
        # Setup
        batch_size, channels, d, h, w = 2, 2, 8, 12, 16
        x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, channels, d, h, w))

        # Execute
        x_ft = standardized_fft(x, spatial_dims=3)

        # Verify
        expected_shape = (batch_size, channels, d, h, w // 2 + 1)
        assert x_ft.shape == expected_shape

        # Check dtype based on JAX precision setting
        expected_dtype = (
            jnp.complex128 if jax.config.read("jax_enable_x64") else jnp.complex64
        )
        assert x_ft.dtype == expected_dtype

    def test_fft_jax_transformations(self):
        """Test that FFT operations work with JAX transformations."""
        # Setup
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16))

        # Test JIT compilation
        fft_jit = jax.jit(lambda x: standardized_fft(x, spatial_dims=1))
        x_ft_jit = fft_jit(x)
        x_ft_direct = standardized_fft(x, spatial_dims=1)

        np.testing.assert_allclose(x_ft_jit, x_ft_direct, rtol=1e-6)

        # Test vmap
        batch_x = jax.random.normal(jax.random.PRNGKey(42), (5, 2, 2, 16))
        fft_vmap = jax.vmap(lambda x: standardized_fft(x, spatial_dims=1))
        x_ft_vmap = fft_vmap(batch_x)

        assert x_ft_vmap.shape == (5, 2, 2, 9)


class TestStandardizedIFFT:
    """Test inverse FFT operations."""

    def test_ifft_1d_basic(self):
        """Test basic 1D inverse FFT."""
        # Setup
        x = jnp.array([[[1.0, 2.0, 3.0, 4.0]]])
        x_fft = standardized_fft(x, spatial_dims=1)

        # Execute
        x_reconstructed = standardized_ifft(x_fft, target_shape=x.shape, spatial_dims=1)

        # Verify
        np.testing.assert_allclose(x, x_reconstructed, rtol=1e-10)

    def test_ifft_2d_basic(self):
        """Test basic 2D inverse FFT."""
        # Setup
        x = jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        x_fft = standardized_fft(x, spatial_dims=2)

        # Execute
        x_reconstructed = standardized_ifft(x_fft, target_shape=x.shape, spatial_dims=2)

        # Verify
        np.testing.assert_allclose(x, x_reconstructed, rtol=1e-10)

    def test_ifft_3d_basic(self):
        """Test basic 3D inverse FFT."""
        # Setup
        x = jnp.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
        x_fft = standardized_fft(x, spatial_dims=3)

        # Execute
        x_reconstructed = standardized_ifft(x_fft, target_shape=x.shape, spatial_dims=3)

        # Verify
        np.testing.assert_allclose(x, x_reconstructed, rtol=1e-10)

    def test_ifft_with_truncation(self):
        """Test inverse FFT with truncation."""
        # Setup
        x = jnp.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
        x_fft = standardized_fft(x, spatial_dims=1)

        # Execute with truncation
        target_shape = (1, 1, 4)
        x_reconstructed = standardized_ifft(
            x_fft, target_shape=target_shape, spatial_dims=1
        )

        # Verify
        assert x_reconstructed.shape == target_shape

    def test_ifft_unsupported_spatial_dims(self):
        """Test error handling for unsupported spatial dimensions in IFFT."""
        # Setup
        x_fft = jnp.array([[[1.0, 2.0, 3.0, 4.0]]])
        target_shape = (1, 1, 4)

        # Test unsupported spatial_dims
        with pytest.raises(ValueError, match="Unsupported spatial_dims: 4"):
            standardized_ifft(x_fft, target_shape=target_shape, spatial_dims=4)


class TestSpectralDerivative:
    """Test spectral derivative computation."""

    def test_derivative_1d_sine(self):
        """Test derivative of sine function in 1D."""
        # Setup: sin(2πx) has derivative 2π*cos(2πx)
        x = jnp.linspace(0, 1, 64, endpoint=False)
        x = x[None, None, :]  # Add batch and channel dims
        y = jnp.sin(2 * jnp.pi * x)

        # Execute
        dy_dx = spectral_derivative(y, spatial_dims=1, dx=1.0 / 64)

        # Verify (analytic derivative)
        expected = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
        # Use both relative and absolute tolerance for small values near zero
        np.testing.assert_allclose(dy_dx, expected, rtol=1e-2, atol=1e-5)

    def test_derivative_2d_gaussian(self):
        """Test gradient of 2D Gaussian."""
        # Setup
        x = jnp.linspace(-2, 2, 32)
        y = jnp.linspace(-2, 2, 32)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        f = jnp.exp(-(X**2 + Y**2))[None, None, :, :]  # Add batch and channel dims

        # Execute
        df_dx, df_dy = spectral_derivative(f, spatial_dims=2, dx=4.0 / 32)

        # Verify dimensions
        assert df_dx.shape == f.shape
        assert df_dy.shape == f.shape

    def test_derivative_invalid_spatial_dims(self):
        """Test error handling for invalid spatial dimensions."""
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16))

        # Test spatial_dims < 1
        with pytest.raises(ValueError, match="Unsupported spatial dimensions: 0"):
            spectral_derivative(x, spatial_dims=0, dx=0.1)

        # Test spatial_dims > 3
        with pytest.raises(ValueError, match="Unsupported spatial dimensions: 4"):
            spectral_derivative(x, spatial_dims=4, dx=0.1)

    def test_derivative_mismatched_dx_length(self):
        """Test error handling for mismatched dx length."""
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16, 20))

        # dx length doesn't match spatial_dims
        with pytest.raises(
            ValueError, match="Grid spacing length 3 doesn't match spatial_dims 2"
        ):
            spectral_derivative(x, spatial_dims=2, dx=[0.1, 0.2, 0.3])

    def test_derivative_axis_out_of_range(self):
        """Test error handling for axis out of range."""
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16, 20))

        # axis >= spatial_dims
        with pytest.raises(
            ValueError, match="Axis 2 out of bounds for 2 spatial dimensions"
        ):
            spectral_derivative(x, spatial_dims=2, dx=0.1, axis=2)

    def test_derivative_single_axis_specification(self):
        """Test single axis derivative specification."""
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16, 20))

        # Test single axis as integer
        dx_single = spectral_derivative(x, spatial_dims=2, dx=0.1, axis=0)
        # Ensure we get a single array, not a tuple
        assert not isinstance(dx_single, tuple)
        assert dx_single.shape == x.shape

        # Test single axis as list - should return same single array since len(derivatives) == 1
        dx_list = spectral_derivative(x, spatial_dims=2, dx=0.1, axis=[0])
        # Ensure we get a single array, not a tuple
        assert not isinstance(dx_list, tuple)
        assert dx_list.shape == x.shape  # Should be single array, not tuple
        np.testing.assert_allclose(dx_single, dx_list, rtol=1e-10)

    def test_derivative_3d_multiple_axes(self):
        """Test 3D derivative with multiple axes."""
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 8, 12, 16))

        # Compute derivative along all axes
        derivs = spectral_derivative(x, spatial_dims=3, dx=[0.1, 0.2, 0.3])

        # Should return tuple of 3 derivatives
        assert isinstance(derivs, tuple)
        assert len(derivs) == 3
        for deriv in derivs:
            assert deriv.shape == x.shape


class TestSpectralFilter:
    """Test spectral filtering operations."""

    def test_lowpass_filter(self):
        """Test low-pass filtering in frequency domain."""
        # Setup: Signal with high-frequency noise
        x = jnp.linspace(0, 1, 128, endpoint=False)
        signal = jnp.sin(2 * jnp.pi * x) + 0.1 * jnp.sin(20 * jnp.pi * x)
        signal = signal[None, None, :]  # Add batch and channel dims

        # Execute
        filtered_signal = spectral_filter(
            signal, cutoff=0.1, spatial_dims=1, filter_type="lowpass"
        )

        # Verify
        assert filtered_signal.shape == signal.shape

        # Check dtype based on JAX precision setting
        expected_dtype = (
            jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
        )
        assert filtered_signal.dtype == expected_dtype

        # High-frequency component should be reduced
        fft_orig = jnp.abs(jnp.fft.fft(signal[0, 0, :]))
        fft_filtered = jnp.abs(jnp.fft.fft(filtered_signal[0, 0, :]))

        # Check that high frequencies are attenuated
        high_freq_orig = jnp.sum(fft_orig[20:])
        high_freq_filtered = jnp.sum(fft_filtered[20:])
        assert high_freq_filtered < high_freq_orig

    def test_highpass_filter(self):
        """Test high-pass filtering."""
        # Setup
        x = jnp.linspace(0, 1, 128, endpoint=False)
        signal = (
            jnp.sin(2 * jnp.pi * x)
            + jnp.sin(10 * jnp.pi * x)
            + jnp.sin(30 * jnp.pi * x)
        )
        signal = signal[None, None, :]

        # Execute: High-pass filter (remove low frequencies)
        filtered_signal = spectral_filter(
            signal,
            cutoff=0.1,  # Remove frequencies below 0.1 (normalized)
            spatial_dims=1,
            filter_type="highpass",
        )

        # Verify
        assert filtered_signal.shape == signal.shape

    def test_filter_3d(self):
        """Test 3D spectral filtering."""
        # Setup 3D signal
        d, h, w = 8, 12, 16
        signal = jax.random.normal(jax.random.PRNGKey(42), (2, 2, d, h, w))

        # Test lowpass filter
        filtered = spectral_filter(
            signal, cutoff=0.3, spatial_dims=3, filter_type="lowpass"
        )
        assert filtered.shape == signal.shape

        # Check dtype based on JAX precision setting
        expected_dtype = (
            jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
        )
        assert filtered.dtype == expected_dtype

        # Test highpass filter
        filtered_hp = spectral_filter(
            signal, cutoff=0.3, spatial_dims=3, filter_type="highpass"
        )
        assert filtered_hp.shape == signal.shape

    def test_filter_unsupported_spatial_dims(self):
        """Test error handling for unsupported spatial dimensions."""
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16))

        with pytest.raises(ValueError, match="Unsupported spatial_dims: 4"):
            spectral_filter(x, cutoff=0.1, spatial_dims=4, filter_type="lowpass")

    def test_filter_unsupported_filter_type(self):
        """Test error handling for unsupported filter type."""
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16))

        with pytest.raises(ValueError, match="Unsupported filter_type: bandpass"):
            spectral_filter(x, cutoff=0.1, spatial_dims=1, filter_type="bandpass")


class TestFFTFrequencyGrid:
    """Test frequency grid generation utilities."""

    def test_frequency_grid_1d(self):
        """Test 1D frequency grid generation."""
        # Setup
        n = 32
        dx = 0.1

        # Execute
        freqs = fft_frequency_grid(shape=(n,), dx=dx)

        # Verify
        assert len(freqs) == 1
        assert freqs[0].shape == (n // 2 + 1,)

        # Check frequency values
        expected_freqs = jnp.fft.rfftfreq(n, dx)
        np.testing.assert_allclose(freqs[0], expected_freqs)

    def test_frequency_grid_2d(self):
        """Test 2D frequency grid generation."""
        # Setup
        h, w = 16, 20
        dx = 0.1

        # Execute
        freqs = fft_frequency_grid(shape=(h, w), dx=dx)

        # Verify
        assert len(freqs) == 2
        assert freqs[0].shape == (h,)
        assert freqs[1].shape == (w // 2 + 1,)

    def test_frequency_grid_wavenumber(self):
        """Test wavenumber grid generation."""
        # Setup
        n = 32
        dx = 0.1

        # Execute
        wavenumbers = fft_frequency_grid(shape=(n,), dx=dx, wavenumber=True)

        # Verify
        expected_k = 2 * jnp.pi * jnp.fft.rfftfreq(n, dx)
        np.testing.assert_allclose(wavenumbers[0], expected_k)

    def test_frequency_grid_mismatched_dx(self):
        """Test error handling for mismatched dx length."""
        with pytest.raises(
            ValueError, match="Grid spacing length 3 doesn't match spatial_dims 2"
        ):
            fft_frequency_grid(shape=(16, 20), dx=[0.1, 0.2, 0.3])


class TestGetSpectralFrequencies:
    """Test get_spectral_frequencies function."""

    def test_get_frequencies_1d(self):
        """Test 1D spectral frequency generation."""
        shape = (32,)
        sample_rate = 2.0

        freqs = get_spectral_frequencies(shape, spatial_dims=1, sample_rate=sample_rate)

        assert isinstance(freqs, tuple)
        assert len(freqs) == 1
        assert freqs[0].shape == (shape[0] // 2 + 1,)

        # Verify against expected
        expected = jnp.fft.rfftfreq(shape[0], d=1.0 / sample_rate)
        np.testing.assert_allclose(freqs[0], expected)

    def test_get_frequencies_2d(self):
        """Test 2D spectral frequency generation."""
        shape = (16, 20)
        sample_rate = 1.5

        freqs = get_spectral_frequencies(shape, spatial_dims=2, sample_rate=sample_rate)

        assert isinstance(freqs, tuple)
        assert len(freqs) == 2
        assert freqs[0].shape == (shape[0],)
        assert freqs[1].shape == (shape[1] // 2 + 1,)

    def test_get_frequencies_3d(self):
        """Test 3D spectral frequency generation."""
        shape = (8, 12, 16)
        sample_rate = 3.0

        freqs = get_spectral_frequencies(shape, spatial_dims=3, sample_rate=sample_rate)

        assert isinstance(freqs, tuple)
        assert len(freqs) == 3
        assert freqs[0].shape == (shape[0],)
        assert freqs[1].shape == (shape[1],)
        assert freqs[2].shape == (shape[2] // 2 + 1,)

    def test_get_frequencies_unsupported_dims(self):
        """Test error handling for unsupported spatial dimensions."""
        shape = (16,)

        with pytest.raises(ValueError, match="Unsupported spatial_dims: 4"):
            get_spectral_frequencies(shape, spatial_dims=4)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_spatial_dims(self):
        """Test error handling for invalid spatial dimensions."""
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16))

        with pytest.raises(ValueError, match="Unsupported spatial_dims"):
            standardized_fft(x, spatial_dims=4)

    def test_empty_input(self):
        """Test handling of empty input."""
        # Test with zero-length spatial dimension
        x = jnp.zeros((1, 1, 0))

        # JAX throws a specific MLIR verification error for empty tensors
        with pytest.raises(ValueError, match="Cannot lower jaxpr"):
            standardized_fft(x, spatial_dims=1)

    def test_single_point_input(self):
        """Test handling of single spatial point."""
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 1))

        x_ft = standardized_fft(x, spatial_dims=1)
        assert x_ft.shape == (2, 2, 1)

    def test_mismatched_ifft_shape(self):
        """Test IFFT behavior with different target shapes."""
        # Note: The original tensor_ops implementation allows flexible shape reconstruction
        # This test verifies that reconstruction works rather than fails
        x = jax.random.normal(jax.random.PRNGKey(42), (2, 2, 16))
        x_ft = standardized_fft(x, spatial_dims=1)

        # Test reconstruction to larger size (should work per original implementation)
        x_reconstructed = standardized_ifft(
            x_ft, target_shape=(2, 2, 32), spatial_dims=1
        )
        assert x_reconstructed.shape == (2, 2, 32)

        # Check dtype based on JAX precision setting
        expected_dtype = (
            jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
        )
        assert x_reconstructed.dtype == expected_dtype


class TestJITCompatibility:
    """Test JIT compilation compatibility for spectral operations."""

    def test_standardized_fft_jit_compilation(self):
        """Test that standardized FFT functions can be JIT compiled."""
        import time

        # Test data
        x_1d = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 32))
        x_2d = jax.random.normal(jax.random.PRNGKey(1), (4, 3, 16, 20))
        x_3d = jax.random.normal(jax.random.PRNGKey(2), (4, 3, 8, 12, 16))

        # JIT compile FFT functions
        jitted_fft_1d = jax.jit(lambda x: standardized_fft(x, 1))
        jitted_fft_2d = jax.jit(lambda x: standardized_fft(x, 2))
        jitted_fft_3d = jax.jit(lambda x: standardized_fft(x, 3))

        # Test JIT compilation works
        fft_1d = jitted_fft_1d(x_1d)
        fft_2d = jitted_fft_2d(x_2d)
        fft_3d = jitted_fft_3d(x_3d)

        assert fft_1d.shape == (4, 3, 17)  # rfft output
        assert fft_2d.shape == (4, 3, 16, 11)  # rfft2 output
        assert fft_3d.shape == (4, 3, 8, 12, 9)  # rfftn output

        # Test performance improvement with JIT
        # Warmup
        _ = jitted_fft_2d(x_2d)

        # Time JIT version
        start_time = time.time()
        for _ in range(10):
            _ = jitted_fft_2d(x_2d)
        jit_time = time.time() - start_time

        # Time non-JIT version
        start_time = time.time()
        for _ in range(10):
            _ = standardized_fft(x_2d, 2)
        non_jit_time = time.time() - start_time

        # JIT should be faster or at least not significantly slower
        assert jit_time <= non_jit_time * 2.0

    def test_standardized_ifft_jit_compilation(self):
        """Test that standardized IFFT functions can be JIT compiled."""
        # Test data
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 16, 20))
        x_fft = standardized_fft(x, 2)

        # JIT compile IFFT function with static shape
        def ifft_static(x_ft):
            return standardized_ifft(x_ft, (4, 3, 16, 20), 2)

        jitted_ifft = jax.jit(ifft_static)

        # Test JIT compilation works
        x_reconstructed = jitted_ifft(x_fft)

        assert x_reconstructed.shape == x.shape
        assert jnp.allclose(x_reconstructed, x, atol=1e-6)

    def test_spectral_derivative_jit_compilation(self):
        """Test that spectral derivative can be JIT compiled."""
        # Test data
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 32))

        # JIT compile derivative function with static dx
        def derivative_1d(x):
            return spectral_derivative(x, 1, 0.1)

        jitted_derivative = jax.jit(derivative_1d)

        # Test JIT compilation works
        dx_dt = jitted_derivative(x)

        assert dx_dt.shape == x.shape
        assert jnp.isfinite(dx_dt).all()

        # Test multi-axis derivative with JIT
        x_2d = jax.random.normal(jax.random.PRNGKey(1), (4, 3, 16, 20))

        def derivative_2d(x):
            return spectral_derivative(x, 2, [0.1, 0.2], axis=None)

        jitted_derivative_2d = jax.jit(derivative_2d)

        derivatives = jitted_derivative_2d(x_2d)
        assert len(derivatives) == 2
        assert all(d.shape == x_2d.shape for d in derivatives)

    def test_spectral_filter_jit_compilation(self):
        """Test that spectral filter can be JIT compiled."""
        # Test data
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 32))
        cutoff = 0.3

        # JIT compile filter functions - filter_type must be static
        def lowpass_filter(x, cutoff):
            return spectral_filter(x, cutoff, 1, "lowpass")

        def highpass_filter(x, cutoff):
            return spectral_filter(x, cutoff, 1, "highpass")

        jitted_lowpass = jax.jit(lowpass_filter)
        jitted_highpass = jax.jit(highpass_filter)

        # Test JIT compilation works
        x_low = jitted_lowpass(x, cutoff)
        x_high = jitted_highpass(x, cutoff)

        assert x_low.shape == x.shape
        assert x_high.shape == x.shape
        assert jnp.isfinite(x_low).all()
        assert jnp.isfinite(x_high).all()

    def test_fft_frequency_grid_jit_compilation(self):
        """Test that FFT frequency grid generation can be JIT compiled using static arguments."""
        # Test data - use JAX array for JIT compatibility
        _ = jnp.array([0.1, 0.2])

        # JIT compile frequency grid function with static shape and dx
        # Use static_argnames to make both shape and dx static for JIT
        def freq_grid_static(shape, dx):
            return fft_frequency_grid(shape, dx)

        def wavenumber_grid_static(shape, dx):
            return fft_frequency_grid(shape, dx, wavenumber=True)

        jitted_freq_grid = jax.jit(freq_grid_static, static_argnames=["shape", "dx"])
        jitted_wavenumber_grid = jax.jit(
            wavenumber_grid_static, static_argnames=["shape", "dx"]
        )

        # Test JIT compilation works with static arguments
        freqs = jitted_freq_grid(shape=(32, 64), dx=(0.1, 0.2))
        wavenumbers = jitted_wavenumber_grid(shape=(32, 64), dx=(0.1, 0.2))

        assert len(freqs) == 2
        assert freqs[0].shape == (32,)
        assert freqs[1].shape == (33,)  # rfft frequency count

        assert len(wavenumbers) == 2
        assert jnp.all(
            jnp.abs(wavenumbers[0]) >= jnp.abs(freqs[0]) * 2 * jnp.pi - 1e-10
        )

    def test_get_spectral_frequencies_jit_compilation(self):
        """Test that spectral frequency generation can be JIT compiled."""
        # Test data - use static shape for JIT compatibility
        sample_rate = 2.0

        # JIT compile frequency function with static shape
        def get_freqs_static(sample_rate):
            return get_spectral_frequencies((32, 64, 48), 3, sample_rate)

        jitted_get_freqs = jax.jit(get_freqs_static)

        # Test JIT compilation works
        freqs = jitted_get_freqs(sample_rate)

        assert len(freqs) == 3
        assert freqs[0].shape == (32,)
        assert freqs[1].shape == (64,)
        assert freqs[2].shape == (25,)  # rfft frequency count for last dimension

    def test_batch_spectral_operations_jit_compatibility(self):
        """Test JIT compilation with batch processing of spectral operations."""

        # Create batch processing function
        def batch_spectral_pipeline(x_batch):
            """Complete spectral processing pipeline."""
            # FFT
            x_fft = jax.vmap(lambda x: standardized_fft(x, 2))(x_batch)

            # Derivative
            def derivative_2d_batch(x):
                return spectral_derivative(x, 2, [0.1, 0.1])

            derivatives = jax.vmap(derivative_2d_batch)(x_batch)

            # Filter
            def lowpass_filter_2d(x):
                return spectral_filter(x, 0.3, 2, "lowpass")

            filtered = jax.vmap(lowpass_filter_2d)(x_batch)

            return x_fft, derivatives, filtered

        # JIT compile batch processing
        jitted_batch_pipeline = jax.jit(batch_spectral_pipeline)

        # Test batch processing
        x_batch = jax.random.normal(jax.random.PRNGKey(0), (8, 3, 16, 20))
        fft_batch, deriv_batch, filt_batch = jitted_batch_pipeline(x_batch)

        assert fft_batch.shape == (8, 3, 16, 11)
        assert len(deriv_batch) == 2
        assert deriv_batch[0].shape == (8, 3, 16, 20)
        assert filt_batch.shape == (8, 3, 16, 20)

    def test_end_to_end_spectral_jit_workflow(self):
        """Test complete JIT-compiled spectral analysis workflow."""

        def spectral_analysis_workflow(x, dx):
            """Complete spectral analysis workflow."""
            # Forward FFT
            x_fft = standardized_fft(x, 2)

            # Compute derivatives
            dx_dx, dx_dy = spectral_derivative(x, 2, [0.1, 0.15], axis=None)

            # Apply filtering
            x_filtered = spectral_filter(x, 0.4, 2, "lowpass")

            # Reconstruct
            x_reconstructed = standardized_ifft(x_fft, x.shape, 2)

            return {
                "fft": x_fft,
                "derivatives": (dx_dx, dx_dy),
                "filtered": x_filtered,
                "reconstructed": x_reconstructed,
            }

        # JIT compile complete workflow
        jitted_workflow = jax.jit(spectral_analysis_workflow)

        # Test complete workflow
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 32, 48))
        dx = [0.1, 0.15]

        results = jitted_workflow(x, dx)

        assert "fft" in results
        assert "derivatives" in results
        assert "filtered" in results
        assert "reconstructed" in results

        assert results["fft"].shape == (4, 3, 32, 25)
        assert len(results["derivatives"]) == 2
        assert results["filtered"].shape == x.shape
        assert results["reconstructed"].shape == x.shape
        assert jnp.allclose(results["reconstructed"], x, atol=1e-6)
