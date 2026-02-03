"""JIT compatibility tests for spectral utilities.

This module tests JIT compilation compatibility for spectral utility functions
to ensure optimal performance in scientific computing applications.
"""

import time

import jax
import jax.numpy as jnp

from opifex.core.spectral.spectral_utils import (
    energy_spectrum,
    power_spectral_density,
    spectral_energy,
    wavenumber_grid,
)


class TestJITCompatibility:
    """Test JIT compilation compatibility for spectral utilities."""

    def test_power_spectral_density_jit_compilation(self):
        """Test that power spectral density can be JIT compiled."""
        # Test data
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 32, 48))

        # JIT compile PSD function with static dx
        def psd_static(x):
            return power_spectral_density(x, 2, [0.1, 0.15])

        jitted_psd = jax.jit(psd_static)

        # Test JIT compilation works
        psd = jitted_psd(x)

        assert psd.shape == (32, 25)  # Reduced over batch/channel dims
        assert jnp.all(psd >= 0)
        assert jnp.isfinite(psd).all()

        # Test performance improvement
        # Warmup
        _ = jitted_psd(x)

        # Time JIT version
        start_time = time.time()
        for _ in range(5):
            _ = jitted_psd(x)
        jit_time = time.time() - start_time

        # Time non-JIT version
        start_time = time.time()
        for _ in range(5):
            _ = power_spectral_density(x, 2, [0.1, 0.15])
        non_jit_time = time.time() - start_time

        # JIT should be faster or at least not significantly slower
        assert jit_time <= non_jit_time * 2.0

    def test_energy_spectrum_jit_compilation(self):
        """Test that energy spectrum can be JIT compiled."""
        # Test data
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 16, 20))

        # JIT compile energy spectrum function with static dx
        def energy_static(x, spatial_dims, dx):
            return energy_spectrum(x, spatial_dims, dx)

        jitted_energy = jax.jit(energy_static, static_argnames=["spatial_dims", "dx"])

        # Test JIT compilation works with static arguments
        energy = jitted_energy(x, spatial_dims=2, dx=(0.1, 0.2))

        assert energy.shape == (4, 3, 16, 11)  # Same as FFT output
        assert jnp.all(energy >= 0)
        assert jnp.isfinite(energy).all()

    def test_spectral_energy_jit_compilation(self):
        """Test that total spectral energy can be JIT compiled."""
        # Test data
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 32))

        # JIT compile spectral energy function with static dx
        def total_energy_static(x):
            return spectral_energy(x, 1, 0.1)

        jitted_total_energy = jax.jit(total_energy_static)

        # Test JIT compilation works
        total_energy = jitted_total_energy(x)

        assert total_energy.shape == (4, 3)  # Reduced over spatial dims
        assert jnp.all(total_energy >= 0)
        assert jnp.isfinite(total_energy).all()

    def test_wavenumber_grid_jit_compilation(self):
        """Test that wavenumber grid generation can be JIT compiled."""

        # JIT compile wavenumber grid functions with static shapes and dx
        def k_arrays_static(shape, dx):
            return wavenumber_grid(shape, dx, magnitude=False)

        def k_magnitude_static(shape, dx):
            return wavenumber_grid(shape, dx, magnitude=True)

        jitted_k_arrays = jax.jit(k_arrays_static, static_argnames=["shape", "dx"])
        jitted_k_magnitude = jax.jit(
            k_magnitude_static, static_argnames=["shape", "dx"]
        )

        # Test JIT compilation works with static arguments
        k_arrays = jitted_k_arrays(shape=(32, 64), dx=(0.1, 0.2))
        k_mag = jitted_k_magnitude(shape=(32, 64), dx=(0.1, 0.2))

        assert len(k_arrays) == 2
        assert k_arrays[0].shape == (32,)
        assert k_arrays[1].shape == (33,)  # rfft frequency count

        assert k_mag.shape == (32, 33)
        assert jnp.all(k_mag >= 0)

    def test_batch_spectral_utils_jit_compatibility(self):
        """Test JIT compilation with batch processing of spectral utilities."""

        # Create batch processing function
        def batch_spectral_analysis(x_batch):
            """Batch spectral analysis pipeline."""

            # Power spectral density for each sample
            def psd_single(x):
                return power_spectral_density(x, 2, [0.1, 0.15])

            psd_batch = jax.vmap(psd_single)(x_batch)

            # Energy spectrum for each sample
            def energy_single(x):
                return energy_spectrum(x, 2, [0.1, 0.15])

            energy_batch = jax.vmap(energy_single)(x_batch)

            # Total energy for each sample
            def total_energy_single(x):
                return spectral_energy(x, 2, [0.1, 0.15])

            total_energy_batch = jax.vmap(total_energy_single)(x_batch)

            return psd_batch, energy_batch, total_energy_batch

        # JIT compile batch processing
        jitted_batch_analysis = jax.jit(batch_spectral_analysis)

        # Test batch processing
        x_batch = jax.random.normal(jax.random.PRNGKey(0), (6, 3, 16, 20))

        psd_batch, energy_batch, total_energy_batch = jitted_batch_analysis(x_batch)

        assert psd_batch.shape == (6, 16, 11)  # Reduced over batch/channel
        assert energy_batch.shape == (6, 3, 16, 11)
        assert total_energy_batch.shape == (6, 3)

        assert jnp.all(psd_batch >= 0)
        assert jnp.all(energy_batch >= 0)
        assert jnp.all(total_energy_batch >= 0)

    def test_end_to_end_spectral_utils_jit_workflow(self):
        """Test complete JIT-compiled spectral utilities workflow."""

        def spectral_utils_workflow(x):
            """Complete spectral utilities analysis workflow."""
            # Compute power spectral density
            psd = power_spectral_density(x, 2, [0.1, 0.15])

            # Compute energy spectrum
            energy_spec = energy_spectrum(x, 2, [0.1, 0.15])

            # Compute total spectral energy
            total_energy = spectral_energy(x, 2, [0.1, 0.15])

            # Generate wavenumber grids
            k_arrays = wavenumber_grid((32, 48), [0.1, 0.15], magnitude=False)
            k_magnitude = wavenumber_grid((32, 48), [0.1, 0.15], magnitude=True)

            return {
                "psd": psd,
                "energy_spectrum": energy_spec,
                "total_energy": total_energy,
                "k_arrays": k_arrays,
                "k_magnitude": k_magnitude,
            }

        # JIT compile complete workflow
        jitted_utils_workflow = jax.jit(spectral_utils_workflow)

        # Test complete workflow
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 32, 48))

        results = jitted_utils_workflow(x)

        assert "psd" in results
        assert "energy_spectrum" in results
        assert "total_energy" in results
        assert "k_arrays" in results
        assert "k_magnitude" in results

        assert results["psd"].shape == (32, 25)
        assert results["energy_spectrum"].shape == (4, 3, 32, 25)
        assert results["total_energy"].shape == (4, 3)
        assert len(results["k_arrays"]) == 2
        assert results["k_magnitude"].shape == (32, 25)
