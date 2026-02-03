"""
Spectral Analysis Utilities

JAX-native spectral analysis tools for frequency domain analysis and
modal decomposition.
Optimized for neural operator preprocessing and data analysis.
"""

import jax
import jax.numpy as jnp


def power_spectral_density(field: jax.Array, normalize: bool = True) -> jax.Array:
    """
    Compute power spectral density using JAX FFT.

    Args:
        field: Input field (2D array)
        normalize: Whether to normalize the PSD

    Returns:
        Power spectral density array
    """
    # Compute 2D FFT
    field_fft = jnp.fft.fft2(field)

    # Compute power spectral density
    psd = jnp.abs(field_fft) ** 2

    if normalize:
        psd = psd / jnp.sum(psd)

    return psd


def modal_analysis(
    field: jax.Array, modes: int, return_coefficients: bool = False
) -> tuple[jax.Array, jax.Array]:
    """
    Extract dominant spatial modes using FFT-based modal decomposition.

    Args:
        field: Input field (2D array)
        modes: Number of dominant modes to extract
        return_coefficients: Whether to return modal coefficients

    Returns:
        Tuple of (reconstructed_field, modal_coefficients) if return_coefficients=True
        Otherwise returns (reconstructed_field, mode_energies)
    """
    # Compute 2D FFT
    field_fft = jnp.fft.fft2(field)

    # Compute mode energies
    energies = jnp.abs(field_fft) ** 2

    # Find indices of top modes - use a more stable approach
    flat_energies = energies.ravel()
    # Clamp modes to avoid out of bounds
    modes = min(modes, flat_energies.size)
    top_indices = jnp.argsort(flat_energies)[-modes:]

    # Create mask using a simpler approach
    mask = jnp.zeros_like(flat_energies, dtype=bool)
    mask = mask.at[top_indices].set(True)
    mask = mask.reshape(energies.shape)

    # Filter field to keep only top modes
    filtered_fft = jnp.where(mask, field_fft, 0.0)

    # Reconstruct field - use more stable approach
    try:
        reconstructed = jnp.real(jnp.fft.ifft2(filtered_fft))
    except Exception:
        # Fallback: return zero field if reconstruction fails
        reconstructed = jnp.zeros_like(field)

    if return_coefficients:
        # Return modal coefficients - extract safely
        coefficients = jnp.where(mask.ravel(), field_fft.ravel(), 0.0)
        coefficients = coefficients[mask.ravel()]  # Only non-zero elements
        # Pad to expected size if needed
        if coefficients.size < modes:
            padding = jnp.zeros(modes - coefficients.size)
            coefficients = jnp.concatenate([coefficients, padding])
        elif coefficients.size > modes:
            coefficients = coefficients[:modes]
        return reconstructed, coefficients

    # Return mode energies
    mode_energies = jnp.where(mask.ravel(), energies.ravel(), 0.0)
    mode_energies = mode_energies[mask.ravel()]  # Only non-zero elements
    # Pad to expected size if needed
    if mode_energies.size < modes:
        padding = jnp.zeros(modes - mode_energies.size)
        mode_energies = jnp.concatenate([mode_energies, padding])
    elif mode_energies.size > modes:
        mode_energies = mode_energies[:modes]
    return reconstructed, mode_energies


def frequency_analysis(
    field: jax.Array, dx: float = 1.0
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """
    Analyze frequency content of a field.

    Args:
        field: Input field (2D array)
        dx: Grid spacing

    Returns:
        Tuple of ((freqs_x, freqs_y), spectrum)
    """
    nx, ny = field.shape

    # Compute frequency arrays
    freqs_x = jnp.fft.fftfreq(nx, dx)
    freqs_y = jnp.fft.fftfreq(ny, dx)

    # Compute 2D FFT
    field_fft = jnp.fft.fft2(field)
    spectrum = jnp.abs(field_fft)

    return (freqs_x, freqs_y), spectrum


def bandpass_filter(
    field: jax.Array, low_freq: float, high_freq: float, dx: float = 1.0
) -> jax.Array:
    """
    Apply bandpass filter to field in frequency domain.

    Args:
        field: Input field (2D array)
        low_freq: Lower frequency cutoff
        high_freq: Higher frequency cutoff
        dx: Grid spacing

    Returns:
        Filtered field
    """
    nx, ny = field.shape

    # Create frequency grids
    freqs_x = jnp.fft.fftfreq(nx, dx)
    freqs_y = jnp.fft.fftfreq(ny, dx)
    FX, FY = jnp.meshgrid(freqs_x, freqs_y, indexing="ij")

    # Compute magnitude of frequency
    freq_mag = jnp.sqrt(FX**2 + FY**2)

    # Create bandpass mask
    mask = (freq_mag >= low_freq) & (freq_mag <= high_freq)

    # Apply filter in frequency domain
    field_fft = jnp.fft.fft2(field)
    filtered_fft = jnp.where(mask, field_fft, 0.0)

    # Transform back to spatial domain
    return jnp.real(jnp.fft.ifft2(filtered_fft))


def compute_energy_spectrum(
    field: jax.Array, dx: float = 1.0
) -> tuple[jax.Array, jax.Array]:
    """
    Compute 1D energy spectrum from 2D field.

    Args:
        field: Input field (2D array)
        dx: Grid spacing

    Returns:
        Tuple of (wavenumbers, energy_spectrum)
    """
    nx, ny = field.shape

    # Compute 2D FFT
    field_fft = jnp.fft.fft2(field)

    # Compute energy density
    energy_density = jnp.abs(field_fft) ** 2

    # Create wavenumber grids
    kx = jnp.fft.fftfreq(nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, dx) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")

    # Compute radial wavenumber
    k_radial = jnp.sqrt(KX**2 + KY**2)

    # Define wavenumber bins
    k_max = jnp.max(k_radial)
    k_bins = jnp.linspace(0, k_max, min(nx, ny) // 2)

    # Bin the energy
    def bin_energy(k_center):
        dk = k_bins[1] - k_bins[0] if len(k_bins) > 1 else 1.0
        mask = jnp.abs(k_radial - k_center) < dk / 2
        return jnp.sum(jnp.where(mask, energy_density, 0.0))

    energy_spectrum = jax.vmap(bin_energy)(k_bins)

    return k_bins, energy_spectrum
