"""Common spectral analysis utilities.

This module provides spectral analysis tools and utilities that are commonly
used across the Opifex framework for frequency domain analysis, energy
calculations, and wavenumber operations.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from opifex.core.spectral.fft_operations import (
    ensure_static_grid_spacing,
    standardized_fft,
)
from opifex.typing import Axis, GridSpacing


def power_spectral_density(
    x: jax.Array,
    spatial_dims: int,
    dx: GridSpacing,
    axis: Axis | None = None,
) -> jax.Array:
    """
    Compute power spectral density of input signal.

    Args:
        x: Input tensor (batch, channels, *spatial)
        spatial_dims: Number of spatial dimensions
        dx: Grid spacing (scalar or sequence for each dimension)
        axis: Axis for PSD computation (None for all spatial dimensions)

    Returns:
        Power spectral density tensor

    Raises:
        ValueError: If spatial dimensions are invalid
    """
    if spatial_dims < 1 or spatial_dims > 3:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    # Ensure dx is properly validated and normalized for the given shape
    dx_list = ensure_static_grid_spacing(dx, spatial_dims)

    # Compute FFT
    x_ft = standardized_fft(x, spatial_dims)

    # Compute power spectral density
    psd = jnp.abs(x_ft) ** 2

    # Normalize by grid spacing and size
    for i, dx_i in enumerate(dx_list):
        spatial_size = x.shape[-(spatial_dims - i)]
        psd = psd / (dx_i * spatial_size)

    # Average over batch and channel dimensions if present
    if axis is None:
        # Average over all non-spatial dimensions
        reduce_axes = tuple(range(len(x.shape) - spatial_dims))
        if reduce_axes:
            psd = jnp.mean(psd, axis=reduce_axes)

    return psd


def energy_spectrum(
    x: jax.Array,
    spatial_dims: int,
    dx: GridSpacing,
    axis: Axis | None = None,
) -> jax.Array:
    """
    Compute energy spectrum of input signal.

    Args:
        x: Input tensor (batch, channels, *spatial)
        spatial_dims: Number of spatial dimensions
        dx: Grid spacing (scalar or sequence for each dimension)

    Returns:
        Energy spectrum tensor

    Raises:
        ValueError: If spatial dimensions are invalid
    """
    if spatial_dims < 1 or spatial_dims > 3:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    # Ensure dx is properly validated and normalized for the given shape
    dx_list = ensure_static_grid_spacing(dx, spatial_dims)

    # Compute FFT
    x_ft = standardized_fft(x, spatial_dims)

    # Compute energy spectrum (squared magnitude)
    energy = jnp.abs(x_ft) ** 2

    # Scale by grid spacing
    for dx_i in dx_list:
        energy = energy * dx_i

    return energy


def spectral_energy(
    x: jax.Array,
    spatial_dims: int,
    dx: GridSpacing,
    axis: Axis | None = None,
) -> jax.Array:
    """
    Compute total spectral energy of input signal.

    Args:
        x: Input tensor (batch, channels, *spatial)
        spatial_dims: Number of spatial dimensions
        dx: Grid spacing (scalar or sequence for each dimension)

    Returns:
        Total spectral energy (scalar per batch/channel)

    Raises:
        ValueError: If spatial dimensions are invalid
    """
    if spatial_dims < 1 or spatial_dims > 3:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    # Ensure dx is properly validated and normalized for the given shape
    _ = ensure_static_grid_spacing(dx, spatial_dims)

    # Compute energy spectrum
    energy_spec = energy_spectrum(x, spatial_dims, dx)

    # Sum over all spatial (frequency) dimensions
    spatial_axes = tuple(range(-spatial_dims, 0))
    return jnp.sum(energy_spec, axis=spatial_axes)


def wavenumber_grid(
    shape: Sequence[int],
    dx: GridSpacing,
    axis: Axis | None = None,
    magnitude: bool = False,
) -> list[jax.Array] | jax.Array:
    """
    Generate wavenumber grids for spectral analysis.

    Args:
        shape: Shape of spatial dimensions
        dx: Grid spacing (scalar or sequence for each dimension)
        magnitude: If True, return magnitude of wavenumber vector

    Returns:
        List of wavenumber arrays for each dimension, or magnitude array

    Raises:
        ValueError: If shape and dx dimensions don't match
    """
    spatial_dims = len(shape)

    if spatial_dims < 1 or spatial_dims > 3:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    # Ensure dx is properly validated and normalized for the given shape
    dx_list = ensure_static_grid_spacing(dx, spatial_dims)

    if len(dx_list) != spatial_dims:
        raise ValueError(
            f"dx length {len(dx_list)} doesn't match shape dimensions {spatial_dims}"
        )

    # Generate wavenumber arrays for each dimension
    k_arrays = []
    for i, (size, dx_i) in enumerate(zip(shape, dx_list, strict=False)):
        if i == spatial_dims - 1:  # Last dimension uses rfftfreq
            k = 2 * jnp.pi * jnp.fft.rfftfreq(size, dx_i)
        else:
            k = 2 * jnp.pi * jnp.fft.fftfreq(size, dx_i)
        k_arrays.append(k)

    if not magnitude:
        return k_arrays

    # Compute magnitude of wavenumber vector
    if spatial_dims == 1:
        k_mag = jnp.abs(k_arrays[0])
    elif spatial_dims == 2:
        k0, k1 = jnp.meshgrid(k_arrays[0], k_arrays[1], indexing="ij")
        k_mag = jnp.sqrt(k0**2 + k1**2)
    elif spatial_dims == 3:
        k0, k1, k2 = jnp.meshgrid(k_arrays[0], k_arrays[1], k_arrays[2], indexing="ij")
        k_mag = jnp.sqrt(k0**2 + k1**2 + k2**2)

    return k_mag
