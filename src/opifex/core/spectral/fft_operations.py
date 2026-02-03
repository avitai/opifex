"""Core FFT operations with standardized API.

This module provides fundamental FFT utilities with consistent interfaces
across different spatial dimensions. All functions are JAX-compatible and
optimized for scientific computing with double-precision accuracy.

The framework uses float64/complex128 precision by default for scientific
computing accuracy when JAX X64 mode is enabled.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from opifex.typing import (
    Axis,
    ensure_optional_axes,
    ensure_static_grid_spacing,
    FrequencyGrid,
    GridSpacing,
    normalize_axis,
    SpectralTensor,
    validate_axis_bounds,
)


def standardized_fft(x: jax.Array, spatial_dims: int) -> jax.Array:
    """Standardized FFT operation with scientific computing precision.

    Args:
        x: Input tensor (batch, channels, *spatial)
        spatial_dims: Number of spatial dimensions

    Returns:
        FFT transformed tensor

    Raises:
        ValueError: If spatial_dims is not 1, 2, or 3
    """
    # Check if input is complex
    is_complex = jnp.iscomplexobj(x)

    if spatial_dims == 1:
        if is_complex:
            return jnp.fft.fft(x, axis=-1)
        return jnp.fft.rfft(x, axis=-1)
    if spatial_dims == 2:
        if is_complex:
            return jnp.fft.fft2(x, axes=(-2, -1))
        return jnp.fft.rfft2(x, axes=(-2, -1))
    if spatial_dims == 3:
        if is_complex:
            return jnp.fft.fftn(x, axes=(-3, -2, -1))
        return jnp.fft.rfftn(x, axes=(-3, -2, -1))
    raise ValueError(f"Unsupported spatial_dims: {spatial_dims}")


def standardized_ifft(
    x: jax.Array, target_shape: tuple, spatial_dims: int
) -> jax.Array:
    """Standardized IFFT operation with scientific computing precision.

    Args:
        x: Input tensor (batch, channels, *spatial)
        target_shape: Target shape for the output
        spatial_dims: Number of spatial dimensions

    Returns:
        IFFT transformed tensor

    Raises:
        ValueError: If spatial_dims is not 1, 2, or 3
    """
    if spatial_dims == 1:
        n = target_shape[-1]
        # Check if this could be rfft result - rfft output has shape (n//2 + 1)
        # where n is the original real signal length. For any reasonable n,
        # rfft(n) produces at most n//2 + 1 frequencies.
        # If x.shape[-1] <= n//2 + 1, it's likely from rfft
        freq_count = x.shape[-1]
        max_rfft_freqs = n // 2 + 1
        is_rfft_result = freq_count <= max_rfft_freqs

        if is_rfft_result:
            return jnp.fft.irfft(x, n=n, axis=-1)
        return jnp.fft.ifft(x, n=n, axis=-1)

    if spatial_dims == 2:
        s = target_shape[-2:]
        _, w = s
        # Similar logic for 2D - check if last dimension suggests rfft2 output
        freq_count = x.shape[-1]
        max_rfft_freqs = w // 2 + 1
        is_rfft_result = freq_count <= max_rfft_freqs

        if is_rfft_result:
            return jnp.fft.irfft2(x, s=s, axes=(-2, -1))
        return jnp.fft.ifft2(x, s=s, axes=(-2, -1))

    if spatial_dims == 3:
        s = target_shape[-3:]
        _, _, w = s
        # Similar logic for 3D - check if last dimension suggests irfftn output
        freq_count = x.shape[-1]
        max_rfft_freqs = w // 2 + 1
        is_rfft_result = freq_count <= max_rfft_freqs

        if is_rfft_result:
            return jnp.fft.irfftn(x, s=s, axes=(-3, -2, -1))
        return jnp.fft.ifftn(x, s=s, axes=(-3, -2, -1))

    raise ValueError(f"Unsupported spatial_dims: {spatial_dims}")


def spectral_derivative(
    x: SpectralTensor,
    spatial_dims: int,
    dx: GridSpacing,
    axis: Axis = None,
) -> SpectralTensor | tuple[SpectralTensor, ...]:
    """Compute spectral derivatives using FFT.

    Uses shape-aware typing for clear tensor contracts. Input tensor should have
    batch and channel dimensions followed by spatial dimensions.

    Args:
        x: Input spectral tensor with shape (batch, channels, *spatial)
        spatial_dims: Number of spatial dimensions
        dx: Grid spacing (scalar, sequence, or JAX array for each dimension)
        axis: Axis or axes for derivative (None for all spatial axes)

    Returns:
        Derivative tensor(s) with same shape as input. Single tensor for 1D
        derivatives, tuple of tensors for multi-dimensional derivatives.

    Raises:
        ValueError: If spatial_dims is not supported, dx dimensions don't match,
                   or axis indices are out of bounds
        TypeError: If axis or dx specification is invalid
    """
    if spatial_dims < 1 or spatial_dims > 3:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    # Ensure axis is statically known for JAX compilation
    axis = ensure_optional_axes(axis)

    # Ensure dx is statically known and properly validated for JAX compilation
    dx_list = ensure_static_grid_spacing(dx, spatial_dims)

    # Handle and validate axis specification
    if axis is None:
        derivative_axes = list(range(spatial_dims))
    elif isinstance(axis, int):
        # Validate single axis bounds
        if not validate_axis_bounds(axis, spatial_dims):
            raise ValueError(
                f"Axis {axis} out of bounds for {spatial_dims} spatial dimensions"
            )
        # Normalize negative indices - we know this returns an int
        normalized_axis = normalize_axis(axis, spatial_dims)
        assert isinstance(normalized_axis, int)  # Type narrowing  # nosec B101
        derivative_axes = [normalized_axis]
    else:
        # Handle sequence of axes
        if not validate_axis_bounds(axis, spatial_dims):
            raise ValueError(
                f"One or more axes in {axis} out of bounds for "
                f"{spatial_dims} spatial dimensions"
            )
        # Normalize all negative indices and convert to list
        normalized_axes = normalize_axis(axis, spatial_dims)
        assert isinstance(normalized_axes, tuple)  # Type narrowing  # nosec B101
        derivative_axes = list(normalized_axes)

    # Remember if input was real
    input_is_real = not jnp.iscomplexobj(x)

    # Compute FFT
    x_ft = standardized_fft(x, spatial_dims)

    derivatives = []

    for ax in derivative_axes:
        # Get frequency grid for this axis
        spatial_size = x.shape[-(spatial_dims - ax)]
        freqs = (
            jnp.fft.rfftfreq(spatial_size, dx_list[ax])
            if ax == spatial_dims - 1
            else jnp.fft.fftfreq(spatial_size, dx_list[ax])
        )

        # Create wavenumber array
        k = 2j * jnp.pi * freqs

        # Broadcast k to match x_ft shape
        k_shape = [1] * x_ft.ndim
        k_shape[-(spatial_dims - ax)] = k.shape[0]
        k = k.reshape(k_shape)

        # Apply derivative in frequency domain
        x_ft_deriv = x_ft * k

        # Transform back to spatial domain
        x_deriv = standardized_ifft(
            x_ft_deriv, target_shape=x.shape, spatial_dims=spatial_dims
        )

        # For real inputs, the derivative should also be real
        # (take real part to remove numerical noise)
        if input_is_real:
            x_deriv = jnp.real(x_deriv)

        derivatives.append(x_deriv)

    if len(derivatives) == 1:
        return derivatives[0]
    return tuple(derivatives)


def spectral_filter(
    x: jax.Array,
    cutoff: float,
    spatial_dims: int,
    filter_type: str = "lowpass",
) -> jax.Array:
    """
    Apply spectral filtering with scientific computing precision.

    Args:
        x: Input tensor to filter
        cutoff: Cutoff frequency (normalized, 0-1)
        spatial_dims: Number of spatial dimensions
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')

    Returns:
        Filtered tensor with same shape as input

    Raises:
        ValueError: If filter_type is not supported
    """
    # Remember if input was real
    input_is_real = not jnp.iscomplexobj(x)

    # Transform to frequency domain
    x_fft = standardized_fft(x, spatial_dims)

    # Create frequency coordinates
    if spatial_dims == 1:
        n = x.shape[-1]
        freqs = jnp.fft.rfftfreq(n)
        mask = freqs <= cutoff
    elif spatial_dims == 2:
        h, w = x.shape[-2:]
        freqs_h = jnp.fft.fftfreq(h)
        freqs_w = jnp.fft.rfftfreq(w)
        freqs_h_grid, freqs_w_grid = jnp.meshgrid(freqs_h, freqs_w, indexing="ij")
        freq_magnitude = jnp.sqrt(freqs_h_grid**2 + freqs_w_grid**2)
        mask = freq_magnitude <= cutoff
    elif spatial_dims == 3:
        d, h, w = x.shape[-3:]
        freqs_d = jnp.fft.fftfreq(d)
        freqs_h = jnp.fft.fftfreq(h)
        freqs_w = jnp.fft.rfftfreq(w)
        freqs_d_grid, freqs_h_grid, freqs_w_grid = jnp.meshgrid(
            freqs_d, freqs_h, freqs_w, indexing="ij"
        )
        freq_magnitude = jnp.sqrt(freqs_d_grid**2 + freqs_h_grid**2 + freqs_w_grid**2)
        mask = freq_magnitude <= cutoff
    else:
        raise ValueError(f"Unsupported spatial_dims: {spatial_dims}")

    # Apply filter
    if filter_type == "lowpass":
        x_fft_filtered = x_fft * mask
    elif filter_type == "highpass":
        x_fft_filtered = x_fft * (1 - mask)
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type}")

    # Transform back to spatial domain
    result = standardized_ifft(
        x_fft_filtered, target_shape=x.shape, spatial_dims=spatial_dims
    )

    # For real inputs, the result should also be real
    # (take real part to remove numerical noise)
    if input_is_real:
        result = jnp.real(result)

    return result


def fft_frequency_grid(
    shape: Sequence[int], dx: GridSpacing, wavenumber: bool = False
) -> list[FrequencyGrid]:
    """
    Generate frequency grid for FFT operations.

    Creates frequency grids for each spatial dimension with shape-aware typing.
    Returns wavenumber grids (2π*freq) when requested.

    Args:
        shape: Spatial shape for each dimension
        dx: Grid spacing (scalar, sequence, or JAX array for each dimension)
        wavenumber: If True, return wavenumbers (2π*freq) instead of frequencies

    Returns:
        List of frequency grid arrays, one for each spatial dimension

    Raises:
        ValueError: If dx dimensions don't match shape dimensions
        TypeError: If dx specification is invalid
    """
    # Ensure dx is properly validated and normalized for the given shape
    spatial_dims = len(shape)
    dx_list = ensure_static_grid_spacing(dx, spatial_dims)

    freqs = []
    for i, (n, spacing) in enumerate(zip(shape, dx_list, strict=False)):
        if i == len(shape) - 1:  # Last dimension uses rfftfreq
            freq = jnp.fft.rfftfreq(n, spacing)
        else:
            freq = jnp.fft.fftfreq(n, spacing)

        if wavenumber:
            freq = 2 * jnp.pi * freq

        freqs.append(freq)

    return freqs


def get_spectral_frequencies(
    shape: Sequence[int],
    spatial_dims: int,
    sample_rate: float = 1.0,
) -> tuple[jax.Array, ...]:
    """
    Get frequency coordinates for spectral operations.

    Args:
        shape: Shape of the spatial dimensions
        spatial_dims: Number of spatial dimensions
        sample_rate: Sample rate for frequency scaling

    Returns:
        Tuple of frequency arrays for each spatial dimension

    Raises:
        ValueError: If spatial_dims is not 1, 2, or 3
    """
    if spatial_dims == 1:
        return (jnp.fft.rfftfreq(shape[0], d=1.0 / sample_rate),)
    if spatial_dims == 2:
        return (
            jnp.fft.fftfreq(shape[0], d=1.0 / sample_rate),
            jnp.fft.rfftfreq(shape[1], d=1.0 / sample_rate),
        )
    if spatial_dims == 3:
        return (
            jnp.fft.fftfreq(shape[0], d=1.0 / sample_rate),
            jnp.fft.fftfreq(shape[1], d=1.0 / sample_rate),
            jnp.fft.rfftfreq(shape[2], d=1.0 / sample_rate),
        )
    raise ValueError(f"Unsupported spatial_dims: {spatial_dims}")
