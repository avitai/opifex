"""Input validation and error handling for spectral operations.

This module provides validation utilities for spectral operations to ensure
proper input shapes, types, and parameters are provided to spectral functions.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from opifex.typing import Axis, GridSpacing


def validate_spectral_input(
    x: jax.Array,
    spatial_dims: int,
    min_spatial_size: int = 2,
) -> None:
    """
    Validate input tensor for spectral operations.

    Args:
        x: Input tensor to validate
        spatial_dims: Expected number of spatial dimensions
        min_spatial_size: Minimum size for spatial dimensions

    Raises:
        TypeError: If input is not a JAX array
        ValueError: If shape or dimensions are invalid
    """
    if not isinstance(x, jax.Array):
        raise TypeError(f"Input must be a JAX array, got {type(x)}")

    if x.ndim < spatial_dims:
        raise ValueError(
            f"Input tensor has {x.ndim} dimensions, but requires at least "
            f"{spatial_dims} for {spatial_dims}D spectral operations"
        )

    # Check spatial dimensions
    spatial_shape = x.shape[-spatial_dims:]

    for i, size in enumerate(spatial_shape):
        if size < min_spatial_size:
            raise ValueError(
                f"Spatial dimension {i} has size {size}, but minimum "
                f"required size is {min_spatial_size}"
            )

    # Check for zero-sized dimensions
    if any(size == 0 for size in x.shape):
        raise ValueError("Input tensor cannot have zero-sized dimensions")

    # Check dtype is numeric
    if not jnp.issubdtype(x.dtype, jnp.floating) and not jnp.issubdtype(
        x.dtype, jnp.complexfloating
    ):
        raise ValueError(
            f"Input dtype {x.dtype} is not supported for spectral operations"
        )


def validate_spatial_dims(spatial_dims: int) -> None:
    """
    Validate spatial dimensions parameter.

    Args:
        spatial_dims: Number of spatial dimensions

    Raises:
        TypeError: If spatial_dims is not an integer
        ValueError: If spatial_dims is out of supported range
    """
    if not isinstance(spatial_dims, int):
        raise TypeError(f"spatial_dims must be an integer, got {type(spatial_dims)}")

    if spatial_dims < 1 or spatial_dims > 3:
        raise ValueError(f"spatial_dims must be 1, 2, or 3, got {spatial_dims}")


def validate_fft_shape(
    x_ft: jax.Array,
    target_shape: Sequence[int],
    spatial_dims: int,
) -> None:
    """
    Validate FFT tensor and target shape for IFFT operations.

    Args:
        x_ft: FFT-transformed tensor
        target_shape: Target shape for IFFT output
        spatial_dims: Number of spatial dimensions

    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If shapes are incompatible
    """
    if not isinstance(x_ft, jax.Array):
        raise TypeError(f"x_ft must be a JAX array, got {type(x_ft)}")

    if not isinstance(target_shape, (list, tuple)):
        raise TypeError(f"target_shape must be a sequence, got {type(target_shape)}")

    # Validate spatial dimensions
    validate_spatial_dims(spatial_dims)

    # Check dimensionality compatibility
    if len(target_shape) != len(x_ft.shape):
        raise ValueError(
            f"Target shape dimensionality {len(target_shape)} doesn't match "
            f"FFT tensor dimensionality {len(x_ft.shape)}"
        )

    # Check that x_ft has complex dtype
    if not jnp.issubdtype(x_ft.dtype, jnp.complexfloating):
        raise ValueError(f"FFT tensor must have complex dtype, got {x_ft.dtype}")

    # Validate target shape elements
    for i, size in enumerate(target_shape):
        if not isinstance(size, int) or size <= 0:
            raise ValueError(
                f"Target shape element {i} must be a positive integer, got {size}"
            )

    # Check FFT dimension compatibility
    target_spatial = target_shape[-spatial_dims:]
    fft_spatial = x_ft.shape[-spatial_dims:]

    # For real FFT, last dimension is compressed
    if spatial_dims >= 1:
        expected_last_dim = target_spatial[-1] // 2 + 1
        if fft_spatial[-1] != expected_last_dim:
            raise ValueError(
                f"FFT last spatial dimension {fft_spatial[-1]} incompatible "
                f"with target size {target_spatial[-1]} (expected {expected_last_dim})"
            )

    # Check non-last spatial dimensions
    if spatial_dims > 1:
        for i in range(spatial_dims - 1):
            fft_dim = fft_spatial[i]
            target_dim = target_spatial[i]
            if fft_dim != target_dim:
                raise ValueError(
                    f"FFT spatial dimension {i} has size {fft_dim}, "
                    f"but target requires {target_dim}"
                )


def validate_grid_spacing(  # noqa: PLR0912
    dx: GridSpacing,
    spatial_dims: int,
) -> list[float]:
    """
    Validate and normalize grid spacing parameter.

    Args:
        dx: Grid spacing (scalar, sequence, or JAX array)
        spatial_dims: Number of spatial dimensions

    Returns:
        List of grid spacing values for each dimension

    Raises:
        TypeError: If dx has incorrect type
        ValueError: If dx values are invalid
    """
    validate_spatial_dims(spatial_dims)

    # Handle scalar dx
    if isinstance(dx, (int, float)):
        if dx <= 0:
            raise ValueError(f"Grid spacing must be positive, got {dx}")
        return [float(dx)] * spatial_dims

    # Handle JAX array dx
    if isinstance(dx, jax.Array):
        if dx.ndim == 0:  # Scalar JAX array
            dx_val = float(dx)
            if dx_val <= 0:
                raise ValueError(f"Grid spacing must be positive, got {dx_val}")
            return [dx_val] * spatial_dims
        # Multi-dimensional JAX array - treat as sequence
        dx_list = [float(dx[i]) for i in range(dx.shape[0])]
        if len(dx_list) != spatial_dims:
            raise ValueError(
                f"Grid spacing length {len(dx_list)} doesn't match "
                f"spatial_dims {spatial_dims}"
            )
        for i, dx_i in enumerate(dx_list):
            if dx_i <= 0:
                raise ValueError(
                    f"Grid spacing element {i} must be positive, got {dx_i}"
                )
        return dx_list

    # Handle sequence dx
    if not isinstance(dx, (list, tuple)):
        raise TypeError(f"dx must be a scalar or sequence, got {type(dx)}")

    dx_list = list(dx)

    if len(dx_list) != spatial_dims:
        raise ValueError(
            f"Grid spacing length {len(dx_list)} doesn't match "
            f"spatial_dims {spatial_dims}"
        )

    # Validate each element
    for i, dx_i in enumerate(dx_list):
        if not isinstance(dx_i, (int, float, jax.Array)):
            raise TypeError(f"dx element {i} must be numeric, got {type(dx_i)}")
        if dx_i <= 0:
            raise ValueError(f"dx element {i} must be positive, got {dx_i}")

    return [float(dx_i) for dx_i in dx_list]


def validate_axis_parameter(
    axis: Axis,
    spatial_dims: int,
) -> list[int]:
    """
    Validate and normalize axis parameter for spectral operations.

    Args:
        axis: Axis specification (int, sequence, or None)
        spatial_dims: Number of spatial dimensions

    Returns:
        List of validated axis indices

    Raises:
        TypeError: If axis has incorrect type
        ValueError: If axis values are out of range
    """
    validate_spatial_dims(spatial_dims)

    # Handle None (all spatial axes)
    if axis is None:
        return list(range(spatial_dims))

    # Handle single axis
    if isinstance(axis, int):
        if axis < 0 or axis >= spatial_dims:
            raise ValueError(
                f"Axis {axis} out of range for {spatial_dims} spatial dimensions"
            )
        return [axis]

    # Handle sequence of axes
    if not isinstance(axis, (list, tuple, jax.Array)):
        raise TypeError(f"axis must be an int, sequence, or None, got {type(axis)}")

    axis_list = list(axis)

    for i, ax in enumerate(axis_list):
        if not isinstance(ax, (int, jax.Array)):
            raise TypeError(f"axis element {i} must be an integer, got {type(ax)}")
        if ax < 0 or ax >= spatial_dims:
            raise ValueError(
                f"Axis element {i} ({ax}) out of range for "
                f"{spatial_dims} spatial dimensions"
            )

    # Check for duplicates
    if len(set(axis_list)) != len(axis_list):
        raise ValueError(f"Duplicate axes found in {axis_list}")

    return axis_list
