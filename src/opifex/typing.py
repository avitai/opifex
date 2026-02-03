"""Type definitions for Opifex.

Provides type annotations and protocols for the Opifex framework,
ensuring type safety and clear interfaces throughout the codebase.
"""  # noqa: A005

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any, TypeAlias

import jax
from jax import core

# jaxtyping support - assumed to be available
from jaxtyping import Array, Float, Int


JAXTYPING_AVAILABLE = True


# Core axis type definitions
Axis: TypeAlias = int | Sequence[int] | None
SingleAxis: TypeAlias = int
MultiAxis: TypeAlias = Sequence[int]
OptionalAxis: TypeAlias = Axis | None

# JAX array types
JAXArray: TypeAlias = jax.Array
ArrayShape: TypeAlias = tuple[int, ...]

# Grid spacing type definitions
GridSpacing: TypeAlias = float | Sequence[float] | jax.Array
ScalarSpacing: TypeAlias = float
SequenceSpacing: TypeAlias = Sequence[float]
ArraySpacing: TypeAlias = jax.Array

# Spectral domain type definitions (jaxtyping-enhanced)
# Shape-aware types for spectral operations
SpatialField: TypeAlias = Float[Array, "batch channels *spatial"]
FrequencyField: TypeAlias = Float[Array, "batch channels *frequencies"]
SpectralTensor: TypeAlias = Float[Array, "batch channels *dims"]

# Grid and coordinate types
GridSpacingArray: TypeAlias = Float[Array, "dims"]  # noqa: F821
CoordinateGrid: TypeAlias = Float[Array, "*spatial_dims coord_dim"]

# Frequency domain types
FrequencyGrid: TypeAlias = Float[Array, "*frequency_dims"]
WavenumberGrid: TypeAlias = Float[Array, "*wavenumber_dims"]

# Neural operator specific types
FourierModes: TypeAlias = Int[Array, "dims"]  # noqa: F821
SpectralWeights: TypeAlias = Float[Array, "in_channels out_channels *modes"]


def ensure_optional_axes(x: Axis) -> Axis:
    """Ensure axis argument is statically known for JAX compilation.

    Args:
        x: Axis specification (int, sequence of ints, or None)

    Returns:
        Validated axis specification

    Raises:
        ValueError: If axis argument is not statically known
    """

    def force(x):
        if x is None:
            return None
        try:
            return operator.index(x)
        except TypeError:
            return tuple(i if isinstance(i, str) else operator.index(i) for i in x)

    return core.concrete_or_error(
        force, x, "The axis argument must be known statically."
    )


def is_single_axis(axis: Any) -> bool:
    """Type guard to check if value is a single axis (int)."""
    return isinstance(axis, int)


def is_multi_axis(axis: Any) -> bool:
    """Type guard to check if value is a multi-axis (sequence of ints)."""
    return (
        isinstance(axis, Sequence)
        and not isinstance(axis, str)
        and all(isinstance(x, int) for x in axis)
    )


def is_valid_axis(axis: Any) -> bool:
    """Type guard to check if value is a valid axis specification."""
    return axis is None or is_single_axis(axis) or is_multi_axis(axis)


def validate_axis_bounds(axis: Axis, ndim: int) -> bool:
    """Validate that axis indices are within bounds for given array dimensions.

    Args:
        axis: Axis specification to validate
        ndim: Number of dimensions in the array

    Returns:
        True if all axis indices are valid, False otherwise
    """
    if axis is None:
        return True

    if isinstance(axis, int):
        return -ndim <= axis < ndim

    if isinstance(axis, Sequence) and not isinstance(axis, str):
        return all(-ndim <= ax < ndim for ax in axis if isinstance(ax, int))

    return False


def normalize_axis(axis: Axis, ndim: int) -> Axis:
    """Normalize axis indices to positive values.

    Args:
        axis: Axis specification (may contain negative indices)
        ndim: Number of dimensions in the array

    Returns:
        Normalized axis specification with positive indices

    Raises:
        ValueError: If axis indices are out of bounds
    """
    if axis is None:
        return None

    if not validate_axis_bounds(axis, ndim):
        raise ValueError(f"Axis indices out of bounds for {ndim}-dimensional array")

    if isinstance(axis, int):
        return axis % ndim

    if isinstance(axis, Sequence) and not isinstance(axis, str):
        return tuple(ax % ndim for ax in axis if isinstance(ax, int))

    raise TypeError(f"Invalid axis type: {type(axis)}")


def assert_valid_axis(axis: Any) -> Axis:
    """Assert that value is a valid axis specification.

    Args:
        axis: Value to check

    Returns:
        The axis value if valid

    Raises:
        TypeError: If axis is not a valid specification
    """
    if not is_valid_axis(axis):
        raise TypeError(f"Expected valid axis specification, got {type(axis).__name__}")
    return axis


# Grid spacing validation functions


def is_scalar_spacing(spacing: Any) -> bool:
    """Type guard to check if value is a scalar spacing (float)."""
    return isinstance(spacing, (int, float))


def is_sequence_spacing(spacing: Any) -> bool:
    """Type guard to check if value is a sequence spacing."""
    return (
        isinstance(spacing, Sequence)
        and not isinstance(spacing, str)
        and all(isinstance(x, (int, float)) for x in spacing)
    )


def is_array_spacing(spacing: Any) -> bool:
    """Type guard to check if value is a JAX array spacing."""
    return hasattr(spacing, "shape") and hasattr(spacing, "dtype")


def is_valid_grid_spacing(spacing: Any) -> bool:
    """Type guard to check if value is a valid grid spacing specification."""
    return (
        is_scalar_spacing(spacing)
        or is_sequence_spacing(spacing)
        or is_array_spacing(spacing)
    )


def validate_grid_spacing_values(spacing: GridSpacing) -> bool:
    """Validate that grid spacing values are positive.

    Args:
        spacing: Grid spacing specification to validate

    Returns:
        True if all spacing values are positive, False otherwise
    """
    if isinstance(spacing, (int, float)):
        return float(spacing) > 0

    if isinstance(spacing, Sequence) and not isinstance(spacing, str):
        return all(float(s) > 0 for s in spacing)

    # For JAX arrays, we can't check values during compilation
    # This should be validated at runtime before JIT
    return hasattr(spacing, "shape") and hasattr(spacing, "dtype")


def normalize_grid_spacing(spacing: GridSpacing, spatial_dims: int) -> list[float]:
    """Normalize grid spacing to a list of floats for each dimension.

    Args:
        spacing: Grid spacing specification
        spatial_dims: Number of spatial dimensions

    Returns:
        List of grid spacing values for each dimension

    Raises:
        ValueError: If spacing values are invalid or dimensions don't match
        TypeError: If spacing type is invalid
    """
    if not is_valid_grid_spacing(spacing):
        raise TypeError(f"Invalid grid spacing type: {type(spacing)}")

    if isinstance(spacing, (int, float)):
        spacing_val = float(spacing)
        if spacing_val <= 0:
            raise ValueError(f"Grid spacing must be positive, got {spacing_val}")
        return [spacing_val] * spatial_dims

    if isinstance(spacing, Sequence) and not isinstance(spacing, str):
        spacing_list = [float(s) for s in spacing]
        if len(spacing_list) != spatial_dims:
            raise ValueError(
                f"Grid spacing length {len(spacing_list)} doesn't match "
                f"spatial_dims {spatial_dims}"
            )
        if not all(s > 0 for s in spacing_list):
            raise ValueError("All grid spacing values must be positive")
        return spacing_list

    if hasattr(spacing, "shape") and hasattr(spacing, "dtype"):
        # Handle JAX array spacing
        spacing_shape = getattr(spacing, "shape", None)
        if (
            spacing_shape is not None
            and len(spacing_shape) > 0
            and spacing_shape[0] != spatial_dims
        ):
            raise ValueError(
                f"Grid spacing shape {spacing_shape} doesn't match "
                f"spatial_dims {spatial_dims}"
            )
        return [float(spacing[i]) for i in range(spatial_dims)]

    raise TypeError(f"Unsupported grid spacing type: {type(spacing)}")


def ensure_static_grid_spacing(spacing: GridSpacing, spatial_dims: int) -> list[float]:
    """Ensure grid spacing is statically known for JAX compilation.

    Args:
        spacing: Grid spacing specification
        spatial_dims: Number of spatial dimensions

    Returns:
        Normalized list of grid spacing values

    Raises:
        ValueError: If spacing is not statically determinable
    """
    # For JAX compilation, we need concrete values
    if hasattr(spacing, "shape") and hasattr(spacing, "dtype"):
        # JAX arrays need to be concretized
        try:
            # This will fail during tracing if values are not concrete
            # We know spacing has shape and dtype, so it supports indexing
            return [float(spacing[i]) for i in range(spatial_dims)]  # type: ignore[index]
        except Exception as e:
            raise ValueError(
                f"Grid spacing must be statically known for JAX compilation: {e}"
            ) from e

    return normalize_grid_spacing(spacing, spatial_dims)


def assert_valid_grid_spacing(spacing: Any) -> GridSpacing:
    """Assert that value is a valid grid spacing specification.

    Args:
        spacing: Value to check

    Returns:
        The spacing value if valid

    Raises:
        TypeError: If spacing is not a valid specification
    """
    if not is_valid_grid_spacing(spacing):
        raise TypeError(
            f"Expected valid grid spacing specification, got {type(spacing).__name__}"
        )
    return spacing


# Export public API
__all__ = [  # noqa: RUF022
    "ArrayShape",
    "ArraySpacing",
    "Axis",
    "CoordinateGrid",
    "FourierModes",
    "FrequencyField",
    "FrequencyGrid",
    "GridSpacing",
    "GridSpacingArray",
    "JAXArray",
    "JAXTYPING_AVAILABLE",
    "MultiAxis",
    "OptionalAxis",
    "ScalarSpacing",
    "SequenceSpacing",
    "SingleAxis",
    "SpatialField",
    "SpectralTensor",
    "SpectralWeights",
    "WavenumberGrid",
    "assert_valid_axis",
    "assert_valid_grid_spacing",
    "ensure_optional_axes",
    "ensure_static_grid_spacing",
    "is_array_spacing",
    "is_multi_axis",
    "is_scalar_spacing",
    "is_sequence_spacing",
    "is_single_axis",
    "is_valid_axis",
    "is_valid_grid_spacing",
    "normalize_axis",
    "normalize_grid_spacing",
    "validate_axis_bounds",
    "validate_grid_spacing_values",
]
