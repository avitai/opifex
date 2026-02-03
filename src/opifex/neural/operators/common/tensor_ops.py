"""
Standardized tensor operations for neural operators.

This module provides consistent, validated tensor operations across all
neural operator implementations, ensuring proper channel handling,
spectral operations, and shape management.
"""

import logging
from collections.abc import Sequence
from functools import wraps
from typing import Any

import jax
import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array


logger = logging.getLogger(__name__)


@beartype
def validate_tensor_shape(
    tensor: jax.Array, expected_dims: int, min_spatial_dims: int = 1
) -> None:
    """
    Validate tensor has expected number of dimensions.

    Args:
        tensor: Input tensor to validate
        expected_dims: Expected number of dimensions
        min_spatial_dims: Minimum spatial dimensions required

    Raises:
        ValueError: If tensor shape is invalid
    """
    if len(tensor.shape) != expected_dims:
        raise ValueError(
            f"Expected {expected_dims}D tensor, got {len(tensor.shape)}D: "
            f"{tensor.shape}"
        )

    if len(tensor.shape) < 2 + min_spatial_dims:
        raise ValueError(
            f"Tensor must have at least batch + channel + "
            f"{min_spatial_dims} spatial dims, got {tensor.shape}"
        )


@beartype
def validate_channel_compatibility(
    input_channels: int, expected_channels: int, operation_name: str = "operation"
) -> None:
    """
    Validate channel dimension compatibility with detailed error messages.

    Args:
        input_channels: Actual input channels
        expected_channels: Expected input channels
        operation_name: Name of operation for error context

    Raises:
        ValueError: If channels don't match with detailed context
    """
    if input_channels != expected_channels:
        raise ValueError(
            f"{operation_name}: Channel mismatch - input has "
            f"{input_channels} channels, but operation expects "
            f"{expected_channels} channels. Consider using channel "
            f"transformation or adjust model configuration."
        )


@beartype
def validate_einsum_dimensions(
    operand1: jax.Array,
    operand2: jax.Array,
    pattern: str,
    operation_name: str = "einsum",
) -> None:
    """
    Validate einsum operand dimensions match pattern requirements.

    Args:
        operand1: First operand array
        operand2: Second operand array
        pattern: Einsum pattern (e.g., "bi...,ij...->bj...")
        operation_name: Name of operation for error context

    Raises:
        ValueError: If dimensions don't match pattern requirements
    """
    try:
        # Test einsum with small arrays to validate dimensions
        test_shape1 = [min(2, s) for s in operand1.shape]
        test_shape2 = [min(2, s) for s in operand2.shape]
        test1 = jnp.ones(test_shape1)
        test2 = jnp.ones(test_shape2)
        jnp.einsum(pattern, test1, test2)
    except ValueError as e:
        raise ValueError(
            f"{operation_name}: Einsum dimension mismatch for pattern '{pattern}'. "
            f"Operand 1 shape: {operand1.shape}, Operand 2 shape: {operand2.shape}. "
            f"Original error: {e!s}"
        ) from e


@beartype
def standardized_fft(x: jax.Array, spatial_dims: int) -> jax.Array:
    """
    Standardized FFT operation with proper dimension handling.

    Args:
        x: Input tensor (batch, channels, *spatial)
        spatial_dims: Number of spatial dimensions

    Returns:
        FFT transformed tensor with consistent complex type
    """
    validate_tensor_shape(x, 2 + spatial_dims, spatial_dims)

    # Get spatial axes (skip batch and channel dimensions)
    spatial_axes = tuple(range(-spatial_dims, 0))

    if spatial_dims == 1:
        return jnp.fft.rfft(x, axis=-1)
    if spatial_dims == 2:
        return jnp.fft.rfft2(x, axes=spatial_axes)
    if spatial_dims == 3:
        return jnp.fft.rfftn(x, axes=spatial_axes)
    raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")


@beartype
def standardized_ifft(
    x_ft: jax.Array, target_shape: Sequence[int], spatial_dims: int
) -> Array:
    """
    Standardized IFFT operation with proper shape restoration.

    Args:
        x_ft: FFT transformed tensor
        target_shape: Target shape for output
        spatial_dims: Number of spatial dimensions

    Returns:
        Real-valued tensor with target shape
    """
    # Get spatial axes and target spatial sizes
    spatial_axes = tuple(range(-spatial_dims, 0))
    target_spatial = target_shape[2:]  # Skip batch and channel dims

    if spatial_dims == 1:
        result = jnp.fft.irfft(x_ft, n=target_spatial[0], axis=-1)
    elif spatial_dims == 2:
        result = jnp.fft.irfft2(x_ft, s=target_spatial, axes=spatial_axes)
    elif spatial_dims == 3:
        result = jnp.fft.irfftn(x_ft, s=target_spatial, axes=spatial_axes)
    else:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

    return jnp.asarray(result.real)


@beartype
def apply_linear_with_channel_transform(x: Array, linear_layer: nnx.Linear) -> Array:
    """
    Apply linear transformation with proper channel dimension handling.

    This function handles the channel transformation by applying the linear
    layer pointwise across spatial dimensions.

    Args:
        x: Input tensor (batch, in_channels, *spatial)
        linear_layer: Flax NNX Linear layer

    Returns:
        Output tensor (batch, out_channels, *spatial)
    """
    # Store original spatial shape
    original_shape = x.shape
    batch_size = original_shape[0]
    spatial_shape = original_shape[2:]

    # Reshape to (batch * spatial_points, in_channels)
    x_flat = x.reshape(batch_size, original_shape[1], -1)  # (batch, channels, points)
    x_flat = jnp.transpose(x_flat, (0, 2, 1))  # (batch, points, channels)
    x_flat = x_flat.reshape(-1, original_shape[1])  # (batch * points, channels)

    # Apply linear transformation
    y_flat = linear_layer(x_flat)  # (batch * points, out_channels)

    # Reshape back to spatial format
    out_channels = y_flat.shape[-1]
    y_reshaped = y_flat.reshape(
        batch_size, -1, out_channels
    )  # (batch, points, out_channels)
    y_reshaped = jnp.transpose(y_reshaped, (0, 2, 1))  # (batch, out_channels, points)
    return y_reshaped.reshape(batch_size, out_channels, *spatial_shape)


@beartype
def safe_spectral_multiply(
    x_modes: Array, weights: Array, modes: Sequence[int]
) -> Array:
    """
    Safe spectral multiplication with automatic shape handling.

    Args:
        x_modes: Input modes tensor (batch, in_channels, *mode_dims)
        weights: Weight tensor (in_channels, out_channels, *mode_dims)
        modes: Target mode counts for each dimension

    Returns:
        Output modes tensor (batch, out_channels, *mode_dims)
    """
    # Validate input shapes
    _, in_channels = x_modes.shape[:2]
    weight_in_channels, _ = weights.shape[:2]

    validate_channel_compatibility(
        in_channels, weight_in_channels, "safe_spectral_multiply"
    )

    # Ensure consistent mode handling between input and weights
    # Both should be truncated to the minimum effective modes
    effective_modes = []
    for i, mode_count in enumerate(modes):
        x_axis = i + 2  # Skip batch and channel dimensions for input
        w_axis = i + 2  # Skip in_channels and out_channels for weights

        x_size = x_modes.shape[x_axis] if x_axis < len(x_modes.shape) else mode_count
        w_size = weights.shape[w_axis] if w_axis < len(weights.shape) else mode_count

        # Use minimum of requested modes and actual sizes
        effective_mode = min(mode_count, x_size, w_size)
        effective_modes.append(effective_mode)

    # Truncate both input and weights to effective modes
    effective_x = x_modes
    effective_weights = weights

    for i, eff_mode in enumerate(effective_modes):
        x_axis = i + 2  # Skip batch and channel dimensions
        w_axis = i + 2  # Skip in_channels and out_channels

        if x_axis < len(effective_x.shape) and effective_x.shape[x_axis] > eff_mode:
            effective_x = jnp.take(effective_x, jnp.arange(eff_mode), axis=x_axis)

        if (
            w_axis < len(effective_weights.shape)
            and effective_weights.shape[w_axis] > eff_mode
        ):
            effective_weights = jnp.take(
                effective_weights, jnp.arange(eff_mode), axis=w_axis
            )

    # Validate einsum dimensions before operation
    pattern = "bi...,ij...->bj..."
    validate_einsum_dimensions(
        effective_x, effective_weights, pattern, "safe_spectral_multiply"
    )

    # Perform spectral multiplication using einsum
    return jnp.einsum(pattern, effective_x, effective_weights)


@beartype
def ensure_channel_compatibility(
    input_channels: int,
    target_channels: int,
    x: Array,
    channel_mapper: nnx.Linear | None = None,
) -> Array:
    """
    Ensure tensor has target number of channels, applying transformation if needed.

    Args:
        input_channels: Current number of channels
        target_channels: Target number of channels
        x: Input tensor (batch, input_channels, *spatial)
        channel_mapper: Optional linear layer for channel transformation

    Returns:
        Tensor with target number of channels (batch, target_channels, *spatial)

    Raises:
        ValueError: If channel_mapper is None and channels don't match
    """
    if input_channels == target_channels:
        return x

    if channel_mapper is None:
        raise ValueError(
            f"Channel mismatch: input has {input_channels} channels, "
            f"target has {target_channels} channels, but no channel_mapper provided"
        )

    return apply_linear_with_channel_transform(x, channel_mapper)


@beartype
def interpolate_spatial_dimensions(
    x: Array, target_spatial_shape: Sequence[int], method: str = "linear"
) -> Array:
    """
    Interpolate tensor to match target spatial dimensions.

    Args:
        x: Input tensor (batch, channels, *spatial)
        target_spatial_shape: Target spatial dimensions
        method: Interpolation method ('linear' or 'nearest')

    Returns:
        Interpolated tensor with target spatial shape
    """
    current_spatial = x.shape[2:]

    if current_spatial == tuple(target_spatial_shape):
        return x

    # Simple implementation using JAX resizing
    # For more sophisticated interpolation, could use jax.image.resize
    indices = []
    for _, (current_size, target_size) in enumerate(
        zip(current_spatial, target_spatial_shape, strict=False)
    ):
        axis_indices = jnp.asarray(
            jnp.round(jnp.linspace(0, current_size - 1, target_size))
        )
        indices.append(axis_indices)

    # Apply interpolation
    result = x
    for i, axis_indices in enumerate(indices):
        axis = i + 2  # Skip batch and channel dimensions
        result = jnp.take(result, axis_indices, axis=axis)

    return result


@beartype
def compute_padding_for_conv(
    input_size: int, kernel_size: int, stride: int = 1, target_size: int | None = None
) -> tuple[int, int]:
    """
    Compute padding for convolution to achieve target output size.

    Args:
        input_size: Input spatial dimension size
        kernel_size: Convolution kernel size
        stride: Convolution stride
        target_size: Target output size (if None, uses input_size)

    Returns:
        Tuple of (left_pad, right_pad)
    """
    if target_size is None:
        target_size = input_size

    total_pad = max(0, (target_size - 1) * stride + kernel_size - input_size)
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad

    return left_pad, right_pad


class StandardSpectralConv(nnx.Module):
    """Standardized spectral convolution with enhanced validation and compatibility."""

    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = tuple(modes)

        # Initialize weights with proper shape validation
        spatial_dims = len(modes)

        # For real FFT, last dimension modes are (modes[-1] // 2 + 1)
        weight_shape: tuple[int, ...]
        if spatial_dims == 1:
            weight_shape = (in_channels, out_channels, modes[0] // 2 + 1)
        elif spatial_dims == 2:
            weight_shape = (in_channels, out_channels, modes[0], modes[1] // 2 + 1)
        elif spatial_dims == 3:
            weight_shape = (
                in_channels,
                out_channels,
                modes[0],
                modes[1],
                modes[2] // 2 + 1,
            )
        else:
            raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

        self.weights = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), weight_shape)
        )

    def __call__(self, x_ft: Array) -> Array:
        """
        Apply spectral convolution with enhanced validation.

        Args:
            x_ft: FFT transformed input (batch, in_channels, *mode_dims)

        Returns:
            Spectral convolution output (batch, out_channels, *mode_dims)
        """
        # Validate input channels
        validate_channel_compatibility(
            x_ft.shape[1], self.in_channels, f"StandardSpectralConv(modes={self.modes})"
        )

        # Apply safe spectral multiplication
        return safe_spectral_multiply(x_ft, self.weights.value, self.modes)


@beartype
def match_spatial_dimensions(x: Array, target_shape: Sequence[int]) -> Array:
    """
    Match spatial dimensions of tensor to target shape using interpolation.

    Args:
        x: Input tensor (batch, channels, *spatial)
        target_shape: Target spatial dimensions only (no batch/channel)

    Returns:
        Tensor with matched spatial dimensions
    """
    current_spatial = x.shape[2:]

    if current_spatial == tuple(target_shape):
        return x

    return interpolate_spatial_dimensions(x, target_shape)


@beartype
def create_channel_mapper(
    in_channels: int, out_channels: int, rngs: nnx.Rngs
) -> nnx.Linear:
    """
    Create a channel mapping layer for dimension compatibility.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        rngs: Random number generator state

    Returns:
        Linear layer for channel transformation
    """
    return nnx.Linear(in_channels, out_channels, rngs=rngs)


class TensorShapeValidator:
    """Comprehensive tensor shape validation for neural operators."""

    @staticmethod
    def validate_einsum(equation: str, *operands: Array) -> None:
        """Validate einsum operation dimensions match equation.

        Args:
            equation: Einstein summation equation
            *operands: Input tensors

        Raises:
            ValueError: If tensor dimensions don't match equation requirements
        """
        try:
            # Parse einsum equation
            if "->" in equation:
                inputs, _ = equation.split("->")
                input_specs = inputs.split(",")
            else:
                input_specs = equation.split(",")

            # Validate number of operands matches equation
            if len(operands) != len(input_specs):
                _raise_einsum_dimension_error(
                    f"Einsum equation expects {len(input_specs)} operands, "
                    f"but {len(operands)} were provided. Equation: {equation}"
                )

            # Validate each operand's dimensions
            for i, (operand, input_spec) in enumerate(
                zip(operands, input_specs, strict=False)
            ):
                spec = input_spec.strip()
                expected_dims = len(spec)
                actual_dims = len(operand.shape)

                if actual_dims != expected_dims:
                    _raise_einsum_dimension_error(
                        f"Operand {i} has {actual_dims} dimensions "
                        f"{operand.shape}, but einsum spec '{spec}' expects "
                        f"{expected_dims} dimensions. Full equation: {equation}"
                    )

            # Collect dimension sizes for consistency checking
            dim_sizes: dict[str, int] = {}
            for operand, input_spec in zip(operands, input_specs, strict=False):
                spec = input_spec.strip()
                for dim_idx, dim_label in enumerate(spec):
                    size = operand.shape[dim_idx]
                    if dim_label in dim_sizes:
                        if dim_sizes[dim_label] != size:
                            _raise_einsum_dimension_error(
                                f"Dimension '{dim_label}' has inconsistent "
                                f"sizes: {dim_sizes[dim_label]} vs {size}. "
                                f"Equation: {equation}, Operand shapes: "
                                f"{[op.shape for op in operands]}"
                            )
                    else:
                        dim_sizes[dim_label] = size

        except Exception:
            logger.exception(f"Einsum validation failed for equation: {equation}")
            _raise_einsum_dimension_error(
                f"Equation: {equation}\nOperand shapes: {[op.shape for op in operands]}"
            )

    @staticmethod
    def validate_attention_dims(query: Array, key: Array, value: Array) -> None:
        """Validate attention mechanism tensor dimensions.

        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]
            value: Value tensor [batch, seq_len, d_model]

        Raises:
            ValueError: If attention tensor dimensions are incompatible
        """
        # Validate basic shape requirements
        if len(query.shape) != 3:
            raise ValueError(
                f"Query must be 3D [batch, seq_len, d_model], got shape {query.shape}"
            )
        if len(key.shape) != 3:
            raise ValueError(
                f"Key must be 3D [batch, seq_len, d_model], got shape {key.shape}"
            )
        if len(value.shape) != 3:
            raise ValueError(
                f"Value must be 3D [batch, seq_len, d_model], got shape {value.shape}"
            )

        batch_q, _, d_q = query.shape
        batch_k, seq_k, d_k = key.shape
        batch_v, seq_v, _ = value.shape

        # Validate batch dimensions match
        if not (batch_q == batch_k == batch_v):
            raise ValueError(
                f"Batch dimensions must match: query={batch_q}, "
                f"key={batch_k}, value={batch_v}"
            )

        # Validate sequence lengths match for key and value
        if seq_k != seq_v:
            raise ValueError(
                f"Key and value sequence lengths must match: key={seq_k}, value={seq_v}"
            )

        # Validate query and key have same model dimension
        if d_q != d_k:
            raise ValueError(
                f"Query and key model dimensions must match: query={d_q}, key={d_k}"
            )

    @staticmethod
    def validate_spectral_conv_dims(input_tensor: Array, weights: Array) -> None:
        """Validate spectral convolution tensor compatibility.

        Args:
            input_tensor: Input tensor [batch, in_channels, *spatial_dims]
            weights: Weight tensor [in_channels, out_channels, *mode_dims]

        Raises:
            ValueError: If spectral convolution dimensions are incompatible
        """
        if len(input_tensor.shape) < 3:
            raise ValueError(
                f"Input tensor must have at least 3 dimensions [batch, channels, ...], "
                f"got shape {input_tensor.shape}"
            )

        if len(weights.shape) < 3:
            raise ValueError(
                f"Weight tensor must have at least 3 dimensions "
                f"[in_channels, out_channels, ...], got shape {weights.shape}"
            )

        _, in_channels = input_tensor.shape[:2]
        spatial_dims = input_tensor.shape[2:]

        weight_in_channels, _ = weights.shape[:2]
        mode_dims = weights.shape[2:]

        # Validate channel dimensions match
        if in_channels != weight_in_channels:
            raise ValueError(
                f"Input channels ({in_channels}) must match weight input "
                f"channels ({weight_in_channels})"
            )

        # Validate spatial and mode dimensions are compatible
        if len(spatial_dims) != len(mode_dims):
            raise ValueError(
                f"Spatial dimensions ({len(spatial_dims)}) must match "
                f"mode dimensions ({len(mode_dims)})"
            )

        # Check that mode dimensions don't exceed spatial dimensions
        for i, (spatial_size, mode_size) in enumerate(
            zip(spatial_dims, mode_dims, strict=False)
        ):
            if mode_size > spatial_size:
                raise ValueError(
                    f"Mode size {mode_size} in dimension {i} exceeds "
                    f"spatial size {spatial_size}"
                )

    @staticmethod
    def validate_tensor_shapes(
        expected_shapes: dict[str, tuple[int, ...]], **tensors: Array
    ) -> None:
        """Validate multiple tensors against expected shapes.

        Args:
            expected_shapes: Dictionary mapping tensor names to expected shapes
            **tensors: Named tensors to validate

        Raises:
            ValueError: If any tensor doesn't match expected shape
        """
        for name, tensor in tensors.items():
            if name in expected_shapes:
                expected = expected_shapes[name]
                actual = tensor.shape
                if actual != expected:
                    raise ValueError(
                        f"Tensor '{name}' has shape {actual}, expected {expected}"
                    )


class EinsumPatterns:
    """Standardized einsum patterns for neural operators."""

    # Attention patterns
    ATTENTION_QK = "bqd,bkd->bqk"  # Query-Key attention scores
    ATTENTION_SCORES_V = "bqk,bkd->bqd"  # Attention scores with values

    # Spectral convolution patterns
    SPECTRAL_CONV_2D = "bixy,ioxy->boxy"  # 2D spectral convolution
    SPECTRAL_CONV_3D = "bixyz,ioxyz->boxyz"  # 3D spectral convolution

    # Multipole interaction patterns
    MULTIPOLE_EXPANSION = "bij,bjk->bik"  # Basic multipole interaction
    MULTIPOLE_COEFFS = "bnlm,nlm->bn"  # Multipole coefficient contraction

    # Tensor factorization patterns
    TUCKER_MODE1 = "ijk,ia->ajk"  # Tucker decomposition mode-1
    TUCKER_MODE2 = "ijk,jb->ibk"  # Tucker decomposition mode-2
    TUCKER_MODE3 = "ijk,kc->ijc"  # Tucker decomposition mode-3

    # Graph neural operator patterns
    NODE_FEATURES = "ni,ij->nj"  # Node feature transformation
    EDGE_AGGREGATION = "eij,ej->ei"  # Edge feature aggregation


def validate_tensor_shapes(func):
    """Decorator to add automatic tensor shape validation to neural operator methods.

    Usage:
        @validate_tensor_shapes
        def forward(self, x):
            # Function implementation
            return output
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            # Call the original function
            result = func(self, *args, **kwargs)

            # Validate output shape if method has _expected_output_shape
            if hasattr(self, "_expected_output_shape") and hasattr(result, "shape"):
                expected = self._expected_output_shape
                actual = result.shape
                if callable(expected):
                    expected = expected(*args, **kwargs)
                if expected is not None and actual != expected:
                    logger.warning(
                        f"Output shape {actual} doesn't match expected {expected} "
                        f"in {func.__name__}"
                    )

            return result

        except Exception:
            logger.exception(f"Tensor operation failed in {func.__name__}")
            if args:
                logger.info(
                    f"Input shapes: {[getattr(arg, 'shape', 'N/A') for arg in args]}"
                )
            raise

    return wrapper


def safe_einsum(equation: str, *operands: Array, **kwargs) -> Array:
    """Safe einsum operation with automatic validation.

    Args:
        equation: Einstein summation equation
        *operands: Input tensors
        **kwargs: Additional arguments passed to jnp.einsum

    Returns:
        Result of einsum operation

    Raises:
        ValueError: If tensor dimensions don't match equation
    """
    # Validate before computation
    TensorShapeValidator.validate_einsum(equation, *operands)

    try:
        return jnp.einsum(equation, *operands, **kwargs)
    except Exception:
        logger.exception("Einsum computation failed")
        logger.info(f"Equation: {equation}")
        logger.info(f"Operand shapes: {[op.shape for op in operands]}")
        raise


def safe_attention(
    query: Array, key: Array, value: Array, mask: Array | None = None
) -> Array:
    """Safe attention computation with validation.

    Args:
        query: Query tensor [batch, seq_len, d_model]
        key: Key tensor [batch, seq_len, d_model]
        value: Value tensor [batch, seq_len, d_model]
        mask: Optional attention mask

    Returns:
        Attention output [batch, seq_len, d_model]
    """
    # Validate input dimensions
    TensorShapeValidator.validate_attention_dims(query, key, value)

    # Compute attention scores
    d_k = query.shape[-1]
    scores = safe_einsum(EinsumPatterns.ATTENTION_QK, query, key) / jnp.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = jnp.where(mask, scores, -jnp.inf)

    # Apply softmax and compute output
    attention_weights = jax.nn.softmax(scores, axis=-1)
    return safe_einsum(EinsumPatterns.ATTENTION_SCORES_V, attention_weights, value)


def safe_spectral_conv(
    input_tensor: Array, weights: Array, modes: tuple[int, ...]
) -> Array:
    """Safe spectral convolution with validation.

    Args:
        input_tensor: Input tensor [batch, in_channels, *spatial_dims]
        weights: Weight tensor [in_channels, out_channels, *mode_dims]
        modes: Number of modes to keep in each dimension

    Returns:
        Convolution output [batch, out_channels, *spatial_dims]
    """
    # Validate input dimensions
    TensorShapeValidator.validate_spectral_conv_dims(input_tensor, weights)

    # Get dimensions
    spatial_dims = input_tensor.shape[2:]

    # Transform to frequency domain
    x_ft = jnp.fft.rfftn(input_tensor, axes=tuple(range(2, len(input_tensor.shape))))

    # Extract modes
    mode_slices = [slice(None), slice(None)]  # batch and channel dimensions
    for i, (spatial_size, mode_count) in enumerate(
        zip(spatial_dims, modes, strict=False)
    ):
        if i == len(spatial_dims) - 1:  # Last dimension for rfft
            mode_slices.append(slice(0, min(mode_count, spatial_size // 2 + 1)))
        else:
            mode_slices.append(slice(0, min(mode_count, spatial_size)))

    x_ft_modes = x_ft[tuple(mode_slices)]

    # Apply spectral convolution
    if len(spatial_dims) == 2:
        pattern = EinsumPatterns.SPECTRAL_CONV_2D
    elif len(spatial_dims) == 3:
        pattern = EinsumPatterns.SPECTRAL_CONV_3D
    else:
        # Fallback for other dimensions
        in_dims = "".join(chr(ord("i") + i) for i in range(len(spatial_dims)))
        out_dims = "".join(chr(ord("o") + i) for i in range(len(spatial_dims)))
        pattern = f"bi{in_dims},io{out_dims}->bo{out_dims}"

    out_ft = safe_einsum(pattern, x_ft_modes, weights)

    # Pad back to original size and transform to spatial domain
    pad_widths = [(0, 0), (0, 0)]  # No padding for batch and channel dims
    for i, (spatial_size, _) in enumerate(zip(spatial_dims, modes, strict=False)):
        current_size = out_ft.shape[2 + i]
        if i == len(spatial_dims) - 1:  # Last dimension for rfft
            target_size = spatial_size // 2 + 1
        else:
            target_size = spatial_size

        if current_size < target_size:
            pad_widths.append((0, target_size - current_size))
        else:
            pad_widths.append((0, 0))

    if any(pad[1] > 0 for pad in pad_widths):
        out_ft = jnp.pad(out_ft, pad_widths)

    # Transform back to spatial domain
    return jnp.fft.irfftn(
        out_ft, s=spatial_dims, axes=tuple(range(2, len(out_ft.shape)))
    )


# Utility functions for common tensor operations
def ensure_tensor_compatibility(*tensors: Array) -> None:
    """Ensure tensors have compatible batch dimensions.

    Args:
        *tensors: Tensors to check

    Raises:
        ValueError: If batch dimensions don't match
    """
    if not tensors:
        return

    batch_size = tensors[0].shape[0]
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"Tensor {i} has batch size {tensor.shape[0]}, "
                f"expected {batch_size} to match tensor 0"
            )


def validate_operator_config(config: dict[str, Any], required_keys: list[str]) -> None:
    """Validate neural operator configuration.

    Args:
        config: Configuration dictionary
        required_keys: List of required configuration keys

    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def get_tensor_info(tensor: Array) -> dict[str, Any]:
    """Get comprehensive information about a tensor.

    Args:
        tensor: Input tensor

    Returns:
        Dictionary with tensor information
    """
    return {
        "shape": tensor.shape,
        "dtype": tensor.dtype,
        "size": tensor.size,
        "ndim": tensor.ndim,
        "memory_mb": tensor.nbytes / (1024 * 1024),
        "min": float(jnp.min(tensor)),
        "max": float(jnp.max(tensor)),
        "mean": float(jnp.mean(tensor)),
        "std": float(jnp.std(tensor)),
    }


def _raise_einsum_dimension_error(message: str) -> None:
    """Helper function to raise einsum dimension errors."""
    raise ValueError(message)
