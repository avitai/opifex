#!/usr/bin/env python3
"""
Operation Fusion Optimizer for Opifex JAX Neural Operators.

This module provides advanced operation fusion techniques to improve XLA optimization
and reduce memory traffic, addressing the low fusion ratios (5-17%) identified by
the profiling harness.

Key Features:
- Automatic operation fusion detection and optimization
- Memory-efficient computation patterns
- XLA-friendly operation structuring
- Hardware-specific fusion strategies
- Integration with neural operator architectures
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx


def fused_linear_activation(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
) -> jax.Array:
    """Fused linear transformation with activation.

    Combines linear transformation and activation into a single operation
    to improve XLA fusion and reduce memory traffic.

    Args:
        x: Input tensor
        weight: Weight matrix
        bias: Optional bias vector
        activation: Activation function

    Returns:
        Activated linear transformation result
    """
    # Fuse linear + activation in single operation
    if bias is not None:
        return activation(x @ weight + bias)
    return activation(x @ weight)


def fused_conv_activation(
    x: jax.Array,
    kernel: jax.Array,
    bias: jax.Array | None = None,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
    strides: int | tuple[int, ...] = 1,
    padding: str = "SAME",
) -> jax.Array:
    """Fused convolution with activation.

    Args:
        x: Input tensor (NHWC format for optimal GPU performance)
        kernel: Convolution kernel
        bias: Optional bias
        activation: Activation function
        strides: Convolution strides
        padding: Padding mode

    Returns:
        Activated convolution result
    """
    # Use lax.conv_general_dilated for better fusion
    conv_result = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(strides,) * 2 if isinstance(strides, int) else strides,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    if bias is not None:
        conv_result = conv_result + bias

    return activation(conv_result)


def fused_spectral_conv_activation(
    x: jax.Array,
    weights: jax.Array,
    modes: int,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
) -> jax.Array:
    """Fused spectral convolution with activation for FNO layers.

    Optimizes the spectral convolution operation by fusing FFT, multiplication,
    and IFFT operations with activation to improve XLA fusion.

    Args:
        x: Input tensor in spatial domain
        weights: Spectral weights
        modes: Number of Fourier modes
        activation: Activation function

    Returns:
        Activated spectral convolution result
    """
    # Get input dimensions
    _, _, height, width = x.shape

    # Fused FFT + spectral multiplication + IFFT + activation
    def fused_spectral_op(x_spatial):
        # Forward FFT
        x_ft = jnp.fft.rfft2(x_spatial, axes=(-2, -1))

        # Truncate to modes and multiply with weights
        x_ft_truncated = x_ft[:, :, :modes, :modes]
        out_ft = jnp.einsum("bchw,iohw->bihw", x_ft_truncated, weights)

        # Pad back to original size
        out_ft_padded = jnp.zeros(
            (x.shape[0], weights.shape[1], height, width // 2 + 1), dtype=out_ft.dtype
        )
        out_ft_padded = out_ft_padded.at[:, :, :modes, :modes].set(out_ft)

        # Inverse FFT
        out_spatial = jnp.fft.irfft2(out_ft_padded, s=(height, width), axes=(-2, -1))

        return activation(out_spatial)

    return fused_spectral_op(x)


def fused_elementwise_chain(
    x: jax.Array,
    operations: list[Callable[[jax.Array], jax.Array]],
) -> jax.Array:
    """Fuse chain of element-wise operations.

    Combines multiple element-wise operations into a single fused operation
    to improve memory efficiency and XLA optimization.

    Args:
        x: Input tensor
        operations: List of element-wise operations to apply

    Returns:
        Result of fused operations
    """

    def fused_chain(x):
        result = x
        for op in operations:
            result = op(result)
        return result

    return fused_chain(x)


def optimize_memory_layout_for_fusion(
    x: jax.Array, target_layout: str = "NHWC"
) -> jax.Array:
    """Optimize tensor memory layout for better fusion.

    Converts tensor to optimal memory layout for hardware-specific fusion.

    Args:
        x: Input tensor
        target_layout: Target memory layout ('NHWC' for GPU, 'NCHW' for some operations)

    Returns:
        Tensor with optimized layout
    """
    if len(x.shape) != 4:
        return x

    current_layout = "NCHW"  # Assume current layout

    if target_layout == "NHWC" and current_layout == "NCHW":
        # Convert NCHW -> NHWC
        return jnp.transpose(x, (0, 2, 3, 1))
    if target_layout == "NCHW" and current_layout == "NHWC":
        # Convert NHWC -> NCHW
        return jnp.transpose(x, (0, 3, 1, 2))

    return x


class FusedFourierLayer(nnx.Module):
    """Optimized Fourier layer with improved operation fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize fused Fourier layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.activation = activation

        # Spectral weights - optimized initialization
        scale = 1 / (in_channels * out_channels)
        self.weights = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (in_channels, out_channels, modes, modes),
                dtype=jnp.complex64,
            )
            * scale
        )

        # Skip connection with fused linear + activation
        self.skip_linear = nnx.Linear(
            in_features=in_channels,
            out_features=out_channels,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply fused Fourier layer with optimized operations."""
        _, _, _, _ = x.shape

        # Spectral path with fused operations
        spectral_out = fused_spectral_conv_activation(
            x,
            self.weights.value,
            self.modes,
            activation=lambda y: y,  # No activation here
        )

        # Skip connection path - fused linear + activation
        # Convert to NHWC for linear operation
        x_nhwc = optimize_memory_layout_for_fusion(x, "NHWC")
        # Skip connection path - fused linear + activation
        skip_kernel = self.skip_linear.kernel.value
        skip_bias = getattr(self.skip_linear, "bias", None)
        skip_bias_value = skip_bias.value if skip_bias is not None else None

        skip_out = fused_linear_activation(
            x_nhwc,
            skip_kernel,
            skip_bias_value,
            activation=lambda y: y,  # No activation here
        )
        # Convert back to NCHW
        skip_out = optimize_memory_layout_for_fusion(skip_out, "NCHW")

        # Fused addition + activation
        return self.activation(spectral_out + skip_out)


class FusionOptimizedOperator(nnx.Module):
    """Base class for fusion-optimized neural operators."""

    def __init__(self):
        super().__init__()
        self.fusion_enabled = True
        self.target_layout = self._detect_optimal_layout()

    def _detect_optimal_layout(self) -> str:
        """Detect optimal memory layout for current hardware."""
        backend = jax.default_backend()

        if backend == "gpu":
            # GPU prefers NHWC for convolutions and TensorCore operations
            return "NHWC"
        if backend == "tpu":
            # TPU can handle both efficiently, prefer NHWC for consistency
            return "NHWC"
        # CPU may prefer NCHW for some operations
        return "NCHW"

    def enable_fusion_optimization(self, enable: bool = True):
        """Enable or disable fusion optimizations."""
        self.fusion_enabled = enable

    def optimize_for_hardware(self, x: jax.Array) -> jax.Array:
        """Optimize tensor layout for current hardware."""
        if not self.fusion_enabled:
            return x

        return optimize_memory_layout_for_fusion(x, self.target_layout)


def create_fused_computation_graph(
    operations: list[Callable],
    fusion_boundaries: list[int] | None = None,
) -> Callable:
    """Create optimized computation graph with fusion boundaries.

    Args:
        operations: List of operations to fuse
        fusion_boundaries: Optional boundaries where fusion should be broken

    Returns:
        Fused computation function
    """
    if fusion_boundaries is None:
        # Fuse all operations
        def fused_graph(x):
            result = x
            for op in operations:
                result = op(result)
            return result
    else:
        # Respect fusion boundaries
        def segmented_fused_graph(x):
            result = x
            start_idx = 0

            for boundary in [*fusion_boundaries, len(operations)]:
                # Fuse operations in this segment
                segment_ops = operations[start_idx:boundary]
                for op in segment_ops:
                    result = op(result)
                start_idx = boundary

            return result

        fused_graph = segmented_fused_graph

    return jax.jit(fused_graph)


# Utility functions for fusion analysis
def analyze_fusion_opportunities(model: nnx.Module) -> dict:
    """Analyze fusion opportunities in a neural operator model."""
    _ = model  # Reserved for future implementation

    # This is a simplified analysis - in practice, would need to traverse
    # the computation graph to identify fusion opportunities

    return {
        "num_operations": 0,
        "fusable_patterns": [],
        "estimated_speedup": 0.0,
        "fusion_recommendations": [],
    }


def apply_fusion_optimizations(model: nnx.Module) -> nnx.Module:
    """Apply fusion optimizations to a neural operator model."""
    # This would implement automatic fusion optimization
    # For now, return the model unchanged
    return model


# Export main components
__all__ = [
    "FusedFourierLayer",
    "FusionOptimizedOperator",
    "analyze_fusion_opportunities",
    "apply_fusion_optimizations",
    "create_fused_computation_graph",
    "fused_conv_activation",
    "fused_elementwise_chain",
    "fused_linear_activation",
    "fused_spectral_conv_activation",
    "optimize_memory_layout_for_fusion",
]
