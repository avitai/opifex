#!/usr/bin/env python3
"""
Memory Layout Optimization for Opifex JAX Neural Operators.

This module provides comprehensive memory layout optimization to improve GPU
performance, specifically addressing the NHWC vs NCHW layout optimization
identified by the profiling harness.

Key Features:
- Automatic layout detection and optimization
- Hardware-specific layout recommendations
- Efficient layout conversion with minimal overhead
- Integration with neural operator architectures
- Performance monitoring and validation
"""

from collections.abc import Callable
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


class MemoryLayout(Enum):
    """Supported memory layout formats."""

    NCHW = "NCHW"  # Batch, Channels, Height, Width
    NHWC = "NHWC"  # Batch, Height, Width, Channels
    NDHWC = "NDHWC"  # Batch, Depth, Height, Width, Channels
    NCDHW = "NCDHW"  # Batch, Channels, Depth, Height, Width


class LayoutOptimizer:
    """Memory layout optimizer for neural operators."""

    def __init__(self):
        self.backend = jax.default_backend()
        self.optimal_layouts = self._get_optimal_layouts()
        self.conversion_cache = {}

    def _get_optimal_layouts(self) -> dict[str, MemoryLayout]:
        """Get optimal memory layouts for different operations and backends."""
        if self.backend == "gpu":
            return {
                "convolution": MemoryLayout.NHWC,  # GPU prefers NHWC for convolutions
                "matmul": MemoryLayout.NHWC,  # Better for TensorCore utilization
                "elementwise": MemoryLayout.NHWC,  # Consistent with conv operations
                "fft": MemoryLayout.NCHW,  # FFT may prefer NCHW for some cases
                "default": MemoryLayout.NHWC,  # Default to NHWC for GPU
            }
        if self.backend == "tpu":
            return {
                "convolution": MemoryLayout.NHWC,  # TPU also prefers NHWC
                "matmul": MemoryLayout.NHWC,
                "elementwise": MemoryLayout.NHWC,
                "fft": MemoryLayout.NHWC,
                "default": MemoryLayout.NHWC,
            }
        # CPU
        return {
            "convolution": MemoryLayout.NCHW,  # CPU may prefer NCHW
            "matmul": MemoryLayout.NCHW,
            "elementwise": MemoryLayout.NCHW,
            "fft": MemoryLayout.NCHW,
            "default": MemoryLayout.NCHW,
        }

    def detect_layout(self, x: jax.Array) -> MemoryLayout | None:
        """Detect the current memory layout of a tensor."""
        if len(x.shape) != 4:
            return None

        # Heuristic: assume smaller dimension is channels
        # This is not always accurate but works for most cases
        _, dim1, dim2, dim3 = x.shape

        if dim1 < dim2 and dim1 < dim3:
            return MemoryLayout.NCHW
        if dim3 < dim1 and dim3 < dim2:
            return MemoryLayout.NHWC
        # Ambiguous, assume NCHW as default
        return MemoryLayout.NCHW

    def convert_layout(
        self,
        x: jax.Array,
        target_layout: MemoryLayout,
        current_layout: MemoryLayout | None = None,
    ) -> jax.Array:
        """Convert tensor to target memory layout."""
        if len(x.shape) != 4:
            return x

        if current_layout is None:
            current_layout = self.detect_layout(x)

        if current_layout is None or current_layout == target_layout:
            return x

        # Check cache for this conversion
        cache_key = (current_layout, target_layout, x.shape)
        if cache_key in self.conversion_cache:
            transpose_axes = self.conversion_cache[cache_key]
        else:
            transpose_axes = self._get_transpose_axes(current_layout, target_layout)
            self.conversion_cache[cache_key] = transpose_axes

        if transpose_axes is None:
            return x

        return jnp.transpose(x, transpose_axes)

    def _get_transpose_axes(
        self, current: MemoryLayout, target: MemoryLayout
    ) -> tuple[int, ...] | None:
        """Get transpose axes for layout conversion."""
        conversions = {
            (MemoryLayout.NCHW, MemoryLayout.NHWC): (0, 2, 3, 1),
            (MemoryLayout.NHWC, MemoryLayout.NCHW): (0, 3, 1, 2),
            (MemoryLayout.NCDHW, MemoryLayout.NDHWC): (0, 2, 3, 4, 1),
            (MemoryLayout.NDHWC, MemoryLayout.NCDHW): (0, 4, 1, 2, 3),
        }

        return conversions.get((current, target))

    def optimize_for_operation(
        self, x: jax.Array, operation_type: str = "default"
    ) -> jax.Array:
        """Optimize tensor layout for specific operation type."""
        target_layout = self.optimal_layouts.get(
            operation_type, self.optimal_layouts["default"]
        )
        return self.convert_layout(x, target_layout)

    def get_performance_score(
        self, x: jax.Array, operation_type: str = "default"
    ) -> float:
        """Get performance score for current tensor layout."""
        current_layout = self.detect_layout(x)
        optimal_layout = self.optimal_layouts.get(
            operation_type, self.optimal_layouts["default"]
        )

        if current_layout == optimal_layout:
            return 1.0
        if current_layout is not None:
            return 0.5  # Suboptimal but valid
        return 0.0  # Unknown layout


class OptimizedConvolution(nnx.Module):
    """Convolution layer with automatic layout optimization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, ...],
        strides: int | tuple[int, ...] = 1,
        padding: str = "SAME",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding

        self.layout_optimizer = LayoutOptimizer()

        # Initialize kernel weights
        kernel_shape = (*self.kernel_size, in_features, out_features)
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), kernel_shape) * 0.1)

        # Optional bias
        self.bias = nnx.Param(jnp.zeros(out_features))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply optimized convolution."""
        # Optimize input layout for convolution
        x_optimized = self.layout_optimizer.optimize_for_operation(x, "convolution")

        # Perform convolution with optimal layout
        conv_result = jax.lax.conv_general_dilated(
            x_optimized,
            self.kernel.value,
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

        # Add bias
        return conv_result + self.bias.value


class OptimizedLinear(nnx.Module):
    """Linear layer with layout optimization for TensorCore utilization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.layout_optimizer = LayoutOptimizer()

        # Align dimensions for TensorCore (multiples of 8/16)
        aligned_in = ((in_features + 15) // 16) * 16
        aligned_out = ((out_features + 15) // 16) * 16

        # Initialize weights with TensorCore alignment
        self.kernel = nnx.Param(
            jax.random.normal(rngs.params(), (aligned_in, aligned_out)) * 0.1
        )
        self.bias = nnx.Param(jnp.zeros(aligned_out))

        # Store original dimensions for output trimming
        self.original_out = out_features

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply optimized linear transformation."""
        # Pad input to aligned dimensions
        input_shape = x.shape
        padded_x = jnp.pad(
            x,
            [(0, 0)] * (len(input_shape) - 1)
            + [(0, self.kernel.value.shape[0] - input_shape[-1])],
        )

        # Optimize layout for matrix multiplication
        padded_x = self.layout_optimizer.optimize_for_operation(padded_x, "matmul")

        # Perform matrix multiplication
        result = padded_x @ self.kernel.value + self.bias.value

        # Trim to original output dimensions
        return result[..., : self.original_out]


def optimize_neural_operator_layout(model: nnx.Module) -> nnx.Module:
    """Optimize memory layout for an entire neural operator model."""
    # This would implement automatic layout optimization for the entire model
    # For now, return the model unchanged
    return model


def benchmark_layout_performance(
    x: jax.Array,
    operation: Callable,
    layouts_to_test: list | None = None,
    num_iterations: int = 100,
) -> dict[str, float]:
    """Benchmark performance of different memory layouts."""
    if layouts_to_test is None:
        layouts_to_test = [MemoryLayout.NCHW, MemoryLayout.NHWC]

    optimizer = LayoutOptimizer()
    results = {}

    for layout in layouts_to_test:
        # Convert to target layout
        x_converted = optimizer.convert_layout(x, layout)

        # JIT compile the operation
        jitted_op = jax.jit(operation)

        # Warmup
        _ = jitted_op(x_converted)

        # Benchmark
        import time

        start_time = time.time()

        for _ in range(num_iterations):
            _ = jitted_op(x_converted)

        # Wait for completion
        jax.block_until_ready(_)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations

        results[layout.value] = avg_time

    return results


def create_layout_optimization_report(
    model: nnx.Module,
    sample_input: jax.Array,
) -> dict[str, Any]:
    """Create comprehensive layout optimization report."""
    optimizer = LayoutOptimizer()

    report = {
        "model": model.__class__.__name__,
        "backend": optimizer.backend,
        "input_shape": sample_input.shape,
        "detected_layout": optimizer.detect_layout(sample_input),
        "optimal_layouts": optimizer.optimal_layouts,
        "performance_scores": {},
        "recommendations": [],
    }

    # Analyze performance for different operations
    for op_type in optimizer.optimal_layouts:
        score = optimizer.get_performance_score(sample_input, op_type)
        report["performance_scores"][op_type] = score

        if score < 1.0:
            optimal_layout = optimizer.optimal_layouts[op_type]
            report["recommendations"].append(
                f"Convert to {optimal_layout.value} for {op_type} operations"
            )

    return report


# Export main components
__all__ = [
    "LayoutOptimizer",
    "MemoryLayout",
    "OptimizedConvolution",
    "OptimizedLinear",
    "benchmark_layout_performance",
    "create_layout_optimization_report",
    "optimize_neural_operator_layout",
]
