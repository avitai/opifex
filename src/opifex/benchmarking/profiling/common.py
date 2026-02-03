"""
Shared utilities for Opifex profiling.

Contains common hardware specifications and timing functions to ensure
consistency and adherence to DRY principles across profilers.
"""

import time
from collections.abc import Callable
from typing import Any

import jax


# Hardware specifications for common accelerators
HARDWARE_SPECS = {
    "tpu_v5e": {
        "peak_flops": 197.0e12,  # 197 TFLOPS (bf16)
        "peak_flops_bf16": 197.0e12,
        "memory_bandwidth": 1600.0e9,  # 1.6 TB/s
        "critical_intensity": 123.125,  # FLOPs/byte
    },
    "a100_80g": {
        "peak_flops": 312.0e12,  # 312 TFLOPS (bf16/fp16)
        "peak_flops_bf16": 312.0e12,
        "memory_bandwidth": 2039.0e9,  # 2.0 TB/s
        "critical_intensity": 153.0,
        "tensor_core_shapes": [(16, 16, 16), (16, 16, 8)],
    },
    "h100": {
        "peak_flops": 989.0e12,  # 989 TFLOPS (bf16/fp16)
        "peak_flops_bf16": 989.0e12,
        "memory_bandwidth": 3350.0e9,  # 3.35 TB/s
        "critical_intensity": 295.0,
        "tensor_core_shapes": [(16, 16, 16)],
    },
    "cpu_generic": {
        "peak_flops": 2.0e12,  # ~2 TFLOPS (optimistic)
        "peak_flops_bf16": 2.0e12,
        "memory_bandwidth": 200.0e9,  # ~200 GB/s
        "critical_intensity": 10.0,
        "simd_width": 8,
    },
}


def detect_hardware_specs() -> dict[str, Any]:
    """Detect current hardware and return appropriate specifications."""
    backend = jax.default_backend()

    if backend == "tpu":
        # Assume TPU v5e for now - could be enhanced with device detection
        return HARDWARE_SPECS["tpu_v5e"]
    if backend == "gpu":
        # Assume A100 for now
        return HARDWARE_SPECS["a100_80g"]

    return HARDWARE_SPECS["cpu_generic"]


def measure_execution_time(
    func: Callable,
    inputs: list[jax.Array],
    warmup: int = 3,
    iterations: int = 10,
) -> float:
    """
    Measure execution time of a JAX function with proper synchronization.

    Args:
        func: Function to benchmark
        inputs: Input arguments
        warmup: Number of warmup iterations
        iterations: Number of timing iterations

    Returns:
        Average execution time in seconds
    """
    compiled_func = jax.jit(func)

    # Warmup
    for _ in range(warmup):
        result = compiled_func(*inputs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, tuple | list):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()

    # Timing
    start_time = time.time()
    for _ in range(iterations):
        result = compiled_func(*inputs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, tuple | list):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()

    total_time = time.time() - start_time
    return total_time / iterations
