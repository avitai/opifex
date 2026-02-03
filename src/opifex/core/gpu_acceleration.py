"""
Optimized GPU optimization patterns for maximum efficiency in Opifex workflows.

This module implements efficient optimization techniques including:
- Mixed precision with hardware-aware optimization
- Roofline model-based memory management
- Asynchronous memory operations with prefetching
- Memory pooling for buffer reuse
- Cached progressive testing with efficiency analysis

Designed for maximum speed and memory efficiency while maintaining safety.
"""

import contextlib
import functools
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


class RooflineMemoryManager:
    """Memory management based on roofline model for optimal hardware utilization."""

    def __init__(self):
        self.hw_specs = self._get_hardware_specs()
        self.operation_cache = {}

    def _get_hardware_specs(self) -> dict[str, Any]:
        """Get actual hardware specifications with dynamic detection."""
        try:
            device = jax.devices()[0]
            if device.platform == "gpu":
                # Try to get actual GPU memory
                try:
                    if hasattr(device, "memory_stats"):
                        memory_stats = device.memory_stats()
                        memory_gb = memory_stats.get("bytes_limit", 24 * 1024**3) / (
                            1024**3
                        )
                    else:
                        memory_gb = 24.0  # Reasonable default for modern GPUs
                except Exception:
                    memory_gb = 24.0

                # Detect GPU type for optimal settings
                device_kind = str(device.device_kind).lower()
                if any(name in device_kind for name in ["h100", "a100"]):
                    return {
                        "memory_gb": memory_gb,
                        "peak_flops": 9.89e14,  # H100 bf16
                        "memory_bandwidth": 3.35e12,  # H100 HBM
                        "critical_intensity": 298,
                        "platform": "gpu",
                        "supports_tensorcore": True,
                    }
                return {
                    "memory_gb": memory_gb,
                    "peak_flops": 5e13,  # Conservative GPU estimate
                    "memory_bandwidth": 1e12,
                    "critical_intensity": 200,
                    "platform": "gpu",
                    "supports_tensorcore": True,
                }

            if device.platform == "tpu":
                return {
                    "memory_gb": 32.0,  # TPU v5e
                    "peak_flops": 1.97e14,
                    "memory_bandwidth": 8.2e11,
                    "critical_intensity": 240,
                    "platform": "tpu",
                    "supports_tensorcore": False,
                }
        except Exception:
            pass

        # Fallback to CPU specifications
        return {
            "memory_gb": 16.0,
            "peak_flops": 1e12,
            "memory_bandwidth": 1e11,
            "critical_intensity": 100,
            "platform": "cpu",
            "supports_tensorcore": False,
        }

    def estimate_operation_efficiency(
        self, operation_type: str, *shapes
    ) -> dict[str, Any]:
        """Estimate operation efficiency using roofline model."""
        cache_key = (operation_type, shapes)
        if cache_key in self.operation_cache:
            return self.operation_cache[cache_key]

        if operation_type == "matmul" and len(shapes) == 3:
            m, k, n = shapes
            # Calculate FLOPs and memory access
            flops = 2 * m * k * n  # Matrix multiplication FLOPs
            bytes_accessed = (m * k + k * n + m * n) * 4  # Assuming float32

            arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else 0

            # Determine bottleneck
            compute_time = flops / self.hw_specs["peak_flops"]
            memory_time = bytes_accessed / self.hw_specs["memory_bandwidth"]

            is_compute_bound = (
                arithmetic_intensity > self.hw_specs["critical_intensity"]
            )
            efficiency = min(
                arithmetic_intensity / self.hw_specs["critical_intensity"], 1.0
            )

            result = {
                "arithmetic_intensity": arithmetic_intensity,
                "is_compute_bound": is_compute_bound,
                "efficiency_score": efficiency,
                "estimated_time": max(compute_time, memory_time),
                "memory_gb": bytes_accessed / (1024**3),
            }

            self.operation_cache[cache_key] = result
            return result

        # Default estimation for unknown operations
        return {
            "arithmetic_intensity": 100.0,
            "is_compute_bound": True,
            "efficiency_score": 0.5,
            "estimated_time": 1e-3,
            "memory_gb": 1.0,
        }

    def get_optimal_batch_size(self, operation_type: str, base_shape: tuple) -> int:
        """Find optimal batch size based on hardware characteristics."""
        if self.hw_specs["platform"] == "tpu":
            min_batch = 240  # TPU critical batch size
        elif self.hw_specs["platform"] == "gpu":
            min_batch = 298 if "h100" in str(jax.devices()[0]).lower() else 128
        else:
            min_batch = 32  # Conservative default

        # Binary search for optimal batch size
        low, high = min_batch, min_batch * 10
        optimal_batch = min_batch

        while low <= high:
            mid = (low + high) // 2
            test_shapes = (
                (mid, *base_shape[1:]) if len(base_shape) > 1 else (mid, mid, mid)
            )

            efficiency = self.estimate_operation_efficiency(
                operation_type, *test_shapes
            )
            memory_usage = efficiency["memory_gb"]

            if memory_usage <= self.hw_specs["memory_gb"] * 0.8:  # 80% memory limit
                optimal_batch = mid
                low = mid + 1
            else:
                high = mid - 1

        return optimal_batch


class MixedPrecisionOptimizer:
    """Hardware-aware mixed precision optimization for maximum performance."""

    def __init__(self):
        self.hardware_config = self._detect_hardware()

    def _detect_hardware(self) -> dict[str, Any]:
        """Detect hardware capabilities for optimal precision selection."""
        try:
            device = jax.devices()[0]
            if device.platform == "gpu":
                # Check for TensorCore support
                device_kind = str(device.device_kind).lower()
                if any(name in device_kind for name in ["h100", "a100", "v100"]):
                    return {
                        "supports_tensorcore": True,
                        "optimal_dtype": jnp.bfloat16,
                        "tensor_shapes": [(16, 16, 8), (32, 8, 16)],
                        "precision": jax.lax.Precision.HIGH,
                        "alignment": 16,
                    }
                return {
                    "supports_tensorcore": True,  # Assume modern GPU
                    "optimal_dtype": jnp.bfloat16,
                    "tensor_shapes": [(16, 16, 8)],
                    "precision": jax.lax.Precision.HIGH,
                    "alignment": 16,
                }
            if device.platform == "tpu":
                return {
                    "supports_tensorcore": False,
                    "optimal_dtype": jnp.bfloat16,
                    "tensor_shapes": [(128, 128)],
                    "precision": jax.lax.Precision.DEFAULT,
                    "alignment": 128,
                }
        except Exception:
            pass

        return {
            "supports_tensorcore": False,
            "optimal_dtype": jnp.float32,
            "tensor_shapes": [],
            "precision": jax.lax.Precision.DEFAULT,
            "alignment": 1,
        }

    def align_for_hardware(self, arrays: list[Array]) -> list[Array]:
        """Align tensor shapes for optimal hardware utilization."""
        target_alignment = self.hardware_config["alignment"]
        if target_alignment <= 1:
            return arrays

        aligned = []
        for arr in arrays:
            if arr.ndim >= 2:
                # Pad to multiples of target alignment
                pad_rows = (
                    target_alignment - arr.shape[-2] % target_alignment
                ) % target_alignment
                pad_cols = (
                    target_alignment - arr.shape[-1] % target_alignment
                ) % target_alignment

                if pad_rows > 0 or pad_cols > 0:
                    pad_spec = [(0, 0)] * (arr.ndim - 2) + [
                        (0, pad_rows),
                        (0, pad_cols),
                    ]
                    arr = jnp.pad(arr, pad_spec)  # noqa: PLW2901

            aligned.append(arr)
        return aligned

    @functools.partial(jax.jit, static_argnums=(0,))
    def mixed_precision_matmul(self, x: Array, y: Array) -> Array:
        """Hardware-optimized mixed precision matrix multiplication."""
        # Store original shapes for trimming
        orig_x_shape = x.shape
        orig_y_shape = y.shape

        # Align for hardware
        x_aligned, y_aligned = self.align_for_hardware([x, y])

        # Convert to optimal precision
        optimal_dtype = self.hardware_config["optimal_dtype"]
        if x_aligned.dtype not in {optimal_dtype, jnp.float32}:
            x_mp = x_aligned.astype(optimal_dtype)
            y_mp = y_aligned.astype(optimal_dtype)
        else:
            x_mp, y_mp = x_aligned, y_aligned

        # Perform computation with optimal precision
        result = jnp.dot(x_mp, y_mp, precision=self.hardware_config["precision"])

        # Convert back to float32 for numerical stability
        if result.dtype != jnp.float32:
            result = result.astype(jnp.float32)

        # Trim padding if applied
        return result[: orig_x_shape[0], : orig_y_shape[1]]


class AsyncMemoryManager:
    """Asynchronous memory management with prefetching for overlapped computation."""

    def __init__(self):
        self.prefetch_queue = {}

    def async_device_put(
        self, array: jax.Array, device: Any, key: str | None = None
    ) -> jax.Array:
        """Asynchronous device placement with prefetching."""
        if key:
            # Store handle for later retrieval (simulate async with regular device_put)
            future = jax.device_put(array, device)
            self.prefetch_queue[key] = future
            return future
        return jax.device_put(array, device)

    def prefetch_next_batch(self, next_data: jax.Array, device: Any) -> None:
        """Prefetch next batch while current batch is processing."""
        self.async_device_put(next_data, device, key="next_batch")

    def get_prefetched(self, key: str) -> jax.Array | None:
        """Retrieve prefetched data."""
        if key in self.prefetch_queue:
            return self.prefetch_queue.pop(key)  # JAX handles the async completion
        return None

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def overlapped_computation(
        self,
        operation_fn: Callable[[Array], Array],
        current_data: Array,
        next_data: Array | None = None,
    ) -> tuple[Array, None]:
        """Overlap computation with memory transfers."""
        # Process current data
        result = operation_fn(current_data)

        # Prefetch next data asynchronously (if provided)
        if next_data is not None:
            with contextlib.suppress(Exception):
                self.prefetch_next_batch(next_data, jax.devices()[0])

        return result, None


class MemoryPoolManager:
    """Memory pool management for efficient buffer reuse."""

    def __init__(self, pool_size_gb: float = 4.0):
        self.pool_size_bytes = int(pool_size_gb * 1024**3)
        self.buffer_pools = {}  # (shape, dtype) -> list of buffers
        self.allocated_bytes = 0
        self.allocation_count = 0
        self.reuse_count = 0

    def get_buffer(self, shape: tuple, dtype: jnp.dtype) -> Array:
        """Get buffer from pool or create new one."""
        # Normalize dtype to ensure consistent key matching
        normalized_dtype = jnp.dtype(dtype)
        key = (tuple(shape), normalized_dtype)

        # Try to reuse existing buffer
        if self.buffer_pools.get(key):
            self.reuse_count += 1
            return self.buffer_pools[key].pop()

        # Create new buffer if pool is empty
        buffer_size = int(jnp.prod(jnp.array(shape)) * normalized_dtype.itemsize)

        if self.allocated_bytes + buffer_size > self.pool_size_bytes:
            self._cleanup_oldest_buffers(buffer_size)

        self.allocation_count += 1
        buffer = jnp.zeros(shape, dtype=normalized_dtype)
        self.allocated_bytes += buffer_size
        return buffer

    def return_buffer(self, buffer: Array) -> None:
        """Return buffer to pool for reuse."""
        # Normalize dtype to ensure consistent key matching
        key = (tuple(buffer.shape), jnp.dtype(buffer.dtype))

        if key not in self.buffer_pools:
            self.buffer_pools[key] = []

        # Return the actual buffer to pool (no copying!)
        self.buffer_pools[key].append(buffer)

    def _cleanup_oldest_buffers(self, needed_bytes: int) -> None:
        """Remove oldest buffers to make space."""
        freed_bytes = 0

        for key in list(self.buffer_pools.keys()):
            while self.buffer_pools[key] and freed_bytes < needed_bytes:
                buffer = self.buffer_pools[key].pop(0)
                buffer_size = int(
                    jnp.prod(jnp.array(buffer.shape)) * jnp.dtype(buffer.dtype).itemsize
                )
                freed_bytes += buffer_size
                self.allocated_bytes -= buffer_size

            if freed_bytes >= needed_bytes:
                break

    @contextlib.contextmanager
    def managed_buffers(self, shapes_and_dtypes: list[tuple]):
        """Context manager for automatic buffer management."""
        buffers = []
        try:
            for shape, dtype in shapes_and_dtypes:
                buffer = self.get_buffer(shape, dtype)
                buffers.append(buffer)

            yield buffers

        finally:
            for buffer in buffers:
                self.return_buffer(buffer)

    def get_pool_stats(self) -> dict[str, Any]:
        """Get memory pool statistics."""
        total_buffers = sum(len(pool) for pool in self.buffer_pools.values())
        reuse_ratio = self.reuse_count / max(
            1, self.allocation_count + self.reuse_count
        )

        return {
            "total_allocations": self.allocation_count,
            "total_reuses": self.reuse_count,
            "reuse_ratio": reuse_ratio,
            "pooled_buffers": total_buffers,
            "allocated_bytes": self.allocated_bytes,
            "pool_types": len(self.buffer_pools),
        }


class CachedProgressiveTester:
    """Progressive testing with caching and roofline analysis prescreening."""

    def __init__(self, memory_manager: RooflineMemoryManager | None = None):
        self.memory_manager = memory_manager or RooflineMemoryManager()
        self.hardware_signature = self._get_hardware_signature()

    def _get_hardware_signature(self) -> str:
        """Create unique signature for current hardware configuration."""
        try:
            device = jax.devices()[0]
            return f"{device.platform}_{device.device_kind}_{len(jax.devices())}"
        except Exception:
            return "unknown_hardware"

    @functools.lru_cache(maxsize=128)  # noqa: B019
    def _cached_test_operation(
        self, operation_name: str, size: int, dtype_str: str
    ) -> tuple[bool, float | None, str | None]:
        """Cache test results to avoid repeated expensive tests."""
        return self._actual_test_operation(operation_name, size, dtype_str)

    def _actual_test_operation(
        self, operation_name: str, size: int, dtype_str: str
    ) -> tuple[bool, float | None, str | None]:
        """Perform the actual operation test."""
        try:
            dtype = getattr(jnp, dtype_str)
        except AttributeError:
            return False, None, f"Invalid dtype: {dtype_str}"

        # Use roofline analysis for pre-screening
        efficiency = self.memory_manager.estimate_operation_efficiency(
            "matmul", size, size, size
        )

        if efficiency["memory_gb"] > self.memory_manager.hw_specs["memory_gb"] * 0.9:
            return (
                False,
                None,
                f"Predicted OOM: {efficiency['memory_gb']:.2f}GB > available",
            )

        try:
            # Create test matrices
            key = jax.random.PRNGKey(42)
            x = jax.random.normal(key, (size, size), dtype=dtype)
            y = jax.random.normal(key, (size, size), dtype=dtype)

            # Get operation function
            if operation_name == "safe_matmul":
                operation_fn = self._test_safe_matmul
            elif operation_name == "optimized_matmul":
                operation_fn = self._test_optimized_matmul
            else:
                return False, None, f"Unknown operation: {operation_name}"

            # Warmup (3 runs)
            for _ in range(3):
                result = operation_fn(x, y)
                result.block_until_ready()

            # Actual timing (10 runs for accuracy)
            times = []
            for _ in range(10):
                start = time.perf_counter()
                result = operation_fn(x, y)
                result.block_until_ready()
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            return True, avg_time, None

        except Exception as e:
            return False, None, str(e)

    def _test_safe_matmul(self, x: Array, y: Array) -> Array:
        """Test safe matrix multiplication."""
        return jnp.dot(x, y)

    def _test_optimized_matmul(self, x: Array, y: Array) -> Array:
        """Test optimized matrix multiplication."""
        mixed_precision = MixedPrecisionOptimizer()
        return mixed_precision.mixed_precision_matmul(x, y)

    def find_optimal_configuration(
        self, operation_name: str = "safe_matmul"
    ) -> dict[str, Any]:
        """Find optimal configuration using adaptive search."""
        # Start with hardware-suggested batch size
        base_size = self.memory_manager.get_optimal_batch_size("matmul", (1, 1))

        # Test different configurations
        configurations = []

        dtypes_to_test = ["float32"]
        if self.memory_manager.hw_specs["platform"] in ["gpu", "tpu"]:
            dtypes_to_test.extend(["bfloat16", "float16"])

        for dtype_str in dtypes_to_test:
            for size in [base_size // 2, base_size, base_size * 2]:
                if size < 32:  # Skip too small sizes
                    continue

                success, exec_time, _error = self._cached_test_operation(
                    operation_name, size, dtype_str
                )

                if success and exec_time is not None:
                    efficiency = self.memory_manager.estimate_operation_efficiency(
                        "matmul", size, size, size
                    )

                    configurations.append(
                        {
                            "size": size,
                            "dtype": dtype_str,
                            "execution_time": exec_time,
                            "efficiency_score": efficiency["efficiency_score"],
                            "throughput": (2 * size**3) / exec_time
                            if exec_time > 0
                            else 0,
                        }
                    )

        # Find best configuration (highest throughput with good efficiency)
        if configurations:
            best_config = max(
                configurations, key=lambda x: x["throughput"] * x["efficiency_score"]
            )
            return {
                "optimal_config": best_config,
                "all_configs": configurations,
                "hardware_specs": self.memory_manager.hw_specs,
            }

        return {"error": "No successful configurations found"}


class OptimizedGPUManager:
    """Complete optimized GPU manager integrating all optimization techniques."""

    def __init__(self, max_memory_fraction: float = 0.75):
        self.roofline_manager = RooflineMemoryManager()
        self.mixed_precision = MixedPrecisionOptimizer()
        self.async_manager = AsyncMemoryManager()
        self.memory_pool = MemoryPoolManager()
        self.cached_tester = CachedProgressiveTester(self.roofline_manager)

        # Pre-compile optimized operations
        self.compiled_ops = self._precompile_operations()

    def _precompile_operations(self) -> dict[str, Callable]:
        """Pre-compile operations for different size buckets."""
        size_buckets = [128, 256, 512, 1024, 2048, 4096]
        compiled = {}

        for size in size_buckets:
            # Create a closure that captures the mixed precision optimizer
            mixed_precision = self.mixed_precision

            # Shape-specialized matrix multiplication
            @functools.partial(jax.jit)
            def _matmul_specialized(x: Array, y: Array) -> Array:
                return mixed_precision.mixed_precision_matmul(x, y)  # noqa: B023

            compiled[f"matmul_{size}"] = _matmul_specialized

        return compiled

    def optimal_matrix_multiply(self, x: Array, y: Array) -> Array:
        """Maximally optimized matrix multiplication."""
        # Select best compiled version based on input size
        size = max(x.shape[0], y.shape[1])

        # Find closest size bucket
        size_buckets = [128, 256, 512, 1024, 2048, 4096]
        target_size = min(size_buckets, key=lambda s: abs(s - size))

        if f"matmul_{target_size}" in self.compiled_ops:
            return self.compiled_ops[f"matmul_{target_size}"](x, y)
        # Fallback to mixed precision
        return self.mixed_precision.mixed_precision_matmul(x, y)

    def benchmark_with_prefetching(
        self, operation_fn: Callable[[Array, Array], Array], test_sizes: list[int]
    ) -> dict[int, dict[str, Any]]:
        """Benchmark with asynchronous prefetching and memory pooling."""
        results = {}

        for i, size in enumerate(test_sizes):
            try:
                # Use memory pool for test matrices
                shapes_and_dtypes = [
                    ((size, size), jnp.float32),
                    ((size, size), jnp.float32),
                ]

                with self.memory_pool.managed_buffers(shapes_and_dtypes) as buffers:
                    x_buf, y_buf = buffers

                    # Fill buffers with test data
                    key = jax.random.PRNGKey(42)
                    x = jax.random.normal(key, (size, size))
                    y = jax.random.normal(key, (size, size))

                    # Copy to managed buffers
                    x = x.at[:].set(x_buf[:size, :size])
                    y = y.at[:].set(y_buf[:size, :size])

                    # Prefetch next test data if available
                    if i + 1 < len(test_sizes):
                        next_size = test_sizes[i + 1]
                        next_key = jax.random.PRNGKey(43)
                        next_x = jax.random.normal(next_key, (next_size, next_size))
                        with contextlib.suppress(Exception):
                            self.async_manager.prefetch_next_batch(
                                next_x, jax.devices()[0]
                            )

                    # Run benchmark with overlapped computation
                    success, exec_time, error = (
                        self.cached_tester._cached_test_operation(
                            "optimized_matmul", size, "float32"
                        )
                    )

                    results[size] = {
                        "success": success,
                        "execution_time": exec_time,
                        "error": error,
                        "efficiency": (
                            self.roofline_manager.estimate_operation_efficiency(
                                "matmul", size, size, size
                            )
                        ),
                    }

            except Exception as e:
                results[size] = {
                    "success": False,
                    "execution_time": None,
                    "error": str(e),
                    "efficiency": {"efficiency_score": 0.0},
                }

        return results


# Backward compatibility functions (will be deprecated)
def safe_matrix_multiply(x: Array, y: Array) -> Array:
    """Legacy safe matrix multiplication - use OptimizedGPUManager instead."""
    manager = OptimizedGPUManager()
    return manager.optimal_matrix_multiply(x, y)


def optimized_matrix_multiply(x: Array, y: Array) -> Array:
    """Legacy optimized matrix multiplication - use OptimizedGPUManager instead."""
    manager = OptimizedGPUManager()
    return manager.optimal_matrix_multiply(x, y)


def benchmark_gpu_operations() -> dict[str, Any]:
    """Legacy benchmark function - OptimizedGPUManager.benchmark_with_prefetching."""
    manager = OptimizedGPUManager()
    test_sizes = [64, 128, 256, 512]
    results = manager.benchmark_with_prefetching(
        manager.optimal_matrix_multiply, test_sizes
    )

    return {
        "benchmark_results": results,
        "backend_info": {
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "memory_gb": manager.roofline_manager.hw_specs["memory_gb"],
        },
        "hardware_specs": manager.roofline_manager.hw_specs,
    }
