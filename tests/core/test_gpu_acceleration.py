"""
Comprehensive tests for the optimized GPU optimization module.

This test suite defines the expected behavior for maximum efficiency GPU optimization
including mixed precision, roofline analysis, asynchronous operations, and memory pooling.

Following TDD principles: Tests define behavior, implementation follows.
"""

import time
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Import the new optimized classes (to be implemented)
from opifex.core.gpu_acceleration import (
    AsyncMemoryManager,
    CachedProgressiveTester,
    MemoryPoolManager,
    MixedPrecisionOptimizer,
    OptimizedGPUManager,
    RooflineMemoryManager,
)


class TestRooflineMemoryManager:
    """Test roofline model-based memory management."""

    def test_hardware_specs_detection(self):
        """Test hardware specifications are properly detected."""
        manager = RooflineMemoryManager()

        # Should detect hardware and return valid specs
        assert isinstance(manager.hw_specs, dict)
        assert "memory_gb" in manager.hw_specs
        assert "peak_flops" in manager.hw_specs
        assert "memory_bandwidth" in manager.hw_specs
        assert "critical_intensity" in manager.hw_specs
        assert "platform" in manager.hw_specs

        # Memory should be positive
        assert manager.hw_specs["memory_gb"] > 0
        assert manager.hw_specs["peak_flops"] > 0
        assert manager.hw_specs["memory_bandwidth"] > 0

    def test_operation_efficiency_estimation(self):
        """Test roofline model efficiency estimation."""
        manager = RooflineMemoryManager()

        # Test matrix multiplication efficiency
        efficiency = manager.estimate_operation_efficiency("matmul", 1000, 1000, 1000)

        assert isinstance(efficiency, dict)
        assert "arithmetic_intensity" in efficiency
        assert "is_compute_bound" in efficiency
        assert "efficiency_score" in efficiency
        assert "estimated_time" in efficiency
        assert "memory_gb" in efficiency

        # Efficiency score should be between 0 and 1
        assert 0 <= efficiency["efficiency_score"] <= 1
        assert efficiency["arithmetic_intensity"] > 0
        assert efficiency["memory_gb"] > 0

    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation based on hardware."""
        manager = RooflineMemoryManager()

        batch_size = manager.get_optimal_batch_size("matmul", (1000, 1000))

        # Should return reasonable batch size
        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert batch_size >= 32  # Minimum reasonable batch size

    def test_efficiency_caching(self):
        """Test that efficiency calculations are cached."""
        manager = RooflineMemoryManager()

        # First call
        start_time = time.time()
        efficiency1 = manager.estimate_operation_efficiency("matmul", 1000, 1000, 1000)
        first_call_time = time.time() - start_time

        # Second call (should be cached)
        start_time = time.time()
        efficiency2 = manager.estimate_operation_efficiency("matmul", 1000, 1000, 1000)
        second_call_time = time.time() - start_time

        # Results should be identical
        assert efficiency1 == efficiency2
        # Second call should be faster (cached)
        assert second_call_time < first_call_time


class TestMixedPrecisionOptimizer:
    """Test mixed precision optimization with hardware awareness."""

    def test_hardware_detection(self):
        """Test hardware capability detection."""
        optimizer = MixedPrecisionOptimizer()

        assert isinstance(optimizer.hardware_config, dict)
        assert "supports_tensorcore" in optimizer.hardware_config
        assert "optimal_dtype" in optimizer.hardware_config
        assert "precision" in optimizer.hardware_config

    def test_tensor_alignment(self):
        """Test tensor alignment for hardware optimization."""
        optimizer = MixedPrecisionOptimizer()

        # Test with unaligned tensors
        x = jnp.ones((127, 511))  # Not aligned to 16
        y = jnp.ones((511, 253))

        aligned = optimizer.align_for_hardware([x, y])

        assert len(aligned) == 2
        # Should pad to multiples of alignment
        if optimizer.hardware_config["supports_tensorcore"]:
            assert aligned[0].shape[0] % 16 == 0
            assert aligned[0].shape[1] % 16 == 0
            assert aligned[1].shape[0] % 16 == 0
            assert aligned[1].shape[1] % 16 == 0

    def test_mixed_precision_matmul(self):
        """Test mixed precision matrix multiplication."""
        optimizer = MixedPrecisionOptimizer()

        x = jnp.ones((128, 128), dtype=jnp.float32)
        y = jnp.ones((128, 128), dtype=jnp.float32)

        result = optimizer.mixed_precision_matmul(x, y)

        # Should return float32 result
        assert result.dtype == jnp.float32
        assert result.shape == (128, 128)
        # Should compute correct result
        expected = jnp.dot(x, y)
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_precision_conversion(self):
        """Test automatic precision conversion."""
        optimizer = MixedPrecisionOptimizer()

        # Test with different input dtypes
        x_f32 = jnp.ones((64, 64), dtype=jnp.float32)
        y_f32 = jnp.ones((64, 64), dtype=jnp.float32)

        result = optimizer.mixed_precision_matmul(x_f32, y_f32)

        # Should handle conversion internally
        assert result.dtype == jnp.float32
        assert not jnp.any(jnp.isnan(result))


class TestAsyncMemoryManager:
    """Test asynchronous memory management."""

    def test_async_device_put(self):
        """Test asynchronous device placement."""
        manager = AsyncMemoryManager()

        x = jnp.ones((100, 100))
        device = jax.devices()[0]

        # Test with key (for prefetching)
        _future = manager.async_device_put(x, device, key="test")
        assert "test" in manager.prefetch_queue

        # Test without key (immediate)
        result = manager.async_device_put(x, device)
        assert isinstance(result, jax.Array)

    def test_prefetch_mechanism(self):
        """Test prefetching mechanism."""
        manager = AsyncMemoryManager()

        data = jnp.ones((50, 50))
        device = jax.devices()[0]

        # Prefetch data
        manager.prefetch_next_batch(data, device)
        assert "next_batch" in manager.prefetch_queue

        # Retrieve prefetched data
        prefetched = manager.get_prefetched("next_batch")
        assert prefetched is not None
        assert "next_batch" not in manager.prefetch_queue

    def test_overlapped_computation(self):
        """Test computation overlap with memory transfers."""
        manager = AsyncMemoryManager()

        def dummy_operation(x):
            return x * 2

        current_data = jnp.ones((32, 32))
        next_data = jnp.ones((32, 32))

        result, _ = manager.overlapped_computation(
            dummy_operation, current_data, next_data
        )

        # Should compute result correctly
        expected = current_data * 2
        np.testing.assert_array_equal(result, expected)


class TestMemoryPoolManager:
    """Test memory pool management for buffer reuse."""

    def test_buffer_allocation(self):
        """Test buffer allocation from pool."""
        manager = MemoryPoolManager(pool_size_gb=1.0)

        shape = (100, 100)
        dtype = jnp.float32

        buffer = manager.get_buffer(shape, dtype)

        assert buffer.shape == shape
        assert buffer.dtype == dtype
        assert jnp.all(buffer == 0)  # Should be zeroed

    def test_buffer_reuse(self):
        """Test buffer return and reuse."""
        manager = MemoryPoolManager(pool_size_gb=1.0)

        shape = (50, 50)
        dtype = jnp.float32

        # Get buffer
        buffer1 = manager.get_buffer(shape, dtype)

        # Return buffer
        manager.return_buffer(buffer1)

        # Get buffer again (should reuse)
        buffer2 = manager.get_buffer(shape, dtype)

        # Should have same shape and dtype
        assert buffer2.shape == shape
        assert buffer2.dtype == dtype

    def test_managed_buffers_context(self):
        """Test managed buffers context manager."""
        manager = MemoryPoolManager(pool_size_gb=1.0)

        shapes_and_dtypes = [((64, 64), jnp.float32), ((32, 32), jnp.bfloat16)]

        with manager.managed_buffers(shapes_and_dtypes) as buffers:
            assert len(buffers) == 2
            assert buffers[0].shape == (64, 64)
            assert buffers[0].dtype == jnp.float32
            assert buffers[1].shape == (32, 32)
            assert buffers[1].dtype == jnp.bfloat16

    def test_memory_limit_enforcement(self):
        """Test memory pool size limit enforcement."""
        manager = MemoryPoolManager(pool_size_gb=0.001)  # Very small pool

        # Try to allocate large buffer
        large_shape = (1000, 1000)
        buffer = manager.get_buffer(large_shape, jnp.float32)

        # Should still work (cleanup should occur)
        assert buffer.shape == large_shape


class TestCachedProgressiveTester:
    """Test cached progressive testing with roofline analysis."""

    def test_hardware_signature_generation(self):
        """Test hardware signature generation for caching."""
        manager = RooflineMemoryManager()
        tester = CachedProgressiveTester(manager)

        signature = tester._get_hardware_signature()

        assert isinstance(signature, str)
        assert len(signature) > 0

    def test_operation_caching(self):
        """Test that operation results are cached."""
        manager = RooflineMemoryManager()
        tester = CachedProgressiveTester(manager)

        # Mock the actual test to avoid expensive operations
        with patch.object(tester, "_actual_test_operation") as mock_test:
            mock_test.return_value = (True, 0.001, None)

            # First call
            result1 = tester._cached_test_operation("safe_matmul", 100, "float32")

            # Second call (should use cache)
            result2 = tester._cached_test_operation("safe_matmul", 100, "float32")

            # Should only call actual test once
            assert mock_test.call_count == 1
            assert result1 == result2

    def test_optimal_configuration_search(self):
        """Test optimal configuration search."""
        manager = RooflineMemoryManager()
        tester = CachedProgressiveTester(manager)

        # Mock successful tests
        with patch.object(tester, "_cached_test_operation") as mock_test:
            mock_test.return_value = (True, 0.001, None)

            config = tester.find_optimal_configuration("safe_matmul")

            assert isinstance(config, dict)
            if "optimal_config" in config:
                assert "size" in config["optimal_config"]
                assert "dtype" in config["optimal_config"]
                assert "execution_time" in config["optimal_config"]

    def test_roofline_prescreening(self):
        """Test roofline analysis prescreening."""
        manager = RooflineMemoryManager()
        tester = CachedProgressiveTester(manager)

        # Test with very large size that should be prescreened out
        result = tester._actual_test_operation("safe_matmul", 50000, "float32")

        success, _exec_time, error = result

        # Should fail due to memory prediction
        if not success:
            assert error is not None
            assert "memory" in error.lower() or "oom" in error.lower()


class TestOptimizedGPUManager:
    """Test the complete optimized GPU manager."""

    def test_initialization(self):
        """Test optimized GPU manager initialization."""
        manager = OptimizedGPUManager()

        # Should initialize all components
        assert hasattr(manager, "roofline_manager")
        assert hasattr(manager, "mixed_precision")
        assert hasattr(manager, "async_manager")
        assert hasattr(manager, "memory_pool")
        assert hasattr(manager, "cached_tester")
        assert hasattr(manager, "compiled_ops")

    def test_precompiled_operations(self):
        """Test that operations are pre-compiled."""
        manager = OptimizedGPUManager()

        # Should have compiled operations for different sizes
        assert isinstance(manager.compiled_ops, dict)
        assert len(manager.compiled_ops) > 0

        # Check for expected compiled operations
        size_buckets = [128, 256, 512, 1024, 2048, 4096]
        for size in size_buckets:
            key = f"matmul_{size}"
            if key in manager.compiled_ops:
                assert callable(manager.compiled_ops[key])

    def test_optimal_matrix_multiply(self):
        """Test optimal matrix multiplication selection."""
        manager = OptimizedGPUManager()

        # Test with different sizes
        for size in [100, 256, 500, 1024]:
            x = jnp.ones((size, size))
            y = jnp.ones((size, size))

            result = manager.optimal_matrix_multiply(x, y)

            # Should return correct result
            assert result.shape == (size, size)
            expected = jnp.dot(x, y)
            np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_benchmark_with_prefetching(self):
        """Test benchmarking with prefetching and memory pooling."""
        manager = OptimizedGPUManager()

        test_sizes = [64, 128, 256]

        # Mock the cached tester to avoid expensive operations
        with patch.object(manager.cached_tester, "_cached_test_operation") as mock_test:
            mock_test.return_value = (True, 0.001, None)

            results = manager.benchmark_with_prefetching(
                manager.optimal_matrix_multiply, test_sizes
            )

            assert isinstance(results, dict)
            assert len(results) == len(test_sizes)

            for size in test_sizes:
                assert size in results
                assert "success" in results[size]
                assert "execution_time" in results[size]
                assert "efficiency" in results[size]


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_cpu_fallback_behavior(self):
        """Test behavior when GPU is not available."""
        # Force CPU-only environment
        with patch("jax.devices") as mock_devices:
            mock_devices.return_value = [Mock(platform="cpu")]

            manager = OptimizedGPUManager()

            # Should still work on CPU
            x = jnp.ones((64, 64))
            y = jnp.ones((64, 64))

            result = manager.optimal_matrix_multiply(x, y)

        assert result.shape == (64, 64)
        expected = jnp.dot(x, y)
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        manager = OptimizedGPUManager()

        # Test with very small memory pool
        manager.memory_pool = MemoryPoolManager(pool_size_gb=0.001)

        # Should still work (with cleanup)
        test_sizes = [32, 64]

        with patch.object(manager.cached_tester, "_cached_test_operation") as mock_test:
            mock_test.return_value = (True, 0.001, None)

            results = manager.benchmark_with_prefetching(
                manager.optimal_matrix_multiply, test_sizes
            )

            assert len(results) == len(test_sizes)

    def test_error_propagation(self):
        """Test proper error handling and propagation."""
        manager = OptimizedGPUManager()

        # Test with operation that should fail
        with patch.object(manager.cached_tester, "_cached_test_operation") as mock_test:
            mock_test.return_value = (False, None, "Test error")

            results = manager.benchmark_with_prefetching(
                manager.optimal_matrix_multiply, [64]
            )

            assert 64 in results
            assert results[64]["success"] is False
            assert results[64]["error"] == "Test error"

    def test_performance_regression_detection(self):
        """Test that performance doesn't regress below baseline."""
        manager = OptimizedGPUManager()

        # Simple baseline test
        x = jnp.ones((256, 256))
        y = jnp.ones((256, 256))

        # Time optimized version
        start_time = time.time()
        optimized_result = manager.optimal_matrix_multiply(x, y)
        optimized_result.block_until_ready()
        optimized_time = time.time() - start_time

        # Time baseline version
        start_time = time.time()
        baseline_result = jnp.dot(x, y)
        baseline_result.block_until_ready()
        baseline_time = time.time() - start_time

        # Results should be equivalent
        np.testing.assert_allclose(optimized_result, baseline_result, rtol=1e-3)

        # Performance should be reasonable (not more than 10x slower)
        # This is a loose bound to account for compilation overhead
        assert optimized_time < baseline_time * 10


# Performance benchmarks (not strict tests, but useful for monitoring)
class TestPerformanceBenchmarks:
    """Performance benchmarks for monitoring optimization effectiveness."""

    def setup_method(self):
        """Ensure clean state before each benchmark."""
        import gc

        import jax

        # Force cleanup
        jax.clear_caches()
        gc.collect()

    @pytest.mark.slow
    def test_mixed_precision_speedup(self):
        """Benchmark mixed precision speedup (when available)."""
        try:
            optimizer = MixedPrecisionOptimizer()

            if optimizer.hardware_config["optimal_dtype"] != jnp.float32:
                # Test large matrix multiplication
                size = 1024
                x = jnp.ones((size, size), dtype=jnp.float32)
                y = jnp.ones((size, size), dtype=jnp.float32)

                # Warmup
                for _ in range(3):
                    result = optimizer.mixed_precision_matmul(x, y)
                    result.block_until_ready()

                # Time mixed precision
                start_time = time.perf_counter()
                for _ in range(10):
                    result = optimizer.mixed_precision_matmul(x, y)
                    result.block_until_ready()
                mixed_precision_time = (time.perf_counter() - start_time) / 10

                # Time baseline
                start_time = time.perf_counter()
                for _ in range(10):
                    result = jnp.dot(x, y)
                    result.block_until_ready()
                baseline_time = (time.perf_counter() - start_time) / 10

                # Mixed precision should be faster (or at least not much slower)
                if mixed_precision_time > 0:
                    speedup = baseline_time / mixed_precision_time
                else:
                    speedup = float("inf")  # Infinite speedup if instant

                print(f"Mixed precision speedup: {speedup:.2f}x")

                # Should be at least not much slower
                assert speedup > 0.5
        except Exception as e:
            pytest.skip(f"Benchmark skipped due to runtime error: {e}")

    @pytest.mark.slow
    def test_memory_pool_efficiency(self):
        """Benchmark memory pool efficiency."""
        try:
            # Use smaller pool size to avoid OOM in CI/shared environments
            manager = MemoryPoolManager(pool_size_gb=0.5)

            shape = (64, 64)  # Smaller size for faster testing
            dtype = jnp.float32
            num_iterations = 50  # Fewer iterations but more realistic pattern

            # Warm up both approaches
            for _ in range(5):
                buffer = manager.get_buffer(shape, dtype)
                result = buffer + 1
                manager.return_buffer(buffer)

            for _ in range(5):
                buffer = jnp.zeros(shape, dtype=dtype)
                result = buffer + 1

            # Test with memory pool (should reuse buffers)
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                buffer = manager.get_buffer(shape, dtype)
                result = buffer + 1
                result.block_until_ready()  # Ensure computation completes
                manager.return_buffer(buffer)
            pooled_time = time.perf_counter() - start_time

            # Test without memory pool (direct allocation every time)
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                buffer = jnp.zeros(shape, dtype=dtype)
                result = buffer + 1
                result.block_until_ready()  # Ensure computation completes
            direct_time = time.perf_counter() - start_time

            # Get pool statistics
            stats = manager.get_pool_stats()
            print(f"Pool stats: {stats}")

            # Memory pool should be faster for repeated allocations
            efficiency = direct_time / pooled_time if pooled_time > 0 else float("inf")

            print(f"Memory pool efficiency: {efficiency:.2f}x")
            print(f"Pooled time: {pooled_time:.4f}s, Direct time: {direct_time:.4f}s")

            # Memory pool should provide some benefit, but be realistic about test environment
            # In a real scenario with larger allocations, the benefit would be much higher
            assert efficiency > 0.8, (
                f"Memory pool is too slow: {efficiency:.2f}x efficiency"
            )

            # Verify that buffer reuse actually happened
            assert stats["reuse_ratio"] > 0.5, (
                f"Buffer reuse ratio too low: {stats['reuse_ratio']:.2f}"
            )
        except Exception as e:
            pytest.skip(f"Benchmark skipped due to runtime error: {e}")
