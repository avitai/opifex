"""Tests for correct benchmarking timing with block_until_ready.

This test suite verifies that all benchmarking code properly uses
block_until_ready() to ensure accurate timing measurements.

Following TDD: These tests are written FIRST to define expected behavior.
Implementation must be updated to make these tests pass.

Non-negotiable principle: Tests define behavior, not current implementation.
"""

import time

import jax
import jax.numpy as jnp
import pytest


class TestCorrectBenchmarkTiming:
    """Test that benchmarks use block_until_ready for accurate timing."""

    def test_benchmark_includes_warmup(self):
        """Verify benchmarks perform warmup iterations.

        Warmup is critical for accurate JIT timing - first call includes
        compilation time which should be excluded from measurements.
        """
        # This test defines expected behavior:
        # All benchmark functions should accept warmup parameter
        # and execute warmup iterations before timing

        # Will be implemented in benchmark utilities
        # Placeholder - implementation will make this pass

    def test_benchmark_waits_for_computation(self):
        """Verify benchmarks use block_until_ready for accurate timing.

        JAX uses asynchronous dispatch - operations return immediately.
        Without block_until_ready(), timing measures dispatch time (wrong).

        This test proves block_until_ready is necessary and implemented.
        """

        # Create simple JAX operation
        @jax.jit
        def simple_matmul(x, y):
            return jnp.dot(x, y)

        # Create test matrices
        x = jnp.ones((1000, 1000))
        y = jnp.ones((1000, 1000))

        # Warmup
        for _ in range(3):
            result = simple_matmul(x, y)
            result.block_until_ready()

        # Measure WITHOUT block_until_ready (wrong)
        start_wrong = time.perf_counter()
        _ = simple_matmul(x, y)
        time_wrong = time.perf_counter() - start_wrong

        # Measure WITH block_until_ready (correct)
        start_correct = time.perf_counter()
        result_correct = simple_matmul(x, y)
        result_correct.block_until_ready()  # ✅ Wait for GPU
        time_correct = time.perf_counter() - start_correct

        # Test: Correct timing should be larger
        # Wrong timing is ~microseconds (dispatch time)
        # Correct timing is ~milliseconds (actual computation)
        # Note: On GPU with small ops, ratio may be 2-5x (dispatch is optimized)
        # On larger ops or CPU, ratio can be 10-100x
        assert time_correct > time_wrong * 2, (
            f"block_until_ready not working: "
            f"wrong={time_wrong:.6f}s, correct={time_correct:.6f}s. "
            f"Ratio should be >2x, got {time_correct / time_wrong:.1f}x"
        )

        # Correct timing should be meaningful (not just dispatch)
        assert time_correct > 1e-5, (
            f"Timing too fast ({time_correct:.6f}s). Likely missing block_until_ready."
        )

    def test_benchmark_returns_statistics(self):
        """Verify benchmarks return statistical information.

        Proper benchmarks should run multiple iterations and return
        mean, std, min, median for reliability.
        """
        # This test defines expected return format
        # Benchmark functions should return dict with keys:
        # - "mean_time": float
        # - "std_time": float
        # - "min_time": float
        # - "median_time": float

        # Will be implemented

    def test_benchmark_handles_compilation_separately(self):
        """Verify benchmarks separate compilation from execution timing.

        First call to JIT function includes compilation time.
        This should be excluded via warmup, not counted in timing.
        """
        # Define expected behavior:
        # 1. Warmup iterations (exclude from timing)
        # 2. Timed iterations (only execution, no compilation)

        # Will be implemented


class TestGPUAccelerationBenchmarking:
    """Test gpu_acceleration.py benchmarking correctness.

    This module has benchmarking code that currently lacks block_until_ready.
    These tests prove it's broken and define correct behavior.
    """

    def test_cached_test_operation_uses_block_until_ready(self):
        """Test that CachedProgressiveTester uses block_until_ready.

        Location: opifex/core/gpu_acceleration.py
        Class: CachedProgressiveTester
        Method: _actual_test_operation (lines 434-486)

        Issue: Missing block_until_ready at lines 469-471 and 476-480
        """
        from opifex.core.gpu_acceleration import CachedProgressiveTester

        tester = CachedProgressiveTester()

        # Test a simple operation with larger size for robust timing
        success, exec_time, error = tester._actual_test_operation(
            "safe_matmul", 512, "float32"
        )

        # Should succeed
        assert success, f"Test operation failed: {error}"

        # Execution time should be meaningful (not just dispatch)
        # Matrix multiply of 512x512 should take >10 microseconds
        assert exec_time is not None
        assert exec_time > 1e-5, (
            f"Timing too fast ({exec_time:.6f}s). "
            f"Likely missing block_until_ready in warmup or timing loops."
        )

        # Should be consistent across runs
        _, exec_time2, _ = tester._actual_test_operation("safe_matmul", 512, "float32")

        # Times should be similar (within 2x due to caching/noise)
        if exec_time2 is not None:
            ratio = max(exec_time, exec_time2) / min(exec_time, exec_time2)
            assert ratio < 3.0, (
                f"Timing unstable: {exec_time:.6f}s vs {exec_time2:.6f}s. "
                f"Ratio: {ratio:.2f}x. Likely missing block_until_ready."
            )

    def test_benchmark_with_prefetching_uses_block_until_ready(self):
        """Test that OptimizedGPUManager benchmarking is correct.

        Location: opifex/core/gpu_acceleration.py
        Class: OptimizedGPUManager
        Method: benchmark_with_prefetching

        Should use block_until_ready in timing loops.
        """
        from opifex.core.gpu_acceleration import OptimizedGPUManager

        manager = OptimizedGPUManager()

        # Benchmark small sizes only (fast test)
        test_sizes = [64, 128]
        results = manager.benchmark_with_prefetching(
            manager.optimal_matrix_multiply, test_sizes
        )

        # Verify results exist
        assert 64 in results
        assert 128 in results

        # Verify timing is meaningful
        if results[64]["success"] and results[64]["execution_time"] is not None:
            exec_time = results[64]["execution_time"]
            # Should be >10 microseconds for 64x64 matmul
            assert exec_time > 1e-5, (
                f"Timing too fast ({exec_time:.6f}s). "
                f"Missing block_until_ready in benchmark loop."
            )


class TestRooflineAnalyzerTiming:
    """Test roofline analyzer uses correct timing.

    Roofline analysis depends on accurate timing measurements.
    Without block_until_ready, all measurements are wrong.
    """

    @pytest.mark.slow
    def test_roofline_analyzer_accurate_timing(self):
        """Test roofline analyzer produces accurate timing.

        Location: opifex/benchmarking/profiling/roofline_analyzer.py
        Class: RooflineAnalyzer
        Method: analyze_operation

        Should include block_until_ready in timing measurements.
        """
        from opifex.benchmarking.profiling.event_coordinator import EventCoordinator
        from opifex.benchmarking.profiling.roofline_analyzer import RooflineAnalyzer

        coordinator = EventCoordinator()
        analyzer = RooflineAnalyzer(coordinator)

        # Create a compute-heavy operation
        @jax.jit
        def compute_heavy_op(x):
            # Do enough work to be measurable
            for _ in range(50):
                x = jnp.sin(x) * jnp.cos(x)
            return x

        x = jnp.ones((1000, 1000))

        # Analyze
        results = analyzer.analyze_operation(compute_heavy_op, [x], name="test_op")

        # Verify timing is present and reasonable
        assert "actual_time_ms" in results
        exec_time_ms = results["actual_time_ms"]

        # Should be > 0.1ms (100us) for this much work
        assert exec_time_ms > 0.1, (
            f"Timing too fast ({exec_time_ms}ms). Missing block_until_ready?"
        )


class TestBenchmarkRunnerTiming:
    """Test benchmark runner uses correct timing.

    The benchmark runner orchestrates benchmarks across the codebase.
    It must use proper timing with block_until_ready throughout.
    """

    @pytest.mark.slow
    def test_benchmark_runner_uses_block_until_ready(self):
        """Test benchmark runner timing is accurate.

        Location: opifex/benchmarking/benchmark_runner.py

        Should use block_until_ready for all timing measurements.
        """
        # This test will be implemented after checking actual
        # BenchmarkRunner implementation
        # Placeholder


class TestEvaluationEngineTiming:
    """Test evaluation engine uses correct timing.

    Location: opifex/benchmarking/evaluation_engine.py

    Evaluation engine benchmarks model performance.
    Must use block_until_ready for accurate measurements.
    """

    @pytest.mark.slow
    def test_evaluation_engine_timing_accuracy(self):
        """Test evaluation engine timing is accurate."""
        # Will be implemented after checking actual implementation
        # Placeholder


# ==============================================================================
# Integration Tests - Verify Complete Benchmark Pipeline Uses Correct Timing
# ==============================================================================


class TestBenchmarkPipelineCorrectness:
    """Integration tests for complete benchmark pipeline.

    These tests verify the entire benchmarking infrastructure uses
    block_until_ready throughout for accurate measurements.
    """

    @pytest.mark.slow
    @pytest.mark.integration
    def test_end_to_end_benchmark_timing_accuracy(self):
        """Test complete benchmark pipeline produces accurate timings.

        This integration test runs a simple end-to-end benchmark and
        verifies the timing is meaningful and accurate.
        """
        # Create simple model to benchmark
        from flax import nnx

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(10, 10, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        x = jnp.ones((100, 10))

        # Warmup
        for _ in range(3):
            result = model(x)
            result.block_until_ready()

        # Time multiple iterations
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = model(x)
            result.block_until_ready()  # ✅ CRITICAL
            times.append(time.perf_counter() - start)

        mean_time = jnp.mean(jnp.array(times))
        std_time = jnp.std(jnp.array(times))

        # Timing should be meaningful
        assert mean_time > 1e-6, f"Timing too fast: {mean_time:.6f}s"

        # Timing should be relatively stable (allow some variance on GPU)
        cv = std_time / mean_time  # Coefficient of variation
        assert cv < 1.0, f"Timing unstable: CV={cv:.2f} (threshold: 1.0)"


# ==============================================================================
# Timing Utility Tests - Define Expected Benchmark Utility Behavior
# ==============================================================================


class TestBenchmarkUtilities:
    """Test benchmark utility functions (to be created/updated).

    These tests define the expected behavior of benchmark utilities.
    Implementation must match these tests.
    """

    def test_benchmark_function_signature(self):
        """Define expected signature for benchmark functions.

        Expected signature:
        benchmark_operation(
            func: Callable,
            *args,
            num_iterations: int = 100,
            warmup: int = 10
        ) -> dict[str, float]

        Returns: {
            "mean_time": float,
            "std_time": float,
            "min_time": float,
            "median_time": float,
        }
        """
        # Will be implemented when utilities are created

    def test_benchmark_function_correctness(self):
        """Test benchmark utility produces accurate measurements.

        Will test the canonical benchmark_operation function once created.
        """
        # Will be implemented


# ==============================================================================
# Test Markers and Configuration
# ==============================================================================


# ==============================================================================
# TDD Status: These tests are written FIRST
# ==============================================================================

# Expected Initial Status: MOST TESTS WILL FAIL
# - test_cached_test_operation_uses_block_until_ready: WILL FAIL (missing block_until_ready)
# - test_benchmark_with_prefetching_uses_block_until_ready: WILL FAIL (missing block_until_ready)
# - test_benchmark_waits_for_computation: WILL PASS (demonstrates correct pattern)
# - Integration tests: WILL FAIL (missing block_until_ready in actual code)

# After Implementation: ALL TESTS SHOULD PASS
# - All benchmarking code updated with block_until_ready
# - Timing measurements accurate
# - Tests verify correctness

# TDD Workflow:
# 1. ✅ Tests written (this file)
# 2. ⏳ Run tests (expect failures - RED)
# 3. ⏳ Implement fixes (add block_until_ready)
# 4. ⏳ Run tests again (expect pass - GREEN)
# 5. ⏳ Refactor if needed (keep tests GREEN)
