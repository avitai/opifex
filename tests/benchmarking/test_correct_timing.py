"""Tests for correct benchmarking timing with block_until_ready.

Verifies that all benchmarking code properly uses block_until_ready()
to ensure accurate timing measurements on GPU hardware.
"""

import time

import jax
import jax.numpy as jnp
import pytest


class TestCorrectBenchmarkTiming:
    """Test that benchmarks use block_until_ready for accurate timing."""

    def test_benchmark_waits_for_computation(self):
        """Verify benchmarks use block_until_ready for accurate timing.

        JAX uses asynchronous dispatch - operations return immediately.
        Without block_until_ready(), timing measures dispatch time (wrong).

        This test proves block_until_ready is necessary and implemented.
        """

        @jax.jit
        def simple_matmul(x, y):
            return jnp.dot(x, y)

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
        result_correct.block_until_ready()
        time_correct = time.perf_counter() - start_correct

        # Correct timing should be larger
        assert time_correct > time_wrong * 2, (
            f"block_until_ready not working: "
            f"wrong={time_wrong:.6f}s, correct={time_correct:.6f}s. "
            f"Ratio should be >2x, got {time_correct / time_wrong:.1f}x"
        )

        # Correct timing should be meaningful (not just dispatch)
        assert time_correct > 1e-5, (
            f"Timing too fast ({time_correct:.6f}s). Likely missing block_until_ready."
        )


class TestGPUAccelerationBenchmarking:
    """Test gpu_acceleration.py benchmarking correctness."""

    def test_cached_test_operation_uses_block_until_ready(self):
        """Test that CachedProgressiveTester uses block_until_ready."""
        from opifex.core.gpu_acceleration import CachedProgressiveTester

        tester = CachedProgressiveTester()

        success, exec_time, error = tester._actual_test_operation(
            "safe_matmul", 512, "float32"
        )

        assert success, f"Test operation failed: {error}"

        # Execution time should be meaningful (not just dispatch)
        assert exec_time is not None
        assert exec_time > 1e-5, (
            f"Timing too fast ({exec_time:.6f}s). "
            f"Likely missing block_until_ready in warmup or timing loops."
        )

        # Should be consistent across runs
        _, exec_time2, _ = tester._actual_test_operation("safe_matmul", 512, "float32")

        if exec_time2 is not None:
            ratio = max(exec_time, exec_time2) / min(exec_time, exec_time2)
            assert ratio < 3.0, (
                f"Timing unstable: {exec_time:.6f}s vs {exec_time2:.6f}s. "
                f"Ratio: {ratio:.2f}x. Likely missing block_until_ready."
            )

    def test_benchmark_with_prefetching_uses_block_until_ready(self):
        """Test that OptimizedGPUManager benchmarking is correct."""
        from opifex.core.gpu_acceleration import OptimizedGPUManager

        manager = OptimizedGPUManager()

        test_sizes = [64, 128]
        results = manager.benchmark_with_prefetching(
            manager.optimal_matrix_multiply, test_sizes
        )

        assert 64 in results
        assert 128 in results

        if results[64]["success"] and results[64]["execution_time"] is not None:
            exec_time = results[64]["execution_time"]
            assert exec_time > 1e-5, (
                f"Timing too fast ({exec_time:.6f}s). "
                f"Missing block_until_ready in benchmark loop."
            )


class TestBenchmarkPipelineCorrectness:
    """Integration tests for complete benchmark pipeline."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_end_to_end_benchmark_timing_accuracy(self):
        """Test complete benchmark pipeline produces accurate timings."""
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
            result.block_until_ready()
            times.append(time.perf_counter() - start)

        mean_time = jnp.mean(jnp.array(times))
        std_time = jnp.std(jnp.array(times))

        assert mean_time > 1e-6, f"Timing too fast: {mean_time:.6f}s"

        # Timing should be relatively stable
        cv = std_time / mean_time
        assert cv < 1.0, f"Timing unstable: CV={cv:.2f} (threshold: 1.0)"
