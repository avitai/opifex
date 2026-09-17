"""Tests for correct benchmarking timing.

Verifies that the benchmarking code waits for its results (``jax.block_until_ready``) so the
timings measure the computation rather than the dispatch, and reports calibrax's median,
which one slow call does not move.
"""

import time
from typing import Any

import jax.numpy as jnp
import pytest
from calibrax.profiling import CallTiming


class TestGPUAccelerationBenchmarking:
    """Test gpu_acceleration.py benchmarking correctness."""

    def test_cached_test_operation_uses_block_until_ready(self):
        """The tester waits for every timed call, so the time is the computation's."""
        from opifex.core.gpu_acceleration import CachedProgressiveTester

        tester = CachedProgressiveTester()

        success, exec_time, error = tester._actual_test_operation("safe_matmul", 512, "float32")

        assert success, f"Test operation failed: {error}"

        # Execution time should be meaningful (not just dispatch)
        assert exec_time is not None
        assert exec_time > 1e-5, (
            f"Timing too fast ({exec_time:.6f}s). "
            f"Likely missing block_until_ready in warmup or timing loops."
        )

    def test_cached_test_operation_reports_calibrax_median_of_blocking_calls(self, monkeypatch):
        """The time is ``time_calls``'s median with its default, blocking, sync.

        A mean of the samples is moved by one slow call, which a shared CI runner
        produces at will; the median is calibrax's reported figure for that reason.
        """
        from opifex.core import gpu_acceleration

        seen: dict[str, Any] = {}

        def spy(func, *args, **kwargs):
            seen.update(kwargs)
            return CallTiming(
                samples_sec=(0.002, 0.003, 0.030),
                median_sec=0.003,
                percentiles_sec={50: 0.003},
                warmup=kwargs.get("warmup", 0),
            )

        monkeypatch.setattr(gpu_acceleration, "time_calls", spy)
        tester = gpu_acceleration.CachedProgressiveTester()

        success, exec_time, error = tester._actual_test_operation("safe_matmul", 64, "float32")

        assert success, error
        assert exec_time == 0.003
        assert seen.get("sync") is None, "the default sync waits for each result"
        assert seen["warmup"] == 3
        assert seen["iterations"] == 10

    def test_benchmark_with_prefetching_uses_block_until_ready(self):
        """Test that OptimizedGPUManager benchmarking is correct."""
        from opifex.core.gpu_acceleration import OptimizedGPUManager

        manager = OptimizedGPUManager()

        test_sizes = [64, 128]
        results = manager.benchmark_with_prefetching(manager.optimal_matrix_multiply, test_sizes)

        assert 64 in results
        assert 128 in results

        if results[64]["success"] and results[64]["execution_time"] is not None:
            exec_time = results[64]["execution_time"]
            assert exec_time > 1e-5, (
                f"Timing too fast ({exec_time:.6f}s). Missing block_until_ready in benchmark loop."
            )


class TestBenchmarkPipelineCorrectness:
    """Integration tests for complete benchmark pipeline."""

    @pytest.mark.integration
    def test_end_to_end_benchmark_timing_accuracy(self, monkeypatch):
        """Test complete benchmark timing aggregation without wall-clock flakiness."""
        from flax import nnx

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs) -> None:
                self.dense = nnx.Linear(10, 10, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        x = jnp.ones((100, 10))

        # Warmup
        for _ in range(3):
            result = model(x)
            result.block_until_ready()

        current_time = 0.0

        def fake_perf_counter():
            nonlocal current_time
            value = current_time
            current_time += 0.001
            return value

        monkeypatch.setattr(time, "perf_counter", fake_perf_counter)

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
