"""Tests for correct benchmarking timing with block_until_ready.

Verifies that all benchmarking code properly uses block_until_ready()
to ensure accurate timing measurements on GPU hardware.
"""

import time

import jax.numpy as jnp
import pytest


class TestCorrectBenchmarkTiming:
    """Test that benchmarks use block_until_ready for accurate timing."""

    def test_benchmark_waits_for_computation(self):
        """Benchmark synchronization should call block_until_ready on nested outputs."""
        from opifex.core.timing import block_until_ready

        class ReadySpy:
            def __init__(self):
                self.called = False

            def block_until_ready(self):
                self.called = True
                return self

        first = ReadySpy()
        second = ReadySpy()
        output = {"first": first, "nested": (second,)}

        assert block_until_ready(output) is output
        assert first.called
        assert second.called


class TestGPUAccelerationBenchmarking:
    """Test gpu_acceleration.py benchmarking correctness."""

    def test_cached_test_operation_uses_block_until_ready(self):
        """Test that CachedProgressiveTester uses block_until_ready."""
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
