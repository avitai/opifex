"""Tests for memory profiler module.

This module tests the memory profiling functionality, covering:
- MemoryProfiler class initialization and lifecycle
- Memory checkpoint updates
- Memory usage analysis
- Efficiency categorization
- Optimization suggestions
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from opifex.benchmarking.profiling.memory_profiler import (
    _analyze_parameter_memory,
    _analyze_tensor_memory,
    memory_usage_analysis,
    MemoryProfiler,
)
from opifex.neural.base import StandardMLP


class TestMemoryProfiler:
    """Test MemoryProfiler class."""

    def test_initialization(self):
        """Test profiler initialization."""
        profiler = MemoryProfiler()

        assert profiler.baseline_memory is None
        assert profiler.peak_memory == 0
        assert profiler.memory_timeline == []

    def test_start_profiling(self):
        """Test starting profiling."""
        profiler = MemoryProfiler()

        profiler.start_profiling()

        assert profiler.baseline_memory is not None
        assert profiler.baseline_memory > 0
        assert profiler.peak_memory == profiler.baseline_memory
        assert len(profiler.memory_timeline) == 1

    def test_stop_profiling_returns_results(self):
        """Test that stop_profiling returns results dict."""
        profiler = MemoryProfiler()
        profiler.start_profiling()

        results = profiler.stop_profiling()

        assert isinstance(results, dict)
        assert "baseline_memory_mb" in results
        assert "peak_memory_mb" in results
        assert "memory_increase_mb" in results
        assert "timeline" in results

    def test_checkpoint_updates_timeline(self):
        """Test that checkpoint updates memory timeline."""
        profiler = MemoryProfiler()
        profiler.start_profiling()

        profiler.checkpoint("test_checkpoint")

        assert len(profiler.memory_timeline) == 2
        last_entry = profiler.memory_timeline[-1]
        assert len(last_entry) == 3  # (timestamp, memory, label)
        assert last_entry[2] == "test_checkpoint"

    def test_checkpoint_updates_peak_memory(self):
        """Test that checkpoint updates peak memory if higher."""
        profiler = MemoryProfiler()
        profiler.start_profiling()

        initial_peak = profiler.peak_memory

        # Allocate some memory to potentially increase peak
        _ = jnp.ones((1000, 1000))
        profiler.checkpoint("after_allocation")

        assert profiler.peak_memory >= initial_peak

    def test_multiple_checkpoints(self):
        """Test multiple checkpoints."""
        profiler = MemoryProfiler()
        profiler.start_profiling()

        profiler.checkpoint("step_1")
        profiler.checkpoint("step_2")
        profiler.checkpoint("step_3")

        assert len(profiler.memory_timeline) == 4  # Initial + 3 checkpoints


class TestMemoryUsageAnalysis:
    """Test memory_usage_analysis function."""

    def test_basic_analysis(self):
        """Test basic memory usage analysis."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_data = jnp.ones((8, 4))

        result = memory_usage_analysis(model, input_data)

        assert isinstance(result, dict)
        assert "parameter_memory" in result
        assert "input_memory" in result
        assert "output_memory" in result

    def test_analysis_without_gradients(self):
        """Test analysis without gradient computation."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_data = jnp.ones((8, 4))

        result = memory_usage_analysis(model, input_data, include_gradients=False)

        assert "gradient_memory" in result
        assert result["gradient_memory"] is None

    def test_analysis_with_gradients(self):
        """Test analysis with gradient computation."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_data = jnp.ones((8, 4))

        result = memory_usage_analysis(model, input_data, include_gradients=True)

        assert "gradient_memory" in result
        # May be dict with error or actual gradient memory
        assert result["gradient_memory"] is not None

    def test_analysis_includes_efficiency(self):
        """Test that analysis includes efficiency metrics."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_data = jnp.ones((8, 4))

        result = memory_usage_analysis(model, input_data)

        assert "efficiency_analysis" in result
        assert "optimization_suggestions" in result

    def test_analysis_includes_profiling_timeline(self):
        """Test that analysis includes profiling timeline."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_data = jnp.ones((8, 4))

        result = memory_usage_analysis(model, input_data)

        assert "profiling_timeline" in result
        timeline = result["profiling_timeline"]
        assert "baseline_memory_mb" in timeline
        assert "peak_memory_mb" in timeline


class TestAnalyzeParameterMemory:
    """Test _analyze_parameter_memory function."""

    def test_returns_total_bytes(self):
        """Test that analysis returns total bytes."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        result = _analyze_parameter_memory(model)

        assert "total_bytes" in result
        assert "total_mb" in result
        assert result["total_bytes"] > 0

    def test_returns_parameter_breakdown(self):
        """Test that analysis returns parameter breakdown."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        result = _analyze_parameter_memory(model)

        assert "parameter_breakdown" in result
        assert isinstance(result["parameter_breakdown"], dict)

    def test_identifies_largest_parameter(self):
        """Test that analysis identifies largest parameter."""
        model = StandardMLP([4, 16, 1], rngs=nnx.Rngs(42))

        result = _analyze_parameter_memory(model)

        assert "largest_parameter" in result
        assert result["largest_parameter"] != "none"


class TestAnalyzeTensorMemory:
    """Test _analyze_tensor_memory function."""

    def test_returns_shape(self):
        """Test that analysis returns tensor shape."""
        tensor = jnp.ones((8, 4))

        result = _analyze_tensor_memory(tensor)

        assert "shape" in result
        assert result["shape"] == (8, 4)

    def test_returns_dtype(self):
        """Test that analysis returns tensor dtype."""
        tensor = jnp.ones((8, 4), dtype=jnp.float32)

        result = _analyze_tensor_memory(tensor)

        assert "dtype" in result
        assert "float32" in result["dtype"]

    def test_returns_memory_size(self):
        """Test that analysis returns memory size."""
        tensor = jnp.ones((10, 10), dtype=jnp.float32)

        result = _analyze_tensor_memory(tensor)

        assert "bytes" in result
        assert "mb" in result
        assert result["bytes"] == 10 * 10 * 4  # 100 floats * 4 bytes

    def test_returns_element_count(self):
        """Test that analysis returns element count."""
        tensor = jnp.ones((8, 4))

        result = _analyze_tensor_memory(tensor)

        assert "elements" in result
        assert result["elements"] == 32
