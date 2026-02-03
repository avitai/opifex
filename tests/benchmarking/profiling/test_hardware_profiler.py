"""Tests for hardware profiler module.

This module tests the hardware profiling functionality, covering:
- TPUProfiler initialization and analysis
- GPUProfiler initialization and analysis
- MXU utilization analysis
- VMEM usage analysis
- Hardware detection
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp

from opifex.benchmarking.profiling.hardware_profiler import (
    GPUProfiler,
    TPUProfiler,
)


class TestTPUProfiler:
    """Test TPUProfiler class."""

    def test_initialization(self):
        """Test profiler initialization."""
        profiler = TPUProfiler()

        assert profiler.coordinator is None
        assert profiler.hardware_specs is not None

    def test_initialization_with_coordinator(self):
        """Test profiler initialization with coordinator."""
        mock_coordinator = MagicMock()

        profiler = TPUProfiler(coordinator=mock_coordinator)

        assert profiler.coordinator is mock_coordinator

    def test_analyze_mxu_utilization(self):
        """Test MXU utilization analysis."""
        profiler = TPUProfiler()

        def dummy_func(*inputs):
            return inputs[0] @ inputs[1]

        # Create matrix inputs
        a = jnp.ones((32, 64))
        b = jnp.ones((64, 32))

        result = profiler.analyze_mxu_utilization(dummy_func, [a, b])

        assert isinstance(result, dict)
        assert "mxu_utilization" in result
        assert "estimated_mxu_ops" in result
        assert "execution_time_ms" in result
        assert "shape_alignment" in result
        assert "recommendations" in result

    def test_analyze_vmem_usage(self):
        """Test VMEM usage analysis."""
        profiler = TPUProfiler()

        def dummy_func(*inputs):
            return inputs[0] + 1

        inputs = [jnp.ones((16, 16))]

        result = profiler.analyze_vmem_usage(dummy_func, inputs)

        assert isinstance(result, dict)
        assert "vmem_eligible" in result
        assert "total_input_size_mb" in result
        assert "vmem_threshold_mb" in result
        assert "speedup_potential" in result
        assert "recommendation" in result

    def test_vmem_eligible_for_small_inputs(self):
        """Test that small inputs are VMEM eligible."""
        profiler = TPUProfiler()

        def dummy_func(*inputs):
            return inputs[0]

        # Small input - should be VMEM eligible
        small_input = jnp.ones((8, 8))

        result = profiler.analyze_vmem_usage(dummy_func, [small_input])

        assert result["vmem_eligible"] is True
        assert result["speedup_potential"] == 22.0

    def test_estimate_mxu_operations(self):
        """Test MXU operation estimation."""
        profiler = TPUProfiler()

        # Matrix multiplication pattern
        a = jnp.ones((32, 64))
        b = jnp.ones((64, 32))

        ops = profiler._estimate_mxu_operations([a, b])

        # 2 * M * K * N = 2 * 32 * 64 * 32 = 131072
        assert ops == 131072

    def test_analyze_mxu_shape_alignment(self):
        """Test MXU shape alignment analysis."""
        profiler = TPUProfiler()

        # Well-aligned shape (multiple of 128)
        aligned = jnp.ones((128, 256))
        result = profiler._analyze_mxu_shape_alignment([aligned])

        assert "individual_scores" in result
        assert "shape_recommendations" in result
        assert "average_alignment_score" in result
        assert "overall_rating" in result


class TestGPUProfiler:
    """Test GPUProfiler class."""

    def test_initialization(self):
        """Test profiler initialization."""
        profiler = GPUProfiler()

        assert profiler.coordinator is None
        assert profiler.hardware_specs is not None

    def test_initialization_with_coordinator(self):
        """Test profiler initialization with coordinator."""
        mock_coordinator = MagicMock()

        profiler = GPUProfiler(coordinator=mock_coordinator)

        assert profiler.coordinator is mock_coordinator

    def test_analyze_tensorcore_utilization(self):
        """Test TensorCore utilization analysis."""
        profiler = GPUProfiler()

        def dummy_func(*inputs):
            return inputs[0] @ inputs[1]

        a = jnp.ones((32, 64))
        b = jnp.ones((64, 32))

        result = profiler.analyze_tensorcore_utilization(dummy_func, [a, b])

        assert isinstance(result, dict)
        assert "tensorcore_utilization" in result
        assert "estimated_tensorcore_ops" in result
        assert "execution_time_ms" in result
        assert "recommendations" in result
        assert "shape_alignment" in result

    def test_analyze_memory_coalescing(self):
        """Test memory coalescing analysis."""
        profiler = GPUProfiler()

        # Create input with specific memory layout
        input_data = jnp.ones((64, 64))

        result = profiler.analyze_memory_coalescing([input_data])

        assert isinstance(result, dict)
        assert "average_coalescing_efficiency" in result
        assert "individual_analysis" in result
        assert "memory_throughput_loss" in result
        assert "optimization_suggestions" in result

    def test_estimate_tensorcore_operations(self):
        """Test TensorCore operation estimation."""
        profiler = GPUProfiler()

        a = jnp.ones((32, 64))
        b = jnp.ones((64, 32))

        ops = profiler._estimate_tensorcore_operations([a, b])

        # Should detect matrix multiplication pattern
        assert ops > 0


class TestHardwareProfilerEdgeCases:
    """Test edge cases in hardware profilers."""

    def test_tpu_profiler_empty_inputs(self):
        """Test TPU profiler with empty inputs."""
        profiler = TPUProfiler()

        result = profiler._estimate_mxu_operations([])

        assert result == 0

    def test_gpu_profiler_empty_inputs(self):
        """Test GPU profiler with empty inputs."""
        profiler = GPUProfiler()

        result = profiler._estimate_tensorcore_operations([])

        assert result == 0

    def test_tpu_profiler_1d_inputs(self):
        """Test TPU profiler with 1D inputs."""
        profiler = TPUProfiler()

        # 1D inputs shouldn't contribute to MXU ops
        vec = jnp.ones((64,))

        ops = profiler._estimate_mxu_operations([vec])

        assert ops == 0

    def test_tpu_profiler_misaligned_shapes(self):
        """Test alignment analysis with misaligned shapes."""
        profiler = TPUProfiler()

        # Misaligned shape (not multiple of 8 or 128)
        misaligned = jnp.ones((17, 33))

        result = profiler._analyze_mxu_shape_alignment([misaligned])

        # Should have low alignment score
        if result["individual_scores"]:
            assert result["individual_scores"][0] < 1.0
        assert result["average_alignment_score"] < 1.0
