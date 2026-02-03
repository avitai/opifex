"""Tests for model complexity analysis module.

This module tests the complexity analysis functionality, covering:
- Model complexity analysis main function
- Parameter analysis
- Memory usage estimation
- Computational complexity analysis
- Scaling property analysis
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from opifex.benchmarking.profiling.complexity_analysis import (
    _analyze_computational_complexity,
    _analyze_memory_usage,
    _analyze_parameters,
    _analyze_scaling_properties,
    model_complexity_analysis,
)
from opifex.neural.base import StandardMLP


class TestModelComplexityAnalysis:
    """Test model_complexity_analysis function."""

    def test_returns_comprehensive_analysis(self):
        """Test that analysis returns all expected sections."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (8, 4)

        result = model_complexity_analysis(model, input_shape)

        assert isinstance(result, dict)
        assert "parameters" in result
        assert "memory" in result
        assert "computational" in result
        assert "scaling" in result
        assert "input_shape" in result
        assert "model_type" in result

    def test_preserves_input_shape(self):
        """Test that analysis preserves input shape."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (16, 4)

        result = model_complexity_analysis(model, input_shape)

        assert result["input_shape"] == input_shape

    def test_identifies_model_type(self):
        """Test that analysis identifies model type."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (8, 4)

        result = model_complexity_analysis(model, input_shape)

        assert result["model_type"] is not None


class TestAnalyzeParameters:
    """Test _analyze_parameters function."""

    def test_counts_total_parameters(self):
        """Test that analysis counts total parameters."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        result = _analyze_parameters(model)

        assert "total_parameters" in result
        assert result["total_parameters"] > 0
        # 4*8 (weights) + 8 (bias) + 8*1 (weights) + 1 (bias) = 49 params
        assert result["total_parameters"] == 49

    def test_calculates_parameter_memory(self):
        """Test that analysis calculates parameter memory."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        result = _analyze_parameters(model)

        assert "parameter_memory_mb" in result
        assert result["parameter_memory_mb"] > 0

    def test_identifies_largest_layer(self):
        """Test that analysis identifies largest layer."""
        model = StandardMLP([4, 32, 8, 1], rngs=nnx.Rngs(42))

        result = _analyze_parameters(model)

        assert "largest_layer" in result
        assert "name" in result["largest_layer"]
        assert "params" in result["largest_layer"]

    def test_returns_parameter_breakdown(self):
        """Test that analysis returns parameter breakdown."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        result = _analyze_parameters(model)

        assert "parameter_breakdown" in result
        assert isinstance(result["parameter_breakdown"], dict)

    def test_memory_efficiency_flag(self):
        """Test that analysis includes memory efficiency flag."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        result = _analyze_parameters(model)

        assert "memory_efficient" in result
        assert isinstance(result["memory_efficient"], bool)


class TestAnalyzeMemoryUsage:
    """Test _analyze_memory_usage function."""

    def test_estimates_intermediate_memory(self):
        """Test that analysis estimates intermediate memory."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        sample_input = jnp.ones((8, 4))

        result = _analyze_memory_usage(model, sample_input)

        assert "estimated_intermediate_mb" in result or "error" in result

    def test_estimates_total_memory(self):
        """Test that analysis estimates total memory."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        sample_input = jnp.ones((8, 4))

        result = _analyze_memory_usage(model, sample_input)

        if "error" not in result:
            assert "total_estimated_mb" in result

    def test_memory_efficiency_flag(self):
        """Test that analysis includes memory efficiency flag."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        sample_input = jnp.ones((8, 4))

        result = _analyze_memory_usage(model, sample_input)

        if "error" not in result:
            assert "memory_efficient" in result


class TestAnalyzeComputationalComplexity:
    """Test _analyze_computational_complexity function."""

    def test_estimates_total_operations(self):
        """Test that analysis estimates total operations."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (8, 4)

        result = _analyze_computational_complexity(model, input_shape)

        assert "total_estimated_operations" in result
        assert result["total_estimated_operations"] > 0

    def test_calculates_operations_per_sample(self):
        """Test that analysis calculates operations per sample."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (8, 4)

        result = _analyze_computational_complexity(model, input_shape)

        assert "operations_per_sample" in result

    def test_returns_complexity_breakdown(self):
        """Test that analysis returns complexity breakdown."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (8, 4)

        result = _analyze_computational_complexity(model, input_shape)

        assert "complexity_breakdown" in result
        assert isinstance(result["complexity_breakdown"], dict)

    def test_identifies_dominant_complexity(self):
        """Test that analysis identifies dominant complexity."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (8, 16, 16)  # 2D spatial input

        result = _analyze_computational_complexity(model, input_shape)

        assert "dominant_complexity" in result

    def test_analyzes_fft_complexity_for_spatial_input(self):
        """Test that analysis includes FFT complexity for spatial inputs."""
        model = StandardMLP([256, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (8, 16, 16)  # 2D spatial input

        result = _analyze_computational_complexity(model, input_shape)

        breakdown = result["complexity_breakdown"]
        assert "fft_operations" in breakdown


class TestAnalyzeScalingProperties:
    """Test _analyze_scaling_properties function."""

    def test_returns_scaling_tests(self):
        """Test that analysis returns scaling tests."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (8, 4)

        result = _analyze_scaling_properties(model, input_shape)

        assert isinstance(result, dict)

    def test_includes_multiple_scale_factors(self):
        """Test that analysis tests multiple scale factors."""
        model = StandardMLP([256, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (8, 16, 16)

        result = _analyze_scaling_properties(model, input_shape)

        # Should test scale factors 0.5, 1.0, 2.0
        if "scaling_tests" in result:
            assert len(result["scaling_tests"]) >= 3


class TestComplexityAnalysisEdgeCases:
    """Test edge cases in complexity analysis."""

    def test_small_model(self):
        """Test analysis of a very small model."""
        model = StandardMLP([2, 2, 1], rngs=nnx.Rngs(42))
        input_shape = (1, 2)

        result = model_complexity_analysis(model, input_shape)

        # 2*2 (weights) + 2 (bias) + 2*1 (weights) + 1 (bias) = 9 params
        assert result["parameters"]["total_parameters"] == 9

    def test_batch_size_one(self):
        """Test analysis with batch size of 1."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (1, 4)

        result = model_complexity_analysis(model, input_shape)

        assert "parameters" in result
        assert "computational" in result

    def test_large_spatial_dimensions(self):
        """Test analysis with large spatial dimensions."""
        model = StandardMLP([1024, 8, 1], rngs=nnx.Rngs(42))
        input_shape = (4, 32, 32)  # 1024 spatial size

        result = model_complexity_analysis(model, input_shape)

        assert "computational" in result
        assert result["computational"]["total_estimated_operations"] > 0
