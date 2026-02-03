"""Tests for roofline analyzer module.

This module tests the roofline analysis functionality, covering:
- RooflineAnalyzer initialization
- Operation analysis (compute-bound vs memory-bound)
- FLOPs estimation
- Recommendation generation
- Alignment score calculation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from opifex.benchmarking.profiling.roofline_analyzer import RooflineAnalyzer


class TestRooflineAnalyzerInitialization:
    """Test RooflineAnalyzer initialization."""

    def test_initialization_with_coordinator(self):
        """Test profiler initialization with coordinator."""
        mock_coordinator = MagicMock()

        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        assert analyzer.coordinator is mock_coordinator
        assert analyzer.hardware_specs is not None

    def test_hardware_specs_populated(self):
        """Test that hardware specs are populated on init."""
        mock_coordinator = MagicMock()

        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        assert "peak_flops" in analyzer.hardware_specs
        assert "memory_bandwidth" in analyzer.hardware_specs
        assert "critical_intensity" in analyzer.hardware_specs


class TestAnalyzeOperation:
    """Test analyze_operation method."""

    def test_returns_expected_keys(self):
        """Test that analyze_operation returns all expected keys."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def dummy_func(*inputs):
            return inputs[0] + inputs[1]

        a = jnp.ones((32, 32))
        b = jnp.ones((32, 32))

        result = analyzer.analyze_operation(dummy_func, [a, b], name="test_add")

        assert "arithmetic_intensity" in result
        assert "critical_intensity" in result
        assert "memory_bandwidth_utilization" in result
        assert "flops_utilization" in result
        assert "bottleneck" in result
        assert "efficiency" in result
        assert "actual_time_ms" in result
        assert "optimization_recommendations" in result

    def test_identifies_memory_bound_operation(self):
        """Test identification of memory-bound operations."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        # Simple element-wise operation is typically memory-bound
        def simple_add(*inputs):
            return inputs[0] + 1

        small_input = jnp.ones((16, 16))

        result = analyzer.analyze_operation(simple_add, [small_input])

        # Low arithmetic intensity should indicate memory-bound
        assert result["arithmetic_intensity"] < result["critical_intensity"]
        assert result["bottleneck"] == "Memory Bandwidth"

    def test_execution_time_positive(self):
        """Test that execution time is positive."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def dummy_func(*inputs):
            return inputs[0] * 2

        input_data = jnp.ones((32, 32))

        result = analyzer.analyze_operation(dummy_func, [input_data])

        assert result["actual_time_ms"] >= 0

    def test_handles_tuple_output(self):
        """Test handling of functions that return tuples."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def multi_output(*inputs):
            return inputs[0], inputs[0] * 2

        input_data = jnp.ones((16, 16))

        result = analyzer.analyze_operation(multi_output, [input_data])

        assert isinstance(result, dict)
        assert "bottleneck" in result

    def test_handles_list_output(self):
        """Test handling of functions that return lists."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def list_output(*inputs):
            return [inputs[0], inputs[0] + 1]

        input_data = jnp.ones((16, 16))

        result = analyzer.analyze_operation(list_output, [input_data])

        assert isinstance(result, dict)
        assert "bottleneck" in result


class TestEstimateOperationFlops:
    """Test _estimate_operation_flops method."""

    def test_flops_proportional_to_input_size(self):
        """Test that FLOPs estimation scales with input size."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def dummy_func(*inputs):
            return inputs[0]

        small_input = jnp.ones((8, 8))  # 64 elements
        large_input = jnp.ones((16, 16))  # 256 elements

        small_flops = analyzer._estimate_operation_flops(dummy_func, [small_input])
        large_flops = analyzer._estimate_operation_flops(dummy_func, [large_input])

        # Large input should result in more FLOPs
        assert large_flops > small_flops

    def test_multiple_inputs_combined(self):
        """Test that FLOPs estimation combines multiple inputs."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def dummy_func(*inputs):
            return inputs[0] + inputs[1]

        single_input = [jnp.ones((16, 16))]  # 256 elements
        double_input = [jnp.ones((16, 16)), jnp.ones((16, 16))]  # 512 elements

        single_flops = analyzer._estimate_operation_flops(dummy_func, single_input)
        double_flops = analyzer._estimate_operation_flops(dummy_func, double_input)

        assert double_flops > single_flops

    def test_empty_inputs_returns_zero(self):
        """Test that empty inputs return zero FLOPs."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def dummy_func():
            return jnp.array(1.0)

        flops = analyzer._estimate_operation_flops(dummy_func, [])

        assert flops == 0


class TestGenerateRecommendations:
    """Test _generate_recommendations method."""

    def test_memory_bound_recommendations(self):
        """Test recommendations for memory-bound operations."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        inputs = [jnp.ones((32, 32))]
        recommendations = analyzer._generate_recommendations(
            arithmetic_intensity=5.0,  # Low intensity
            efficiency=0.3,
            bottleneck="Memory Bandwidth",
            inputs=inputs,
            achieved_flops=1e9,
        )

        # Should recommend memory-related optimizations
        rec_text = " ".join(recommendations)
        assert "Memory Bound" in rec_text or "batch size" in rec_text.lower()

    def test_compute_bound_recommendations(self):
        """Test recommendations for compute-bound operations."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        inputs = [jnp.ones((32, 32))]
        recommendations = analyzer._generate_recommendations(
            arithmetic_intensity=200.0,  # High intensity
            efficiency=0.6,
            bottleneck="Compute (FLOPs)",
            inputs=inputs,
            achieved_flops=100e9,
        )

        # Should recommend compute-related optimizations
        rec_text = " ".join(recommendations)
        assert "Compute Bound" in rec_text or "FLOPs" in rec_text

    def test_low_efficiency_recommendations(self):
        """Test recommendations for low efficiency operations."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        inputs = [jnp.ones((32, 32))]
        recommendations = analyzer._generate_recommendations(
            arithmetic_intensity=50.0,
            efficiency=0.1,  # Very low efficiency
            bottleneck="Memory Bandwidth",
            inputs=inputs,
            achieved_flops=0.5e9,
        )

        # Should have performance warning
        rec_text = " ".join(recommendations)
        assert "Low Performance" in rec_text or "GFLOPS" in rec_text

    def test_moderate_efficiency_recommendations(self):
        """Test recommendations for moderate efficiency operations."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        inputs = [jnp.ones((32, 32))]
        recommendations = analyzer._generate_recommendations(
            arithmetic_intensity=50.0,
            efficiency=0.35,  # Moderate efficiency
            bottleneck="Memory Bandwidth",
            inputs=inputs,
            achieved_flops=10e9,
        )

        # Should have moderate efficiency guidance
        rec_text = " ".join(recommendations)
        assert "Moderate" in rec_text or "efficiency" in rec_text.lower()

    def test_poor_alignment_recommendation(self):
        """Test recommendation for poorly aligned tensors."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        # Misaligned shape (not multiple of 128)
        inputs = [jnp.ones((17, 33))]
        recommendations = analyzer._generate_recommendations(
            arithmetic_intensity=50.0,
            efficiency=0.6,
            bottleneck="Memory Bandwidth",
            inputs=inputs,
            achieved_flops=10e9,
        )

        # Should have alignment warning
        rec_text = " ".join(recommendations)
        assert "alignment" in rec_text.lower() or "Pad" in rec_text

    def test_well_aligned_no_alignment_warning(self):
        """Test no alignment warning for well-aligned tensors."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        # Well-aligned shape (multiple of 128)
        inputs = [jnp.ones((32, 128))]
        recommendations = analyzer._generate_recommendations(
            arithmetic_intensity=50.0,
            efficiency=0.8,  # High efficiency
            bottleneck="Memory Bandwidth",
            inputs=inputs,
            achieved_flops=100e9,
        )

        # Should not have alignment warning
        rec_text = " ".join(recommendations)
        assert "Poor tensor alignment" not in rec_text

    def test_empty_inputs_handled(self):
        """Test that empty inputs don't cause errors."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        recommendations = analyzer._generate_recommendations(
            arithmetic_intensity=50.0,
            efficiency=0.6,
            bottleneck="Compute (FLOPs)",
            inputs=[],
            achieved_flops=10e9,
        )

        assert isinstance(recommendations, list)


class TestCalculateAlignmentScore:
    """Test _calculate_alignment_score method."""

    def test_perfect_alignment_128(self):
        """Test perfect alignment for multiple of 128."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        score = analyzer._calculate_alignment_score((32, 128))

        assert score == 1.0

    def test_perfect_alignment_256(self):
        """Test perfect alignment for multiple of 256."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        score = analyzer._calculate_alignment_score((32, 256))

        assert score == 1.0

    def test_good_alignment_32(self):
        """Test good alignment for multiple of 32."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        score = analyzer._calculate_alignment_score((16, 32))

        assert score == 0.8

    def test_moderate_alignment_8(self):
        """Test moderate alignment for multiple of 8."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        score = analyzer._calculate_alignment_score((16, 24))

        assert score == 0.5

    def test_poor_alignment(self):
        """Test poor alignment for odd dimensions."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        score = analyzer._calculate_alignment_score((17, 33))

        assert score == 0.2

    def test_empty_shape(self):
        """Test that empty shape returns 1.0."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        score = analyzer._calculate_alignment_score(())

        assert score == 1.0

    def test_1d_shape_alignment(self):
        """Test alignment calculation for 1D shapes."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        aligned_score = analyzer._calculate_alignment_score((128,))
        misaligned_score = analyzer._calculate_alignment_score((17,))

        assert aligned_score == 1.0
        assert misaligned_score == 0.2


class TestRooflineAnalyzerEdgeCases:
    """Test edge cases in roofline analyzer."""

    def test_scalar_output(self):
        """Test handling of scalar output."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def reduce_sum(*inputs):
            return jnp.sum(inputs[0])

        input_data = jnp.ones((16, 16))

        result = analyzer.analyze_operation(reduce_sum, [input_data])

        assert isinstance(result, dict)
        assert "bottleneck" in result

    def test_very_small_input(self):
        """Test with very small input."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def identity(*inputs):
            return inputs[0]

        tiny_input = jnp.ones((2, 2))

        result = analyzer.analyze_operation(identity, [tiny_input])

        assert isinstance(result, dict)
        assert result["arithmetic_intensity"] >= 0

    def test_high_dimensional_input(self):
        """Test with high-dimensional input."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def sum_op(*inputs):
            return jnp.sum(inputs[0], axis=-1)

        high_dim_input = jnp.ones((2, 4, 8, 16))

        result = analyzer.analyze_operation(sum_op, [high_dim_input])

        assert isinstance(result, dict)
        assert "bottleneck" in result

    def test_matmul_operation(self):
        """Test with matrix multiplication operation."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def matmul(*inputs):
            return inputs[0] @ inputs[1]

        a = jnp.ones((32, 64))
        b = jnp.ones((64, 32))

        result = analyzer.analyze_operation(matmul, [a, b])

        assert isinstance(result, dict)
        # Matmul has higher arithmetic intensity than element-wise ops
        assert result["arithmetic_intensity"] > 0

    def test_utilization_values_reasonable(self):
        """Test that utilization values are in reasonable range."""
        mock_coordinator = MagicMock()
        analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

        def dummy_op(*inputs):
            return inputs[0] * 2

        input_data = jnp.ones((32, 32))

        result = analyzer.analyze_operation(dummy_op, [input_data])

        # Utilization should be between 0 and 1 (or slightly above in edge cases)
        assert result["flops_utilization"] >= 0
        assert result["memory_bandwidth_utilization"] >= 0


class TestRooflineAnalyzerWithMockedHardware:
    """Test RooflineAnalyzer with mocked hardware specs."""

    def test_with_custom_hardware_specs(self):
        """Test analyzer with mocked hardware specs for determinism."""
        mock_coordinator = MagicMock()

        with patch(
            "opifex.benchmarking.profiling.roofline_analyzer.detect_hardware_specs"
        ) as mock_detect:
            mock_detect.return_value = {
                "peak_flops": 100e12,
                "memory_bandwidth": 1000e9,
                "critical_intensity": 100.0,
            }

            analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

            assert analyzer.hardware_specs["peak_flops"] == 100e12
            assert analyzer.hardware_specs["memory_bandwidth"] == 1000e9
            assert analyzer.hardware_specs["critical_intensity"] == 100.0

    def test_bottleneck_determination_with_known_specs(self):
        """Test bottleneck determination with known hardware specs."""
        mock_coordinator = MagicMock()

        with patch(
            "opifex.benchmarking.profiling.roofline_analyzer.detect_hardware_specs"
        ) as mock_detect:
            mock_detect.return_value = {
                "peak_flops": 100e12,
                "memory_bandwidth": 1000e9,
                "critical_intensity": 100.0,  # Low critical intensity
            }

            analyzer = RooflineAnalyzer(coordinator=mock_coordinator)

            def simple_op(*inputs):
                return inputs[0] + 1

            input_data = jnp.ones((16, 16))
            result = analyzer.analyze_operation(simple_op, [input_data])

            # With low critical intensity, most simple ops should be compute-bound
            # or memory-bound depending on their arithmetic intensity
            assert result["bottleneck"] in ["Memory Bandwidth", "Compute (FLOPs)"]
