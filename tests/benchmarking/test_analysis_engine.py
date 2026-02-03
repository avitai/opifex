"""Tests for benchmarking analysis engine functionality.

This module tests the AnalysisEngine class, covering:
- Operator comparison with minimum 2 operators required
- Performance rankings and winner determination (via calibrax)
- Scaling behavior analysis
- Complexity estimation (O(n), O(n^2), etc.)
- Performance insights generation
- Domain inference from dataset names
- Statistical significance testing
"""

import warnings

import numpy as np
import pytest
from calibrax.core.models import Metric

from opifex.benchmarking.analysis_engine import (
    _calculate_confidence,
    _estimate_complexity,
    _infer_domain,
    _test_statistical_significance,
    AnalysisEngine,
    ComparisonReport,
    InsightReport,
    RecommendationReport,
    ScalingAnalysis,
)
from opifex.benchmarking.evaluation_engine import BenchmarkResult


def _make_result(
    name: str,
    dataset: str,
    metrics: dict[str, float],
    execution_time: float = 1.0,
    memory_usage: float | None = None,
) -> BenchmarkResult:
    """Helper to build a BenchmarkResult with the calibrax API."""
    metadata: dict = {
        "execution_time": execution_time,
        "framework_version": "flax_nnx",
    }
    if memory_usage is not None:
        metadata["memory_usage"] = memory_usage
    return BenchmarkResult(
        name=name,
        tags={"dataset": dataset},
        metrics={k: Metric(value=v) for k, v in metrics.items()},
        metadata=metadata,
    )


class TestAnalysisEngineInitialization:
    """Test AnalysisEngine initialization."""

    def test_default_initialization(self):
        """Test default initialization with default significance threshold."""
        engine = AnalysisEngine()
        assert engine.significance_threshold == 0.05

    def test_custom_significance_threshold(self):
        """Test initialization with custom significance threshold."""
        engine = AnalysisEngine(significance_threshold=0.01)
        assert engine.significance_threshold == 0.01

    def test_significance_threshold_stored(self):
        """Test that significance threshold is stored."""
        engine = AnalysisEngine(significance_threshold=0.01)
        assert engine.significance_threshold == 0.01


class TestOperatorComparison:
    """Test operator comparison functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalysisEngine()

    def test_compare_operators_minimum_two_required(self):
        """Test that comparison requires at least 2 operators."""
        result = _make_result("FNO", "TestDataset", {"mse": 0.01})

        with pytest.raises(ValueError, match="Need at least 2 operators"):
            self.engine.compare_operators({"FNO": result})

    def test_compare_two_operators(self):
        """Test comparison of two operators."""
        result1 = _make_result("FNO", "TestDataset", {"mse": 0.01, "mae": 0.05})
        result2 = _make_result(
            "DeepONet", "TestDataset", {"mse": 0.02, "mae": 0.08}, execution_time=2.0
        )

        report = self.engine.compare_operators({"FNO": result1, "DeepONet": result2})

        assert isinstance(report, ComparisonReport)
        assert report.benchmark_name == "TestDataset"
        assert "FNO" in report.operators_compared
        assert "DeepONet" in report.operators_compared

    def test_performance_rankings_lower_is_better(self):
        """Test that lower values are ranked higher for error metrics."""
        result1 = _make_result("FNO", "TestDataset", {"mse": 0.01})
        result2 = _make_result(
            "DeepONet", "TestDataset", {"mse": 0.02}, execution_time=2.0
        )

        report = self.engine.compare_operators({"FNO": result1, "DeepONet": result2})

        # FNO has lower MSE, so should be ranked first
        assert report.performance_rankings["mse"][0] == "FNO"
        assert report.winner_by_metric["mse"] == "FNO"

    def test_winner_determination(self):
        """Test overall winner determination using weighted scoring."""
        result1 = _make_result("FNO", "TestDataset", {"mse": 0.01, "mae": 0.05})
        result2 = _make_result(
            "DeepONet", "TestDataset", {"mse": 0.02, "mae": 0.08}, execution_time=2.0
        )

        report = self.engine.compare_operators({"FNO": result1, "DeepONet": result2})

        # FNO is better on all metrics
        assert report.overall_winner == "FNO"

    def test_improvement_factors_computed(self):
        """Test that improvement factors are computed for each metric."""
        result1 = _make_result("FNO", "TestDataset", {"mse": 0.01})
        result2 = _make_result(
            "DeepONet", "TestDataset", {"mse": 0.02}, execution_time=2.0
        )

        report = self.engine.compare_operators({"FNO": result1, "DeepONet": result2})

        assert "mse" in report.improvement_factors
        # FNO should have improvement factor of 1.0 (best)
        assert report.improvement_factors["mse"]["FNO"] == pytest.approx(1.0, rel=1e-5)
        # DeepONet should have improvement factor of 0.5 (2x worse)
        assert report.improvement_factors["mse"]["DeepONet"] == pytest.approx(
            0.5, rel=1e-5
        )


class TestScalingBehaviorAnalysis:
    """Test scaling behavior analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalysisEngine()

    def test_scaling_analysis_minimum_sizes(self):
        """Test that warning is issued for insufficient problem sizes."""
        result1 = _make_result("FNO", "TestDataset", {"mse": 0.01})
        result2 = _make_result("FNO", "TestDataset", {"mse": 0.01}, execution_time=2.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.engine.analyze_scaling_behavior({100: result1, 200: result2})
            assert len(w) == 1
            assert "at least 3 problem sizes" in str(w[0].message)

    def test_scaling_analysis_returns_analysis(self):
        """Test that scaling analysis returns proper ScalingAnalysis."""
        results = {}
        for i, size in enumerate([100, 200, 400, 800]):
            results[size] = _make_result(
                "FNO",
                "TestDataset",
                {"mse": 0.01 / (i + 1)},
                execution_time=size * 0.001,  # Linear scaling
            )

        analysis = self.engine.analyze_scaling_behavior(results)

        assert isinstance(analysis, ScalingAnalysis)
        assert analysis.operator_name == "FNO"
        assert analysis.problem_sizes == [100, 200, 400, 800]

    def test_scaling_coefficients_computed(self):
        """Test that scaling coefficients are computed."""
        results = {}
        for size in [100, 200, 400, 800]:
            # Quadratic scaling: execution_time ~ O(n^2)
            results[size] = _make_result(
                "FNO",
                "TestDataset",
                {"mse": 0.01},
                execution_time=size**2 * 1e-6,
            )

        analysis = self.engine.analyze_scaling_behavior(results)

        assert "execution_time" in analysis.scaling_coefficients
        # Should be close to 2.0 for O(n^2)
        assert analysis.scaling_coefficients["execution_time"] == pytest.approx(
            2.0, rel=0.1
        )


class TestComplexityEstimation:
    """Test computational complexity estimation."""

    def test_linear_complexity_estimation(self):
        """Test O(n) complexity estimation."""
        complexity = _estimate_complexity("execution_time", 1.0)
        assert complexity == "O(n)"

    def test_quadratic_complexity_estimation(self):
        """Test O(n^2) complexity estimation."""
        complexity = _estimate_complexity("execution_time", 2.0)
        assert complexity == "O(n\u00b2)"

    def test_cubic_complexity_estimation(self):
        """Test O(n^3) complexity estimation."""
        complexity = _estimate_complexity("execution_time", 3.0)
        assert complexity == "O(n\u00b3)"

    def test_higher_complexity_estimation(self):
        """Test higher complexity estimation."""
        complexity = _estimate_complexity("execution_time", 4.5)
        assert "O(n^4.5)" in complexity

    def test_non_execution_time_metric(self):
        """Test complexity estimation for non-execution_time metrics."""
        complexity = _estimate_complexity("mse", 1.5)
        assert "scaling exponent" in complexity


class TestPerformanceInsightsGeneration:
    """Test performance insights generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalysisEngine()

    def test_insights_report_structure(self, benchmark_result):
        """Test that insight report has expected structure."""
        report = self.engine.generate_performance_insights(benchmark_result)

        assert isinstance(report, InsightReport)
        assert report.benchmark_name == "TestDataset"
        assert report.operator_name == "TestModel"
        assert isinstance(report.key_insights, list)
        assert isinstance(report.performance_bottlenecks, list)
        assert isinstance(report.optimization_suggestions, list)
        assert isinstance(report.domain_specific_observations, list)

    def test_high_execution_time_bottleneck(self):
        """Test bottleneck identification for high execution time."""
        result = _make_result("FNO", "TestDataset", {"mse": 0.01}, execution_time=15.0)

        report = self.engine.generate_performance_insights(result)

        assert any(
            "execution time" in bottleneck.lower()
            for bottleneck in report.performance_bottlenecks
        )

    def test_excellent_efficiency_insight(self):
        """Test insight for excellent computational efficiency."""
        result = _make_result("FNO", "TestDataset", {"mse": 0.01}, execution_time=0.05)

        report = self.engine.generate_performance_insights(result)

        assert any(
            "computational efficiency" in insight.lower()
            for insight in report.key_insights
        )

    def test_exceptional_accuracy_insight(self):
        """Test insight for exceptional accuracy."""
        result = _make_result("FNO", "TestDataset", {"mse": 1e-7})

        report = self.engine.generate_performance_insights(result)

        assert any(
            "exceptional accuracy" in insight.lower() for insight in report.key_insights
        )

    def test_poor_accuracy_bottleneck(self):
        """Test bottleneck identification for poor accuracy."""
        result = _make_result("FNO", "TestDataset", {"mse": 0.5})

        report = self.engine.generate_performance_insights(result)

        assert any(
            "accuracy" in bottleneck.lower()
            for bottleneck in report.performance_bottlenecks
        )

    def test_high_memory_usage_bottleneck(self):
        """Test bottleneck identification for high memory usage."""
        result = _make_result(
            "FNO",
            "TestDataset",
            {"mse": 0.01},
            memory_usage=20 * 1024**3,  # 20 GB
        )

        report = self.engine.generate_performance_insights(result)

        assert any(
            "memory" in bottleneck.lower()
            for bottleneck in report.performance_bottlenecks
        )


class TestDomainInference:
    """Test domain inference from dataset names."""

    def test_quantum_computing_domain(self):
        """Test inference of quantum computing domain."""
        assert _infer_domain("QuantumChemistry") == "quantum_computing"
        assert _infer_domain("dft_energies") == "quantum_computing"
        assert _infer_domain("molecular_dynamics") == "quantum_computing"

    def test_fluid_dynamics_domain(self):
        """Test inference of fluid dynamics domain."""
        assert _infer_domain("NavierStokes2D") == "fluid_dynamics"
        assert _infer_domain("Burgers1D") == "fluid_dynamics"
        assert _infer_domain("DarcyFlow") == "fluid_dynamics"

    def test_materials_science_domain(self):
        """Test inference of materials science domain."""
        assert _infer_domain("CrystalEnergies") == "materials_science"
        assert _infer_domain("SolidMechanics") == "materials_science"
        assert _infer_domain("material_properties") == "materials_science"

    def test_climate_modeling_domain(self):
        """Test inference of climate modeling domain."""
        assert _infer_domain("WeatherPrediction") == "climate_modeling"
        assert _infer_domain("climate_data") == "climate_modeling"
        assert _infer_domain("atmospheric_pressure") == "climate_modeling"

    def test_general_domain_fallback(self):
        """Test fallback to general domain."""
        assert _infer_domain("SomeRandomDataset") == "general"
        assert _infer_domain("TestDataset") == "general"


class TestOperatorRecommendations:
    """Test operator recommendations generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalysisEngine()

    def test_recommendations_for_pde_solving(self):
        """Test recommendations for PDE solving problem type."""
        report = self.engine.create_operator_recommendations(
            "pde_solving", "fluid_dynamics"
        )

        assert isinstance(report, RecommendationReport)
        assert report.problem_type == "pde_solving"
        assert report.domain == "fluid_dynamics"
        assert len(report.recommended_operators) > 0

        # Check that FNO is in recommendations for PDE solving
        operator_names = [op["operator"] for op in report.recommended_operators]
        assert "FNO" in operator_names

    def test_recommendations_include_domain_considerations(self):
        """Test that recommendations include domain-specific considerations."""
        report = self.engine.create_operator_recommendations(
            "pde_solving", "quantum_computing"
        )

        # Should include quantum-specific considerations
        assert any(
            "quantum" in consideration.lower()
            for consideration in report.implementation_considerations
        )

    def test_recommendations_include_use_cases(self):
        """Test that recommendations include use case specific recommendations."""
        report = self.engine.create_operator_recommendations("pde_solving", "general")

        assert "high_accuracy" in report.use_case_specific_recommendations
        assert "fast_inference" in report.use_case_specific_recommendations

    def test_recommendations_include_trade_offs(self):
        """Test that recommendations include performance trade-offs."""
        report = self.engine.create_operator_recommendations("pde_solving", "general")

        assert "accuracy_vs_speed" in report.performance_trade_offs


class TestStatisticalSignificanceTesting:
    """Test statistical significance testing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalysisEngine()

    def test_single_run_significance_testing(self):
        """Test significance testing with single run results."""
        result1 = _make_result("FNO", "TestDataset", {"mse": 0.01})
        result2 = _make_result(
            "DeepONet", "TestDataset", {"mse": 0.05}, execution_time=2.0
        )

        significance = _test_statistical_significance(
            {"FNO": result1, "DeepONet": result2}, self.engine.significance_threshold
        )

        # Should find significance due to large relative difference
        assert "FNO" in significance
        assert "DeepONet" in significance["FNO"]

    def test_multi_run_significance_testing(self):
        """Test significance testing with multiple runs."""
        rng = np.random.default_rng(42)
        # Create multiple runs with clear difference
        fno_results = [
            _make_result(
                "FNO",
                "TestDataset",
                {"mse": 0.01 + rng.normal(0, 0.001)},
            )
            for _ in range(10)
        ]
        deeponet_results = [
            _make_result(
                "DeepONet",
                "TestDataset",
                {"mse": 0.05 + rng.normal(0, 0.001)},
                execution_time=2.0,
            )
            for _ in range(10)
        ]

        significance = self.engine.test_statistical_significance_multi_run(
            {"FNO": fno_results, "DeepONet": deeponet_results}
        )

        # Should find significance
        assert "FNO" in significance or "DeepONet" in significance
        if "FNO" in significance and "DeepONet" in significance["FNO"]:
            result = significance["FNO"]["DeepONet"]
            assert "significant" in result
            assert "t_pvalue" in result
            assert "u_pvalue" in result


class TestConfidenceCalculation:
    """Test confidence level calculation."""

    def test_confidence_base_level(self):
        """Test base confidence level."""
        result = _make_result("FNO", "TestDataset", {}, execution_time=0)

        confidence = _calculate_confidence(result)

        # Should be base confidence (0.8) since no adjustments apply
        assert confidence >= 0.8

    def test_confidence_increases_with_good_metrics(self):
        """Test that confidence increases with good metrics."""
        result = _make_result("FNO", "TestDataset", {"mse": 1e-5})

        confidence = _calculate_confidence(result)

        # Should be higher than base confidence
        assert confidence >= 0.9

    def test_confidence_max_is_one(self):
        """Test that confidence is capped at 1.0."""
        result = _make_result("FNO", "TestDataset", {"mse": 1e-5})

        confidence = _calculate_confidence(result)

        assert confidence <= 1.0
