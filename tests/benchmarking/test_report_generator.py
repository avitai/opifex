"""Tests for PDEBench report generation functionality.

This module tests the PDEBenchReportGenerator class, covering:
- Report generation with various input combinations
- Metadata generation with dataset/model info
- Evaluation summary generation with MSE thresholds
- Detailed metrics extraction
- Statistical analysis
- Recommendations generation
- Report formatting (JSON, text)
- File saving
"""

import json

import jax.numpy as jnp
import pytest

from opifex.benchmarking.report_generator import PDEBenchReportGenerator


class TestPDEBenchReportGeneratorInitialization:
    """Test PDEBenchReportGenerator initialization."""

    def test_default_initialization(self):
        """Test default initialization with JSON format."""
        generator = PDEBenchReportGenerator()
        assert generator.report_format == "json"
        assert generator.generation_timestamp is not None

    def test_custom_format_initialization(self):
        """Test initialization with custom format."""
        generator = PDEBenchReportGenerator(report_format="text")
        assert generator.report_format == "text"

    def test_timestamp_is_iso_format(self):
        """Test that timestamp is in ISO format."""
        generator = PDEBenchReportGenerator()
        # The timestamp should be in ISO format with 'T' separator
        assert "T" in generator.generation_timestamp


class TestMetadataGeneration:
    """Test metadata generation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()

    def test_metadata_without_dataset_or_model_info(self):
        """Test metadata generation with no dataset or model info."""
        metadata = self.generator._generate_metadata(None, None)

        assert "report_type" in metadata
        assert metadata["report_type"] == "PDEBench Evaluation Report"
        assert "generated_at" in metadata
        assert "dataset" not in metadata
        assert "model" not in metadata

    def test_metadata_with_dataset_info(self):
        """Test metadata generation with dataset info."""
        dataset_info = {
            "name": "Burgers2D",
            "type": "PDE",
            "size": "1000 samples",
            "description": "2D Burgers equation dataset",
        }

        metadata = self.generator._generate_metadata(dataset_info, None)

        assert "dataset" in metadata
        assert metadata["dataset"]["name"] == "Burgers2D"
        assert metadata["dataset"]["type"] == "PDE"
        assert metadata["dataset"]["size"] == "1000 samples"
        assert metadata["dataset"]["description"] == "2D Burgers equation dataset"

    def test_metadata_with_model_info(self):
        """Test metadata generation with model info."""
        model_info = {
            "name": "FNO",
            "type": "Neural Operator",
            "parameters": "1M",
            "architecture": "Fourier Neural Operator",
        }

        metadata = self.generator._generate_metadata(None, model_info)

        assert "model" in metadata
        assert metadata["model"]["name"] == "FNO"
        assert metadata["model"]["type"] == "Neural Operator"
        assert metadata["model"]["parameters"] == "1M"
        assert metadata["model"]["architecture"] == "Fourier Neural Operator"

    def test_metadata_with_partial_info(self):
        """Test metadata generation with partial info (missing keys)."""
        dataset_info = {"name": "TestDataset"}  # Missing other keys
        model_info = {"name": "TestModel"}  # Missing other keys

        metadata = self.generator._generate_metadata(dataset_info, model_info)

        # Should use "Unknown" for missing keys
        assert metadata["dataset"]["type"] == "Unknown"
        assert metadata["model"]["parameters"] == "Unknown"


class TestEvaluationSummaryGeneration:
    """Test evaluation summary generation with MSE thresholds."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()

    def test_excellent_performance_mse(self):
        """Test excellent performance classification (MSE < 0.01)."""
        results = {"mse": 0.005}
        summary = self.generator._generate_evaluation_summary(results)

        assert summary["overall_performance"] == "Excellent"
        assert summary["key_metrics"]["mse"] == 0.005

    def test_good_performance_mse(self):
        """Test good performance classification (0.01 <= MSE < 0.1)."""
        results = {"mse": 0.05}
        summary = self.generator._generate_evaluation_summary(results)

        assert summary["overall_performance"] == "Good"

    def test_fair_performance_mse(self):
        """Test fair performance classification (0.1 <= MSE < 1.0)."""
        results = {"mse": 0.5}
        summary = self.generator._generate_evaluation_summary(results)

        assert summary["overall_performance"] == "Fair"

    def test_needs_improvement_mse(self):
        """Test needs improvement classification (MSE >= 1.0)."""
        results = {"mse": 1.5}
        summary = self.generator._generate_evaluation_summary(results)

        assert summary["overall_performance"] == "Needs Improvement"

    def test_r2_excellent_correlation_finding(self):
        """Test that high R² score generates excellent correlation finding."""
        results = {"mse": 0.01, "r2_score": 0.98}
        summary = self.generator._generate_evaluation_summary(results)

        assert "Excellent correlation with ground truth" in summary["notable_findings"]

    def test_r2_low_correlation_finding(self):
        """Test that low R² score generates warning finding."""
        results = {"mse": 0.5, "r2_score": 0.3}
        summary = self.generator._generate_evaluation_summary(results)

        assert any(
            "Low correlation" in finding for finding in summary["notable_findings"]
        )

    def test_fast_inference_finding(self):
        """Test that fast evaluation time generates finding."""
        results = {"mse": 0.05, "evaluation_time": 0.5}
        summary = self.generator._generate_evaluation_summary(results)

        assert "Very fast inference time" in summary["notable_findings"]

    def test_slow_inference_finding(self):
        """Test that slow evaluation time generates warning finding."""
        results = {"mse": 0.05, "evaluation_time": 15.0}
        summary = self.generator._generate_evaluation_summary(results)

        assert any(
            "Slow inference" in finding for finding in summary["notable_findings"]
        )


class TestDetailedMetricsExtraction:
    """Test detailed metrics extraction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()

    def test_accuracy_metrics_extraction(self):
        """Test extraction of accuracy-related metrics."""
        results = {
            "mse": 0.01,
            "mae": 0.05,
            "rmse": 0.1,
            "r2_score": 0.95,
            "relative_error": 0.02,
        }
        detailed = self.generator._generate_detailed_metrics(results)

        assert detailed["accuracy_metrics"]["mse"] == 0.01
        assert detailed["accuracy_metrics"]["mae"] == 0.05
        assert detailed["accuracy_metrics"]["rmse"] == 0.1
        assert detailed["accuracy_metrics"]["r2_score"] == 0.95
        assert detailed["accuracy_metrics"]["relative_error"] == 0.02

    def test_efficiency_metrics_extraction(self):
        """Test extraction of efficiency-related metrics."""
        results = {
            "evaluation_time": 1.5,
            "memory_usage": 1024.0,
            "flops": 1e9,
        }
        detailed = self.generator._generate_detailed_metrics(results)

        assert detailed["efficiency_metrics"]["evaluation_time"] == 1.5
        assert detailed["efficiency_metrics"]["memory_usage"] == 1024.0
        assert detailed["efficiency_metrics"]["flops"] == 1e9

    def test_statistical_metrics_extraction(self):
        """Test extraction of statistical metrics."""
        results = {
            "mean_prediction": 0.5,
            "std_prediction": 0.1,
            "confidence_interval": [0.3, 0.7],
        }
        detailed = self.generator._generate_detailed_metrics(results)

        assert detailed["statistical_metrics"]["mean_prediction"] == 0.5
        assert detailed["statistical_metrics"]["std_prediction"] == 0.1
        assert detailed["statistical_metrics"]["confidence_interval"] == [0.3, 0.7]

    def test_empty_results(self):
        """Test handling of empty results dictionary."""
        results = {}
        detailed = self.generator._generate_detailed_metrics(results)

        assert detailed["accuracy_metrics"] == {}
        assert detailed["efficiency_metrics"] == {}
        assert detailed["statistical_metrics"] == {}


class TestStatisticalAnalysis:
    """Test statistical analysis generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()

    def test_distribution_analysis_with_predictions(self):
        """Test distribution analysis when predictions are provided."""
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        results = {"predictions": predictions}

        analysis = self.generator._generate_statistical_analysis(results)

        assert "distribution_analysis" in analysis
        dist = analysis["distribution_analysis"]
        assert dist["mean"] == pytest.approx(3.0, rel=1e-5)
        assert dist["min"] == pytest.approx(1.0, rel=1e-5)
        assert dist["max"] == pytest.approx(5.0, rel=1e-5)
        assert dist["median"] == pytest.approx(3.0, rel=1e-5)

    def test_uncertainty_quantification(self):
        """Test uncertainty quantification extraction."""
        results = {
            "epistemic_uncertainty": 0.1,
            "aleatoric_uncertainty": 0.2,
        }
        analysis = self.generator._generate_statistical_analysis(results)

        assert analysis["uncertainty_quantification"]["epistemic"] == 0.1
        assert analysis["uncertainty_quantification"]["aleatoric"] == 0.2

    def test_significance_tests_placeholder(self):
        """Test that significance tests show 'Not computed' status."""
        results = {}
        analysis = self.generator._generate_statistical_analysis(results)

        assert analysis["significance_tests"]["status"] == "Not computed"


class TestRecommendationsGeneration:
    """Test recommendations generation based on results."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()

    def test_high_mse_recommendation(self):
        """Test recommendation for high MSE."""
        results = {"mse": 2.0}
        recommendations = self.generator._generate_recommendations(results)

        assert any("increasing model complexity" in rec for rec in recommendations)

    def test_excellent_mse_recommendation(self):
        """Test recommendation for excellent MSE."""
        results = {"mse": 0.0005}
        recommendations = self.generator._generate_recommendations(results)

        assert any("ready for production" in rec for rec in recommendations)

    def test_slow_inference_recommendation(self):
        """Test recommendation for slow inference time."""
        results = {"evaluation_time": 10.0}
        recommendations = self.generator._generate_recommendations(results)

        assert any("optimization techniques" in rec for rec in recommendations)

    def test_high_epistemic_uncertainty_recommendation(self):
        """Test recommendation for high epistemic uncertainty."""
        results = {"epistemic_uncertainty": 0.7}
        recommendations = self.generator._generate_recommendations(results)

        assert any("ensemble methods" in rec for rec in recommendations)

    def test_low_r2_recommendation(self):
        """Test recommendation for low R² score."""
        results = {"r2_score": 0.5}
        recommendations = self.generator._generate_recommendations(results)

        assert any(
            "data quality" in rec or "underfitting" in rec for rec in recommendations
        )

    def test_default_recommendation(self):
        """Test default recommendation when all metrics are good."""
        results = {}
        recommendations = self.generator._generate_recommendations(results)

        assert any("satisfactory" in rec for rec in recommendations)


class TestBaselineComparison:
    """Test baseline comparison generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()

    def test_no_baseline_comparisons(self):
        """Test handling when no baseline comparisons available."""
        comparison = self.generator._generate_baseline_comparison(None)

        assert comparison["status"] == "No baseline comparisons available"

    def test_baseline_comparison_with_data(self):
        """Test baseline comparison with actual data."""
        baseline_data = {
            "baselines": {
                "FNO": {"mse": 0.01, "relative_improvement": 0.5},
                "DeepONet": {"mse": 0.02, "relative_improvement": 0.3},
            },
            "current_model_performance": {
                "rank": 1,
                "total": 3,
                "percentile": 95.0,
            },
        }

        comparison = self.generator._generate_baseline_comparison(baseline_data)

        assert len(comparison["baseline_models"]) == 2
        assert comparison["performance_ranking"]["current_model_rank"] == 1
        assert comparison["performance_ranking"]["total_models"] == 3
        assert comparison["performance_ranking"]["percentile"] == 95.0


class TestReportFormatting:
    """Test report formatting functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()
        self.sample_report = self.generator.generate_evaluation_report(
            evaluation_results={"mse": 0.05, "mae": 0.1},
            dataset_info={"name": "TestDataset"},
            model_info={"name": "TestModel"},
        )

    def test_format_as_text(self):
        """Test text formatting of report."""
        text_report = self.generator.format_report_as_text(self.sample_report)

        assert "PDEBench Evaluation Report" in text_report
        assert "METADATA" in text_report
        assert "EVALUATION SUMMARY" in text_report
        assert "RECOMMENDATIONS" in text_report
        assert "TestDataset" in text_report
        assert "TestModel" in text_report

    def test_text_format_contains_metrics(self):
        """Test that text format contains key metrics."""
        text_report = self.generator.format_report_as_text(self.sample_report)

        assert "mse" in text_report.lower()


class TestReportSaving:
    """Test report saving functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()
        self.sample_report = self.generator.generate_evaluation_report(
            evaluation_results={"mse": 0.05}
        )

    def test_save_as_json(self, temp_directory):
        """Test saving report as JSON."""
        filepath = temp_directory / "report.json"
        self.generator.save_report(self.sample_report, str(filepath), "json")

        assert filepath.exists()

        # Verify it's valid JSON
        with open(filepath) as f:
            loaded = json.load(f)

        assert "metadata" in loaded
        assert "evaluation_summary" in loaded

    def test_save_as_text(self, temp_directory):
        """Test saving report as text."""
        filepath = temp_directory / "report.txt"
        self.generator.save_report(self.sample_report, str(filepath), "text")

        assert filepath.exists()

        with open(filepath) as f:
            content = f.read()

        assert "PDEBench Evaluation Report" in content

    def test_save_with_default_format(self, temp_directory):
        """Test saving with default format from initialization."""
        generator = PDEBenchReportGenerator(report_format="text")
        filepath = temp_directory / "report_default.txt"

        generator.save_report(self.sample_report, str(filepath))

        with open(filepath) as f:
            content = f.read()

        assert "PDEBench Evaluation Report" in content

    def test_save_invalid_format(self, temp_directory):
        """Test that invalid format raises ValueError."""
        filepath = temp_directory / "report.xml"

        with pytest.raises(ValueError, match="Unsupported format"):
            self.generator.save_report(self.sample_report, str(filepath), "xml")


class TestGenerateEvaluationReport:
    """Test the main generate_evaluation_report method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()

    def test_complete_report_structure(self):
        """Test that complete report has all expected sections."""
        report = self.generator.generate_evaluation_report(
            evaluation_results={"mse": 0.05, "mae": 0.1},
            baseline_comparisons=None,
            dataset_info={"name": "TestDataset"},
            model_info={"name": "TestModel"},
        )

        assert "metadata" in report
        assert "evaluation_summary" in report
        assert "detailed_metrics" in report
        assert "statistical_analysis" in report
        assert "baseline_comparison" in report
        assert "recommendations" in report
        assert "generation_info" in report

    def test_generation_info(self):
        """Test generation info section."""
        report = self.generator.generate_evaluation_report(
            evaluation_results={"mse": 0.05}
        )

        gen_info = report["generation_info"]
        assert "timestamp" in gen_info
        assert gen_info["format"] == "json"
        assert gen_info["generator_version"] == "1.0.0"


class TestSummaryStatistics:
    """Test summary statistics across multiple reports."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()

    def test_empty_reports_list(self):
        """Test handling of empty reports list."""
        summary = self.generator.generate_summary_statistics([])

        assert "error" in summary
        assert "No reports provided" in summary["error"]

    def test_summary_statistics_computation(self):
        """Test summary statistics computation across multiple reports."""
        reports = [
            {
                "detailed_metrics": {
                    "accuracy_metrics": {"mse": 0.01, "mae": 0.05},
                }
            },
            {
                "detailed_metrics": {
                    "accuracy_metrics": {"mse": 0.02, "mae": 0.06},
                }
            },
        ]

        summary = self.generator.generate_summary_statistics(reports)

        assert summary["total_reports"] == 2
        assert "aggregated_metrics" in summary

        # Check aggregated metrics
        agg = summary["aggregated_metrics"]["accuracy_metrics"]["mse"]
        assert agg["mean"] == pytest.approx(0.015, rel=1e-5)
        assert agg["min"] == pytest.approx(0.01, rel=1e-5)
        assert agg["max"] == pytest.approx(0.02, rel=1e-5)


class TestComprehensiveReport:
    """Test comprehensive report generation from BenchmarkResult objects."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PDEBenchReportGenerator()

    def test_comprehensive_report_empty_results(self):
        """Test handling of empty results list."""
        report = self.generator.generate_comprehensive_report([])

        assert "error" in report
        assert "No results provided" in report["error"]

    def test_comprehensive_report_with_results(self, benchmark_result):
        """Test comprehensive report with actual results."""
        report = self.generator.generate_comprehensive_report(
            [benchmark_result],
            include_baseline_comparison=True,
            include_statistical_analysis=True,
        )

        assert "metadata" in report
        assert "detailed_results" in report
        assert len(report["detailed_results"]) == 1
        assert report["detailed_results"][0]["model_name"] == "TestModel"

    def test_cross_dataset_analysis(self, benchmark_result):
        """Test cross-dataset performance analysis."""
        report = self.generator.generate_comprehensive_report(
            [benchmark_result],
            include_statistical_analysis=True,
        )

        assert "statistical_analysis" in report
        assert "cross_dataset_analysis" in report["statistical_analysis"]
