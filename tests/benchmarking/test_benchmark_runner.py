"""Tests for benchmark runner module.

This module tests the benchmark runner functionality, covering:
- BenchmarkRunner initialization
- DomainResults and PublicationReport dataclasses
- Error handling for missing operators/benchmarks
- Output directory creation
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from opifex.benchmarking.benchmark_runner import (
    BenchmarkRunner,
    DomainResults,
    PublicationReport,
)


class TestDomainResultsDataclass:
    """Test DomainResults dataclass."""

    def test_default_values(self):
        """Test default values for DomainResults."""
        result = DomainResults(domain="pde", benchmark_results={})

        assert result.domain == "pde"
        assert result.benchmark_results == {}
        assert result.validation_reports == {}
        assert result.comparison_reports == {}
        assert result.insight_reports == {}
        assert result.summary_statistics == {}

    def test_custom_values(self):
        """Test DomainResults with custom values."""
        benchmark_results = {"test_benchmark": {"operator1": MagicMock()}}
        validation_reports = {"test_benchmark": {"operator1": MagicMock()}}

        result = DomainResults(
            domain="quantum",
            benchmark_results=benchmark_results,  # pyright: ignore[reportArgumentType]
            validation_reports=validation_reports,  # pyright: ignore[reportArgumentType]
        )

        assert result.domain == "quantum"
        assert "test_benchmark" in result.benchmark_results
        assert "test_benchmark" in result.validation_reports


class TestPublicationReportDataclass:
    """Test PublicationReport dataclass."""

    def test_required_fields(self):
        """Test PublicationReport with required fields."""
        report = PublicationReport(
            title="Test Report",
            abstract="Test abstract",
            methodology="Test methodology",
            results_summary={"accuracy": 0.95},
        )

        assert report.title == "Test Report"
        assert report.abstract == "Test abstract"
        assert report.methodology == "Test methodology"
        assert report.results_summary == {"accuracy": 0.95}

    def test_default_lists(self):
        """Test that list fields default to empty lists."""
        report = PublicationReport(
            title="Test",
            abstract="Test",
            methodology="Test",
            results_summary={},
        )

        assert report.comparison_tables == []
        assert report.figures == []
        assert report.key_findings == []
        assert report.recommendations == []
        assert report.appendix_data == {}

    def test_with_optional_fields(self):
        """Test PublicationReport with all optional fields."""
        report = PublicationReport(
            title="Test",
            abstract="Test",
            methodology="Test",
            results_summary={},
            key_findings=["Finding 1", "Finding 2"],
            recommendations=["Rec 1"],
            comparison_tables=[Path("table1.csv")],
            figures=[Path("fig1.png")],
        )

        assert len(report.key_findings) == 2
        assert len(report.recommendations) == 1
        assert len(report.comparison_tables) == 1
        assert len(report.figures) == 1


class TestBenchmarkRunnerInitialization:
    """Test BenchmarkRunner initialization."""

    def test_default_initialization(self, temp_directory):
        """Test runner with default components."""
        output_dir = str(temp_directory / "benchmark_results")

        with patch(
            "opifex.benchmarking.benchmark_runner.BenchmarkRegistry"
        ) as mock_registry:
            mock_registry.return_value.list_available_operators.return_value = []
            runner = BenchmarkRunner(output_dir=output_dir)

            assert runner.output_dir == Path(output_dir)
            assert runner.registry is not None
            assert runner.evaluator is not None
            assert runner.validator is not None
            assert runner.analyzer is not None

    def test_output_directory_created(self, temp_directory):
        """Test that output directory is created."""
        output_dir = temp_directory / "new_benchmark_dir"

        with patch(
            "opifex.benchmarking.benchmark_runner.BenchmarkRegistry"
        ) as mock_registry:
            mock_registry.return_value.list_available_operators.return_value = []
            BenchmarkRunner(output_dir=str(output_dir))

            assert output_dir.exists()

    def test_custom_components(self, temp_directory):
        """Test runner with custom components."""
        mock_registry = MagicMock()
        mock_registry.list_available_operators.return_value = ["op1"]
        mock_evaluator = MagicMock()
        mock_validator = MagicMock()
        mock_analyzer = MagicMock()

        runner = BenchmarkRunner(
            registry=mock_registry,
            evaluator=mock_evaluator,
            validator=mock_validator,
            analyzer=mock_analyzer,
            output_dir=str(temp_directory),
        )

        assert runner.registry is mock_registry
        assert runner.evaluator is mock_evaluator
        assert runner.validator is mock_validator
        assert runner.analyzer is mock_analyzer


class TestBenchmarkRunnerErrors:
    """Test error handling in BenchmarkRunner."""

    def test_no_operators_error(self, temp_directory):
        """Test error when no operators available."""
        mock_registry = MagicMock()
        mock_registry.list_available_operators.return_value = []
        mock_registry.list_available_benchmarks.return_value = ["benchmark1"]

        runner = BenchmarkRunner(registry=mock_registry, output_dir=str(temp_directory))

        with pytest.raises(ValueError, match="No operators available"):
            runner.run_comprehensive_benchmark()

    def test_no_benchmarks_error(self, temp_directory):
        """Test error when no benchmarks available."""
        mock_registry = MagicMock()
        mock_registry.list_available_operators.return_value = ["op1"]
        mock_registry.list_available_benchmarks.return_value = []

        runner = BenchmarkRunner(registry=mock_registry, output_dir=str(temp_directory))

        with pytest.raises(ValueError, match="No benchmarks available"):
            runner.run_comprehensive_benchmark()
