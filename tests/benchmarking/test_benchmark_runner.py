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
    BenchmarkFailure,
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

        with patch("opifex.benchmarking.benchmark_runner.BenchmarkRegistry") as mock_registry:
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

        with patch("opifex.benchmarking.benchmark_runner.BenchmarkRegistry") as mock_registry:
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


class TestBenchmarkRunnerFailureRecording:
    """A failing benchmark must be recorded and surfaced, not silently masked."""

    def _build_runner(self, temp_directory, failing_operator):
        """Construct a runner whose _run_single_benchmark fails for one operator."""
        mock_registry = MagicMock()
        mock_registry.list_available_operators.return_value = ["good_op", "bad_op"]
        mock_registry.list_available_benchmarks.return_value = ["bench1"]
        mock_registry.list_compatible_operators.return_value = ["good_op", "bad_op"]
        mock_registry.get_benchmark_config.return_value = MagicMock()

        runner = BenchmarkRunner(
            registry=mock_registry,
            evaluator=MagicMock(),
            validator=MagicMock(),
            analyzer=MagicMock(),
            results_manager=MagicMock(),
            output_dir=str(temp_directory),
        )

        good_result = MagicMock(name="good_result")

        def fake_single(operator_name, _config):
            if operator_name == failing_operator:
                raise RuntimeError("operator blew up during execution")
            return good_result

        runner._run_single_benchmark = MagicMock(side_effect=fake_single)
        runner._validate_result = MagicMock(return_value=MagicMock())
        return runner, good_result

    def test_failing_operator_does_not_abort_suite(self, temp_directory):
        """The healthy operator still produces a result despite the failure."""
        runner, good_result = self._build_runner(temp_directory, failing_operator="bad_op")

        results = runner.run_comprehensive_benchmark(generate_analysis=False)

        # Loop continued: the good operator's result is present.
        assert results["bench1"]["good_op"] is good_result
        # The failing operator left no successful result.
        assert "bad_op" not in results["bench1"]

    def test_failing_operator_is_recorded_not_masked(self, temp_directory):
        """The failure is captured on the runner so callers can inspect it."""
        runner, _ = self._build_runner(temp_directory, failing_operator="bad_op")

        runner.run_comprehensive_benchmark(generate_analysis=False)

        assert len(runner.failed_runs) == 1
        failure = runner.failed_runs[0]
        assert isinstance(failure, BenchmarkFailure)
        assert failure.benchmark_name == "bench1"
        assert failure.operator_name == "bad_op"
        assert isinstance(failure.error, RuntimeError)
        assert "blew up" in str(failure.error)

    def test_failures_logged_via_logger_exception(self, temp_directory, caplog):
        """The failure is logged at ERROR level with a traceback."""
        import logging

        runner, _ = self._build_runner(temp_directory, failing_operator="bad_op")

        with caplog.at_level(logging.ERROR):
            runner.run_comprehensive_benchmark(generate_analysis=False)

        assert any(
            "bad_op" in record.message and record.levelno == logging.ERROR
            for record in caplog.records
        )
