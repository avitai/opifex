"""Tests for BenchmarkResult → calibrax Run adapter."""

from __future__ import annotations

import pytest
from calibrax.core.models import Metric, MetricDef, MetricDirection, MetricPriority, Run

from opifex.benchmarking.adapters import default_metric_defs, results_to_run


@pytest.fixture
def sample_results() -> list:
    """Create sample BenchmarkResult objects for testing."""
    from calibrax.core.result import BenchmarkResult

    return [
        BenchmarkResult(
            name="FNO",
            tags={"dataset": "darcy", "resolution": "64"},
            metrics={
                "mse": Metric(value=0.001),
                "mae": Metric(value=0.01),
                "relative_error": Metric(value=0.05),
            },
            metadata={"execution_time": 1.5},
        ),
        BenchmarkResult(
            name="UNO",
            tags={"dataset": "darcy", "resolution": "64"},
            metrics={
                "mse": Metric(value=0.002),
                "mae": Metric(value=0.02),
            },
            metadata={"execution_time": 2.0},
        ),
    ]


class TestResultsToRun:
    """Tests for results_to_run() adapter function."""

    def test_converts_results_to_run(self, sample_results: list) -> None:
        """Run has correct number of points."""
        run = results_to_run(sample_results)
        assert isinstance(run, Run)
        assert len(run.points) == 2

    def test_maps_name_to_point_name(self, sample_results: list) -> None:
        """BenchmarkResult.name maps to Point.name."""
        run = results_to_run(sample_results)
        assert run.points[0].name == "FNO"
        assert run.points[1].name == "UNO"

    def test_maps_dataset_tag_to_scenario(self, sample_results: list) -> None:
        """BenchmarkResult.tags['dataset'] maps to Point.scenario."""
        run = results_to_run(sample_results)
        assert run.points[0].scenario == "darcy"
        assert run.points[1].scenario == "darcy"

    def test_maps_tags_to_point_tags(self, sample_results: list) -> None:
        """BenchmarkResult.tags pass through to Point.tags."""
        run = results_to_run(sample_results)
        assert run.points[0].tags["resolution"] == "64"

    def test_maps_metrics_directly(self, sample_results: list) -> None:
        """BenchmarkResult.metrics pass through to Point.metrics (same Metric type)."""
        run = results_to_run(sample_results)
        assert run.points[0].metrics["mse"].value == pytest.approx(0.001)
        assert run.points[0].metrics["mae"].value == pytest.approx(0.01)

    def test_empty_results_gives_empty_run(self) -> None:
        """Empty result list produces a Run with zero points."""
        run = results_to_run([])
        assert len(run.points) == 0

    def test_missing_dataset_tag_uses_unknown(self) -> None:
        """Missing 'dataset' tag defaults scenario to 'unknown'."""
        from calibrax.core.result import BenchmarkResult

        result = BenchmarkResult(name="FNO", tags={"framework": "flax"})
        run = results_to_run([result])
        assert run.points[0].scenario == "unknown"

    def test_accepts_commit_and_branch(self, sample_results: list) -> None:
        """Optional commit and branch are forwarded to the Run."""
        run = results_to_run(sample_results, commit="abc123", branch="main")
        assert run.commit == "abc123"
        assert run.branch == "main"

    def test_attaches_metric_defs(self, sample_results: list) -> None:
        """Metric definitions are attached to the Run."""
        defs = default_metric_defs()
        run = results_to_run(sample_results, metric_defs=defs)
        assert "mse" in run.metric_defs
        assert run.metric_defs["mse"].direction == MetricDirection.LOWER

    def test_roundtrip_serialization(self, sample_results: list) -> None:
        """Run can be serialized and deserialized without data loss."""
        run = results_to_run(sample_results, metric_defs=default_metric_defs())
        data = run.to_dict()
        restored = Run.from_dict(data)
        assert len(restored.points) == len(run.points)
        assert restored.points[0].name == run.points[0].name


class TestDefaultMetricDefs:
    """Tests for default_metric_defs() factory."""

    def test_returns_metric_defs_dict(self) -> None:
        """Returns a dict[str, MetricDef]."""
        defs = default_metric_defs()
        assert isinstance(defs, dict)
        for key, md in defs.items():
            assert isinstance(key, str)
            assert isinstance(md, MetricDef)

    def test_contains_standard_metrics(self) -> None:
        """Contains MSE, MAE, relative_error, R2."""
        defs = default_metric_defs()
        assert "mse" in defs
        assert "mae" in defs
        assert "relative_error" in defs
        assert "r2_score" in defs

    def test_error_metrics_are_lower_is_better(self) -> None:
        """Error metrics have direction=LOWER."""
        defs = default_metric_defs()
        assert defs["mse"].direction == MetricDirection.LOWER
        assert defs["mae"].direction == MetricDirection.LOWER
        assert defs["relative_error"].direction == MetricDirection.LOWER

    def test_r2_is_higher_is_better(self) -> None:
        """R2 metric has direction=HIGHER."""
        defs = default_metric_defs()
        assert defs["r2_score"].direction == MetricDirection.HIGHER

    def test_primary_metrics_marked(self) -> None:
        """MSE and relative_error are primary metrics."""
        defs = default_metric_defs()
        assert defs["mse"].priority == MetricPriority.PRIMARY
        assert defs["relative_error"].priority == MetricPriority.PRIMARY
