"""ResultsManager persists results and renders publication output through calibrax.

Plots and tables are calibrax's ``PublicationGenerator``; the manager groups
opifex results into the runs and series it takes.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import matplotlib as mpl
import pytest
from calibrax.core import BenchmarkResult
from calibrax.core.models import Metric

from opifex.benchmarking.results_manager import ResultsManager


if TYPE_CHECKING:
    from pathlib import Path


mpl.use("Agg")


def _result(
    name: str,
    dataset: str,
    metrics: dict[str, float],
    *,
    execution_time: float = 1.0,
    problem_size: int | None = None,
    loss_history: list[float] | None = None,
) -> BenchmarkResult:
    metadata: dict[str, object] = {"execution_time": execution_time}
    if problem_size is not None:
        metadata["problem_size"] = problem_size
    if loss_history is not None:
        metadata["loss_history"] = loss_history
    return BenchmarkResult(
        name=name,
        tags={"dataset": dataset},
        metrics={key: Metric(value=value) for key, value in metrics.items()},
        metadata=metadata,
    )


@pytest.fixture
def manager(tmp_path: Path) -> ResultsManager:
    return ResultsManager(storage_path=str(tmp_path / "results"))


def _assert_written(paths: list[Path]) -> None:
    assert paths
    for path in paths:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


class TestComparisonPlots:
    def test_one_figure_per_dataset_with_two_or_more_models(self, manager: ResultsManager) -> None:
        results = [
            _result("FNO", "darcy", {"mse": 0.01, "mae": 0.05}, execution_time=1.5),
            _result("DeepONet", "darcy", {"mse": 0.02, "mae": 0.07}, execution_time=0.9),
            _result("FNO", "burgers", {"mse": 0.03}),
        ]

        paths = manager.export_publication_plots(results, plot_type="comparison")

        _assert_written(paths)
        assert paths == [manager.plots_path / "darcy" / "comparison.png"]

    def test_no_dataset_with_two_models_means_no_figure(self, manager: ResultsManager) -> None:
        results = [_result("FNO", "darcy", {"mse": 0.01}), _result("FNO", "burgers", {"mse": 0.02})]

        assert manager.export_publication_plots(results, plot_type="comparison") == []


class TestScalingPlots:
    def test_one_figure_per_model_and_metric_over_problem_sizes(
        self, manager: ResultsManager
    ) -> None:
        sizes = [64, 128, 256, 512]
        results = [
            _result(
                "FNO",
                "darcy",
                {"mse": 1.0 / size},
                execution_time=size / 64.0,
                problem_size=size,
            )
            for size in sizes
        ]

        paths = manager.export_publication_plots(results, plot_type="scaling", output_format="svg")

        _assert_written(paths)
        assert {path.name for path in paths} == {"scaling_execution_time.svg", "scaling_mse.svg"}
        assert {path.parent for path in paths} == {manager.plots_path / "FNO"}

    def test_scaling_needs_two_distinct_sizes(self, manager: ResultsManager) -> None:
        results = [
            _result("FNO", "darcy", {"mse": 0.1}, problem_size=64),
            _result("FNO", "darcy", {"mse": 0.2}, problem_size=64),
            _result("DeepONet", "darcy", {"mse": 0.2}),
        ]

        assert manager.export_publication_plots(results, plot_type="scaling") == []


class TestConvergencePlots:
    def test_one_figure_per_result_with_a_loss_history(self, manager: ResultsManager) -> None:
        history = [10.0 / (epoch + 1) for epoch in range(20)]
        results = [
            _result("FNO", "darcy", {"mse": 0.05}, loss_history=history),
            _result("FNO", "burgers", {"mse": 0.05}, loss_history=history[:5]),
            _result("DeepONet", "darcy", {"mse": 0.05}),
        ]

        paths = manager.export_publication_plots(results, plot_type="convergence")

        _assert_written(paths)
        assert paths == [
            manager.plots_path / "FNO" / "darcy" / "convergence_loss.png",
            manager.plots_path / "FNO" / "burgers" / "convergence_loss.png",
        ]

    def test_no_history_means_no_figure(self, manager: ResultsManager) -> None:
        assert (
            manager.export_publication_plots(
                [_result("FNO", "darcy", {"mse": 0.05})], plot_type="convergence"
            )
            == []
        )


class TestComparisonTables:
    def test_latest_result_per_operator_and_dataset_makes_one_row(
        self, manager: ResultsManager
    ) -> None:
        manager.save_benchmark_results(_result("FNO", "darcy", {"mse": 0.02, "mae": 0.05}))
        manager.save_benchmark_results(_result("FNO", "darcy", {"mse": 0.01, "mae": 0.04}))
        manager.save_benchmark_results(
            _result("DeepONet", "darcy", {"mse": 0.03}, execution_time=2.0)
        )
        manager.save_benchmark_results(_result("UNet", "darcy", {"mse": 0.001}))

        path = manager.generate_comparison_tables(
            ["FNO", "DeepONet"], ["mse", "mae", "execution_time"], output_format="csv"
        )

        assert path == manager.tables_path / "table.csv"
        with path.open() as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == ["Framework", "mse", "mae", "execution_time"]
        assert rows[1:] == [
            ["FNO (darcy)", "0.01", "0.04", "1.0"],
            ["DeepONet (darcy)", "0.03", "", "2.0"],
        ]

    def test_latex_table_marks_the_best_value(self, manager: ResultsManager) -> None:
        manager.save_benchmark_results(_result("FNO", "darcy", {"mse": 0.02}))
        manager.save_benchmark_results(_result("DeepONet", "darcy", {"mse": 0.03}))

        path = manager.generate_comparison_tables(["FNO", "DeepONet"], ["mse"])

        assert path == manager.tables_path / "table.tex"
        assert "\\textbf{0.0200}" in path.read_text()

    def test_unknown_format_is_an_error(self, manager: ResultsManager) -> None:
        with pytest.raises(ValueError, match="markdown"):
            manager.generate_comparison_tables(["FNO"], ["mse"], output_format="markdown")  # type: ignore[arg-type]


class TestPersistence:
    def test_saved_results_round_trip_and_are_queryable(self, manager: ResultsManager) -> None:
        result = _result("FNO", "darcy", {"mse": 0.01, "mae": 0.05}, execution_time=1.5)

        result_id = manager.save_benchmark_results(result, extra_metadata={"gpu": "a100"})

        loaded = manager.load_results(result_id)
        assert loaded is not None
        assert loaded.name == "FNO"
        assert loaded.metrics["mse"].value == 0.01
        assert manager.load_results("missing") is None
        (entry,) = manager.query_results(name="FNO", metric_filter={"mse": (0.0, 0.02)})
        assert entry["id"] == result_id
        assert entry["metrics_summary"] == {"mse": 0.01, "mae": 0.05, "execution_time": 1.5}
        assert entry["extra_metadata"] == {"gpu": "a100"}
        assert manager.query_results(dataset="burgers") == []

    def test_statistics_summarise_the_database(self, manager: ResultsManager) -> None:
        assert manager.get_database_statistics()["total_results"] == 0
        manager.save_benchmark_results(_result("FNO", "darcy", {"mse": 0.01}, execution_time=1.0))
        manager.save_benchmark_results(_result("FNO", "burgers", {"mse": 0.01}, execution_time=3.0))

        stats = manager.get_database_statistics()

        assert stats["total_results"] == 2
        assert stats["name_counts"] == {"FNO": 2}
        assert stats["dataset_counts"] == {"darcy": 1, "burgers": 1}
        assert stats["execution_time_stats"] == {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0}

    def test_corrupt_database_is_an_error_not_an_empty_database(self, tmp_path: Path) -> None:
        storage = tmp_path / "results"
        storage.mkdir()
        (storage / "benchmark_database.json").write_text("{not json")

        with pytest.raises(ValueError, match=r"benchmark_database\.json"):
            ResultsManager(storage_path=str(storage))

    def test_export_database_writes_json(self, manager: ResultsManager, tmp_path: Path) -> None:
        manager.save_benchmark_results(_result("FNO", "darcy", {"mse": 0.01}))

        manager.export_database(str(tmp_path / "export.json"))

        exported = json.loads((tmp_path / "export.json").read_text())
        assert [entry["name"] for entry in exported["results"]] == ["FNO"]
        with pytest.raises(ValueError, match="yaml"):
            manager.export_database(str(tmp_path / "export.yaml"), output_format="yaml")
