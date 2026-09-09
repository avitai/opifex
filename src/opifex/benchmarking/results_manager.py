"""Results persistence and publication output for the benchmarking system.

``ResultsManager`` keeps a JSON database of saved results, writes each one
through to a calibrax ``Store``, answers queries over the database, and renders
publication plots and tables with calibrax's ``PublicationGenerator``.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from calibrax.core import BenchmarkResult
from calibrax.core.models import Metric, Point, Run, TrendPoint, TrendSeries
from calibrax.exporters.publication import PublicationGenerator
from calibrax.storage.store import Store

from opifex.benchmarking.adapters import default_metric_defs, metric_values, results_to_run


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


logger = logging.getLogger(__name__)

type PlotType = Literal["comparison", "scaling", "convergence"]
type TableFormat = Literal["latex", "html", "csv"]

_EXECUTION_TIME = "execution_time"
_LOSS = "loss"


class ResultsManager:
    """Persist benchmark results and render publication output.

    Args:
        storage_path: Directory holding the database, the raw results, the calibrax
            store, and the ``plots`` and ``tables`` output directories.
        database_path: The database file; ``<storage_path>/benchmark_database.json``
            by default; a file that exists and is not valid JSON is a ``ValueError``.
    """

    def __init__(
        self,
        storage_path: str = "./benchmark_results",
        database_path: str | None = None,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.database_path = (
            self.storage_path / "benchmark_database.json"
            if database_path is None
            else Path(database_path)
        )
        self.plots_path = self.storage_path / "plots"
        self.tables_path = self.storage_path / "tables"
        self.raw_results_path = self.storage_path / "raw_results"
        for directory in (
            self.storage_path,
            self.plots_path,
            self.tables_path,
            self.raw_results_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self._store = Store(self.storage_path / "store")
        self._tables = PublicationGenerator(self.tables_path)
        self.database = self._load_database()

    def _load_database(self) -> dict[str, Any]:
        """Read the database, or start an empty one when the file is absent.

        Returns:
            The database record.

        Raises:
            ValueError: If the file is not valid JSON.
        """
        if not self.database_path.exists():
            return {"results": [], "metadata": {}}
        try:
            with self.database_path.open() as handle:
                return json.load(handle)
        except json.JSONDecodeError as err:
            raise ValueError(f"{self.database_path} is not a valid benchmark database") from err

    def _save_database(self) -> None:
        """Write the database."""
        with self.database_path.open("w") as handle:
            json.dump(self.database, handle, indent=2)

    def save_benchmark_results(
        self,
        result: BenchmarkResult,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a result to the raw results, the database and the calibrax store.

        Args:
            result: The result to save.
            extra_metadata: Recorded next to the result and in its database entry.

        Returns:
            The identifier of the saved result.
        """
        dataset = result.tags.get("dataset", "unknown")
        result_id = f"{result.name}_{dataset}_{datetime.now(UTC):%Y%m%d_%H%M%S}"

        result_data = result.to_dict()
        if extra_metadata:
            result_data["extra_metadata"] = extra_metadata
        result_file = self.raw_results_path / f"{result_id}.json"
        with result_file.open("w") as handle:
            json.dump(result_data, handle, indent=2)

        entry: dict[str, Any] = {
            "id": result_id,
            "name": result.name,
            "dataset": dataset,
            "timestamp": result.timestamp,
            "file_path": str(result_file),
            "metrics_summary": metric_values(result),
            "execution_time": result.metadata.get(_EXECUTION_TIME, 0.0),
        }
        if extra_metadata:
            entry["extra_metadata"] = extra_metadata
        self.database["results"].append(entry)
        self._save_database()

        try:
            self._store.save(results_to_run([result], metric_defs=default_metric_defs()))
        except (OSError, ValueError, TypeError):
            logger.warning(
                "Failed to write result %s to the calibrax store", result_id, exc_info=True
            )

        return result_id

    def load_results(self, result_id: str) -> BenchmarkResult | None:
        """Load a saved result by identifier; ``None`` when the identifier is unknown."""
        for entry in self.database["results"]:
            if entry["id"] == result_id:
                with Path(entry["file_path"]).open() as handle:
                    result_data = json.load(handle)
                result_data.pop("extra_metadata", None)
                return BenchmarkResult.from_dict(result_data)
        return None

    def query_results(
        self,
        name: str | None = None,
        dataset: str | None = None,
        metric_filter: dict[str, tuple[float, float]] | None = None,
    ) -> list[dict[str, Any]]:
        """Database entries matching every given filter.

        Args:
            name: Benchmark name.
            dataset: Dataset tag.
            metric_filter: Inclusive ``{metric: (low, high)}`` ranges every entry
                must satisfy; an entry without the metric does not match.

        Returns:
            The matching entries.
        """
        entries: Iterable[dict[str, Any]] = self.database["results"]
        if name:
            entries = [entry for entry in entries if entry["name"] == name]
        if dataset:
            entries = [entry for entry in entries if entry["dataset"] == dataset]
        if metric_filter:
            entries = [
                entry for entry in entries if _within(entry["metrics_summary"], metric_filter)
            ]
        return list(entries)

    def get_database_statistics(self) -> dict[str, Any]:
        """Counts per name and dataset, and execution time statistics."""
        entries = self.database["results"]
        name_counts: dict[str, int] = defaultdict(int)
        dataset_counts: dict[str, int] = defaultdict(int)
        for entry in entries:
            name_counts[entry["name"]] += 1
            dataset_counts[entry["dataset"]] += 1
        stats: dict[str, Any] = {
            "total_results": len(entries),
            "unique_names": len(name_counts),
            "unique_datasets": len(dataset_counts),
            "name_counts": dict(name_counts),
            "dataset_counts": dict(dataset_counts),
        }
        times = [
            entry[_EXECUTION_TIME] for entry in entries if entry.get(_EXECUTION_TIME) is not None
        ]
        if times:
            stats["execution_time_stats"] = {
                "mean": statistics.fmean(times),
                "std": statistics.pstdev(times),
                "min": min(times),
                "max": max(times),
            }
        return stats

    def export_database(self, export_path: str, output_format: str = "json") -> None:
        """Write the whole database to ``export_path``.

        Args:
            export_path: The file to write.
            output_format: ``"json"``, the one supported format.

        Raises:
            ValueError: If ``output_format`` is not ``"json"``.
        """
        if output_format != "json":
            raise ValueError(f"Unsupported export format: {output_format}")
        with Path(export_path).open("w") as handle:
            json.dump(self.database, handle, indent=2)

    def export_publication_plots(
        self,
        results: list[BenchmarkResult],
        plot_type: PlotType = "comparison",
        output_format: str = "png",
    ) -> list[Path]:
        """Render publication plots under ``plots_path``.

        ``comparison`` draws one figure per dataset with at least two results
        (``<dataset>/comparison``), ``scaling`` one figure per model and metric over
        the ``problem_size`` metadata (``<model>/scaling_<metric>``), and
        ``convergence`` one figure per result carrying a ``loss_history``
        (``<model>/<dataset>/convergence_loss``).

        Args:
            results: The results to plot.
            plot_type: Which figures to draw.
            output_format: Image format (``png``, ``pdf``, ``svg``).

        Returns:
            The written figures; empty when no result carries the data the plot needs.
        """
        renderers: dict[PlotType, Callable[[list[BenchmarkResult], str], list[Path]]] = {
            "comparison": self._comparison_plots,
            "scaling": self._scaling_plots,
            "convergence": self._convergence_plots,
        }
        return renderers[plot_type](results, output_format)

    def _comparison_plots(self, results: list[BenchmarkResult], output_format: str) -> list[Path]:
        """One bar-chart figure per dataset comparing the models measured on it."""
        paths: list[Path] = []
        for dataset, group in _grouped(results, lambda r: r.tags.get("dataset", r.name)).items():
            if len(group) < 2:
                continue
            run = results_to_run(group, metric_defs=default_metric_defs())
            generator = PublicationGenerator(self.plots_path / dataset)
            path = generator.generate_comparison_plot(run, output_format=output_format)
            if path is not None:
                paths.append(path)
        return paths

    def _scaling_plots(self, results: list[BenchmarkResult], output_format: str) -> list[Path]:
        """One figure per model and metric over the problem sizes it was measured at."""
        paths: list[Path] = []
        for model, group in _grouped(results, lambda r: r.name).items():
            sized = sorted(
                (r for r in group if r.metadata.get("problem_size") is not None),
                key=lambda r: int(r.metadata["problem_size"]),
            )
            sizes = [int(r.metadata["problem_size"]) for r in sized]
            if len(set(sizes)) < 2:
                continue
            generator = PublicationGenerator(self.plots_path / model)
            for metric_name, values in _shared_metrics(sized).items():
                path = generator.generate_scaling_plot(
                    sizes, values, metric_name=metric_name, output_format=output_format
                )
                if path is not None:
                    paths.append(path)
        return paths

    def _convergence_plots(self, results: list[BenchmarkResult], output_format: str) -> list[Path]:
        """One loss-curve figure per result that recorded a loss history."""
        paths: list[Path] = []
        for result in results:
            history = result.metadata.get("loss_history")
            if not history:
                continue
            recorded = datetime.fromtimestamp(result.timestamp, UTC)
            series = TrendSeries(
                metric=_LOSS,
                point_name=result.name,
                tags=dict(result.tags),
                points=tuple(
                    TrendPoint(run_id=f"epoch-{epoch}", timestamp=recorded, value=float(value))
                    for epoch, value in enumerate(history, start=1)
                ),
            )
            dataset = result.tags.get("dataset", "unknown")
            generator = PublicationGenerator(self.plots_path / result.name / dataset)
            path = generator.generate_convergence_plot(series, output_format=output_format)
            if path is not None:
                paths.append(path)
        return paths

    def generate_comparison_tables(
        self,
        operators: list[str],
        metrics: list[str],
        output_format: TableFormat = "latex",
    ) -> Path:
        """Render a table of the latest saved result per operator and dataset.

        Rows are labelled ``<operator> (<dataset>)``; the best value of each metric
        is marked in the LaTeX and HTML formats.

        Args:
            operators: The operators to include, in row order.
            metrics: The metric columns.
            output_format: ``latex``, ``html`` or ``csv``.

        Returns:
            The written table, ``tables_path / table.<ext>``.
        """
        points: list[Point] = []
        for operator in operators:
            entries = [entry for entry in self.database["results"] if entry["name"] == operator]
            for dataset, entry in sorted(_latest_per_dataset(entries).items()):
                summary = entry["metrics_summary"]
                points.append(
                    Point(
                        name=operator,
                        scenario=dataset,
                        tags={"dataset": dataset, "table_row": f"{operator} ({dataset})"},
                        metrics={
                            name: Metric(value=float(summary[name]))
                            for name in metrics
                            if name in summary
                        },
                    )
                )
        run = Run(points=tuple(points), metric_defs=default_metric_defs())
        return self._tables.generate_table(
            run, metrics, output_format=output_format, group_by_tag="table_row"
        )


def _within(summary: dict[str, float], metric_filter: dict[str, tuple[float, float]]) -> bool:
    """Whether every filtered metric is present and inside its range."""
    return all(
        summary.get(metric) is not None and low <= summary[metric] <= high
        for metric, (low, high) in metric_filter.items()
    )


def _grouped(
    results: list[BenchmarkResult], key: Callable[[BenchmarkResult], str]
) -> dict[str, list[BenchmarkResult]]:
    """Results by ``key``, in first-seen order."""
    groups: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        groups[key(result)].append(result)
    return dict(groups)


def _shared_metrics(results: list[BenchmarkResult]) -> dict[str, list[float]]:
    """The metrics every result records, each as the list of values in result order."""
    values = [metric_values(result) for result in results]
    shared = set.intersection(*(set(v) for v in values)) if values else set()
    return {name: [v[name] for v in values] for name in sorted(shared)}


def _latest_per_dataset(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The most recently recorded entry for each dataset."""
    latest: dict[str, dict[str, Any]] = {}
    for entry in entries:
        dataset = entry["dataset"]
        if dataset not in latest or entry["timestamp"] > latest[dataset]["timestamp"]:
            latest[dataset] = entry
    return latest
