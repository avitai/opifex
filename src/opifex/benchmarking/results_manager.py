"""Results Manager for Opifex Advanced Benchmarking System.

Data persistence and publication-ready export capabilities.
Provides results storage, publication plot generation, comparison tables,
and benchmark database management. Each saved result is also persisted to a
calibrax Store for cross-tool interoperability.
"""

import json
import logging
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Literal

import numpy as np
from calibrax.core import BenchmarkResult
from calibrax.storage.store import Store

from opifex.benchmarking.adapters import default_metric_defs, results_to_run


logger = logging.getLogger(__name__)


class ResultsManager:
    """Data persistence and publication-ready export capabilities.

    Provides:
    - Persistent storage of benchmark results with metadata
    - calibrax Store write-through for cross-tool interoperability
    - Publication-ready plot and table generation
    - Benchmark database maintenance and querying
    - Export formats for different publication venues
    """

    def __init__(
        self,
        storage_path: str = "./benchmark_results",
        database_path: str | None = None,
    ) -> None:
        """Initialize results manager.

        Args:
            storage_path: Base path for storing benchmark results.
            database_path: Path to benchmark database file.
        """
        self.storage_path = Path(storage_path)
        if database_path is None:
            self.database_path = self.storage_path / "benchmark_database.json"
        else:
            self.database_path = Path(database_path)

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.plots_path = self.storage_path / "plots"
        self.plots_path.mkdir(exist_ok=True)
        self.tables_path = self.storage_path / "tables"
        self.tables_path.mkdir(exist_ok=True)
        self.raw_results_path = self.storage_path / "raw_results"
        self.raw_results_path.mkdir(exist_ok=True)

        self._store = Store(self.storage_path / "store")
        self.database = self._load_database()

    def _load_database(self) -> dict[str, Any]:
        """Load benchmark database from file."""
        if self.database_path.exists():
            try:
                with open(self.database_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {"results": [], "metadata": {}}
        return {"results": [], "metadata": {}}

    def _save_database(self) -> None:
        """Save benchmark database to file."""
        with open(self.database_path, "w") as f:
            json.dump(self.database, f, indent=2)

    def save_benchmark_results(
        self,
        result: BenchmarkResult,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save benchmark results with metadata.

        Args:
            result: Benchmark result to save.
            extra_metadata: Additional metadata to store alongside.

        Returns:
            Unique identifier for saved results.
        """
        dataset = result.tags.get("dataset", "unknown")
        timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        result_id = f"{result.name}_{dataset}_{timestamp_str}"

        result_data = result.to_dict()
        if extra_metadata:
            result_data["extra_metadata"] = extra_metadata

        result_file = self.raw_results_path / f"{result_id}.json"
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)

        exec_time = result.metadata.get("execution_time", 0.0)
        metrics_summary = {k: v.value for k, v in result.metrics.items()}

        database_entry = {
            "id": result_id,
            "name": result.name,
            "dataset": dataset,
            "timestamp": result.timestamp,
            "file_path": str(result_file),
            "metrics_summary": metrics_summary,
            "execution_time": exec_time,
        }

        if extra_metadata:
            database_entry["extra_metadata"] = extra_metadata

        self.database["results"].append(database_entry)
        self._save_database()

        # Persist to calibrax Store for cross-tool interop
        try:
            run = results_to_run([result], metric_defs=default_metric_defs())
            self._store.save(run)
        except (OSError, ValueError, TypeError):
            logger.warning("Failed to write result to calibrax Store", exc_info=True)

        return result_id

    def load_result(self, result_id: str) -> BenchmarkResult | None:
        """Load benchmark result by ID.

        Args:
            result_id: Unique identifier for results.

        Returns:
            Loaded BenchmarkResult or None if not found.
        """
        entry = None
        for result_entry in self.database["results"]:
            if result_entry["id"] == result_id:
                entry = result_entry
                break

        if entry is None:
            return None

        try:
            with open(entry["file_path"]) as f:
                result_data = json.load(f)
            result_data.pop("extra_metadata", None)
            return BenchmarkResult.from_dict(result_data)
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def load_results(self, result_id: str) -> BenchmarkResult | None:
        """Load benchmark results by ID.

        Alias for :meth:`load_result` for backward compatibility.
        """
        return self.load_result(result_id)

    def query_results(
        self,
        name: str | None = None,
        dataset: str | None = None,
        metric_filter: dict[str, tuple[float, float]] | None = None,
    ) -> list[dict[str, Any]]:
        """Query benchmark database with filters.

        Args:
            name: Filter by benchmark name.
            dataset: Filter by dataset tag.
            metric_filter: Filter by metric ranges {metric: (min, max)}.

        Returns:
            List of matching database entries.
        """
        results = self.database["results"]

        if name:
            results = [r for r in results if r["name"] == name]

        if dataset:
            results = [r for r in results if r["dataset"] == dataset]

        if metric_filter:
            filtered = []
            for result in results:
                matches = True
                for metric, (min_val, max_val) in metric_filter.items():
                    metric_val = result["metrics_summary"].get(metric)
                    if metric_val is None or not (min_val <= metric_val <= max_val):
                        matches = False
                        break
                if matches:
                    filtered.append(result)
            results = filtered

        return results

    def get_database_statistics(self) -> dict[str, Any]:
        """Get statistics about the benchmark database.

        Returns:
            Database statistics summary.
        """
        results = self.database["results"]

        if not results:
            return {
                "total_results": 0,
                "unique_names": 0,
                "unique_datasets": 0,
                "name_counts": {},
                "dataset_counts": {},
            }

        name_counts: dict[str, int] = {}
        dataset_counts: dict[str, int] = {}

        for result in results:
            n = result["name"]
            d = result["dataset"]
            name_counts[n] = name_counts.get(n, 0) + 1
            dataset_counts[d] = dataset_counts.get(d, 0) + 1

        exec_times = [
            r["execution_time"] for r in results if r.get("execution_time") is not None
        ]

        stats: dict[str, Any] = {
            "total_results": len(results),
            "unique_names": len(name_counts),
            "unique_datasets": len(dataset_counts),
            "name_counts": name_counts,
            "dataset_counts": dataset_counts,
        }

        if exec_times:
            stats["execution_time_stats"] = {
                "mean": float(np.mean(exec_times)),
                "std": float(np.std(exec_times)),
                "min": float(np.min(exec_times)),
                "max": float(np.max(exec_times)),
            }

        return stats

    def create_benchmark_database_entry(
        self, result: BenchmarkResult
    ) -> dict[str, Any]:
        """Create standardized database entry for benchmark results.

        Args:
            result: Benchmark result.

        Returns:
            Standardized database entry dictionary.
        """
        return {
            "id": (
                f"{result.name}"
                f"_{result.tags.get('dataset', 'unknown')}"
                f"_{result.timestamp}"
            ),
            "name": result.name,
            "dataset": result.tags.get("dataset", "unknown"),
            "timestamp": result.timestamp,
            "metrics": {k: v.value for k, v in result.metrics.items()},
            "execution_time": result.metadata.get("execution_time", 0.0),
            "memory_usage": result.metadata.get("memory_usage"),
            "config": result.config,
        }

    def export_database(self, export_path: str, output_format: str = "json") -> None:
        """Export entire benchmark database.

        Args:
            export_path: Path to export file.
            output_format: Export format (``"json"``).
        """
        export_file = Path(export_path)

        if output_format == "json":
            with open(export_file, "w") as f:
                json.dump(self.database, f, indent=2)
        else:
            msg = f"Unsupported export format: {output_format}"
            raise ValueError(msg)

    def export_publication_plots(
        self,
        results: list[BenchmarkResult],
        plot_type: Literal["comparison", "scaling", "convergence"] = "comparison",
        output_format: str = "png",
    ) -> list[Path]:
        """Export publication-ready plots.

        Args:
            results: List of benchmark results to plot.
            plot_type: Type of plot to generate.
            output_format: Output format (png, pdf, svg).

        Returns:
            List of paths to generated plot files.
        """
        if plot_type == "comparison":
            return _generate_comparison_plots(results, self.plots_path, output_format)
        if plot_type == "scaling":
            return _generate_scaling_plots(results, self.plots_path, output_format)
        if plot_type == "convergence":
            return _generate_convergence_plots(results, self.plots_path, output_format)
        return []

    def generate_comparison_tables(
        self,
        operators: list[str],
        metrics: list[str],
        output_format: Literal["latex", "html", "csv"] = "latex",
    ) -> Path:
        """Generate publication-ready comparison tables.

        Queries the local benchmark database and generates a formatted
        comparison table in the requested output format.

        Args:
            operators: List of operator names to include.
            metrics: List of metrics to include in table.
            output_format: Output format.

        Returns:
            Path to generated table file.
        """
        table_data: list[dict[str, Any]] = []

        for operator in operators:
            operator_results = [
                entry for entry in self.database["results"] if entry["name"] == operator
            ]

            if operator_results:
                datasets: dict[str, dict[str, Any]] = {}
                for result in operator_results:
                    ds = result["dataset"]
                    if (
                        ds not in datasets
                        or result["timestamp"] > datasets[ds]["timestamp"]
                    ):
                        datasets[ds] = result

                for ds, result in datasets.items():
                    row: dict[str, Any] = {
                        "Operator": operator,
                        "Dataset": ds,
                    }
                    for metric in metrics:
                        row[metric] = result["metrics_summary"].get(metric, "N/A")
                    table_data.append(row)

        timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        table_file = (
            self.tables_path / f"comparison_table_{timestamp_str}.{output_format}"
        )

        _generate_table_fallback(table_data, table_file, metrics, output_format)

        return table_file


# ---------------------------------------------------------------------------
# Plot generation helpers
# ---------------------------------------------------------------------------


def _generate_comparison_plots(
    results: list[BenchmarkResult],
    plots_path: Path,
    output_format: str,
) -> list[Path]:
    """Generate comparison plots between different models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plot_files: list[Path] = []
    datasets = list({r.tags.get("dataset", r.name) for r in results})

    for dataset in datasets:
        dataset_results = [
            r for r in results if r.tags.get("dataset", r.name) == dataset
        ]
        if len(dataset_results) < 2:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Performance Comparison - {dataset}", fontsize=16)

        models = [r.name for r in dataset_results]

        mse_values = [
            r.metrics["mse"].value for r in dataset_results if "mse" in r.metrics
        ]
        if len(mse_values) == len(dataset_results):
            axes[0, 0].bar(models, mse_values)
            axes[0, 0].set_title("Mean Squared Error")
            axes[0, 0].set_ylabel("MSE")
            axes[0, 0].tick_params(axis="x", rotation=45)

        exec_times = [r.metadata.get("execution_time", 0.0) for r in dataset_results]
        axes[0, 1].bar(models, exec_times)
        axes[0, 1].set_title("Execution Time")
        axes[0, 1].set_ylabel("Time (s)")
        axes[0, 1].tick_params(axis="x", rotation=45)

        mae_values = [
            r.metrics["mae"].value for r in dataset_results if "mae" in r.metrics
        ]
        if len(mae_values) == len(dataset_results):
            axes[1, 0].bar(models, mae_values)
            axes[1, 0].set_title("Mean Absolute Error")
            axes[1, 0].set_ylabel("MAE")
            axes[1, 0].tick_params(axis="x", rotation=45)

        rel_values = [
            r.metrics["relative_error"].value
            for r in dataset_results
            if "relative_error" in r.metrics
        ]
        if len(rel_values) == len(dataset_results):
            axes[1, 1].bar(models, rel_values)
            axes[1, 1].set_title("Relative Error")
            axes[1, 1].set_ylabel("Relative Error")
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plot_file = plots_path / f"comparison_{dataset}.{output_format}"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        plot_files.append(plot_file)

    return plot_files


def _generate_scaling_plots(
    results: list[BenchmarkResult],
    plots_path: Path,
    output_format: str,
) -> list[Path]:
    """Generate scaling behavior plots."""
    # Placeholder for future implementation
    _ = results, plots_path, output_format
    return []


def _generate_convergence_plots(
    results: list[BenchmarkResult],
    plots_path: Path,
    output_format: str,
) -> list[Path]:
    """Generate convergence analysis plots."""
    # Placeholder for future implementation
    _ = results, plots_path, output_format
    return []


# ---------------------------------------------------------------------------
# Table generation helpers
# ---------------------------------------------------------------------------


def _generate_table_fallback(
    data: list[dict[str, Any]],
    file_path: Path,
    metrics: list[str],
    output_format: str,
) -> None:
    """Generate a simple table as fallback."""
    if output_format == "latex":
        _generate_latex_table(data, file_path, metrics)
    elif output_format == "html":
        _generate_html_table(data, file_path, metrics)
    elif output_format == "csv":
        _generate_csv_table(data, file_path, metrics)


def _generate_latex_table(
    data: list[dict[str, Any]], file_path: Path, metrics: list[str]
) -> None:
    """Generate LaTeX table."""
    with open(file_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Comparison}\n")
        f.write("\\label{tab:performance_comparison}\n")

        n_cols = 2 + len(metrics)
        f.write(f"\\begin{{tabular}}{{{'|c' * n_cols}|}}\n")
        f.write("\\hline\n")

        headers = ["Operator", "Dataset", *metrics]
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\hline\n")

        for row in data:
            values = [str(row["Operator"]), str(row["Dataset"])]
            for metric in metrics:
                val = row.get(metric, "N/A")
                if isinstance(val, float):
                    values.append(f"{val:.4e}")
                else:
                    values.append(str(val))
            f.write(" & ".join(values) + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def _generate_html_table(
    data: list[dict[str, Any]], file_path: Path, metrics: list[str]
) -> None:
    """Generate HTML table."""
    with open(file_path, "w") as f:
        f.write("<html><body>\n")
        f.write("<table border='1'>\n")
        f.write("<caption>Performance Comparison</caption>\n")

        f.write("<tr>")
        for header in ["Operator", "Dataset", *metrics]:
            f.write(f"<th>{header}</th>")
        f.write("</tr>\n")

        for row in data:
            f.write("<tr>")
            f.write(f"<td>{row['Operator']}</td>")
            f.write(f"<td>{row['Dataset']}</td>")
            for metric in metrics:
                val = row.get(metric, "N/A")
                if isinstance(val, float):
                    f.write(f"<td>{val:.4e}</td>")
                else:
                    f.write(f"<td>{val}</td>")
            f.write("</tr>\n")

        f.write("</table>\n")
        f.write("</body></html>\n")


def _generate_csv_table(
    data: list[dict[str, Any]], file_path: Path, metrics: list[str]
) -> None:
    """Generate CSV table."""
    import csv

    with open(file_path, "w", newline="") as f:
        if data:
            headers = ["Operator", "Dataset", *metrics]
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for row in data:
                csv_row: dict[str, Any] = {
                    "Operator": row["Operator"],
                    "Dataset": row["Dataset"],
                }
                for metric in metrics:
                    csv_row[metric] = row.get(metric, "N/A")
                writer.writerow(csv_row)
