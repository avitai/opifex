"""Results Manager for Opifex Advanced Benchmarking System

Data persistence and publication-ready export capabilities.
Provides results storage, publication plot generation, comparison tables,
and benchmark database management.
"""

import json
import pickle  # nosec B403 # Used for legitimate data serialization in scientific computing
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Literal

import numpy as np

from opifex.benchmarking.evaluation_engine import BenchmarkResult


class ResultsManager:
    """Data persistence and publication-ready export capabilities.

    This manager provides comprehensive results management including:
    - Persistent storage of benchmark results with metadata
    - Publication-ready plot and table generation
    - Benchmark database maintenance and querying
    - Export formats for different publication venues
    """

    def __init__(
        self,
        storage_path: str = "./benchmark_results",
        database_path: str | None = None,
    ):
        """Initialize results manager.

        Args:
            storage_path: Base path for storing benchmark results
            database_path: Path to benchmark database file
        """
        self.storage_path = Path(storage_path)
        if database_path is None:
            self.database_path = self.storage_path / "benchmark_database.json"
        else:
            self.database_path = Path(database_path)

        # Create directories with proper structure
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.plots_path = self.storage_path / "plots"
        self.plots_path.mkdir(exist_ok=True)
        self.tables_path = self.storage_path / "tables"
        self.tables_path.mkdir(exist_ok=True)
        self.raw_results_path = self.storage_path / "raw_results"
        self.raw_results_path.mkdir(exist_ok=True)

        # Load existing database
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
        self, results: BenchmarkResult, metadata: dict[str, Any] | None = None
    ) -> str:
        """Save benchmark results with metadata.

        Args:
            results: Benchmark results to save
            metadata: Additional metadata

        Returns:
            Unique identifier for saved results
        """
        # Generate unique ID
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        result_id = f"{results.model_name}_{results.dataset_name}_{timestamp}"

        # Prepare result data for storage
        result_data = asdict(results)
        if metadata:
            result_data["additional_metadata"] = metadata

        # Save to individual file in raw_results subdirectory
        result_file = self.raw_results_path / f"{result_id}.json"
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)

        # Add to database
        database_entry = {
            "id": result_id,
            "model_name": results.model_name,
            "dataset_name": results.dataset_name,
            "timestamp": results.timestamp,
            "file_path": str(result_file),
            "metrics_summary": results.metrics,
            "execution_time": results.execution_time,
        }

        if metadata:
            database_entry["metadata"] = metadata

        self.database["results"].append(database_entry)
        self._save_database()

        return result_id

    def load_results(self, result_id: str) -> BenchmarkResult | None:
        """Load benchmark results by ID.

        Args:
            result_id: Unique identifier for results

        Returns:
            Loaded benchmark results or None if not found
        """
        # Find in database
        entry = None
        for result_entry in self.database["results"]:
            if result_entry["id"] == result_id:
                entry = result_entry
                break

        if entry is None:
            return None

        # Load from file
        try:
            with open(entry["file_path"]) as f:
                result_data = json.load(f)

            # Remove additional metadata before creating BenchmarkResult
            result_data.pop("additional_metadata", None)
            return BenchmarkResult(**result_data)

        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def export_publication_plots(
        self,
        results: list[BenchmarkResult],
        plot_type: Literal["comparison", "scaling", "convergence"] = "comparison",
        output_format: str = "png",
    ) -> list[Path]:
        """Export publication-ready plots.

        Args:
            results: List of benchmark results to plot
            plot_type: Type of plot to generate
            format: Output format (png, pdf, svg)

        Returns:
            List of paths to generated plot files
        """
        plot_files = []

        try:
            import matplotlib.pyplot as plt

            try:
                import seaborn as sns  # type: ignore[import-untyped]

                # Set publication style
                plt.style.use("seaborn-v0_8-whitegrid")
                sns.set_palette("husl")
            except ImportError:
                # Use matplotlib without seaborn
                plt.style.use("default")

        except ImportError:
            # Fallback without plotting if matplotlib not available
            return plot_files

        if plot_type == "comparison":
            plot_files.extend(self._generate_comparison_plots(results, output_format))
        elif plot_type == "scaling":
            plot_files.extend(self._generate_scaling_plots(results, output_format))
        elif plot_type == "convergence":
            plot_files.extend(self._generate_convergence_plots(results, output_format))

        return plot_files

    def _generate_comparison_plots(
        self, results: list[BenchmarkResult], output_format: str
    ) -> list[Path]:
        """Generate comparison plots between different models."""
        try:
            import matplotlib.pyplot as plt

            plot_files = []

            # Extract data for plotting
            [r.model_name for r in results]
            datasets = list({r.dataset_name for r in results})

            for dataset in datasets:
                dataset_results = [r for r in results if r.dataset_name == dataset]

                if len(dataset_results) < 2:
                    continue

                # Performance comparison plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"Performance Comparison - {dataset}", fontsize=16)

                # MSE comparison
                if all("mse" in r.metrics for r in dataset_results):
                    models_subset = [r.model_name for r in dataset_results]
                    mse_values = [r.metrics["mse"] for r in dataset_results]

                    axes[0, 0].bar(models_subset, mse_values)
                    axes[0, 0].set_title("Mean Squared Error")
                    axes[0, 0].set_ylabel("MSE")
                    axes[0, 0].tick_params(axis="x", rotation=45)

                # Execution time comparison
                exec_times = [r.execution_time for r in dataset_results]
                axes[0, 1].bar(models_subset, exec_times)
                axes[0, 1].set_title("Execution Time")
                axes[0, 1].set_ylabel("Time (s)")
                axes[0, 1].tick_params(axis="x", rotation=45)

                # MAE comparison
                if all("mae" in r.metrics for r in dataset_results):
                    mae_values = [r.metrics["mae"] for r in dataset_results]
                    axes[1, 0].bar(models_subset, mae_values)
                    axes[1, 0].set_title("Mean Absolute Error")
                    axes[1, 0].set_ylabel("MAE")
                    axes[1, 0].tick_params(axis="x", rotation=45)

                # Relative error comparison
                if all("relative_error" in r.metrics for r in dataset_results):
                    rel_err_values = [
                        r.metrics["relative_error"] for r in dataset_results
                    ]
                    axes[1, 1].bar(models_subset, rel_err_values)
                    axes[1, 1].set_title("Relative Error")
                    axes[1, 1].set_ylabel("Relative Error")
                    axes[1, 1].tick_params(axis="x", rotation=45)

                plt.tight_layout()

                plot_file = self.plots_path / f"comparison_{dataset}.{output_format}"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close()

                plot_files.append(plot_file)

            return plot_files

        except Exception:
            return []

    def _generate_scaling_plots(
        self, results: list[BenchmarkResult], output_format: str
    ) -> list[Path]:
        """Generate scaling behavior plots."""
        # Placeholder - would implement scaling analysis plots
        return []

    def _generate_convergence_plots(
        self, results: list[BenchmarkResult], output_format: str
    ) -> list[Path]:
        """Generate convergence analysis plots."""
        # Placeholder - would implement convergence plots
        return []

    def generate_comparison_tables(
        self,
        operators: list[str],
        metrics: list[str],
        output_format: Literal["latex", "html", "csv"] = "latex",
    ) -> Path:
        """Generate publication-ready comparison tables.

        Args:
            operators: List of operator names to include
            metrics: List of metrics to include in table
            format: Output format

        Returns:
            Path to generated table file
        """
        # Query database for relevant results
        table_data = []

        for operator in operators:
            operator_results = [
                entry
                for entry in self.database["results"]
                if entry["model_name"] == operator
            ]

            if operator_results:
                # Use most recent result for each dataset
                datasets = {}
                for result in operator_results:
                    dataset = result["dataset_name"]
                    if (
                        dataset not in datasets
                        or result["timestamp"] > datasets[dataset]["timestamp"]
                    ):
                        datasets[dataset] = result

                for dataset, result in datasets.items():
                    row = {"Operator": operator, "Dataset": dataset}
                    for metric in metrics:
                        row[metric] = result["metrics_summary"].get(metric, "N/A")
                    table_data.append(row)

        # Generate table in requested format
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        if output_format == "latex":
            table_file = self.tables_path / f"comparison_table_{timestamp}.tex"
            self._generate_latex_table(table_data, table_file, metrics)
        elif output_format == "html":
            table_file = self.tables_path / f"comparison_table_{timestamp}.html"
            self._generate_html_table(table_data, table_file, metrics)
        elif output_format == "csv":
            table_file = self.tables_path / f"comparison_table_{timestamp}.csv"
            self._generate_csv_table(table_data, table_file, metrics)
        else:
            raise ValueError(f"Unsupported table format: {output_format}")

        return table_file

    def _generate_latex_table(
        self, data: list[dict], file_path: Path, metrics: list[str]
    ) -> None:
        """Generate LaTeX table."""
        with open(file_path, "w") as f:
            # Table header
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Performance Comparison}\n")
            f.write("\\label{tab:performance_comparison}\n")

            # Column specification
            n_cols = 2 + len(metrics)  # Operator, Dataset, + metrics
            f.write(f"\\begin{{tabular}}{{{'|c' * n_cols}|}}\n")
            f.write("\\hline\n")

            # Header row
            headers = ["Operator", "Dataset", *metrics]
            f.write(" & ".join(headers) + " \\\\\n")
            f.write("\\hline\n")

            # Data rows
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
        self, data: list[dict], file_path: Path, metrics: list[str]
    ) -> None:
        """Generate HTML table."""
        with open(file_path, "w") as f:
            f.write("<html><body>\n")
            f.write("<table border='1'>\n")
            f.write("<caption>Performance Comparison</caption>\n")

            # Header
            f.write("<tr>")
            headers = ["Operator", "Dataset", *metrics]
            for header in headers:
                f.write(f"<th>{header}</th>")
            f.write("</tr>\n")

            # Data rows
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
        self, data: list[dict], file_path: Path, metrics: list[str]
    ) -> None:
        """Generate CSV table."""
        import csv

        with open(file_path, "w", newline="") as f:
            if data:
                headers = ["Operator", "Dataset", *metrics]
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

                for row in data:
                    csv_row = {
                        "Operator": row["Operator"],
                        "Dataset": row["Dataset"],
                    }
                    for metric in metrics:
                        csv_row[metric] = row.get(metric, "N/A")
                    writer.writerow(csv_row)

    def create_benchmark_database_entry(
        self, results: BenchmarkResult
    ) -> dict[str, Any]:
        """Create standardized database entry for benchmark results.

        Args:
            results: Benchmark results

        Returns:
            Standardized database entry
        """
        return {
            "id": f"{results.model_name}_{results.dataset_name}_{results.timestamp}",
            "model_name": results.model_name,
            "dataset_name": results.dataset_name,
            "timestamp": results.timestamp,
            "metrics": results.metrics,
            "execution_time": results.execution_time,
            "memory_usage": results.memory_usage,
            "gpu_memory_usage": results.gpu_memory_usage,
            "framework_version": results.framework_version,
            "system_info": results.system_info,
            "hyperparameters": results.hyperparameters,
        }

    def query_results(
        self,
        model_name: str | None = None,
        dataset_name: str | None = None,
        metric_filter: dict[str, tuple[float, float]] | None = None,
    ) -> list[dict[str, Any]]:
        """Query benchmark database with filters.

        Args:
            model_name: Filter by model name
            dataset_name: Filter by dataset name
            metric_filter: Filter by metric ranges (metric_name: (min_val, max_val))

        Returns:
            List of matching database entries
        """
        results = self.database["results"]

        if model_name:
            results = [r for r in results if r["model_name"] == model_name]

        if dataset_name:
            results = [r for r in results if r["dataset_name"] == dataset_name]

        if metric_filter:
            filtered_results = []
            for result in results:
                matches_filter = True
                for metric, (min_val, max_val) in metric_filter.items():
                    metric_val = result["metrics_summary"].get(metric)
                    if metric_val is None or not (min_val <= metric_val <= max_val):
                        matches_filter = False
                        break
                if matches_filter:
                    filtered_results.append(result)
            results = filtered_results

        return results

    def export_database(self, export_path: str, output_format: str = "json") -> None:
        """Export entire benchmark database.

        Args:
            export_path: Path to export file
            output_format: Export format (json, pickle)
        """
        export_file = Path(export_path)

        if output_format == "json":
            with open(export_file, "w") as f:
                json.dump(self.database, f, indent=2)
        elif output_format == "pickle":
            with open(export_file, "wb") as f:
                pickle.dump(self.database, f)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")

    def get_database_statistics(self) -> dict[str, Any]:
        """Get statistics about the benchmark database.

        Returns:
            Database statistics summary
        """
        results = self.database["results"]

        if not results:
            return {
                "total_results": 0,
                "unique_models": 0,
                "unique_datasets": 0,
                "model_counts": {},
                "dataset_counts": {},
            }

        # Count by model
        model_counts = {}
        dataset_counts = {}

        for result in results:
            model = result["model_name"]
            dataset = result["dataset_name"]

            model_counts[model] = model_counts.get(model, 0) + 1
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

        # Execution time statistics
        exec_times = [
            r["execution_time"] for r in results if r["execution_time"] is not None
        ]

        stats = {
            "total_results": len(results),
            "unique_models": len(model_counts),
            "unique_datasets": len(dataset_counts),
            "model_counts": model_counts,
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
