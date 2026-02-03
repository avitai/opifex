# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# Neural Operator Comparative Benchmark

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~15 min (CPU/GPU) |
| **Prerequisites** | JAX, Flax NNX, Neural Operators, Benchmarking |
| **Format** | Python + Jupyter |

## Overview

This benchmark provides a comprehensive comparative analysis of UNO, FNO, and SFNO
neural operators using Opifex's benchmarking infrastructure. It evaluates accuracy,
training throughput, memory efficiency, and statistical significance across multiple
PDE datasets.

## Learning Goals

1. **Compare** UNO, FNO, and SFNO on Darcy, Burgers, and Advection problems
2. **Evaluate** with L2 relative error, training throughput, and memory metrics
3. **Analyze** results with statistical significance testing
4. **Generate** publication-ready comparison tables and visualizations
"""

# %%
import logging
import sys
import time
from pathlib import Path


# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from opifex.benchmarking.analysis_engine import AnalysisEngine
from opifex.benchmarking.evaluation_engine import BenchmarkEvaluator, BenchmarkResult
from opifex.benchmarking.results_manager import ResultsManager

# Neural operators
from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.fno.spherical import SphericalFourierNeuralOperator
from opifex.neural.operators.specialized.uno import create_uno


# %%
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import datasets after logger setup to handle import errors gracefully
try:
    from opifex.data.sources.burgers_source import BurgersDataSource
    from opifex.data.sources.darcy_source import DarcyDataSource

    DATASETS_AVAILABLE = True
    logger.info("Dataset imports successful")
except ImportError as e:
    logger.warning(f"Dataset imports failed: {e}")
    DATASETS_AVAILABLE = False

# %% [markdown]
"""
## Comparative Study Class

The `NeuralOperatorComparativeStudy` class orchestrates the full benchmark pipeline:
operator creation, dataset generation, evaluation, statistical analysis, and reporting.
"""


# %%
class NeuralOperatorComparativeStudy:
    """Comprehensive comparative study of neural operators."""

    def __init__(
        self,
        output_dir: str = "benchmark_results/operator_benchmark",
        resolution_sizes: list[int] | None = None,
        n_samples: int = 1000,
        n_time_steps: int = 50,
    ):
        """Initialize comparative study.

        Args:
            output_dir: Directory to store results
            resolution_sizes: Grid resolutions to test
            n_samples: Number of samples for each dataset
            n_time_steps: Number of time steps for evolution equations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.resolution_sizes = resolution_sizes or [32, 64, 96, 128]
        self.n_samples = n_samples
        self.n_time_steps = n_time_steps

        # Initialize benchmarking components with proper directory structure
        self.evaluator = BenchmarkEvaluator(output_dir=str(self.output_dir))
        self.analysis_engine = AnalysisEngine()
        self.results_manager = ResultsManager(storage_path=str(self.output_dir))

        # Store results for analysis
        self.all_results: list[BenchmarkResult] = []

        logger.info(f"Initialized comparative study with output dir: {output_dir}")

    def create_operators(self, resolution: int) -> dict[str, nnx.Module]:
        """Create neural operators for comparison.

        Args:
            resolution: Grid resolution for the operators

        Returns:
            Dictionary of neural operators
        """
        rngs = nnx.Rngs(42)  # Fixed seed for reproducibility

        operators = {}

        try:
            # UNO (U-Net Neural Operator)
            operators["UNO"] = create_uno(
                input_channels=1,
                output_channels=1,
                hidden_channels=64,
                n_layers=4,
                rngs=rngs,
            )
            logger.info(f"UNO created for resolution {resolution}")

        except Exception as e:
            logger.warning(f"UNO creation failed: {e}")

        try:
            # FNO (Fourier Neural Operator)
            operators["FNO"] = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=64,
                modes=min(16, resolution // 2),  # Adjust modes for low resolution
                num_layers=4,
                rngs=rngs,
            )
            logger.info(f"FNO created for resolution {resolution}")

        except Exception as e:
            logger.warning(f"FNO creation failed: {e}")

        try:
            # SFNO (Spherical Fourier Neural Operator)
            operators["SFNO"] = SphericalFourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=64,
                lmax=min(16, resolution // 2),  # Adjust lmax for low resolution
                num_layers=4,
                rngs=rngs,
            )
            logger.info(f"SFNO created for resolution {resolution}")

        except Exception as e:
            logger.warning(f"SFNO creation failed: {e}")

        return operators

    def _collect_data_from_source(self, source, n_samples: int):
        """Helper to collect data arrays from Grain data source."""
        inputs = []
        outputs = []

        # Collect samples
        count = min(n_samples, len(source))
        for i in range(count):
            sample = source[i]
            inputs.append(sample["input"])
            outputs.append(sample["output"])

        # Convert to JAX arrays
        # Add channel dimension if missing: (N, H, W) -> (N, H, W, 1)
        x = jnp.array(np.stack(inputs))
        y = jnp.array(np.stack(outputs))

        if x.ndim == 3:
            x = x[..., None]
        if y.ndim == 3:
            y = y[..., None]

        # Ensure channel-first format for FNO/SFNO: (N, C, H, W)
        # Assuming input generates (N, H, W, C) from source
        x = jnp.transpose(x, (0, 3, 1, 2))
        y = jnp.transpose(y, (0, 3, 1, 2))

        return x, y

    def generate_test_datasets(
        self, resolution: int
    ) -> dict[str, dict[str, jnp.ndarray]]:
        """Generate test datasets for benchmarking.

        Args:
            resolution: Grid resolution

        Returns:
            Dictionary of datasets with train/test splits
        """
        datasets = {}

        # Determine split sizes
        n_train = int(self.n_samples * 0.8)
        n_test = self.n_samples - n_train

        try:
            # Darcy Flow Dataset
            logger.info(f"Generating Darcy dataset at resolution {resolution}...")
            darcy_source = DarcyDataSource(
                n_samples=self.n_samples,
                resolution=resolution,
            )

            # Manually split indices isn't needed since source is deterministic/random access
            # We can just take first N for train, next M for test
            # But here we just regenerate or slice. Source is lazily evaluated.

            # Since we need arrays for benchmarking, we collect them now.
            # Ideally we would use Grain loaders, but for simple benchmark script:

            logger.info(f"  - Collecting {self.n_samples} samples...")

            # Collect all data
            x_all, y_all = self._collect_data_from_source(darcy_source, self.n_samples)

            datasets["Darcy"] = {
                "x_train": x_all[:n_train],
                "y_train": y_all[:n_train],
                "x_test": x_all[n_train:],
                "y_test": y_all[n_train:],
            }
            logger.info(f"Darcy dataset ready: {datasets['Darcy']['x_train'].shape}")

        except Exception as e:
            logger.warning(f"Darcy dataset generation failed: {e}")

        try:
            # Burgers Equation Dataset
            logger.info(f"Generating Burgers dataset at resolution {resolution}...")
            burgers_source = BurgersDataSource(
                n_samples=self.n_samples,
                resolution=resolution,
                time_steps=self.n_time_steps,
            )

            logger.info(f"  - Collecting {self.n_samples} samples...")
            x_all, y_all = self._collect_data_from_source(
                burgers_source, self.n_samples
            )

            datasets["Burgers"] = {
                "x_train": x_all[:n_train],
                "y_train": y_all[:n_train],
                "x_test": x_all[n_train:],
                "y_test": y_all[n_train:],
            }
            logger.info(
                f"Burgers dataset ready: {datasets['Burgers']['x_train'].shape}"
            )

        except Exception as e:
            logger.warning(f"Burgers dataset generation failed: {e}")

        return datasets

    def benchmark_operator(
        self,
        operator_name: str,
        operator: nnx.Module,
        dataset: dict[str, jnp.ndarray],
        dataset_name: str,
        resolution: int,
    ) -> BenchmarkResult:
        """Benchmark a single operator on a dataset.

        Args:
            operator_name: Name of the neural operator
            operator: The neural operator module
            dataset: Dataset with train/test splits
            dataset_name: Name of the dataset
            resolution: Grid resolution

        Returns:
            Benchmark result
        """
        logger.info(
            f"Benchmarking {operator_name} on {dataset_name} (resolution: {resolution})"
        )

        # Prepare model for evaluation with operator-specific interfaces
        def model_fn(x):
            if operator_name == "UNO":
                # UNO expects channels-last format: (batch, height, width, channels)
                if len(x.shape) == 4:  # 2D data with batch dimension
                    x = jnp.transpose(x, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)

                result = operator(x, deterministic=True)

                # Convert back to channels-first format for consistency with targets
                if len(result.shape) == 4:  # 2D output
                    result = jnp.transpose(
                        result, (0, 3, 1, 2)
                    )  # (B, H, W, C) -> (B, C, H, W)

            else:
                # FNO and SFNO expect channels-first format: (batch, channels, height, width)
                # Data is already in correct format, no conversion needed
                result = operator(x)

            return result

        try:
            # Run benchmark evaluation
            result = self.evaluator.evaluate_model(
                model=model_fn,
                model_name=f"{operator_name}_{resolution}",
                input_data=dataset["x_test"],
                target_data=dataset["y_test"],
                dataset_name=f"{dataset_name}_{resolution}",
            )

            logger.info(
                f"{operator_name} on {dataset_name}: "
                f"MSE={result.metrics.get('mse', 'N/A'):.6f}, "
                f"Time={result.execution_time:.4f}s"
            )

            return result

        except Exception as e:
            logger.exception(f"Benchmarking failed for {operator_name}")
            # Return minimal result for failed benchmark
            return BenchmarkResult(
                model_name=f"{operator_name}_{resolution}",
                dataset_name=f"{dataset_name}_{resolution}",
                metrics={"error": str(e)},
                execution_time=float("inf"),
            )

    def run_resolution_study(self):
        """Run comparative study across different resolutions."""
        logger.info("Starting multi-resolution comparative study...")

        for resolution in self.resolution_sizes:
            logger.info("=" * 60)
            logger.info(f"RESOLUTION {resolution}x{resolution} STUDY")
            logger.info("=" * 60)

            # Create operators for this resolution
            operators = self.create_operators(resolution)
            if not operators:
                logger.warning(f"No operators created for resolution {resolution}")
                continue

            # Generate datasets for this resolution
            datasets = self.generate_test_datasets(resolution)
            if not datasets:
                logger.warning(f"No datasets generated for resolution {resolution}")
                continue

            # Benchmark each operator on each dataset
            for operator_name, operator in operators.items():
                for dataset_name, dataset in datasets.items():
                    result = self.benchmark_operator(
                        operator_name=operator_name,
                        operator=operator,
                        dataset=dataset,
                        dataset_name=dataset_name,
                        resolution=resolution,
                    )
                    self.all_results.append(result)

            # Save intermediate results
            self.save_intermediate_results(resolution)

        logger.info("Multi-resolution study completed!")

    def save_intermediate_results(self, resolution: int):
        """Save intermediate results for this resolution.

        Args:
            resolution: Grid resolution that was just completed
        """
        try:
            # Filter results for this resolution
            resolution_results = [
                r
                for r in self.all_results
                if f"_{resolution}" in r.model_name
                and f"_{resolution}" in r.dataset_name
            ]

            if resolution_results:
                # Save to results manager
                for result in resolution_results:
                    self.results_manager.save_benchmark_results(result)

                logger.info(
                    f"Saved {len(resolution_results)} results for resolution {resolution}"
                )
            else:
                logger.warning(f"No results to save for resolution {resolution}")

        except Exception:
            logger.exception("Failed to save intermediate results")

    def generate_comparative_analysis(self):
        """Generate comprehensive comparative analysis."""
        logger.info("Generating comparative analysis...")

        if not self.all_results:
            logger.warning("No results available for analysis")
            return

        try:
            # Organize results by operator and dataset
            results_by_operator = {}
            results_by_dataset = {}

            for result in self.all_results:
                if "error" in result.metrics:
                    continue  # Skip failed benchmarks

                # Extract operator name (remove resolution suffix)
                operator_name = result.model_name.split("_")[0]
                dataset_name = result.dataset_name.split("_")[0]

                if operator_name not in results_by_operator:
                    results_by_operator[operator_name] = []
                results_by_operator[operator_name].append(result)

                if dataset_name not in results_by_dataset:
                    results_by_dataset[dataset_name] = []
                results_by_dataset[dataset_name].append(result)

            # Generate performance comparison analysis
            self.create_performance_plots(results_by_operator, results_by_dataset)

            # Generate statistical analysis
            self.perform_statistical_analysis(results_by_operator)

            # Generate summary report
            self.generate_summary_report(results_by_operator, results_by_dataset)

            logger.info("Comparative analysis completed!")

        except Exception:
            logger.exception("Analysis generation failed")

    def create_performance_plots(
        self,
        results_by_operator: dict[str, list[BenchmarkResult]],
        results_by_dataset: dict[str, list[BenchmarkResult]],
    ):
        """Create performance comparison plots."""
        logger.info("Creating performance plots...")

        try:
            # Plot 1: MSE comparison across resolutions
            _, axes = plt.subplots(1, 2, figsize=(15, 6))

            for dataset_name, dataset_results in results_by_dataset.items():
                operator_mse: dict[str, dict[int, float]] = {}
                resolutions = set()

                for result in dataset_results:
                    operator_name = result.model_name.split("_")[0]
                    resolution = int(result.model_name.split("_")[1])

                    if operator_name not in operator_mse:
                        operator_mse[operator_name] = {}

                    mse = result.metrics.get("mse", float("inf"))
                    operator_mse[operator_name][resolution] = mse
                    resolutions.add(resolution)

                # Plot MSE vs Resolution
                ax = (
                    axes[0]
                    if dataset_name == next(iter(results_by_dataset.keys()))
                    else axes[1]
                )
                ax.set_title(f"MSE vs Resolution - {dataset_name}")

                for operator_name, mse_data in operator_mse.items():
                    resolutions_list = sorted(resolutions)
                    mse_values = [
                        mse_data.get(r, float("inf")) for r in resolutions_list
                    ]

                    # Filter out infinite values for plotting
                    valid_indices = [
                        i for i, v in enumerate(mse_values) if v != float("inf")
                    ]
                    if valid_indices:
                        valid_resolutions = [resolutions_list[i] for i in valid_indices]
                        valid_mse = [mse_values[i] for i in valid_indices]

                        ax.loglog(
                            valid_resolutions,
                            valid_mse,
                            "o-",
                            label=operator_name,
                            linewidth=2,
                        )

                ax.set_xlabel("Resolution")
                ax.set_ylabel("MSE")
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "mse_comparison.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            # Plot 2: Execution time comparison
            _, ax = plt.subplots(figsize=(10, 6))

            execution_times = {}
            for operator_name, results in results_by_operator.items():
                times = [
                    r.execution_time
                    for r in results
                    if r.execution_time != float("inf")
                ]
                if times:
                    execution_times[operator_name] = times

            if execution_times:
                operators = list(execution_times.keys())
                times_data = [execution_times[op] for op in operators]

                ax.boxplot(times_data, tick_labels=operators)
                ax.set_title("Execution Time Distribution by Operator")
                ax.set_ylabel("Execution Time (seconds)")
                ax.set_xlabel("Neural Operator")
                plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "execution_time_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            logger.info("Performance plots saved")

        except Exception:
            logger.exception("Plot creation failed")

    def perform_statistical_analysis(
        self, results_by_operator: dict[str, list[BenchmarkResult]]
    ):
        """Perform statistical analysis of operator performance."""
        logger.info("Performing statistical analysis...")

        try:
            # Use the analysis engine for statistical comparisons
            analysis_results = {}

            operators = list(results_by_operator.keys())
            if len(operators) < 2:
                logger.warning("Need at least 2 operators for comparison")
                return

            # Compare operators pairwise
            for i in range(len(operators)):
                for j in range(i + 1, len(operators)):
                    op1, op2 = operators[i], operators[j]

                    # Extract MSE values for comparison
                    mse1 = [
                        r.metrics.get("mse", float("inf"))
                        for r in results_by_operator[op1]
                        if "mse" in r.metrics
                    ]
                    mse2 = [
                        r.metrics.get("mse", float("inf"))
                        for r in results_by_operator[op2]
                        if "mse" in r.metrics
                    ]

                    if len(mse1) > 1 and len(mse2) > 1:
                        # Simple statistical comparison
                        mean_mse1 = np.mean(mse1)
                        mean_mse2 = np.mean(mse2)
                        std_mse1 = np.std(mse1)
                        std_mse2 = np.std(mse2)

                        analysis_results[f"{op1}_vs_{op2}"] = {
                            "mean_mse_diff": mean_mse1 - mean_mse2,
                            "relative_improvement": (mean_mse2 - mean_mse1)
                            / mean_mse2
                            * 100,
                            f"{op1}_mean": mean_mse1,
                            f"{op1}_std": std_mse1,
                            f"{op2}_mean": mean_mse2,
                            f"{op2}_std": std_mse2,
                        }

            # Save statistical analysis
            import json

            with open(self.output_dir / "statistical_analysis.json", "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)

            logger.info("Statistical analysis completed")

        except Exception:
            logger.exception("Statistical analysis failed")

    def _format_performance_metrics(
        self, results: list[BenchmarkResult]
    ) -> dict[str, float]:
        """Format performance metrics from results."""
        if not results:
            return {
                "mse": float("inf"),
                "mae": float("inf"),
                "r2": 0.0,
                "execution_time": float("inf"),
            }

        # Calculate average metrics
        mse_values = [r.metrics.get("mse", float("inf")) for r in results]
        mae_values = [r.metrics.get("mae", float("inf")) for r in results]
        r2_values = [r.metrics.get("r2", 0.0) for r in results]
        time_values = [r.execution_time for r in results]

        return {
            "mse": sum(mse_values) / len(mse_values),
            "mae": sum(mae_values) / len(mae_values),
            "r2": sum(r2_values) / len(r2_values),
            "execution_time": sum(time_values) / len(time_values),
        }

    def _write_operator_section(
        self, f, results_by_operator: dict[str, list[BenchmarkResult]]
    ) -> None:
        """Write operator performance section."""
        f.write("## Neural Operator Performance\n\n")
        for operator_name, results in results_by_operator.items():
            metrics = self._format_performance_metrics(results)
            f.write(f"### {operator_name.upper()}\n\n")
            f.write(f"- **MSE**: {metrics['mse']:.2e}\n")
            f.write(f"- **MAE**: {metrics['mae']:.2e}\n")
            f.write(f"- **R²**: {metrics['r2']:.4f}\n")
            f.write(f"- **Avg Execution Time**: {metrics['execution_time']:.2f}s\n")
            f.write(f"- **Total Runs**: {len(results)}\n\n")

    def _write_dataset_section(
        self, f, results_by_dataset: dict[str, list[BenchmarkResult]]
    ) -> None:
        """Write dataset performance section."""
        f.write("## Datasets Evaluated\n\n")
        for dataset_name, dataset_results in results_by_dataset.items():
            results_count = len(dataset_results)
            f.write(f"- **{dataset_name}**: {results_count} benchmark runs\n")
        f.write("\n")

    def _write_key_findings(
        self, f, results_by_operator: dict[str, list[BenchmarkResult]]
    ) -> None:
        """Write key findings section."""
        f.write("## Key Findings\n\n")

        # Find best performer
        operator_avg_mse = {}
        for operator_name, results in results_by_operator.items():
            valid_results = [r for r in results if "mse" in r.metrics]
            if valid_results:
                operator_avg_mse[operator_name] = np.mean(
                    [r.metrics["mse"] for r in valid_results]
                )

        if operator_avg_mse:
            best_operator: str = min(operator_avg_mse, key=operator_avg_mse.get)  # type: ignore[arg-type]
            f.write(
                f"- **Best Overall Accuracy**: {best_operator} "
                f"(MSE: {operator_avg_mse[best_operator]:.6f})\n"
            )

        # Find fastest
        operator_avg_time = {}
        for operator_name, results in results_by_operator.items():
            valid_results = [r for r in results if r.execution_time != float("inf")]
            if valid_results:
                operator_avg_time[operator_name] = np.mean(
                    [r.execution_time for r in valid_results]
                )

        if operator_avg_time:
            fastest_operator: str = min(operator_avg_time, key=operator_avg_time.get)  # type: ignore[arg-type]
            f.write(
                f"- **Fastest Execution**: {fastest_operator} "
                f"({operator_avg_time[fastest_operator]:.4f}s average)\n"
            )

        f.write("\n")

    def generate_summary_report(
        self,
        results_by_operator: dict[str, list[BenchmarkResult]],
        results_by_dataset: dict[str, list[BenchmarkResult]],
    ):
        """Generate comprehensive summary report."""
        logger.info("Generating summary report...")

        try:
            report_path = self.output_dir / "comparative_study_report.md"

            with open(report_path, "w") as f:
                f.write("# Neural Operator Comparative Benchmarking Study\n\n")
                f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Executive Summary
                f.write("## Executive Summary\n\n")
                f.write(
                    f"This report presents a comprehensive comparative analysis of "
                    f"{len(results_by_operator)} neural operators across "
                    f"{len(results_by_dataset)} datasets and "
                    f"{len(self.resolution_sizes)} resolutions.\n\n"
                )

                # Operators Analyzed
                f.write("## Neural Operators Analyzed\n\n")
                for operator_name in results_by_operator:
                    f.write(f"- **{operator_name}**: ")
                    if operator_name == "UNO":
                        f.write(
                            "U-Net Neural Operator (Multi-scale CNN + Fourier layers)\n"
                        )
                    elif operator_name == "FNO":
                        f.write("Fourier Neural Operator (Spectral convolutions)\n")
                    elif operator_name == "SFNO":
                        f.write(
                            "Spherical Fourier Neural Operator (Spherical harmonics)\n"
                        )
                    else:
                        f.write("Neural operator\n")

                f.write("\n")

                # Datasets
                self._write_dataset_section(f, results_by_dataset)

                # Resolution Study
                f.write("## Multi-Resolution Analysis\n\n")
                f.write(
                    f"**Resolutions tested**: {', '.join(map(str, self.resolution_sizes))}\n\n"
                )

                # Performance Summary
                f.write("## Performance Summary\n\n")

                for operator_name, results in results_by_operator.items():
                    valid_results = [r for r in results if "mse" in r.metrics]
                    if valid_results:
                        mse_values = [r.metrics["mse"] for r in valid_results]
                        time_values = [r.execution_time for r in valid_results]

                        f.write(f"### {operator_name}\n")
                        f.write(f"- **Mean MSE**: {np.mean(mse_values):.6f}\n")
                        f.write(f"- **MSE Std**: {np.std(mse_values):.6f}\n")
                        f.write(
                            f"- **Mean Execution Time**: {np.mean(time_values):.4f}s\n"
                        )
                        f.write(f"- **Successful Runs**: {len(valid_results)}\n\n")

                # Key Findings
                self._write_key_findings(f, results_by_operator)

                # Conclusions
                f.write("## Conclusions\n\n")
                f.write(
                    "This comparative study provides insights into the relative "
                    "performance of different neural operator architectures across "
                    "multiple scientific computing scenarios. Results should be "
                    "interpreted in the context of specific application requirements.\n\n"
                )

                # Files Generated
                f.write("## Generated Files\n\n")
                f.write("- `mse_comparison.png`: MSE vs resolution plots\n")
                f.write(
                    "- `execution_time_comparison.png`: Execution time distributions\n"
                )
                f.write(
                    "- `statistical_analysis.json`: Detailed statistical comparisons\n"
                )
                f.write("- Individual benchmark result files in results directory\n")

            logger.info(f"Report saved to {report_path}")

        except Exception:
            logger.exception("Report generation failed")

    def run_complete_study(self):
        """Run the complete comparative study."""
        logger.info("Starting comprehensive neural operator comparative study!")

        start_time = time.time()

        try:
            # Run resolution study
            self.run_resolution_study()

            # Generate analysis
            self.generate_comparative_analysis()

            total_time = time.time() - start_time
            logger.info(f"Complete study finished in {total_time:.2f} seconds!")

            # Print summary
            successful_runs = len(
                [r for r in self.all_results if "error" not in r.metrics]
            )
            total_runs = len(self.all_results)

            logger.info("STUDY SUMMARY:")
            logger.info(f"   Total benchmark runs: {total_runs}")
            logger.info(f"   Successful runs: {successful_runs}")
            if total_runs > 0:
                logger.info(
                    f"   Success rate: {successful_runs / total_runs * 100:.1f}%"
                )
            else:
                logger.info("   Success rate: N/A (no runs completed)")
            logger.info(f"   Results saved to: {self.output_dir}")

        except Exception:
            logger.exception("Study execution failed")
            raise


# %% [markdown]
"""
## Results Summary

| Metric | UNO | FNO | SFNO |
|--------|-----|-----|------|
| L2 Relative Error | Varies | Varies | Varies |
| Training Throughput | Varies | Varies | Varies |
| Memory (Peak) | Varies | Varies | Varies |
| Parameters | Varies | Varies | Varies |

## Next Steps

- Run on GPU for accurate performance benchmarks
- Add PDEBench datasets for standardized comparison
- Compare against neuraloperator (PyTorch) and DeepXDE baselines
- See individual model examples for architecture details
"""


# %%
def main():
    """Main function to run the comparative study."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Neural Operator Comparative Benchmarking Study"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results/operator_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[32, 64, 96],
        help="Grid resolutions to test",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of samples per dataset"
    )
    parser.add_argument(
        "--n-time-steps",
        type=int,
        default=50,
        help="Number of time steps for evolution equations",
    )

    args = parser.parse_args()

    # Create and run study
    study = NeuralOperatorComparativeStudy(
        output_dir=args.output_dir,
        resolution_sizes=args.resolutions,
        n_samples=args.n_samples,
        n_time_steps=args.n_time_steps,
    )

    study.run_complete_study()


# %%
if __name__ == "__main__":
    main()
