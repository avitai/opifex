"""Core benchmarking evaluation engine for Opifex framework.

This module provides model evaluation capabilities using calibrax for metrics
and statistical analysis. BenchmarkEvaluator orchestrates evaluation runs,
profiling, and result management.
"""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
from calibrax.core import BenchmarkResult
from calibrax.core.models import Metric
from calibrax.metrics import calculate_all as calculate_all_metrics
from calibrax.profiling import TimingCollector
from calibrax.statistics import StatisticalAnalyzer


logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Main benchmark evaluator for Opifex models.

    Provides comprehensive evaluation capabilities including model assessment,
    performance profiling, batch evaluation, and result management.
    """

    def __init__(
        self,
        output_dir: str = "./benchmark_results",
        save_detailed_results: bool = True,
        enable_gpu_profiling: bool = False,
    ) -> None:
        """Initialize benchmark evaluator.

        Args:
            output_dir: Directory for saving results.
            save_detailed_results: Whether to save detailed results to files.
            enable_gpu_profiling: Whether to enable GPU profiling.
        """
        self.output_dir = Path(output_dir)
        self.save_detailed_results = save_detailed_results
        self.enable_gpu_profiling = enable_gpu_profiling

        # Create output directory with proper structure
        if self.save_detailed_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.raw_results_dir = self.output_dir / "raw_evaluations"
            self.raw_results_dir.mkdir(exist_ok=True)

        # Initialize calibrax statistical analyzer
        self.statistical_analyzer = StatisticalAnalyzer()

        # Store results
        self.results: list[BenchmarkResult] = []

    def evaluate_model(
        self,
        model: Any,
        model_name: str,
        input_data: jax.Array | tuple[jax.Array, ...],
        target_data: jax.Array,
        dataset_name: str,
        forward_fn: Callable | None = None,
        custom_metrics: dict[str, Callable] | None = None,
    ) -> BenchmarkResult:
        """Evaluate a model on given data with extensive metrics.

        Args:
            model: Model to evaluate.
            model_name: Name identifier for the model.
            input_data: Input data for evaluation.
            target_data: Expected target outputs.
            dataset_name: Name of the dataset being used.
            forward_fn: Optional custom forward function.
            custom_metrics: Optional dictionary of custom metric functions.

        Returns:
            BenchmarkResult with evaluation metrics and metadata.
        """
        if forward_fn is None:
            forward_fn = _default_forward

        jit_forward = jax.jit(forward_fn, static_argnums=(0,))

        # Warm-up run
        try:
            result_warmup = jit_forward(model, input_data)
            if hasattr(result_warmup, "block_until_ready"):
                result_warmup.block_until_ready()
        except Exception as e:
            raise RuntimeError(f"Model forward pass failed during warm-up: {e}") from e

        # Timed evaluation
        start_time = time.perf_counter()
        try:
            predictions = jit_forward(model, input_data)
            if hasattr(predictions, "block_until_ready"):
                predictions.block_until_ready()
        except Exception as e:
            raise RuntimeError(
                f"Model forward pass failed during evaluation: {e}"
            ) from e
        execution_time = time.perf_counter() - start_time

        # Calculate metrics via calibrax
        raw_metrics = calculate_all_metrics(predictions, target_data)

        # Add custom metrics if provided
        if custom_metrics:
            for metric_name, metric_fn in custom_metrics.items():
                try:
                    raw_metrics[metric_name] = float(
                        metric_fn(predictions, target_data)
                    )
                except (ValueError, TypeError, ArithmeticError) as e:
                    logger.warning("Custom metric '%s' failed: %s", metric_name, e)

        # Wrap raw floats into calibrax Metric objects
        metrics = {k: Metric(value=v) for k, v in raw_metrics.items()}

        result = BenchmarkResult(
            name=model_name,
            domain="scientific_ml",
            tags={"dataset": dataset_name},
            metrics=metrics,
            metadata={
                "execution_time": execution_time,
                "framework_version": "1.0.0",
            },
        )

        self.results.append(result)

        if self.save_detailed_results:
            self._save_result(result)

        return result

    def batch_evaluate(
        self,
        models: list[tuple[str, Any]],
        datasets: list[tuple[str, Any, jax.Array, Callable | None]],
    ) -> list[BenchmarkResult]:
        """Evaluate multiple models on multiple datasets.

        Args:
            models: List of (model_name, model) tuples.
            datasets: List of (dataset_name, input_data, target_data, forward_fn)
                tuples.

        Returns:
            List of BenchmarkResults for all model-dataset combinations.
        """
        all_results = []

        for model_name, model in models:
            for dataset_name, input_data, target_data, forward_fn in datasets:
                try:
                    result = self.evaluate_model(
                        model=model,
                        model_name=model_name,
                        input_data=input_data,
                        target_data=target_data,
                        dataset_name=dataset_name,
                        forward_fn=forward_fn,
                    )
                    all_results.append(result)
                except RuntimeError:
                    logger.warning(
                        "Evaluation failed for model '%s' on dataset '%s'",
                        model_name,
                        dataset_name,
                        exc_info=True,
                    )

        return all_results

    def profile_model_performance(
        self,
        model: Any,
        input_data: jax.Array | tuple[jax.Array, ...],
        num_runs: int = 10,
        forward_fn: Callable | None = None,
    ) -> dict[str, float]:
        """Profile model performance with multiple runs.

        Args:
            model: Model to profile.
            input_data: Input data for profiling.
            num_runs: Number of runs for statistics.
            forward_fn: Custom forward function.

        Returns:
            Dictionary with performance statistics.
        """
        if forward_fn is None:
            forward_fn = _default_forward

        jit_forward = jax.jit(forward_fn, static_argnums=(0,))

        # Use TimingCollector for consistent, GPU-synced timing with warmup
        def _sync() -> None:
            jax.numpy.array(0.0).block_until_ready()

        collector = TimingCollector(sync_fn=_sync, warmup_iterations=1)

        def _forward_iter():
            for _ in range(num_runs + 1):
                result_iter = jit_forward(model, input_data)
                if hasattr(result_iter, "block_until_ready"):
                    result_iter.block_until_ready()
                yield result_iter

        sample = collector.measure_iteration(_forward_iter(), num_batches=num_runs + 1)
        execution_times = list(sample.per_batch_times)

        stats = self.statistical_analyzer.summarize(execution_times)

        return {
            "mean_execution_time": stats.mean,
            "std_execution_time": stats.std,
            "min_execution_time": stats.min,
            "max_execution_time": stats.max,
            "memory_usage": 0,
        }

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to file.

        Args:
            result: BenchmarkResult to save.
        """
        if not self.save_detailed_results:
            return

        dataset = result.tags.get("dataset", "unknown")
        timestamp = str(result.timestamp).replace(".", "-")
        filename = f"{result.name}_{dataset}_{timestamp}.json"
        filepath = self.raw_results_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def load_results(self) -> list[BenchmarkResult]:
        """Load all benchmark results from files.

        Returns:
            List of BenchmarkResults.
        """
        if not self.save_detailed_results or not self.output_dir.exists():
            return []

        results = []
        search_dir = (
            self.raw_results_dir
            if hasattr(self, "raw_results_dir") and self.raw_results_dir.exists()
            else self.output_dir
        )
        for filepath in search_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    result_dict = json.load(f)
                result = BenchmarkResult.from_dict(result_dict)
                results.append(result)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load result from %s: %s", filepath, e)

        return results

    def generate_summary_report(self) -> dict[str, Any]:
        """Generate complete summary report of all evaluations.

        Returns:
            Dictionary with summary statistics and analysis.
        """
        if not self.results:
            return {"error": "No benchmark results available"}

        # Group results by model
        model_performance: dict[str, list[float]] = {}
        dataset_difficulty: dict[str, list[float]] = {}

        for result in self.results:
            mse_val = result.metrics.get("mse")
            if mse_val is None:
                continue

            mse_value = mse_val.value

            if result.name not in model_performance:
                model_performance[result.name] = []
            model_performance[result.name].append(mse_value)

            dataset = result.tags.get("dataset", "unknown")
            if dataset not in dataset_difficulty:
                dataset_difficulty[dataset] = []
            dataset_difficulty[dataset].append(mse_value)

        summary: dict[str, Any] = {
            "total_evaluations": len(self.results),
            "unique_models": len(model_performance),
            "unique_datasets": len(dataset_difficulty),
            "model_rankings": {},
            "dataset_difficulty_rankings": {},
        }

        for model_name, mse_values in model_performance.items():
            stats = self.statistical_analyzer.summarize(mse_values)
            summary["model_rankings"][model_name] = {
                "average_mse": stats.mean,
                "confidence_interval": [stats.ci_lower, stats.ci_upper],
                "num_evaluations": len(mse_values),
            }

        for dataset_name, mse_values in dataset_difficulty.items():
            stats = self.statistical_analyzer.summarize(mse_values)
            summary["dataset_difficulty_rankings"][dataset_name] = {
                "average_mse": stats.mean,
                "num_evaluations": len(mse_values),
            }

        return summary


def _default_forward(model: Any, inputs: Any) -> Any:
    """Default forward function for model evaluation.

    Args:
        model: Model to evaluate.
        inputs: Input data.

    Returns:
        Model predictions.
    """
    if callable(model):
        return model(inputs)
    if hasattr(model, "forward"):
        return model.forward(inputs)
    raise ValueError(f"Model {type(model)} has no callable interface")


__all__ = [
    "BenchmarkEvaluator",
    "BenchmarkResult",
]
