"""Core benchmarking evaluation engine for Opifex framework.

This module provides complete benchmarking capabilities including model evaluation,
statistical analysis, performance profiling, and result management. Designed to
integrate seamlessly with all Opifex neural operators and provide publication-ready
results.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


# Model safety functionality removed


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result data structure.

    Stores all relevant information about a benchmark run including metrics,
    performance data, system information, and metadata for reproducibility.
    """

    model_name: str
    dataset_name: str
    metrics: dict[str, float]
    execution_time: float
    memory_usage: int | None = None
    gpu_memory_usage: int | None = None
    framework_version: str | None = None
    timestamp: str | None = None
    system_info: dict[str, Any] | None = None
    hyperparameters: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate benchmark result data."""
        if not self.model_name or self.model_name.strip() == "":
            raise ValueError("model_name is required")

        if self.execution_time < 0:
            raise ValueError("execution_time must be positive")

        if self.timestamp is None:
            self.timestamp = datetime.now(tz=UTC).isoformat()


class EvaluationMetrics:
    """Comprehensive evaluation metrics for model assessment.

    Provides standard regression and classification metrics with JAX-native
    implementations for optimal performance and compatibility.
    """

    def __init__(self):
        """Initialize evaluation metrics calculator."""
        self.available_metrics = {
            "mse": self.calculate_mse,
            "mae": self.calculate_mae,
            "relative_error": self.calculate_relative_error,
            "r2_score": self.calculate_r2_score,
            "rmse": self.calculate_rmse,
            "mape": self.calculate_mape,
        }

    def get_available_metrics(self) -> list[str]:
        """Get list of available metrics."""
        return list(self.available_metrics.keys())

    def calculate_mse(self, predictions: jax.Array, targets: jax.Array) -> float:
        """Calculate Mean Squared Error."""
        return float(jnp.mean((predictions - targets) ** 2))

    def calculate_mae(self, predictions: jax.Array, targets: jax.Array) -> float:
        """Calculate Mean Absolute Error."""
        return float(jnp.mean(jnp.abs(predictions - targets)))

    def calculate_relative_error(
        self, predictions: jax.Array, targets: jax.Array
    ) -> float:
        """Calculate Mean Relative Error."""
        return float(
            jnp.mean(jnp.abs(predictions - targets) / (jnp.abs(targets) + 1e-8))
        )

    def calculate_r2_score(self, predictions: jax.Array, targets: jax.Array) -> float:
        """Calculate R² (coefficient of determination)."""
        ss_res = jnp.sum((targets - predictions) ** 2)
        ss_tot = jnp.sum((targets - jnp.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        return float(jnp.clip(r2, 0.0, 1.0))

    def calculate_rmse(self, predictions: jax.Array, targets: jax.Array) -> float:
        """Calculate Root Mean Squared Error."""
        mse = self.calculate_mse(predictions, targets)
        return float(jnp.sqrt(mse))

    def calculate_mape(self, predictions: jax.Array, targets: jax.Array) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return float(
            jnp.mean(jnp.abs((targets - predictions) / (targets + 1e-8))) * 100
        )

    def calculate_all_metrics(
        self, predictions: jax.Array, targets: jax.Array
    ) -> dict[str, float]:
        """Calculate all available metrics."""
        return {
            name: metric_fn(predictions, targets)
            for name, metric_fn in self.available_metrics.items()
        }


class StatisticalAnalyzer:
    """Statistical analysis for benchmark results.

    Provides confidence intervals, significance testing, and other statistical
    analysis tools for rigorous evaluation of model performance.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 1000,
        random_seed: int = 42,
    ):
        """Initialize statistical analyzer.

        Args:
            confidence_level: Confidence level for intervals (default: 0.95)
            bootstrap_samples: Number of bootstrap samples (default: 1000)
            random_seed: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed
        self.key = jax.random.PRNGKey(random_seed)

    def calculate_confidence_interval(self, data: jax.Array) -> tuple[float, float]:
        """Calculate confidence interval using bootstrap method.

        Args:
            data: Array of data points

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data)
        alpha = 1 - self.confidence_level

        # Bootstrap sampling
        bootstrap_means: list[float] = []
        for i in range(self.bootstrap_samples):
            key = jax.random.fold_in(self.key, i)
            indices = jax.random.choice(key, n, shape=(n,), replace=True)
            bootstrap_sample = data[indices]
            bootstrap_means.append(float(jnp.mean(bootstrap_sample)))

        bootstrap_means_array = jnp.array(bootstrap_means)

        # Calculate percentiles
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = float(jnp.percentile(bootstrap_means_array, lower_percentile))
        ci_upper = float(jnp.percentile(bootstrap_means_array, upper_percentile))

        return ci_lower, ci_upper

    def test_significance(
        self,
        data1: jax.Array,
        data2: jax.Array,
        alpha: float = 0.05,
    ) -> tuple[float, bool]:
        """Test statistical significance between two datasets using permutation test.

        Args:
            data1: First dataset
            data2: Second dataset
            alpha: Significance level (default: 0.05)

        Returns:
            Tuple of (p_value, is_significant)
        """
        # Calculate observed difference in means
        observed_diff = jnp.abs(jnp.mean(data1) - jnp.mean(data2))

        # Combine datasets for permutation
        combined = jnp.concatenate([data1, data2])
        n1 = len(data1)

        # Permutation test
        greater_count = 0
        for i in range(self.bootstrap_samples):
            key = jax.random.fold_in(self.key, i + 1000)  # Different seed space
            permuted = jax.random.permutation(key, combined)

            perm_data1 = permuted[:n1]
            perm_data2 = permuted[n1:]

            perm_diff = jnp.abs(jnp.mean(perm_data1) - jnp.mean(perm_data2))

            if perm_diff >= observed_diff:
                greater_count += 1

        p_value = greater_count / self.bootstrap_samples
        is_significant = p_value < alpha

        return float(p_value), bool(is_significant)

    def calculate_effect_size(self, data1: jax.Array, data2: jax.Array) -> float:
        """Calculate Cohen's d effect size between two datasets."""
        mean1, mean2 = jnp.mean(data1), jnp.mean(data2)
        std1, std2 = jnp.std(data1), jnp.std(data2)

        # Pooled standard deviation
        n1, n2 = len(data1), len(data2)
        pooled_std = jnp.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        cohen_d = (mean1 - mean2) / (pooled_std + 1e-8)
        return float(jnp.abs(cohen_d))


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
        metrics_config: dict[str, Any] | None = None,
    ):
        """Initialize benchmark evaluator.

        Args:
            output_dir: Directory for saving results
            save_detailed_results: Whether to save detailed results to files
            enable_gpu_profiling: Whether to enable GPU profiling
            metrics_config: Configuration for metrics calculation
        """
        self.output_dir = Path(output_dir)
        self.save_detailed_results = save_detailed_results
        self.enable_gpu_profiling = enable_gpu_profiling

        # Create output directory with proper structure
        if self.save_detailed_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Create subdirectory for raw evaluation results
            self.raw_results_dir = self.output_dir / "raw_evaluations"
            self.raw_results_dir.mkdir(exist_ok=True)

        # Initialize components
        self.metrics = EvaluationMetrics()
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
            model: Model to evaluate
            model_name: Name identifier for the model
            input_data: Input data for evaluation
            target_data: Expected target outputs
            dataset_name: Name of the dataset being used
            forward_fn: Optional custom forward function
            custom_metrics: Optional dictionary of custom metric functions

        Returns:
            BenchmarkResult with evaluation metrics and metadata
        """
        # Prepare forward function with direct execution
        if forward_fn is None:
            # Use direct model call for default forward function
            def default_forward(model, inputs):
                if callable(model):
                    return model(inputs)
                if hasattr(model, "forward"):
                    return model.forward(inputs)
                raise ValueError(f"Model {type(model)} has no callable interface")

            forward_fn = default_forward

        # Use JIT compilation for performance
        # Mark model as static since neural network models are not JAX pytrees
        jit_forward = jax.jit(forward_fn, static_argnums=(0,))

        # Warm-up run with safe execution
        try:
            result_warmup = jit_forward(model, input_data)
            # ✅ CRITICAL: Wait for GPU/TPU computation to complete
            if hasattr(result_warmup, "block_until_ready"):
                result_warmup.block_until_ready()
        except Exception as e:
            raise RuntimeError(f"Model forward pass failed during warm-up: {e}") from e

        # Timed evaluation with safe execution
        start_time = time.time()
        try:
            predictions = jit_forward(model, input_data)
            # ✅ CRITICAL: Wait for GPU/TPU computation before measuring time
            if hasattr(predictions, "block_until_ready"):
                predictions.block_until_ready()
        except Exception as e:
            raise RuntimeError(
                f"Model forward pass failed during evaluation: {e}"
            ) from e
        end_time = time.time()

        execution_time = end_time - start_time

        # Calculate metrics
        metrics = self.metrics.calculate_all_metrics(predictions, target_data)

        # Add custom metrics if provided
        if custom_metrics:
            for metric_name, metric_fn in custom_metrics.items():
                try:
                    metrics[metric_name] = float(metric_fn(predictions, target_data))
                except Exception as e:
                    print(f"Warning: Custom metric '{metric_name}' failed: {e}")

        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            execution_time=execution_time,
            framework_version="1.0.0",  # Could be made configurable
        )

        # Store result
        self.results.append(result)

        # Save if enabled
        if self.save_detailed_results:
            self.save_result(result)

        return result

    def batch_evaluate(
        self,
        models: list[tuple[str, Any]],
        datasets: list[tuple[str, Any, jax.Array, Callable | None]],
    ) -> list[BenchmarkResult]:
        """Evaluate multiple models on multiple datasets.

        Args:
            models: List of (model_name, model) tuples
            datasets: List of (dataset_name, input_data, target_data, forward_fn) tuples

        Returns:
            List of BenchmarkResults for all model-dataset combinations
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
                except Exception as e:
                    print(
                        f"Warning: Evaluation failed for {model_name} on "
                        f"{dataset_name}: {e}"
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
            model: Model to profile
            input_data: Input data for profiling
            num_runs: Number of runs for statistics
            forward_fn: Custom forward function

        Returns:
            Dictionary with performance statistics
        """
        # Use direct model execution for profiling
        if forward_fn is None:
            # Use direct model call for default forward function
            def default_forward(model, inputs):
                if callable(model):
                    return model(inputs)
                if hasattr(model, "forward"):
                    return model.forward(inputs)
                raise ValueError(f"Model {type(model)} has no callable interface")

            forward_fn = default_forward

        # Use JIT compilation for performance
        # Mark model as static since neural network models are not JAX pytrees
        jit_forward = jax.jit(forward_fn, static_argnums=(0,))

        # Warm-up with safe execution
        result_warmup = jit_forward(model, input_data)
        # ✅ CRITICAL: Wait for GPU/TPU computation to complete during warmup
        if hasattr(result_warmup, "block_until_ready"):
            result_warmup.block_until_ready()

        # Multiple timed runs with safe execution
        execution_times: list[float] = []
        for _ in range(num_runs):
            start_time = time.time()
            result_timed = jit_forward(model, input_data)
            # ✅ CRITICAL: Wait for GPU/TPU computation before measuring time
            if hasattr(result_timed, "block_until_ready"):
                result_timed.block_until_ready()
            end_time = time.time()
            execution_times.append(end_time - start_time)

        execution_times_array = jnp.array(execution_times)

        return {
            "mean_execution_time": float(jnp.mean(execution_times_array)),
            "std_execution_time": float(jnp.std(execution_times_array)),
            "min_execution_time": float(jnp.min(execution_times_array)),
            "max_execution_time": float(jnp.max(execution_times_array)),
            "memory_usage": 0,  # Placeholder - could be implemented with JAX profiling
        }

    def save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to file.

        Args:
            result: BenchmarkResult to save
        """
        if not self.save_detailed_results:
            return

        # Ensure timestamp is not None (should be set in __post_init__)
        timestamp_str = result.timestamp or datetime.now(tz=UTC).isoformat()
        timestamp = timestamp_str.replace(":", "-").replace(".", "-")
        filename = f"{result.model_name}_{result.dataset_name}_{timestamp}.json"
        # Save to raw_evaluations subdirectory
        filepath = self.raw_results_dir / filename

        # Convert to serializable format
        result_dict = asdict(result)

        with open(filepath, "w") as f:
            json.dump(result_dict, f, indent=2)

    def load_results(self) -> list[BenchmarkResult]:
        """Load all benchmark results from files.

        Returns:
            List of BenchmarkResults
        """
        if not self.save_detailed_results or not self.output_dir.exists():
            return []

        results = []
        # Look in the raw_evaluations subdirectory where files are actually saved
        search_dir = (
            self.raw_results_dir
            if hasattr(self, "raw_results_dir") and self.raw_results_dir.exists()
            else self.output_dir
        )
        for filepath in search_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    result_dict = json.load(f)

                # Convert back to BenchmarkResult
                result = BenchmarkResult(**result_dict)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to load result from {filepath}: {e}")

        return results

    def generate_summary_report(self) -> dict[str, Any]:
        """Generate complete summary report of all evaluations.

        Returns:
            Dictionary with summary statistics and analysis
        """
        if not self.results:
            return {"error": "No benchmark results available"}

        # Group results by model and dataset
        model_performance: dict[str, list[float]] = {}
        dataset_difficulty: dict[str, list[float]] = {}

        for result in self.results:
            # Model performance tracking
            if result.model_name not in model_performance:
                model_performance[result.model_name] = []
            model_performance[result.model_name].append(result.metrics["mse"])

            # Dataset difficulty tracking
            if result.dataset_name not in dataset_difficulty:
                dataset_difficulty[result.dataset_name] = []
            dataset_difficulty[result.dataset_name].append(result.metrics["mse"])

        # Calculate summary statistics
        summary: dict[str, Any] = {
            "total_evaluations": len(self.results),
            "unique_models": len(model_performance),
            "unique_datasets": len(dataset_difficulty),
            "model_rankings": {},
            "dataset_difficulty_rankings": {},
        }

        # Rank models by average performance
        for model_name, mse_values in model_performance.items():
            avg_mse = float(jnp.mean(jnp.array(mse_values)))
            ci_lower, ci_upper = (
                self.statistical_analyzer.calculate_confidence_interval(
                    jnp.array(mse_values)
                )
            )
            summary["model_rankings"][model_name] = {
                "average_mse": avg_mse,
                "confidence_interval": [ci_lower, ci_upper],
                "num_evaluations": len(mse_values),
            }

        # Rank datasets by difficulty (average MSE across all models)
        for dataset_name, mse_values in dataset_difficulty.items():
            avg_mse = float(jnp.mean(jnp.array(mse_values)))
            summary["dataset_difficulty_rankings"][dataset_name] = {
                "average_mse": avg_mse,
                "num_evaluations": len(mse_values),
            }

        return summary


# Export main classes
__all__ = [
    "BenchmarkEvaluator",
    "BenchmarkResult",
    "EvaluationMetrics",
    "StatisticalAnalyzer",
]
