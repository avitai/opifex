"""
Visualization Tools Module

This module provides visualization utilities for PDEBench benchmarking results.
It focuses on generating figure metadata and configuration rather than
actual plotting to integrate optimally with the core scientific framework.

Key Features:
- Figure metadata generation for comparison charts
- Configuration for publication-ready visualizations
- Support for multiple chart types and metrics
- Integration with benchmarking infrastructure

Following Critical Technical Guidelines:
- JAX-native data processing
- Type hints and comprehensive documentation
- No external plotting dependencies (metadata only)
"""

from typing import Any

from calibrax.core import BenchmarkResult

from opifex.benchmarking._shared import LOWER_IS_BETTER


def _name(r: BenchmarkResult) -> str:
    return r.name


def _dataset(r: BenchmarkResult) -> str:
    return r.tags.get("dataset", r.name)


def _exec_time(r: BenchmarkResult) -> float:
    return r.metadata.get("execution_time", 0.0)


def _metric_val(r: BenchmarkResult, metric: str) -> float | None:
    m = r.metrics.get(metric)
    return m.value if m is not None else None


class PDEBenchVisualizer:
    """
    Visualization utilities for PDEBench benchmark results.

    This class generates figure metadata and configurations for
    creating charts and plots of benchmark results. It avoids
    direct plotting to maintain lightweight dependencies.
    """

    def __init__(self) -> None:
        """Initialize PDEBench visualizer."""
        self.supported_metrics = [
            "mse",
            "mae",
            "r2_score",
            "relative_error",
            "rmse",
            "mape",
        ]

        self.chart_types = [
            "comparison_chart",
            "performance_trends",
            "error_distribution",
            "baseline_comparison",
            "model_ranking",
        ]

    def create_comparison_chart(
        self,
        results: list[BenchmarkResult],
        metric: str,
        title: str = "Model Comparison",
        sort_by_performance: bool = True,
    ) -> dict[str, Any]:
        """
        Create metadata for a model comparison chart.

        Args:
            results: List of benchmark results to compare
            metric: Metric to use for comparison
            title: Chart title
            sort_by_performance: Whether to sort results by performance

        Returns:
            Dictionary with figure metadata and configuration
        """
        if metric not in self.supported_metrics:
            raise ValueError(f"Unsupported metric: {metric}")

        # Extract and sort data
        chart_data = []
        for result in results:
            val = _metric_val(result, metric)
            if val is not None:
                chart_data.append(
                    {
                        "model_name": _name(result),
                        "dataset_name": _dataset(result),
                        "value": val,
                        "execution_time": _exec_time(result),
                    }
                )

        # Sort by performance if requested
        if sort_by_performance:
            # Lower is better for error metrics, higher for others
            reverse = metric.lower() not in [
                "mse",
                "mae",
                "rmse",
                "relative_error",
                "mape",
            ]
            chart_data.sort(key=lambda x: x["value"], reverse=reverse)

        return {
            "figure_type": "comparison_chart",
            "title": title,
            "metric": metric,
            "data": chart_data,
            "metrics_compared": [metric],
            "num_models": len(chart_data),
            "sort_order": "ascending"
            if not sort_by_performance or metric.lower() in LOWER_IS_BETTER
            else "descending",
            "chart_config": {
                "x_axis": "model_name",
                "y_axis": metric,
                "chart_type": "bar",
                "color_scheme": "performance_based",
            },
        }

    def create_multi_metric_comparison(
        self,
        results: list[BenchmarkResult],
        metrics: list[str],
        title: str = "Multi-Metric Comparison",
    ) -> dict[str, Any]:
        """
        Create metadata for multi-metric comparison chart.

        Args:
            results: List of benchmark results
            metrics: List of metrics to compare
            title: Chart title

        Returns:
            Dictionary with figure metadata
        """
        # Validate metrics
        invalid_metrics = [m for m in metrics if m not in self.supported_metrics]
        if invalid_metrics:
            raise ValueError(f"Unsupported metrics: {invalid_metrics}")

        # Organize data by model and metric
        model_data: dict[str, dict[str, Any]] = {}
        for result in results:
            model_key = f"{_name(result)}_{_dataset(result)}"
            if model_key not in model_data:
                model_data[model_key] = {
                    "model_name": _name(result),
                    "dataset_name": _dataset(result),
                    "metrics": {},
                }

            for metric in metrics:
                val = _metric_val(result, metric)
                if val is not None:
                    model_data[model_key]["metrics"][metric] = val

        return {
            "figure_type": "multi_metric_comparison",
            "title": title,
            "metrics": metrics,
            "data": list(model_data.values()),
            "metrics_compared": metrics,
            "num_models": len(model_data),
            "chart_config": {
                "chart_type": "radar" if len(metrics) > 3 else "grouped_bar",
                "normalize_metrics": True,
                "color_scheme": "categorical",
            },
        }

    def create_performance_trends(
        self,
        results: list[BenchmarkResult],
        group_by: str = "dataset_name",
        metric: str = "mse",
    ) -> dict[str, Any]:
        """
        Create metadata for performance trends visualization.

        Args:
            results: List of benchmark results
            group_by: Field to group results by
            metric: Metric to track trends for

        Returns:
            Dictionary with trend visualization metadata
        """
        # Group results
        grouped_data: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            if group_by == "dataset_name":
                group_key = _dataset(result)
            else:
                group_key = getattr(result, group_by, "Unknown")
            if group_key not in grouped_data:
                grouped_data[group_key] = []

            val = _metric_val(result, metric)
            if val is not None:
                grouped_data[group_key].append(
                    {
                        "model_name": _name(result),
                        "value": val,
                        "execution_time": _exec_time(result),
                    }
                )

        return {
            "figure_type": "performance_trends",
            "title": f"Performance Trends by {group_by.replace('_', ' ').title()}",
            "metric": metric,
            "group_by": group_by,
            "data": grouped_data,
            "metrics_compared": [metric],
            "num_groups": len(grouped_data),
            "chart_config": {
                "chart_type": "line" if len(grouped_data) > 1 else "scatter",
                "x_axis": group_by,
                "y_axis": metric,
                "show_trend_lines": True,
            },
        }

    def create_baseline_comparison(
        self,
        results: list[BenchmarkResult],
        baseline_metrics: dict[str, dict[str, float]],
        metric: str = "mse",
    ) -> dict[str, Any]:
        """
        Create metadata for baseline comparison visualization.

        Args:
            results: Test results to compare
            baseline_metrics: Dictionary of baseline metrics by model type
            metric: Metric to use for comparison

        Returns:
            Dictionary with baseline comparison metadata
        """
        comparison_data = []

        for result in results:
            val = _metric_val(result, metric)
            if val is None:
                continue

            # Find matching baseline
            baseline_value = None
            for baseline_model, baseline_data in baseline_metrics.items():
                if baseline_model.lower() in _name(result).lower():
                    baseline_value = baseline_data.get(metric)
                    break

            comparison_data.append(
                {
                    "model_name": _name(result),
                    "dataset_name": _dataset(result),
                    "test_value": val,
                    "baseline_value": baseline_value,
                    "improvement": (baseline_value - val) / baseline_value
                    if baseline_value
                    else None,
                }
            )

        return {
            "figure_type": "baseline_comparison",
            "title": f"Baseline Comparison - {metric.upper()}",
            "metric": metric,
            "data": comparison_data,
            "metrics_compared": [metric],
            "has_baselines": any(
                d["baseline_value"] is not None for d in comparison_data
            ),
            "chart_config": {
                "chart_type": "paired_bar",
                "show_improvement": True,
                "color_scheme": "improvement_based",
            },
        }

    def create_error_distribution(
        self, results: list[BenchmarkResult], error_metric: str = "mae"
    ) -> dict[str, Any]:
        """
        Create metadata for error distribution visualization.

        Args:
            results: List of benchmark results
            error_metric: Error metric to analyze distribution for

        Returns:
            Dictionary with error distribution metadata
        """
        error_values: list[float] = []
        model_errors: dict[str, list[float]] = {}

        for result in results:
            val = _metric_val(result, error_metric)
            if val is not None:
                error_values.append(val)

                model_key = _name(result)
                if model_key not in model_errors:
                    model_errors[model_key] = []
                model_errors[model_key].append(val)

        # Calculate basic statistics
        if error_values:
            import statistics

            stats = {
                "mean": statistics.mean(error_values),
                "median": statistics.median(error_values),
                "std_dev": statistics.stdev(error_values)
                if len(error_values) > 1
                else 0,
                "min": min(error_values),
                "max": max(error_values),
            }
        else:
            stats = {}

        return {
            "figure_type": "error_distribution",
            "title": f"Error Distribution - {error_metric.upper()}",
            "metric": error_metric,
            "error_values": error_values,
            "model_errors": model_errors,
            "statistics": stats,
            "metrics_compared": [error_metric],
            "num_samples": len(error_values),
            "chart_config": {
                "chart_type": "histogram",
                "show_statistics": True,
                "bins": min(20, len(error_values) // 5) if error_values else 10,
            },
        }

    def create_model_ranking(
        self,
        results: list[BenchmarkResult],
        ranking_metrics: list[str],
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Create metadata for model ranking visualization.

        Args:
            results: List of benchmark results
            ranking_metrics: Metrics to use for ranking
            weights: Optional weights for each metric

        Returns:
            Dictionary with model ranking metadata
        """
        if weights is None:
            weights = dict.fromkeys(ranking_metrics, 1.0)

        # Calculate composite scores
        model_scores: dict[str, dict[str, Any]] = {}
        for result in results:
            model_key = f"{_name(result)}_{_dataset(result)}"

            score = 0.0
            total_weight = 0.0

            for metric in ranking_metrics:
                val = _metric_val(result, metric)
                if val is not None and metric in weights:
                    weight = weights[metric]

                    # Normalize score (higher is better)
                    if metric.lower() in [
                        "mse",
                        "mae",
                        "rmse",
                        "relative_error",
                        "mape",
                    ]:
                        # For error metrics, invert (lower error = higher score)
                        normalized_score = 1.0 / (1.0 + val)
                    else:
                        # For performance metrics, use direct value
                        normalized_score = val

                    score += normalized_score * weight
                    total_weight += weight

            if total_weight > 0:
                model_scores[model_key] = {
                    "model_name": _name(result),
                    "dataset_name": _dataset(result),
                    "composite_score": score / total_weight,
                    "individual_metrics": {
                        m: _metric_val(result, m) for m in ranking_metrics
                    },
                }

        # Sort by composite score
        ranked_models = sorted(
            model_scores.values(), key=lambda x: x["composite_score"], reverse=True
        )

        return {
            "figure_type": "model_ranking",
            "title": "Model Performance Ranking",
            "ranking_metrics": ranking_metrics,
            "weights": weights,
            "ranked_models": ranked_models,
            "metrics_compared": ranking_metrics,
            "num_models": len(ranked_models),
            "chart_config": {
                "chart_type": "horizontal_bar",
                "show_scores": True,
                "color_scheme": "ranking_based",
            },
        }

    def get_visualization_summary(
        self, results: list[BenchmarkResult]
    ) -> dict[str, Any]:
        """
        Generate a summary of available visualization options.

        Args:
            results: List of benchmark results

        Returns:
            Dictionary with visualization recommendations
        """
        # Analyze available data
        available_metrics_set: set[str] = set()
        model_names_set: set[str] = set()
        dataset_names_set: set[str] = set()

        for result in results:
            available_metrics_set.update(result.metrics.keys())
            model_names_set.add(_name(result))
            dataset_names_set.add(_dataset(result))

        available_metrics = list(available_metrics_set)
        model_names = list(model_names_set)
        dataset_names = list(dataset_names_set)

        # Recommend visualizations
        recommendations = []

        if len(model_names) > 1:
            recommendations.append("comparison_chart")
            recommendations.append("model_ranking")

        if len(dataset_names) > 1:
            recommendations.append("performance_trends")

        if len(available_metrics) > 2:
            recommendations.append("multi_metric_comparison")

        if len(results) > 5:
            recommendations.append("error_distribution")

        return {
            "num_results": len(results),
            "available_metrics": available_metrics,
            "model_names": model_names,
            "dataset_names": dataset_names,
            "recommended_visualizations": recommendations,
            "supported_chart_types": self.chart_types,
        }
