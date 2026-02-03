"""Report generation for PDEBench evaluation and benchmarking results.

This module provides comprehensive report generation capabilities for PDEBench
evaluation results, including statistical analysis, baseline comparisons, and
publication-ready formatted outputs.
"""

import json
from datetime import datetime, UTC
from typing import Any

# pyright: reportArgumentType=false, reportReturnType=false, reportAssignmentType=false
import jax.numpy as jnp


class PDEBenchReportGenerator:
    """Generator for comprehensive PDEBench evaluation reports.

    Creates detailed reports from evaluation results including statistical
    analysis, baseline comparisons, and multiple output formats for both
    programmatic access and human readability.
    """

    def __init__(self, report_format: str = "json"):
        """Initialize the report generator.

        Args:
            report_format: Default output format ("json" or "text")
        """
        self.report_format = report_format
        self.generation_timestamp = datetime.now(UTC).isoformat()

    def generate_evaluation_report(
        self,
        evaluation_results: dict[str, Any],
        baseline_comparisons: dict[str, Any] | None = None,
        dataset_info: dict[str, str] | None = None,
        model_info: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive evaluation report.

        Args:
            evaluation_results: Results from benchmarking evaluation
            baseline_comparisons: Optional baseline comparison data
            dataset_info: Optional dataset metadata
            model_info: Optional model metadata

        Returns:
            Complete evaluation report dictionary
        """
        return {
            "metadata": self._generate_metadata(dataset_info, model_info),
            "evaluation_summary": self._generate_evaluation_summary(evaluation_results),
            "detailed_metrics": self._generate_detailed_metrics(evaluation_results),
            "statistical_analysis": self._generate_statistical_analysis(
                evaluation_results
            ),
            "baseline_comparison": self._generate_baseline_comparison(
                baseline_comparisons
            ),
            "recommendations": self._generate_recommendations(evaluation_results),
            "generation_info": {
                "timestamp": self.generation_timestamp,
                "format": self.report_format,
                "generator_version": "1.0.0",
            },
        }

    def _generate_metadata(
        self, dataset_info: dict[str, str] | None, model_info: dict[str, str] | None
    ) -> dict[str, Any]:
        """Generate report metadata section."""
        metadata: dict[str, Any] = {
            "report_type": "PDEBench Evaluation Report",
            "generated_at": self.generation_timestamp,
        }

        if dataset_info:
            metadata["dataset"] = {
                "name": dataset_info.get("name", "Unknown"),
                "type": dataset_info.get("type", "Unknown"),
                "size": dataset_info.get("size", "Unknown"),
                "description": dataset_info.get(
                    "description", "No description available"
                ),
            }

        if model_info:
            metadata["model"] = {
                "name": model_info.get("name", "Unknown"),
                "type": model_info.get("type", "Unknown"),
                "parameters": model_info.get("parameters", "Unknown"),
                "architecture": model_info.get("architecture", "Unknown"),
            }

        return metadata

    def _generate_evaluation_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate high-level evaluation summary."""
        summary: dict[str, Any] = {
            "overall_performance": "Good",  # Will be determined by metrics
            "key_metrics": {},
            "notable_findings": [],
        }

        # Extract key metrics
        if "mse" in results:
            mse_value = float(results["mse"])
            summary["key_metrics"]["mse"] = mse_value

            # Determine performance level
            if mse_value < 0.01:
                summary["overall_performance"] = "Excellent"
            elif mse_value < 0.1:
                summary["overall_performance"] = "Good"
            elif mse_value < 1.0:
                summary["overall_performance"] = "Fair"
            else:
                summary["overall_performance"] = "Needs Improvement"

        if "mae" in results:
            summary["key_metrics"]["mae"] = float(results["mae"])

        if "r2_score" in results:
            r2_value = float(results["r2_score"])
            summary["key_metrics"]["r2_score"] = r2_value

            if r2_value > 0.95:
                summary["notable_findings"].append(
                    "Excellent correlation with ground truth"
                )
            elif r2_value < 0.5:
                summary["notable_findings"].append(
                    "Low correlation with ground truth - model may need improvement"
                )

        # Add computational efficiency notes
        if "evaluation_time" in results:
            eval_time = float(results["evaluation_time"])
            summary["key_metrics"]["evaluation_time_seconds"] = eval_time

            if eval_time < 1.0:
                summary["notable_findings"].append("Very fast inference time")
            elif eval_time > 10.0:
                summary["notable_findings"].append(
                    "Slow inference time - consider optimization"
                )

        return summary

    def _generate_detailed_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate detailed metrics breakdown."""
        detailed = {
            "accuracy_metrics": {},
            "efficiency_metrics": {},
            "statistical_metrics": {},
        }

        # Accuracy metrics
        accuracy_keys = ["mse", "mae", "rmse", "r2_score", "relative_error"]
        for key in accuracy_keys:
            if key in results:
                detailed["accuracy_metrics"][key] = float(results[key])

        # Efficiency metrics
        efficiency_keys = ["evaluation_time", "memory_usage", "flops"]
        for key in efficiency_keys:
            if key in results:
                detailed["efficiency_metrics"][key] = float(results[key])

        # Statistical metrics
        statistical_keys = ["mean_prediction", "std_prediction", "confidence_interval"]
        for key in statistical_keys:
            if key in results:
                if key == "confidence_interval" and isinstance(
                    results[key], (list, tuple)
                ):
                    detailed["statistical_metrics"][key] = [
                        float(x) for x in results[key]
                    ]
                else:
                    detailed["statistical_metrics"][key] = float(results[key])

        return detailed

    def _generate_statistical_analysis(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate statistical analysis section."""
        analysis = {
            "distribution_analysis": {},
            "uncertainty_quantification": {},
            "significance_tests": {},
        }

        # Distribution analysis
        if "predictions" in results and hasattr(results["predictions"], "__len__"):
            predictions = jnp.array(results["predictions"])
            analysis["distribution_analysis"] = {
                "mean": float(jnp.mean(predictions)),
                "std": float(jnp.std(predictions)),
                "min": float(jnp.min(predictions)),
                "max": float(jnp.max(predictions)),
                "median": float(jnp.median(predictions)),
            }

        # Uncertainty quantification
        if "epistemic_uncertainty" in results:
            analysis["uncertainty_quantification"]["epistemic"] = float(
                results["epistemic_uncertainty"]
            )

        if "aleatoric_uncertainty" in results:
            analysis["uncertainty_quantification"]["aleatoric"] = float(
                results["aleatoric_uncertainty"]
            )

        # Significance tests (placeholder for future implementation)
        analysis["significance_tests"]["status"] = "Not computed"

        return analysis

    def _generate_baseline_comparison(
        self, baseline_comparisons: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Generate baseline comparison section."""
        if not baseline_comparisons:
            return {"status": "No baseline comparisons available"}

        comparison = {
            "baseline_models": [],
            "performance_ranking": {},
            "improvement_analysis": {},
        }

        # Extract baseline information
        if "baselines" in baseline_comparisons:
            for baseline_name, baseline_data in baseline_comparisons[
                "baselines"
            ].items():
                baseline_info = {
                    "name": baseline_name,
                    "mse": float(baseline_data.get("mse", 0.0)),
                    "relative_improvement": float(
                        baseline_data.get("relative_improvement", 0.0)
                    ),
                }
                comparison["baseline_models"].append(baseline_info)

        # Performance ranking
        if "current_model_performance" in baseline_comparisons:
            current_perf = baseline_comparisons["current_model_performance"]
            comparison["performance_ranking"] = {
                "current_model_rank": int(current_perf.get("rank", 0)),
                "total_models": int(current_perf.get("total", 1)),
                "percentile": float(current_perf.get("percentile", 0.0)),
            }

        return comparison

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        # Performance-based recommendations
        if "mse" in results:
            mse_value = float(results["mse"])
            if mse_value > 1.0:
                recommendations.append(
                    "Consider increasing model complexity or training time to "
                    "improve accuracy"
                )
            elif mse_value < 0.001:
                recommendations.append(
                    "Excellent accuracy achieved - model is ready for production use"
                )

        # Efficiency recommendations
        if "evaluation_time" in results:
            eval_time = float(results["evaluation_time"])
            if eval_time > 5.0:
                recommendations.append(
                    "Consider model optimization techniques to reduce inference time"
                )

        # Uncertainty recommendations
        if "epistemic_uncertainty" in results:
            epistemic = float(results["epistemic_uncertainty"])
            if epistemic > 0.5:
                recommendations.append(
                    "High epistemic uncertainty - consider ensemble methods or "
                    "more training data"
                )

        # Data quality recommendations
        if "r2_score" in results:
            r2 = float(results["r2_score"])
            if r2 < 0.7:
                recommendations.append(
                    "Low R² score suggests potential data quality issues or "
                    "model underfitting"
                )

        if not recommendations:
            recommendations.append(
                "Model performance appears satisfactory across all evaluated metrics"
            )

        return recommendations

    def format_report_as_text(self, report: dict[str, Any]) -> str:
        """Format report as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("PDEBench Evaluation Report")
        lines.append("=" * 60)
        lines.append("")

        # Metadata
        if "metadata" in report:
            lines.append("METADATA")
            lines.append("-" * 30)
            metadata = report["metadata"]
            lines.append(f"Generated: {metadata.get('generated_at', 'Unknown')}")

            if "dataset" in metadata:
                dataset = metadata["dataset"]
                lines.append(
                    f"Dataset: {dataset.get('name', 'Unknown')} "
                    f"({dataset.get('type', 'Unknown')})"
                )

            if "model" in metadata:
                model = metadata["model"]
                lines.append(
                    f"Model: {model.get('name', 'Unknown')} "
                    f"({model.get('type', 'Unknown')})"
                )
            lines.append("")

        # Summary
        if "evaluation_summary" in report:
            lines.append("EVALUATION SUMMARY")
            lines.append("-" * 30)
            summary = report["evaluation_summary"]
            lines.append(
                f"Overall Performance: {summary.get('overall_performance', 'Unknown')}"
            )

            if "key_metrics" in summary:
                lines.append("\nKey Metrics:")
                for metric, value in summary["key_metrics"].items():
                    lines.append(f"  {metric}: {value}")

            if summary.get("notable_findings"):
                lines.append("\nNotable Findings:")
                for finding in summary["notable_findings"]:
                    lines.append(f"  • {finding}")
            lines.append("")

        # Recommendations
        if "recommendations" in report:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 30)
            for i, rec in enumerate(report["recommendations"], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_report(
        self, report: dict[str, Any], filepath: str, format_type: str | None = None
    ) -> None:
        """Save report to file.

        Args:
            report: Report data to save
            filepath: Output file path
            format_type: Output format ("json" or "text"), defaults to
                self.report_format
        """
        format_type = format_type or self.report_format

        if format_type == "json":
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)
        elif format_type == "text":
            text_report = self.format_report_as_text(report)
            with open(filepath, "w") as f:
                f.write(text_report)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def generate_summary_statistics(
        self, reports: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate summary statistics across multiple reports.

        Args:
            reports: List of evaluation reports to analyze

        Returns:
            Summary statistics across all reports
        """
        if not reports:
            return {"error": "No reports provided for analysis"}

        summary = {
            "total_reports": len(reports),
            "aggregated_metrics": {},
            "performance_trends": {},
            "recommendations": [],
        }

        # Aggregate metrics across reports
        all_metrics: dict[str, dict[str, list[float]]] = {}
        for report in reports:
            if "detailed_metrics" in report:
                for category, metrics in report["detailed_metrics"].items():
                    if category not in all_metrics:
                        all_metrics[category] = {}
                    for metric, value in metrics.items():
                        if metric not in all_metrics[category]:
                            all_metrics[category][metric] = []
                        all_metrics[category][metric].append(float(value))

        # Compute summary statistics
        for category, category_metrics in all_metrics.items():
            summary["aggregated_metrics"][category] = {}
            for metric, values in category_metrics.items():
                values_array = jnp.array(values)
                summary["aggregated_metrics"][category][metric] = {
                    "mean": float(jnp.mean(values_array)),
                    "std": float(jnp.std(values_array)),
                    "min": float(jnp.min(values_array)),
                    "max": float(jnp.max(values_array)),
                }

        return summary

    def generate_comprehensive_report(
        self,
        results: list,
        include_baseline_comparison: bool = True,
        include_statistical_analysis: bool = True,
    ) -> dict[str, Any]:
        """Generate comprehensive report from benchmark results.

        Args:
            results: List of BenchmarkResult objects
            include_baseline_comparison: Whether to include baseline comparisons
            include_statistical_analysis: Whether to include statistical analysis

        Returns:
            Comprehensive report dictionary
        """
        if not results:
            return {"error": "No results provided for report generation"}

        # Convert results to evaluation format
        evaluation_results = {}
        detailed_results = []

        for result in results:
            # Extract metrics from BenchmarkResult
            result_data = {
                "model_name": result.model_name,
                "dataset_name": result.dataset_name,
                "metrics": result.metrics,
                "execution_time": result.execution_time,
            }
            detailed_results.append(result_data)

            # Aggregate metrics for overall evaluation
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in evaluation_results:
                    evaluation_results[metric_name] = []
                evaluation_results[metric_name].append(metric_value)

        # Compute aggregate metrics
        for metric_name, values in evaluation_results.items():
            values_array = jnp.array(values)
            evaluation_results[metric_name] = float(jnp.mean(values_array))

        # Generate the base report
        report = self.generate_evaluation_report(
            evaluation_results=evaluation_results,
            baseline_comparisons=None if not include_baseline_comparison else {},
            dataset_info={"name": "Multiple Datasets"},
            model_info={"name": "Multiple Models"},
        )

        # Add detailed results
        report["detailed_results"] = detailed_results

        # Add statistical analysis if requested
        if include_statistical_analysis:
            report["statistical_analysis"]["cross_dataset_analysis"] = (
                self._analyze_cross_dataset_performance(detailed_results)
            )

        return report

    def _analyze_cross_dataset_performance(
        self, detailed_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze performance across different datasets.

        Args:
            detailed_results: List of detailed result dictionaries

        Returns:
            Cross-dataset performance analysis
        """
        analysis = {
            "dataset_performance": {},
            "model_rankings": {},
            "consistency_metrics": {},
        }

        # Group results by dataset
        dataset_results = {}
        for result in detailed_results:
            dataset = result["dataset_name"]
            if dataset not in dataset_results:
                dataset_results[dataset] = []
            dataset_results[dataset].append(result)

        # Analyze performance per dataset
        for dataset, results in dataset_results.items():
            if not results:
                continue

            # Aggregate metrics for this dataset
            dataset_metrics = {}
            for result in results:
                for metric_name, metric_value in result["metrics"].items():
                    if metric_name not in dataset_metrics:
                        dataset_metrics[metric_name] = []
                    dataset_metrics[metric_name].append(metric_value)

            # Compute statistics
            analysis["dataset_performance"][dataset] = {}
            for metric_name, values in dataset_metrics.items():
                values_array = jnp.array(values)
                analysis["dataset_performance"][dataset][metric_name] = {
                    "mean": float(jnp.mean(values_array)),
                    "std": float(jnp.std(values_array)),
                    "best": float(jnp.min(values_array))
                    if metric_name in ["mse", "mae"]
                    else float(jnp.max(values_array)),
                }

        return analysis
