"""Report generation for PDEBench evaluation and benchmarking results.

This module provides full report generation capabilities for PDEBench
evaluation results, including statistical analysis, baseline comparisons, and
publication-ready formatted outputs.
"""

import json
from datetime import datetime, UTC
from typing import Any

import numpy as np
from calibrax.core import BenchmarkResult
from calibrax.statistics import paired_significance_test

from opifex.benchmarking._shared import ACCURACY_METRIC_KEYS


def _performance_label(mse: float) -> str:
    """Map an MSE value to a qualitative performance label."""
    if mse < 0.01:
        return "Excellent"
    if mse < 0.1:
        return "Good"
    if mse < 1.0:
        return "Fair"
    return "Needs Improvement"


def _correlation_finding(r2_score: float) -> str | None:
    """Notable finding for an R-squared correlation value, if any."""
    if r2_score > 0.95:
        return "Excellent correlation with ground truth"
    if r2_score < 0.5:
        return "Low correlation with ground truth - model may need improvement"
    return None


def _timing_finding(evaluation_time: float) -> str | None:
    """Notable finding for an evaluation-time value, if any."""
    if evaluation_time < 1.0:
        return "Very fast inference time"
    if evaluation_time > 10.0:
        return "Slow inference time - consider optimization"
    return None


def _mse_recommendation(mse: float) -> str | None:
    """Accuracy recommendation driven by the MSE value."""
    if mse > 1.0:
        return "Consider increasing model complexity or training time to improve accuracy"
    if mse < 0.001:
        return "Excellent accuracy achieved - model is ready for production use"
    return None


def _timing_recommendation(evaluation_time: float) -> str | None:
    """Efficiency recommendation driven by the evaluation time."""
    if evaluation_time > 5.0:
        return "Consider model optimization techniques to reduce inference time"
    return None


def _uncertainty_recommendation(epistemic: float) -> str | None:
    """Recommendation driven by epistemic uncertainty."""
    if epistemic > 0.5:
        return "High epistemic uncertainty - consider ensemble methods or more training data"
    return None


def _r2_recommendation(r2_score: float) -> str | None:
    """Recommendation driven by the R-squared score."""
    if r2_score < 0.7:
        return "Low R-squared score suggests potential data quality issues or model underfitting"
    return None


_RECOMMENDATION_RULES = (
    ("mse", _mse_recommendation),
    ("evaluation_time", _timing_recommendation),
    ("epistemic_uncertainty", _uncertainty_recommendation),
    ("r2_score", _r2_recommendation),
)


def _format_metadata_section(metadata: dict[str, Any]) -> list[str]:
    """Render the METADATA section lines for the text report."""
    lines = ["METADATA", "-" * 30, f"Generated: {metadata.get('generated_at', 'Unknown')}"]
    if "dataset" in metadata:
        dataset = metadata["dataset"]
        lines.append(
            f"Dataset: {dataset.get('name', 'Unknown')} ({dataset.get('type', 'Unknown')})"
        )
    if "model" in metadata:
        model = metadata["model"]
        lines.append(f"Model: {model.get('name', 'Unknown')} ({model.get('type', 'Unknown')})")
    lines.append("")
    return lines


def _format_summary_section(summary: dict[str, Any]) -> list[str]:
    """Render the EVALUATION SUMMARY section lines for the text report."""
    lines = [
        "EVALUATION SUMMARY",
        "-" * 30,
        f"Overall Performance: {summary.get('overall_performance', 'Unknown')}",
    ]
    if "key_metrics" in summary:
        lines.append("\nKey Metrics:")
        lines.extend(f"  {metric}: {value}" for metric, value in summary["key_metrics"].items())
    if summary.get("notable_findings"):
        lines.append("\nNotable Findings:")
        lines.extend(f"  • {finding}" for finding in summary["notable_findings"])
    lines.append("")
    return lines


def _format_recommendations_section(recommendations: list[str]) -> list[str]:
    """Render the RECOMMENDATIONS section lines for the text report."""
    lines = ["RECOMMENDATIONS", "-" * 30]
    lines.extend(f"{index}. {rec}" for index, rec in enumerate(recommendations, 1))
    lines.append("")
    return lines


class PDEBenchReportGenerator:
    """Generator for full PDEBench evaluation reports.

    Creates detailed reports from evaluation results including statistical
    analysis, baseline comparisons, and multiple output formats for both
    programmatic access and human readability.
    """

    def __init__(self, report_format: str = "json") -> None:
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
        """Generate full evaluation report.

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
            "statistical_analysis": self._generate_statistical_analysis(evaluation_results),
            "baseline_comparison": self._generate_baseline_comparison(baseline_comparisons),
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
                "description": dataset_info.get("description", "No description available"),
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
            summary["overall_performance"] = _performance_label(mse_value)

        if "mae" in results:
            summary["key_metrics"]["mae"] = float(results["mae"])

        if "r2_score" in results:
            r2_value = float(results["r2_score"])
            summary["key_metrics"]["r2_score"] = r2_value
            finding = _correlation_finding(r2_value)
            if finding is not None:
                summary["notable_findings"].append(finding)

        # Add computational efficiency notes
        if "evaluation_time" in results:
            eval_time = float(results["evaluation_time"])
            summary["key_metrics"]["evaluation_time_seconds"] = eval_time
            finding = _timing_finding(eval_time)
            if finding is not None:
                summary["notable_findings"].append(finding)

        return summary

    def _generate_detailed_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate detailed metrics breakdown."""
        detailed = {
            "accuracy_metrics": {},
            "efficiency_metrics": {},
            "statistical_metrics": {},
        }

        # Accuracy metrics
        accuracy_keys = list(ACCURACY_METRIC_KEYS)
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
                if key == "confidence_interval" and isinstance(results[key], list | tuple):
                    detailed["statistical_metrics"][key] = [float(x) for x in results[key]]
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
            predictions = np.array(results["predictions"])
            analysis["distribution_analysis"] = {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
                "median": float(np.median(predictions)),
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

        analysis["significance_tests"] = self._compute_significance(results)

        return analysis

    @staticmethod
    def _compute_significance(results: dict[str, Any]) -> dict[str, Any]:
        """Run a paired significance test of the model against a baseline.

        Benchmark comparisons evaluate the model and a baseline on the *same*
        test samples, so the two per-sample error sequences are paired. This
        method applies the Wilcoxon signed-rank test (the non-parametric paired
        test; see Wilcoxon, 1945, "Individual Comparisons by Ranking Methods",
        Biometrics Bulletin 1(6):80-83) via
        :func:`calibrax.statistics.paired_significance_test`.

        Args:
            results: Evaluation results. The test runs only when both
                ``per_sample_errors`` (model) and ``baseline_per_sample_errors``
                (paired baseline) are present and of equal, non-empty length.

        Returns:
            A dict with ``status`` of ``"computed"`` (plus ``p_value``,
            ``statistic``, ``effect_size``, ``significant`` and ``method``) when
            paired data is available, otherwise ``{"status": "not_tested", ...}``
            with an honest reason. No silent or fabricated p-values.
        """
        model_errors = results.get("per_sample_errors")
        baseline_errors = results.get("baseline_per_sample_errors")

        if model_errors is None or baseline_errors is None:
            return {
                "status": "not_tested",
                "reason": "paired per-sample errors not provided",
            }

        model_seq = [float(x) for x in model_errors]
        baseline_seq = [float(x) for x in baseline_errors]

        if not model_seq or len(model_seq) != len(baseline_seq):
            return {
                "status": "not_tested",
                "reason": "paired samples are empty or of unequal length",
            }

        result = paired_significance_test(model_seq, baseline_seq)
        return {"status": "computed", **result.to_dict()}

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
            for baseline_name, baseline_data in baseline_comparisons["baselines"].items():
                baseline_info = {
                    "name": baseline_name,
                    "mse": float(baseline_data.get("mse", 0.0)),
                    "relative_improvement": float(baseline_data.get("relative_improvement", 0.0)),
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
        for key, rule in _RECOMMENDATION_RULES:
            if key in results:
                recommendation = rule(float(results[key]))
                if recommendation is not None:
                    recommendations.append(recommendation)

        if not recommendations:
            recommendations.append(
                "Model performance appears satisfactory across all evaluated metrics"
            )

        return recommendations

    def format_report_as_text(self, report: dict[str, Any]) -> str:
        """Format report as human-readable text."""
        lines = ["=" * 60, "PDEBench Evaluation Report", "=" * 60, ""]

        if "metadata" in report:
            lines.extend(_format_metadata_section(report["metadata"]))

        if "evaluation_summary" in report:
            lines.extend(_format_summary_section(report["evaluation_summary"]))

        if "recommendations" in report:
            lines.extend(_format_recommendations_section(report["recommendations"]))

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

    def generate_summary_statistics(self, reports: list[dict[str, Any]]) -> dict[str, Any]:
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
                values_array = np.array(values)
                summary["aggregated_metrics"][category][metric] = {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                }

        return summary

    def generate_comprehensive_report(
        self,
        results: list[BenchmarkResult],
        include_baseline_comparison: bool = True,
        include_statistical_analysis: bool = True,
    ) -> dict[str, Any]:
        """Generate full report from benchmark results.

        Args:
            results: List of BenchmarkResult objects
            include_baseline_comparison: Whether to include baseline comparisons
            include_statistical_analysis: Whether to include statistical analysis

        Returns:
            Full report dictionary
        """
        if not results:
            return {"error": "No results provided for report generation"}

        # Convert results to evaluation format
        evaluation_results = {}
        detailed_results = []

        for result in results:
            # Extract metrics from BenchmarkResult (calibrax API)
            metrics_dict = {k: m.value for k, m in result.metrics.items()}
            result_data = {
                "model_name": result.name,
                "dataset_name": result.tags.get("dataset", result.name),
                "metrics": metrics_dict,
                "execution_time": result.metadata.get("execution_time", 0.0),
            }
            detailed_results.append(result_data)

            # Aggregate metrics for overall evaluation
            for metric_name, metric_value in metrics_dict.items():
                if metric_name not in evaluation_results:
                    evaluation_results[metric_name] = []
                evaluation_results[metric_name].append(metric_value)

        # Compute aggregate metrics
        for metric_name, values in evaluation_results.items():
            values_array = np.array(values)
            evaluation_results[metric_name] = float(np.mean(values_array))

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
                values_array = np.array(values)
                analysis["dataset_performance"][dataset][metric_name] = {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "best": float(np.min(values_array))
                    if metric_name in ["mse", "mae"]
                    else float(np.max(values_array)),
                }

        return analysis
