"""Analysis Engine for Opifex Advanced Benchmarking System.

Comparative analysis and performance insights generation for scientific
computing benchmarks. Operator comparison and statistical testing delegate
to calibrax.analysis and calibrax.statistics. Domain-specific recommendation
logic and scaling analysis are retained here.
"""

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from calibrax.analysis.comparison import compare_configurations
from calibrax.core import BenchmarkResult
from calibrax.core.models import Metric, Run
from calibrax.statistics import (
    mann_whitney_u,
    welch_t_test,
)

from opifex.benchmarking._shared import infer_domain, LOWER_IS_BETTER
from opifex.benchmarking.adapters import default_metric_defs, results_to_run


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonReport:
    """Report comparing multiple operators on the same benchmark."""

    benchmark_name: str
    operators_compared: list[str]
    metric_comparisons: dict[str, dict[str, float]]
    performance_rankings: dict[str, list[str]]
    statistical_significance: dict[str, dict[str, bool]]
    winner_by_metric: dict[str, str]
    overall_winner: str
    improvement_factors: dict[str, dict[str, float]] = field(
        default_factory=dict,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalingAnalysis:
    """Analysis of scaling behavior across problem sizes."""

    operator_name: str
    problem_sizes: list[int]
    scaling_metrics: dict[str, dict[int, float]]
    scaling_coefficients: dict[str, float]
    complexity_estimates: dict[str, str]
    efficiency_scores: dict[int, float]
    optimal_problem_size: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InsightReport:
    """Performance insights for a specific benchmark run."""

    benchmark_name: str
    operator_name: str
    key_insights: list[str]
    performance_bottlenecks: list[str]
    optimization_suggestions: list[str]
    domain_specific_observations: list[str]
    confidence_level: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class RecommendationReport:
    """Recommendations for optimal operator selection."""

    problem_type: str
    domain: str
    recommended_operators: list[dict[str, Any]]
    use_case_specific_recommendations: dict[str, str]
    performance_trade_offs: dict[str, str]
    implementation_considerations: list[str]


class AnalysisEngine:
    """Comparative analysis and performance insights for scientific benchmarks.

    Provides:
    - Multi-operator performance comparisons with statistical significance
    - Scaling behavior analysis across problem sizes
    - Performance insights and bottleneck identification
    - Intelligent operator recommendations for specific use cases

    Statistical significance testing delegates to calibrax.statistics
    (welch_t_test, mann_whitney_u) for multi-run comparisons.
    """

    def __init__(self, significance_threshold: float = 0.05) -> None:
        """Initialize analysis engine.

        Args:
            significance_threshold: Threshold for statistical significance.
        """
        self.significance_threshold = significance_threshold

    def compare_operators(
        self, results_dict: dict[str, BenchmarkResult]
    ) -> ComparisonReport:
        """Compare multiple operators on the same benchmark.

        Delegates ranking and overall-winner determination to
        ``calibrax.analysis.compare_configurations()``. Domain-specific
        features (improvement_factors, statistical_significance, weighted
        scoring) are retained here because calibrax lacks equivalents.

        Args:
            results_dict: Dictionary mapping operator names to benchmark results.

        Returns:
            Comparison report with rankings and improvement factors.
        """
        if len(results_dict) < 2:
            msg = "Need at least 2 operators for comparison"
            raise ValueError(msg)

        benchmark_name = next(iter(results_dict.values())).tags.get(
            "dataset", next(iter(results_dict.values())).name
        )
        operators = list(results_dict.keys())

        # Build calibrax Runs with execution_time promoted to a Metric
        metric_defs = default_metric_defs()
        runs: dict[str, Run] = {}
        for label, result in results_dict.items():
            enriched_metrics = dict(result.metrics)
            exec_time = result.metadata.get("execution_time", 0.0)
            enriched_metrics["execution_time"] = Metric(value=exec_time)

            enriched_result = BenchmarkResult(
                name=result.name,
                domain=result.domain,
                tags=result.tags,
                metrics=enriched_metrics,
                metadata=result.metadata,
            )
            runs[label] = results_to_run([enriched_result], metric_defs=metric_defs)

        calibrax_report = compare_configurations(runs)

        # Extract rankings and winner_by_metric from calibrax report
        performance_rankings: dict[str, list[str]] = {}
        winner_by_metric: dict[str, str] = {}
        for mc in calibrax_report.metric_comparisons:
            performance_rankings[mc.metric_name] = [r.label for r in mc.rankings]
            winner_by_metric[mc.metric_name] = mc.best_label

        # Keep local improvement_factors (calibrax has them per-metric but
        # opifex ComparisonReport uses a different structure)
        metric_comparisons, all_metrics = _organize_metrics_for_comparison(results_dict)
        improvement_factors = _calculate_improvement_factors(
            metric_comparisons, all_metrics
        )

        statistical_significance = _test_statistical_significance(
            results_dict, self.significance_threshold
        )

        return ComparisonReport(
            benchmark_name=benchmark_name,
            operators_compared=operators,
            metric_comparisons=dict(metric_comparisons),
            performance_rankings=performance_rankings,
            statistical_significance=statistical_significance,
            winner_by_metric=winner_by_metric,
            overall_winner=calibrax_report.overall_winner,
            improvement_factors=improvement_factors,
        )

    def test_statistical_significance_multi_run(
        self,
        multi_run_results: dict[str, list[BenchmarkResult]],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Test statistical significance with multiple runs per operator.

        Delegates to calibrax.statistics.welch_t_test and mann_whitney_u
        for proper parametric and non-parametric testing.

        Args:
            multi_run_results: Operator names mapped to lists of results.

        Returns:
            Pairwise significance results with p-values and statistics.
        """
        significance: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        operators = list(multi_run_results.keys())

        for i, op1 in enumerate(operators):
            for op2 in operators[i + 1 :]:
                results1 = multi_run_results[op1]
                results2 = multi_run_results[op2]

                mse1 = [r.metrics["mse"].value for r in results1 if "mse" in r.metrics]
                mse2 = [r.metrics["mse"].value for r in results2 if "mse" in r.metrics]

                if len(mse1) >= 2 and len(mse2) >= 2:
                    t_stat, t_pval = welch_t_test(mse1, mse2)
                    u_stat, u_pval = mann_whitney_u(mse1, mse2)

                    is_significant = min(t_pval, u_pval) < self.significance_threshold

                    pair_result: dict[str, Any] = {
                        "significant": is_significant,
                        "t_statistic": float(t_stat),
                        "t_pvalue": float(t_pval),
                        "u_statistic": float(u_stat),
                        "u_pvalue": float(u_pval),
                    }
                    significance[op1][op2] = pair_result
                    significance[op2][op1] = pair_result

        return dict(significance)

    def create_operator_recommendations(
        self, problem_type: str, domain: str = "general"
    ) -> RecommendationReport:
        """Create operator recommendations for specific problem types.

        Args:
            problem_type: Type of problem (e.g., "pde_solving", "time_series").
            domain: Scientific domain.

        Returns:
            Operator recommendation report.
        """
        return _build_recommendation_report(problem_type, domain)

    def analyze_scaling_behavior(
        self, performance_data: dict[int, BenchmarkResult]
    ) -> ScalingAnalysis:
        """Analyze scaling behavior across different problem sizes.

        Args:
            performance_data: Dictionary mapping problem sizes to benchmark results.

        Returns:
            Scaling behavior analysis.
        """
        if len(performance_data) < 3:
            warnings.warn(
                "Need at least 3 problem sizes for reliable scaling analysis",
                stacklevel=2,
            )

        problem_sizes = sorted(performance_data.keys())
        operator_name = next(iter(performance_data.values())).name

        scaling_metrics = _extract_scaling_metrics(performance_data)

        scaling_coefficients, complexity_estimates = _calculate_scaling_coefficients(
            scaling_metrics
        )

        efficiency_scores = _calculate_efficiency_scores(scaling_metrics, problem_sizes)
        optimal_size = _find_optimal_problem_size(efficiency_scores)

        return ScalingAnalysis(
            operator_name=operator_name,
            problem_sizes=problem_sizes,
            scaling_metrics=dict(scaling_metrics),
            scaling_coefficients=scaling_coefficients,
            complexity_estimates=complexity_estimates,
            efficiency_scores=efficiency_scores,
            optimal_problem_size=optimal_size,
        )

    def generate_performance_insights(self, result: BenchmarkResult) -> InsightReport:
        """Generate performance insights for a benchmark run.

        Args:
            result: Benchmark result to analyze.

        Returns:
            Performance insights report.
        """
        insights: list[str] = []
        bottlenecks: list[str] = []
        suggestions: list[str] = []
        domain_observations: list[str] = []

        exec_time = result.metadata.get("execution_time", 0.0)
        _analyze_execution_performance(exec_time, insights, bottlenecks, suggestions)

        metrics_float = _metrics_to_float(result)
        _analyze_accuracy_metrics(metrics_float, insights, bottlenecks, suggestions)

        dataset = result.tags.get("dataset", result.name)
        _analyze_domain_specific_aspects(dataset, metrics_float, domain_observations)

        memory_usage = result.metadata.get("memory_usage")
        _analyze_memory_usage(memory_usage, bottlenecks, suggestions)

        _analyze_performance_accuracy_tradeoff(
            metrics_float, exec_time, insights, suggestions
        )

        confidence = _calculate_confidence(result)

        return InsightReport(
            benchmark_name=dataset,
            operator_name=result.name,
            key_insights=insights,
            performance_bottlenecks=bottlenecks,
            optimization_suggestions=suggestions,
            domain_specific_observations=domain_observations,
            confidence_level=confidence,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (no self needed)
# ---------------------------------------------------------------------------


def _metrics_to_float(result: BenchmarkResult) -> dict[str, float]:
    """Extract plain float values from calibrax Metric objects."""
    return {k: v.value for k, v in result.metrics.items()}


def _extract_scaling_metrics(
    performance_data: dict[int, BenchmarkResult],
) -> dict[str, dict[int, float]]:
    """Extract metrics vs problem size from performance data."""
    scaling_metrics: dict[str, dict[int, float]] = defaultdict(dict)

    for size, result in performance_data.items():
        exec_time = result.metadata.get("execution_time", 0.0)
        scaling_metrics["execution_time"][size] = exec_time

        memory = result.metadata.get("memory_usage", 0)
        scaling_metrics["memory_usage"][size] = memory

        for metric_name, metric in result.metrics.items():
            scaling_metrics[metric_name][size] = metric.value

    return dict(scaling_metrics)


def _calculate_scaling_coefficients(
    scaling_metrics: dict[str, dict[int, float]],
) -> tuple[dict[str, float], dict[str, str]]:
    """Calculate scaling coefficients and complexity estimates.

    Delegates to calibrax.analysis.scaling_fit for the log-log regression.
    """
    from calibrax.analysis import scaling_fit

    scaling_coefficients: dict[str, float] = {}
    complexity_estimates: dict[str, str] = {}

    for metric, size_values in scaling_metrics.items():
        if len(size_values) >= 3:
            try:
                sizes = [float(s) for s in size_values]
                values = [float(size_values[s]) for s in size_values]

                law = scaling_fit(sizes, values)
                scaling_coefficients[metric] = law.exponent
                complexity_estimates[metric] = _estimate_complexity(
                    metric, law.exponent
                )

            except (ValueError, RuntimeWarning):
                scaling_coefficients[metric] = float("nan")
                complexity_estimates[metric] = "undetermined"

    return scaling_coefficients, complexity_estimates


def _estimate_complexity(metric: str, scaling_exponent: float) -> str:
    """Estimate computational complexity from scaling exponent."""
    if metric == "execution_time":
        if scaling_exponent < 1.5:
            return "O(n)"
        if scaling_exponent < 2.5:
            return "O(n²)"
        if scaling_exponent < 3.5:
            return "O(n³)"
        return f"O(n^{scaling_exponent:.1f})"
    return f"scaling exponent: {scaling_exponent:.2f}"


def _calculate_efficiency_scores(
    scaling_metrics: dict[str, dict[int, float]], problem_sizes: list[int]
) -> dict[int, float]:
    """Calculate efficiency scores (performance per computational cost)."""
    efficiency_scores: dict[int, float] = {}
    for size in problem_sizes:
        exec_time_map = scaling_metrics.get("execution_time", {})
        if size in exec_time_map:
            exec_time = exec_time_map[size]
            mse_map = scaling_metrics.get("mse", {})
            performance = 1.0 / (mse_map.get(size, 1.0) + 1e-6)
            efficiency_scores[size] = performance / (exec_time + 1e-6)

    return efficiency_scores


def _find_optimal_problem_size(
    efficiency_scores: dict[int, float],
) -> int | None:
    """Find optimal problem size with highest efficiency."""
    if efficiency_scores:
        return max(efficiency_scores.items(), key=lambda x: x[1])[0]
    return None


def _analyze_execution_performance(
    exec_time: float,
    insights: list[str],
    bottlenecks: list[str],
    suggestions: list[str],
) -> None:
    """Analyze execution time performance."""
    if exec_time > 10.0:
        bottlenecks.append("High execution time indicates computational bottleneck")
        suggestions.append("Consider model compression or optimization techniques")
    elif exec_time < 0.1:
        insights.append("Excellent computational efficiency achieved")


def _analyze_accuracy_metrics(
    metrics: dict[str, float],
    insights: list[str],
    bottlenecks: list[str],
    suggestions: list[str],
) -> None:
    """Analyze accuracy-related metrics."""
    mse = metrics.get("mse", float("inf"))

    if mse < 1e-6:
        insights.append("Exceptional accuracy achieved (MSE < 1e-6)")
    elif mse > 1e-2:
        bottlenecks.append("Poor accuracy performance (MSE > 1e-2)")
        suggestions.append("Consider increasing model capacity or training longer")


def _analyze_domain_specific_aspects(
    dataset_name: str,
    metrics: dict[str, float],
    domain_observations: list[str],
) -> None:
    """Analyze domain-specific performance aspects."""
    domain = _infer_domain(dataset_name)
    mse = metrics.get("mse", float("inf"))
    relative_error = metrics.get("relative_error", float("inf"))

    if domain == "quantum_computing":
        if mse < 1e-3:
            domain_observations.append("Chemical accuracy achieved for quantum system")
        else:
            domain_observations.append(
                "Chemical accuracy not achieved - "
                "may need specialized quantum architecture"
            )
    elif domain == "fluid_dynamics":
        if relative_error < 0.05:
            domain_observations.append("Good accuracy for fluid dynamics simulation")
        else:
            domain_observations.append(
                "High relative error - may need physics-informed constraints"
            )


def _analyze_memory_usage(
    memory_usage: float | None,
    bottlenecks: list[str],
    suggestions: list[str],
) -> None:
    """Analyze memory usage and provide recommendations."""
    if memory_usage:
        memory_gb = memory_usage / (1024**3)
        if memory_gb > 16:
            bottlenecks.append("High memory usage may limit scalability")
            suggestions.append(
                "Consider memory-efficient architectures or gradient checkpointing"
            )


def _analyze_performance_accuracy_tradeoff(
    metrics: dict[str, float],
    exec_time: float,
    insights: list[str],
    suggestions: list[str],
) -> None:
    """Analyze the performance-accuracy trade-off."""
    mse = metrics.get("mse", float("inf"))

    accuracy_score = 1.0 / (mse + 1e-6)
    efficiency_score = accuracy_score / (exec_time + 1e-6)

    if efficiency_score > 100:
        insights.append("Excellent performance-accuracy trade-off")
    elif efficiency_score < 1:
        suggestions.append(
            "Consider balancing accuracy requirements with computational cost"
        )


_infer_domain = infer_domain  # Re-export for backward compat with tests


def _calculate_confidence(result: BenchmarkResult) -> float:
    """Calculate confidence level for benchmark results."""
    confidence = 0.8

    exec_time = result.metadata.get("execution_time", 0.0)
    if 0 < exec_time < 100:
        confidence += 0.1

    mse_metric = result.metrics.get("mse")
    if mse_metric is not None and 1e-8 < mse_metric.value < 1e-1:
        confidence += 0.1

    return min(confidence, 1.0)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

_LOWER_IS_BETTER = LOWER_IS_BETTER  # Re-export from shared module


def _organize_metrics_for_comparison(
    results_dict: dict[str, BenchmarkResult],
) -> tuple[dict[str, dict[str, float]], set[str]]:
    """Organize metrics from results for comparison."""
    metric_comparisons: dict[str, dict[str, float]] = defaultdict(dict)
    all_metrics: set[str] = set()

    for operator, result in results_dict.items():
        for metric_name, metric in result.metrics.items():
            metric_comparisons[metric_name][operator] = metric.value
            all_metrics.add(metric_name)

        exec_time = result.metadata.get("execution_time", 0.0)
        metric_comparisons["execution_time"][operator] = exec_time
    all_metrics.add("execution_time")

    return metric_comparisons, all_metrics


def _calculate_improvement_factors(
    metric_comparisons: dict[str, dict[str, float]],
    all_metrics: set[str],
) -> dict[str, dict[str, float]]:
    """Calculate improvement factors for each metric."""
    improvement_factors: dict[str, dict[str, float]] = {}

    for metric in all_metrics:
        if metric not in metric_comparisons:
            continue
        values = metric_comparisons[metric]
        lower_better = metric in _LOWER_IS_BETTER
        best_value = min(values.values()) if lower_better else max(values.values())

        improvement_factors[metric] = {}
        for operator, value in values.items():
            if lower_better:
                improvement_factors[metric][operator] = best_value / (value + 1e-12)
            else:
                improvement_factors[metric][operator] = value / (best_value + 1e-12)

    return improvement_factors


def _test_statistical_significance(
    results_dict: dict[str, BenchmarkResult],
    significance_threshold: float,
) -> dict[str, dict[str, bool]]:
    """Test statistical significance between operator results.

    Uses relative difference as a heuristic when only single-run results
    are available. For proper significance testing with multiple runs,
    use ``AnalysisEngine.test_statistical_significance_multi_run``.
    """
    significance: dict[str, dict[str, bool]] = defaultdict(dict)
    operators = list(results_dict.keys())

    for i, op1 in enumerate(operators):
        for j, op2 in enumerate(operators):
            if i == j:
                continue
            result1 = results_dict[op1]
            result2 = results_dict[op2]

            mse1_metric = result1.metrics.get("mse")
            mse2_metric = result2.metrics.get("mse")
            if mse1_metric is not None and mse2_metric is not None:
                mse1, mse2 = mse1_metric.value, mse2_metric.value
                relative_diff = abs(mse1 - mse2) / (min(mse1, mse2) + 1e-12)
                significance[op1][op2] = relative_diff > significance_threshold * 2

    return dict(significance)


# ---------------------------------------------------------------------------
# Recommendation helpers
# ---------------------------------------------------------------------------

_OPERATOR_PROFILES: dict[str, dict[str, dict[str, list[str]]]] = {
    "pde_solving": {
        "FNO": {
            "strengths": [
                "Spectral efficiency",
                "Translation equivariance",
                "Fast inference",
            ],
            "weaknesses": [
                "Fixed resolution",
                "Periodic boundary assumption",
            ],
            "best_for": [
                "Periodic domains",
                "Fast inference requirements",
            ],
        },
        "DeepONet": {
            "strengths": [
                "Operator learning",
                "Variable geometry",
                "Theoretical foundation",
            ],
            "weaknesses": [
                "Slower than FNO",
                "Branch network complexity",
            ],
            "best_for": [
                "Variable geometries",
                "Operator learning tasks",
            ],
        },
        "SFNO": {
            "strengths": [
                "Spherical domains",
                "Climate modeling",
                "Global problems",
            ],
            "weaknesses": [
                "Specialized to spheres",
                "Higher memory usage",
            ],
            "best_for": [
                "Climate/weather modeling",
                "Global atmospheric problems",
            ],
        },
    },
    "time_series": {
        "Neural ODE": {
            "strengths": [
                "Continuous time",
                "Irregularly sampled data",
                "Physical interpretation",
            ],
            "weaknesses": [
                "Slower training",
                "ODE solver dependency",
            ],
            "best_for": [
                "Irregular time series",
                "Physics-based modeling",
            ],
        },
    },
}

_DOMAIN_CONSIDERATIONS: dict[str, list[str]] = {
    "fluid_dynamics": [
        "Consider physics-informed losses for conservation laws",
        "FNO excellent for periodic boundary conditions",
        "DeepONet better for complex geometries",
    ],
    "quantum_computing": [
        "High precision requirements - use double precision",
        "Consider specialized quantum neural networks",
        "Validate against DFT reference methods",
    ],
    "materials_science": [
        "Consider crystal symmetries in architecture design",
        "Graph neural networks for atomic systems",
        "Validate force predictions carefully",
    ],
}


def _build_recommendation_report(
    problem_type: str, domain: str
) -> RecommendationReport:
    """Build a recommendation report for a problem type and domain."""
    recommended_operators: list[dict[str, Any]] = []

    if problem_type in _OPERATOR_PROFILES:
        for operator, profile in _OPERATOR_PROFILES[problem_type].items():
            recommended_operators.append(
                {
                    "operator": operator,
                    "recommendation_score": 0.8,
                    "strengths": profile["strengths"],
                    "weaknesses": profile["weaknesses"],
                    "best_use_cases": profile["best_for"],
                }
            )

    use_case_recommendations = {
        "high_accuracy": ("Consider ensemble methods or higher precision"),
        "fast_inference": ("FNO or optimized neural operators with JIT compilation"),
        "limited_data": ("DeepONet with physics-informed regularization"),
        "complex_geometry": "DeepONet or graph neural networks",
    }

    trade_offs = {
        "accuracy_vs_speed": (
            "FNO fastest, DeepONet most accurate, PINN most physics-informed"
        ),
        "memory_vs_accuracy": ("Compressed operators for memory constraints"),
        "training_vs_inference": ("Some operators slow to train but fast at inference"),
    }

    implementation_considerations = [
        "Use JAX for automatic differentiation and JIT compilation",
        "Consider GPU memory constraints for large problems",
        "Validate on physics-based test cases",
        "Use appropriate loss functions for the domain",
    ]

    if domain in _DOMAIN_CONSIDERATIONS:
        implementation_considerations.extend(_DOMAIN_CONSIDERATIONS[domain])

    return RecommendationReport(
        problem_type=problem_type,
        domain=domain,
        recommended_operators=recommended_operators,
        use_case_specific_recommendations=use_case_recommendations,
        performance_trade_offs=trade_offs,
        implementation_considerations=implementation_considerations,
    )
