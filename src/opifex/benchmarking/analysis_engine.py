"""Analysis Engine for Opifex Advanced Benchmarking System

Comparative analysis and performance insights generation.
Provides operator comparison, scaling analysis, performance insights,
and recommendation generation for optimal model selection.
"""

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from opifex.benchmarking.evaluation_engine import BenchmarkResult


@dataclass
class ComparisonReport:
    """Report comparing multiple operators on the same benchmark."""

    benchmark_name: str
    operators_compared: list[str]
    metric_comparisons: dict[str, dict[str, float]]
    performance_rankings: dict[str, list[str]]
    statistical_significance: dict[str, dict[str, bool]]
    winner_by_metric: dict[str, str]
    overall_winner: str
    improvement_factors: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ScalingAnalysis:
    """Analysis of scaling behavior across problem sizes."""

    operator_name: str
    problem_sizes: list[int]
    scaling_metrics: dict[str, dict[str, float]]
    scaling_coefficients: dict[str, float]
    complexity_estimates: dict[str, str]
    efficiency_scores: dict[str, float]
    optimal_problem_size: int | None = None


@dataclass
class InsightReport:
    """Performance insights for a specific benchmark run."""

    benchmark_name: str
    operator_name: str
    key_insights: list[str]
    performance_bottlenecks: list[str]
    optimization_suggestions: list[str]
    domain_specific_observations: list[str]
    confidence_level: float = 0.0


@dataclass
class RecommendationReport:
    """Recommendations for optimal operator selection."""

    problem_type: str
    domain: str
    recommended_operators: list[dict[str, Any]]
    use_case_specific_recommendations: dict[str, str]
    performance_trade_offs: dict[str, str]
    implementation_considerations: list[str]


class AnalysisEngine:
    """Comparative analysis and performance insights generation.

    This engine provides comprehensive analysis capabilities including:
    - Multi-operator performance comparisons with statistical significance
    - Scaling behavior analysis across problem sizes
    - Performance insights and bottleneck identification
    - Intelligent operator recommendations for specific use cases
    """

    def __init__(self, significance_threshold: float = 0.05):
        """Initialize analysis engine.

        Args:
            significance_threshold: Threshold for statistical significance testing
        """
        self.significance_threshold = significance_threshold

        # Performance weights for different metrics (domain-specific)
        self.metric_weights = {
            "fluid_dynamics": {
                "mse": 0.3,
                "mae": 0.2,
                "relative_error": 0.3,
                "execution_time": 0.2,
            },
            "quantum_computing": {
                "mse": 0.4,
                "chemical_accuracy": 0.4,
                "execution_time": 0.2,
            },
            "materials_science": {
                "mse": 0.25,
                "mae": 0.25,
                "force_accuracy": 0.3,
                "execution_time": 0.2,
            },
            "default": {
                "mse": 0.3,
                "mae": 0.2,
                "relative_error": 0.3,
                "execution_time": 0.2,
            },
        }

    def compare_operators(
        self, results_dict: dict[str, BenchmarkResult]
    ) -> ComparisonReport:
        """Compare multiple operators on the same benchmark.

        Args:
            results_dict: Dictionary mapping operator names to benchmark results

        Returns:
            Comprehensive comparison report
        """
        if len(results_dict) < 2:
            raise ValueError("Need at least 2 operators for comparison")

        # Extract benchmark name (assume all results are for same benchmark)
        benchmark_name = next(iter(results_dict.values())).dataset_name
        operators = list(results_dict.keys())

        # Organize metrics for comparison
        metric_comparisons, all_metrics = self._organize_metrics_for_comparison(
            results_dict
        )

        # Rank operators by each metric
        performance_rankings, winner_by_metric = self._calculate_performance_rankings(
            metric_comparisons, all_metrics
        )

        # Calculate improvement factors
        improvement_factors = self._calculate_improvement_factors(
            metric_comparisons, all_metrics
        )

        # Determine overall winner using weighted scoring
        overall_winner = self._determine_overall_winner(
            operators, metric_comparisons, benchmark_name
        )

        # Statistical significance testing (simplified)
        statistical_significance = self._test_statistical_significance(results_dict)

        return ComparisonReport(
            benchmark_name=benchmark_name,
            operators_compared=operators,
            metric_comparisons=dict(metric_comparisons),
            performance_rankings=performance_rankings,
            statistical_significance=statistical_significance,
            winner_by_metric=winner_by_metric,
            overall_winner=overall_winner,
            improvement_factors=improvement_factors,
        )

    def _organize_metrics_for_comparison(
        self, results_dict: dict[str, BenchmarkResult]
    ) -> tuple[dict, set]:
        """Organize metrics from results for comparison."""
        metric_comparisons = defaultdict(dict)
        all_metrics = set()

        for operator, result in results_dict.items():
            for metric, value in result.metrics.items():
                metric_comparisons[metric][operator] = value
                all_metrics.add(metric)

        # Add execution time as a metric
        for operator, result in results_dict.items():
            metric_comparisons["execution_time"][operator] = result.execution_time
        all_metrics.add("execution_time")

        return metric_comparisons, all_metrics

    def _calculate_performance_rankings(
        self, metric_comparisons: dict, all_metrics: set
    ) -> tuple[dict, dict]:
        """Calculate performance rankings for each metric."""
        performance_rankings = {}
        winner_by_metric = {}

        for metric in all_metrics:
            if metric in metric_comparisons:
                metric_values = metric_comparisons[metric]

                # For error metrics (lower is better)
                if metric in ["mse", "mae", "relative_error", "execution_time"]:
                    sorted_operators = sorted(metric_values.items(), key=lambda x: x[1])
                else:
                    # For accuracy metrics (higher is better)
                    sorted_operators = sorted(
                        metric_values.items(), key=lambda x: x[1], reverse=True
                    )

                performance_rankings[metric] = [op for op, _ in sorted_operators]
                winner_by_metric[metric] = sorted_operators[0][0]

        return performance_rankings, winner_by_metric

    def _calculate_improvement_factors(
        self, metric_comparisons: dict, all_metrics: set
    ) -> dict:
        """Calculate improvement factors for each metric."""
        improvement_factors = {}

        for metric in all_metrics:
            if metric in metric_comparisons:
                values = metric_comparisons[metric]
                best_value = (
                    min(values.values())
                    if metric in ["mse", "mae", "relative_error", "execution_time"]
                    else max(values.values())
                )

                improvement_factors[metric] = {}
                for operator, value in values.items():
                    if metric in ["mse", "mae", "relative_error", "execution_time"]:
                        # For error metrics: improvement = baseline/value
                        improvement_factors[metric][operator] = best_value / (
                            value + 1e-12
                        )
                    else:
                        # For accuracy metrics: improvement = value/baseline
                        improvement_factors[metric][operator] = value / (
                            best_value + 1e-12
                        )

        return improvement_factors

    def _determine_overall_winner(
        self, operators: list, metric_comparisons: dict, benchmark_name: str
    ) -> str:
        """Determine overall winner using weighted scoring."""
        domain = self._infer_domain(benchmark_name)
        weights = self.metric_weights.get(domain, self.metric_weights["default"])

        overall_scores = defaultdict(float)
        for operator in operators:
            for metric, weight in weights.items():
                if (
                    metric in metric_comparisons
                    and operator in metric_comparisons[metric]
                ):
                    score = self._normalize_metric_score(
                        metric, operator, metric_comparisons[metric]
                    )
                    overall_scores[operator] += weight * score

        return max(overall_scores.items(), key=lambda x: x[1])[0]

    def _normalize_metric_score(
        self, metric: str, operator: str, metric_values: dict
    ) -> float:
        """Normalize a metric score to 0-1 scale where 1 is best."""
        values = list(metric_values.values())
        operator_value = metric_values[operator]

        if metric in ["mse", "mae", "relative_error", "execution_time"]:
            # Lower is better
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                return 1 - (operator_value - min_val) / (max_val - min_val)
            return 1.0
        # Higher is better
        min_val, max_val = min(values), max(values)
        if max_val > min_val:
            return (operator_value - min_val) / (max_val - min_val)
        return 1.0

    def analyze_scaling_behavior(
        self, performance_data: dict[int, BenchmarkResult]
    ) -> ScalingAnalysis:
        """Analyze scaling behavior across different problem sizes.

        Args:
            performance_data: Dictionary mapping problem sizes to benchmark results

        Returns:
            Scaling behavior analysis
        """
        if len(performance_data) < 3:
            warnings.warn(
                "Need at least 3 problem sizes for reliable scaling analysis",
                stacklevel=2,
            )

        problem_sizes = sorted(performance_data.keys())
        operator_name = next(iter(performance_data.values())).model_name

        # Extract metrics vs problem size
        scaling_metrics = self._extract_scaling_metrics(performance_data)

        # Fit scaling coefficients and complexity estimates
        scaling_coefficients, complexity_estimates = (
            self._calculate_scaling_coefficients(scaling_metrics)
        )

        # Calculate efficiency scores and find optimal size
        efficiency_scores = self._calculate_efficiency_scores(
            scaling_metrics, problem_sizes
        )
        optimal_size = self._find_optimal_problem_size(efficiency_scores)

        return ScalingAnalysis(
            operator_name=operator_name,
            problem_sizes=problem_sizes,
            scaling_metrics=dict(scaling_metrics),
            scaling_coefficients=scaling_coefficients,
            complexity_estimates=complexity_estimates,
            efficiency_scores=efficiency_scores,
            optimal_problem_size=optimal_size,
        )

    def _extract_scaling_metrics(
        self, performance_data: dict[int, BenchmarkResult]
    ) -> dict:
        """Extract metrics vs problem size from performance data."""
        scaling_metrics = defaultdict(dict)

        for size, result in performance_data.items():
            scaling_metrics["execution_time"][size] = result.execution_time
            scaling_metrics["memory_usage"][size] = result.memory_usage or 0

            for metric, value in result.metrics.items():
                scaling_metrics[metric][size] = value

        return scaling_metrics

    def _calculate_scaling_coefficients(
        self, scaling_metrics: dict
    ) -> tuple[dict, dict]:
        """Calculate scaling coefficients and complexity estimates."""
        scaling_coefficients = {}
        complexity_estimates = {}

        for metric, size_values in scaling_metrics.items():
            if len(size_values) >= 3:
                try:
                    sizes = np.array(list(size_values.keys()))
                    values = np.array(list(size_values.values()))

                    # Log-linear fit: log(value) = log(a) + b * log(size)
                    log_sizes = np.log(sizes)
                    log_values = np.log(values + 1e-12)  # Avoid log(0)

                    # Use np.polyfit for robust linear regression
                    b, _ = np.polyfit(log_sizes, log_values, 1)
                    scaling_coefficients[metric] = float(b)
                    complexity_estimates[metric] = self._estimate_complexity(metric, b)

                except (ValueError, RuntimeWarning):
                    scaling_coefficients[metric] = float("nan")
                    complexity_estimates[metric] = "undetermined"

        return scaling_coefficients, complexity_estimates

    def _estimate_complexity(self, metric: str, scaling_exponent: float) -> str:
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
        self, scaling_metrics: dict, problem_sizes: list[int]
    ) -> dict:
        """Calculate efficiency scores (performance per computational cost)."""
        efficiency_scores = {}
        for size in problem_sizes:
            if size in scaling_metrics["execution_time"]:
                exec_time = scaling_metrics["execution_time"][size]
                # Use inverse of MSE as performance measure
                performance = 1.0 / (
                    scaling_metrics.get("mse", {}).get(size, 1.0) + 1e-6
                )
                efficiency_scores[size] = performance / (exec_time + 1e-6)

        return efficiency_scores

    def _find_optimal_problem_size(self, efficiency_scores: dict) -> int | None:
        """Find optimal problem size with highest efficiency."""
        if efficiency_scores:
            return max(efficiency_scores.items(), key=lambda x: x[1])[0]
        return None

    def generate_performance_insights(
        self, benchmark_results: BenchmarkResult
    ) -> InsightReport:
        """Generate performance insights for a benchmark run.

        Args:
            benchmark_results: Benchmark results to analyze

        Returns:
            Performance insights report
        """
        insights = []
        bottlenecks = []
        suggestions = []
        domain_observations = []

        # Analyze different aspects of performance
        self._analyze_execution_performance(
            benchmark_results.execution_time, insights, bottlenecks, suggestions
        )

        self._analyze_accuracy_metrics(
            benchmark_results.metrics, insights, bottlenecks, suggestions
        )

        self._analyze_domain_specific_aspects(benchmark_results, domain_observations)

        self._analyze_memory_usage(
            benchmark_results.memory_usage, bottlenecks, suggestions
        )

        self._analyze_performance_accuracy_tradeoff(
            benchmark_results, insights, suggestions
        )

        # Confidence level based on metric consistency
        confidence = self._calculate_confidence(benchmark_results)

        return InsightReport(
            benchmark_name=benchmark_results.dataset_name,
            operator_name=benchmark_results.model_name,
            key_insights=insights,
            performance_bottlenecks=bottlenecks,
            optimization_suggestions=suggestions,
            domain_specific_observations=domain_observations,
            confidence_level=confidence,
        )

    def _analyze_execution_performance(
        self, exec_time: float, insights: list, bottlenecks: list, suggestions: list
    ) -> None:
        """Analyze execution time performance."""
        if exec_time > 10.0:  # seconds
            bottlenecks.append("High execution time indicates computational bottleneck")
            suggestions.append("Consider model compression or optimization techniques")
        elif exec_time < 0.1:
            insights.append("Excellent computational efficiency achieved")

    def _analyze_accuracy_metrics(
        self, metrics: dict, insights: list, bottlenecks: list, suggestions: list
    ) -> None:
        """Analyze accuracy-related metrics."""
        mse = metrics.get("mse", float("inf"))

        if mse < 1e-6:
            insights.append("Exceptional accuracy achieved (MSE < 1e-6)")
        elif mse > 1e-2:
            bottlenecks.append("Poor accuracy performance (MSE > 1e-2)")
            suggestions.append("Consider increasing model capacity or training longer")

    def _analyze_domain_specific_aspects(
        self, benchmark_results: BenchmarkResult, domain_observations: list
    ) -> None:
        """Analyze domain-specific performance aspects."""
        domain = self._infer_domain(benchmark_results.dataset_name)
        mse = benchmark_results.metrics.get("mse", float("inf"))
        relative_error = benchmark_results.metrics.get("relative_error", float("inf"))

        if domain == "quantum_computing":
            self._analyze_quantum_performance(mse, domain_observations)
        elif domain == "fluid_dynamics":
            self._analyze_fluid_dynamics_performance(
                relative_error, domain_observations
            )

    def _analyze_quantum_performance(
        self, mse: float, domain_observations: list
    ) -> None:
        """Analyze quantum computing specific performance."""
        if mse < 1e-3:  # Chemical accuracy threshold
            domain_observations.append("Chemical accuracy achieved for quantum system")
        else:
            domain_observations.append(
                "Chemical accuracy not achieved - may need specialized "
                "quantum architecture"
            )

    def _analyze_fluid_dynamics_performance(
        self, relative_error: float, domain_observations: list
    ) -> None:
        """Analyze fluid dynamics specific performance."""
        if relative_error < 0.05:
            domain_observations.append("Good accuracy for fluid dynamics simulation")
        else:
            domain_observations.append(
                "High relative error - may need physics-informed constraints"
            )

    def _analyze_memory_usage(
        self, memory_usage: float | None, bottlenecks: list, suggestions: list
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
        self, benchmark_results: BenchmarkResult, insights: list, suggestions: list
    ) -> None:
        """Analyze the performance-accuracy trade-off."""
        mse = benchmark_results.metrics.get("mse", float("inf"))
        exec_time = benchmark_results.execution_time

        accuracy_score = 1.0 / (mse + 1e-6)  # Higher is better
        efficiency_score = accuracy_score / (exec_time + 1e-6)

        if efficiency_score > 100:
            insights.append("Excellent performance-accuracy trade-off")
        elif efficiency_score < 1:
            suggestions.append(
                "Consider balancing accuracy requirements with computational cost"
            )

    def create_operator_recommendations(
        self, problem_type: str, domain: str = "general"
    ) -> RecommendationReport:
        """Create operator recommendations for specific problem types.

        Args:
            problem_type: Type of problem (e.g., "pde_solving", "regression",
                         "classification")
            domain: Scientific domain

        Returns:
            Operator recommendation report
        """
        # Define operator characteristics for different problem types
        operator_profiles = {
            "pde_solving": {
                "FNO": {
                    "strengths": [
                        "Spectral efficiency",
                        "Translation equivariance",
                        "Fast inference",
                    ],
                    "weaknesses": ["Fixed resolution", "Periodic boundary assumption"],
                    "best_for": ["Periodic domains", "Fast inference requirements"],
                },
                "DeepONet": {
                    "strengths": [
                        "Operator learning",
                        "Variable geometry",
                        "Theoretical foundation",
                    ],
                    "weaknesses": ["Slower than FNO", "Branch network complexity"],
                    "best_for": ["Variable geometries", "Operator learning tasks"],
                },
                "SFNO": {
                    "strengths": [
                        "Spherical domains",
                        "Climate modeling",
                        "Global problems",
                    ],
                    "weaknesses": ["Specialized to spheres", "Higher memory usage"],
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
                    "weaknesses": ["Slower training", "ODE solver dependency"],
                    "best_for": ["Irregular time series", "Physics-based modeling"],
                },
            },
        }

        # Domain-specific considerations
        domain_considerations = {
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

        # Generate recommendations
        recommended_operators = []

        if problem_type in operator_profiles:
            for operator, profile in operator_profiles[problem_type].items():
                recommended_operators.append(
                    {
                        "operator": operator,
                        "recommendation_score": 0.8,  # Placeholder scoring
                        "strengths": profile["strengths"],
                        "weaknesses": profile["weaknesses"],
                        "best_use_cases": profile["best_for"],
                    }
                )

        # Use case specific recommendations
        use_case_recommendations = {
            "high_accuracy": "Consider ensemble methods or higher precision",
            "fast_inference": "FNO or optimized neural operators with JIT compilation",
            "limited_data": "DeepONet with physics-informed regularization",
            "complex_geometry": "DeepONet or graph neural networks",
        }

        # Performance trade-offs
        trade_offs = {
            "accuracy_vs_speed": (
                "FNO fastest, DeepONet most accurate, PINN most physics-informed"
            ),
            "memory_vs_accuracy": "Compressed operators for memory constraints",
            "training_vs_inference": (
                "Some operators slow to train but fast at inference"
            ),
        }

        implementation_considerations = [
            "Use JAX for automatic differentiation and JIT compilation",
            "Consider GPU memory constraints for large problems",
            "Validate on physics-based test cases",
            "Use appropriate loss functions for the domain",
        ]

        if domain in domain_considerations:
            implementation_considerations.extend(domain_considerations[domain])

        return RecommendationReport(
            problem_type=problem_type,
            domain=domain,
            recommended_operators=recommended_operators,
            use_case_specific_recommendations=use_case_recommendations,
            performance_trade_offs=trade_offs,
            implementation_considerations=implementation_considerations,
        )

    def _infer_domain(self, dataset_name: str) -> str:
        """Infer scientific domain from dataset name."""
        dataset_lower = dataset_name.lower()

        if any(term in dataset_lower for term in ["quantum", "dft", "molecular"]):
            return "quantum_computing"
        if any(
            term in dataset_lower for term in ["fluid", "burgers", "navier", "darcy"]
        ):
            return "fluid_dynamics"
        if any(term in dataset_lower for term in ["material", "crystal", "solid"]):
            return "materials_science"
        if any(term in dataset_lower for term in ["climate", "weather", "atmospheric"]):
            return "climate_modeling"
        return "general"

    def _test_statistical_significance(
        self, results_dict: dict[str, BenchmarkResult]
    ) -> dict[str, dict[str, bool]]:
        """Test statistical significance between operator results.

        Note: Full statistical testing (t-test, Mann-Whitney U) requires multiple
        runs per operator. This implementation uses a relative difference threshold
        as a heuristic when only single-run results are available. For proper
        significance testing, use `test_statistical_significance_multi_run`
        with multiple benchmark runs.
        """
        significance = defaultdict(dict)

        operators = list(results_dict.keys())
        for i, op1 in enumerate(operators):
            for j, op2 in enumerate(operators):
                if i != j:
                    result1 = results_dict[op1]
                    result2 = results_dict[op2]

                    if "mse" in result1.metrics and "mse" in result2.metrics:
                        mse1, mse2 = result1.metrics["mse"], result2.metrics["mse"]
                        # Use relative difference as heuristic for single-run data
                        relative_diff = abs(mse1 - mse2) / (min(mse1, mse2) + 1e-12)
                        significance[op1][op2] = (
                            relative_diff > self.significance_threshold * 2
                        )

        return dict(significance)

    def test_statistical_significance_multi_run(
        self, multi_run_results: dict[str, list[BenchmarkResult]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Test statistical significance between operators with multiple runs.

        Uses scipy.stats for proper statistical testing when multiple benchmark
        runs are available for each operator.

        Args:
            multi_run_results: Dictionary mapping operator names to lists of
                benchmark results from multiple runs.

        Returns:
            Dictionary with pairwise significance results including p-values
            and test statistics.
        """
        from scipy.stats import mannwhitneyu, ttest_ind

        significance = defaultdict(dict)
        operators = list(multi_run_results.keys())

        for i, op1 in enumerate(operators):
            for j, op2 in enumerate(operators):
                if i >= j:  # Only compute upper triangle
                    continue

                results1 = multi_run_results[op1]
                results2 = multi_run_results[op2]

                # Extract MSE values from each run
                mse1 = [r.metrics.get("mse", float("inf")) for r in results1]
                mse2 = [r.metrics.get("mse", float("inf")) for r in results2]

                if len(mse1) >= 2 and len(mse2) >= 2:
                    # Perform both parametric and non-parametric tests
                    t_result = ttest_ind(mse1, mse2)
                    u_result = mannwhitneyu(mse1, mse2, alternative="two-sided")

                    t_stat_val = float(t_result.statistic)  # pyright: ignore[reportAttributeAccessIssue]
                    t_pvalue_val = float(t_result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
                    u_stat_val = float(u_result.statistic)
                    u_pvalue_val = float(u_result.pvalue)

                    is_significant = (
                        min(t_pvalue_val, u_pvalue_val) < self.significance_threshold
                    )

                    significance[op1][op2] = {
                        "significant": is_significant,
                        "t_statistic": t_stat_val,
                        "t_pvalue": t_pvalue_val,
                        "u_statistic": u_stat_val,
                        "u_pvalue": u_pvalue_val,
                    }
                    # Mirror for symmetric access
                    significance[op2][op1] = significance[op1][op2]

        return dict(significance)

    def _calculate_confidence(self, results: BenchmarkResult) -> float:
        """Calculate confidence level for benchmark results."""
        # Simple confidence calculation based on metric consistency
        confidence = 0.8  # Base confidence

        # Adjust based on execution time consistency (proxy for stability)
        if results.execution_time > 0 and results.execution_time < 100:
            confidence += 0.1

        # Adjust based on metric reasonableness
        if "mse" in results.metrics and 1e-8 < results.metrics["mse"] < 1e-1:
            confidence += 0.1

        return min(confidence, 1.0)
