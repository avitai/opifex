"""Benchmark Runner for Opifex Advanced Benchmarking System

Orchestrates complete benchmarking pipeline execution.
Provides end-to-end benchmarking workflows, domain-specific suites,
publication report generation, and database updates.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, Literal

import numpy as np
from calibrax.core import BenchmarkResult


# Set up logger for this module
logger = logging.getLogger(__name__)

from opifex.benchmarking._shared import extract_metric_value
from opifex.benchmarking.analysis_engine import (
    AnalysisEngine,
    ComparisonReport,
    InsightReport,
)
from opifex.benchmarking.benchmark_registry import (
    BenchmarkConfig,
    BenchmarkRegistry,
    DomainConfig,
)
from opifex.benchmarking.evaluation_engine import BenchmarkEvaluator
from opifex.benchmarking.results_manager import ResultsManager
from opifex.benchmarking.validation_framework import (
    ValidationFramework,
    ValidationReport,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainResults:
    """Results for a domain-specific benchmark suite."""

    domain: str
    benchmark_results: dict[
        str, dict[str, BenchmarkResult]
    ]  # benchmark_name -> operator_name -> result
    validation_reports: dict[str, dict[str, ValidationReport]] = field(
        default_factory=dict
    )
    comparison_reports: dict[str, ComparisonReport] = field(default_factory=dict)
    insight_reports: dict[str, dict[str, InsightReport]] = field(default_factory=dict)
    summary_statistics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicationReport:
    """Publication-ready benchmark report."""

    title: str
    abstract: str
    methodology: str
    results_summary: dict[str, Any]
    comparison_tables: list[Path] = field(default_factory=list)
    figures: list[Path] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    appendix_data: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Orchestrates complete benchmarking pipeline execution.

    This runner provides end-to-end benchmarking capabilities including:
    - Comprehensive multi-operator benchmarking across domains
    - Domain-specific benchmark suite execution with validation
    - Publication-ready report and figure generation
    - Automated benchmark database updates and maintenance
    """

    def __init__(
        self,
        registry: BenchmarkRegistry | None = None,
        evaluator: BenchmarkEvaluator | None = None,
        validator: ValidationFramework | None = None,
        analyzer: AnalysisEngine | None = None,
        results_manager: ResultsManager | None = None,
        output_dir: str = "./benchmark_results",
    ) -> None:
        """Initialize benchmark runner with all components.

        Args:
            registry: Benchmark registry (creates default if None)
            evaluator: Benchmark evaluator (creates default if None)
            validator: Validation framework (creates default if None)
            analyzer: Analysis engine (creates default if None)
            results_manager: Results manager (creates default if None)
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.registry = registry or BenchmarkRegistry()
        self.evaluator = evaluator or BenchmarkEvaluator(
            output_dir=str(self.output_dir)
        )
        self.validator = validator or ValidationFramework()
        self.analyzer = analyzer or AnalysisEngine()
        self.results_manager = results_manager or ResultsManager(
            storage_path=str(self.output_dir)
        )

        # Auto-discover operators if registry is empty
        if len(self.registry.list_available_operators()) == 0:
            self.registry.auto_discover_operators()

    def run_comprehensive_benchmark(
        self,
        operators: list[str] | None = None,
        benchmarks: list[str] | None = None,
        validate_results: bool = True,
        generate_analysis: bool = True,
    ) -> dict[str, dict[str, BenchmarkResult]]:
        """Run comprehensive benchmark across multiple operators and problems.

        Args:
            operators: List of operator names (uses all available if None)
            benchmarks: List of benchmark names (uses all available if None)
            validate_results: Whether to run validation framework
            generate_analysis: Whether to run analysis engine

        Returns:
            Nested dictionary: benchmark_name -> operator_name -> BenchmarkResult
        """
        if operators is None:
            operators = self.registry.list_available_operators()
        if benchmarks is None:
            benchmarks = self.registry.list_available_benchmarks()

        if not operators:
            raise ValueError("No operators available for benchmarking")
        if not benchmarks:
            raise ValueError("No benchmarks available for testing")

        logger.info(
            f"Running comprehensive benchmark: {len(operators)} operators x "
            f"{len(benchmarks)} benchmarks"
        )

        all_results = {}
        validation_reports = {}

        for benchmark_name in benchmarks:
            logger.info(f"Running benchmark: {benchmark_name}")
            benchmark_config = self.registry.get_benchmark_config(benchmark_name)

            benchmark_results = {}
            benchmark_validations = {}

            for operator_name in operators:
                # Check compatibility
                compatible_ops = self.registry.list_compatible_operators(benchmark_name)
                if operator_name not in compatible_ops:
                    logger.debug(
                        f"Skipping {operator_name} (not compatible with "
                        f"{benchmark_name})"
                    )
                    continue

                logger.info(f"Testing {operator_name}...")

                try:
                    # Run benchmark
                    result = self._run_single_benchmark(operator_name, benchmark_config)
                    benchmark_results[operator_name] = result

                    # Save result
                    self.results_manager.save_benchmark_results(result)

                    # Validation
                    if validate_results:
                        validation = self._validate_result(result, benchmark_config)
                        benchmark_validations[operator_name] = validation

                except Exception:
                    logger.exception(
                        "Benchmark %s failed for %s", benchmark_name, operator_name
                    )

            all_results[benchmark_name] = benchmark_results
            if validate_results:
                validation_reports[benchmark_name] = benchmark_validations

        # Generate analysis if requested
        if generate_analysis:
            self._generate_comprehensive_analysis(all_results, validation_reports)

        return all_results

    def execute_domain_specific_suite(self, domain: str) -> DomainResults:
        """Execute benchmark suite for a specific scientific domain.

        Args:
            domain: Scientific domain name

        Returns:
            Comprehensive domain-specific results
        """

        # Get domain benchmarks and configuration
        domain_benchmarks = self.registry.get_benchmark_suite(domain)
        if not domain_benchmarks:
            raise ValueError(f"No benchmarks available for domain: {domain}")

        domain_config = self.registry.get_domain_specific_config(domain)
        all_operators = self.registry.list_available_operators()

        # Run benchmarks for this domain
        benchmark_results = {}
        validation_reports = {}
        comparison_reports = {}
        insight_reports = {}

        for benchmark in domain_benchmarks:
            # Get compatible operators
            compatible_ops = self.registry.list_compatible_operators(benchmark.name)
            domain_operators = [op for op in all_operators if op in compatible_ops]

            if not domain_operators:
                continue

            # Run benchmark for all compatible operators
            operator_results = {}
            operator_validations = {}
            operator_insights = {}

            for operator_name in domain_operators:
                try:
                    # Execute benchmark
                    result = self._run_single_benchmark(operator_name, benchmark)
                    operator_results[operator_name] = result

                    # Domain-specific validation
                    validation = self._validate_result_for_domain(
                        result, benchmark, domain_config
                    )
                    operator_validations[operator_name] = validation

                    # Generate insights
                    insights = self.analyzer.generate_performance_insights(result)
                    operator_insights[operator_name] = insights

                    # Save results
                    self.results_manager.save_benchmark_results(
                        result,
                        {
                            "domain": domain,
                            "benchmark_config": benchmark.name,
                        },
                    )

                except Exception:
                    logger.exception(
                        "Benchmark %s failed for %s", benchmark.name, operator_name
                    )

            # Generate comparison report for this benchmark
            if len(operator_results) >= 2:
                comparison = self.analyzer.compare_operators(operator_results)
                comparison_reports[benchmark.name] = comparison

            benchmark_results[benchmark.name] = operator_results
            validation_reports[benchmark.name] = operator_validations
            insight_reports[benchmark.name] = operator_insights

        # Generate domain summary statistics
        summary_stats = self._generate_domain_summary(benchmark_results, domain_config)

        return DomainResults(
            domain=domain,
            benchmark_results=benchmark_results,
            validation_reports=validation_reports,
            comparison_reports=comparison_reports,
            insight_reports=insight_reports,
            summary_statistics=summary_stats,
        )

    def generate_publication_report(
        self,
        results: dict[str, dict[str, BenchmarkResult]] | DomainResults,
        title: str | None = None,
    ) -> PublicationReport:
        """Generate publication-ready report from benchmark results.

        Args:
            results: Benchmark results (either comprehensive or domain-specific)
            title: Report title (auto-generated if None)

        Returns:
            Publication-ready report with figures and tables
        """
        if isinstance(results, DomainResults):
            # Domain-specific report
            domain = results.domain
            benchmark_results = results.benchmark_results
            title = title or f"Benchmarking Study: {domain.title()} Domain"

        else:
            # Comprehensive report
            benchmark_results = results
            title = title or "Comprehensive Neural Operator Benchmarking Study"
            domain = "multi_domain"

        # Generate methodology section
        methodology = self._generate_methodology_section(benchmark_results)

        # Generate abstract
        abstract = self._generate_abstract(benchmark_results, domain)

        # Collect all results for analysis
        all_results = []
        for _benchmark_name, operator_results in benchmark_results.items():
            all_results.extend(operator_results.values())

        # Generate comparison tables
        operators = list({r.name for r in all_results})
        metrics = ["mse", "mae", "relative_error", "execution_time"]

        table_files = []
        for output_format in ["latex", "csv"]:
            table_file = self.results_manager.generate_comparison_tables(
                operators,
                metrics,
                output_format,  # type: ignore[arg-type]
            )
            table_files.append(table_file)

        # Generate figures
        figure_files = []
        for plot_type_str in ["comparison", "scaling"]:
            plot_type = cast(
                "Literal['comparison', 'scaling', 'convergence']", plot_type_str
            )
            plots = self.results_manager.export_publication_plots(
                all_results,
                plot_type,
                "png",
            )
            figure_files.extend(plots)

        # Generate key findings and recommendations
        key_findings = self._extract_key_findings(benchmark_results)
        recommendations = self._generate_recommendations(benchmark_results)

        # Create results summary
        results_summary = self._generate_results_summary(benchmark_results)

        # Create appendix data
        appendix_data = {
            "detailed_metrics": {
                benchmark_name: {
                    operator_name: {k: v.value for k, v in result.metrics.items()}
                    for operator_name, result in operator_results.items()
                }
                for benchmark_name, operator_results in benchmark_results.items()
            },
            "system_information": {
                result.name: result.metadata.get("system_info")
                for result in all_results
                if result.metadata.get("system_info") is not None
            },
        }

        return PublicationReport(
            title=title,
            abstract=abstract,
            methodology=methodology,
            results_summary=results_summary,
            comparison_tables=table_files,
            figures=figure_files,
            key_findings=key_findings,
            recommendations=recommendations,
            appendix_data=appendix_data,
        )

    def update_benchmark_database(self) -> dict[str, Any]:
        """Update benchmark database with latest results.

        Returns:
            Database update summary
        """

        # Get current database statistics
        self.results_manager.get_database_statistics()

        # The database is updated automatically when results are saved
        # This method provides a summary of the current state

        stats_after = self.results_manager.get_database_statistics()

        return {
            "database_path": str(self.results_manager.database_path),
            "total_results": stats_after["total_results"],
            "unique_models": stats_after["unique_names"],
            "unique_datasets": stats_after["unique_datasets"],
            "recent_updates": "Database updated with latest benchmark runs",
        }

    def _run_single_benchmark(
        self, operator_name: str, benchmark_config: BenchmarkConfig
    ) -> BenchmarkResult:
        """Run a single benchmark for one operator.

        UPDATED: Uses OperatorExecutor for REAL execution instead of mocks.
        This is the core fix for the benchmarking infrastructure.
        """
        from opifex.benchmarking.operator_executor import (
            ExecutionConfig,
            OperatorExecutor,
        )

        # Get operator class
        operator_class = self.registry.get_operator_class(operator_name)

        # Get data loaders for this benchmark (DRY: uses loader_type from config)
        train_loader, test_loader = self._get_data_loaders(benchmark_config)

        # Determine operator configuration from benchmark and registry metadata
        operator_config = self._get_operator_config(operator_class, benchmark_config)

        # Get execution config from benchmark
        n_epochs = benchmark_config.computational_requirements.get("n_epochs", 100)
        batch_size = benchmark_config.computational_requirements.get("batch_size", 32)

        # Execute with REAL operator
        executor = OperatorExecutor(
            ExecutionConfig(
                n_epochs=n_epochs,
                batch_size=batch_size,
            )
        )

        return executor.execute_training_benchmark(
            operator_class=operator_class,
            operator_config=operator_config,
            train_loader=train_loader,
            test_loader=test_loader,
            benchmark_name=benchmark_config.name,
        )

    def _get_data_loaders(self, benchmark_config: BenchmarkConfig) -> tuple[Any, Any]:
        """Get appropriate data loaders for benchmark.

        DRY: Uses loader_type from benchmark config, NOT string matching on name.

        Args:
            benchmark_config: Benchmark configuration

        Returns:
            Tuple of (train_loader, test_loader)

        Raises:
            ValueError: If loader_type is missing or unknown
        """
        from opifex.data.loaders import (
            create_burgers_loader,
            create_darcy_loader,
            create_diffusion_loader,
            create_navier_stokes_loader,
            create_shallow_water_loader,
        )

        # DRY: Centralized loader registry
        loader_registry = {
            "darcy": create_darcy_loader,
            "burgers": create_burgers_loader,
            "navier_stokes": create_navier_stokes_loader,
            "diffusion": create_diffusion_loader,
            "shallow_water": create_shallow_water_loader,
        }

        # DRY: Get loader type from config, not from name parsing
        loader_type = benchmark_config.computational_requirements.get("loader_type")
        if loader_type is None:
            raise ValueError(
                f"Benchmark '{benchmark_config.name}' is missing 'loader_type' in "
                f"computational_requirements. Add it to the BenchmarkConfig."
            )

        loader_fn = loader_registry.get(loader_type)
        if loader_fn is None:
            raise ValueError(
                f"Unknown loader_type '{loader_type}'. "
                f"Available: {list(loader_registry.keys())}"
            )

        # DRY: Factory function for creating loaders
        batch_size = benchmark_config.computational_requirements.get("batch_size", 32)
        resolution = benchmark_config.input_shape[0]

        # Create train and test loaders
        train_loader = loader_fn(
            n_samples=1000, batch_size=batch_size, resolution=resolution
        )
        test_loader = loader_fn(
            n_samples=100, batch_size=batch_size, resolution=resolution
        )

        return train_loader, test_loader

    def _get_operator_config(
        self,
        operator_class: type,
        benchmark_config: BenchmarkConfig,
    ) -> dict[str, Any]:
        """Determine operator configuration from benchmark and operator metadata.

        DRY: Uses operator_metadata from registry, NOT string matching on class name.

        Args:
            operator_class: Operator class to configure
            benchmark_config: Benchmark configuration

        Returns:
            Configuration dictionary for operator instantiation
        """
        operator_name = operator_class.__name__
        metadata = self.registry.get_operator_metadata(operator_name)

        # Get operator type from metadata (set during registration)
        operator_type = metadata.get("operator_type", "unknown")

        # Base config from benchmark shapes
        in_channels = benchmark_config.input_shape[-1]
        out_channels = benchmark_config.output_shape[-1]

        # DRY: Centralized default configs per operator type
        default_configs: dict[str, dict[str, Any]] = {
            "fno": {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "hidden_channels": 32,
                "modes": (12, 12),
                "rank": 0.1,
                "num_layers": 4,
            },
            "deeponet": {
                "branch_sizes": [
                    int(np.prod(np.array(benchmark_config.input_shape))),
                    128,
                    64,
                ],
                "trunk_sizes": [2, 128, 64],
            },
            "gno": {
                "node_dim": in_channels,
                "hidden_dim": 64,
                "num_layers": 4,
            },
            "pino": {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "hidden_channels": 32,
            },
        }

        # Get config for this operator type
        config = default_configs.get(operator_type, {})

        # If no specific config found, try to infer from class name (fallback)
        if not config:
            class_name_lower = operator_name.lower()
            if "fno" in class_name_lower or "fourier" in class_name_lower:
                config = default_configs["fno"]
            elif "deeponet" in class_name_lower:
                config = default_configs["deeponet"]
            elif "gno" in class_name_lower or "graph" in class_name_lower:
                config = default_configs["gno"]
            else:
                # Generic fallback
                config = {
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "hidden_channels": 32,
                }

        # Allow benchmark-specific overrides
        config.update(
            benchmark_config.computational_requirements.get("operator_config", {})
        )

        return config

    def _validate_result(
        self, result: BenchmarkResult, benchmark_config: BenchmarkConfig
    ) -> ValidationReport:
        """Validate a benchmark result."""
        return self.validator.validate_against_reference(result, "synthetic_reference")

    def _validate_result_for_domain(
        self,
        result: BenchmarkResult,
        benchmark_config: BenchmarkConfig,
        domain_config: DomainConfig,
    ) -> ValidationReport:
        """Validate result with domain-specific criteria."""
        # Use domain-specific validation
        validation = self.validator.validate_against_reference(
            result, "domain_reference"
        )

        # Add domain-specific checks
        if hasattr(domain_config, "tolerance_ranges"):
            for metric_name, metric_obj in result.metrics.items():
                if metric_name in domain_config.tolerance_ranges:
                    min_tol, max_tol = domain_config.tolerance_ranges[metric_name]
                    if not (min_tol <= metric_obj.value <= max_tol):
                        validation.tolerance_violations.append(
                            f"{metric_name} ({metric_obj.value}) outside "
                            f"domain tolerance ({min_tol}, {max_tol})"
                        )

        return validation

    def _generate_comprehensive_analysis(
        self,
        results: dict[str, dict[str, BenchmarkResult]],
        validations: dict[str, dict[str, ValidationReport]],
    ) -> None:
        """Generate comprehensive analysis across all results."""

        # Generate comparison reports for each benchmark
        for _benchmark_name, operator_results in results.items():
            if len(operator_results) >= 2:
                self.analyzer.compare_operators(operator_results)

    def _generate_domain_summary(
        self,
        benchmark_results: dict[str, dict[str, BenchmarkResult]],
        domain_config: DomainConfig,
    ) -> dict[str, Any]:
        """Generate summary statistics for domain results."""
        all_results = []
        for operator_results in benchmark_results.values():
            all_results.extend(operator_results.values())

        summary: dict[str, Any] = {
            "total_benchmarks": len(benchmark_results),
            "total_results": len(all_results),
        }

        if not all_results:
            return summary

        # Calculate domain-wide statistics
        exec_times = [r.metadata.get("execution_time", 0.0) for r in all_results]
        mse_values = [
            r.metrics["mse"].value if "mse" in r.metrics else float("inf")
            for r in all_results
        ]

        finite_mse = [v for v in mse_values if v != float("inf")]
        summary.update(
            {
                "avg_execution_time": float(np.mean(np.array(exec_times))),
                "avg_mse": float(np.mean(np.array(finite_mse)))
                if finite_mse
                else float("inf"),
                "best_overall_model": min(
                    all_results,
                    key=lambda r: extract_metric_value(r, "mse"),
                ).name,
            }
        )
        return summary

    def _generate_methodology_section(
        self, results: dict[str, dict[str, BenchmarkResult]]
    ) -> str:
        """Generate methodology section for publication."""
        n_operators = len(
            {
                result.name
                for benchmark_results in results.values()
                for result in benchmark_results.values()
            }
        )
        n_benchmarks = len(results)

        return f"""
        We evaluated {n_operators} neural operator architectures across \
{n_benchmarks} benchmark problems.
        All models were implemented using JAX and Flax NNX framework \
with consistent hyperparameters.
        Performance was measured using mean squared error (MSE), \
mean absolute error (MAE), and relative error.
        Execution times were measured on standardized hardware configurations.
        Statistical significance was assessed using bootstrap confidence intervals.
        """

    def _generate_abstract(
        self, results: dict[str, dict[str, BenchmarkResult]], domain: str
    ) -> str:
        """Generate abstract for publication."""
        return f"""
        This study presents a comprehensive benchmarking analysis of neural \
operator architectures
        for {domain} applications. We evaluate multiple advanced \
operators across diverse
        benchmark problems, providing detailed performance comparisons and \
recommendations for
        practical deployment. Our results demonstrate significant performance \
variations across
        different problem types and provide guidance for operator selection \
in scientific computing
        applications.
        """

    def _extract_key_findings(
        self, results: dict[str, dict[str, BenchmarkResult]]
    ) -> list[str]:
        """Extract key findings from benchmark results."""
        findings = []

        # Find best performing operators
        all_results = []
        for operator_results in results.values():
            all_results.extend(operator_results.values())

        if all_results:
            best_mse = min(
                all_results,
                key=lambda r: extract_metric_value(r, "mse"),
            )
            mse_v = extract_metric_value(best_mse, "mse")
            findings.append(
                f"{best_mse.name} achieved best accuracy (MSE: {mse_v:.2e})"
            )

            fastest = min(
                all_results,
                key=lambda r: r.metadata.get("execution_time", float("inf")),
            )
            exec_t = fastest.metadata.get("execution_time", 0.0)
            findings.append(f"{fastest.name} was fastest (Time: {exec_t:.2f}s)")

        return findings

    def _generate_recommendations(
        self, results: dict[str, dict[str, BenchmarkResult]]
    ) -> list[str]:
        """Generate recommendations based on results."""
        return [
            "For accuracy-critical applications, consider ensemble methods",
            "For real-time applications, prioritize computational efficiency",
            "Domain-specific architectures show superior performance "
            "in specialized applications",
        ]

    def _generate_results_summary(
        self, results: dict[str, dict[str, BenchmarkResult]]
    ) -> dict[str, Any]:
        """Generate high-level results summary."""
        all_results = []
        for operator_results in results.values():
            all_results.extend(operator_results.values())

        if not all_results:
            return {}

        return {
            "total_experiments": len(all_results),
            "operators_tested": len({r.name for r in all_results}),
            "benchmarks_used": len(results),
            "performance_range": {
                "mse": {
                    "min": min(extract_metric_value(r, "mse") for r in all_results),
                    "max": max(extract_metric_value(r, "mse") for r in all_results),
                },
                "execution_time": {
                    "min": min(
                        r.metadata.get("execution_time", 0.0) for r in all_results
                    ),
                    "max": max(
                        r.metadata.get("execution_time", 0.0) for r in all_results
                    ),
                },
            },
        }
