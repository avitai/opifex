"""Opifex Advanced Benchmarking System

Comprehensive benchmarking infrastructure for neural operators and scientific ML models.
Provides performance evaluation, scientific validation, comparative analysis, and
publication-ready results.

Main Components:
- BenchmarkRegistry: Manages benchmarks and neural operators
- BenchmarkEvaluator: Performance measurement and profiling
- ValidationFramework: Scientific accuracy validation
- AnalysisEngine: Comparative analysis and insights
- ResultsManager: Publication-ready export capabilities
- BenchmarkRunner: Orchestrates complete benchmarking pipeline
- BaselineRepository: Manages baseline performance metrics
"""

from calibrax.core import BenchmarkResult
from calibrax.statistics import StatisticalAnalyzer

from opifex.benchmarking.adapters import default_metric_defs, results_to_run
from opifex.benchmarking.analysis_engine import AnalysisEngine
from opifex.benchmarking.baseline_repository import BaselineRepository
from opifex.benchmarking.benchmark_registry import BenchmarkRegistry
from opifex.benchmarking.benchmark_runner import BenchmarkRunner
from opifex.benchmarking.cli import main as run_benchmark_cli, parse_args, run_cli
from opifex.benchmarking.evaluation_engine import BenchmarkEvaluator
from opifex.benchmarking.operator_executor import ExecutionConfig, OperatorExecutor
from opifex.benchmarking.pdebench_configs import (
    PDEBENCH_BENCHMARKS,
    REALPDEBENCH_BENCHMARKS,
    register_all_benchmarks,
    register_pdebench_benchmarks,
    register_realpdebench_benchmarks,
)
from opifex.benchmarking.results_manager import ResultsManager
from opifex.benchmarking.validation_framework import ValidationFramework


__all__ = [
    "PDEBENCH_BENCHMARKS",
    "REALPDEBENCH_BENCHMARKS",
    "AnalysisEngine",
    "BaselineRepository",
    "BenchmarkEvaluator",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ExecutionConfig",
    "OperatorExecutor",
    "ResultsManager",
    "StatisticalAnalyzer",
    "ValidationFramework",
    "default_metric_defs",
    "parse_args",
    "register_all_benchmarks",
    "register_pdebench_benchmarks",
    "register_realpdebench_benchmarks",
    "results_to_run",
    "run_benchmark_cli",
    "run_cli",
]
