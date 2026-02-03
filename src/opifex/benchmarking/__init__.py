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

from opifex.benchmarking.analysis_engine import AnalysisEngine
from opifex.benchmarking.baseline_repository import BaselineRepository
from opifex.benchmarking.benchmark_registry import BenchmarkRegistry
from opifex.benchmarking.benchmark_runner import BenchmarkRunner
from opifex.benchmarking.evaluation_engine import (
    BenchmarkEvaluator,
    BenchmarkResult,
    EvaluationMetrics,
    StatisticalAnalyzer,
)
from opifex.benchmarking.results_manager import ResultsManager
from opifex.benchmarking.validation_framework import ValidationFramework


__all__ = [
    "AnalysisEngine",
    "BaselineRepository",
    "BenchmarkEvaluator",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "BenchmarkRunner",
    "EvaluationMetrics",
    "ResultsManager",
    "StatisticalAnalyzer",
    "ValidationFramework",
]
