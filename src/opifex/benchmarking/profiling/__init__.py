"""
Performance profiling tools for Opifex framework.

This module provides JAX-native performance analysis utilities including
FLOPS counting, memory usage analysis, computational complexity metrics,
and comprehensive hardware-aware profiling capabilities.
"""

from opifex.benchmarking.profiling.compilation_profiler import CompilationProfiler
from opifex.benchmarking.profiling.complexity_analysis import model_complexity_analysis
from opifex.benchmarking.profiling.event_coordinator import EventCoordinator
from opifex.benchmarking.profiling.flops_counter import JAXFlopCounter
from opifex.benchmarking.profiling.hardware_profiler import HardwareAwareProfiler
from opifex.benchmarking.profiling.memory_profiler import memory_usage_analysis
from opifex.benchmarking.profiling.profiling_harness import OpifexProfilingHarness
from opifex.benchmarking.profiling.roofline_analyzer import RooflineAnalyzer


__all__ = [
    "CompilationProfiler",
    "EventCoordinator",
    "HardwareAwareProfiler",
    "JAXFlopCounter",
    "OpifexProfilingHarness",
    "RooflineAnalyzer",
    "memory_usage_analysis",
    "model_complexity_analysis",
]
