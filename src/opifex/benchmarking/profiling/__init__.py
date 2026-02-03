"""
Performance profiling tools for Opifex framework.

This module provides JAX-native performance analysis utilities including
FLOPS counting, memory usage analysis, computational complexity metrics,
and comprehensive hardware-aware profiling capabilities.
"""

from .compilation_profiler import CompilationProfiler
from .complexity_analysis import model_complexity_analysis
from .event_coordinator import EventCoordinator
from .flops_counter import JAXFlopCounter
from .hardware_profiler import HardwareAwareProfiler
from .memory_profiler import memory_usage_analysis
from .profiling_harness import OpifexProfilingHarness
from .roofline_analyzer import RooflineAnalyzer


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
