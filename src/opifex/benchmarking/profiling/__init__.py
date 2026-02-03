"""Performance profiling tools for Opifex framework.

Low-level profiling utilities (FLOPS counting, memory analysis, roofline,
compilation, hardware detection) are now provided by calibrax.profiling.
This package retains the Opifex-specific orchestration layer:
- EventCoordinator: session management and profiler lifecycle
- OpifexProfilingHarness: neural operator profiling integration
"""

from calibrax.profiling import (
    analyze_complexity,
    CompilationProfiler,
    detect_hardware_specs,
    FlopsCounter,
    ResourceMonitor,
    RooflineAnalyzer,
)

from opifex.benchmarking.profiling.event_coordinator import EventCoordinator
from opifex.benchmarking.profiling.profiling_harness import (
    OpifexProfilingHarness,
    OptimizationReport,
)


__all__ = [
    "CompilationProfiler",
    "EventCoordinator",
    "FlopsCounter",
    "OpifexProfilingHarness",
    "OptimizationReport",
    "ResourceMonitor",
    "RooflineAnalyzer",
    "analyze_complexity",
    "detect_hardware_specs",
]
