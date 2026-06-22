"""Opifex Optimization Module.

Production optimisation (JIT kernel fusion, GPU memory planning, and physics/numerical
validation), the learn-to-optimize subsystem (:mod:`opifex.optimization.l2o`), the
meta-optimization framework (:mod:`opifex.optimization.meta_optimization`), and differentiable
control (:mod:`opifex.optimization.control`).

Serving telemetry, autoscaling, edge distribution, and deployment orchestration are out of
scope — those concerns belong to external infrastructure (KServe / Ray Serve / k8s HPA /
Prometheus), not to this numerical-optimisation layer.
"""

from opifex.optimization.production import (
    AdaptiveJAXOptimizer,
    HybridPerformancePlatform,
    IntelligentGPUMemoryManager,
    OptimizationStrategy,
    OptimizedModel,
    PerformanceMetrics,
    WorkloadProfile,
)

# Scientific Computing Integration
from opifex.optimization.scientific_integration import (
    ConservationCheckResult,
    ConservationLaw,
    NumericalValidationResult,
    NumericalValidator,
    PhysicsDomain,
    PhysicsMetrics,
    PhysicsProfiler,
    ScientificBenchmarkResult,
    ScientificBenchmarkValidator,
    ScientificComputingIntegrator,
)


__all__ = [
    # Core production optimization
    "AdaptiveJAXOptimizer",
    # Scientific computing integration
    "ConservationCheckResult",
    "ConservationLaw",
    "HybridPerformancePlatform",
    "IntelligentGPUMemoryManager",
    "NumericalValidationResult",
    "NumericalValidator",
    "OptimizationStrategy",
    "OptimizedModel",
    "PerformanceMetrics",
    "PhysicsDomain",
    "PhysicsMetrics",
    "PhysicsProfiler",
    "ScientificBenchmarkResult",
    "ScientificBenchmarkValidator",
    "ScientificComputingIntegrator",
    "WorkloadProfile",
]
