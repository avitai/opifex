"""Opifex Optimization Module.

This module provides full optimization components for the Opifex framework,
including production optimization, L2O meta-optimization, and Version 7.4 enhancements.

Version 7.4 Production Optimization Components:
- Hybrid Performance Platform with adaptive JIT optimization
- AI-powered performance monitoring and predictive scaling
- Scientific computing integration with physics-informed validation
- Intelligent edge network with global distribution and sub-ms latency
- Adaptive deployment system with AI-driven strategies and rollback automation
- Global resource management with multi-cloud optimization and cost intelligence
"""

# Core optimization components
# Version 7.4: Performance Monitoring & Prediction
# Version 7.4: Adaptive Deployment System
#
# Note: resource_management lives in opifex.deployment because deployment
# owns the concrete cloud/sustainability concerns. Callers that need the
# concrete classes should import them from
# ``opifex.deployment.resource_management`` directly — we no longer
# re-export them here, which kept the optimization layer artificially
# coupled to deployment-side symbols nobody was importing through this
# facade.
from opifex.optimization.adaptive_deployment import (
    AdaptiveDeploymentSystem,
    CanaryController,
    DeploymentAI,
    DeploymentConfig,
    DeploymentMetrics,
    DeploymentState,
    DeploymentStatus,
    DeploymentStrategy,
    RollbackDecision,
    RollbackEngine,
    RollbackTrigger,
    TrafficShaper,
)

# Version 7.4: Intelligent Edge Network
from opifex.optimization.edge_network import (
    EdgeCache,
    EdgeGateway,
    EdgeNodeMetrics,
    EdgeRegion,
    FailoverResult,
    FailoverStrategy,
    IntelligentEdgeNetwork,
    LatencyOptimizer,
    LatencyProfile,
    RegionalFailover,
)
from opifex.optimization.performance_monitoring import (
    AIAnomalyDetector,
    Anomaly,
    AnomalySeverity,
    PerformanceMetrics,
    PerformanceMonitor,
    PerformancePredictor,
    PredictionResult,
    PredictiveScaler,
)
from opifex.optimization.production import (
    AdaptiveJAXOptimizer,
    HybridPerformancePlatform,
    IntelligentGPUMemoryManager,
    OptimizationStrategy,
    OptimizedModel,
    PerformanceMetrics as ProductionPerformanceMetrics,
    WorkloadProfile,
)

# Version 7.4: Scientific Computing Integration
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
    # Performance monitoring & prediction
    "AIAnomalyDetector",
    # Adaptive deployment system
    "AdaptiveDeploymentSystem",
    # Core production optimization
    "AdaptiveJAXOptimizer",
    "Anomaly",
    "AnomalySeverity",
    "CanaryController",
    # Scientific computing integration
    "ConservationCheckResult",
    "ConservationLaw",
    "DeploymentAI",
    "DeploymentConfig",
    "DeploymentMetrics",
    "DeploymentState",
    "DeploymentStatus",
    "DeploymentStrategy",
    # Intelligent edge network
    "EdgeCache",
    "EdgeGateway",
    "EdgeNodeMetrics",
    "EdgeRegion",
    "FailoverResult",
    "FailoverStrategy",
    "HybridPerformancePlatform",
    "IntelligentEdgeNetwork",
    "IntelligentGPUMemoryManager",
    "LatencyOptimizer",
    "LatencyProfile",
    "NumericalValidationResult",
    "NumericalValidator",
    "OptimizationStrategy",
    "OptimizedModel",
    "PerformanceMetrics",
    "PerformanceMonitor",
    "PerformancePredictor",
    "PhysicsDomain",
    "PhysicsMetrics",
    "PhysicsProfiler",
    "PredictionResult",
    "PredictiveScaler",
    "ProductionPerformanceMetrics",
    "RegionalFailover",
    "RollbackDecision",
    "RollbackEngine",
    "RollbackTrigger",
    "ScientificBenchmarkResult",
    "ScientificBenchmarkValidator",
    "ScientificComputingIntegrator",
    "TrafficShaper",
    "WorkloadProfile",
]
