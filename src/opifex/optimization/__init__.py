"""Opifex Optimization Module.

This module provides comprehensive optimization components for the Opifex framework,
including production optimization, L2O meta-optimization, and Phase 7.4 enhancements.

Phase 7.4 Production Optimization Components:
- Hybrid Performance Platform with adaptive JIT optimization
- AI-powered performance monitoring and predictive scaling
- Scientific computing integration with physics-informed validation
- Intelligent edge network with global distribution and sub-ms latency
- Adaptive deployment system with AI-driven strategies and rollback automation
- Global resource management with multi-cloud optimization and cost intelligence
"""

# Core optimization components
# Phase 7.4: Performance Monitoring & Prediction
# Phase 7.4: Adaptive Deployment System
# Phase 7.4: Global Resource Management
from opifex.deployment.resource_management import (
    CloudProvider,
    CostController,
    CostOptimization,
    GlobalResourceManager,
    GPUPoolManager,
    OptimizationObjective,
    ResourceAllocation,
    ResourceOrchestrator,
    ResourcePool,
    ResourceType,
    SustainabilityMetrics,
    SustainabilityTracker,
)
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

# Phase 7.4: Intelligent Edge Network
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

# Phase 7.4: Scientific Computing Integration
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
    # Global resource management
    "CloudProvider",
    # Scientific computing integration
    "ConservationCheckResult",
    "ConservationLaw",
    "CostController",
    "CostOptimization",
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
    "GPUPoolManager",
    "GlobalResourceManager",
    "HybridPerformancePlatform",
    "IntelligentEdgeNetwork",
    "IntelligentGPUMemoryManager",
    "LatencyOptimizer",
    "LatencyProfile",
    "NumericalValidationResult",
    "NumericalValidator",
    "OptimizationObjective",
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
    "ResourceAllocation",
    "ResourceOrchestrator",
    "ResourcePool",
    "ResourceType",
    "RollbackDecision",
    "RollbackEngine",
    "RollbackTrigger",
    "ScientificBenchmarkResult",
    "ScientificBenchmarkValidator",
    "ScientificComputingIntegrator",
    "SustainabilityMetrics",
    "SustainabilityTracker",
    "TrafficShaper",
    "WorkloadProfile",
]
