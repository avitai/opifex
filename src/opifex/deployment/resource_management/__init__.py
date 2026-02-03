"""Global Resource Management for Opifex deployment and production.

This package implements multi-cloud optimization, GPU pool management,
cost intelligence, and sustainability tracking for production deployments.

NO BACKWARD COMPATIBILITY - Clean package structure with breaking changes.
"""

# Import all types
# Import managers
from opifex.deployment.resource_management.cost_controller import CostController
from opifex.deployment.resource_management.global_manager import GlobalResourceManager
from opifex.deployment.resource_management.gpu_manager import GPUPoolManager
from opifex.deployment.resource_management.orchestrator import ResourceOrchestrator
from opifex.deployment.resource_management.sustainability import SustainabilityTracker
from opifex.deployment.resource_management.types import (
    CloudProvider,
    CostOptimization,
    OptimizationObjective,
    ResourceAllocation,
    ResourcePool,
    ResourceType,
    SustainabilityMetrics,
)


__all__ = [
    # Types
    "CloudProvider",
    # Managers
    "CostController",
    "CostOptimization",
    "GPUPoolManager",
    "GlobalResourceManager",
    "OptimizationObjective",
    "ResourceAllocation",
    "ResourceOrchestrator",
    "ResourcePool",
    "ResourceType",
    "SustainabilityMetrics",
    "SustainabilityTracker",
]
