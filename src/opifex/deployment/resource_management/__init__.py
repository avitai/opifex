"""Global Resource Management for Opifex deployment and production.

This package implements multi-cloud optimization, GPU pool management,
cost intelligence, and sustainability tracking for production deployments.

NO BACKWARD COMPATIBILITY - Clean package structure with breaking changes.
"""

# Import all types
# Import managers
from .cost_controller import CostController
from .global_manager import GlobalResourceManager
from .gpu_manager import GPUPoolManager
from .orchestrator import ResourceOrchestrator
from .sustainability import SustainabilityTracker
from .types import (
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
