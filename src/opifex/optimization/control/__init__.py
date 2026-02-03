"""Control module for Learn-to-Optimize (L2O) framework.

This module implements differentiable predictive control components including
system identification networks and model predictive control frameworks.

Sprint 5.2: Differentiable Predictive Control
- Task 5.2.1: System Identification Networks (COMPLETE)
- Task 5.2.2: Model Predictive Control Framework (COMPLETE)
"""

from .mpc import (
    BatchMPCResult,
    ConstraintProjector,
    ControlBarrier,
    DifferentiableMPC,
    MPCConfig,
    MPCObjective,
    MPCResult,
    OptimizationResult,
    PredictiveModel,
    RealTimeOptimizer,
    RecedingHorizonController,
    SafetyCriticalMPC,
)
from .system_id import (
    BenchmarkValidationResult,
    ControlIntegratedSystemID,
    OnlineSystemLearner,
    PhysicsConstrainedSystemID,
    PhysicsConstraint,
    SystemDynamicsModel,
    SystemIdentifier,
)


__all__ = [
    # System Identification (Task 5.2.1)
    "BatchMPCResult",
    "BenchmarkValidationResult",
    "ConstraintProjector",
    "ControlBarrier",
    "ControlIntegratedSystemID",
    "DifferentiableMPC",
    "MPCConfig",
    "MPCObjective",
    "MPCResult",
    "OnlineSystemLearner",
    "OptimizationResult",
    "PhysicsConstrainedSystemID",
    "PhysicsConstraint",
    "PredictiveModel",
    "RealTimeOptimizer",
    "RecedingHorizonController",
    "SafetyCriticalMPC",
    "SystemDynamicsModel",
    "SystemIdentifier",
]
