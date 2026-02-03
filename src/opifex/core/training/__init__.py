"""Core training utilities for Opifex framework.

This module provides centralized, DRY-compliant training infrastructure including:
- Training configuration classes
- Optimizer creation and configuration
- Learning rate schedules
- Gradient clipping

All training components follow strict TDD principles and are designed for
high performance with JAX compatibility.
"""

from __future__ import annotations

from opifex.core.training.config import (
    CheckpointConfig,
    LossConfig,
    MetaOptimizerConfig,
    OptimizationConfig,
    QuantumTrainingConfig,
    TrainingConfig,
    ValidationConfig,
)
from opifex.core.training.optimizers import (
    create_adam,
    create_adamw,
    create_optimizer,
    create_rmsprop,
    create_schedule,
    create_sgd,
    OptimizerConfig,
    with_gradient_clipping,
    with_schedule,
)
from opifex.core.training.physics_configs import (
    BoundaryConfig,
    ConservationConfig,
    ConstraintConfig,
    DFTConfig,
    ElectronicStructureConfig,
    LoggingConfig,
    MetricsTrackingConfig,
    MultiScaleConfig,
    PerformanceConfig,
    SCFConfig,
)
from opifex.core.training.trainer import Trainer


__all__ = [
    # Physics configuration classes (composable)
    "BoundaryConfig",
    # Configuration classes
    "CheckpointConfig",
    "ConservationConfig",
    "ConstraintConfig",
    "DFTConfig",
    "ElectronicStructureConfig",
    "LoggingConfig",
    "LossConfig",
    "MetaOptimizerConfig",
    "MetricsTrackingConfig",
    "MultiScaleConfig",
    "OptimizationConfig",
    # Optimizer config and functions
    "OptimizerConfig",
    "PerformanceConfig",
    "QuantumTrainingConfig",
    "SCFConfig",
    # Trainer
    "Trainer",
    "TrainingConfig",
    "ValidationConfig",
    "create_adam",
    "create_adamw",
    "create_optimizer",
    "create_rmsprop",
    "create_schedule",
    "create_sgd",
    "with_gradient_clipping",
    "with_schedule",
]
