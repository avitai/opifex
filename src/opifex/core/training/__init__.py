"""Core training utilities for Opifex framework.

This module provides centralized, DRY-compliant training infrastructure including:
- Training configuration classes
- The mapping from ``OptimizationConfig`` to the optimizer substrax builds
- Learning rate schedules

All training components follow strict TDD principles and are designed for
high performance with JAX compatibility.
"""

from __future__ import annotations

from opifex.core.training.callbacks import EarlyStopping, PlateauMode, ReduceLROnPlateau
from opifex.core.training.config import (
    CheckpointConfig,
    LossConfig,
    MetaOptimizerConfig,
    OptimizationConfig,
    QuantumTrainingConfig,
    TrainingConfig,
    ValidationConfig,
)
from opifex.core.training.optimizers import create_schedule, optimizer_spec
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
    "EarlyStopping",
    "ElectronicStructureConfig",
    "LoggingConfig",
    "LossConfig",
    "MetaOptimizerConfig",
    "MetricsTrackingConfig",
    "MultiScaleConfig",
    "OptimizationConfig",
    "PerformanceConfig",
    "PlateauMode",
    "QuantumTrainingConfig",
    "ReduceLROnPlateau",
    "SCFConfig",
    # Trainer
    "Trainer",
    "TrainingConfig",
    "ValidationConfig",
    "create_schedule",
    "optimizer_spec",
]
