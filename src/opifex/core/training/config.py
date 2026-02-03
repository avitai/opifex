"""Centralized training configuration classes for Opifex framework.

This module provides comprehensive configuration management for training,
including quantum-aware settings, loss configuration, optimization parameters,
validation settings, checkpointing options, and meta-optimization.

This is the single source of truth for all training configurations, eliminating
duplication across the codebase and following strict DRY principles.

Following strict TDD principles - all classes are implemented to pass
the tests defined in test_config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
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


@dataclass
class QuantumTrainingConfig:
    """Configuration for quantum-aware training.

    Attributes:
        chemical_accuracy_target: Target chemical accuracy in kcal/mol
        scf_max_iterations: Maximum SCF iterations
        scf_tolerance: SCF convergence tolerance
        enable_symmetry_enforcement: Enable symmetry enforcement
        enable_density_constraints: Enable density constraints
        enable_energy_conservation: Enable energy conservation
    """

    chemical_accuracy_target: float = 1e-3  # kcal/mol
    scf_max_iterations: int = 100
    scf_tolerance: float = 1e-6
    enable_symmetry_enforcement: bool = True
    enable_density_constraints: bool = True
    enable_energy_conservation: bool = True


@dataclass
class LossConfig:
    """Configuration for loss computation.

    Attributes:
        loss_type: Type of loss function ('mse', 'mae', 'quantum_energy')
        physics_weight: Weight for physics loss component
        boundary_weight: Weight for boundary loss component
        quantum_constraint_weight: Weight for quantum constraints
        density_constraint_weight: Weight for density constraints
        regularization_weight: Weight for regularization term
    """

    loss_type: str = "mse"  # 'mse', 'mae', 'quantum_energy'
    physics_weight: float = 1.0
    boundary_weight: float = 1.0
    quantum_constraint_weight: float = 1.0
    density_constraint_weight: float = 1.0
    regularization_weight: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for optimization.

    Attributes:
        optimizer: Optimizer type ('adam', 'sgd', 'rmsprop', 'adamw')
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        momentum: Momentum coefficient (for SGD)
        eps: Epsilon for numerical stability (for Adam)
        beta1: Beta1 for Adam optimizer
        beta2: Beta2 for Adam optimizer
    """

    optimizer: str = "adam"  # 'adam', 'sgd', 'rmsprop', 'adamw'
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9  # for SGD
    eps: float = 1e-8  # for Adam
    beta1: float = 0.9  # for Adam
    beta2: float = 0.999  # for Adam


@dataclass
class ValidationConfig:
    """Configuration for validation.

    Attributes:
        validation_frequency: Frequency of validation steps
        early_stopping_patience: Early stopping patience
        early_stopping_min_delta: Minimum delta for early stopping
        compute_val_metrics: Whether to compute validation metrics
    """

    validation_frequency: int = 10
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6
    compute_val_metrics: bool = True


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing.

    Attributes:
        checkpoint_dir: Directory for saving checkpoints
        save_frequency: Frequency of checkpoint saves
        max_to_keep: Maximum number of checkpoints to keep
        save_best_only: Whether to save only the best checkpoint
    """

    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 50
    max_to_keep: int = 5
    save_best_only: bool = False


@dataclass
class MetaOptimizerConfig:
    """Configuration for meta-optimization algorithms.

    This configuration class defines all parameters for meta-optimization
    including algorithm selection, adaptation strategies, and performance
    monitoring settings.

    Attributes:
        meta_algorithm: Meta-optimization algorithm ('l2o', 'adaptive_lr', 'warm_start')
        base_optimizer: Base optimizer to enhance ('adam', 'sgd', 'rmsprop', 'adamw')
        meta_learning_rate: Learning rate for meta-parameters
        adaptation_steps: Number of steps for adaptation
        warm_start_strategy: Strategy for warm-starting
            ('previous_params', 'similar_problems')
        performance_tracking: Enable performance monitoring
        memory_efficient: Use memory-efficient implementations
        quantum_aware: Enable quantum-specific adaptations
        scf_adaptation: Enable SCF convergence acceleration
        energy_convergence_tracking: Track energy convergence for quantum systems
        chemical_accuracy_target: Target chemical accuracy (kcal/mol)
    """

    meta_algorithm: str = "l2o"
    base_optimizer: str = "adam"
    meta_learning_rate: float = 1e-4
    adaptation_steps: int = 10
    warm_start_strategy: str = "previous_params"
    performance_tracking: bool = True
    memory_efficient: bool = True
    quantum_aware: bool = False
    scf_adaptation: bool = False
    energy_convergence_tracking: bool = False
    chemical_accuracy_target: float = 1e-3

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_algorithms = ["l2o", "adaptive_lr", "warm_start"]
        if self.meta_algorithm not in valid_algorithms:
            raise ValueError(
                f"Invalid meta algorithm '{self.meta_algorithm}'. "
                f"Must be one of: {valid_algorithms}"
            )

        valid_optimizers = ["adam", "sgd", "rmsprop", "adamw"]
        if self.base_optimizer not in valid_optimizers:
            raise ValueError(
                f"Invalid base optimizer '{self.base_optimizer}'. "
                f"Must be one of: {valid_optimizers}"
            )


@dataclass
class TrainingConfig:
    """Main training configuration.

    This is the primary configuration class that composes all other
    configuration classes for comprehensive training setup.

    Attributes:
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate (synced to optimization_config)
        validation_frequency: Validation frequency (synced to validation_config)
        checkpoint_frequency: Checkpoint frequency (synced to checkpoint_config)
        progress_callback: Optional callback for progress updates
        verbose: Whether to print basic progress information
        loss_config: Loss computation configuration
        optimization_config: Optimization configuration
        validation_config: Validation configuration
        checkpoint_config: Checkpointing configuration
        quantum_config: Optional quantum training configuration
    """

    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    validation_frequency: int = 10
    checkpoint_frequency: int = 50

    # Progress feedback configuration
    progress_callback: Any = None  # Optional callback for progress updates
    verbose: bool = True  # Whether to print basic progress information

    # Sub-configurations
    loss_config: LossConfig = field(default_factory=LossConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)
    quantum_config: QuantumTrainingConfig | None = None

    # Physics sub-configs (composable, all optional)
    constraint_config: ConstraintConfig | None = None
    conservation_config: ConservationConfig | None = None
    multiscale_config: MultiScaleConfig | None = None
    boundary_config: BoundaryConfig | None = None
    dft_config: DFTConfig | None = None
    scf_config: SCFConfig | None = None
    electronic_structure_config: ElectronicStructureConfig | None = None
    metrics_tracking_config: MetricsTrackingConfig | None = None
    logging_config: LoggingConfig | None = None
    performance_config: PerformanceConfig | None = None

    def __post_init__(self):
        """Update sub-configs with main config values.

        This ensures that the main config values are synchronized to the
        corresponding sub-configuration instances.
        """
        self.optimization_config.learning_rate = self.learning_rate
        self.validation_config.validation_frequency = self.validation_frequency
        self.checkpoint_config.save_frequency = self.checkpoint_frequency


__all__ = [
    "CheckpointConfig",
    "LossConfig",
    "MetaOptimizerConfig",
    "OptimizationConfig",
    "QuantumTrainingConfig",
    "TrainingConfig",
    "ValidationConfig",
]
