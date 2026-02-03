"""Composable physics configuration classes for Opifex framework.

This module provides small, focused configuration classes that compose together
following strict DRY principles. Each config class has a single responsibility
and can be independently tested and composed.

Following strict TDD principles - all classes are implemented to pass
the tests defined in test_physics_configs.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConstraintConfig:
    """Configuration for physics constraint enforcement.

    Attributes:
        constraints: List of constraint names to enforce
        weights: Constraint weights (constraint_name -> weight)
        adaptive_weighting: Enable adaptive constraint weighting
        adaptation_rate: Rate of weight adaptation (0.0-1.0)
        violation_threshold: Threshold for violation detection
        violation_monitoring: Enable real-time violation monitoring
    """

    constraints: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    adaptive_weighting: bool = False
    adaptation_rate: float = 0.1
    violation_threshold: float = 0.01
    violation_monitoring: bool = False


@dataclass
class ConservationConfig:
    """Configuration for conservation law enforcement.

    Attributes:
        laws: List of conservation laws to enforce
        energy_tolerance: Tolerance for energy conservation
        energy_monitoring: Enable energy conservation monitoring
        momentum_tolerance: Tolerance for momentum conservation
        momentum_components: Momentum components to track
        symmetry_tolerance: Tolerance for symmetry preservation
        symmetry_groups: Symmetry groups to preserve
        particle_tolerance: Tolerance for particle number
        target_particle_number: Target particle number
    """

    laws: list[str] = field(default_factory=list)
    energy_tolerance: float = 1e-6
    energy_monitoring: bool = False
    momentum_tolerance: float = 1e-5
    momentum_components: list[str] = field(default_factory=list)
    symmetry_tolerance: float = 1e-6
    symmetry_groups: list[str] = field(default_factory=list)
    particle_tolerance: float = 1e-4
    target_particle_number: float = 0.0


@dataclass
class MultiScaleConfig:
    """Configuration for multi-scale physics integration.

    Attributes:
        scales: Physics scales to integrate
        weights: Scale weights (scale_name -> weight)
        coupling: Enable coupling between scales
    """

    scales: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    coupling: bool = False


@dataclass
class BoundaryConfig:
    """Configuration for boundary conditions.

    Attributes:
        weight: Weight for boundary loss
        enforce: Enable boundary condition enforcement
    """

    weight: float = 0.0
    enforce: bool = False


@dataclass
class DFTConfig:
    """Configuration for density functional theory.

    Attributes:
        functional: DFT functional type
        electron_density_threshold: Electron density threshold
        exchange_correlation_weight: Exchange-correlation weight
    """

    functional: str = "pbe"
    electron_density_threshold: float = 1e-8
    exchange_correlation_weight: float = 0.3


@dataclass
class SCFConfig:
    """Configuration for self-consistent field calculations.

    Attributes:
        max_iterations: Maximum SCF iterations
        tolerance: SCF convergence tolerance
        mixing_parameter: SCF mixing parameter
    """

    max_iterations: int = 50
    tolerance: float = 1e-6
    mixing_parameter: float = 0.5


@dataclass
class ElectronicStructureConfig:
    """Configuration for electronic structure optimization.

    Attributes:
        enabled: Enable electronic structure optimization
        orbital_optimization: Enable orbital optimization
        temperature: Electronic temperature (K)
    """

    enabled: bool = False
    orbital_optimization: bool = False
    temperature: float = 300.0


@dataclass
class ChemicalAccuracyTracking:
    """Configuration for chemical accuracy tracking.

    Attributes:
        enabled: Enable chemical accuracy tracking
        target: Target chemical accuracy threshold (kcal/mol)
        history_size: Number of historical values to retain
    """

    enabled: bool = False
    target: float = 1e-3  # kcal/mol
    history_size: int = 100


@dataclass
class SCFConvergenceTracking:
    """Configuration for SCF convergence tracking.

    Attributes:
        enabled: Enable SCF convergence tracking
        tolerance: SCF convergence tolerance
        history_size: Number of historical values to retain
    """

    enabled: bool = False
    tolerance: float = 1e-6
    history_size: int = 100


@dataclass
class ConservationViolationTracking:
    """Configuration for conservation violation tracking.

    Attributes:
        enabled: Enable conservation violation tracking
        threshold: Violation threshold for alerting
        history_size: Number of historical values to retain
    """

    enabled: bool = False
    threshold: float = 0.01
    history_size: int = 100


@dataclass
class MetricsTrackingConfig:
    """Configuration for physics metrics tracking.

    Attributes:
        detailed: Enable detailed tracking
        physics_metrics: Physics metrics to track
        quantum_states: Track quantum states
        state_history: Track quantum state history
        chemical_accuracy: Chemical accuracy tracking configuration
        scf_convergence: SCF convergence tracking configuration
        conservation_violations: Conservation violation tracking configuration
    """

    detailed: bool = True
    physics_metrics: list[str] = field(default_factory=lambda: ["chemical_accuracy"])
    quantum_states: bool = False
    state_history: bool = False
    chemical_accuracy: ChemicalAccuracyTracking | None = None
    scf_convergence: SCFConvergenceTracking | None = None
    conservation_violations: ConservationViolationTracking | None = None


@dataclass
class LoggingConfig:
    """Configuration for physics-specific logging.

    Attributes:
        backends: Logging backends to use
        level: Logging level
        frequency: Metrics logging frequency
        real_time: Enable real-time monitoring
        alert_thresholds: Alert thresholds (metric_name -> threshold)
    """

    backends: list[str] = field(default_factory=lambda: ["tensorboard"])
    level: str = "INFO"
    frequency: int = 1
    real_time: bool = False
    alert_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceConfig:
    """Configuration for performance analytics.

    Attributes:
        analytics: Enable performance analytics
        timing: Enable timing analysis
        memory: Enable memory tracking
        convergence: Enable convergence analysis
    """

    analytics: bool = False
    timing: bool = False
    memory: bool = False
    convergence: bool = False


__all__ = [
    "BoundaryConfig",
    "ChemicalAccuracyTracking",
    "ConservationConfig",
    "ConservationViolationTracking",
    "ConstraintConfig",
    "DFTConfig",
    "ElectronicStructureConfig",
    "LoggingConfig",
    "MetricsTrackingConfig",
    "MultiScaleConfig",
    "PerformanceConfig",
    "SCFConfig",
    "SCFConvergenceTracking",
]
