"""Core experiment tracking interfaces for scientific computing.

This module provides physics-informed metadata and experiment tracking
capabilities for scientific machine learning workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import datetime


class PhysicsDomain(Enum):
    """Scientific computing domains supported by Opifex."""

    NEURAL_OPERATORS = "neural-operators"
    L2O = "l2o"
    NEURAL_DFT = "neural-dft"
    PINN = "pinn"
    QUANTUM_COMPUTING = "quantum-computing"


class Framework(Enum):
    """Machine learning frameworks supported by Opifex."""

    JAX = "jax"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


@dataclass
class PhysicsMetadata:
    """Physics-informed metadata for scientific computing experiments."""

    # PDE and physics information
    pde_type: str | None = None
    dimensionality: int | None = None
    boundary_conditions: list[str] | None = None
    conservation_laws: list[str] | None = None
    symmetries: list[str] | None = None
    physical_constants: dict[str, float] | None = None
    coordinate_system: str | None = None
    temporal_scheme: str | None = None

    # Domain-specific information
    domain_bounds: tuple[float, ...] | None = None
    grid_resolution: tuple[int, ...] | None = None
    time_horizon: float | None = None

    # Material and system properties
    material_properties: dict[str, Any] | None = None
    system_parameters: dict[str, Any] | None = None

    # Validation and verification
    analytical_solution: str | None = None
    reference_data_source: str | None = None
    validation_metrics: dict[str, str] | None = None


@dataclass
class NeuralOperatorMetrics:
    """Metrics specific to neural operator experiments."""

    # Core required metrics
    train_loss: float
    val_loss: float
    spectral_accuracy: float
    relative_l2_error: float
    max_absolute_error: float
    pde_residual: float
    conservation_error: float
    boundary_condition_error: float
    stability_measure: float
    physical_consistency: float
    inference_time_per_sample: float
    memory_usage_mb: float

    # Optional metrics
    test_loss: float | None = None
    gpu_utilization_percent: float | None = None
    out_of_distribution_error: float | None = None
    parameter_extrapolation_error: float | None = None


@dataclass
class L2OMetrics:
    """Metrics specific to learning-to-optimize experiments."""

    # Optimization performance (required)
    final_objective_value: float
    convergence_iterations: int
    convergence_time_seconds: float

    # Learning metrics (required)
    meta_loss: float
    adaptation_loss: float
    meta_gradient_norm: float

    # Task generalization (required)
    few_shot_performance: dict[str, float]
    zero_shot_performance: dict[str, float]

    # Optimizer analysis (required)
    update_magnitude: float
    gradient_scaling_factor: float

    # Optional metrics
    task_similarity_scores: dict[str, float] | None = None
    learned_lr_schedule: list[float] | None = None
    momentum_schedule: list[float] | None = None


@dataclass
class NeuralDFTMetrics:
    """Metrics specific to neural density functional theory experiments."""

    # DFT accuracy metrics (required)
    total_energy_error_hartree: float
    forces_error_ev_per_angstrom: float
    density_mse: float
    exchange_correlation_error: float

    # Chemical accuracy (required)
    atomization_energy_error_kcal_per_mol: float
    bond_length_error_angstrom: float

    # Physical constraints satisfaction (required)
    particle_number_conservation: float
    density_positivity_violation: float
    symmetry_preservation: float

    # Computational efficiency (required)
    scf_iterations: int
    scf_convergence_time: float
    density_optimization_time: float

    # Optional metrics
    vibrational_frequency_error_cm_minus_1: float | None = None


@dataclass
class PINNMetrics:
    """Metrics specific to physics-informed neural network experiments."""

    # PINN loss components (required)
    data_loss: float
    physics_loss: float
    boundary_loss: float
    initial_condition_loss: float

    # Solution accuracy (required)
    solution_l2_error: float
    solution_max_error: float
    derivative_accuracy: dict[str, float]

    # Physics compliance (required)
    pde_residual_l2: float
    pde_residual_max: float
    conservation_violation: float

    # Training dynamics (required)
    loss_balance: dict[str, float]
    gradient_pathology_measure: float
    training_stability: float

    # Optional metrics
    causality_violation: float | None = None


@dataclass
class QuantumMetrics:
    """Metrics specific to quantum computing experiments."""

    # Quantum state fidelity (required)
    state_fidelity: float

    # Circuit metrics (required)
    circuit_depth: int
    gate_count: int
    connectivity_score: float

    # Error metrics (required)
    coherence_error: float
    measurement_error: float
    readout_error: float

    # Quantum advantage (required)
    quantum_execution_time: float

    # Optional metrics
    process_fidelity: float | None = None
    average_gate_fidelity: float | None = None
    crosstalk_error: float | None = None
    classical_simulation_time: float | None = None
    quantum_volume: int | None = None
    energy_variance: float | None = None
    parameter_gradient_magnitude: float | None = None


@dataclass
class ExperimentConfig:
    """Configuration for scientific computing experiments."""

    # Basic experiment info
    name: str
    physics_domain: PhysicsDomain
    framework: Framework
    description: str | None = None
    tags: list[str] = field(default_factory=list)

    # Scientific metadata
    physics_metadata: PhysicsMetadata | None = None

    # Backend configuration
    backend: str = "auto"  # auto, mlflow, wandb, neptune, opifex
    backend_config: dict[str, Any] = field(default_factory=dict)

    # Research context
    research_group: str | None = None
    project_id: str | None = None
    paper_reference: str | None = None
    dataset_id: str | None = None

    # Reproducibility
    random_seed: int | None = None
    environment_hash: str | None = None
    git_commit: str | None = None

    # Performance tracking
    enable_gpu_tracking: bool = True
    enable_memory_tracking: bool = True
    enable_physics_validation: bool = True


class Experiment(ABC):
    """Abstract base class for scientific computing experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.id: str | None = None
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.status: str = "created"
        self._metrics: dict[str, Any] = {}
        self._artifacts: dict[str, str] = {}
        self._parameters: dict[str, Any] = {}

    @abstractmethod
    async def start(self) -> str:
        """Start the experiment and return experiment ID."""

    @abstractmethod
    async def log_metrics(
        self, metrics: dict[str, float | int], step: int | None = None
    ):
        """Log scalar metrics."""

    @abstractmethod
    async def log_physics_metrics(
        self,
        metrics: NeuralOperatorMetrics
        | L2OMetrics
        | NeuralDFTMetrics
        | PINNMetrics
        | QuantumMetrics,
        step: int | None = None,
    ):
        """Log physics-informed metrics specific to the domain."""

    @abstractmethod
    async def log_parameters(self, params: dict[str, Any]):
        """Log experiment parameters and hyperparameters."""

    @abstractmethod
    async def log_artifact(self, local_path: str, artifact_path: str | None = None):
        """Log an artifact (model, plot, data file)."""

    @abstractmethod
    async def log_model(
        self,
        model: Any,
        model_name: str,
        physics_metadata: PhysicsMetadata | None = None,
    ):
        """Log a trained model with scientific metadata."""

    @abstractmethod
    async def end(self, status: str = "completed"):
        """End the experiment."""

    def get_experiment_url(self) -> str | None:
        """Get the URL to view this experiment in the backend UI."""
        return None

    def get_metrics(self) -> dict[str, Any]:
        """Get all logged metrics."""
        return self._metrics.copy()

    def get_parameters(self) -> dict[str, Any]:
        """Get all logged parameters."""
        return self._parameters.copy()

    def get_artifacts(self) -> dict[str, str]:
        """Get all logged artifacts."""
        return self._artifacts.copy()


class ExperimentTracker:
    """Factory for creating experiment instances with appropriate backends."""

    def __init__(self, default_backend: str = "auto"):
        self.default_backend = default_backend
        self._backend_registry: dict[str, type[Experiment]] = {}

    def register_backend(self, name: str, backend_class):
        """Register a new backend implementation."""
        self._backend_registry[name] = backend_class

    async def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """Create an experiment with the appropriate backend."""
        backend = (
            config.backend if config.backend != "auto" else self._select_backend(config)
        )

        if backend not in self._backend_registry:
            available = list(self._backend_registry.keys())
            raise ValueError(f"Backend '{backend}' not found. Available: {available}")

        backend_class = self._backend_registry[backend]
        return backend_class(config)

    def _select_backend(self, config: ExperimentConfig) -> str:
        """Auto-select the best backend for the experiment configuration."""
        # Logic for automatic backend selection based on:
        # - Physics domain
        # - Available backends
        # - User preferences
        # - Performance requirements

        if config.physics_domain in [
            PhysicsDomain.NEURAL_DFT,
            PhysicsDomain.QUANTUM_COMPUTING,
        ]:
            # These domains benefit from detailed scientific metadata
            return "opifex"
        if config.research_group and "collaboration" in config.tags:
            # Research collaboration scenarios
            return "wandb"
        # Default enterprise backend
        return "mlflow"
