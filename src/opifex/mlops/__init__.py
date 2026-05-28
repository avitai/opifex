"""Opifex MLOps - Unified experiment tracking and model lifecycle management.

This package provides a tool-agnostic interface for experiment tracking, model
versioning, and deployment automation optimized for scientific machine learning
workflows.
"""

from opifex.mlops._uq_capabilities import MLOPS_CAPABILITIES
from opifex.mlops.backends import MLFLOW_AVAILABLE, MLflowBackend
from opifex.mlops.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentTracker,
    Framework,
    L2OMetrics,
    NeuralDFTMetrics,
    NeuralOperatorMetrics,
    PhysicsDomain,
    PhysicsMetadata,
    PINNMetrics,
    QuantumMetrics,
)
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.5. Guarded against duplicate
# registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in MLOPS_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__version__ = "1.0.0"
__author__ = "Opifex Team"
__email__ = "team@opifex.io"

__all__ = [
    "MLFLOW_AVAILABLE",
    "MLOPS_CAPABILITIES",
    "Experiment",
    "ExperimentConfig",
    "ExperimentTracker",
    "Framework",
    "L2OMetrics",
    "MLflowBackend",
    "NeuralDFTMetrics",
    "NeuralOperatorMetrics",
    "PINNMetrics",
    "PhysicsDomain",
    "PhysicsMetadata",
    "QuantumMetrics",
]

# Package metadata
SUPPORTED_PHYSICS_DOMAINS = [
    "neural-operators",
    "l2o",
    "neural-dft",
    "pinn",
    "quantum-computing",
]

SUPPORTED_FRAMEWORKS = ["jax", "pytorch", "tensorflow"]

SUPPORTED_BACKENDS = ["mlflow"] if MLFLOW_AVAILABLE else []
