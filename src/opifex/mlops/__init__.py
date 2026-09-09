"""Experiment tracking for scientific machine learning.

``ExperimentTracker`` creates an ``Experiment`` on a registered backend; the MLflow
backend records physics-informed metadata, domain metrics records and Orbax model
checkpoints in an MLflow run through substrax's tracking and checkpoint layers.
"""

from opifex.mlops._uq_capabilities import MLOPS_CAPABILITIES, register_mlops_capabilities
from opifex.mlops.backends import MLflowBackend
from opifex.mlops.experiment import (
    Experiment,
    ExperimentConfig,
    Framework,
    L2OMetrics,
    NeuralDFTMetrics,
    NeuralOperatorMetrics,
    PhysicsDomain,
    PhysicsMetadata,
    PINNMetrics,
    QuantumMetrics,
)
from opifex.mlops.tracker import ExperimentTracker


__all__ = [
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
    "register_mlops_capabilities",
]
