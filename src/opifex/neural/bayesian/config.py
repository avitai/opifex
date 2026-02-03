"""
Configuration classes for Bayesian Physics-Informed Neural Networks.

This module contains configuration data structures for multi-fidelity
and probabilistic PINN models.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FidelityConfig:
    """Configuration for a single fidelity level."""

    data_points: int
    noise_level: float
    spatial_resolution: int
    temporal_resolution: int
    physics_weight: float = 1.0
    data_weight: float = 1.0


@dataclass
class MultiFidelityConfig:
    """
    Complete configuration for multi-fidelity PINN with all required keys.

    This configuration ensures all required attributes are present
    and properly initialized to prevent KeyError issues.
    """

    # CRITICAL: Core required keys that must be present
    low_fidelity: FidelityConfig | None = None
    high_fidelity_count: int = 1

    # Network architecture parameters
    network_params: dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_layers": [64, 64, 64],
            "activation": "tanh",
            "use_batch_norm": False,
            "dropout_rate": 0.0,
        }
    )

    # Training parameters
    training_params: dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": 1e-3,
            "batch_size": 256,
            "num_epochs": 1000,
            "optimizer": "adam",
            "weight_decay": 0.0,
        }
    )

    # Multi-fidelity specific parameters
    fidelity_weights: list[float] = field(default_factory=lambda: [0.1, 1.0])
    uncertainty_threshold: float = 0.1
    adaptive_sampling: bool = True
    ensemble_size: int = 10

    # Physics-informed parameters
    physics_params: dict[str, Any] = field(
        default_factory=lambda: {
            "pde_weight": 1.0,
            "boundary_weight": 1.0,
            "initial_weight": 1.0,
            "residual_weight": 1.0,
        }
    )

    # Uncertainty quantification parameters
    uncertainty_params: dict[str, Any] = field(
        default_factory=lambda: {
            "use_epistemic": True,
            "use_aleatoric": True,
            "monte_carlo_samples": 100,
            "prior_std": 1.0,
        }
    )

    def __post_init__(self):
        """Initialize default configurations and validate required keys."""
        # FIXED: Ensure low_fidelity is always initialized
        if self.low_fidelity is None:
            self.low_fidelity = FidelityConfig(
                data_points=1000,
                noise_level=0.01,
                spatial_resolution=32,
                temporal_resolution=50,
                physics_weight=1.0,
                data_weight=1.0,
            )

        # Validate required attributes
        self._validate_config()

    def _validate_config(self):
        """Validate that all required configuration keys are present."""
        required_keys = [
            "low_fidelity",
            "high_fidelity_count",
            "network_params",
            "training_params",
            "fidelity_weights",
            "uncertainty_threshold",
        ]

        for key in required_keys:
            if not hasattr(self, key):
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate values
        if self.high_fidelity_count <= 0:
            raise ValueError("high_fidelity_count must be positive")

        if (
            not isinstance(self.fidelity_weights, list)
            or len(self.fidelity_weights) < 2
        ):
            raise ValueError("fidelity_weights must be a list with at least 2 elements")

        if not (0.0 <= self.uncertainty_threshold <= 1.0):
            raise ValueError("uncertainty_threshold must be between 0 and 1")
