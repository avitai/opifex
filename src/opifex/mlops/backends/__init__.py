"""
MLOps backends for different experiment tracking systems.

This module provides backend implementations for various experiment tracking platforms
including MLflow, Wandb, Neptune, and a custom Opifex backend.
"""

from typing import Any


# Handle optional MLflow import
try:
    from opifex.mlops.backends.mlflow_backend import MLflowBackend

    _mlflow_available = True
except ImportError:
    _mlflow_available = False

    class _MLflowBackendUnavailable:
        """Fallback MLflow backend when MLflow is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("MLflow is not installed. Install with: uv add mlflow")

    # Use the fallback class with the expected name
    MLflowBackend = _MLflowBackendUnavailable  # type: ignore[assignment, misc]

# Export the availability as a constant
MLFLOW_AVAILABLE: bool = _mlflow_available


__all__ = ["MLFLOW_AVAILABLE", "MLflowBackend"]
