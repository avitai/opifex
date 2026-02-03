"""
MLflow backend implementation for Opifex unified experiment tracking.

This backend provides MLflow integration with scientific computing optimizations
including physics-informed metadata, GPU tracking, and large model artifact handling.
"""

import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    # Type checking imports - will be resolved during static analysis
    import mlflow  # type: ignore[import-untyped]
    import mlflow.pytorch  # type: ignore[import-untyped]
    import mlflow.sklearn  # type: ignore[import-untyped]
    import mlflow.tensorflow  # type: ignore[import-untyped]
    from mlflow.tracking import MlflowClient  # type: ignore[import-untyped]
else:
    # Runtime imports - only used when MLflow is actually available
    try:
        import mlflow
        import mlflow.pytorch

        # mlflow.sklearn imported but not used - remove it
        import mlflow.tensorflow
        from mlflow.tracking import MlflowClient
    except ImportError as e:
        raise ImportError(
            "MLflow dependencies not available. This backend should only be "
            "imported when MLflow is installed."
        ) from e

from opifex.mlops.experiment import (
    Experiment,
    ExperimentConfig,
    L2OMetrics,
    NeuralDFTMetrics,
    NeuralOperatorMetrics,
    PhysicsMetadata,
    PINNMetrics,
    QuantumMetrics,
)


class MLflowBackend(Experiment):
    """MLflow backend implementation with scientific computing optimizations."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.client = None
        self.experiment_id = None
        self.run_id = None
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow client and configuration."""
        # Configure MLflow tracking URI
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", "http://mlflow-tracking-server:5000"
        )
        mlflow.set_tracking_uri(tracking_uri)

        # Initialize client
        self.client = MlflowClient()

        # Set experiment
        experiment_name = (
            f"opifex_{self.config.physics_domain.value}_{self.config.name}"
        )
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    experiment_name,
                    tags={
                        "opifex.physics_domain": self.config.physics_domain.value,
                        "opifex.framework": self.config.framework.value,
                        "opifex.version": "1.0.0",
                        "opifex.research_group": self.config.research_group
                        or "default",
                    },
                )
            else:
                self.experiment_id = experiment.experiment_id
        except Exception:
            # Fallback to default experiment
            self.experiment_id = "0"

    async def start(self) -> str:
        """Start MLflow run with scientific metadata."""
        self.start_time = datetime.now(UTC)

        # Start MLflow run
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"{self.config.name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            tags={
                "opifex.physics_domain": self.config.physics_domain.value,
                "opifex.framework": self.config.framework.value,
                "opifex.backend": "mlflow",
                "opifex.research_group": self.config.research_group or "default",
                "opifex.project_id": self.config.project_id or "default",
                "opifex.git_commit": self.config.git_commit or "unknown",
                "opifex.description": self.config.description or "",
            },
        )

        self.run_id = run.info.run_id
        self.id = self.run_id
        self.status = "running"

        # Log initial configuration
        await self._log_initial_config()

        return self.run_id

    async def _log_initial_config(self):
        """Log initial experiment configuration and metadata."""
        # Log basic parameters
        params = {
            "physics_domain": self.config.physics_domain.value,
            "framework": self.config.framework.value,
            "random_seed": self.config.random_seed,
            "enable_gpu_tracking": self.config.enable_gpu_tracking,
            "enable_physics_validation": self.config.enable_physics_validation,
        }

        if self.config.backend_config:
            for key, value in self.config.backend_config.items():
                params[f"backend.{key}"] = value

        mlflow.log_params(params)

        # Log physics metadata if available
        if self.config.physics_metadata:
            await self._log_physics_metadata(self.config.physics_metadata)

    async def _log_physics_metadata(self, metadata: PhysicsMetadata):
        """Log physics-informed metadata."""
        physics_params = {}

        # Log basic physics parameters
        self._add_basic_physics_params(metadata, physics_params)

        # Log physics collections (laws, symmetries, conditions)
        self._add_physics_collections(metadata, physics_params)

        # Log physics constants and system parameters
        self._add_physics_mappings(metadata, physics_params)

        mlflow.log_params(physics_params)

    def _add_basic_physics_params(self, metadata: PhysicsMetadata, params: dict):
        """Add basic physics parameters to the params dictionary."""
        basic_fields = [
            ("pde_type", "physics.pde_type"),
            ("dimensionality", "physics.dimensionality"),
            ("coordinate_system", "physics.coordinate_system"),
            ("temporal_scheme", "physics.temporal_scheme"),
            ("time_horizon", "physics.time_horizon"),
        ]

        for field_name, param_key in basic_fields:
            value = getattr(metadata, field_name)
            if value:
                params[param_key] = value

        # Handle special string conversion cases
        if metadata.domain_bounds:
            params["physics.domain_bounds"] = str(metadata.domain_bounds)
        if metadata.grid_resolution:
            params["physics.grid_resolution"] = str(metadata.grid_resolution)

    def _add_physics_collections(self, metadata: PhysicsMetadata, params: dict):
        """Add physics collections (laws, symmetries, conditions) to params."""
        collection_fields = [
            ("conservation_laws", "physics.conservation_laws"),
            ("symmetries", "physics.symmetries"),
            ("boundary_conditions", "physics.boundary_conditions"),
        ]

        for field_name, param_key in collection_fields:
            value = getattr(metadata, field_name)
            if value:
                params[param_key] = ",".join(value)

    def _add_physics_mappings(self, metadata: PhysicsMetadata, params: dict):
        """Add physics constants and system parameters to params."""
        if metadata.physical_constants:
            for name, value in metadata.physical_constants.items():
                params[f"physics.constants.{name}"] = value

        if metadata.system_parameters:
            for name, value in metadata.system_parameters.items():
                params[f"physics.system.{name}"] = value

    async def log_metrics(
        self, metrics: dict[str, float | int], step: int | None = None
    ):
        """Log scalar metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)
        self._metrics.update(metrics)

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

        # Convert dataclass to dictionary
        if hasattr(metrics, "__dataclass_fields__"):
            metrics_dict = {}
            for field_name, field_value in metrics.__dict__.items():
                if field_value is not None:
                    if isinstance(field_value, dict):
                        # Flatten nested dictionaries
                        for sub_key, sub_value in field_value.items():
                            metrics_dict[f"{field_name}.{sub_key}"] = sub_value
                    elif isinstance(field_value, list):
                        # Convert lists to strings or summary statistics
                        if all(isinstance(x, (int, float)) for x in field_value):
                            metrics_dict[f"{field_name}.mean"] = sum(field_value) / len(
                                field_value
                            )
                            metrics_dict[f"{field_name}.min"] = min(field_value)
                            metrics_dict[f"{field_name}.max"] = max(field_value)
                        else:
                            metrics_dict[field_name] = str(field_value)
                    else:
                        metrics_dict[field_name] = field_value

            await self.log_metrics(metrics_dict, step=step)
        else:
            # Convert to dict for non-dataclass metrics
            metrics_dict = metrics.__dict__ if hasattr(metrics, "__dict__") else {}
            await self.log_metrics(metrics_dict, step=step)

    async def log_parameters(self, params: dict[str, Any]):
        """Log experiment parameters and hyperparameters."""
        # Convert complex types to strings for MLflow compatibility
        mlflow_params = {}
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                mlflow_params[key] = str(value)
            else:
                mlflow_params[key] = value

        mlflow.log_params(mlflow_params)
        self._parameters.update(params)

    async def log_artifact(self, local_path: str, artifact_path: str | None = None):
        """Log an artifact (model, plot, data file)."""
        mlflow.log_artifact(local_path, artifact_path)

        # Track artifact in internal registry
        artifact_name = artifact_path or Path(local_path).name
        self._artifacts[artifact_name] = local_path

    async def log_model(
        self,
        model: Any,
        model_name: str,
        physics_metadata: PhysicsMetadata | None = None,
    ):
        """Log a trained model with scientific metadata."""

        # Determine framework and log accordingly
        model_info = None

        try:
            # Try JAX/Flax model logging
            if hasattr(model, "params") or (
                isinstance(model, (dict)) and "params" in str(type(model))
            ):
                # Custom JAX model logging
                model_info = await self._log_jax_model(
                    model, model_name, physics_metadata
                )
        except ImportError:
            pass

        try:
            # Try PyTorch model logging
            import torch  # type: ignore[import-untyped]

            if isinstance(model, torch.nn.Module):
                model_info = mlflow.pytorch.log_model(
                    model,
                    model_name,
                    signature=await self._infer_model_signature(model),
                    metadata={
                        "physics_domain": self.config.physics_domain.value,
                        "framework": "pytorch",
                        **(physics_metadata.__dict__ if physics_metadata else {}),
                    },
                )
        except ImportError:
            pass

        try:
            # Try TensorFlow model logging
            import tensorflow as tf  # type: ignore[import-untyped]

            if isinstance(model, (tf.keras.Model, tf.Module)):  # pyright: ignore[reportAttributeAccessIssue]
                model_info = mlflow.tensorflow.log_model(
                    model,
                    model_name,
                    signature=await self._infer_model_signature(model),
                    metadata={
                        "physics_domain": self.config.physics_domain.value,
                        "framework": "tensorflow",
                        **(physics_metadata.__dict__ if physics_metadata else {}),
                    },
                )
        except ImportError:
            pass

        if model_info is None:
            # Fallback: serialize model as pickle
            import pickle  # nosec B403
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(model, f)
                await self.log_artifact(f.name, f"{model_name}/model.pkl")

    async def _log_jax_model(
        self,
        model: Any,
        model_name: str,
        physics_metadata: PhysicsMetadata | None = None,
    ):
        """Log JAX/Flax model with custom serialization."""
        import json
        import pickle  # nosec B403
        import tempfile

        # Create model directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model state
            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Save metadata
            metadata = {
                "framework": "jax",
                "physics_domain": self.config.physics_domain.value,
                "model_type": str(type(model)),
                **(physics_metadata.__dict__ if physics_metadata else {}),
            }

            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            # Log as artifact directory
            mlflow.log_artifacts(str(model_dir), model_name)

            return {"artifact_path": model_name}

    async def _infer_model_signature(self, model: Any):
        """Infer MLflow model signature from model."""
        # This would need implementation based on model inspection
        # For now, return None to allow model logging without signature
        return

    async def end(self, status: str = "completed"):
        """End the MLflow run."""
        self.end_time = datetime.now(UTC)
        self.status = status

        # Log final metrics
        if self.start_time:
            duration_seconds = (self.end_time - self.start_time).total_seconds()
            await self.log_metrics(
                {
                    "experiment.duration_seconds": duration_seconds,
                    "experiment.status": 1 if status == "completed" else 0,
                }
            )

        # End MLflow run
        mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")

    def get_experiment_url(self) -> str | None:
        """Get the URL to view this experiment in MLflow UI."""
        if self.run_id:
            tracking_uri = mlflow.get_tracking_uri()
            return (
                f"{tracking_uri}/#/experiments/{self.experiment_id}/runs/{self.run_id}"
            )
        return None
