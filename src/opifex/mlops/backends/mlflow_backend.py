"""MLflow backend: an ``Experiment`` whose run is a substrax ``MLFlowLogger``."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, TYPE_CHECKING

from substrax.checkpoint import OrbaxCheckpointStore
from substrax.tracking import MLFlowLogger

from opifex.mlops.experiment import Experiment
from opifex.mlops.records import flatten_physics_metrics, physics_metadata_params


if TYPE_CHECKING:
    from substrax.checkpoint import ModelLike

    from opifex.mlops.backends.run_logger import RunLogger
    from opifex.mlops.experiment import ExperimentConfig, PhysicsMetadata, PhysicsMetrics


_TRACKING_URI_VARIABLE = "MLFLOW_TRACKING_URI"
_CONTEXT_FIELDS = (
    "description",
    "research_group",
    "project_id",
    "paper_reference",
    "dataset_id",
    "git_commit",
    "environment_hash",
    "random_seed",
)


class MLflowBackend(Experiment):
    """Records an experiment in an MLflow run.

    ``start`` opens the run through substrax's ``MLFlowLogger`` in the experiment
    ``opifex_<physics domain>_<name>`` on the tracking URI from
    ``backend_config["tracking_uri"]`` or ``MLFLOW_TRACKING_URI`` (MLflow's default
    store otherwise). A ``logger`` given at construction is used instead, which is
    how tests and other run stores plug in. Models are logged as Orbax checkpoints
    written by substrax's store.

    Args:
        config: The experiment configuration.
        logger: The run to record in; opened on ``start`` when omitted.
    """

    def __init__(self, config: ExperimentConfig, *, logger: RunLogger | None = None) -> None:
        super().__init__(config)
        self._logger = logger
        self._tracking_uri: str | None = config.backend_config.get(
            "tracking_uri"
        ) or os.environ.get(_TRACKING_URI_VARIABLE)

    @property
    def logger(self) -> RunLogger:
        """The run being recorded in.

        Raises:
            RuntimeError: If the experiment has not been started.
        """
        if self.id is None or self._logger is None:
            raise RuntimeError("The experiment has not been started; call start() first.")
        return self._logger

    async def start(self) -> str:
        """Open the run and log the configuration; returns the run id."""
        self.start_time = datetime.now(UTC)
        if self._logger is None:
            self._logger = self._open_run(self.start_time)
        self.id = self._logger.run_id
        self.status = "running"
        self._logger.log_hyperparams(self._initial_params())
        return self.id

    def _open_run(self, started: datetime) -> MLFlowLogger:
        """Start an MLflow run for this experiment."""
        # MLflow 3.13+ refuses the filesystem store unless this opt-out is set; local
        # ``file://`` stores are supported, so honour them without overriding an
        # explicit operator choice.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        return MLFlowLogger(
            self.config.name,
            experiment_name=f"opifex_{self.config.physics_domain.value}_{self.config.name}",
            run_name=f"{self.config.name}_{started:%Y%m%d_%H%M%S}",
            tracking_uri=self._tracking_uri,
        )

    def _initial_params(self) -> dict[str, Any]:
        """The configuration as run parameters: identity, context, backend, physics."""
        config = self.config
        params: dict[str, Any] = {
            "physics_domain": config.physics_domain.value,
            "framework": config.framework.value,
            "enable_gpu_tracking": config.enable_gpu_tracking,
            "enable_memory_tracking": config.enable_memory_tracking,
            "enable_physics_validation": config.enable_physics_validation,
        }
        context = {name: getattr(config, name) for name in _CONTEXT_FIELDS}
        params.update({name: value for name, value in context.items() if value is not None})
        params.update({f"backend.{key}": value for key, value in config.backend_config.items()})
        if config.physics_metadata is not None:
            params.update(physics_metadata_params(config.physics_metadata))
        return params

    async def log_metrics(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """Log scalar metrics at ``step``."""
        self.logger.log_scalars(metrics, step)
        self._metrics.update(metrics)

    async def log_physics_metrics(self, metrics: PhysicsMetrics, step: int | None = None) -> None:
        """Log a physics metrics record, flattened to one metric per field."""
        await self.log_metrics(flatten_physics_metrics(metrics), step=step)

    async def log_parameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.logger.log_hyperparams(params)
        self._parameters.update(params)

    async def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log one file under ``artifact_path`` in the run's artifacts."""
        self.logger.log_artifact(local_path, artifact_path)
        self._artifacts[artifact_path or Path(local_path).name] = local_path

    async def log_model(
        self,
        model: ModelLike,
        model_name: str,
        physics_metadata: PhysicsMetadata | None = None,
    ) -> None:
        """Log ``model`` as an Orbax checkpoint under ``model_name`` in the run's artifacts.

        The checkpoint is substrax's: the model state under step 0 next to a JSON
        metadata record carrying the framework, the physics domain and
        ``physics_metadata``. ``OrbaxCheckpointStore(<artifact dir>).restore(step=0)``
        reads it back.

        Args:
            model: An ``nnx.Module``, a Flax ``TrainState`` or a state dictionary.
            model_name: Artifact directory of the checkpoint within the run.
            physics_metadata: Recorded in the checkpoint's metadata when given.
        """
        additional_metadata = {
            "framework": self.config.framework.value,
            "physics_domain": self.config.physics_domain.value,
        }
        with tempfile.TemporaryDirectory() as staging:
            directory = Path(staging) / model_name
            with OrbaxCheckpointStore(directory, max_to_keep=1) as store:
                store.save(
                    model,
                    step=0,
                    physics_metadata=None if physics_metadata is None else asdict(physics_metadata),
                    additional_metadata=additional_metadata,
                )
            self.logger.log_artifacts(directory, model_name)
        self._artifacts[model_name] = model_name

    async def end(self, status: str = "completed") -> None:
        """Log the duration and the outcome, then close the run."""
        self.end_time = datetime.now(UTC)
        self.status = status
        duration = (
            0.0 if self.start_time is None else (self.end_time - self.start_time).total_seconds()
        )
        await self.log_metrics(
            {
                "experiment.duration_seconds": duration,
                "experiment.status": 1.0 if status == "completed" else 0.0,
            }
        )
        self.logger.close()

    def get_experiment_url(self) -> str | None:
        """The run's page in the MLflow UI, while a run opened on a known tracking URI is active."""
        logger = self._logger
        if (
            not isinstance(logger, MLFlowLogger)
            or logger.active_run is None
            or self._tracking_uri is None
        ):
            return None
        experiment_id = logger.active_run.info.experiment_id
        return f"{self._tracking_uri}/#/experiments/{experiment_id}/runs/{logger.run_id}"
