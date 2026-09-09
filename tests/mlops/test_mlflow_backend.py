"""The MLflow backend is an ``Experiment`` over an injected run logger.

Every logging call is forwarded to a ``RunLogger`` (substrax's ``MLFlowLogger`` by
default); models are logged as Orbax checkpoints written by substrax's store. The
``TestWithMLflow`` class exercises the real logger against a local file store.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any, TYPE_CHECKING

import jax.numpy as jnp
import pytest
import pytest_asyncio
from substrax.checkpoint import OrbaxCheckpointStore

from opifex.mlops.backends.mlflow_backend import MLflowBackend
from opifex.mlops.experiment import (
    ExperimentConfig,
    Framework,
    PhysicsDomain,
    PhysicsMetadata,
    PINNMetrics,
)
from opifex.mlops.records import flatten_physics_metrics, physics_metadata_params


if TYPE_CHECKING:
    from collections.abc import Mapping


pytestmark = pytest.mark.asyncio


class RecordingRunLogger:
    """A ``RunLogger`` that records every call and copies logged directories."""

    def __init__(self, artifact_root: Path) -> None:
        self.run_id = "run-1234"
        self.artifact_root = artifact_root
        self.scalars: list[tuple[dict[str, float], int | None]] = []
        self.params: list[dict[str, Any]] = []
        self.artifacts: list[tuple[Path, str | None]] = []
        self.closed = False

    def log_scalars(self, scalars: Mapping[str, float], step: int | None = None) -> None:
        self.scalars.append((dict(scalars), step))

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self.params.append(dict(params))

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        self.artifacts.append((Path(local_path), artifact_path))

    def log_artifacts(self, local_dir: str | Path, artifact_path: str | None = None) -> None:
        destination = self.artifact_root / (artifact_path or Path(local_dir).name)
        shutil.copytree(local_dir, destination)
        self.artifacts.append((destination, artifact_path))

    def close(self) -> None:
        self.closed = True


_METADATA = PhysicsMetadata(pde_type="burgers", dimensionality=1, boundary_conditions=["periodic"])


def _config(**overrides: Any) -> ExperimentConfig:
    values: dict[str, Any] = {
        "name": "burgers_fno",
        "physics_domain": PhysicsDomain.NEURAL_OPERATORS,
        "framework": Framework.JAX,
        "physics_metadata": _METADATA,
        "random_seed": 7,
        "backend_config": {"artifact_bucket": "s3://runs"},
    }
    values.update(overrides)
    return ExperimentConfig(**values)


@pytest.fixture
def recorder(tmp_path: Path) -> RecordingRunLogger:
    return RecordingRunLogger(tmp_path / "artifacts")


@pytest_asyncio.fixture
async def experiment(recorder: RecordingRunLogger) -> MLflowBackend:
    backend = MLflowBackend(_config(), logger=recorder)
    await backend.start()
    return backend


class TestLifecycle:
    """``start`` opens the run and logs the configuration; ``end`` closes it."""

    async def test_start_returns_the_run_id_and_logs_the_configuration(
        self, recorder: RecordingRunLogger
    ) -> None:
        backend = MLflowBackend(_config(), logger=recorder)

        run_id = await backend.start()

        assert run_id == "run-1234"
        assert backend.id == "run-1234"
        assert backend.status == "running"
        assert backend.start_time is not None
        (params,) = recorder.params
        assert params["physics_domain"] == "neural-operators"
        assert params["framework"] == "jax"
        assert params["random_seed"] == 7
        assert params["backend.artifact_bucket"] == "s3://runs"
        for key, value in physics_metadata_params(_METADATA).items():
            assert params[key] == value

    async def test_logging_before_start_is_an_error(self, recorder: RecordingRunLogger) -> None:
        backend = MLflowBackend(_config(), logger=recorder)

        with pytest.raises(RuntimeError, match="start"):
            await backend.log_metrics({"loss": 1.0})

    async def test_end_logs_the_duration_and_closes_the_logger(
        self, experiment: MLflowBackend, recorder: RecordingRunLogger
    ) -> None:
        await experiment.end()

        assert experiment.status == "completed"
        assert experiment.end_time is not None
        final, _ = recorder.scalars[-1]
        assert final["experiment.status"] == 1.0
        assert final["experiment.duration_seconds"] >= 0.0
        assert recorder.closed

    async def test_failed_end_records_status_zero(
        self, experiment: MLflowBackend, recorder: RecordingRunLogger
    ) -> None:
        await experiment.end(status="failed")

        final, _ = recorder.scalars[-1]
        assert final["experiment.status"] == 0.0
        assert experiment.status == "failed"


class TestLogging:
    """Metrics, physics metrics, parameters and artifacts go to the logger."""

    async def test_metrics_are_forwarded_with_their_step(
        self, experiment: MLflowBackend, recorder: RecordingRunLogger
    ) -> None:
        await experiment.log_metrics({"loss": 0.25, "epoch": 3}, step=3)

        assert recorder.scalars[-1] == ({"loss": 0.25, "epoch": 3}, 3)
        assert experiment.get_metrics() == {"loss": 0.25, "epoch": 3}

    async def test_physics_metrics_are_flattened(
        self, experiment: MLflowBackend, recorder: RecordingRunLogger
    ) -> None:
        metrics = PINNMetrics(
            data_loss=0.1,
            physics_loss=0.2,
            boundary_loss=0.05,
            initial_condition_loss=0.01,
            solution_l2_error=0.02,
            solution_max_error=0.1,
            derivative_accuracy={"dx": 0.9, "dt": 0.8},
            pde_residual_l2=0.001,
            pde_residual_max=0.01,
            conservation_violation=0.0,
            loss_balance={"data": 0.5, "physics": 0.5},
            gradient_pathology_measure=0.1,
            training_stability=0.99,
        )

        await experiment.log_physics_metrics(metrics, step=10)

        assert recorder.scalars[-1] == (flatten_physics_metrics(metrics), 10)
        assert experiment.get_metrics()["derivative_accuracy.dx"] == 0.9

    async def test_parameters_are_forwarded_and_kept(
        self, experiment: MLflowBackend, recorder: RecordingRunLogger
    ) -> None:
        await experiment.log_parameters({"learning_rate": 1e-3, "modes": [12, 12]})

        assert recorder.params[-1] == {"learning_rate": 1e-3, "modes": [12, 12]}
        assert experiment.get_parameters() == {"learning_rate": 1e-3, "modes": [12, 12]}

    async def test_artifacts_are_forwarded_and_indexed_by_name(
        self, experiment: MLflowBackend, recorder: RecordingRunLogger, tmp_path: Path
    ) -> None:
        plot = tmp_path / "loss.png"
        plot.write_bytes(b"png")

        await experiment.log_artifact(str(plot))
        await experiment.log_artifact(str(plot), "plots/final")

        assert recorder.artifacts == [(plot, None), (plot, "plots/final")]
        assert experiment.get_artifacts() == {"loss.png": str(plot), "plots/final": str(plot)}


class TestLogModel:
    """A model is logged as an Orbax checkpoint directory, not a pickle."""

    async def test_log_model_writes_an_orbax_checkpoint(
        self, experiment: MLflowBackend, recorder: RecordingRunLogger
    ) -> None:
        state = {"kernel": jnp.arange(6.0).reshape(2, 3), "bias": jnp.zeros(3)}

        await experiment.log_model(state, "fno", physics_metadata=_METADATA)

        (logged, artifact_path) = recorder.artifacts[-1]
        assert artifact_path == "fno"
        assert not list(logged.rglob("*.pkl"))
        with OrbaxCheckpointStore(logged, create=False) as store:
            assert store.list_steps() == [0]
            restored, metadata = store.restore(step=0)
        assert isinstance(restored, dict)
        assert jnp.array_equal(restored["kernel"], state["kernel"])
        assert metadata["physics_metadata"]["pde_type"] == "burgers"
        assert metadata["physics_domain"] == "neural-operators"
        assert metadata["framework"] == "jax"
        assert experiment.get_artifacts()["fno"] == "fno"

    async def test_log_model_without_metadata_records_the_domain_only(
        self, experiment: MLflowBackend, recorder: RecordingRunLogger
    ) -> None:
        await experiment.log_model({"w": jnp.ones(2)}, "plain")

        (logged, _) = recorder.artifacts[-1]
        with OrbaxCheckpointStore(logged, create=False) as store:
            _, metadata = store.restore(step=0)
        assert "physics_metadata" not in metadata
        assert metadata["physics_domain"] == "neural-operators"


@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="mlflow extra absent")
class TestWithMLflow:
    """The default logger is substrax's ``MLFlowLogger`` on the configured tracking URI."""

    async def test_run_is_recorded_on_a_local_file_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mlflow

        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
        backend = MLflowBackend(_config(name="file_store"))

        run_id = await backend.start()
        await backend.log_metrics({"loss": 0.5}, step=1)
        await backend.log_parameters({"lr": 0.01})
        await backend.end()

        run = mlflow.get_run(run_id)
        assert run.data.metrics["loss"] == 0.5
        assert run.data.params["lr"] == "0.01"
        assert run.data.params["physics.pde_type"] == "burgers"
        assert run.info.status == "FINISHED"
        experiment = mlflow.get_experiment(run.info.experiment_id)
        assert experiment.name == "opifex_neural-operators_file_store"

    async def test_experiment_url_points_at_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
        backend = MLflowBackend(_config())
        assert backend.get_experiment_url() is None

        run_id = await backend.start()
        url = backend.get_experiment_url()
        await backend.end()

        assert url is not None
        assert url.endswith(f"/runs/{run_id}")
        assert url.startswith(f"file://{tmp_path / 'mlruns'}/#/experiments/")
