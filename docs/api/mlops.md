# MLOps API

`opifex.mlops` records scientific experiments in a tracking run. An
`ExperimentTracker` creates an `Experiment` on a registered backend; the MLflow
backend writes physics-informed metadata, domain metrics records, artifacts and
Orbax model checkpoints to an MLflow run through substrax's tracking and
checkpoint layers.

```python
from opifex.mlops import (
    ExperimentConfig,
    ExperimentTracker,
    Framework,
    NeuralOperatorMetrics,
    PhysicsDomain,
    PhysicsMetadata,
)
```

## Installation

The MLflow SDK ships in the `mlflow` extra. Importing `opifex.mlops` never imports
it; `Experiment.start()` loads it and raises `ImportError` naming the extra when it
is absent.

```bash
uv add "opifex[mlflow]"
```

## Quick start

Every `Experiment` method is a coroutine.

```python
import asyncio
import tempfile

import jax.numpy as jnp
import optax
from flax import nnx

from opifex.data.sources import generate_burgers
from opifex.mlops import (
    ExperimentConfig,
    ExperimentTracker,
    Framework,
    NeuralOperatorMetrics,
    PhysicsDomain,
    PhysicsMetadata,
)
from opifex.neural.operators.fno.base import FourierNeuralOperator

tracking_uri = f"file://{tempfile.mkdtemp()}"  # or your tracking server's URI
data = generate_burgers(n_samples=16, resolution=64)
inputs, targets = jnp.asarray(data["input"]), jnp.asarray(data["output"])

model = FourierNeuralOperator(1, 1, 16, 8, 2, spatial_dims=1, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


@nnx.jit
def train_step(model: FourierNeuralOperator, optimizer: nnx.Optimizer) -> jnp.ndarray:
    def loss_fn(model: FourierNeuralOperator) -> jnp.ndarray:
        return jnp.mean((model(inputs) - targets) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


physics = PhysicsMetadata(
    pde_type="burgers",
    dimensionality=1,
    boundary_conditions=["periodic"],
    physical_constants={"viscosity": 0.1},
)


async def main() -> str:
    config = ExperimentConfig(
        name="burgers_fno",
        physics_domain=PhysicsDomain.NEURAL_OPERATORS,
        framework=Framework.JAX,
        physics_metadata=physics,
        random_seed=0,
        backend_config={"tracking_uri": tracking_uri},
    )
    experiment = await ExperimentTracker().create_experiment(config)

    run_id = await experiment.start()
    await experiment.log_parameters({"learning_rate": 1e-3, "modes": 8})
    for step in range(20):
        await experiment.log_metrics({"loss": float(train_step(model, optimizer))}, step=step)
    relative_l2 = float(jnp.linalg.norm(model(inputs) - targets) / jnp.linalg.norm(targets))
    await experiment.log_physics_metrics(
        NeuralOperatorMetrics(
            train_loss=float(jnp.mean((model(inputs) - targets) ** 2)),
            val_loss=0.0,
            spectral_accuracy=0.0,
            relative_l2_error=relative_l2,
            max_absolute_error=float(jnp.max(jnp.abs(model(inputs) - targets))),
            pde_residual=0.0,
            conservation_error=0.0,
            boundary_condition_error=0.0,
            stability_measure=1.0,
            physical_consistency=0.0,
            inference_time_per_sample=0.0,
            memory_usage_mb=0.0,
        )
    )
    await experiment.log_model(model, "fno", physics_metadata=physics)
    await experiment.end()
    return run_id


run_id = asyncio.run(main())
```

## Configuration

`ExperimentConfig` is a frozen, keyword-only dataclass.

| Field | Meaning |
| --- | --- |
| `name` | Experiment name; the MLflow experiment is `opifex_<physics_domain>_<name>` |
| `physics_domain` | One of `PhysicsDomain`: `NEURAL_OPERATORS`, `L2O`, `NEURAL_DFT`, `PINN`, `QUANTUM_COMPUTING` |
| `framework` | `Framework.JAX` |
| `description`, `tags` | Free text and labels |
| `physics_metadata` | A `PhysicsMetadata`; logged as run parameters on `start` |
| `backend` | Backend name, or `"auto"` for the tracker's default |
| `backend_config` | Backend options; the MLflow backend reads `tracking_uri` and logs every entry as `backend.<key>` |
| `research_group`, `project_id`, `paper_reference`, `dataset_id` | Research context, logged as parameters when set |
| `random_seed`, `environment_hash`, `git_commit` | Reproducibility, logged as parameters when set |
| `enable_gpu_tracking`, `enable_memory_tracking`, `enable_physics_validation` | Flags, logged as parameters |

### Physics metadata

`PhysicsMetadata` carries the PDE type, dimensionality, boundary conditions,
conservation laws, symmetries, physical constants, coordinate system, temporal
scheme, domain bounds, grid resolution, time horizon, material properties, system
parameters and the validation references (`analytical_solution`,
`reference_data_source`, `validation_metrics`). Every field defaults to `None`.

`start` logs it as run parameters under the `physics.` prefix
(`opifex.mlops.records.physics_metadata_params`): scalars as they are, string
lists comma-joined, tuples through `str`, and one parameter per mapping key
(`physics.physical_constants.viscosity`). Unset fields log nothing.

## Metrics records

One frozen dataclass per domain: `NeuralOperatorMetrics`, `L2OMetrics`,
`NeuralDFTMetrics`, `PINNMetrics` and `QuantumMetrics`. `log_physics_metrics`
flattens a record to one metric per field
(`opifex.mlops.records.flatten_physics_metrics`): mappings become dotted names
(`derivative_accuracy.dx`), number sequences contribute `.mean`, `.min` and
`.max`, and unset optional fields log nothing. Any other value is a `TypeError`.

Plain scalar metrics go through `log_metrics(metrics, step=None)`; hyperparameters
through `log_parameters`. `get_metrics()`, `get_parameters()` and
`get_artifacts()` return what the experiment has logged so far.

## Models and artifacts

`log_artifact(local_path, artifact_path=None)` logs one file.

`log_model(model, model_name, physics_metadata=None)` writes an Orbax checkpoint
with substrax's `OrbaxCheckpointStore` and logs the directory under `model_name`
in the run's artifacts. `model` is an `nnx.Module` (its `nnx.state` is saved) or a
state mapping, stored as the `model` item at step 0; the record's extra values
carry the framework, the physics domain and `physics_metadata`. Read it back with
the same store, onto the state of a module built the same way:

```python
import mlflow
from substrax.checkpoint import OrbaxCheckpointStore

downloaded_artifact_dir = mlflow.artifacts.download_artifacts(
    run_id=run_id, artifact_path="fno", tracking_uri=tracking_uri
)
restored = FourierNeuralOperator(1, 1, 16, 8, 2, spatial_dims=1, rngs=nnx.Rngs(1))
with OrbaxCheckpointStore(downloaded_artifact_dir) as store:
    checkpoint = store.restore(0, templates={"model": nnx.state(restored)})
nnx.update(restored, checkpoint.items["model"])
checkpoint.metadata.extra["physics_metadata"]
```

## Backends

`ExperimentTracker(default_backend="mlflow")` starts with the MLflow backend
registered; `backends` lists the registered names, `register_backend(name, cls)`
adds one, and `create_experiment(config)` instantiates the backend `config.backend`
names, or the default for `"auto"`. An unregistered name is a `ValueError`.

### MLflow

`MLflowBackend(config, *, logger=None)` opens its run on `start` through substrax's
`MLFlowLogger`, on the tracking URI from `backend_config["tracking_uri"]` or the
`MLFLOW_TRACKING_URI` environment variable (MLflow's default store otherwise).
Local `file://` stores work; the backend sets `MLFLOW_ALLOW_FILE_STORE` when the
environment does not. `get_experiment_url()` returns the run's page while the run
is open on a known tracking URI.

`logger` injects the run to record in. Anything satisfying
`opifex.mlops.backends.RunLogger` fits: a `run_id`, `log_scalars`,
`log_hyperparams`, `log_artifact`, `log_artifacts` and `close`. Tests inject a
recording logger; a Weights & Biases run can be adapted the same way.

### Custom backends

Subclass `Experiment` and implement `start`, `log_metrics`,
`log_physics_metrics`, `log_parameters`, `log_artifact`, `log_model` and `end`,
then register it:

```python
from typing import Any

from opifex.mlops import Experiment


class ConsoleExperiment(Experiment):
    """Prints what it is asked to record."""

    async def start(self) -> str:
        print(f"start {self.config.name}")
        return self.config.name

    async def log_metrics(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        print(f"step {step}: {metrics}")

    async def log_physics_metrics(self, metrics: Any, step: int | None = None) -> None:
        print(f"physics {type(metrics).__name__}")

    async def log_parameters(self, params: dict[str, Any]) -> None:
        print(f"params {params}")

    async def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        print(f"artifact {local_path}")

    async def log_model(self, model: Any, model_name: str, physics_metadata: Any = None) -> None:
        print(f"model {model_name}")

    async def end(self, status: str = "completed") -> None:
        print(f"end {status}")


tracker = ExperimentTracker(default_backend="console")
tracker.register_backend("console", ConsoleExperiment)


async def console_run() -> None:
    config = ExperimentConfig(
        name="console_demo", physics_domain=PhysicsDomain.NEURAL_OPERATORS, framework=Framework.JAX
    )
    experiment = await tracker.create_experiment(config)
    await experiment.start()
    await experiment.log_metrics({"loss": 0.5}, step=0)
    await experiment.end()


asyncio.run(console_run())
```

## Capability declaration

`MLOPS_CAPABILITIES` declares `mlops:ExperimentTracker` as `UNSUPPORTED`: the tracker
publishes uncertainty metrics and computes none. The declaration is collected into
`opifex.capabilities.UQ_CAPABILITIES`; importing the package registers nothing.

## See Also

- [Platform API](platform.md): Model registry and versioning
- [Training API](training.md): Training infrastructure
- [Deployment API](deployment.md): Model serving
- [Benchmarking API](benchmarking.md): Performance benchmarking
