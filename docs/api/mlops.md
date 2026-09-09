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

from opifex.mlops import (
    ExperimentConfig,
    ExperimentTracker,
    Framework,
    NeuralOperatorMetrics,
    PhysicsDomain,
    PhysicsMetadata,
)


async def main() -> None:
    config = ExperimentConfig(
        name="burgers_fno",
        physics_domain=PhysicsDomain.NEURAL_OPERATORS,
        framework=Framework.JAX,
        physics_metadata=PhysicsMetadata(
            pde_type="burgers",
            dimensionality=1,
            boundary_conditions=["periodic"],
            physical_constants={"viscosity": 0.01},
        ),
        random_seed=0,
    )
    experiment = await ExperimentTracker().create_experiment(config)

    run_id = await experiment.start()
    await experiment.log_parameters({"learning_rate": 1e-3, "modes": 12})
    for step, loss in enumerate(train()):
        await experiment.log_metrics({"loss": loss}, step=step)
    await experiment.log_physics_metrics(
        NeuralOperatorMetrics(
            train_loss=0.01,
            val_loss=0.012,
            spectral_accuracy=0.98,
            relative_l2_error=0.02,
            max_absolute_error=0.1,
            pde_residual=1e-3,
            conservation_error=1e-4,
            boundary_condition_error=1e-4,
            stability_measure=1.0,
            physical_consistency=0.99,
            inference_time_per_sample=2e-3,
            memory_usage_mb=512.0,
        )
    )
    await experiment.log_model(model, "fno")
    await experiment.end()
    print(run_id, experiment.get_experiment_url())


asyncio.run(main())
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
in the run's artifacts. `model` is an `nnx.Module`, a Flax `TrainState` or a state
dictionary; the checkpoint's metadata records the framework, the physics domain
and `physics_metadata`. Read it back with the same store:

```python
from substrax.checkpoint import OrbaxCheckpointStore

with OrbaxCheckpointStore(downloaded_artifact_dir, create=False) as store:
    state, metadata = store.restore(step=0)
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
tracker = ExperimentTracker(default_backend="console")
tracker.register_backend("console", ConsoleExperiment)
```

## Capability registry

`register_mlops_capabilities(registry)` records `mlops:ExperimentTracker` in a
`UQRegistry` as `UNSUPPORTED`: the tracker publishes uncertainty metrics and
computes none. Importing the package registers nothing.

## See Also

- [Platform API](platform.md): Model registry and versioning
- [Training API](training.md): Training infrastructure
- [Deployment API](deployment.md): Model serving
- [Benchmarking API](benchmarking.md): Performance benchmarking
