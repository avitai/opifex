# opifex.mlops

Experiment tracking for scientific machine learning. `ExperimentTracker` creates an
`Experiment` on a registered backend; `MLflowBackend` records physics-informed
metadata, domain metrics records, artifacts and Orbax model checkpoints in an
MLflow run through substrax's `MLFlowLogger` and `OrbaxCheckpointStore`.

```python
import asyncio
import tempfile

from flax import nnx

from opifex.mlops import ExperimentConfig, ExperimentTracker, Framework, PhysicsDomain, PhysicsMetadata
from opifex.neural.operators.fno.base import FourierNeuralOperator

model = FourierNeuralOperator(1, 1, 16, 8, 2, spatial_dims=1, rngs=nnx.Rngs(0))


async def main() -> None:
    config = ExperimentConfig(
        name="burgers_fno",
        physics_domain=PhysicsDomain.NEURAL_OPERATORS,
        framework=Framework.JAX,
        physics_metadata=PhysicsMetadata(pde_type="burgers", dimensionality=1),
        backend_config={"tracking_uri": f"file://{tempfile.mkdtemp()}"},
    )
    experiment = await ExperimentTracker().create_experiment(config)
    await experiment.start()
    await experiment.log_metrics({"loss": 0.1}, step=1)
    await experiment.log_model(model, "fno")
    await experiment.end()


asyncio.run(main())
```

| Module | Contents |
| --- | --- |
| `experiment` | `Experiment` (the async base class), `ExperimentConfig`, `PhysicsDomain`, `Framework`, `PhysicsMetadata`, the five metrics records and the `PhysicsMetrics` union |
| `tracker` | `ExperimentTracker`: backend registry and `create_experiment` |
| `records` | `physics_metadata_params` and `flatten_physics_metrics`, the pure shaping of records into parameters and metrics |
| `backends` | `MLflowBackend` and the `RunLogger` protocol it records through |
| `_uq_capabilities` | `MLOPS_CAPABILITIES`, collected into `opifex.capabilities.UQ_CAPABILITIES` |

The MLflow SDK ships in the `mlflow` extra (`uv add "opifex[mlflow]"`); importing
the package never imports it. The full reference is `docs/api/mlops.md`.
