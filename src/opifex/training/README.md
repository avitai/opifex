# Opifex Training

The unified training loop lives in **`opifex.core.training.Trainer`** — an `nnx.Module` that uses
`nnx.Optimizer` (automatic optax-state management), supports physics-informed losses, boundary /
constraint / conservation configs, callbacks/hooks, checkpointing, and distributed sharding. Use
it for supervised, PINN, and neural-operator training.

```python
import jax
import jax.numpy as jnp
from flax import nnx

from opifex.core.training import Trainer
from opifex.core.training.config import OptimizationConfig, TrainingConfig

x = jnp.linspace(-1.0, 1.0, 256)[:, None]
y = jnp.sin(3.0 * x)
x_train, y_train, x_val, y_val = x[::2], y[::2], x[1::2], y[1::2]


class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.hidden = nnx.Linear(1, 32, rngs=rngs)
        self.out = nnx.Linear(32, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.out(nnx.tanh(self.hidden(x)))


model = MLP(nnx.Rngs(0))
config = TrainingConfig(
    num_epochs=200, batch_size=64, optimization_config=OptimizationConfig(learning_rate=1e-2)
)
trainer = Trainer(model, config)
trained_model, history = trainer.fit((x_train, y_train), val_data=(x_val, y_val))
```

## What lives in this package

- **`uncertainty_trainers.py`** — uncertainty-guided helpers used *alongside* `Trainer` (these are
  sample-selection / uncertainty-propagation utilities, not standalone training loops):
  - **`UncertaintyGuidedTrainer`** — select the most uncertain pool samples and compute adaptive
    per-sample loss weights.
  - **`MultiFidelityUncertaintyTrainer`** — propagate uncertainty across model fidelities
    (Kennedy–O'Hagan weighting).
  - **`ActiveUncertaintyLearner`** — acquire informative samples (delegates to
    `opifex.uncertainty.active.acquisition`).
- **`_uq_capabilities.py`** — UQ capability declarations for the above, collected into
  `opifex.capabilities.UQ_CAPABILITIES`.

Physics-loss composition (`PhysicsInformedLoss`, `AdaptiveWeightScheduler`,
`ConservationLawEnforcer`, PDE residual computers) lives in **`opifex.core.physics.losses`** and is
consumed by `Trainer`.

## Atomistic / molecular potentials

For machine-learning interatomic potentials, assemble a backbone (SchNet / PaiNN / NequIP) into an
`AtomisticModel` (`opifex.neural.atomistic`) with energy / forces / stress heads, then train it with
the standard `Trainer`. See the
[Atomistic Potentials guide](../../../docs/methods/atomistic-potentials.md).

## Integration with other packages

- **[Core Package](../core/README.md)** — the unified `Trainer`, physics losses, and configs.
- **[Neural Package](../neural/README.md)** — `StandardMLP` plus the `neural.atomistic` potentials.
- **[Uncertainty Package](../uncertainty/README.md)** — the quantifiers/acquisition the utilities
  here build on.
- **[Optimization Package](../optimization/README.md)** — meta-optimization and the learn-to-optimize
  subsystem.
