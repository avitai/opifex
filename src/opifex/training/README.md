# Opifex Training

The unified training loop lives in **`opifex.core.training.Trainer`** — an `nnx.Module` that uses
`nnx.Optimizer` (automatic optax-state management), supports physics-informed losses, boundary /
constraint / conservation configs, callbacks/hooks, checkpointing, and distributed sharding. Use
it for supervised, PINN, and neural-operator training.

```python
from opifex.core.training import Trainer
from opifex.core.training.config import TrainingConfig

trainer = Trainer(model, TrainingConfig(num_epochs=2000, batch_size=64, learning_rate=1e-3))
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
- **`_uq_capabilities.py`** — UQ capability declarations for the above, registered in the
  `UQRegistry`.

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
