# Opifex Bayesian Neural Networks

This module hosts the Bayesian model surface of Opifex: variational training
helpers, calibration tooling, and amortised posterior frameworks. Probabilistic
PINNs and the trainable Bayesian dense / spectral-convolution layers live in
`opifex.uncertainty` so that operator and PINN code share a single
implementation. Everything below is currently exported and exercised by the
test suite.

## Module map

| Module | Public symbols |
|--------|----------------|
| `variational_framework` | `AmortizedVariationalFramework`, `VariationalConfig`, `PriorConfig`, `MeanFieldGaussian`, `UncertaintyEncoder` |
| `calibration_tools` | `CalibrationTools`, `PlattScaling`, `IsotonicRegression`, `TemperatureScaling` |
| `probabilistic_pinns` | `ProbabilisticPINN`, `MultiFidelityPINN`, `RobustPINNOptimizer`, factory helpers |

Trainable Bayesian layers (`BayesianLinear`, `BayesianSpectralConvolution`)
are re-exported from `opifex.uncertainty` and consumed directly by the PINN
and neural-operator code paths. Aggregators (`UncertaintyQuantifier`,
`EnhancedUncertaintyQuantifier`, `MultiSourceUncertaintyAggregator`) live in
`opifex.uncertainty.aggregators`. Physics-informed prior modules
(`PhysicsInformedPriors`, `ConservationLawPriors`, `DomainSpecificPriors`,
`HierarchicalBayesianFramework`, `PhysicsAwareUncertaintyPropagation`) live
in `opifex.uncertainty.priors_physics`. MCMC sampling is delivered by
`opifex.uncertainty.inference_backends.BlackJAXBackend`, a thin adapter over
sibling Artifex's HMC / NUTS / MALA wrappers.

## Bayesian dense layers

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.uncertainty.layers.bayesian import BayesianLinear

rngs = nnx.Rngs(0)
layer = BayesianLinear(in_features=8, out_features=16, prior_std=1.0, rngs=rngs)

x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))

# Posterior sample (uses the caller-owned RNG bundle).
sample_rngs = nnx.Rngs(posterior=2)
y_sampled = layer(x, deterministic=False, rngs=sample_rngs)

# Posterior mean (no sampling).
y_mean = layer(x, deterministic=True)

# Closed-form diagonal-Gaussian KL between posterior and prior.
kl = layer.kl_divergence()
```

`BayesianSpectralConvolution` follows the same call convention and powers the
trainable Bayesian path in `opifex.neural.operators` (FNO and operator
layers).

## Variational training surface

`opifex.neural.bayesian.probabilistic_pinns.ProbabilisticPINN` implements the
`VariationalModule` protocol from `opifex.uncertainty.protocols`. The
canonical training-time entry points are `loss_components` and
`negative_elbo` (both return a `UQLossComponents` driven by an
`ObjectiveConfig`), and the canonical inference-time entry point is
`predict_distribution` (returns a `PredictiveDistribution`).

```python
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.bayesian.probabilistic_pinns import ProbabilisticPINN
from opifex.uncertainty.objectives import ObjectiveConfig

rngs = nnx.Rngs(0)
pinn = ProbabilisticPINN(
    input_dim=2,
    output_dim=1,
    hidden_dims=(32, 32),
    use_bayesian=True,
    rngs=rngs,
)

objective = ObjectiveConfig(
    kl_weight=1e-3,
    dataset_size=512,
    physics_weight=1.0,
    data_weight=1.0,
    boundary_weight=1.0,
    initial_condition_weight=0.0,
    regularization_weight=0.0,
    calibration_weight=0.0,
    conformal_weight=0.0,
    pac_bayes_weight=0.0,
)

batch = {
    "x": jnp.zeros((16, 2)),
    "y": jnp.zeros((16, 1)),
}

components = pinn.loss_components(batch, rngs=nnx.Rngs(elbo=1), objective=objective)
elbo_components = pinn.negative_elbo(batch, rngs=nnx.Rngs(elbo=2), objective=objective)
predictive = pinn.predict_distribution(batch["x"], rngs=nnx.Rngs(predict=3))
```

`UQLossComponents` exposes per-term breakdowns (`data`, `physics_residual`,
`boundary`, `regularization`, `kl`, `negative_elbo`, ...) so trainers can log
diagnostics without recomputing weighted sums.

## Amortised variational framework

`AmortizedVariationalFramework` wraps any Flax NNX module and adds a
mean-field Gaussian posterior plus an input-conditioned uncertainty encoder.

```python
import flax.nnx as nnx
from opifex.neural.base import StandardMLP
from opifex.neural.bayesian import (
    AmortizedVariationalFramework,
    VariationalConfig,
    PriorConfig,
    MeanFieldGaussian,
    UncertaintyEncoder,
)

rngs = nnx.Rngs(0)

base_model = StandardMLP(
    layer_sizes=[10, 64, 64, 1],
    activation="gelu",
    rngs=rngs,
)

framework = AmortizedVariationalFramework(
    base_model=base_model,
    prior_config=PriorConfig(prior_scale=1.0),
    variational_config=VariationalConfig(
        input_dim=10,
        hidden_dims=(64, 32),
        num_samples=10,
        kl_weight=1.0,
    ),
    rngs=rngs,
)
```

`MeanFieldGaussian` and `UncertaintyEncoder` are also re-exported for use as
standalone NNX modules when the orchestration wrapper is not needed.

## Calibration tools

```python
import flax.nnx as nnx
from opifex.neural.bayesian import (
    CalibrationTools,
    PlattScaling,
    IsotonicRegression,
)
from opifex.uncertainty.calibration import TemperatureScaling
from opifex.uncertainty.conformal import SplitConformalRegressor

rngs = nnx.Rngs(0)

# Trainable per-domain calibration helpers (NNX modules).
calibration = CalibrationTools(rngs=rngs)
platt = PlattScaling(rngs=rngs)
isotonic = IsotonicRegression(n_bins=100, rngs=rngs)

# Caller-driven value-object calibrators from the platform layer.
temperature = TemperatureScaling()
# state = temperature.fit(logits=val_logits, targets=val_labels)
# probs = temperature.with_state(state).predict(test_logits)

conformal = SplitConformalRegressor(alpha=0.1)
# state = conformal.fit(predictions=val_preds, targets=val_targets)
# interval = conformal.with_state(state).predict(predictions=test_preds)
```

The CalibraX-backed `TemperatureScaling` / `SplitConformalRegressor` objects
are pure value objects: their `fit` returns a state and `with_state(state)`
returns a fresh callable. The neural NNX wrappers (`CalibrationTools`,
`PlattScaling`, `IsotonicRegression`) maintain learnable parameters so they
can be composed into a Flax model graph.

## MCMC via the BlackJAX backend

```python
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.uncertainty import BlackJAXBackend

def log_density(theta):
    return -0.5 * jnp.sum(theta * theta)

backend = BlackJAXBackend(
    target_log_prob=log_density,
    init_state=jnp.zeros(10),
    n_samples=1000,
    n_burnin=1000,
    method="nuts",
    step_size=1e-3,
)

result = backend.fit(log_density, rngs=nnx.Rngs(sample=0))
samples = result.sampler_state

predictive = backend.predict_distribution(jnp.zeros((4, 10)), rngs=nnx.Rngs(predict=1))
```

The backend conforms to `InferenceBackendProtocol`. Sampler families that
are not yet wrapped (SGLD, SGHMC, SMC) raise `UnsupportedBackendError` from
`opifex.uncertainty.inference_backends`. Mean-field VI, SVGD, and Pathfinder
are available as separate backends (`ADVIBackend`, `SVGDBackend`,
`PathfinderBackend`) and are routed through the same protocol.

## Practical guidance

- Always pass an explicit `nnx.Rngs` to Bayesian layers and modules; the
  module never owns its own seed.
- Prefer `loss_components` / `negative_elbo` over assembling
  `data_loss + kl_weight * kl_divergence()` by hand. The objective config
  centralises the weights and the loss container records the breakdown.
- Use `predict_distribution` (returns a `PredictiveDistribution`) for any
  inference-time uncertainty surface; downstream calibrators and conformal
  wrappers expect that container.
- Check calibration on a held-out set with metrics from
  `opifex.uncertainty.calibration` (`expected_calibration_error`,
  `regression_calibration_error`, `picp`, `mpiw`, ...) before reporting
  predictive intervals.

## Related modules

- [Neural operators](../operators/README.md) — operator learning with
  uncertainty surfaces.
- [Quantum neural networks](../quantum/README.md) — neural DFT components.
- [Core framework](../../core/README.md) — physics losses and shared
  mathematical primitives.
