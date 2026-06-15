# Probabilistic Numerics

## Overview

Probabilistic numerics treats numerical computation as a statistical inference
problem. Instead of producing point estimates, probabilistic numerical methods
quantify uncertainty in computational results, enabling more robust
decision-making and a clearer accounting of numerical error.

The Opifex probabilistic stack provides Bayesian neural network layers,
variational inference, MCMC sampling (via BlackJAX), uncertainty
quantification utilities, physics-informed priors, and calibration tools, all
exposed through the `opifex.uncertainty` and `opifex.neural.bayesian`
namespaces.

## Theoretical foundation

### Probabilistic perspective on computation

Traditional numerical methods provide deterministic outputs. Probabilistic
numerics acknowledges that:

1. **Finite precision** — every computation involves approximations.
2. **Model uncertainty** — our mathematical models are uncertain.
3. **Data uncertainty** — measurements contain noise.
4. **Computational uncertainty** — numerical algorithms introduce error.

### Bayesian framework

Probabilistic numerics propagates uncertainty through Bayes' rule:

$$p(\text{solution} \mid \text{data}, \text{model}) = \frac{p(\text{data} \mid \text{solution}, \text{model})\, p(\text{solution} \mid \text{model})}{p(\text{data} \mid \text{model})}$$

where $p(\text{solution} \mid \text{model})$ is the prior over solutions,
$p(\text{data} \mid \text{solution}, \text{model})$ is the likelihood, and
$p(\text{solution} \mid \text{data}, \text{model})$ is the posterior.

### Uncertainty types

Probabilistic numerics distinguishes between:

- **Aleatoric uncertainty** — inherent randomness in the system (data noise).
- **Epistemic uncertainty** — uncertainty due to limited knowledge.
- **Computational uncertainty** — uncertainty from numerical approximations.

## Core probabilistic components

### 1. Bayesian neural network layers

Opifex provides trainable diagonal-Gaussian variational layers. `BayesianLinear`
is the dense layer; `BayesianSpectralConvolution` is the spectral counterpart
that powers the Bayesian variant of the Fourier neural operator.

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.uncertainty.layers.bayesian import BayesianLinear, BayesianSpectralConvolution

rngs = nnx.Rngs(0)

layer = BayesianLinear(
    in_features=10,
    out_features=64,
    prior_std=1.0,
    rngs=rngs,
)

# Forward pass with weight sampling. The caller owns the RNG and passes
# either an ``nnx.Rngs`` bundle (which advances its ``posterior`` stream
# across calls) or an explicit ``jax.Array`` key. Mode follows the
# canonical ``nnx.Dropout`` convention: a per-call ``deterministic``
# overrides the module's stored attribute set by the NNX inference toggle.
x = jax.random.normal(jax.random.PRNGKey(0), (32, 10))
sample_rngs = nnx.Rngs(posterior=0)
output_sampled = layer(x, deterministic=False, rngs=sample_rngs)

# Forward pass at the posterior mean (no sampling).
output_mean = layer(x, deterministic=True)
```

Each `BayesianLinear` maintains the variational parameters
(`weight_mean`, `weight_logvar`, `bias_mean`, `bias_logvar`) and samples
weights via the reparameterisation trick when non-deterministic.
`BayesianLinear.kl_divergence()` returns the closed-form diagonal Gaussian
KL between the posterior and the prior, summed across weight and bias
parameters.

For modules built on top of these layers (`ProbabilisticPINN`,
`UncertaintyQuantificationNeuralOperator`, ...), the shared platform exposes
built-in objectives. `model.loss_components(batch, *, rngs, objective)` and
`model.negative_elbo(batch, *, rngs, objective)` both return a
`UQLossComponents` (data, physics_residual, boundary, regularization, kl)
computed from an `ObjectiveConfig`. Prefer these over assembling
`data_loss + kl_weight * kl_divergence()` by hand.

### 2. Amortized variational framework

`AmortizedVariationalFramework` wraps any Flax NNX model with a mean-field
Gaussian posterior and an input-conditioned uncertainty encoder.

```python
import flax.nnx as nnx
from opifex.neural.base import StandardMLP
from opifex.neural.bayesian import (
    AmortizedVariationalFramework,
    VariationalConfig,
    PriorConfig,
)

rngs = nnx.Rngs(0)

base_model = StandardMLP(
    layer_sizes=[10, 64, 64, 1],
    activation="gelu",
    rngs=rngs,
)

config = VariationalConfig(
    input_dim=10,
    hidden_dims=(64, 32),
    num_samples=10,
    kl_weight=1.0,
    temperature=1.0,
)

framework = AmortizedVariationalFramework(
    base_model=base_model,
    prior_config=PriorConfig(),
    variational_config=config,
    rngs=rngs,
)
```

The framework owns a `MeanFieldGaussian` posterior and an
`UncertaintyEncoder` that maps inputs to uncertainty estimates. Both modules
are re-exported from `opifex.neural.bayesian` for standalone use.

### 3. BlackJAX MCMC backend

For full Bayesian posterior sampling, Opifex provides `BlackJAXBackend`,
which delegates to the BlackJAX HMC / NUTS / MALA samplers.

```python
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.uncertainty import BlackJAXBackend

def log_density(theta):
    # Replace with the model + likelihood log-density of interest.
    return -0.5 * jnp.sum(theta * theta)

rngs = nnx.Rngs(sample=42)

backend = BlackJAXBackend(
    target_log_prob=log_density,
    init_state=jnp.zeros(10),
    n_samples=1000,
    n_burnin=1000,
    method="nuts",       # "nuts", "hmc", or "mala"
    step_size=1e-3,
)

result = backend.fit(log_density, rngs=rngs)
samples = result.sampler_state   # shape (n_samples, ...)

# Convert posterior samples into a PredictiveDistribution for downstream use.
predictive = backend.predict_distribution(jnp.zeros((4, 10)), rngs=rngs)
```

The backend conforms to `InferenceBackendProtocol`. Sampler families that
are not yet wrapped (SGLD, SGHMC, SMC) raise `UnsupportedBackendError` from
`opifex.uncertainty.inference_backends` until the upstream BlackJAX adapter
grows a wrapper for them.

### 4. Variational backends

The same protocol routes a mean-field VI backend, a Stein-gradient particle
backend, and a quasi-Newton variational backend. Each produces a
`PredictiveDistribution` through the canonical `predict_distribution` hook.

```python
import jax.numpy as jnp
from opifex.uncertainty.inference_backends import (
    ADVIBackend,
    SVGDBackend,
    PathfinderBackend,
    InferenceBackendProtocol,
    UnsupportedBackendError,
)

advi = ADVIBackend(init_state=jnp.zeros(10))
svgd = SVGDBackend(init_state=jnp.zeros(10))
pathfinder = PathfinderBackend(init_state=jnp.zeros(10))
```

### 5. Uncertainty quantification utilities

The `opifex.uncertainty.aggregators` module provides tools for decomposing
and analysing uncertainty.

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.uncertainty.layers.bayesian import BayesianLinear
from opifex.uncertainty.aggregators import (
    UncertaintyQuantifier,
    EpistemicUncertainty,
    AleatoricUncertainty,
    UncertaintyComponents,
    CalibrationMetrics,
)

# Build a small Bayesian model and gather Monte Carlo samples.
rngs = nnx.Rngs(0)
model = BayesianLinear(in_features=4, out_features=1, rngs=rngs)
x = jax.random.normal(jax.random.PRNGKey(0), (8, 4))

# predictions shape: (num_samples, batch_size, output_dim)
sample_rngs = nnx.Rngs(posterior=0)
predictions = jnp.stack([model(x, rngs=sample_rngs) for _ in range(100)])

# Compute epistemic uncertainty (model uncertainty).
epistemic_var = EpistemicUncertainty.compute_variance(predictions)

# Mean prediction.
mean_prediction = jnp.mean(predictions, axis=0)

# Total standard deviation across MC samples.
total_std = jnp.std(predictions, axis=0)
```

### 6. Physics-informed priors

`PhysicsInformedPriors` enforces conservation laws and boundary conditions
through learnable constraint weights.

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.uncertainty.priors_physics import PhysicsInformedPriors

priors = PhysicsInformedPriors(
    conservation_laws=["energy", "momentum"],
    boundary_conditions=["dirichlet"],
    penalty_weight=1.0,
    rngs=nnx.Rngs(0),
)

# Apply physics constraints to sampled parameters.
unconstrained_params = jax.random.normal(jax.random.PRNGKey(0), (16,))
constrained_params = priors.apply_constraints(unconstrained_params)
```

Additional prior classes are available:

- `ConservationLawPriors` — specialised conservation-law enforcement.
- `DomainSpecificPriors` — domain-adapted prior distributions.
- `HierarchicalBayesianFramework` — hierarchical prior structures.
- `PhysicsAwareUncertaintyPropagation` — propagates uncertainty through
  physics constraints.

### 7. Calibration tools

Opifex provides calibration helpers that operate on top of any model
producing a `PredictiveDistribution`.

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

# Trainable per-domain calibration NNX modules.
calibration = CalibrationTools(rngs=rngs)
platt = PlattScaling(rngs=rngs)
isotonic = IsotonicRegression(n_bins=100, rngs=rngs)

# Pure value-object calibrators backed by sibling CalibraX.
temperature = TemperatureScaling()
# state = temperature.fit(logits=val_logits, targets=val_labels)
# probs = temperature.with_state(state).predict(test_logits)

# Split-conformal regressor for distribution-free coverage guarantees.
regressor = SplitConformalRegressor(alpha=0.1)
# state = regressor.fit(predictions=val_preds, targets=val_targets)
# interval = regressor.with_state(state).predict(predictions=test_preds)
```

### 8. Planned components

The following are planned but not yet implemented:

- **Gaussian processes** — non-parametric Bayesian models for function
  approximation.
- **Bayesian optimisation** — efficient optimisation of expensive
  black-box functions.
- **Stochastic differential equations** — neural networks for stochastic
  dynamics.
- **Probabilistic solvers** — Bayesian approaches to numerical integration.

## Scientific applications

### Physics-informed probabilistic models

Combine physical laws with probabilistic modelling using
`PhysicsInformedLoss` and Bayesian layers.

```python
import flax.nnx as nnx
from opifex.core.physics.losses import PhysicsInformedLoss, PhysicsLossConfig
from opifex.uncertainty.layers.bayesian import BayesianLinear

rngs = nnx.Rngs(0)

class BayesianPINN(nnx.Module):
    def __init__(self, *, rngs):
        self.rngs = rngs
        self.layer1 = BayesianLinear(2, 64, rngs=rngs)
        self.layer2 = BayesianLinear(64, 64, rngs=rngs)
        self.layer3 = BayesianLinear(64, 1, rngs=rngs)

    def __call__(self, x):
        h = nnx.tanh(self.layer1(x, rngs=self.rngs))
        h = nnx.tanh(self.layer2(h, rngs=self.rngs))
        return self.layer3(h, rngs=self.rngs)

model = BayesianPINN(rngs=rngs)

# Physics-informed loss for the heat equation.
config = PhysicsLossConfig(
    data_loss_weight=10.0,
    physics_loss_weight=1.0,
    boundary_loss_weight=10.0,
)

physics_loss = PhysicsInformedLoss(
    config=config,
    equation_type="heat",
    domain_type="2d",
)
```

For a turnkey variational PINN that already implements
`loss_components` / `negative_elbo` / `predict_distribution`, use
`opifex.neural.bayesian.probabilistic_pinns.ProbabilisticPINN` directly.

### Probabilistic neural operators

For uncertainty-aware neural operators, wrap a Fourier operator with the
`AmortizedVariationalFramework`.

```python
import flax.nnx as nnx
from opifex.neural.operators.fno import FourierNeuralOperator
from opifex.neural.bayesian import (
    AmortizedVariationalFramework,
    PriorConfig,
    VariationalConfig,
)

rngs = nnx.Rngs(0)

fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    modes=16,
    num_layers=4,
    rngs=rngs,
)

config = VariationalConfig(
    input_dim=64,
    hidden_dims=(64, 32),
    num_samples=10,
    kl_weight=0.1,
)

prob_fno = AmortizedVariationalFramework(
    base_model=fno,
    prior_config=PriorConfig(),
    variational_config=config,
    rngs=rngs,
)
```

Alternatively, the Bayesian variant of an FNO can be assembled from
`BayesianSpectralConvolution` blocks directly, in which case the resulting
module exposes the platform `VariationalModule` surface
(`loss_components`, `negative_elbo`, `predict_distribution`,
`kl_divergence`).

## Best practices

### Choosing a method

| Need | Suggested entry point |
|------|------------------------|
| Aleatoric uncertainty only | Heteroscedastic regression head |
| Epistemic uncertainty only | `AmortizedVariationalFramework` or `BlackJAXBackend` |
| Both, dense models | `BayesianLinear`-based models or deep ensembles |
| Both, spectral operators | `BayesianSpectralConvolution`-based FNO |
| Distribution-free intervals | `SplitConformalRegressor` |

Computational budget matters: a single network with `TemperatureScaling` is
fast; `AmortizedVariationalFramework` with `num_samples=10` is the medium
tier; `BlackJAXBackend` with NUTS is the high-budget reference.

### Calibration workflow

After training a probabilistic model, always check calibration:

1. Compute predictions with uncertainty on a held-out set.
2. Use the metrics in `opifex.uncertainty.calibration`
   (`expected_calibration_error`, `regression_calibration_error`, `picp`,
   `mpiw`, `gaussian_nll`, `pinball_loss`) to assess reliability.
3. Apply `TemperatureScaling` or `SplitConformalRegressor` to improve
   calibration.
4. Verify coverage of confidence intervals matches the nominal level.

## Future directions

### Emerging methods

- **Neural differential equations** — probabilistic neural ODEs and
  uncertainty in neural SDEs.
- **Self-calibrating models** — adaptive uncertainty estimation.
- **Scalable inference** — distributed Bayesian inference.

### Planned enhancements

- GPU-accelerated MCMC sampling improvements and better calibration.
- Gaussian-process integration and causal uncertainty quantification.
- Automated probabilistic modelling pipelines.

## References

1. Hennig, P., Osborne, M. A., & Girolami, M. "Probabilistic numerics and
   uncertainty in computations." *Proceedings of the Royal Society A* 471,
   20150142 (2015).
2. Gal, Y., & Ghahramani, Z. "Dropout as a Bayesian approximation:
   Representing model uncertainty in deep learning." *ICML* 2016.
3. Lakshminarayanan, B., Pritzel, A., & Blundell, C. "Simple and scalable
   predictive uncertainty estimation using deep ensembles." *NIPS* 2017.
4. Blundell, C., et al. "Weight uncertainty in neural networks." *ICML*
   2015.
5. Hoffman, M. D., & Gelman, A. "The No-U-Turn sampler: adaptively setting
   path lengths in Hamiltonian Monte Carlo." *JMLR* 15, 1593-1623 (2014).

## See also

- [Bayesian neural networks](../user-guide/neural-networks.md#bayesian-neural-networks)
  — architecture guide.
- [Neural network training](../user-guide/training.md) — training
  infrastructure with uncertainty quantification.
- [Optimisation methods](../user-guide/optimization.md) — optimisation
  techniques.
- [Bayesian networks API](../api/neural.md#bayesian-networks) — API
  reference.
