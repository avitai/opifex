# Probabilistic Numerics

## Overview

Probabilistic numerics represents a paradigm shift in scientific computing that treats numerical computation as a statistical inference problem. Instead of providing point estimates, probabilistic numerical methods quantify uncertainty in computational results, enabling more robust decision-making and better understanding of numerical errors.

The Opifex probabilistic framework provides implementations of Bayesian neural network layers, variational inference, MCMC sampling (via BlackJAX), uncertainty quantification, physics-informed priors, and calibration tools -- enabling principled uncertainty-aware scientific computing.

## Theoretical Foundation

### Probabilistic Perspective on Computation

Traditional numerical methods provide deterministic outputs, but probabilistic numerics acknowledges that:

1. **Finite Precision**: All computations involve approximations
2. **Model Uncertainty**: Our mathematical models are uncertain
3. **Data Uncertainty**: Measurements contain noise
4. **Computational Uncertainty**: Numerical algorithms introduce errors

### Bayesian Framework

Probabilistic numerics uses Bayesian inference to propagate uncertainty:

$$p(\text{solution} | \text{data}, \text{model}) = \frac{p(\text{data} | \text{solution}, \text{model}) p(\text{solution} | \text{model})}{p(\text{data} | \text{model})}$$

where:

- $p(\text{solution} | \text{model})$ is the prior belief about the solution
- $p(\text{data} | \text{solution}, \text{model})$ is the likelihood of observations
- $p(\text{solution} | \text{data}, \text{model})$ is the posterior distribution over solutions

### Uncertainty Types

Probabilistic numerics distinguishes between:

- **Aleatoric Uncertainty**: Inherent randomness in the system (data noise)
- **Epistemic Uncertainty**: Uncertainty due to limited knowledge (model uncertainty)
- **Computational Uncertainty**: Uncertainty from numerical approximations

## Core Probabilistic Components

### 1. Bayesian Neural Network Layers

Opifex provides `BayesianLinear` -- a variational diagonal-Gaussian dense layer with learnable weight distributions for epistemic uncertainty estimation:

```python
from opifex.uncertainty import BayesianLinear
import flax.nnx as nnx
import jax
import jax.numpy as jnp

rngs = nnx.Rngs(42)

# Create a Bayesian linear layer
layer = BayesianLinear(
    in_features=10,
    out_features=64,
    prior_std=1.0,
    rngs=rngs,
)

# Forward pass with weight sampling. Caller owns the RNG — pass an
# ``nnx.Rngs`` (which advances its ``posterior`` stream across calls)
# or an explicit ``jax.Array`` key. Mode follows the canonical
# ``nnx.Dropout`` convention: per-call ``deterministic`` overrides the
# module's ``self.deterministic`` attribute set by the NNX inference
# toggle.
x = jax.random.normal(jax.random.PRNGKey(0), (32, 10))
sample_rngs = nnx.Rngs(posterior=0)
output_sampled = layer(x, deterministic=False, rngs=sample_rngs)

# Forward pass at the posterior mean (no sampling).
output_mean = layer(x, deterministic=True)
```

Each `BayesianLinear` maintains variational parameters (`weight_mean`, `weight_logvar`, `bias_mean`, `bias_logvar`) and samples weights via the reparameterization trick when non-deterministic. `BayesianLinear.kl_divergence()` returns the closed-form diagonal Gaussian KL between the posterior and the prior, summed across weight and bias parameters.

For modules built on top of these layers (`ProbabilisticPINN`, `UncertaintyQuantificationNeuralOperator`, etc.) the shared platform exposes built-in objectives — `model.loss_components(batch, *, rngs, objective)` and `model.negative_elbo(batch, *, rngs, objective)` both return a `UQLossComponents` (data, physics_residual, boundary, regularization, kl) computed from an `ObjectiveConfig`. Prefer these over assembling `data_loss + kl_weight * kl_divergence()` by hand.

### 2. Amortized Variational Framework

The `AmortizedVariationalFramework` wraps any Flax NNX model to add amortized uncertainty estimation:

```python
from opifex.neural.bayesian import (
    AmortizedVariationalFramework,
    VariationalConfig,
    PriorConfig,
)
from opifex.neural.base import StandardMLP
import flax.nnx as nnx

rngs = nnx.Rngs(42)

# Create a base model
base_model = StandardMLP(
    layer_sizes=[10, 64, 64, 1],
    activation="gelu",
    rngs=rngs,
)

# Wrap with variational framework
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

The framework includes a `MeanFieldGaussian` posterior and an `UncertaintyEncoder` that maps inputs to uncertainty estimates.

### 3. BlackJAX MCMC Backend

For full Bayesian posterior sampling, Opifex provides the `BlackJAXBackend`
inference backend, which delegates to the BlackJAX HMC / NUTS / MALA
samplers:

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

The backend conforms to `InferenceBackendProtocol`. Unsupported sampler
families (SGLD, SGHMC, SMC, ADVI, Pathfinder) raise `UnsupportedBackendError`
until the upstream BlackJAX adapter grows a wrapper for them.

### 4. Uncertainty Quantification Utilities

The `opifex.uncertainty.aggregators` module provides tools for decomposing and analyzing uncertainty:

```python
from opifex.neural.bayesian import (
    UncertaintyQuantifier,
    EpistemicUncertainty,
    AleatoricUncertainty,
    UncertaintyComponents,
    CalibrationMetrics,
)
from opifex.uncertainty import BayesianLinear
from flax import nnx
import jax
import jax.numpy as jnp

# Build a small Bayesian model and gather Monte Carlo samples.
rngs = nnx.Rngs(42)
model = BayesianLinear(in_features=4, out_features=1, rngs=rngs)
x = jax.random.normal(jax.random.PRNGKey(0), (8, 4))

# predictions shape: (num_samples, batch_size, output_dim)
sample_rngs = nnx.Rngs(posterior=0)
predictions = jnp.stack([model(x, rngs=sample_rngs) for _ in range(100)])

# Compute epistemic uncertainty (model uncertainty)
epistemic_var = EpistemicUncertainty.compute_variance(predictions)

# Mean prediction
mean_prediction = jnp.mean(predictions, axis=0)

# Total uncertainty
total_std = jnp.std(predictions, axis=0)
print(f"Prediction: {mean_prediction[0]} +/- {total_std[0]}")
```

### 5. Physics-Informed Priors

The `PhysicsInformedPriors` class enforces conservation laws and boundary conditions through learnable constraint weights:

```python
from opifex.neural.bayesian import PhysicsInformedPriors
import flax.nnx as nnx
import jax
import jax.numpy as jnp

priors = PhysicsInformedPriors(
    conservation_laws=["energy", "momentum"],
    boundary_conditions=["dirichlet"],
    penalty_weight=1.0,
    rngs=nnx.Rngs(42),
)

# Apply physics constraints to sampled parameters
unconstrained_params = jax.random.normal(jax.random.PRNGKey(0), (16,))
constrained_params = priors.apply_constraints(unconstrained_params)
```

Additional prior classes are available:

- `ConservationLawPriors` -- specialized conservation law enforcement
- `DomainSpecificPriors` -- domain-adapted prior distributions
- `HierarchicalBayesianFramework` -- hierarchical prior structures
- `PhysicsAwareUncertaintyPropagation` -- propagates uncertainty through physics constraints

### 6. Calibration Tools

Opifex provides calibration methods to ensure uncertainty estimates are well-calibrated:

```python
from opifex.neural.bayesian import (
    CalibrationTools,
    TemperatureScaling,
    PlattScaling,
    IsotonicRegression,
    ConformalPrediction,
)

# Temperature scaling for calibration
calibrator = TemperatureScaling(rngs=nnx.Rngs(42))

# Conformal prediction for distribution-free coverage guarantees.
# ConformalPredictor wraps any point predictor (PINN, neural operator, etc.).
from opifex.neural.bayesian import ConformalPredictor, ConformalConfig
from opifex.neural.base import StandardMLP

point_model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(0))
conformal = ConformalPredictor(model=point_model, config=ConformalConfig())
```

### 7. Planned Components

The following are planned but not yet implemented:

- **Gaussian Processes**: Non-parametric Bayesian models for function approximation
- **Bayesian Optimization**: Efficient optimization of expensive black-box functions
- **Stochastic Differential Equations**: Neural networks for stochastic dynamics
- **Probabilistic Solvers**: Bayesian approaches to numerical integration

## Scientific Applications

### Physics-Informed Probabilistic Models

Combine physical laws with probabilistic modeling using `PhysicsInformedLoss` and Bayesian layers:

```python
from opifex.core.physics.losses import PhysicsInformedLoss, PhysicsLossConfig
from opifex.uncertainty import BayesianLinear
from opifex.neural.base import StandardMLP
import flax.nnx as nnx
import jax.numpy as jnp

rngs = nnx.Rngs(42)

# Create a model with Bayesian layers for uncertainty
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

# Physics-informed loss for heat equation
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

### Probabilistic Neural Operators

For uncertainty-aware neural operators, wrap an FNO with the `AmortizedVariationalFramework`:

```python
from opifex.neural.operators.fno import FourierNeuralOperator
from opifex.neural.bayesian import (
    AmortizedVariationalFramework,
    PriorConfig,
    VariationalConfig,
)
import flax.nnx as nnx

rngs = nnx.Rngs(42)

# Create base FNO
fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    modes=16,
    num_layers=4,
    rngs=rngs,
)

# Wrap with variational framework for uncertainty
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

## Best Practices

### Model Selection Guidelines

```python
# Decision guide for choosing probabilistic methods:
model_selection_criteria = {
    "uncertainty_type": {
        "aleatoric_only": "Use heteroscedastic regression",
        "epistemic_only": "Use AmortizedVariationalFramework or BlackJAXBackend",
        "both": "Use BayesianLinear-based models or deep ensembles",
    },
    "computational_budget": {
        "low": "Use single model with TemperatureScaling calibration",
        "medium": "Use AmortizedVariationalFramework with num_samples=10",
        "high": "Use BlackJAXBackend with NUTS sampling",
    },
    "data_size": {
        "small": "Use PhysicsInformedPriors with strong constraints",
        "medium": "Use AmortizedVariationalFramework",
        "large": "Use ensemble methods or variational inference",
    },
}
```

### Calibration Workflow

After training a probabilistic model, always check calibration:

1. Compute predictions with uncertainty on a held-out set
2. Use `CalibrationMetrics` to assess expected calibration error
3. Apply `TemperatureScaling` or `ConformalPredictor` to improve calibration
4. Verify coverage of confidence intervals matches nominal level

## Future Directions

### Emerging Methods

- **Neural Differential Equations**: Probabilistic neural ODEs and uncertainty in neural SDEs
- **Automated Uncertainty**: Self-calibrating models and adaptive uncertainty estimation
- **Scalable Inference**: Distributed Bayesian inference

### Planned Enhancements

- **Short-term**: GPU-accelerated MCMC sampling improvements, better calibration methods
- **Medium-term**: Gaussian process integration, causal uncertainty quantification
- **Long-term**: Universal uncertainty quantification, automated probabilistic modeling

## References

1. Hennig, P., Osborne, M. A., & Girolami, M. "Probabilistic numerics and uncertainty in computations." Proceedings of the Royal Society A 471, 20150142 (2015).
2. Gal, Y., & Ghahramani, Z. "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning." ICML 2016.
3. Lakshminarayanan, B., Pritzel, A., & Blundell, C. "Simple and scalable predictive uncertainty estimation using deep ensembles." NIPS 2017.
4. Blundell, C., et al. "Weight uncertainty in neural networks." ICML 2015.
5. Hoffman, M. D., & Gelman, A. "The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." JMLR 15, 1593-1623 (2014).

## See Also

- [Bayesian Neural Networks](../user-guide/neural-networks.md#bayesian-neural-networks) - Bayesian neural network architectures
- [Neural Network Training](../user-guide/training.md) - Training infrastructure with uncertainty quantification
- [Optimization Methods](../user-guide/optimization.md) - Optimization techniques
- [Bayesian Networks API](../api/neural.md#bayesian-networks) - API documentation
