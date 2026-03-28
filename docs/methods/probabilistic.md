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

Opifex provides `BayesianLayer` -- a variational Bayesian layer with learnable weight distributions for epistemic uncertainty estimation:

```python
from opifex.neural.bayesian.layers import BayesianLayer
import flax.nnx as nnx
import jax
import jax.numpy as jnp

rngs = nnx.Rngs(42)

# Create a Bayesian linear layer
layer = BayesianLayer(
    in_features=10,
    out_features=64,
    prior_std=1.0,
    rngs=rngs,
)

# Forward pass with weight sampling (training mode)
x = jax.random.normal(jax.random.PRNGKey(0), (32, 10))
output_sampled = layer(x, training=True, sample=True)

# Forward pass with mean weights (inference mode)
output_mean = layer(x, training=False, sample=False)
```

Each `BayesianLayer` maintains variational parameters (`weight_mean`, `weight_logvar`, `bias_mean`, `bias_logvar`) and samples weights via the reparameterization trick during training.

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
    config=config,
    rngs=rngs,
)
```

The framework includes a `MeanFieldGaussian` posterior and an `UncertaintyEncoder` that maps inputs to uncertainty estimates.

### 3. BlackJAX MCMC Integration

For full Bayesian posterior sampling, Opifex integrates with BlackJAX:

```python
from opifex.neural.bayesian import BlackJAXIntegration
from opifex.neural.base import StandardMLP
import flax.nnx as nnx

rngs = nnx.Rngs(42)

base_model = StandardMLP(
    layer_sizes=[10, 32, 32, 1],
    activation="gelu",
    rngs=rngs,
)

# Create MCMC sampler (NUTS, HMC, or MALA)
mcmc = BlackJAXIntegration(
    base_model=base_model,
    sampler_type="nuts",       # "nuts", "hmc", or "mala"
    num_warmup=1000,
    num_samples=1000,
    step_size=1e-3,
    rngs=rngs,
)
```

This provides NUTS, HMC, and MALA samplers for posterior inference over neural network parameters, enabling rigorous uncertainty quantification without variational approximations.

### 4. Uncertainty Quantification Utilities

The `opifex.neural.bayesian.uncertainty_quantification` module provides tools for decomposing and analyzing uncertainty:

```python
from opifex.neural.bayesian import (
    UncertaintyQuantifier,
    EpistemicUncertainty,
    AleatoricUncertainty,
    UncertaintyComponents,
    CalibrationMetrics,
)
import jax.numpy as jnp

# Given Monte Carlo samples from a Bayesian model
# predictions shape: (num_samples, batch_size, output_dim)
predictions = jnp.stack([model(x) for _ in range(100)])

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

priors = PhysicsInformedPriors(
    conservation_laws=["energy", "momentum"],
    boundary_conditions=["dirichlet"],
    penalty_weight=1.0,
    rngs=nnx.Rngs(42),
)

# Apply physics constraints to sampled parameters
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
calibrator = TemperatureScaling()

# Conformal prediction for distribution-free coverage guarantees
from opifex.neural.bayesian import ConformalPredictor, ConformalConfig

config = ConformalConfig()
conformal = ConformalPredictor(config=config)
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
from opifex.neural.bayesian.layers import BayesianLayer
from opifex.neural.base import StandardMLP
import flax.nnx as nnx
import jax.numpy as jnp

rngs = nnx.Rngs(42)

# Create a model with Bayesian layers for uncertainty
class BayesianPINN(nnx.Module):
    def __init__(self, *, rngs):
        self.layer1 = BayesianLayer(2, 64, rngs=rngs)
        self.layer2 = BayesianLayer(64, 64, rngs=rngs)
        self.layer3 = BayesianLayer(64, 1, rngs=rngs)

    def __call__(self, x):
        h = nnx.tanh(self.layer1(x))
        h = nnx.tanh(self.layer2(h))
        return self.layer3(h)

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
from opifex.neural.bayesian import AmortizedVariationalFramework, VariationalConfig
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
    config=config,
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
        "epistemic_only": "Use AmortizedVariationalFramework or BlackJAXIntegration",
        "both": "Use BayesianLayer-based models or deep ensembles",
    },
    "computational_budget": {
        "low": "Use single model with TemperatureScaling calibration",
        "medium": "Use AmortizedVariationalFramework with num_samples=10",
        "high": "Use BlackJAXIntegration with NUTS sampling",
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
