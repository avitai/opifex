# `opifex.uncertainty` API Reference

The `opifex.uncertainty` package provides predictive-distribution containers,
loss-component aggregation, pure-JAX Bayesian kernels, trainable Bayesian
layers, structural protocols for UQ-aware modules, a capability registry, and
adapter/backend interfaces for downstream inference engines.

## Module layout

| Module | Purpose |
|---|---|
| `opifex.uncertainty.types` | Predictive value objects: `PredictiveDistribution`, `PredictionInterval`, `PredictionSet`, `PredictiveMode`. |
| `opifex.uncertainty.objectives` | `ObjectiveConfig` (loss weights, dataset metadata) and `UQLossComponents` (optimizer-facing loss decomposition); `scale_kl(...)` helper. |
| `opifex.uncertainty.kernels.bayesian` | Pure JAX Bayesian helpers: `diagonal_gaussian_kl`, `sample_diagonal_gaussian`. |
| `opifex.uncertainty.layers.bayesian` | Trainable NNX Bayesian layers: `BayesianLinear`, `BayesianSpectralConvolution`. |
| `opifex.uncertainty.protocols` | Structural protocols: `UncertaintyAwareModule`, `VariationalModule`, `Calibrator`, `Conformalizer`, `UncertaintyEstimator`. |
| `opifex.uncertainty.registry` | Capability metadata: `UQCapability`, `DefaultStrategy`, `UQRegistry`, `register_uq_capability`. |
| `opifex.uncertainty.inference_backends` | Backend protocol + base result/spec/diagnostics containers (`InferenceBackendProtocol`, `BackendResult`, `BackendDiagnostics`, `InferenceBackendSpec`, `UnsupportedBackendError`). |
| `opifex.uncertainty.adapters` | Distribution / model-uncertainty adapter protocols: `DistributionAdapterProtocol`, `ModelUncertaintyAdapterProtocol`, `DistributionAdapterSpec`. |
| `opifex.uncertainty.likelihoods` | Backend-neutral log-likelihoods: Gaussian, heteroscedastic Gaussian, Laplace, Student-t, mixture. |
| `opifex.uncertainty.priors` | Diagonal-Gaussian log prior. |

## Trainable Bayesian Layers

### `BayesianLinear`

Variational diagonal-Gaussian dense layer; weights and bias each carry a
`(mean, log-variance)` posterior. Sampling uses the reparameterization trick
and requires caller-owned RNG ownership at call time.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.uncertainty import BayesianLinear

layer = BayesianLinear(in_features=4, out_features=3, prior_std=1.0, rngs=nnx.Rngs(0))

# Deterministic mode (no sampling)
x = jnp.ones((2, 4))
mean_output = layer(x, sample=False)

# Stochastic mode — caller owns the RNG
rngs = nnx.Rngs(posterior=42)
sample = layer(x, sample=True, rngs=rngs)
also_sample = layer(x, sample=True, rngs=rngs)  # different (stream advanced)

# Or pass an explicit JAX key
import jax
key = jax.random.PRNGKey(7)
deterministic = layer(x, sample=True, rngs=key)
also_deterministic = layer(x, sample=True, rngs=key)  # same (same key)

# KL divergence (used in ELBO objectives)
kl = layer.kl_divergence()  # scalar; sums weight + bias KLs
```

### `BayesianSpectralConvolution`

Variational Fourier-spectral convolution with complex Gaussian weights (1D
and 2D modes). Implements the canonical Zongyi Li FNO spectral block
(Li et al. 2021, `arXiv:2010.08895`) with diagonal-Gaussian posteriors over
the complex weights:

* 1D — one weight tensor of shape `(out, in, modes[0])`.
* 2D — two weight tensors of shape `(out, in, modes[0], modes[1] // 2 + 1)`
  covering positive and negative H-axis low-frequency bands respectively.

Same RNG-safety contract as `BayesianLinear`.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.uncertainty import BayesianSpectralConvolution

# 2D spectral conv (Fourier neural operator-style block)
layer = BayesianSpectralConvolution(
    in_channels=2, out_channels=3, modes=(8, 8), prior_std=1.0, rngs=nnx.Rngs(0)
)

# Input: (batch, in_channels, height, width)
x = jnp.ones((1, 2, 32, 32))
mean_output = layer(x, sample=False)         # (1, 3, 32, 32)
sample = layer(x, sample=True, rngs=nnx.Rngs(posterior=0))

# KL over real + imaginary Fourier weights
kl = layer.kl_divergence()
```

## Predictive Value Objects

```python
import jax.numpy as jnp
from opifex.uncertainty import (
    PredictiveDistribution,
    PredictionInterval,
    PredictionSet,
    PredictiveMode,
)

mean = jnp.zeros((4,))
variance = jnp.ones((4,))
pd = PredictiveDistribution(mean=mean, variance=variance)
std = pd.std()  # sqrt(variance)
```

`PredictiveDistribution` is a JAX pytree (round-trips through `jit` / `vmap`)
with optional `samples`, `covariance`, `epistemic`, `aleatoric`,
`total_uncertainty`, `quantiles`, `interval`, and `prediction_set` fields, plus
tuple-of-pairs `metadata`.

## Objectives

```python
from opifex.uncertainty import ObjectiveConfig, UQLossComponents

config = ObjectiveConfig(
    kl_weight=1.0,
    dataset_size=1000,
    physics_weight=1.0,
    data_weight=1.0,
    boundary_weight=1.0,
    initial_condition_weight=1.0,
    regularization_weight=1.0,
    calibration_weight=1.0,
    conformal_weight=1.0,
    pac_bayes_weight=1.0,
)

components = UQLossComponents.from_components(
    config=config,
    data=data_loss,
    physics_residual=physics_loss,
    kl=layer.kl_divergence(),
)

scalar_for_optimizer = components.total
```

## Canonical KL helper

```python
from opifex.uncertainty.kernels.bayesian import diagonal_gaussian_kl

# Standard N(0, 1) prior delegates to Artifex gaussian_kl_divergence:
kl = diagonal_gaussian_kl(mean, logvar, prior_mean=0.0, prior_std=1.0)

# Parametric (prior_mean, prior_std) extension is provided by Opifex:
kl_parametric = diagonal_gaussian_kl(mean, logvar, prior_mean=2.0, prior_std=3.0)
```

## Capability Registry

```python
from opifex.uncertainty import (
    UQCapability,
    UQRegistry,
    DefaultStrategy,
    register_uq_capability,
)

cap = UQCapability(
    native_bayesian=True,
    native_nnx_module=True,
    supports_function_space=True,
    default_strategy=DefaultStrategy.BAYESIAN,
    source_package="opifex",
)

@register_uq_capability("my_model", cap)
class MyModel:
    pass

registry = UQRegistry()
assert registry.get("my_model") is cap
```
