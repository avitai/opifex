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
| `opifex.uncertainty.inference_backends` | Backend protocol + base result/spec/diagnostics containers (`InferenceBackendProtocol`, `BackendResult`, `BackendDiagnostics`, `InferenceBackendSpec`, `UnsupportedBackendError`), `BlackJAXBackend` for HMC / NUTS / MALA posterior sampling, `BackendRouter` for Artifex-first family routing, and `OptionalBackendSpec` declarations for TFP / bijx / FlowJAX / Bayeux / NumPyro / GPJax / sbiax / flowMC / oryx / traceax / matfree / kfac-jax. |
| `opifex.uncertainty.distributions` | `ArtifexDistributionAdapter`, `DistrAxAdapter`, and the `from_distribution(...)` dispatcher that wraps a backend distribution into a `PredictiveDistribution`. |
| `opifex.uncertainty.adapters` | Distribution / model-uncertainty adapter protocols + concrete adapters. Protocols: `DistributionAdapterProtocol`, `ModelUncertaintyAdapterProtocol`. Concrete adapters: `ModelUncertaintyAdapter` (deterministic), `DeepEnsembleAdapter`, `MCDropoutAdapter`. Fitted-state pytrees: `DeepEnsembleState`, `SnapshotEnsembleState`, `SWAGState`, `BatchEnsembleState`, `MCDropoutState`. Capability spec dataclasses: `DistributionAdapterSpec`, `OperatorAdapterSpec`, the FNO operator adapter family (`FNOConformalAdapterSpec`, `FNODeepEnsembleAdapterSpec`, `FNOMCDropoutAdapterSpec`), the DeepONet operator adapter family (`DeepONetConformalAdapterSpec`, `DeepONetDeepEnsembleAdapterSpec`, `DeepONetMCDropoutAdapterSpec`), and deferred backend specs (`BayesianLastLayerAdapterSpec`, `LaplaceAdapterSpec`, `SNGPAdapterSpec`, `VBLLAdapterSpec`, `SnapshotEnsembleAdapterSpec`, `SWAGAdapterSpec`, `BatchEnsembleAdapterSpec`, `DUEAdapterSpec`, `TestTimeAugmentationAdapterSpec`) — each raises an actionable `NotImplementedError` from `.wrap()` until backends land. |
| `opifex.uncertainty.likelihoods` | Backend-neutral log-likelihoods: Gaussian, heteroscedastic Gaussian, Laplace, Student-t, mixture. |
| `opifex.uncertainty.priors` | Diagonal-Gaussian log prior. |
| `opifex.uncertainty.losses` | Uncertainty-aware loss functions: `PointwiseQuantileLoss` (pinball loss for UQNO residual-quantile training, mirrors `neuralop.losses.data_losses.PointwiseQuantileLoss`). |
| `opifex.uncertainty.metrics` | Ensemble + interval scoring metrics with no canonical CalibraX home: `predictive_entropy(ensemble_probabilities)` (Gal & Ghahramani 2016), `mutual_information(ensemble_probabilities)` (BALD epistemic decomposition, Houlsby et al. 2011), `interval_score`/`winkler_score(lower, upper, targets, alpha)` (Gneiting & Raftery 2007 strictly proper rule). Pure jax.Array kernels; no forward re-exports — Brier/ECE/pinball/Gaussian-NLL stay in `opifex.uncertainty.calibration`. |
| `opifex.uncertainty.calibration` | Calibration metrics and calibrators. Metrics: `gaussian_nll`, `picp`, `mpiw`, `regression_calibration_error` (Opifex-local), plus thin wrappers around CalibraX functionals: `brier_score`, `expected_calibration_error`, `pinball_loss`. Calibrator: `TemperatureScaling` + `TemperatureScalingState` (Guo et al. 2017 temperature scaling for multiclass logits; single-scalar L-BFGS fit on validation NLL). |
| `opifex.uncertainty.conformal` | Conformal prediction subsystem. Score helpers: `absolute_residual_score` (Lei et al. 2018), `cqr_score` (Romano, Patterson, Candes 2019, arXiv:1905.03222), `conformal_quantile` (finite-sample `ceil((n+1)(1-α))/n` rank). Regression calibrators: `SplitConformalRegressor` + `SplitConformalState`, `ConformalizedQuantileRegressor` + `CQRState`, `GroupedSplitConformalRegressor` + `GroupedSplitConformalState` (return `PredictionInterval`). Classification scores and calibrator: `lac_score` (Sadinle et al. 2019), `aps_score` (Romano et al. 2020), `raps_score` (Angelopoulos et al. 2021, arXiv:2009.14193), `aps_prediction_set`, `LACConformalClassifier` + `LACConformalState` (return `PredictionSet`). Cross-conformal / weighted: `jackknife_plus_intervals`, `cv_plus_intervals` (Barber et al. 2021 arXiv:1905.02928), `weighted_conformal_quantile`, `weighted_split_conformal_intervals` (Tibshirani et al. 2019 arXiv:1904.06019). Time-series: `EnbPIState` + `enbpi_update`/`enbpi_predict` (Xu & Xie 2021 arXiv:2010.09107), `AdaptiveConformalState` + `aci_update`/`aci_metadata` (Gibbs & Candès 2021 arXiv:2106.00170). Field/function-space: `field_l2_score`, `field_linf_score`, `field_h1_score`, `FieldSplitConformalRegressor` + `FieldSplitConformalState` with explicit norm over spatial axes. Risk control: `RiskControlConfig` (Pattern A), `RiskControllerState` (Pattern B), `hoeffding_upper_bound`, `rcps_threshold_kernel`, `select_threshold_rcps`, `bootstrap_threshold_ci` — RCPS / Learn-then-Test threshold selection (Bates et al. 2021 arXiv:2101.02703, Angelopoulos et al. 2022 arXiv:2110.01052) with `calibrax.statistics.analyzer.StatisticalAnalyzer.bootstrap_ci` reused for CI reporting. Exchangeability diagnostic: `check_exchangeability`, `ks_two_sample_pvalue`, `ExchangeabilityReport` (Vovk et al. 2005 §2.4). Numerical core for split/CQR/LAC/APS/RAPS aligned with the canonical `aangelopoulos/conformal-prediction` reference notebooks (`'higher'` interpolation rule in `conformal_quantile`). |

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

# Posterior-mean mode (no sampling). Mode resolution follows the
# canonical ``nnx.Dropout`` convention: per-call ``deterministic``
# overrides ``self.deterministic`` set by the NNX inference toggle.
x = jnp.ones((2, 4))
mean_output = layer(x, deterministic=True)

# Stochastic mode — caller owns the RNG
rngs = nnx.Rngs(posterior=42)
sample = layer(x, deterministic=False, rngs=rngs)
also_sample = layer(x, deterministic=False, rngs=rngs)  # different (stream advanced)

# Or pass an explicit JAX key
import jax
key = jax.random.PRNGKey(7)
deterministic_a = layer(x, deterministic=False, rngs=key)
deterministic_b = layer(x, deterministic=False, rngs=key)  # same (same key)

# KL divergence — used by the shared `loss_components` / `negative_elbo`
# objectives on modules built atop these layers (e.g. ProbabilisticPINN,
# UncertaintyQuantificationNeuralOperator). Prefer those built-in
# objectives over assembling ``data_loss + kl_weight * kl`` by hand.
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

## Inference Backends

### `BackendRouter`

Selects the highest-priority available backend for a given family
(`"flow"` / `"sampler"` / `"distribution"`) using **Artifex-first**
resolution order. Optional backends remain available by name but raise
`ImportError` on instantiation when their package is not installed.

```python
from opifex.uncertainty import BackendRouter

router = BackendRouter()
sampler_spec = router.resolve("sampler")               # default = BlackJAX
flow_spec = router.resolve("flow")                     # default = Artifex RealNVP
dist_spec = router.resolve("distribution")             # default = Artifex Distribution
print(sampler_spec.name, flow_spec.name, dist_spec.name)
```

Listing all registered specs for a family:

```python
from opifex.uncertainty import BackendRouter

router = BackendRouter()
for spec in router.available("sampler"):
    status = "installed" if spec.probe() else "needs install"
    print(f"  {spec.name} ({spec.source_package}) — {status}")
```

### `BlackJAXBackend`

Thin adapter over Artifex's BlackJAX HMC / NUTS / MALA samplers conforming
to `InferenceBackendProtocol`. Supports `method` in
`{"hmc", "nuts", "mala"}`; other sampler families raise
`UnsupportedBackendError`.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.uncertainty import BlackJAXBackend


def log_density(theta):
    return -0.5 * jnp.sum(theta * theta)


backend = BlackJAXBackend(
    target_log_prob=log_density,
    init_state=jnp.zeros(10),
    n_samples=1000,
    n_burnin=500,
    method="nuts",
    step_size=0.1,
)
result = backend.fit(log_density, rngs=nnx.Rngs(sample=0))
samples = result.sampler_state             # (n_samples, ...)
diagnostics = result.diagnostics           # BackendDiagnostics

# Convert posterior samples into a predictive distribution.
predictive = backend.predict_distribution(jnp.zeros((4, 10)), rngs=nnx.Rngs(sample=1))
```

## Distribution Adapters

```python
from opifex.uncertainty import from_distribution
from artifex.generative_models.core.distributions.continuous import Normal
from flax import nnx

# Wrap an Artifex distribution.
dist = Normal(loc=jnp.zeros(3), scale=jnp.ones(3), rngs=nnx.Rngs(0))
predictive = from_distribution(dist)       # PredictiveDistribution

# Distrax-like objects (anything exposing sample/log_prob/mean/variance) are
# accepted as a secondary fallback.
```

`from_distribution` resolves Artifex `Distribution` first, then falls back
to Distrax-like objects. Unsupported objects raise `TypeError`.

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
import jax.numpy as jnp
from flax import nnx
from opifex.uncertainty import (
    BayesianLinear,
    ObjectiveConfig,
    UQLossComponents,
)

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

# Stand-in loss tensors from your training step.
data_loss = jnp.array(0.5)
physics_loss = jnp.array(0.3)
layer = BayesianLinear(in_features=4, out_features=3, rngs=nnx.Rngs(0))

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
import jax.numpy as jnp
from opifex.uncertainty.kernels.bayesian import diagonal_gaussian_kl

mean = jnp.zeros(4)
logvar = jnp.zeros(4)

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
