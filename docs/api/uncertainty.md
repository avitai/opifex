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
| `opifex.uncertainty.adapters` | Distribution / model-uncertainty adapter protocols + concrete adapters. Protocols: `DistributionAdapterProtocol`, `ModelUncertaintyAdapterProtocol`. Concrete adapters: `ModelUncertaintyAdapter` (deterministic), `DeepEnsembleAdapter`, `MCDropoutAdapter`. Fitted-state pytrees: `DeepEnsembleState`, `SnapshotEnsembleState`, `SWAGState`, `BatchEnsembleState`, `MCDropoutState`. Capability spec dataclasses: `DistributionAdapterSpec`, `OperatorAdapterSpec`, the FNO operator adapter family (`FNOConformalAdapterSpec`, `FNODeepEnsembleAdapterSpec`, `FNOMCDropoutAdapterSpec`), the DeepONet operator adapter family (`DeepONetConformalAdapterSpec`, `DeepONetDeepEnsembleAdapterSpec`, `DeepONetMCDropoutAdapterSpec`), and deferred backend specs (`BayesianLastLayerAdapterSpec`, `SNGPAdapterSpec`, `VBLLAdapterSpec`, `SnapshotEnsembleAdapterSpec`, `SWAGAdapterSpec`, `BatchEnsembleAdapterSpec`, `DUEAdapterSpec`, `TestTimeAugmentationAdapterSpec`) — each raises an actionable `NotImplementedError` from `.wrap()` until backends land. The concrete diagonal-Laplace adapter (`LaplaceAdapterSpec` + `LaplaceState`) now lives in `opifex.uncertainty.curvature` alongside the curvature kernels it consumes. |
| `opifex.uncertainty.curvature` | Curvature primitives for second-order UQ: `hessian_vector_product`, `ggn_vector_product`, `empirical_fisher_diagonal`, `diagonal_laplace_posterior` + `DiagonalLaplacePosterior`. Diagonal-Laplace adapter: `LaplaceAdapterSpec` + `LaplaceState`. Linearised Neural Operator predictive: `linearized_neural_operator_posterior(model_fn=, laplace_posterior=, x=)` returning a `PredictiveDistribution` whose marginal variance is `diag(J Σ Jᵀ)` for `Σ = diag(1 / precision_diagonal)` (Magnani et al. 2024, arXiv:2406.04317). Pure-JAX; passes `jit` / `vmap` smokes. |
| `opifex.uncertainty.likelihoods` | Backend-neutral log-likelihoods: Gaussian, heteroscedastic Gaussian, Laplace, Student-t, mixture. |
| `opifex.uncertainty.priors` | Diagonal-Gaussian log prior. |
| `opifex.uncertainty.losses` | Uncertainty-aware loss functions: `PointwiseQuantileLoss` (pinball loss for UQNO residual-quantile training, mirrors `neuralop.losses.data_losses.PointwiseQuantileLoss`). |
| `opifex.uncertainty.metrics` | Ensemble + interval scoring metrics with no canonical CalibraX home: `predictive_entropy(ensemble_probabilities)` (Gal & Ghahramani 2016), `mutual_information(ensemble_probabilities)` (BALD epistemic decomposition, Houlsby et al. 2011), `interval_score`/`winkler_score(lower, upper, targets, alpha)` (Gneiting & Raftery 2007 strictly proper rule). Pure jax.Array kernels; no forward re-exports — Brier/ECE/pinball/Gaussian-NLL stay in `opifex.uncertainty.calibration`. |
| `opifex.uncertainty.forecasting_metrics` | Probabilistic forecast metrics: empirical `crps` (CalibraX-canonical formula), `fair_crps` (Ferro 2014; WeatherBenchX `CRPSEnsemble(fair=True)` reference), `energy_score` (Gneiting & Raftery 2007 §4.2), `rank_histogram` (Hamill 2001), `spread_skill_ratio` (Fortin et al. 2014), `pit_histogram` (Diebold-Gunther-Tay 1998), `ranked_probability_score` (Epstein 1969), `event_reliability` (Murphy 1973 Brier decomposition). |
| `opifex.uncertainty.scientific.domain_metrics` | Domain reliability metrics with the `DomainMetricSummary` value object: PINN (`physics_residual_coverage`, `boundary_condition_coverage`), neural operator (`spectral_coverage`), quantum chemistry (`chemical_accuracy_coverage` — caller-supplied tolerance, no hard-coded default), optimization / L2O (`regret_interval_summary`, `feasibility_coverage`), assimilation (`sensor_reliability_summary` — reduced χ²), parameter inference (`parameter_credible_interval_coverage`). Phase 8 deferred capabilities surface as explicit `UNSUPPORTED_LIKELIHOOD_FREE`, `UNSUPPORTED_ACTIVE_LEARNING`, `UNSUPPORTED_PAC_BAYES` `UQCapability` constants; Phase 8 Task 8.5 flips the `supports_*` flags and `default_strategy` when those backends ship. |
| `opifex.uncertainty.scientific.fields` | Canonical `FieldMetadata` (Pattern A) schema plus field/function-space UQ metrics: `spatial_calibration_error`, `function_space_l2_coverage`, `conservation_law_residual_summary`, `residual_uncertainty_alignment`. `FieldMetadata` is the canonical home referenced by `opifex.uncertainty.conformal.fields.FieldSplitConformalRegressor`. |
| `opifex.uncertainty.reports` | `UQReliabilityReport` — Pattern-B aggregated report carrying optional metric leaves from every UQ subsystem (calibration, conformal, forecasting, OOD, selective). `validate()` requires at least one populated metric; `to_dict()` returns a deterministic JSON-compatible payload. Failed exchangeability / shift / OOD warnings propagate verbatim into `metadata["assumption_warnings"]` and `metadata["assumption_status"]`. Data class + serializer only — never an evaluator. |
| `opifex.uncertainty.monitoring` | `MonitoringInputs` (Pattern A provenance container) + `build_reliability_report(inputs=, ...metrics)` builder that assembles a validated `UQReliabilityReport` from already-computed metrics. Validates the payload before returning so half-populated reports never propagate silently. |
| `opifex.uncertainty.selective` | Selective prediction: `risk_coverage_curve(confidences, errors)` returns the per-threshold ``(coverages, risks)`` arrays; `area_under_risk_coverage(confidences, errors)` returns the AURC scalar; `abstention_decision(confidences, threshold)` returns an `AbstentionDecision` value object with named `accepted_mask` / `rejected_mask` + metadata (Geifman & El-Yaniv 2017, arXiv:1705.08500). |
| `opifex.uncertainty.ood` | Out-of-distribution detection scores and residual-shift diagnostics: `max_softmax_probability` (Hendrycks & Gimpel 2017, arXiv:1610.02136), `fpr95`, `ShiftReport` + `residual_shift_diagnostic` (reuses `opifex.uncertainty.conformal.ks_two_sample_pvalue`). AUROC / AUPRC are NOT re-exported — import directly from `calibrax.metrics.functional.classification.{roc_auc, average_precision}`. Predictive entropy / mutual information OOD signals live in `opifex.uncertainty.metrics`. |
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

### `SolutionDistribution` (solver-side, per-field)

```python
import jax.numpy as jnp
from opifex.uncertainty.scientific import SolutionDistribution

sd = SolutionDistribution(
    mean={"u": jnp.zeros((64,)), "p": jnp.ones((64,))},
    epistemic={"u": jnp.full((64,), 0.1), "p": jnp.full((64,), 0.2)},
    aleatoric={"u": jnp.full((64,), 0.3), "p": jnp.full((64,), 0.4)},
    total_uncertainty={"u": jnp.full((64,), 0.4), "p": jnp.full((64,), 0.6)},
    metadata=(
        ("uncertainty_sources", ("numerical", "parameter")),
        ("spatial_axes", (0,)),
        ("function_space_norm", "L2"),
        ("covariance_form", "diag"),
    ),
)
sd.validate()  # public preflight; never called during pytree unflatten

# Project a single field back onto the canonical PredictiveDistribution
# contract so downstream calibration / conformal code consumes either
# container without solver-specific awareness.
pd_u = sd.as_predictive_distribution("u")
```

`SolutionDistribution` is the multi-field solver counterpart of
`PredictiveDistribution`. Every uncertainty leaf (`samples`, `variance`,
`covariance`, `epistemic`, `aleatoric`, `total_uncertainty`) is a
`dict[str, jax.Array]` keyed by field name; `metadata` is static aux_data and
must declare a tuple-of-strings `uncertainty_sources` drawn from
``{"numerical", "parameter", "observation", "model_discrepancy", "ensemble",
"calibration"}``. The same `_VARIANCE_RTOL` / `_VARIANCE_ATOL` tolerances as
`PredictiveDistribution` apply to per-field variance additivity, so a
`SolutionDistribution` round-tripped through `as_predictive_distribution(field)`
does not flap downstream additivity checks.

### `aggregate_solver_solutions` (solver-side Monte-Carlo / ensemble aggregation)

```python
from opifex.uncertainty.scientific import aggregate_solver_solutions

# Caller writes the explicit replay or ensemble loop.
replays = [solver.solve(problem, rngs=nnx.Rngs(seed)) for seed in range(num_samples)]
out = aggregate_solver_solutions(
    replays,
    quantiles=(0.05, 0.95),
    metadata=(("method", "monte_carlo_empirical"),),
)
mean_field_u = out.fields["u"]
band_lower = out.auxiliary_data["uq"]["quantiles"][0.05]["u"]
band_upper = out.auxiliary_data["uq"]["quantiles"][0.95]["u"]
```

Stacks per-field arrays across the sequence, reports the mean in
`Solution.fields`, and stores the full UQ payload (samples, ddof=1
variance, optional quantile bands, metadata) under
`auxiliary_data["uq"]`. Replaces four previous wrapper classes
(`BayesianWrapper` / `ConformalWrapper` / `EnsembleWrapper` /
`GenerativeWrapper`); deep ensemble, stochastic-replay, and
empirical-interval flows all reduce to this single call.

### `summarize_stacked_sample_solution` (generative-sampler aggregation)

```python
from opifex.uncertainty.scientific import summarize_stacked_sample_solution

# Generative base solver returns one Solution whose fields are stacked
# (num_samples, *field_shape) arrays — typically a Diffusion or
# Flow-Matching model from Artifex.
raw = generative_solver.solve(problem)
out = summarize_stacked_sample_solution(raw, quantiles=(0.05, 0.95))
```

Aggregates along the sample axis without overwriting the underlying
batch. Scalar fields pass through unchanged; non-scalar fields land in
`auxiliary_data["uq"]["samples"]` and a mean lands in
`Solution.fields[key]`. Replaces the `GenerativeWrapper` class.

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

## Protocols

The structural-typing protocols in `opifex.uncertainty.protocols` describe
what a model surface must offer to interoperate with the rest of the UQ
platform. None of them require inheritance — any class that exposes the
named methods passes ``isinstance(model, Protocol)`` via
``@runtime_checkable``.

```python
from opifex.uncertainty.protocols import (
    UncertaintyAwareModule,
    VariationalModule,
    UncertaintyEstimator,
    Calibrator,
    Conformalizer,
)
```

* `UncertaintyAwareModule` — exposes ``predict_distribution(x, *, rngs)
  -> PredictiveDistribution``; the minimum every UQ-aware model must
  implement so calibration / conformal code can drive it generically.
* `VariationalModule` — extends `UncertaintyAwareModule` with
  ``loss_components(batch, *, rngs, objective) -> UQLossComponents``,
  ``negative_elbo(batch, *, rngs, objective) -> UQLossComponents`` and
  ``kl_divergence() -> jax.Array``. Used by Bayesian layers and
  `ProbabilisticPINN`.
* `UncertaintyEstimator` — produces uncertainty arrays from raw
  predictions / ensemble outputs.
* `Calibrator` — exposes the ``fit(...) / predict(...) / with_state(...)``
  triple used by calibration tools (`TemperatureScaling`, …).
* `Conformalizer` — exposes the same triple shape but the predict
  output is a `PredictionInterval` or `PredictionSet`. Implemented by
  every class under `opifex.uncertainty.conformal`.

## Inference-Backend and Distribution-Adapter Specs

```python
from opifex.uncertainty.inference_backends import (
    InferenceBackendProtocol,
    InferenceBackendSpec,
    BackendResult,
    BackendDiagnostics,
    UnsupportedBackendError,
)
from opifex.uncertainty.adapters import (
    DistributionAdapterProtocol,
    DistributionAdapterSpec,
    ModelUncertaintyAdapterProtocol,
)
```

* `InferenceBackendProtocol` — ``fit(...) -> BackendResult`` plus
  ``posterior_predictive(...) -> PredictiveDistribution`` and
  ``predict_distribution(...) -> PredictiveDistribution``. Implemented
  by `BlackJAXBackend`; optional NumPyro / Bayeux / FlowJAX backends
  follow the same shape.
* `BackendResult` carries the raw sampler / fitted-state object
  unchanged (e.g. `BlackJAXSamplerState`) so callers can inspect it.
* `BackendDiagnostics` — Pattern-B `flax.struct.dataclass(slots=True,
  kw_only=True)` with `ess`, `r_hat`, `acceptance_rate`, `divergences`,
  `step_size`, `tree_depth` leaves (all optional). Survives
  ``flax.struct.replace()`` and pytree round-tripping.
* `InferenceBackendSpec` — frozen capability declaration with
  ``sampler_names: tuple[str, ...]``. The router walks the tuple
  left-to-right and selects the first installed backend.
* `DistributionAdapterProtocol` and `DistributionAdapterSpec` — wrap
  backend distribution objects (Artifex `Distribution`, Distrax-like)
  into the canonical `PredictiveDistribution` value object via
  `from_distribution(...)`.
* `ModelUncertaintyAdapterProtocol` — wraps deterministic /
  ensemble / dropout / Laplace-style models. The concrete adapters live
  in `opifex.uncertainty.adapters.{model,ensemble,operators}` and
  `opifex.uncertainty.curvature` (concrete `LaplaceAdapterSpec`); the
  remaining deferred spec classes (`SWAGAdapterSpec`,
  `FNOConformalAdapterSpec`, etc.) raise actionable
  `NotImplementedError` from `.wrap()` until the backend lands.

## Model UQ Adapters

```python
from opifex.uncertainty.adapters import (
    ModelUncertaintyAdapter,
    DeepEnsembleAdapter,
    MCDropoutAdapter,
    DeepEnsembleState,
    MCDropoutState,
)
```

* `ModelUncertaintyAdapter` — wraps a single deterministic callable.
  Rejects any non-`DefaultStrategy.DETERMINISTIC` capability claim so a
  raw point estimator cannot silently advertise epistemic uncertainty.
* `DeepEnsembleAdapter` — aggregates mean / variance over a fixed
  member tuple (`DeepEnsembleState`).
* `MCDropoutAdapter` — caller-owned `nnx.Rngs` at predict time;
  deterministic at `mode="deterministic"`, stochastic at
  `mode="bayesian"`.

The neural-operator family ships parallel specs in
`opifex.uncertainty.adapters.operators` (`OperatorAdapterSpec`,
`FNOConformalAdapterSpec`, `FNODeepEnsembleAdapterSpec`,
`FNOMCDropoutAdapterSpec`, `DeepONetConformalAdapterSpec`, …) that
declare operator-family axes (`spatial_axes`, `spectral_axes`),
supported metrics, and required capabilities.

## Likelihoods

```python
import jax.numpy as jnp
from opifex.uncertainty.likelihoods import (
    gaussian_log_likelihood,
    heteroscedastic_gaussian_log_likelihood,
    laplace_log_likelihood,
    student_t_log_likelihood,
    mixture_log_likelihood,
    LikelihoodSpec,
)

ll = gaussian_log_likelihood(
    predictions=jnp.zeros((10,)),
    targets=jnp.zeros((10,)),
    scale=1.0,
)
```

Each helper returns the per-sample log-likelihood as a `jax.Array`. The
heteroscedastic Gaussian takes a per-sample `scale` array; Student-t
takes a `df` parameter; the mixture form takes ``(weights, means,
scales)`` triplets and returns the log-sum-exp likelihood. `LikelihoodSpec`
is the Pattern-A frozen-dataclass capability container that registers a
likelihood family with the UQ registry.

## Priors

```python
from opifex.uncertainty.priors import diagonal_gaussian_log_prior, PriorSpec

log_prior = diagonal_gaussian_log_prior(
    params=jnp.zeros((4,)),
    prior_mean=0.0,
    prior_std=1.0,
)
```

`diagonal_gaussian_log_prior` returns the sum-over-features log-prior
under a diagonal-Gaussian `N(prior_mean, prior_std² I)`. `PriorSpec`
records the prior family (`name`, `family`, `parameter_names`) for the
registry, mirroring `LikelihoodSpec`.

## Physics-Informed Priors

`opifex.uncertainty.priors_physics` exposes five NNX-based prior
modules for physics-aware Bayesian modelling:

* `PhysicsInformedPriors` — hard-constraint projector with
  conservation-law and boundary-condition application.
* `ConservationLawPriors` — uncertainty modifier that inflates
  uncertainty based on per-law violation magnitude.
* `DomainSpecificPriors` — domain-tailored parameter ranges and
  distribution families (quantum chemistry, fluid dynamics, …).
* `HierarchicalBayesianFramework` — multi-level uncertainty
  propagation across hierarchy levels (`multiplicative` or
  `additive` modes).
* `PhysicsAwareUncertaintyPropagation` — confidence adjustment under
  physics-constraint violations.

Each module follows the canonical NNX convention: parameters declared
as `nnx.Param`, indexed with `[...]` (not the deprecated `.value`
property), pure-array methods trace under `jax.jit` / `jax.grad`.
