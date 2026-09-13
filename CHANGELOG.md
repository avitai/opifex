# Changelog

All notable changes to the Opifex framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The `backend:pathfinder`, `backend:svgd` and `backend:advi` capabilities in the UQ registry
  declare `source_package="opifex"`, as `PathfinderBackend`, `SVGDBackend` and `ADVIBackend`
  already do. They declared `"blackjax"`, although the three backends are implemented in opifex.
- The Markov power-EP and posterior-linearisation Student-t predictors,
  `predict_studentst_markov_pep_gp` and `predict_studentst_markov_pl_gp`, record
  `likelihood="students_t"` in their metadata, the label the dense Laplace GP and the Markov
  Laplace and variational predictors already use. They recorded `"studentst"`.
- `StateSpaceKernel` is a pytree. Its leaves are its matrices and the rates or angular frequencies
  of its closed-form transition, so a kernel passed to a jitted function reuses the compiled
  program for new hyperparameter values. `state_transition(dt)` returns the closed-form transition
  of the kernel's SDE, `discretize(dt)` returns the transition and the process noise of one step,
  and
  `discretize_steps(steps)` returns them for a sequence of steps. The Markov GP paths and the
  spatio-temporal GP discretise each time grid with one `discretize_steps` call. A kernel built from
  `feedback`, `noise_effect`, `diffusion`, `measurement` and `stationary_cov` alone discretises
  from its feedback matrix.
- `tfp-nightly` is a declared runtime dependency, for `bessel_ive`. Every install already had it
  through `avitai-artifex`, and it leaves the `probabilistic` extra.

### Deprecated

- `opifex.uncertainty.statespace.kernels.i0e_vector` is deprecated in favour of
  `tensorflow_probability.substrates.jax.math.bessel_ive`. The old name delegates to it and emits a
  `DeprecationWarning`.
- Constructing `StateSpaceKernel` with `state_transition=` is deprecated; omit it and the transition
  is computed from the feedback matrix. A kernel built with it discretises as in 0.2.5, with the
  supplied transition and the process noise `P_inf - A P_inf A^T`, and a `DeprecationWarning` is
  emitted.

### Fixed

- `log_expected_improvement` is accurate when the mean lies well above the incumbent. For
  `u < -1` it approximated `1 + u Phi(u) / phi(u)` by the series `1/u^2 - 3/u^4 + 15/u^6`, which
  is 13 at `u = -1` where the exact value is 0.344, so log-EI jumped by 3.63 nats at the branch
  boundary and was still off by 1.6 at `u = -1.5`. It now evaluates eq. 9 of Ament et al. (2023)
  with `erfcx` from TensorFlow Probability's JAX substrate, switching to the asymptotic branch
  below `-1e3` in float32 and `-1e6` in float64. `jax.scipy.special.erfcx` is not used: it
  returns 0 for arguments in [9.195, 9.419] in float32 and [26.544, 26.641] in float64.
- `kalman_smoother_parallel` smooths each step with the transition out of that step, as
  `kalman_smoother` does. It used the transition into the step, so on unevenly spaced times its
  means differed from the sequential smoother, by up to 0.73 on a 60-point Matérn-3/2 example.
  Evenly spaced times hid the difference.
- `kalman_filter_parallel` stays finite when a step adds no process noise. Its combine step
  inverted the covariance of the earlier element, which is singular without process noise, so the
  cosine and periodic kernels returned NaN from the fourth step on. The combine step now solves with
  `I + C_i J_j` (Särkkä & García-Fernández 2021, eqs. 13 and 14).
- The Beta response predictors `predict_beta_laplace_gp`, `predict_beta_markov_laplace_gp`,
  `predict_beta_markov_vi_gp`, `predict_beta_markov_pep_gp` and `predict_beta_markov_pl_gp` return
  the predictive mean and variance of the response. They returned the Beta variance
  `m (1 - m) / (s + 1)` at a single mean and left out the latent uncertainty. Over latent means in
  [-4, 4], latent variances from 0.1 to 4 and precisions from 10 to 50, that variance was as little
  as a twentieth of the predictive variance. The power-EP and posterior-linearisation predictors
  also took the mean at the latent mean, off by up to 0.11. Each predictor now integrates the Beta
  conditional moments over the latent Gaussian with 20-point Gauss-Hermite quadrature,
  `Var[y] = E[Var[y | f]] + Var[E[y | f]]`. `epistemic` still carries the latent variance.
- The learn-to-optimize surfaces declare their uncertainty capability again. Rebuilding the
  subsystem removed `BayesianSchedulerOptimizer` together with the only `l2o:` entry in the
  capability registry, so no L2O surface declared a UQ strategy. `L2OEngine` and
  `LearnedOptimizer` now register as unsupported. The engine's benchmark reports measured
  point summaries (mean learning curves, per-task and median speedup) and produces no
  predictive distribution. Registration is explicit through
  `opifex.optimization.l2o.register_l2o_capabilities`, and importing the package registers
  nothing.
- `periodic_kernel` weights its harmonics with correct Bessel values. The weights
  `I_n(x) e^{-x}` at `x = lengthscale**-2` came from forward recurrence, which amplifies
  rounding by roughly `(2n/x)^n` whenever the order exceeds `x`. In float32 with `order=12`
  the state-space covariance missed the closed-form periodic kernel by 2.4e3 at lengthscale 1
  and 2.1e9 at lengthscale 2. The values now come from TensorFlow Probability's `bessel_ive`
  (Temme's series below order 50, Olver's uniform asymptotic expansion above), and
  `quasi_periodic_matern12_kernel` inherits the correction.
- `quasi_periodic_matern12_kernel` declares the diffusion that balances its stationary
  covariance. It declared zero diffusion, so its SDE did not satisfy the Lyapunov equation,
  and any consumer that discretises `(F, L, Q_c)` received zero process noise. The spatio-temporal
  GP is one such consumer. The diffusion is now the Matérn-1/2 diffusion scaled by the periodic
  stationary covariance.
- The spatio-temporal GP returns finite predictions on time grids with long gaps. It built each
  step's process noise with the Van Loan block exponential, whose `exp(-F^T dt)` block overflows
  float32. On a grid with gaps of about 95 temporal lengthscales every prediction was NaN.
- Markov GP process noise is accurate in float32 at short steps. `P_inf - A P_inf A^T` subtracts
  nearly equal matrices when the step is short: in float32 the smallest process-noise components of
  a Matern-7/2 kernel were wrong by up to 2.4e15 times their size, and on 1000 points spaced 1e-3
  lengthscales apart the Gaussian evidence missed the exact dense GP by 3.4 nats. In float32 the
  process noise of Matern-3/2 to Matern-7/2 and of the quasi-periodic kernel now comes from the
  Stillfjord and Tronarp Gramian, which brings that evidence within 1.1e-4 nats. Matern-1/2 uses its
  closed form `sigma^2 (-expm1(-2 dt / ell))`, cosine and periodic kernels add no process noise, and
  float64 keeps `P_inf - A P_inf A^T`, which matches the dense GP to 1e-11. The float32 method is
  chosen while tracing, so float64 fits cost what they did. Per 1000 float32 steps a Gaussian
  Markov-Laplace fit of a Matern-7/2 kernel takes 28 ms instead of 10 ms and its gradient 90 ms
  instead of 49 ms; Matern-5/2 fits and 25-iteration Bernoulli fits are within the timing noise.
  At a zero step the derivative of the float32 process noise with respect to that step is zero.
- `discretize_lti_sde` is accurate in every component and stays finite at coarse steps. It
  exponentiated Van Loan's block `[[F, L Q_c L^T], [0, -F^T]] dt` in one go, which overflows float32
  and exceeds the squaring limit of `jax.scipy.linalg.expm`: Matern SDEs came back NaN at 100
  lengthscales, Matern-5/2 was off by 0.89 at 10, a third-order integrated Wiener process was off by
  5.0e-4 at dt = 100, and gradients were NaN at coarse steps. It now uses the exponential-and-Gramian
  doubling of Stillfjord and Tronarp (arXiv:2310.13462), with a
  fixed-length doubling loop so that reverse-mode gradients work. Against extended-precision
  references for Matern-1/2 to Matern-7/2, integrated Wiener processes of order 1 to 4 and
  integrated Ornstein-Uhlenbeck processes, at steps from 1e-4 to 1e4, every process-noise entry
  `Q_ij` is within `1.7e-5 sqrt(Q_ii Q_jj)` in float32 and `1.1e-13 sqrt(Q_ii Q_jj)` in float64.
  Gradients stay finite for growing drifts, a singular positive semi-definite `Q_c` is supported, and
  steps needing more than 32 doublings return NaN. Per 1000 float32 steps under `vmap` it takes 2.3
  to 3.0 ms instead of 8.1 to 8.7 ms for a three-state SDE and 26 to 27 ms instead of 2.3 to 5.9 ms
  for a four-state SDE; gradients take 8.4 ms instead of 43 ms and 93 ms instead of 20 ms.
- The Markov GP evidence values are the published energies. `fit_markov_vi_gp` returned the
  expected log likelihood minus a log-determinant penalty, `fit_markov_laplace_gp` the log
  likelihood at the mode minus the same penalty, and `fit_markov_pep_gp` the sum of the cavity log
  normalisers divided by the power. For a Gaussian likelihood, where each method recovers the exact
  posterior, they missed the exact log marginal likelihood by up to 42.6 nats (VI), 70.7 (Laplace)
  and 376 (power EP at power 0.1) on a grid of kernel hyperparameters. A Bernoulli Laplace evidence
  missed the dense Laplace approximation by 5.6. The ELBO is now eq. (11) of Chang, Wilkinson, Khan
  and Solin (2020), and the Laplace and power-EP evidences are eqs. (17) and (27) of Wilkinson,
  Särkkä and Solin (JMLR 2023). Each is computed from the Kalman log likelihood of the
  pseudo-observation model, and matches the exact value, or the dense Laplace
  approximation, to four decimals. The Gaussian power-EP log normaliser was also `½ log(1/power)`
  too high per observation and now includes the power-EP constant.
- `fit_laplace_gp` runs with float64 inputs. It started the Newton iteration from a float32 latent
  and objective, so with x64 enabled and a float64 kernel `jax.lax.scan` raised "carry input and
  carry output must have equal types". The initial carry now takes the dtype of the kernel matrix
  and the targets.

## [0.2.5] - 2026-09-11

### Fixed

- `compute_jacobian`, and with it `compute_empirical_ntk` and `NTKWrapper`, differentiated
  all of a model's state rather than its parameters. BatchNorm running statistics entered
  the kernel as if they were parameters, putting the eval-mode NTK of a small BatchNorm
  network 14% off, and models carrying RNG state, such as those using `nnx.Dropout`,
  raised `TypeError`. Only `nnx.Param` state is differentiated now, the Jacobian is an
  `nnx.State` of the parameters, and the calls to the deprecated `flax.nnx.State` API are
  gone. Models whose state is all parameters get the same kernel as before.

## [0.2.4] - 2026-09-11

### Changed

- `QH9PaddedSource.get_batch_at` is renamed `read_batch(start, size)`. It reads molecules on
  the host from a concrete position, while datarax treats a source implementing
  `get_batch_at` as JAX-traceable indexed access that `Pipeline` drives inside a compiled
  step. The unused `key` argument is gone, and `iterate_padded_batches` calls `read_batch`.
- Requires `datarax>=0.1.9`. `PDEBenchSource` and `VTKMeshSource` implement `get_batch_at`, and
  from datarax 0.1.9 that alone gives them indexed access, so `for batch in` the loaders from
  `create_pdebench_loader` and `create_vtk_mesh_loader` serves one epoch, batch for batch what
  `step()` returns; on earlier datarax it failed with `AttributeError: get_batch`.

## [0.2.3] - 2026-09-09

### Removed

- The one-release deprecation wrappers 0.2.2 introduced, and `opifex._deprecated`
  with them. `opifex.core.get_device_info`, `get_platform` and `is_gpu_available`:
  use `substrax.devices.detect_devices()`. `opifex.uncertainty.forecasting_metrics`
  (the whole package), `opifex.uncertainty.metrics`, `opifex.core.metrics`,
  `opifex.uncertainty.calibration.base` and `opifex.uncertainty.calibration.regression`:
  the 26 metric functions are `calibrax.metrics.functional.{forecasting, uncertainty,
  calibration, regression}` (`crps`, `fair_crps`, `energy_score`, `rank_histogram`,
  `spread_skill_ratio`, `pit_histogram`, `ranked_probability_score`,
  `event_reliability`, `ensemble_ranked_probability_score`,
  `ranked_probability_skill_score`, `predictive_entropy`, `mutual_information` (calibrax:
  `ensemble_mutual_information`), `interval_score`, `winkler_score`, `anees`,
  `non_credibility_index`, `chi2_confidence_intervals` (calibrax:
  `chi2_confidence_interval`), `gaussian_nll`, `brier_score`,
  `expected_calibration_error`, `pinball_loss` (calibrax: `quantile_loss`), `picp`,
  `mpiw`, `regression_calibration_error`, `per_sample_relative_l2`,
  `relative_l2_error`). `opifex.uncertainty.calibration` keeps the temperature-scaling
  calibrator. `opifex.benchmarking.BenchmarkRegistry` and
  `opifex.benchmarking.benchmark_registry.BenchmarkRegistry`: the class is
  `OperatorBenchmarkRegistry`. No sibling or consumer repository imported any of
  these names (fluctifex moved to calibrax in 0.1.1).

## [0.2.2] - 2026-09-09

### Fixed

- The PIKAN reference in `opifex.neural.kan.pikan` names its authors (Toscano et
  al., arXiv:2410.13228); it read "Hao et al.".
- Distributed training on jax 0.11: `jax.make_mesh` now defaults to explicit
  axis types, under which the backward pass of any layer over a batch sharded
  along `data` raised `ShardingTypeError` ("Contracting dimensions are
  sharded"), and the FNO's spectral scatter raised it on a data-sharded batch.
  `DistributedManager` builds its meshes through substrax 0.1.4, whose
  `DeviceMeshManager` creates `Auto` axes unless asked otherwise, and the
  regression test pins the axis type; the two distributed trainer tests and the
  distributed PDE example pass again.

### Added

- `UNO` in `opifex.neural.operators.OPERATOR_REGISTRY`: `UNeuralOperator` had a
  guide and an example but could not be created through `create_operator`.
- `scripts/derive_status.py`, which renders the README's Neural Operators bullet
  from the registry and, with `--check` in the quality checks, fails when the
  README names an architecture the registry does not hold or miscounts them (it
  said 26 and named DISCO, which does not exist; the registry holds 20).
- `docs/comparisons/jax-pde-landscape.md`: jNO, jinns, PINNx, DeepXDE and
  NeuralPDE.jl next to opifex, by what each states about itself.
- README install path from PyPI (`uv add opifex`), and the `mlflow` extra.
- Dependency floors: `substrax>=0.1.4`, `datarax>=0.1.6`, `calibrax>=0.1.5`,
  `avitai-artifex>=0.1.5`.

### Changed

- `opifex.benchmarking.ResultsManager` renders its publication output with
  calibrax's `PublicationGenerator`: `export_publication_plots` writes one
  comparison figure per dataset (`plots/<dataset>/comparison.<ext>`), one scaling
  figure per model and metric (`plots/<model>/scaling_<metric>.<ext>`) and one
  convergence figure per result with a loss history
  (`plots/<model>/<dataset>/convergence_loss.<ext>`); `generate_comparison_tables`
  writes `tables/table.<ext>` with rows labelled `<operator> (<dataset>)` and the
  best value marked. The execution time recorded in a result's metadata is a
  metric in the database summary, the calibrax store and the figures
  (`opifex.benchmarking.adapters.metric_values`). A corrupt database file is a
  `ValueError` instead of a silently emptied database.
- `opifex.benchmarking.BenchmarkRegistry` is `OperatorBenchmarkRegistry`, since it
  registers operators and benchmark configurations and is not calibrax's
  `BenchmarkRegistry`. The old name still resolves with a `DeprecationWarning`
  and is removed in 0.2.3.
- `opifex.mlops.MLflowBackend` records through substrax's `MLFlowLogger` (or any
  injected `opifex.mlops.backends.RunLogger`), which it opens on `start` in the
  experiment `opifex_<domain>_<name>` on `backend_config["tracking_uri"]` or
  `MLFLOW_TRACKING_URI`. `log_model` writes an Orbax checkpoint through substrax's
  `OrbaxCheckpointStore` and logs the directory; nothing is pickled. Physics
  metadata becomes `physics.<field>` and `physics.<field>.<key>` run parameters
  and metrics records are flattened by `opifex.mlops.records` (the slotted
  records could not be logged before: the backend read `__dict__`, which
  `slots=True` removes). The MLflow SDK is the new `mlflow` extra.
- `opifex.mlops.ExperimentTracker` (now `opifex.mlops.tracker`) registers the
  MLflow backend from the start, defaults to it, lists `backends`, and resolves
  `backend="auto"` to its `default_backend`; the physics-domain heuristic that
  chose backends which did not exist is gone. Importing `opifex.mlops` no longer
  registers `mlops:ExperimentTracker` in the `UQRegistry`;
  `register_mlops_capabilities(registry)` does, idempotently.

### Deprecated

- `opifex.core.get_device_info`, `get_platform` and `is_gpu_available` are
  wrappers over `substrax.devices.detect_devices()` that emit a
  `DeprecationWarning`; they are removed in 0.2.3. `configure_jax_precision`
  stays.
- The metric functions of `opifex.uncertainty.forecasting_metrics`,
  `opifex.uncertainty.metrics`, `opifex.uncertainty.calibration` (the
  calibrator stays) and `opifex.core.metrics` are calibrax's since calibrax
  0.1.3 (`calibrax.metrics.functional.{forecasting,uncertainty,calibration,regression}`).
  The 26 opifex names remain for this release as keyword-argument wrappers that
  emit a `DeprecationWarning` naming the calibrax function, and are removed in
  0.2.3: `opifex.uncertainty.forecasting_metrics.{crps, fair_crps, energy_score,
  rank_histogram, spread_skill_ratio, pit_histogram, ranked_probability_score,
  event_reliability, ensemble_ranked_probability_score,
  ranked_probability_skill_score}`, `opifex.uncertainty.metrics.{predictive_entropy,
  mutual_information, interval_score, winkler_score, anees, non_credibility_index,
  chi2_confidence_intervals}`, `opifex.uncertainty.calibration.{gaussian_nll,
  brier_score, expected_calibration_error, pinball_loss, picp, mpiw,
  regression_calibration_error}` and `opifex.core.metrics.{per_sample_relative_l2,
  relative_l2_error}`. Internal call sites (the trainer's relative L2 loss, the
  calibration aggregators' ECE, MCE and reliability bins, the examples and the
  guides) call calibrax directly; opifex's tests of the metric formulas went with
  the code, calibrax's suite carries them.
- `opifex.benchmarking.BenchmarkRegistry` and
  `opifex.benchmarking.benchmark_registry.BenchmarkRegistry` resolve to
  `OperatorBenchmarkRegistry` through a module `__getattr__` that emits a
  `DeprecationWarning`; the old name is removed in 0.2.3.

### Removed

- `ResultsManager.create_benchmark_database_entry`, which had no caller, and the
  manager's own matplotlib figures and LaTeX, HTML and CSV writers.
- `opifex.mlops.Framework.PYTORCH` and `TENSORFLOW`, the PyTorch and TensorFlow
  branches of `MLflowBackend.log_model`, the pickle fallback,
  `opifex.mlops.{MLFLOW_AVAILABLE, SUPPORTED_BACKENDS, SUPPORTED_FRAMEWORKS,
  SUPPORTED_PHYSICS_DOMAINS, __version__, __author__, __email__}` and the
  `ImportError`-swallowing stand-in in `opifex.mlops.backends`.
- `opifex.core.training.monitoring.flops.FlopsCounter`, an estimator that
  multiplied the parameter count by the input size (times 1.2). FLOP counting is
  calibrax's `FlopsCounter`, which reads XLA's cost analysis of the lowered
  function; `opifex.benchmarking.profiling` re-exports it.
- opifex's copy of the checkpoint store. `opifex.core.training.components.checkpoint_store`
  re-exports substrax's `CheckpointStore`, `ModelLike` and `OrbaxCheckpointStore`
  (the same save/restore/list/best-step surface, TrainState helpers included); the
  store's tests live with it in substrax.
- `opifex.core.training.strategies.mixed_precision` (`MixedPrecisionTrainer`, whose
  training step raised `TypeError` and had no caller, `MixedPrecisionConfig`,
  `MixedPrecisionState`, `scale_gradients`, `update_loss_scale`,
  `create_mixed_precision_policy`, `create_mixed_precision_optimizer`,
  `optimize_batch_size_for_hardware`, `align_for_tensorcore`) and
  `strategies.mixed_precision_ops.check_for_overflow`, together with the second
  hand-rolled loss scaler inside `MixedPrecisionComponent`. The component now
  composes `flax.training.dynamic_scale.DynamicScale`: `value_and_grad(loss_fn)`
  returns unscaled gradients with an `is_finite` flag, the scale backs off on a
  non-finite step and grows after `growth_interval` finite ones, `loss_scale`,
  `overflow_count` and `step_count` report the state, and `dynamic_loss_scaling=False`
  pins the scale.
- opifex's copies of the best-metric tracker, `EarlyStopping` and `PlateauMode`;
  `opifex.core.training.callbacks` re-exports substrax's and keeps
  `ReduceLROnPlateau` composed on substrax's tracker (it acts once per epoch on
  a validation metric between scanned epochs, which optax's per-step
  `reduce_on_plateau` transformation does not express).

### Changed

- `RooflineMemoryManager` reads its peak throughput, bandwidth and ridge point
  from calibrax's `detect_hardware_specs()` (the spec table every Avitai library
  shares) with the platform from substrax and the device's own memory
  statistics; `MixedPrecisionOptimizer` reads the accelerator class from
  substrax and the tensor-core shapes from calibrax instead of matching device
  names. The integration harness reads `substrax.devices.detect_devices()`.
- Depends on `substrax>=0.1.4`; `opifex.distributed` composes `substrax.mesh` and
  `substrax.spmd` (the `datarax.distributed` package it used is gone in datarax
  0.1.6). Floors: `datarax>=0.1.6`, `calibrax>=0.1.5`.
- CI runs the gates the sibling repositories run: the lockfile check, `twine
  check --strict`, a blocking bandit, one blocking pip-audit, the 80 percent
  coverage floor in `[tool.coverage.report]`, pytest `--strict-config`, and the
  validate-pyproject, interrogate (floor 95, measured 95.2) and pydoclint hooks
  (a checked-in baseline of 551 findings that can only shrink). Ruff selects the
  `ANN` and `D` families (Google convention) with the exemptions the pydocstyle
  hook carried; the 163 file-rule pairs `src/` carried at adoption live in
  `quality/ruff_baseline.json`, rendered into the per-file-ignores table by
  `scripts/check_ruff_baseline.py`, which fails when a pair grows, a cleared
  pair is still listed, or the table drifts. The pydocstyle hook is gone.
- Publishing uses PyPI trusted publishing (OIDC); no API token is stored.

## [0.2.1] - 2026-08-29

### Changed

- **Requires Python 3.12 or later.** jax 0.11.0 dropped 3.11, and this release
  takes that jax line.
- **The `gpu` extra is renamed `cuda12`,** and resolves `jax[cuda12]` rather than
  `jax[cuda12_local]`. The local variant expects a CUDA toolkit already on the
  machine, which contradicts what `setup.sh` promises; the hand-listed NVIDIA
  wheels existed only to supply what the pip-managed variant already provides.
  The exact jax pin is dropped: uv resolves one universal lockfile across every
  extra, so it governed the jax that `dev`, `test` and `docs` received while no
  workflow installed this extra at all.
- Resolves to jax 0.11.1, jaxlib 0.11.1, flax 0.12.9, optax 0.2.8 and grain
  0.2.18, matching the sibling packages, and raises the floors on
  `avitai-artifex` to 0.1.4, `calibrax` to 0.1.2 and `datarax` to 0.1.5.
- `distrax>=0.1.9` is now required. 0.1.7 calls `jax.core.is_concrete` and
  `jax.core.valid_jaxtype`, both removed in jax 0.11.0.
- Call sites moved off `jnp.clip(a_min=, a_max=)`, `jax.experimental.enable_x64`
  and `jax.core.concrete_or_error`, removed in jax 0.10.0, 0.9.0 and 0.11.0.

- **Breaking: neural operators take their training mode from nnx, not a `training`
  argument.** The `training` parameter is removed from every operator entry point
  (26 functions across 10 modules). Mode is module state, as it is throughout Flax
  NNX: call `model.train()` or `model.eval()`, which set `deterministic` and
  `use_running_average` recursively, or build a view with
  `nnx.view(model, use_running_average=True)`.

  ```python
  # before
  y = operator(x, training=False)

  # after
  operator.eval()
  y = operator(x)
  ```

  Two consequences worth checking when upgrading:

  - Operators that previously defaulted to `training=False` now follow the nnx
    default of training mode, so dropout is **active** unless `eval()` is called.
    Anything relying on a deterministic forward must now say so explicitly.
  - `PowerIteration` and `SensorOptimization` gained a `use_running_average`
    attribute and a `set_view` method, matching `nnx.BatchNorm`. Under `eval()`
    the spectral-norm vectors are no longer written back; previously inference
    silently mutated them, since the flag defaulted to `True`.

### Known limitations

- `kfac_jax` cannot be imported on the jax this release requires. Its latest
  release, 0.0.8, annotates a loss tag with `jax.core.Effects`, which jax removed
  in 0.11.0, so importing it raises `AttributeError`. The `quantum-chemistry`
  extra still declares `kfac-jax>=0.0.8` so a future compatible release is picked
  up automatically, but `opifex.neural.quantum.vmc.kfac_preconditioner` is
  unusable until then, and its tests skip with that reason.

### Fixed

- `PowerIteration` sizes its `u` and `v` vectors for the weight they normalize.
  They had been scalar placeholders re-drawn inside `__call__`, so under `eval()`
  nothing was ever written back and every forward pass restarted the estimate
  from a fresh random vector instead of sharpening it.
- `SpectralNorm` writes its re-estimated vectors through `.value`, which replaces
  them, rather than an indexed assignment that cannot change shape.
- `WavefunctionBC` keeps the imaginary part of a complex boundary value; the
  result dtype had been forced to the input's real dtype, discarding it silently.

## [0.2.0] - 2026-06-24

### Added

- **Uncertainty quantification platform**: conformal prediction, calibration and
  reliability metrics, the Gaussian-process family, Bayesian quadrature and
  probabilistic-ODE inference, state-space Kalman filtering, curvature operators,
  matrix-free probabilistic linear algebra, sensitivity analysis, simulation-based
  inference, active learning, and a model/operator UQ adapter suite with a
  capability registry.
- **Bayesian-linear UQ surfaces** on `MeanFieldGaussian` (`predict_distribution`,
  `loss_components`, `negative_elbo`) so it satisfies the `UncertaintyAwareModule`
  and `VariationalModule` protocols.
- **Quantum chemistry**: E(3)-equivariant core and atomistic models, differentiable
  Kohn–Sham DFT, variational Monte Carlo, equivariant Hamiltonian prediction, and
  the QH9 pipeline.
- **Atomistic potentials**: faithful NequIP with MACE-style higher body-order via
  symmetric contraction, plus an ASE calculator.
- **Physics solvers**: general pseudo-spectral ETDRK4 solver for semilinear PDEs.
- **Data**: PDE data layer migrated to datarax with spectral generators; PDEBench
  and VTK unstructured-mesh sources on the datarax Source/Pipeline contract.

### Changed

- **Learn-to-optimize subsystem rebuilt** on a unified Task/Optimizer abstraction:
  learned MLP and Adafactor optimizers, persistent-evolution-strategy meta-training,
  MAML/Reptile, and distribution-tuned baselines.
- **Neural-operator suite and examples** reworked into SOTA-competitive showcases;
  faithful UNO rebuild and resolution-invariant FNO domain padding.
- Adopted flax NNX best practices across the core stack and consolidated training
  onto the NNX-native `Trainer`.
- Single-sourced package metadata: `__version__` / `__author__` / `__email__` are
  read from the installed package metadata instead of being duplicated in code.
- Refreshed the README and feature documentation.

### Fixed

- Physics-operator and solver correctness fixes across operators and the platform.
- Resolved dependency security advisories (jupyter-server, jupyterlab, msgpack) and
  repinned a yanked grpcio.

### CI

- Sharded the unit-test matrix with `pytest-split` (combined with xdist), moved the
  coverage gate to the aggregated `coverage` job, and fixed latent
  test-isolation/precision/timing failures the sharding exposed.

## [0.1.0] - 2026-05-01

### Changed

- **BREAKING: package renamed from SciML to Opifex** — the package directory, the
  `pyproject.toml` name, all imports (`from sciml.` → `from opifex.`), CLI commands
  (`sciml-*` → `opifex-*`), environment variables (`SCIML_*` → `OPIFEX_*`), and the
  Kubernetes / Docker / documentation references. *From Latin "opifex" — worker,
  skilled maker.*

### Added

- **Uncertainty quantification**: multi-source aggregation with adaptive weighting
  (reliability-based, inverse-variance, entropy-based, uniform), epistemic (ensemble
  disagreement, predictive diversity) and aleatoric (Gaussian / Laplace / mixture)
  decomposition, and quality assessment (coverage probability, calibration,
  reliability).
- Bayesian API reference and uncertainty-quantification usage examples.
