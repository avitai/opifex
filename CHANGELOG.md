# Changelog

All notable changes to the Opifex framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Distributed training on jax 0.11: `jax.make_mesh` now defaults to explicit
  axis types, under which the backward pass of any layer over a batch sharded
  along `data` raised `ShardingTypeError` ("Contracting dimensions are
  sharded"), and the FNO's spectral scatter raised it on a data-sharded batch.
  `DistributedManager` builds its meshes through substrax 0.1.4, whose
  `DeviceMeshManager` creates `Auto` axes unless asked otherwise, and the
  regression test pins the axis type; the two distributed trainer tests and the
  distributed PDE example pass again.

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
  The opifex names remain for this release as keyword-argument wrappers that
  emit a `DeprecationWarning` naming the calibrax function, and are removed in
  0.2.3. Internal call sites (the trainer's relative L2 loss, the calibration
  aggregators' ECE, MCE and reliability bins, the examples and the guides) call
  calibrax directly; opifex's tests of the metric formulas went with the code,
  calibrax's suite carries them.

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
