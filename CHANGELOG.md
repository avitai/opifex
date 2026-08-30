# Changelog

All notable changes to the Opifex framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
