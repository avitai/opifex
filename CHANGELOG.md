# Changelog

All notable changes to the Opifex framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
