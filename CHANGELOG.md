# Changelog

All notable changes to the Opifex framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `GroupedConformalizedQuantileRegressor` and `GroupedCQRState` in `opifex.uncertainty.conformal` for Mondrian (group-conditional) Conformalized Quantile Regression (Romano, Patterson & Candès 2019; Vovk et al. 2005), returning `(-inf, +inf)` intervals for underfilled calibration groups (`< minimum_calibration_size`) and unseen test groups.
- `cv_plus_cqr_intervals`, `CrossConformalizedQuantileRegressor`, and `CrossCQRState` in `opifex.uncertainty.conformal` for K-fold cross-conformalized quantile regression using exact Barber et al. (2021) CV+ finite-sample order statistics.
- `feedback_covariate_shift_weights` and `feedback_covariate_shift_intervals` in `opifex.uncertainty.conformal` for conformal prediction under adaptive design / feedback covariate shift (Fannjiang et al., *PNAS* 2022, Theorem 1), supporting both factorized `(log_calibration_ratios, log_test_ratios)` and swapped-policy `(n_test, n_calib + 1)` log-likelihood-ratio matrices with row-max log-space stabilization.
- `excess_aurc` in `opifex.uncertainty.selective` (Geifman, Uziel & El-Yaniv 2019), paired with tie-invariant `risk_coverage_curve` (`jax.ops.segment_sum` group-mean error averaging across tied confidences) so constant or tied confidence rankings are permutation-invariant.
- `Status.INITIAL_FAILURE` (`10`) in `opifex.core.solver.status` for violated traced solver preconditions (such as non-monotone save times `ts` under `jax.jit`).
- Checkpointed bounded while-loop primitive (`bounded_while_loop` and `custom_vmap` `unvmap_any`) in `opifex.core.solver` for early-exiting batched adaptive loops with forward- and reverse-mode automatic differentiation.
- Implicit-differentiation helpers (`TangentSolverFn`, `dense_tangent_solve`, `lsmr_tangent_solve`, stagnation-aware `gmres_tangent_solve`, `infer_inexact_dtype`, `cast_inexact_pytree`, and `guard_unconverged_tangents`) in `opifex.core.solver`: `dense_tangent_solve` materializes and LU-factors the Jacobian once per operator across batched right-hand sides with an `O(n^2)` alternating triangular-solve condition gate and normwise backward-error contract; `lsmr_tangent_solve` runs matrix-free Fong & Saunders (2011) Golub-Kahan bidiagonalization without forming `A^T A`; and `gmres_tangent_solve` tracks consecutive non-decreasing Arnoldi cycles (`stagnation_iters=20`), all returning `NaN` on singular or stagnated systems.
- Adaptive and constant-step 7-stage FSAL Tsitouras (2011) 5(4) integrator `tsit5_solve` in `opifex.solvers.ude` (`UDEConfig.adaptive=True` by default) with recursive checkpointed bounded `while_loop` batch early exit, in-step cursor-driven 4th-degree continuous-extension dense output (`O(h^5)` global dense accuracy) across `ts`, Hairer-Nørsett-Wanner (§II.4) deadbeat I-controller step-size adaptation (`dtmin`, `force_dtmin`, and number-spacing floor returning `Status.STEP_TOO_SMALL`), traced `ts` monotonicity validation (`Status.INITIAL_FAILURE`), and discretize-then-optimize reverse-mode AD.
- Public `kernel_diagonal` helper in `opifex.uncertainty.gp` for `O(m)` marginal prior variance evaluation across `ExactGP`, `SVGP`, `LaplaceGP`, `StochasticSVGP`, `LinearMultiFidelityGP`, and `NonLinearMultiFidelityGP` (replacing `output_scale**2` prior variance assumptions on composite or multi-output kernels).
- Optional discrete candidate-grid parameter `x_grid` `(g, d)` on `mumbo_acquisition` in `opifex.uncertainty.multi_fidelity` for target-level Gumbel max-value estimation on discrete candidate domains, preserving `float32` candidate dtypes when `jax_enable_x64` is enabled.

### Fixed

- `ipcw_conformal_quantile` normalizes finite weights by their maximum before prefix-sum tail evaluation (Tibshirani et al. 2019, Eq. 5), selects the exact `conformal_quantile_rank(n, alpha)` order statistic under equal positive weights in `float32` and `float64`, propagates `NaN` inputs, and returns `+inf` when calibration mass cannot reach `1 - alpha`.
- `fit_exact_gp`, `fit_heteroscedastic_exact_gp`, and `nonpositive_message` preserve input floating dtypes and delegate concrete scalar extraction to `calibrax.core.eager_scalar_float`.
- `predict_linear_multi_fidelity_gp` validates `target_level` bounds with `ValueError` and clamps posterior variance at `0.0` instead of `1e-6`.
- `local_penalization` in `opifex.uncertainty.active` uses `optax.safe_norm` so gradients at coinciding pending points are finite `0.0`.
- `LevenbergMarquardt` in `opifex.solvers.nonlinear_least_squares` routes implicit differentiation through `opifex.core.solver` (`tangent_solver`, returning `NaN` on singular Hessians consistently with `BFGSSolver`) and runs an early-exiting `jax.lax.while_loop` under `implicit_diff=True` and `bounded_while_loop` under `implicit_diff=False`.
- `solve_stiff_ode` in `opifex.solvers.stiff_ode` deduplicates working floating-dtype inference and pytree casting through `opifex.core.solver`.

### Changed

- `.python-version` pins 3.12, the oldest Python the package supports, so `./setup.sh` and
  any uv command without `--python` build on it rather than on the newest interpreter
  installed (a free-threaded 3.14 on one development machine).
  Every CI job with a Python matrix sets `UV_PYTHON` to its matrix version, which the pin
  would otherwise override for its uv commands.
- `_DeterministicLinear.__call__` in `opifex.neural.bayesian.probabilistic_pinns` drops unread `deterministic` and `rngs` flags (dispatching via `_apply_pinn_layer`) and `BayesianLinear` and `BayesianSpectralConvolution` drop redundant `substrax-allow` comments.
- Raised minimum dependency floors to `jax>=0.11.1`, `flax>=0.12.10`, `calibrax[publication]>=0.1.23`, and `substrax>=0.1.37`. Under `calibrax>=0.1.23`, `CompilationProfiler` times tracing, lowering, and XLA compilation (`lower(...).compile()`) only rather than the first program execution, so `avg_compilation_time_ms` and `max_compilation_time_ms` in `opifex.benchmarking.profiling.profiling_harness` read lower wherever a first run was slow.
- **Breaking:** Moved `lsmr` from `opifex.uncertainty.linalg` to `opifex.core.solver.lsmr` and updated `opifex.fields.pressure` and `opifex.solvers.nonlinear_least_squares` to import it from `opifex.core.solver`.
- **Breaking:** `NeuralODE.solve` and `UDESolver.solve` return `Solution` directly (`fields["y"]`, `fields["t"]`, `status`, and `stats["num_steps"]`), dispatch through `self.vector_field`, validate that submodules do not carry active `.train()` stochastic `Dropout` (`deterministic=False`) or batch-statistics `BatchNorm` (`use_running_average=False`), and emit `NaN` from `UDESolver.trajectory_loss` when `not is_successful(sol.status)`; re-exported `nonlinear_least_squares`, `stiff_ode`, and `ude` symbols from `opifex.solvers`.
- **Breaking:** `fit_nonlinear_multi_fidelity_gp` and keyword-only `flax.struct.dataclass` `NonLinearMultiFidelityGPState` evaluate the Perdikaris et al. (2017, Eq. 2.12) two-term composite kernel (`nargp_composite_kernel`) on unnormalized `[x, f_{*,i-1}(x)]` with unit output scale on `k_f` for multiplicative variance identifiability and traceable `jax.Array` PyTree leaves (`lengthscales`, `output_scales`, `noise_stds`, `f_lengthscales`, `bias_lengthscales`, `bias_output_scales`).
- **Breaking:** `weighted_conformal_quantile` requires keyword-only `test_weight` and `weighted_split_conformal_intervals` requires keyword-only `test_weights` alongside 1-D `predictions`.
- **Breaking:** `CensoredSplitConformalRegressor` delegates left-continuous reverse Kaplan-Meier censoring survival `P(C >= t)` to `calibrax.metrics.functional.reverse_kaplan_meier(..., left_continuous=True)` over the full risk set `observed_times >= u_k` (Uno et al. 2011; Candès, Lei & Ren 2023), preserves `G_hat(c_0)` at the cutoff as `default_test_survival` under general right censoring, validates `censoring_survival` `(0, 1]` bounds and 1-D shapes, and requires test `censoring_survival` in `predict()` whenever `fit()` calibrated against explicit `censoring_survival`.
- **Breaking:** All conformal quantile, jackknife+, CV+, Mondrian, and censored entry points validate static `alpha` via `calibrax.statistics` (raising `ValueError` outside `(0, 1)` and `TypeError` on `bfloat16` or traced `alpha`).
- **Breaking:** `AbstentionDecision` in `opifex.uncertainty.selective` stores `threshold` and count metadata entries as traceable `jax.Array` PyTree leaves rather than static Python scalars.

### Removed

- `opifex.uncertainty.linalg.lstsq` (moved to `opifex.core.solver.lsmr`).

## [0.2.14] - 2026-11-21

### Changed

- Requires `calibrax[publication]>=0.1.21`, whose `Registry.register` refuses a name the
  registry already holds with `DuplicateRegistrationError` (a `ValueError`, exported from
  `calibrax.core`) instead of replacing the item registered first.
- Requires `substrax>=0.1.32` (and its `modal`, `mlflow`, `notebooks` and `testing` extras at
  that floor). `substrax.testing.gradients.check_parameter_gradients` and
  `check_input_gradients`, used by the solver gradient tests, compare against finite
  differences along a direction of unit norm (before 0.1.31 the step grew with the square root
  of the parameter count), with a step of 1e-6 (0.1.32; it was 1e-4).
- `src/opifex` is checked against `substrax.testing.nnx_rules`
  (`tests/core/test_nnx_source_rules.py`). `noop-super-init`: the 133 `super().__init__()`
  calls in direct `nnx.Module` subclasses are removed; `nnx.Module` defines no `__init__`, so
  each reached `object.__init__`. `trainable-metadata`: `SystemDynamicsModel`'s linear `A`
  and `B` matrices no longer pass `trainable=True` to `nnx.Param`, metadata flax never reads.
- `SearchEngine`'s keyword embedding is scikit-learn's feature hashing
  (`FeatureHasher`: signed MurmurHash3 buckets), L2-normalised, of the extracted
  keywords, in place of the builtin `hash`, whose value for a `str` changes with
  `PYTHONHASHSEED`, so the same text embeds, and ranks, the same in every interpreter
  (`salted-hash`). Colliding keywords take opposite signs half the time, so cosine scores
  can be negative. Requires `scikit-learn>=1.9.1`, now a declared dependency.
  `AdaptiveJAXOptimizer`'s cache key holds the workload's text instead of its hash.
- **Breaking:** `Trainer` takes no `rngs` and `TrainingState` has no `rngs` field. Nothing
  drew from them; a missing `rngs` was replaced by `nnx.Rngs(0)`. A model keeps its own
  `nnx.Rngs`. Pass `Trainer(model, config)`; the examples and docs are updated.
- **Breaking:** `LoRALinear` and `apply_lora` require `rngs`; a missing one was replaced by
  `nnx.Rngs(0)`, so every adapter built without it drew the same `A` factor
  (`rngs-fallback`).
- **Breaking:** `fit_galerkin_surrogate(rngs=...)` takes an `nnx.Rngs` (stream `sg_samples`,
  else `sample`, else `default`) or a key, in place of a mapping of stream names to keys;
  `fit_collocation_surrogate` takes no `rngs`, which it never read. The ADVI, Pathfinder
  and SVGD backends take their keys through `substrax.rng.key_from` instead of probing the
  `nnx.Rngs` (`rngs-probe`).
- **Breaking:** the data loaders (`create_{burgers,darcy,diffusion,navier_stokes,
  shallow_water}_loader`, `create_rmd17_loader`, `create_pdebench_loader`,
  `create_vtk_mesh_loader`), `RMD17Config` (now keyword-only) and `ExecutionConfig` require
  `seed`, and the PDE generators (`generate_*`) require a keyword-only `key`, split into one
  key per sample. The loaders generate from their seed's `data` stream and shuffle from its
  `shuffle` stream (`substrax.rng.rngs_from_seed`). Per-sample keys were `PRNGKey(seed + i)`,
  so seeds `s` and `s + 1` shared all but one sample; every generated dataset changes.
  `OperatorExecutor` requires its `ExecutionConfig`. `BenchmarkRunner` takes a keyword-only
  `seed`, from which every run generates its data and initialises its operator;
  `opifex-benchmark --seed` passes it. Before, the CLI wrote `--seed` into the benchmark
  configuration and nothing read it, so every benchmark ran on seed 42 (`seed-literal`).
- `benchmark_with_prefetching` draws no random operands: the two it drew were overwritten
  from the pool buffers and never read, and its prefetch payload, whose values nothing reads,
  is zeros.
- **Breaking:** `sample_boundary_points` and `compute_shape_area` require a keyword-only
  `key`; they drew from fixed keys 42 and 123.
- **Breaking:** geometry membership is exact and vectorised, as DeepXDE's `inside`:
  `contains` takes points of shape `(..., d)` and returns a boolean array of shape `(...)`,
  where it returned a Python `bool` for one point. `Rectangle`, `Circle` and `Interval` are
  closed sets, `Polygon` uses the winding number, and `CSGUnion`, `CSGIntersection` and
  `CSGDifference` combine their operands' memberships with `|`, `&` and `& ~`. They decided
  membership by the sign of a smoothed signed distance, which put points of a rectangle near
  its corners outside the union. Every shape has an exact `bbox`; `compute_shape_area` draws
  its points from it and takes no `bbox` argument (it raised `TracerBoolConversionError`
  before, because it vmapped `contains`). `validate_implementation`, which failed on a
  correct implementation and was tested by nothing, is removed.
- CSG sampling follows DeepXDE's samplers. `CSGUnion.sample_interior` keeps uniform points of
  its bounding box that the union contains, so it samples the union uniformly; it drew half
  its points from each operand, which put twice the density in the overlap.
  `CSGIntersection`, `CSGDifference` and `Polygon` draw rounds until they hold `n` points
  instead of repeating their last point to fill `n`. Each CSG node's `sample_boundary` keeps
  each operand's boundary points that lie on the result's boundary, so a difference's
  boundary includes its hole, which it never sampled, and an intersection whose boundary
  filter found nothing no longer returns a single point at the origin.
- **Breaking:** `optimistix_minimise`, `WarmStartingStrategy.get_warm_start_params` and
  `MetaOptimizer.get_warm_start_params` require a keyword-only `key`. They fell back to
  keys 0 and 42, so every warm start drew the same perturbation or initialisation.
- **Breaking:** `DifferentiableMPC`, `SafetyCriticalMPC` and `RecedingHorizonController`
  require a keyword-only `rngs`, from which the default `PredictiveModel` is initialised;
  it was built from key 42 for every controller.
- **Breaking:** `FunctionEmbedding` requires `rngs` (its projection was initialised from
  `nnx.Rngs(0)`), and `benchmark_tensorly_integration` requires a keyword-only `key` for
  its test tensor (it was key 42).
- **Breaking:** `PowerIteration` requires `weight_shape`: its vectors are created at their
  real size, and a weight that flattens to another matrix is refused with `ValueError`.
  Without it the vectors were shape-`(1,)` placeholders that every call replaced with draws
  from `nnx.Rngs(0)` and `nnx.Rngs(1)`, so the estimate never carried forward. `SpectralNorm`
  refuses a layer with no `kernel` or `weight` at construction instead of at its first call.
  The power iteration follows PyTorch's spectral norm: `u` and `v` are `nnx.BatchStat`
  state, outside the gradient, drawn from `rngs.params()`. They were `nnx.Param`, so an
  optimizer over parameters trained them, and the gradient flowed through them. Training
  calls run the power method and keep the vectors. Eval calls use the stored vectors as
  they are, where before they re-estimated the vectors and discarded the result. The weight
  is divided by `sigma = u . (W v)`, which stays the parameter. `update_stats=False` replaces
  `use_running_average=True` in the constructor, `num_iterations` below one is refused, and a
  write of the vectors that fails is no longer silently skipped. Checkpoints that hold `u`
  and `v` as parameters do not restore.
- **Breaking:** `PINNConfig` requires `seed`, and `PINNSolver.solve` requires its `config`;
  a call without one sampled its collocation points from seed 42.
- `ScientificComputingIntegrator.comprehensive_scientific_validation` stamps its result
  with `time.time()`; its `timestamp` was a uniform draw from key 0 times 1000, the same
  value on every call.
- **Breaking:** `PDEBenchConfig` and `create_pdebench_loader` live in
  `opifex.data.sources.pdebench`, which imports `h5py` at its top, instead of
  `opifex.data.sources.scientific`, which imported it inside `PDEBenchSource.__init__`.
  `opifex.data.sources` still exports them and loads the module on their first use
  (`lazy_loader.attach_stub` over `__init__.pyi`, as `datarax.sources` loads its optional
  integrations), so `import opifex.data.sources` does not import `h5py`. Requires
  `lazy-loader>=0.6`, now a declared dependency.
- **Breaking:** PDEBench files are read as PDEBench's own loader reads them
  (`FNODatasetSingle`): `load_pdebench(PDEBenchConfig(...))` returns a split's `inputs`
  (the first `initial_step` steps), `targets` (whole trajectories) and shared `grid`, every
  field laid out `[N, x..., t, V]`, and `create_pdebench_loader` serves them through a
  datarax `MemorySource` and `Pipeline`. It reads scalar files (`tensor`), Darcy flow (`nu`,
  then the solution, along time), compressible flow (`density`, `pressure`, `Vx`, `Vy`, `Vz`)
  and grouped `.h5` files, and refuses PDEBench's incompressible Navier-Stokes layout, which
  PDEBench's loaders do not read either. `PDEBenchConfig` takes PDEBench's `initial_step`,
  `test_ratio`, `num_samples_max` and `reduced_batch`/`reduced_resolution`/
  `reduced_resolution_t`, all keyword-only, with `file_path` required. It is a frozen
  dataclass rather than a datarax `StructuralConfig`; its `shuffle` and `dtype` fields are
  removed, since the loader shuffles the train split from its `seed` and every field is read
  as `float32`. The test split is the first `test_ratio` of the samples, as in
  PDEBench, where the train split was taken first. Strides are read through h5py's hyperslab
  selection, so a file is no longer read whole. `PDEBenchSource`, the sliding windows
  (`input_steps`, `output_steps`), `train_split`, `dataset_name`, `field_keys` and the
  split-local min-max normalisation are removed: they read a 1D file as steady (input equal to
  target), read no Darcy `nu`, found no coordinates and scaled each split by its own
  statistics. Normalise with a `GaussianNormalizer` fitted on the train split.
- **Breaking:** `PDEBenchLoader.load_dataset` reads a real PDEBench file under `data_root`
  (matched by its published name; `file_name` chooses one when several match) through
  `load_pdebench`, and returns PDEBench's first training pair: the `initial_step` input frames
  stacked as channels, `(n, initial_step * V, x...)`, and the next frame, `(n, V, x...)`; a
  DeepONet trunk takes the file's grid. It returned synthetic data in place of a real file,
  without the opt-in or the warning the synthetic path requires. Incompressible Navier-Stokes
  files are refused (PDEBench has no reader for them), and `split` is `"train"` or `"test"`.
- **Breaking:** `PDEBenchLoader.load_dataset` and `PDEBenchEvaluationPipeline` take
  `synthetic_key` in place of `allow_synthetic`: a key opts in to synthetic data, of the real
  pair's shapes, drawn from the key folded by the dataset's name, so two datasets share no
  draw. The data was drawn from key 42. `diffusion_reaction` and `compressible_navier_stokes`
  have schemas (they raised `KeyError`). `normalize` is removed: it scaled each split by its
  own statistics. `PDEBenchLoader` takes no `cache_dir` (unused) and creates no directories,
  and `PDEBenchEvaluationPipeline.evaluate_model_on_datasets` lets a failing dataset's error
  propagate instead of warning and skipping it.
- `seed-literal` holds over `src/opifex`, so opifex's own fixed-key checks, which looked
  only for `jax.random.PRNGKey(...)` in four uncertainty modules
  (`tests/uncertainty/test_rng_safety.py` and a test in
  `tests/uncertainty/layers/test_bayesian.py`), are removed.
- **Breaking:** `QH9PaddedConfig` takes `split_seed` (43, QH9-Stable's reference split) and
  `shuffle_seed`, which `shuffle=True` requires, in place of `seed`: one seed fixed both the
  reference split and the training order, so the order was QH9's split seed rather than the
  caller's. `scripts/train_qh9_blocks.py` takes `--seed` (default 0) for its model and its
  shuffle.
- **Breaking:** `TensorLyTuckerInitializer.decompose_tensor` and
  `TensorLyCPInitializer.decompose_tensor` take `key`, from which TensorLy's random draws come:
  they were fixed at 42, which decided Tucker's random initialisation and the CP factor columns
  beyond a mode's size. `TensorLyEnhancedDecomposition` draws it from its `rngs`, and
  `MultiGridTuckerDecomposition` folds it from each band's key. The Tucker initializer's
  tests fail instead of skipping on an error.
- **Breaking:** `CliffordFourierBlock`, `CliffordFNO`, `PIKAN`, `SincKAN`, `UNeuralOperator`,
  `FourierContinuationExtender`, `FourierBoundaryHandler`, `SystemIdentifier` and
  `SystemDynamicsModel` take no `deterministic` argument. Each ignored it or passed it to
  layers that ignore it (`mode-flag-default`).
- **Breaking:** `StandardMLP` takes no `deterministic` argument: its dropout follows the
  module's mode, so `.eval()` turns it off and `.train()` on. Before, the flag defaulted to
  `False`, so dropout stayed on after `.eval()`. The DeepONets (`DeepONet`,
  `MultiFidelityDeepONet`, `AdaptiveDeepONet`, `FourierEnhancedDeepONet`,
  `PODDeepONet`, `DeepONetTrainerAdapter`), `PhysicsInformedOperator`,
  `LaplaceNeuralOperator` and Transolver's `PhysicsAttention`, `TransolverBlock` and
  `Transolver` drop the flag they forwarded to it. Transolver defaulted it to `True`, so
  with `dropout_rate > 0` its dropout now runs in training mode; call `.eval()` for
  inference.
- **Breaking:** `NeuralXCFunctional` and its parts take no `deterministic` argument: the
  functional's `__call__`, `energy_density_from_sigma`, `xc_potential_components`,
  `compute_functional_derivative` and `assess_chemical_accuracy`, `DensityFeatureExtractor` and
  the functional's attention layer. Dropout follows the module's mode (`.train()`/`.eval()`).
  `NeuralXCSpec.from_functional` splits an evaluation view of the functional
  (`nnx.view(..., deterministic=True)`), so the SCF evaluates the functional without dropout
  whatever mode the caller's module is in, and the caller's module is left unchanged.
- **Breaking:** the Bayesian layers and PINNs follow flax's mode handling. `BayesianLinear`
  and `BayesianSpectralConvolution` take `deterministic` keyword-only, as `nnx.Dropout` does,
  and gain `set_view`, so `nnx.view(layer, deterministic=True)` returns the posterior mean.
  `ProbabilisticPINN` (whose `deterministic` was stored and never read) and
  `ComputationAwareSpectralConvolution` (whose forward always used the mean) take no
  `deterministic` argument; `ProbabilisticPINN.__call__` and `MultiFidelityPINN.__call__` take
  none either: `.eval()` or `nnx.view(pinn, deterministic=True)` gives the posterior mean,
  `.train()` samples. Neither PINN keeps a reference to the caller's `nnx.Rngs`: each
  `BayesianLinear` samples from the stream it forks at construction, and
  `ProbabilisticPINN.robust_loss` takes `rngs`. A stored `nnx.Rngs` had a key drawn by every
  `nnx.view`, and the view failed under `jax.jvp`.
- `AtomisticModelRegistry`, `BackboneRegistry`, `PropertyHeadRegistry` and
  `ServableModelRegistry` leave the refusal of a taken name to calibrax: registering under
  a taken name raises `DuplicateRegistrationError` and keeps the class registered first.
  The opifex copies of that check are removed, so the error is that subclass of
  `ValueError` and its message reads `'<name>' is already registered in <Registry>; remove
  it first to replace it.` `ServableModelRegistry` still refuses a class that is not an
  `nnx.Module` subclass with `TypeError`.
- **Breaking:** `OperatorBenchmarkRegistry.auto_discover_operators()` refuses with
  `DuplicateRegistrationError` when any operator it discovers is already registered, naming
  every such operator, and registers none of them. It skipped those names before, so a
  second discovery, or a discovery after registering a built-in operator by hand, passed
  silently. `BenchmarkRunner` discovers only into an empty registry and is unaffected.
- CI no longer installs ripgrep. The three tests that ran `rg` over `src/` only checked that
  removed names (`BlackJAXIntegration`, `_FieldConformalMetadata` and a TODO tag) were
  absent, and are removed. On the `ubuntu-26.04` runner image, `apt-get install ripgrep`
  found no package, so the unit shard failed before its tests ran.

## [0.2.13] - 2026-11-19

### Added

- `OperatorBenchmarkRegistry.set_operator_metadata(operator_name, key, value)` sets one metadata
  key of a registered operator and refuses an unregistered name with `ValueError`. Registering
  and annotating are separate operations: an operator the runner discovered gets its
  `operator_type` from this call, never from a second `register_operator` of the same name.

### Fixed

- `opifex.core.quantum.ao_layout`, `opifex.data`, `opifex.benchmarking.qh9` and
  `opifex.neural.quantum.hamiltonian` import with numpy 2.5. Their jaxtyping annotations
  wrapped numpy's `NDArray[...]` (`Int[NDArray[np.int64], " n"]`), which numpy 2.5 defines
  as a generic type alias that jaxtyping 0.3.11 cannot unpack, so the import raised
  `TypeError: issubclass() arg 1 must be a class`. They now use jaxtyping's documented numpy
  array type with the dtype in the jaxtyping class (`Int64[np.ndarray, " n"]`), which also
  makes a runtime check refuse an array of another dtype.

### Changed

- The clean-environment build check installs the wheel with the `dev` and `test` extras from
  PyPI and imports every module through `scripts/check_every_module_imports.py`, where it
  imported `opifex` and `opifex.core.training.trainer` alone. The unit tests run on the lock,
  so this step is the one that meets the dependency versions a fresh install resolves.

## [0.2.12] - 2026-11-18

### Added

- `NonlinearLeastSquaresProblem`, `LevenbergMarquardt`, `LinearSolverKind`, `ScalingKind`, and `levenberg_marquardt` in `opifex.solvers.nonlinear_least_squares` (`solver:LevenbergMarquardt` in `SOLVER_CAPABILITIES`): pure-JAX Madsen-Nielsen / Moré damped Gauss-Newton solver (`"qr"` Householder QR, `"cholesky"`, and matrix-free `"lsmr"` via `opifex.uncertainty.linalg.lstsq.lsmr`) on `jax.lax.scan` with `jax.lax.custom_root` Pearlmutter HVP implicit differentiation and zero `equinox`-stack dependencies.
- `InitialValueProblem`, `Rodas4`, `Ros34PW2`, `Kvaerno3`, `GridKind`, and `solve_stiff_ode` in `opifex.solvers.stiff_ode` (`solver:Rodas4`, `solver:Ros34PW2`, and `solver:Kvaerno3` in `SOLVER_CAPABILITIES`): pure-JAX Hairer & Wanner (1996) 4th-order `Rodas4`, Rang & Angermann (2005) 3rd-order `Ros34PW2` W-method, and Kværnø (2004) 3rd-order `Kvaerno3` ESDIRK solvers built on `opifex.core.solver.integrate` with `jax.checkpoint` reverse-mode AD and uniform/geometric grids.
- `CensoredSplitConformalRegressor`, `CensoredConformalState`, `signed_lower_residual_score`, `effective_sample_size`, and `ipcw_conformal_quantile` in `opifex.uncertainty.conformal` (plus specification in `docs/clean_room_specs/censored_conformal_prediction.md`) for right-censored IPCW and doubly-robust lower prediction bounds (Candès, Lei & Ren 2023; Tibshirani et al. 2019; Sesia & Svetnik 2025).
- `censored_gaussian_log_likelihood`, `tobit_nll`, `satisficing_censored_deficit`, `satisficing_censored_loss`, and `LIKELIHOOD_CAPABILITIES` in `opifex.uncertainty.likelihoods` for exact, left-censored, right-censored, and two-sided interval observations with cancellation-free Gauss-Legendre narrow-interval quadrature and `log_ndtr` + `log1mexp` tail symmetry.
- `correlated_difference_predictive` and `linear_contrast_predictive` in `opifex.uncertainty.distributions` for cross-covariance-aware pairwise differences and multi-output linear contrasts, with zero-safe `PredictiveDistribution.std()` gradients at zero variance.
- `AcquiredBatch.auxiliary_scores` (`jax.Array | None = None`) on `opifex.uncertainty.active.AcquiredBatch` so traced batch acquisitions can return per-candidate pool scores without concrete Python metadata conversions.
- `opifex.uncertainty.statespace.cakf_filter` and `cakf_truncate`: the computation-aware Kalman
  filter loop (predict, update, truncate, carrying the mean and the low-rank factor) and the
  factor truncation, ported from ComputationAwareKalman.jl's `filter!` and `truncate`
  (Pförtner et al. 2024). On scalar observations with one CG iteration, and with `max_iter` equal
  to the observation dimension, the filter equals `kalman_filter`.

### Changed

- The example directories `neural-operators`, `quantum-chemistry`, `domain-decomposition`,
  `advanced-training` and `getting-started` are `neural_operators`, `quantum_chemistry`,
  `domain_decomposition`, `advanced_training` and `getting_started`, as substrax's example guide
  names a path below `examples/` in snake_case; every reference to an example's source moves with
  them, and the compute jobs and budgets name the new paths. The hand-written pages under
  `docs/examples/` keep their paths until generated pages replace them.
- Every example's markdown cells are triple-quoted strings, as substrax's guide writes them: 124
  cells written as comments in 19 examples change form, their text unchanged; `data/` examples
  open with their title cell, without a shebang line; `data/darcy_flow_analysis.py` reads no
  command-line arguments, its sizes being named constants, and `main()` takes none.
- The examples make typed keys with `jax.random.key`, never `jax.random.PRNGKey` (32 examples),
  and each random draw has its own key: `quantum_chemistry/neural_xc_functional.py` drew the
  centre count and the electron count from one key, `uncertainty/calibration.py` its features,
  noise, labels and targets, and its predictions and uncertainties, and
  `advanced_training/ntk_analysis.py` sampled a key it then split.
  `data/darcy_flow_spectral_analysis.py` generated its data with seed 0 whatever key it was given,
  the first word of a raw key below 2**32; it now takes the seed itself, 42.
- Each example's `main()` prints its summary and returns it, and its last cell ends with a bare
  `if __name__ == "__main__": main()` (56 examples), so a notebook run keeps the value
  `main()` returns: the five uncertainty result tests read it rather than `None`.
- Every example opens with substrax's metadata table, directly under its title: Tier, Devices and
  Prerequisites. Tier follows the old Level (Advanced is an Advanced Guide; Intermediate and
  Beginner are Tutorials) or, for the 20 examples without one, the guide's definitions; Devices is
  CPU-compatible for the 39 examples that finish within 5 minutes on a CPU runner and GPU-optional
  for the other 20; Prerequisites names the knowledge each example assumes and, for
  `atomistic/nequip_md17.py`, the rMD17 aspirin dataset (154 MB). The Level, Runtime, Memory and
  Format rows are gone: a page records its run's real duration and device instead.
- Require `calibrax>=0.1.20` and `substrax>=0.1.28`; the `test` extra installs
  `substrax[notebooks,testing]`, which brings the notebook kernel and converter, so `ipykernel`
  and `jupytext` leave `dev`.
- The `[tool.substrax.compute]` example jobs take substrax 0.1.27's layout: `resources`,
  `command_timeout_seconds` and `examples_mode = "test"`, and run on `["cuda", "cpu"]`, since with
  `cuda` alone jax starts no CPU backend for a host callback. `[tool.substrax.examples]` holds the
  examples' budgets: a Bazel timeout category from the nightly run's measured time for the 36
  examples that finished in it; the other 23 have no measured time yet and fall back to
  `budget_seconds`.
  `tests/packaging/test_compute_jobs.py` resolves every job as `substrax-compute` does.
- The example tests are substrax's shared ones, run by `substrax-notebooks test`;
  `tests/examples/test_examples.py` is removed, `--ignore=tests/examples` keeps the directory out
  of the default pytest run, and the uncertainty result tests read each example's run through the
  `example_run` fixture.
- The documentation build sets `validation.nav.omitted_files: warn`, so under `--strict` a page
  left out of the navigation fails the build; `development/nequip-symmetric-contraction.md`, the one
  such page, joins the Development section.
- ruff leaves `examples/**/*.ipynb` to `substrax-notebooks convert` (`extend-exclude` with
  `force-exclude`, so a hook naming a notebook skips it too) and lints the scripts;
  `trailing-whitespace` leaves the generated pages under `docs/examples/` alone.
- Update `README.md` Key Features (neural operators, PINNs/KANs, differentiable PDE/stiff-ODE/nonlinear solvers, atomistic & quantum chemistry, equivariant geometry, uncertainty quantification, and active learning), installation extras (`opifex[cuda12]`), and Basic Usage snippets (`jax.random.key`, `jax.lax.scan` Lorenz RK4 trajectory for SINDy, and `SplitConformalRegressor`).

### Removed

- `docs/development/example-documentation-design.md`. Examples follow substrax's
  [Writing examples](https://substrax.readthedocs.io/en/latest/guides/writing-examples/) guide,
  which the Development navigation and the examples index link instead.
- `scripts/jupytext_converter.py`, `.jupytext.toml`, and the `nbstripout` and `nbqa-ruff` hooks,
  which only ever reached the example notebooks; `substrax-notebooks convert` writes those.
- `mkdocs-ipynb` and `mkdocs-jupyter` from the `docs` extra, and the commented-out
  `mkdocs-jupyter` plugin from `mkdocs.yml`: no page is a notebook, and the strict build passes
  without either installed.

### Fixed

- Rewrite `opifex.uncertainty.active.batch_bald` using Kirsch, van Amersfoort & Gal (2019, Algorithm 1) incremental joint log-probability caching over `jax.lax.scan` and `jax.vmap` (`O(B C M K)` with static `(num_mc_samples, K)` state, dtype-scaled `eps^2` variance floor, and traced `AcquiredBatch.auxiliary_scores`), eliminating Python `float(...)` branches, growing `(t + 1,)` slices, and `ConcretizationTypeError` under `jax.jit` and `jax.vmap`.
- `HyperbolicManifold.geodesic_distance` measures with calibrax's `poincare_distance`,
  `2 asinh(sqrt(t))`, scaled by the radius, in place of its own `arccosh(1 + 2t)`: a pair
  1e-5 apart read 0 in float32 and now reads 2.1e-5, and identical points have a zero gradient
  where they had NaN. `parallel_transport` therefore rescales pairs 1e-7 or more apart that it
  treated as one point. A point projected onto the disk's edge (norm `1 - 1e-6`) is 13.34 from
  the origin, not 14.50, at calibrax's floor on `1 - |x|^2`.
- `cakf_update` clamps `max_iter` to the observation dimension, where iterations beyond it lost
  their column, and stops once the residual norm is at most `max(abstol, reltol * ||y||)`, as the
  reference's `update` does.
- The CAKF example (`examples/uncertainty/statespace/cakf_smoothing.py`) dropped the factor at
  every step and gave CAKF unmasked observation noise, so it reported a 0.73 gap from the exact
  filter; it now runs `cakf_filter` on the masked noise, and its result test asserts the
  reference's property, CAKF equal to the exact filter within `sqrt(eps)` of float32.
- Pass `--timeout=1800` to the `extended-tests` job in `.github/workflows/tests-extended.yml` so nightly slow and integration suites do not inherit `pyproject.toml`'s 300 s per-test timeout, guarded by `tests/packaging/test_ci_workflow_contracts.py`.
- Raise transitive security constraint floors in `[tool.uv] constraint-dependencies` (`tornado>=6.6.1`, `opentelemetry-api>=1.45.2`, `oauthlib>=4.0.1`, `soupsieve>=4.1.1`) and refresh `uv.lock` so `pip-audit` passes cleanly across all extras.

## [0.2.11] - 2026-11-10

### Added

- `opifex.capabilities.UQ_CAPABILITIES`: every uncertainty-quantification capability opifex
  declares, collected once from each package's public table into a read-only mapping, with
  `require_uq_capability(name)` and `assemble(*tables)`, which refuses a name declared twice.
  `opifex.neural.operators.OPERATOR_CAPABILITIES` and `opifex.uncertainty.sbi.SBI_CAPABILITIES`
  are now public.
- `opifex.core.quantum.ao_layout.ao_permutation(atomic_numbers, convention)` and the two
  QH9 AO conventions `PYSCF_DEF2SVP` and `BACK2PYSCF` (Yu et al. 2023), each an
  `AoConvention` of four tables; `native_ao_count` gives a molecule's AO count on the host.
- `opifex.core.solver` is a package exporting the solve protocol. `Problem` is the base
  of a problem family and `Method[P]` a way of solving family `P`, both frozen pytrees;
  `solve(problem, method)` raises `IncompatibleProblemError`, a `TypeError`, for a pairing
  outside the method's family, which a type checker also rejects. `integrate` is the
  shared driver for time-stepping methods: a static number of steps under `lax.scan`,
  saved on a `Snapshots` schedule, each step rematerialised in the backward pass unless
  asked otherwise. `Solution` moves to `opifex.core.solver.solution`.
- `opifex.physics.spectral.solve_advection_diffusion_exact`: the exact Fourier solution
  of `u_t + v . grad u = D lap u` on a periodic domain in one to three dimensions, one
  FFT pair with no time stepping. It is `jit`, `vmap` and `grad` compatible in the
  diffusivity and velocity, and reproduces the closed-form single-mode solution to
  1e-12 in float64.
- A staggered (MAC) layer in `opifex.fields`, differentiable in reverse mode throughout:
  `staggered.StaggeredGrid` with `divergence`, `gradient` and `face_count`, whose
  divergence is exactly minus the adjoint of the gradient under periodic,
  prescribed-velocity and zero-gradient boundaries; `staggered_pressure.project`, a direct
  DFT or DCT-II projection with residual divergence flat at about 3e-07 from 8 to 128
  cells; `staggered_convection.convect`, the energy-conserving divergence form with the
  wall treatment of Sanderse, Verstappen and Koren (2014); `staggered_diffusion.laplacian`
  with `WallVelocity` and `wall_source` for prescribed tangential wall velocities; and
  `staggered_navier_stokes.tendency`, `step`, `integrate` and `stable_step_count`, RK4 with
  a projection at every stage. `CARRIED_BOUNDARIES` and `require_boundary` state which
  boundaries each operation carries.
- `solve_lid_driven_cavity(resolution, nu, lid_velocity, total_time, num_steps)`: a walled
  cavity driven by the regularised lid of Shen (1991), whose step count defaults to the
  scheme's stability limit.
- `opifex.physics.flow_problem.IncompressibleFlowProblem`, which bundles an incompressible
  flow and checks that its grid and wall velocities agree.
- `opifex.core.solver.status` (`Status`, `is_successful`, `message`, `combine_statuses`) and
  `opifex.core.solver.report` (`SolveReport`, `report`, `timed_solve`).
- `shift_status(report)` renders a `ShiftReport` for a reader.
- `TrainingMetrics` gains `best_val_epoch`, `reset`, `update_physics_loss` and
  `update_boundary_loss`.
- `minimum_calibration_size(alpha=...)` -- the smallest calibration size whose
  `1 - alpha` conformal quantile exists, `ceil(1 / alpha) - 1`. Exported from
  `opifex.uncertainty.conformal` so callers can check before fitting.
- `PredictionInterval` and `PredictionSet` are re-exported from
  `opifex.uncertainty.conformal`. They are defined in `opifex.uncertainty.types` with the
  rest of the predictive value objects and are what every conformal calibrator returns;
  both facts were documented and correct, and together left the return type unimportable
  from the module that returns it, so annotating a function that wraps a calibrator failed
  on the natural import.
- `GroupedSplitConformalState` records the calibrated group ids as
  `metadata['calibrated_groups']`, which travels on every `PredictionInterval` `predict`
  returns. `num_groups` alone could not distinguish a sample whose group was absent from
  one that was calibrated.

### Changed

- `UQCapability` and `DefaultStrategy` move from `opifex.uncertainty.registry` (and the
  `opifex.uncertainty` namespace) to `opifex.core.capabilities`. They are metadata that
  every declaring package imports, and reaching them through `opifex.uncertainty` loaded the
  whole uncertainty stack: 1.47 s against 0.87 s for the new module, medians of three
  fresh processes.
- `BayesianSINDy` and `PosteriorOverTerms` move from `opifex.uncertainty.scientific` to
  `opifex.discovery.sindy.bayesian`. The regularised-horseshoe regression is a SINDy
  method -- PySINDy keeps it among its SINDy optimizers -- and from `uncertainty` it
  imported the discovery package above it. It is not re-exported from
  `opifex.discovery.sindy`, so importing SINDy does not load BlackJAX.
- `evaluate_examples`, `evaluate_qh9_test_set` and `QH9TestSetMetrics` move from
  `opifex.neural.quantum.hamiltonian` (and its `qh9_eval` module) to
  `opifex.benchmarking.qh9`. They join the predictor to the QH9 dataset, which put a
  data-layer import inside the network package; the per-molecule metrics,
  `predict_fock` and the checkpoint helpers stay where they were, and
  `FOCK_METRIC_KEYS` names the keys `evaluate_fock` returns.
- The def2-SVP AO layout moves to `opifex.core.quantum.ao_layout` (`FULL_ORBITALS`,
  `ORBITAL_MASK`, `atom_orbital_counts`, `block_validity_mask`): the QH9 data sources
  and the Hamiltonian predictor both need it, and it lived in a private module of the
  network package that the data sources imported across the layering. The four names
  are no longer re-exported from `opifex.neural.quantum.hamiltonian`, which keeps
  `BLOCK_IRREPS`, the irreps of a Fock block.
- `radius_graph`, `scatter_sum`, `scatter_mean` and `scatter_max` move from
  `opifex.neural.equivariant` to `opifex.geometry.graph`: they operate on point clouds,
  and `opifex.core` needs the neighbour search, which it could reach only by importing
  the network package above it. Import them from `opifex.geometry.graph`.
- Requires `avitai-artifex>=0.1.15`, for the estimator above. The relock also drops ten
  packages that reached the closure only through the BlackJAX 1.3 chain artifex no longer
  pulls -- `jaxopt`, `python-fasthtml`, `fastcore`, `apsw` and the rest -- and adds none.
- The macOS jobs run on `macos-15`: the package build that pushes to `main` add, and the
  nightly macOS unit lane in `tests-extended.yml`, whose dependency cache is keyed by the new
  runner. GitHub retires the `macos-14` image on 2026-11-02 (actions/runner-images#13518), and
  its scheduled brownouts already cancel jobs that ask for it.
- calibrax `0.1.17`, substrax `0.1.25` and datarax `0.1.17` are the floors, the three being
  the current releases of the sibling set. calibrax 0.1.17 ends `ResourceMonitor`'s window
  with a reading, so a block shorter than one sampling interval still reports its peak and
  growth; substrax 0.1.25 is what calibrax 0.1.17 requires, and its 0.1.24 types an
  example's summary as `JsonValue`; datarax 0.1.17 moves the checkpoint-state protocol to
  substrax's `Checkpointable`, a name this package does not import.
- The unit suite is sharded twelve ways rather than six. The gate's wall clock is one
  shard's wall clock, and measured on run 42874850076 the twelve jobs spent 296 min of
  compute while the critical path was the slowest single shard at 30.7 min against a
  25.6 min mean -- so the work divides almost linearly where the fixed cost, 24 s of
  checkout and `uv sync` per job, does not. Measured after: 16.2 min, for 6% more compute.
  pytest-split still reports no durations file, so the split is by test count and the
  1.20 max-to-mean ratio is unchanged; a `.test_durations` file is the complement to this.
- GP states are registered as pytrees, so `jax.tree` operations see their arrays as leaves
  rather than one opaque node, and the DUE adapter's state flattens with them.

### Fixed

- `conformal_quantile` no longer reports a threshold for a calibration set too small to
  carry one. The finite-sample rank `ceil((n + 1)(1 - alpha)) / n` is a rank of `n` scores
  only while `ceil((n + 1)(1 - alpha)) <= n`; below that no threshold drawn from them
  attains `1 - alpha` and the conformal quantile is `+inf`. The rank was instead clamped to
  1.0, which returns the largest score: finite, indistinguishable from a valid threshold,
  and short of the requested level by construction. Measured marginal coverage was exactly
  `n / (n + 1)` -- 0.673 at `n = 2`, 0.796 at `n = 4` and 0.886 at `n = 8` against a
  requested 0.900, over 20,000 draws each. It is now `+inf`, so the interval is unbounded
  and the shortfall is visible. MAPIE returns `signed * np.inf` on the same condition and
  crepes emits maximum-size intervals; `n` and `alpha` are static, so the branch resolves at
  trace time and jit and vmap are unaffected.
- `GroupedSplitConformalRegressor.fit` warns when a group cannot carry the requested level.
  The guarantee is per group, so each group needs its own minimum however large the whole
  calibration set is, and partitioning is exactly what makes small groups ordinary: a
  203-point calibration set split 200/3 produced 0.747 coverage for the three-point group
  while `predict` reported `coverage=0.9` for both. The undersized groups are now named in a
  `UserWarning` and recorded in `metadata['underfilled_groups']`, as crepes does for its
  Mondrian bins.
- `dominated_hypervolume` is exact in two objectives for any front. The staircase took each
  box's height from that point's own objective 1, so a dominated point left a zero-width
  slot where the tall box belonged and contributed a short box of its own:
  `[[1,3],[2,2],[3,1],[3,3]]` under reference `[4,4]` returned 4.0 against a true 6.0. It
  also measured each width out to the raw coordinate, so a point past the reference widened
  the box *before* it, only the final negative width being clamped away:
  `[[1,3],[6,2],[3,1]]` returned 11.0 against a true 5.0. The height now comes from the
  running minimum and every coordinate is capped at the reference point -- the line the
  general grid path already applied, and the traceable form of BoTorch's `better_than_ref`
  row filter. Over 2,000 randomised two-objective fronts on a dyadic grid, where every
  partial sum of both paths is exactly representable in float32, the two paths now agree
  bit-exactly on all of them; before, they disagreed on 920.
- `GroupedSplitConformalRegressor.predict` gives an unbounded interval, not a zero-width
  one, for a group absent at `fit` time. The per-group gather is a one-hot match, so an
  absent group's row is all-`False` and its sum was `0.0` -- indistinguishable from a
  legitimately tiny quantile, and reported at `coverage = 1 - alpha`, so a point prediction
  was presented as a 90% interval. Such a row is now `+inf`, the same statement
  `conformal_quantile` makes for a calibration set too small to carry the level. The
  condition is per sample and data-dependent, so unlike that function's static branch it
  cannot resolve at trace time; `jnp.where` keeps the gather jit- and vmap-compatible.
  MAPIE 0.9's `MondrianCP` refused an unseen group eagerly, which a traced function cannot
  do, and crepes returns the same zero-width interval (`henrikbostrom/crepes#54`, open).
- GP hyperparameters are traceable, so `exact_gp_loocv_log_predictive` can be
  differentiated with respect to the quantities it exists to select. `fit_exact_gp` raised
  `TracerBoolConversionError` under `jax.jit` on its `if noise_std <= 0.0` guard, so the
  leave-one-out objective of Rasmussen & Williams 2006 §5.4.2 could not be used for the
  model selection that section is about.
- `examples/data/darcy_flow_spectral_analysis.py`'s `main()` returns finite scalar metrics
  rather than the analysis dict. That dict carries the `(50,)` spectra the plots need, and
  the example harness encodes a return with `json.dumps`, converting only 0-d arrays, so it
  refused `spectral_data`; the example's test had failed in every scheduled Extended Tests
  run since 2026-11-03. The analysis function is unchanged and still returns its arrays.

## [0.2.10] - 2026-09-25

### Fixed

- The BlackJAX backend reports an effective sample size that can be believed. Its own
  estimator truncated Geyer's initial positive sequence at `min(n // 2, 64)`, which has no
  basis in the method and biases the estimate down, then clamped the result to `[1, n]`.
  The clamp hid both ends: a chain whose draws never move is worth no draws and read the
  chain length instead, and a chain that alternates about its mean is legitimately worth
  more than its length and was cut back to it. It now comes from
  `artifex.generative_models.core.sampling.effective_sample_size`, which is the same place
  the samplers come from.
- `solve_navier_stokes_2d` returns a divergence-free field. Its Jacobi solve inverted the
  compact five-point Laplacian while the correction subtracted a two-point central gradient,
  so the projection left most of the divergence in place: a divergent start still measured
  0.31 after a step, and an evolved Taylor-Green vortex 6.4e-03. Both are now at the
  round-off of the difference that measures them, 4.7e-10 and 2.4e-07. Trajectories
  generated through `opifex.data.sources.pde_generation` before this carry the old field.
- The pressure solves project. Both inverted an operator other than the one the projection
  applies -- `pressure_solve_spectral` the continuous symbol and the Jacobi solve the compact
  three-point `laplacian`, while the correction subtracts the two-point central `gradient`,
  whose composition with `divergence` reaches `i +/- 2`. Both now invert
  `divergence(gradient(.))` itself, which takes the residual divergence of a 32x32 periodic
  field from 2.3% of the incoming divergence to 8e-07.
- `lsmr` returns NaN once the bidiagonalisation is exhausted, which a zero right-hand side
  reaches immediately and an exactly solvable system reaches after a few steps past `rank`:
  the recurrence divided by the lengths the rotations leave on the diagonal, and those are
  zero there. Those divisions and `jnp.linalg.norm`'s NaN gradient at the origin are guarded,
  so an exhausted iterate holds still and stays differentiable.
- `Burgers2DSolver.solve` integrates on the device. It read the CFL step back to the host
  every sub-step and grew its trajectory in a Python list, so it could be neither jitted nor
  vmapped; it now sub-steps under `lax.while_loop` between fixed save times gathered by
  `lax.scan`, the structure `solve_burgers_2d` already uses.
- `solve_navier_stokes_2d` uses the operators and the projection in `opifex.fields` instead
  of its own copies of gradient, divergence, Laplacian and the Poisson solve. The copies
  were bit-for-bit identical to the shared ones and compile to the same kernel, so the
  duplication bought nothing and hid the defect above from the fix in `fields`.
- `pressure_solve_lsmr(velocity, num_matvecs=500)` replaces `pressure_solve_jacobi`. The
  composed operator is neither symmetric nor invertible under a zero-gradient boundary, where
  a relaxation or a conjugate-gradient iterate drifts into its null space and diverges; a
  least-squares solve takes the minimum-norm solution instead. Prefer
  `pressure_solve_spectral` for periodic fields, which is about a hundred times faster.
- `Burgers2DSolver.solve(initial_condition, time_final, num_saves=1)` replaces `save_every`:
  the saved times are `num_saves + 1` equally spaced values, which a traced solve can shape
  its output around, where a count of adaptive steps cannot.
- `get_activation("mish")` is `jax.nn.mish`; the local copy computed the same values.
- An unknown activation name is refused with the names that are registered.
- `Solution` is a `flax.struct.PyTreeNode` of arrays, so it can be built inside `jit` and
  returned through `jit`, `vmap` and `grad`. An `int32` `status` replaces `converged`, and
  `is_converged` returns a boolean array with one answer per batch element; a Python
  bool could not be produced from a traced comparison, and as static metadata it
  compiled one program per outcome. `execution_time` is removed: a clock read inside
  `jit` is evaluated once at trace time, so it reported the first compilation's duration
  on every later call. Time a solve with `timed_solve` instead.
- A solver is `Solver = Callable[[Problem], Solution]`, which the type checker verifies
  against the signature. It replaces the `SciMLSolver` protocol, which `isinstance`
  accepted for any object with a `solve` attribute. `SupportsSolve` remains for solvers
  held as objects.
- `additive_hybrid(classical, neural)` replaces the `HybridSolver` class. The combined
  solver runs under `jit`, `vmap` and `grad`, since it no longer calls `float()` on device
  arrays, and it keeps the first unsuccessful status of its two parts. Its capability is
  registered as `solver:additive_hybrid`, a JAX function rather than an NNX module.
- `NeuralOperatorSolver` given a problem with no data reports `Status.DEFAULT`; it
  reported success unconditionally. Ensemble aggregation combines member statuses where
  it took `all()` of flags, and no longer sums the members' execution times.
- `SCFResult.status` replaces `converged`, and `n_iterations` is an array, so a result can
  be built inside a transform. The direct-minimisation path reports the minimiser's own
  outcome. It returned `converged=True` with the iteration budget as its step count, even
  when stopped 0.033 Ha above the fixed point. `SCFSolver` refuses a convergence tolerance below
  `4 * eps` of the working precision. The float32 default of 1e-8 could never be reached,
  so the solve spent its whole budget and reported `MAX_ITERS`.
- `ShiftReport.passes` is a `jax.Array` leaf, and the status strings come from
  `shift_status` instead of static metadata, so passing and failing reports are one pytree
  structure that compiles once and batches under `vmap`.
- `AdvancedMetricsCollector` is renamed `MetricsCollector`. `TrainingMetrics` series are
  read-only tuples recorded through their `update_*` methods, the regression figures and
  best-value tracking come from `calibrax.metrics`, and `update_learning_rate` takes
  `learning_rate`.
- `solve_navier_stokes_2d` and `solve_burgers_2d` take a `substeps` count.

### Removed

- `opifex.uncertainty.registry` with `UQRegistry` and `register_uq_capability`,
  `register_mlops_capabilities`, `register_bayesian_capabilities`, `bayesian_uq_registry`,
  `register_l2o_capabilities` and `opifex.neural.operators.OPERATOR_CAPABILITY_REGISTRY`.
  Eighteen packages filled the singleton at import, three more through explicit calls, and
  nothing in the library ever read it; the test suite needed an autouse fixture snapshotting
  and restoring it around every test to stop leaks between tests. Read
  `opifex.capabilities.UQ_CAPABILITIES` or call `require_uq_capability(name)` instead;
  `get_operator_capability` keeps its contract.
- `ElectronicStructureProblem.compute_energy`, `compute_forces` and `scf_solver`, and the
  abstract `compute_energy`/`compute_forces` of `QuantumProblem`: a problem states what to
  solve and no longer solves itself. The methods duplicated `SCFSolver`, reached it through
  a `TYPE_CHECKING` block and function-level imports that hid an import of the network
  package from `opifex.core`, and silently computed with LDA whenever the requested
  functional was not a PBE variant, `neural_xc` included. As a PySCF molecule is solved by
  `RKS(mol, xc=...)`, use
  `SCFSolver(problem.molecular_system, functional=...).energy_and_forces()`, naming the
  functional. `QuantumProblem` is no longer abstract.
- `opifex.data.sources.qh9_source.def2svp_decode_indices`: use
  `ao_permutation(atomic_numbers, PYSCF_DEF2SVP)` from `opifex.core.quantum.ao_layout`. The
  QH9 convention tables were held twice, once in the data source and once in the QH9
  evaluation, and the permutation built from them written out twice.
- The console commands `opifex-check-compatibility`, `opifex-env-activate`,
  `opifex-fresh-setup`, `opifex-gpu-test`, `opifex-gpu-utils`, `opifex-gpu-verify` and
  `opifex-verify-examples`. Each pointed into the repository's `scripts/` directory, which
  the wheel does not contain, so every one failed at start with `ModuleNotFoundError`; five
  of the modules they named did not exist at all. The GPU tools remain repository scripts,
  run from a checkout as `python scripts/<name>.py`.
- `get_derivative_activation`, which had no caller and returned a boolean array for ReLU;
  `jax.grad` differentiates any activation in the registry.
- `SciMLSolver`, `HybridSolver` and `ArtifexSolverAdapter` with the
  `opifex.solvers.adapters` package; every adapter method raised `NotImplementedError`.
- `create_lid_driven_cavity_ic`, which imposed a smooth profile as an initial condition in
  a periodic box. With no wall to drive it the flow decays instead of recirculating, and
  it peaked at 0.679 when asked for a lid velocity of 1.0. Use `solve_lid_driven_cavity`.
- `MPCResult.computation_time`. It was read inside an `nnx.jit`-compiled method, so it
  held the tracing cost: 0.606 s against a real call of 0.0006 s, on every call. Time
  the step with `calibrax.profiling.time_calls`.

### Fixed

- `atom_orbital_counts` and `block_validity_mask` gave any element above fluorine
  fluorine's fourteen AO slots: the lookup used JAX's default gather, which promises
  in-bounds indices and clamps the rest to the last row, and it wrapped negative numbers
  to it too. Elements outside the basis now populate nothing, under `jit` and `vmap`.
- The QH9 AO count no longer counts any element heavier than helium as fourteen orbitals;
  it reads the layout and refuses an element outside it with `KeyError`, as the AO
  permutation already did.
- `opifex-benchmark` runs the benchmarking CLI. It pointed at `scripts.benchmark_runner`, a
  module that does not exist and would not have been in the wheel, instead of
  `opifex.benchmarking.cli`, which declares itself as that command.
- `generate_diffusion` returns finite, correct labels at its shipped resolution. It
  stepped forward Euler with a fixed `dt = 0.01`, whose diffusive limit `D < dx^2 / (4 dt)`
  is 0.0061 at 64 cells -- below the whole sampled range of 0.01 to 0.1 -- so 0% of the
  default output was finite; its test ran at 16 cells, where the limit covers the range.
  Even at a stable step the scheme's first-order upwind advection left the labels 39-46%
  wrong in the L2 norm. The labels are now the exact Fourier solution at a stated
  `time_final` (default 1.0, the time the old step count implied), with the mean
  conserved to round-off and no stability limit at any resolution. The initial
  conditions are now periodic and sampled at `x_j = j / N` (they were sampled on
  `linspace(0, 1, N)` while the solver assumed spacing `1 / N`): the Gaussian is wrapped
  on the torus, the wave has an integer number of periods, and the centre, width and
  size of a bump are drawn independently instead of from one shared uniform number.
- `solve_navier_stokes_2d` solves the equations it names. Its advection was first-order
  upwind, whose numerical viscosity of `|u| dx / 2` stood an order of magnitude above a
  physical `nu` of 0.01 at 32 cells, so a Taylor-Green vortex decayed 8% too fast at
  `t = 1` and 29% at `t = 5`, with the error falling only by half whenever the grid did.
  It now integrates on a staggered layout whose convective term produces no energy and
  whose projection is exact, and the decay error is 6.4e-05 and 3.2e-04 at those times --
  second order, with the relative error times `n^2` flat at 0.066 per unit time across 16
  to 128 cells. The strict xfail that held the requirement is removed.
- `solve_navier_stokes_2d`, `solve_burgers_2d` and `Burgers2DSolver.solve` differentiate
  in reverse mode. Each stepped adaptively to its CFL limit inside a `lax.while_loop`,
  which traces and maps but carries no transpose rule, so `jax.grad` of anything reaching
  them raised outright -- a correctness defect in a differentiable-physics library, and
  one that `jit` and `vmap` tests cannot see. They take a fixed sub-step count under
  `lax.scan` instead; `stable_substep_count` and `stable_step_count` derive that count
  from the same Courant condition the adaptive versions enforced.
- `_legendre_basis` and `_hermite_basis` differentiate in reverse mode, for the same
  reason and with no behaviour change: their trip count is the requested degree, a Python
  int, so `lax.fori_loop` was always the right construct. Values move by 5.96e-08, which
  is float32 round-off, and still match `numpy.polynomial` to 6.3e-07.
- The BlackJAX backend reports an effective sample size that can be believed. Its own
  estimator truncated Geyer's initial positive sequence at `min(n // 2, 64)`, which has no
  basis in the method and biases the estimate down, then clamped the result to `[1, n]`.
  The clamp hid both ends: a chain whose draws never move is worth no draws and read the
  chain length instead, and a chain that alternates about its mean is legitimately worth
  more than its length and was cut back to it. It now comes from
  `artifex.generative_models.core.sampling.effective_sample_size`, which is the same place
  the samplers come from.
- `opifex.uncertainty.linalg.lsmr` is the LSMR recurrence of Fong and Saunders (2011, SIAM J.
  Sci. Comput. 33(5), 2950): a bidiagonalisation kept factorised by two Givens rotations per
  step. It projected onto the Krylov basis and then solved the normal equations there, which
  squares the condition number: on a well-conditioned 30x6 system it missed SciPy's solution
  by 1.7e-3 where the recurrence misses it by 8e-8. The solutions now agree with
  `scipy.sparse.linalg.lsmr`, the authors' own implementation, at every iteration budget.
  Its docstring records what precision the iteration needs: a 60x10 system of condition 1e4
  converges in 22 float64 steps, while in float32 the iterates drift away.
- `get_activation("prelu")` returned `jnp.maximum`, which raises when called with one
  argument. PReLU learns its negative slope, so it is `flax.nnx.PReLU`, a module, and the
  name now says so instead of resolving to a broken function.
- `list_activations()` listed fourteen names while `get_activation` accepted twenty-five,
  hiding the broken entry; both now read the one registry.
- `ks_two_sample_pvalue` returned zero for identical samples, so the exchangeability check
  reported a shift for data compared against itself. The alternating series needs terms
  up to `k > 4.29 / lambda`, so a hundred terms were not enough below about 0.007. It now
  splits at the median between that series and the theta form (van Mulbregt 2018,
  sec. 4.1), and matches `scipy.special.kolmogorov` to 1.14e-07 over `[0, 5]`.
- The Anderson mixer's Tikhonov stabiliser in the SCF solver works at both ends of a
  solve. While the history filled, its scale came from identity padding and averaged the
  Pulay coefficients to 0.4988 where the answer is 0.0099. Once the history filled, its
  fixed 1e-10 factor sat below float32 epsilon, and the H2 density returned NaN. The scale
  is now the mean squared residual over filled slots and the factor is the square root of
  the working precision's epsilon.
- `opifex.optimization.production` measured forward-pass latency without synchronising,
  timing the dispatch rather than the work: 0.55 ms against a true 67.8 ms on a compiled
  2048-wide model, the figure compared with `target_latency_ms`. It measures through
  `calibrax.profiling.time_calls`. `apply_latency_optimization` compiled every model at a
  hardcoded width of 64 and now reads the model's input width.

### Changed

- Requires `datarax>=0.1.16`; the relock moves datarax from 0.1.15 and nothing else.
  - No batch is padded any more. A loader's validation split serves every record once, and
    its final batch holds the records left instead of wrapped-around repeats. The benchmark
    executor therefore scores every row it is given, and `valid_mask` is gone from batches
    and docs.
  - A training split smaller than `batch_size` is refused when the loaders are built. It
    used to yield no batches.
  - `PDEBenchSource` and `VTKMeshSource` implement `get_records(indices)`. Their shuffle is
    datarax's keyed bijection (`resolve_wrapped_indices`), O(1) per record. It replaces a
    `jax.random.permutation` of the whole dataset drawn on every batch, and per-record
    randomness is now keyed on the record rather than its position.
- Three examples that drained loader pipelines into arrays now generate the arrays with
  `generate_darcy`, and so train and test on the samples they ask for. They had been
  training and testing on repeated and missing samples:
  - `getting-started/first_neural_operator` trained on 1024 rows for 1000 samples (the
    validation split's padded final batch was appended), and its recorded test sets held 200
    rows for 100 samples.
  - `uncertainty/bayesian_fno` asked for its whole generation in one batch. It trained on
    the 200 validation samples wrapped around to 1000 rows, with 20 and 20 distinct samples
    in its 100-row calibration and test splits.
  - `neural-operators/uno_darcy` tested on 128 rows for 100 samples. Its "independent"
    super-resolution set, drawn from `seed + 1`, repeated training samples 1-100; it now
    starts after the last training and test seed.

  datarax 0.1.16 refuses the last two setups, and all three pages' recorded outputs come
  from the old data.
- Requires `avitai-artifex>=0.1.15` and `calibrax>=0.1.14`.
  - artifex 0.1.14 supplies the effective sample size above, and 0.1.15 runs on datarax 0.1.16
    (its `Trainer` evaluates unpadded batches). The artifex relock also dropped ten packages
    that reached the closure only through the BlackJAX 1.3 chain artifex no longer pulls --
    `jaxopt`, `python-fasthtml`, `fastcore`, `apsw` and the rest -- and added none.
  - calibrax 0.1.13 ranks tied values by their average rank in `spearman_rank_correlation`,
    which `active_learning_acquisition_reliability` reports, and takes roofline memory traffic
    from XLA's bytes accessed, which the profiling harness's roofline analysis reads.
- `typing_extensions` is no longer a declared dependency; nothing in opifex imported it.
- `solve_navier_stokes_2d` uses the operators and the projection in `opifex.fields` instead
  of its own copies of gradient, divergence, Laplacian and the Poisson solve. The copies
  were bit-for-bit identical to the shared ones and compile to the same kernel, so the
  duplication bought nothing and hid the defect above from the fix in `fields`.
- `pressure_solve_lsmr(velocity, num_matvecs=500)` replaces `pressure_solve_jacobi`. The
  composed operator is neither symmetric nor invertible under a zero-gradient boundary, where
  a relaxation or a conjugate-gradient iterate drifts into its null space and diverges; a
  least-squares solve takes the minimum-norm solution instead. Prefer
  `pressure_solve_spectral` for periodic fields, which is about a hundred times faster.
- `Burgers2DSolver.solve(initial_condition, time_final, num_saves=1)` replaces `save_every`:
  the saved times are `num_saves + 1` equally spaced values, which a traced solve can shape
  its output around, where a count of adaptive steps cannot.
- `get_activation("mish")` is `jax.nn.mish`; the local copy computed the same values.
- An unknown activation name is refused with the names that are registered.

### Removed

- `get_derivative_activation`, which had no caller and returned a boolean array for ReLU;
  `jax.grad` differentiates any activation in the registry.

### Known issues

- `solve_navier_stokes_2d` dissipates faster than the equations it solves: the advection
  term is first-order upwind, whose numerical viscosity is `|u| dx / 2`, an order of
  magnitude above a physical `nu` of 0.01 at 32 cells. A Taylor-Green vortex decays 8%
  faster than `exp(-2 nu t)` at `t = 1` and 29% faster at `t = 5`, and the error falls by
  half whenever the grid does, which is the first-order convergence that names the cause.
  `test_the_vortex_decays_at_the_analytic_rate` holds the requirement as a strict xfail.

## [0.2.9] - 2026-09-21

### Security

- The lock moves anyio from 4.12.1 to 4.14.2 for CVE-2026-63374 and CVE-2026-64847; nothing else moves. 4.14.2 is the first fixed release; 4.15.1 needs typing-extensions 4.16.0, which a single-package upgrade does not allow to move.

### Changed

- Requires `calibrax[publication]>=0.1.12`, `substrax>=0.1.16`, `avitai-artifex>=0.1.13` and
  `datarax>=0.1.15`, the latest releases; the lock moves those four and lazy-loader, nothing else.
  calibrax 0.1.11 removed the APIs opifex 0.2.8 imported, so on a fresh install 0.2.8 lost
  `opifex.benchmarking` and `opifex.uncertainty.conformal` at import and failed in timing and
  hardware detection at run time; this release moves every call site onto calibrax's current API.
- `bootstrap_threshold_ci(*, samples, key, confidence=0.95, bootstrap_resamples=1000)` takes the
  resampling key (a `jax.Array` or an `nnx.Rngs` with a `sample` or `default` stream) and is
  `calibrax.statistics.bootstrap_interval` of the mean; `seed` is removed. It traces under `jax.jit`
  and `nnx.jit`, and maps under `vmap` over keys.
- `EvaluationEngine.generate_summary_report(*, key)` takes the key its bootstrap intervals are drawn
  from; `profile_model_performance` summarizes with `calibrax.statistics.summarize` and returns
  floats.
- `opifex.benchmarking` no longer re-exports `StatisticalAnalyzer`, which calibrax removed.
- `ResultsManager` draws its comparison, scaling and convergence plots with calibrax's
  `PlotGenerator`, from the `calibrax[publication]` extra opifex now requires.
- Recorded benchmark metadata is read through `calibrax.core.read_metadata_entry`: a value of the
  wrong type is refused naming its key instead of being used as given.
- The roofline, hardware detection and the profiling harness read
  `calibrax.profiling.resolve_hardware_spec(dtype=jnp.float32)`: the listed spec of the visible
  devices, or an unlisted device's ceilings measured once per process. The profiling harness reports calibrax's roofline and
  compilation results under the keys calibrax writes (execution time and cache hit rate were read
  from keys calibrax does not write and always showed 0), and its hardware section names the spec
  and the device count.
- Checkpoints: under substrax 0.1.16 the store reads the one format it writes; a root written
  through the module-only layout of earlier releases is refused with `UnsupportedCheckpointError`.

## [0.2.8] - 2026-09-18

### Changed

- Requires `avitai-artifex>=0.1.12`; the lock moves it from 0.1.11 and nothing else. opifex imports
  artifex's sampling, distributions, conditional flow and rematerialisation helpers, which 0.1.12
  leaves as they were; its changes are the trainer's checkpointing and its floors.
- Requires `substrax>=0.1.11` (and `substrax[mlflow]>=0.1.11`); the lock moves it from 0.1.10 and
  nothing else. 0.1.11 caps jax below 0.11.2, whose renamed `jax.experimental.hijax.HiPrimitive`
  flax 0.12.9 imports at module load; a resolver given `substrax>=0.1.10` keeps jax 0.11.2 and
  picks 0.1.10 instead, so a fresh install of opifex resolved the failing pair until the floor
  moved. The build-verification smoke now installs the wheel with `--refresh` and imports
  `opifex.core.training.trainer`, which loads the stack; `import opifex` alone never did.
- CI: the unit suite's macOS lane runs nightly in the extended workflow, sharded under a cap
  of three runners at a time, and the push and pull-request gate runs on ubuntu alone. A push
  to main used to add twelve macOS-14 shards that held the organisation's macOS runners for
  hours and queued every other repository's macOS job behind them. Every workflow a push
  triggers now cancels the run a newer push supersedes.
- Checkpoints are substrax's format 3. `Trainer.save_checkpoint(step, loss,
  physics_metadata=None)` saves the model's state as the `model` item with the trainer's
  epoch, the loss as the `loss` metric, opifex as the producer and the physics values in the
  record's extra, and returns the checkpoint's directory. `Trainer.load_checkpoint(step)`
  restores that state into the live model and returns substrax's `Checkpoint`; a step the
  directory does not hold raises `CheckpointNotFoundError` (it returned `(None, {})`), and a
  trainer without a store raises `ValueError`. Checkpoints written by 0.2.7 and earlier load
  through substrax's module-only layout; `python -m substrax.checkpoint upgrade` rewrites
  them.
- `CheckpointConfig.checkpoint_dir` defaults to `None`: a trainer saves nothing unless a
  directory is named (it wrote into `./checkpoints` under the working directory, where every
  default-configured run collided at the same steps). `fit` saves the global step every
  `checkpoint_frequency` epochs, not the epoch index, so a second `fit` on the same trainer
  adds checkpoints instead of rewriting the first one's. `CheckpointComponent` likewise
  creates a directory only when its configuration names one (it made `./checkpoints` under
  the working directory for checkpoints it keeps in memory).
- The serving registry stores a model's weights as the `model` item of its checkpoint and
  restores them onto the abstract state; a missing checkpoint raises the store's
  `CheckpointNotFoundError`, not `RuntimeError`.
- `Experiment.log_model(model, ...)` takes an `nnx.Module` or a state mapping (Flax
  `TrainState` payloads are gone with substrax's `ModelLike`) and records the framework, the
  physics domain and `physics_metadata` in the checkpoint record's extra values.
- Requires `substrax>=0.1.10`, `datarax>=0.1.13` and `avitai-artifex>=0.1.11`; the lock
  moves exactly those three.

### Fixed

- `Trainer.fit` advances the global step once per batch it dispatches. The counter was
  incremented inside the jitted training step, where the Python increment ran only when the
  step was traced, so after any number of epochs `state.step` read 1 and every epoch's
  checkpoint carried the same step. `training_step` itself no longer moves the counter; a
  loop of its own around it advances `trainer.state.step`.

### Removed

- `opifex.core.training.components.checkpoint_store.ModelLike`.

### Changed

- The loader factories' training split drops the epoch's ragged final batch
  (`drop_last=True`, PyTorch's rule), so every training batch holds `batch_size` records and
  `len(loaders.train)` is the batches per epoch; the validation split keeps every record, its
  final batch padded and marked in `valid_mask`. Every batch a pipeline serves carries that
  leaf.
- The operator benchmark executor trains every configured epoch (it reset nothing between
  epochs and trained one), reports `epochs_trained`, and evaluates the records only, leaving
  the padded rows of the last batch out of its MSE, MAE and relative error.
- `iterate_padded_batches(..., drop_last=False)` attaches a per-molecule `valid_mask` to each
  QH9 batch, false for the molecules the final batch wraps, and drops that batch under
  `drop_last=True`; the block-form training script trains with `drop_last=True`, so no
  molecule is trained on twice in an epoch, and weights its validation MAE by the real
  molecule count.
- Requires `datarax>=0.1.12`; the lock moves it from 0.1.11.

### Changed

- The progressive GPU tester times its operations through `calibrax.profiling.time_calls`
  (three warm-up calls, ten timed calls that each wait for their result) and reports its
  median: the mean it reported was moved by one slow call, which a shared CI runner produces
  at will (two consecutive means 3.01x apart on a macOS shard), and the GST-PINN
  self-training loss averages over the reliable points through `calibrax.metrics.reduce_values`,
  so an empty reliable set gives zero as before without a hand-written guard.
- Requires `calibrax>=0.1.8` and `avitai-artifex>=0.1.10`; the lock moves both.

### Removed

- `opifex.core.timing.block_until_ready`; the benchmarking code calls `jax.block_until_ready`,
  which already synchronises every array in a nested value and returns it.

## [0.2.7] - 2026-09-17

### Changed

- `OptimizationConfig` is the one owner of the optimizer's settings and the trainer builds
  its optimizer through `substrax.optim.create_optimizer` from it, with the configured
  schedule as the optimizer's learning rate. A schedule now reaches the update: the
  previous factory chained `scale_by_schedule` before the base optimizer, where Adam's
  normalisation cancelled it, so a decaying schedule never decayed. `TrainingConfig` no
  longer takes `learning_rate`; pass
  `optimization_config=OptimizationConfig(learning_rate=...)`. The rate set there is the
  rate applied (`TrainingConfig.__post_init__` used to overwrite it with its own default),
  and the step metrics report the rate the last update applied. `OptimizationConfig` gains
  the schedule fields `end_value`, `boundaries_and_values`, `peak_value` and `warmup_steps`
  and the clip fields `gradient_clip_norm` and `gradient_clip_value`; a weight decay on an
  optimizer without decoupled decay (`adam`, `sgd`, ...) is refused where it was silently
  dropped, and `momentum` reaches only `sgd` and `rmsprop`.
  `FlexibleOptimizerFactory` reads its mapping into an `OptimizationConfig` (its `grad_clip`
  key is the global-norm clip) and `fit_atomistic` takes a `substrax.optim.OptimizerConfig`.
- Every key drawn at a sampling entry point comes through `substrax.rng.key_from`; an
  `nnx.Rngs` holding none of a call's streams raises `MissingRngStreamError`, naming the
  call, and nothing falls back to a default seed.
- Requires `substrax>=0.1.9` (`substrax.rng`, `substrax.optim`, and an `OptimizerConfig`
  that refuses an unknown optimizer or a non-positive clip at construction),
  `datarax>=0.1.11`, `calibrax>=0.1.6` and `avitai-artifex>=0.1.8`; the lock holds each.
- `BenchmarkRunner`, `BenchmarkEvaluator` and `ResultsManager` resolve their output
  directory through `substrax.artifacts.resolve_output_dir`: with no directory given they
  write under `benchmarks` in `$AVITAI_OUTPUT_DIR` when it is set and in a per-process
  temporary directory otherwise, never `./benchmark_results` in the working tree; a
  directory given is resolved against the working directory. The `output_dir` and
  `storage_path` arguments accept a `Path` and default to `None`.
- Requires `substrax>=0.1.7`, the release that carries `substrax.artifacts`,
  `substrax.runtime` and `substrax.testing`; the lock already held it.
- Every example resolves its output directory through `substrax.artifacts.resolve_output_dir`
  inside `main()`, under the name of its former `docs/assets/examples/<name>` directory:
  with `AVITAI_OUTPUT_DIR` set the figures land in `<name>` under it, otherwise in a
  per-process temporary directory. Running an example no longer writes into the working
  tree, and importing one no longer names a directory. The two Darcy-flow analysis
  examples and the operator benchmark's study take `None` for their directory arguments
  and resolve the same way; a directory given is resolved against the working directory.
  `AVITAI_OUTPUT_DIR="$PWD/docs/assets/examples"` regenerates the documentation figures in
  place.
- The example tests run every example in its own interpreter through
  `substrax.testing.run_example`, on the CPU backend with outputs under a temporary
  directory, and read the summary `main()` returns; the in-process loader is gone. The
  four examples that ran a reduced workload under `OPIFEX_EXAMPLE_SMOKE` (adaptive
  sampling, GradNorm, NequIP on rMD17, the Darcy-flow analysis) run at full size in the
  extended job, and the variable is gone. The per-example budget is the suite's
  `--timeout`, enforced by the runner.
- Importing an example configures nothing: the operator benchmark's logging is set up
  inside `main()` through `substrax.runtime.configure_entry_point_logging`, and the two
  float64 examples (Hamiltonian prediction, VMC atoms) enable 64-bit types and the `high`
  matmul precision inside `main()`, with their module-level geometry as numpy arrays so it
  takes the run's precision.

### Removed

- `opifex.core.training.optimizers.OptimizerConfig`, `create_optimizer`, `create_adam`,
  `create_adamw`, `create_sgd`, `create_rmsprop`, `with_gradient_clipping` and
  `with_schedule`, and `TrainingConfig.learning_rate`. The optimizer is described by
  `OptimizationConfig` and built by `substrax.optim` (`optimizer_spec` maps one onto the
  other; `create_schedule` stays, over `OptimizationConfig`).
- The K-FAC natural-gradient preconditioner for variational Monte Carlo
  (`opifex.neural.quantum.vmc.kfac_preconditioner`) and the `kfac-jax` dependency of the
  `quantum-chemistry` extra. The module wrapped `kfac_jax`, whose published releases stop
  at 0.0.8 and fail to import on JAX 0.11 (`jax.core.Effects` is gone); `VMCDriver` never
  offered it (its optimizers are `adam`, `minsr` and `spring`), and its docstring's claim
  that a unit-variance normal predictive tag makes kfac_jax's Fisher the quantum geometric
  tensor is wrong (kfac_jax defaults the variance to 0.5, and the tensor is centred). The
  optional-backend specs and the curvature capability no longer name kfac-jax. Restoring
  a working preconditioner is tracked in avitai/opifex#31.
- `opifex.setup_jax_optimization`, `opifex.core.configure_jax_precision` and the
  `opifex.core.device_utils` module, with the `OPIFEX_XLA_CACHE_DIR` variable the first read.
  A process declares the JAX settings it starts with through `substrax.runtime`: a
  `JaxRuntime` names the platforms, 64-bit types, matmul precision, compilation cache
  directory and XLA flags, `apply_runtime` applies it in the running process and
  `runtime_environment` renders it for a child. The helper applied a fixed flag set for
  the backend it found and a `.cache/jax` directory in the working tree; a run that wants
  those settings names them in its `JaxRuntime`.

## [0.2.6] - 2026-09-16

### Changed

- Requires `datarax>=0.1.10`; the lock moves datarax from 0.1.9 to 0.1.10 and, through it,
  substrax from 0.1.4 to 0.1.7. datarax's `MapOperatorConfig` no longer takes
  `precomputed_stats` and `DataraxModuleConfig` no longer has `cacheable`: the PDEBench
  normalisation stage passes no statistics (its constants are baked into its function), and
  `QH9PaddedConfig`'s shuffle override copies no `cacheable`, which raised `AttributeError`
  on the first split whose order differed from the config's. The two Fock operators name
  the fourth `apply` argument `key`, the record's PRNG key, which they do not use.
- The `backend:pathfinder`, `backend:svgd` and `backend:advi` capabilities in the UQ registry
  declare `source_package="opifex"`, as `PathfinderBackend`, `SVGDBackend` and `ADVIBackend`
  already do. They declared `"blackjax"`, although the three backends are implemented in opifex.
- The Markov power-EP and posterior-linearisation Student-t predictors,
  `predict_studentst_markov_pep_gp` and `predict_studentst_markov_pl_gp`, record
  `likelihood="students_t"` in their metadata, the label the dense Laplace GP and the Markov
  Laplace and variational predictors already use. They recorded `"studentst"`.
- `StateSpaceKernel` is a pytree. Its leaves are its matrices and the rates or angular frequencies
  of its closed-form transition, so a kernel passed to a jitted function reuses the compiled program
  for new hyperparameter values. `state_transition(dt)` returns the closed-form transition of the
  kernel's SDE, `discretize(dt)` returns the transition and the process noise of one step, and
  `discretize_steps(steps)` returns them for a sequence of steps. The Markov GP paths and the
  spatio-temporal GP discretise each time grid with one `discretize_steps` call. A kernel built from
  `feedback`, `noise_effect`, `diffusion`, `measurement` and `stationary_cov` alone discretises from
  its feedback matrix.
- `mumbo_acquisition` takes an odd `num_quadrature_points`, default 5001, and raises `ValueError`
  for an even count. Composite Simpson's rule needs an even number of intervals; the previous
  default of 1000 weighted the last interval as a half panel. When `sqrt(1 - rho^2)` is below the
  Simpson spacing the skew factor of eq. 5 steps and the quadrature error is first order in the
  spacing, so the default is five times finer.
- `HamiltonianBlockExpansion` learns a bias for every expansion path whose input degree is 0, the
  s-s, p-p and d-d shell pairs, as QHNet does (Yu et al. 2023, arXiv:2306.04922), so `num_bias` is
  14 for the `3x0e + 2x1e + 1x2e` block. Only the 9 s-s paths had a bias, so the p-p and d-d
  on-site and off-site sub-blocks had no term independent of the feature. The readout of each block
  head of `BlockHamiltonianPredictor` gains 5 outputs, so `load_predictor_checkpoint` raises
  `ValueError` for a checkpoint saved before this change.
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

- Citations: DALTON is Wu & Lysy, *Data-Adaptive Probabilistic Likelihood Approximation for Ordinary
  Differential Equations* (AISTATS 2024); DUE (arXiv:2102.11409) is by van Amersfoort, Smith,
  Jesson, Key and Gal and did not appear at ICML 2021; U-NO appeared in TMLR in 2023; and
  arXiv:1810.03440 is Tronarp et al., *Probabilistic Solutions To Ordinary Differential Equations As
  Non-Linear Bayesian Filtering*. The Euler-angle Wigner-D factorisation and its `J_l` tables no
  longer cite Geiger & Smidt (2022), and the FBPINN oscillator example no longer attributes its
  problem to the FBPINN paper; neither paper contains them.
  `DenseOutputSamplingSpec` and `ExpEKSpec` no longer cite §5 of Tronarp et al. (2019), which
  describes neither method, and `scripts/eval_qh9_blocks.py` names canonical orthogonalization
  instead of Löwdin's.
- `scripts/eval_qh9_blocks.py` logs `eps-MAE(occ)`, the orbital-energy MAE over the occupied
  orbitals that QH9 reports (Yu et al. 2023, arXiv:2306.09549), beside `eps-MAE(all)`, and its
  docstring names `orbital_energy_mae_occ` as the benchmark metric. It logged the all-orbital MAE
  as `eps-MAE` and said every reported metric mirrors the benchmark.
- The `kalman_update` docstring names the standard covariance update `P - K H P` that it computes,
  as do the square-root Kalman tests; they called it the Joseph form. The state-space kernel
  references cite Särkkä & Solin (2019) eqs. (12.34)-(12.35) and Example 12.7 in place of a
  Table 12.2, which the book does not have.
- `gibbon` computes the GIBBON acquisition of Moss et al. (2021) at batch size one, Definition 4
  applied to samples of the minimum, and accepts `noise_variance` (default 0.0) for the correlation
  `rho^2 = var / (var + noise_variance)` between an observation and the objective. It returned
  `min_value_entropy_search` on the claim that the two coincide at batch size one; Definition 4 is
  a lower bound on that information gain instead. In float32 the score loses accuracy once a
  sampled minimum lies more than a few standard deviations above the mean without observation
  noise, and is not finite at `gamma = -40`.
- `morris_screening` keeps every trajectory inside `[lower, upper]`. A dimension that stepped down
  started at the base value and moved to `x* - Delta`, below the box: with 5 inputs, about 1450 of
  1500 trajectories left the unit cube for 2 to 8 levels, and a model defined only on the box, such
  as `sqrt`, returned NaN statistics. It now starts at `x* + Delta` and steps down to `x*`, as in
  the random orientation `B*` of Morris (1991), p. 164, so each coordinate takes `x*` or
  `x* + Delta`.
- Pathfinder centres each local Gaussian on `theta + Sigma grad log p(theta)`, as in Algorithm 4,
  line 8 of Zhang et al. (2022). `bfgs_sample` added `Sigma` times the gradient of `-log p` that
  `pathfinder_approximate` passes, which reflected every mean away from the mode. At converged
  iterations the gradient is near zero and the two agree; on a 10-dimensional Gaussian with L-BFGS
  stopped after 3 iterations, KL(q || p) of the selected approximation falls from 28.8 to 6.57.
  Because an iterate short of the mode can now centre its Gaussian on the mode, the ELBO may select
  it, so `PathfinderState.position` is not necessarily the last or most converged iterate; the
  returned draws come from the selected approximation, as in Algorithm 1.
- Pathfinder's `lbfgs_recover_alpha`, `lbfgs_inverse_hessian_factors` and `bfgs_sample` no longer
  build `N x N` matrices. They formed `diag(alpha)` as a dense matrix, and `bfgs_sample` evaluated
  `beta @ gamma @ beta^T @ grad` from the left, so memory and time grew with the square of the
  dimension: one float32 `bfgs_sample` call took 489 ms on CPU at `N = 20000`. They now scale by
  `alpha` elementwise and apply `beta @ (gamma @ (beta^T @ grad))`, which keeps the
  `O(J N + J^2)` cost of Algorithm 4 of Zhang et al. (2022); the same call takes 2.9 ms.
- Pathfinder's `bfgs_sample` returns finite log densities in high dimensions. It evaluated
  `log|Sigma|` as `log(prod(alpha)) + 2 log(det(L))`, so in float32 the product of 128 diagonal
  factors of 2 overflowed and 127 factors of 0.5 underflowed, and `log_q` was `-inf` or `+inf`,
  which also discarded those iterations from the ELBO selection. It now sums logarithms, as in
  Algorithm 4, step 7 of Zhang et al. (2022).
- `mumbo_acquisition` evaluates eq. 5 of Moss, Leslie & Rayson (2020) for samples of the target
  level's maximum. Its Gumbel fit to the quartiles (Wang & Jegelka 2017, §3.1) had a negative scale,
  so its samples were `2 y_0.25 - y*` and fell below the lower quartile. Its extended skew Gaussian
  conditioned on `g > g*` instead of `g < g*`. `gamma` used the candidate's mean and standard
  deviation instead of the target level's. The correlation left the observation noise out of the
  candidate's variance. On a two-level example the scores were off by up to 1.5 nats. The
  expectation in eq. 5 is integrated about the mean `-rho phi(gamma) / Phi(gamma)` of the density in
  Appendix A.1 (eq. 7 prints `+rho`); at `rho = 1` eq. 5 is MES.
- `min_value_entropy_search` scores information about the minimum. It applied the maximum-value
  form of Wang & Jegelka (2017), eq. 6, `gamma = (y* - mu) / sigma` with `Phi(gamma)`, to samples
  of the minimum, which ranks candidates in reverse: with sampled minima -3 and -2.5 and unit
  variance, means 0, 2 and 5 scored 1.62, 2.05 and 32.6, where the entropy reductions are 0.018,
  2e-5 and 0. It now uses `gamma = (mu - y*) / sigma`, eq. 6 applied to `-f`.
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
  5.0e-4 at dt = 100, and gradients were NaN at coarse steps. It now uses the
  exponential-and-Gramian doubling of Stillfjord and Tronarp (arXiv:2310.13462), with a fixed-length
  doubling loop so that reverse-mode gradients work. Against extended-precision references for
  Matern-1/2 to Matern-7/2, integrated Wiener processes of order 1 to 4 and integrated
  Ornstein-Uhlenbeck processes, at steps from 1e-4 to 1e4, every process-noise entry `Q_ij` is
  within `1.7e-5 sqrt(Q_ii Q_jj)` in float32 and `1.1e-13 sqrt(Q_ii Q_jj)` in float64. Gradients
  stay finite for growing drifts, a singular positive semi-definite `Q_c` is supported, and steps
  needing more than 32 doublings return NaN. Per 1000 float32 steps under `vmap` it takes 2.3 to 3.0
  ms instead of 8.1 to 8.7 ms for a three-state SDE and 26 to 27 ms instead of 2.3 to 5.9 ms for a
  four-state SDE; gradients take 8.4 ms instead of 43 ms and 93 ms instead of 20 ms.
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
