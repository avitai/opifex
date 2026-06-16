# Neural Network API Reference

The `opifex.neural` package provides the building blocks for scientific machine learning models, built on top of Flax NNX.

## Base Architectures

### Standard MLP

::: opifex.neural.base.StandardMLP
    options:
        show_root_heading: true
        show_source: true

## Atomistic Models {: #atomistic-models }

Machine-learning interatomic potentials live in `opifex.neural.atomistic`. They
follow a **backbone → typed property heads** assembly: a backbone produces
per-atom embeddings and named heads read them out into energy, forces and stress.
See the [Atomistic Potentials guide](../methods/atomistic-potentials.md) for the
design, the three backbones (SchNet, PaiNN, NequIP) and a registry-driven build.

::: opifex.neural.atomistic.base.AtomisticModel
    options:
        show_root_heading: true
        show_source: false

::: opifex.neural.atomistic.backbones
    options:
        show_root_heading: true
        show_source: false
        members:
            - SchNet
            - SchNetConfig
            - PaiNN
            - PaiNNConfig
            - NequIP
            - NequIPConfig

::: opifex.neural.atomistic.heads
    options:
        show_root_heading: true
        show_source: false
        members:
            - EnergyHead
            - ForcesHead
            - StressHead

## Neural Quantum {: #opifex.neural.quantum }

::: opifex.neural.quantum
    options:
        show_root_heading: true
        show_source: false

### Kohn-Sham DFT solver

::: opifex.neural.quantum.dft.scf
    options:
        show_source: false

### SCF acceleration from a predicted Fock

::: opifex.neural.quantum.dft.scf_acceleration
    options:
        show_source: false

### Exchange-correlation functionals

::: opifex.neural.quantum.dft.xc
    options:
        show_source: false

### Hamiltonian prediction

::: opifex.neural.quantum.hamiltonian
    options:
        show_root_heading: true
        show_source: false

## Neural Operators

::: opifex.neural.operators
    options:
        show_root_heading: true
        show_source: false

## Bayesian Networks {: #bayesian-networks }

::: opifex.neural.bayesian
    options:
        show_root_heading: true
        show_source: false

### `ProbabilisticPINN` shared-objective surface

``opifex.neural.bayesian.ProbabilisticPINN`` is an `nnx.Module` that
implements the canonical `VariationalModule` protocol from
``opifex.uncertainty.protocols``:

* ``kl_divergence() -> jax.Array`` — total KL across every Bayesian
  layer in the network.
* ``predict_distribution(x, *, rngs, mode) -> PredictiveDistribution``
  — returns the canonical Phase-1 contract. ``mode`` is a
  ``PredictiveMode`` value (deterministic / single-sample / monte-carlo
  ensemble); unknown modes raise ``ValueError``.
* ``loss_components(batch, *, rngs, objective) -> UQLossComponents`` —
  returns the data / KL / physics / boundary / initial-condition terms
  as the canonical pattern-B container.
* ``negative_elbo(batch, *, rngs, objective) -> UQLossComponents`` —
  ``UQLossComponents.from_components`` evaluated with sign flipped for
  optimisers that maximise ELBO.

All four methods take a traced ``rngs: nnx.Rngs`` argument; no module
holds a hidden fallback RNG. The shared objective API replaces hand-rolled
``data_loss + kl_weight * kl`` assembly in the example notebooks.

### `RobustPINNOptimizer` (uncertainty-guided training)

``RobustPINNOptimizer.compute_loss_components(batch, *, rngs, objective)``
returns the same ``UQLossComponents`` pattern-B container as
``ProbabilisticPINN`` so a robust-PINN training loop can plug into the
shared objective surface without diverging.
``uncertainty_guided_sampling(x_candidates, num_samples, *, rngs)``
selects the highest-uncertainty samples for the next training batch.

### `ComputationAwareSpectralConvolution` (CASpec)

``opifex.neural.operators.fno.bayesian.ComputationAwareSpectralConvolution``
is a sibling of ``BayesianSpectralConvolution`` whose uncertainty over
the flattened spectral weights is maintained as a *low-rank CAKF
posterior* — the implicit ``posterior_cov = prior_cov - factor @
factor^T`` representation of Pförtner+ 2024 (arXiv:2405.08971) and the
CAGP precursor Wenger+ 2023 (arXiv:2306.07879). The constructor mirrors
``BayesianSpectralConvolution``; ``__call__`` runs the deterministic
spectral conv using the BSC posterior-mean weights, and
``cakf_refine(observation=, observation_matrix=, observation_cov=,
max_iter=)`` returns a ``_CAKFSpectralRefinement`` carrying the
updated ``(cakf_mean, cakf_factor)`` pair (rank gained per call ==
``max_iter``). The same module also re-exports
``BayesianSpectralConvolution`` from its canonical home at
``opifex.uncertainty.layers.bayesian`` so callers can import either
sibling from a single namespace.

### `gp_pinn_predictive_posterior` (GP-PINN)

``opifex.neural.pinns.gp_pinn.gp_pinn_predictive_posterior(*, pinn_forward,
laplace_posterior, coordinates, gp_adapter_spec)`` returns a
function-valued GP predictive over a trained PINN via the
linearised-Laplace equivalence (Immer, Korzepa, Bauer 2021, AISTATS,
arXiv:2008.08400 §3). The math is identical to LUNO
(``opifex.uncertainty.curvature.linearized_neural_operator_posterior``)
and is reused directly; what differs is the *context*: the input is a
PINN forward consuming spatial / spatio-temporal coordinates, and the
``gp_adapter_spec`` parameter (a GP adapter spec such as
``TinygpAdapterSpec`` or ``GPJaxAdapterSpec``) is recorded in the
predictive metadata so consumers can resolve the linearised-Laplace ↔
GP correspondence. Concrete GP fit / predict is available through the
``opifex.uncertainty.gp`` subpackage.

### `ProbabilisticFourierNeuralOperator` (PNO)

``opifex.neural.operators.fno.probabilistic.ProbabilisticFourierNeuralOperator``
equips a standard FNO backbone with twin pointwise heads — a mean head
and a log-variance head — producing a per-location
heteroscedastic-Gaussian ``PredictiveDistribution`` (Kendall & Gal 2017,
arXiv:1703.04977 §3.1; companion to the Magnani+ 2024 LUNO
function-uncertainty thread, arXiv:2406.04317). The training objective
is the elementwise heteroscedastic-Gaussian negative log-likelihood,
exposed as ``probabilistic_fno_negative_log_likelihood(model, x, y)``;
the predictive uncertainty is *aleatoric* by construction. Epistemic
uncertainty is supplied orthogonally by wrapping a fitted PNO with the
existing ``LaplaceAdapterSpec`` (``opifex.uncertainty.curvature``) or a
deep-ensemble adapter (``FNODeepEnsembleAdapterSpec``). The log-variance
head is clipped to ``[log_variance_floor, log_variance_ceiling]``
(defaults ``[-10, 10]``) for numerical stability.

### `UncertaintyQuantificationNeuralOperator` (UQNO)

The conformal neural operator under
``opifex.neural.operators.specialized.uqno`` is composed of three NNX
modules:

* ``UQNOBaseSolutionOperator`` — the underlying FNO that produces point
  predictions of the PDE solution field.
* ``UQNOResidualOperator`` — a Bayesian residual-magnitude operator
  built on shared ``BayesianSpectralConvolution`` layers; predicts
  per-pixel calibrated uncertainty.
* ``UQNOConformalCalibrator`` — applies pointwise conformal calibration
  to the residual output so the resulting bands carry the requested
  empirical coverage.

The three-stage pipeline (``predict_base`` → ``calibrate`` →
``predict_with_bands``) is documented end-to-end in the
``examples/uncertainty/uqno_darcy`` example. UQNO exposes the
conformal contract only — no Bayesian-objective surface
(`predict_distribution` / `loss_components` / `negative_elbo` are
intentionally absent on UQNO itself; those live on
``ProbabilisticPINN`` and the shared layers).

## Domain Decomposition PINNs {: #domain-decomposition }

Domain decomposition methods for physics-informed neural networks, enabling efficient training on complex geometries.

### Base Classes

::: opifex.neural.pinns.domain_decomposition.base
    options:
        show_root_heading: true
        show_source: false
        members:
            - Subdomain
            - Interface
            - DomainDecompositionPINN
            - SubdomainNetwork
            - uniform_partition

### XPINN (Extended PINN)

::: opifex.neural.pinns.domain_decomposition.xpinn
    options:
        show_root_heading: true
        show_source: false
        members:
            - XPINN
            - XPINNConfig

### FBPINN (Finite Basis PINN)

::: opifex.neural.pinns.domain_decomposition.fbpinn
    options:
        show_root_heading: true
        show_source: false
        members:
            - FBPINN
            - FBPINNConfig
            - WindowFunction
            - CosineWindow
            - GaussianWindow

### CPINN (Conservative PINN)

::: opifex.neural.pinns.domain_decomposition.cpinn
    options:
        show_root_heading: true
        show_source: false
        members:
            - CPINN
            - CPINNConfig

### APINN (Augmented PINN)

::: opifex.neural.pinns.domain_decomposition.apinn
    options:
        show_root_heading: true
        show_source: false
        members:
            - APINN
            - APINNConfig
            - GatingNetwork

For usage examples and best practices, see the [Domain Decomposition PINNs Guide](../methods/domain-decomposition-pinns.md).

## Activations

::: opifex.neural.activations
    options:
        show_root_heading: true
        show_source: false
