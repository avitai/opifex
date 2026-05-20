# Neural Network API Reference

The `opifex.neural` package provides the building blocks for scientific machine learning models, built on top of Flax NNX.

## Base Architectures

### Standard MLP

::: opifex.neural.base.StandardMLP
    options:
        show_root_heading: true
        show_source: true

### Quantum MLP {: #quantum-networks }

::: opifex.neural.base.QuantumMLP
    options:
        show_root_heading: true
        show_source: true

## Neural Quantum {: #opifex.neural.quantum }

::: opifex.neural.quantum
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
