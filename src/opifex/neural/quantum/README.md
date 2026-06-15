# Opifex Quantum Neural Networks: Neural Density Functional Theory

This module hosts the Neural Density Functional Theory (Neural DFT) building
blocks in Opifex: a top-level `NeuralDFT` driver, a neural exchange-correlation
functional, and a neural-accelerated self-consistent field solver. Every
public symbol below is currently exported and JAX/NNX compatible.

## Module map

| File | Public symbols |
|------|----------------|
| `neural_dft.py` | `NeuralDFT`, `DFTResult` |
| `neural_xc.py` | `NeuralXCFunctional` |
| `neural_scf.py` | `NeuralSCFSolver` |

`NeuralDFT` composes `NeuralXCFunctional` and `NeuralSCFSolver` through its
internal `_initialize_neural_components` hook, so users typically interact
with the top-level driver and only construct the subcomponents directly when
swapping functionals or solver strategies.

## Neural DFT driver

`NeuralDFT` is a Flax NNX module. It runs an SCF loop with an optional
neural mixing strategy and a neural XC functional, and reports convergence
plus precision diagnostics through `DFTResult`.

```python
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.quantum import NeuralDFT

rngs = nnx.Rngs(0)

neural_dft = NeuralDFT(
    grid_size=512,
    convergence_threshold=1e-8,
    max_scf_iterations=100,
    xc_functional_type="neural",
    mixing_strategy="neural",
    use_neural_scf=True,
    chemical_accuracy_target=0.043,   # 1 kcal/mol expressed in Hartree
    enable_high_precision=True,
    rngs=rngs,
)

# `molecular_system` is any mapping/object the caller's data layer
# produces; NeuralDFT consumes it through duck-typed accessors so it can be
# swapped between PySCF, Datarax, or in-house molecular containers.
molecular_system = {
    "atomic_numbers": jnp.array([8, 1, 1]),
    "positions": jnp.array(
        [
            [0.0,  0.0,  0.1173],
            [0.0,  0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ]
    ),
    "charge": 0,
    "multiplicity": 1,
}

result = neural_dft.compute_energy(molecular_system)
# result.total_energy, result.converged, result.iterations
# result.chemical_accuracy_achieved, result.precision_metrics

predicted_accuracy = neural_dft.predict_chemical_accuracy(molecular_system)
```

`DFTResult` carries `electronic_energy`, `nuclear_repulsion_energy`,
`total_energy`, `xc_energy`, `converged`, `iterations`,
`convergence_history`, `final_density`, `molecular_orbitals`,
`orbital_energies`, `chemical_accuracy_achieved`, and `precision_metrics`.

## Neural exchange-correlation functional

`NeuralXCFunctional` evaluates the exchange-correlation energy density and
its functional derivative. The functional is constructed with the standard
hidden-size tuple and supports attention-style mixing.

```python
import flax.nnx as nnx
from opifex.neural.quantum import NeuralXCFunctional

rngs = nnx.Rngs(0)

neural_xc = NeuralXCFunctional(
    hidden_sizes=(128, 128, 64),
    use_attention=True,
    num_attention_heads=8,
    use_advanced_features=True,
    dropout_rate=0.0,
    rngs=rngs,
)

# Inspect the functional's accuracy assessment hook on a density input.
# `density` here is a JAX array supplied by the caller's data pipeline.
# accuracy = neural_xc.assess_chemical_accuracy(density)
# v_xc = neural_xc.compute_functional_derivative(density)
```

## Neural self-consistent field solver

`NeuralSCFSolver` accelerates the SCF iteration with a neural mixing
strategy and a convergence predictor. It can be used standalone when
embedding the solver inside an alternative DFT driver.

```python
import flax.nnx as nnx
from opifex.neural.quantum import NeuralSCFSolver

rngs = nnx.Rngs(0)

neural_scf = NeuralSCFSolver(
    convergence_threshold=1e-8,
    max_iterations=100,
    mixing_strategy="neural",
    grid_size=512,
    chemical_accuracy_target=1e-6,
    rngs=rngs,
)

# scf_result = neural_scf.solve_scf(molecular_system=...)
# predicted_iterations = neural_scf.predict_convergence_iterations(...)
```

## Integration with the Bayesian platform

Adding a posterior over the XC parameters lets a `NeuralXCFunctional` quote
predictive uncertainty over energies and densities. The
`AmortizedVariationalFramework` wraps any Flax NNX module, including
`NeuralXCFunctional`, with a mean-field Gaussian posterior and an
input-conditioned uncertainty encoder.

```python
import flax.nnx as nnx
from opifex.neural.bayesian import (
    AmortizedVariationalFramework,
    PriorConfig,
    VariationalConfig,
)
from opifex.neural.quantum import NeuralXCFunctional

rngs = nnx.Rngs(0)

xc = NeuralXCFunctional(rngs=rngs)

config = VariationalConfig(
    input_dim=64,
    hidden_dims=(64, 32),
    num_samples=10,
    kl_weight=0.1,
)

prob_xc = AmortizedVariationalFramework(
    base_model=xc,
    prior_config=PriorConfig(),
    variational_config=config,
    rngs=rngs,
)

# predictive = prob_xc.predict_distribution(features, rngs=nnx.Rngs(predict=1))
```

For full posterior sampling, the platform's `BlackJAXBackend` (NUTS / HMC /
MALA) and the variational backends `ADVIBackend`, `SVGDBackend`, and
`PathfinderBackend` are routed through `InferenceBackendProtocol` and
return `PredictiveDistribution` objects suitable for downstream calibration.

## Pairing with neural operators

The Fourier neural operator (FNO) can serve as a density-to-XC-feature
operator that feeds `NeuralXCFunctional`. The operator preserves the NNX
interface, so it composes directly inside an SCF loop.

```python
import flax.nnx as nnx
from opifex.neural.operators.fno import FourierNeuralOperator

rngs = nnx.Rngs(0)

density_operator = FourierNeuralOperator(
    in_channels=4,
    out_channels=1,
    hidden_channels=64,
    modes=16,
    num_layers=4,
    rngs=rngs,
)
```

## Practical guidance

- Always pass a caller-owned `nnx.Rngs` to `NeuralDFT`, `NeuralXCFunctional`,
  and `NeuralSCFSolver`. The modules never construct hidden seeds.
- `chemical_accuracy_target` is expressed in Hartree (0.043 Ha is roughly
  1 kcal/mol). `enable_high_precision=True` promotes critical inner loops
  to float64.
- Use `result.precision_metrics` and `result.chemical_accuracy_achieved` to
  decide whether to retry with a tighter convergence threshold or a
  different XC functional.
- When experimenting with classical XC functionals, set
  `xc_functional_type="lda"` or `"pbe"`; the neural component is skipped
  and the solver falls back to standard density mixing.

## Related modules

- [Bayesian neural networks](../bayesian/README.md) — posterior frameworks,
  calibration helpers, and probabilistic PINNs.
- [Neural operators](../operators/README.md) — Fourier and DeepONet
  operators that share the NNX interface with `NeuralXCFunctional`.
- [Core framework](../../core/README.md) — physics losses, conservation
  utilities, and shared mathematical primitives.
