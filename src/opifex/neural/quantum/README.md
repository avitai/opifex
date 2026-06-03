# Opifex Quantum Neural Networks: Differentiable Kohn-Sham DFT

This module hosts Opifex's differentiable molecular density-functional theory:
a native-JAX restricted Kohn-Sham (RKS) self-consistent-field solver and a
trainable neural exchange-correlation functional. Every public symbol below is
exported and JAX/NNX compatible.

## Module map

| File | Public symbols |
|------|----------------|
| `dft/` | `SCFSolver`, `SCFResult`, `Functional`, `SolverMode`, the molecular grid and LDA/PBE XC primitives |
| `neural_xc.py` | `NeuralXCFunctional` |

`SCFSolver` runs the RKS SCF on the McMurchie-Davidson Gaussian-integral
backend (`opifex.core.quantum.backend.JaxGaussianBackend`) with the LDA
(Slater + VWN5), PBE GGA, or a learned `NeuralXCFunctional`. The converged
total energy is a pure, differentiable function of the nuclear coordinates, and
analytic forces come from implicit differentiation of the SCF fixed point.

## Kohn-Sham SCF solver

```python
import jax
import jax.numpy as jnp

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.dft import SCFSolver

with jax.enable_x64(True):
    # H2 at the equilibrium bond length (positions in Bohr).
    system = MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        basis_set="sto-3g",
    )
    solver = SCFSolver(system, functional="lda")

    result = solver.solve()          # SCFResult: total_energy, orbitals, density
    energy = solver.energy()         # converged total energy (Hartree)
    energy, forces = solver.energy_and_forces()  # analytic forces (-dE/dR)
```

`SCFResult` carries `total_energy`, `orbital_energies`, `density_matrix`,
`coefficients`, `n_iterations`, and `converged`. The DIIS forward solve and the
direct-minimisation mode are selected through the `mode` argument; the
implicit-diff energy/force path is used by `energy_from_positions`,
`compute_forces`, and `energy_and_forces`.

The closed-shell RKS solver requires an even electron count, and the bundled
STO-3G minimal basis covers H, C, N and O. The LDA energies are validated
against PySCF in the test suite (e.g. H2 LDA/STO-3G agrees with `pyscf` to
~1e-7 Hartree).

## Neural exchange-correlation functional

`NeuralXCFunctional` is a constrained, attention-based exchange-correlation
functional that drives the same real SCF. It is wired into the solver through
the `neural_functional` argument, and `jax.grad` of the converged energy with
respect to its parameters (via `SCFSolver.energy_from_state`) gives an exact
`dE/dtheta` through the implicit-diff SCF for end-to-end learned-XC training.

```python
import flax.nnx as nnx

from opifex.neural.quantum import NeuralXCFunctional
from opifex.neural.quantum.dft import SCFSolver

functional = NeuralXCFunctional(
    hidden_sizes=(128, 128, 64),
    use_attention=True,
    num_attention_heads=8,
    use_advanced_features=True,
    dropout_rate=0.0,
    rngs=nnx.Rngs(0),
)

solver = SCFSolver(system, neural_functional=functional)
# graphdef, state = nnx.split(functional)
# gradient = jax.grad(solver.energy_from_state)(state)  # exact dE/dtheta
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

## Practical guidance

- Always pass a caller-owned `nnx.Rngs` to `NeuralXCFunctional`; the module
  never constructs hidden seeds.
- The Gaussian integrals are float64; wrap solves in `jax.enable_x64(True)`.
- Pre-build the solver (its AO basis / grid are eager static metadata) before
  `jax.jit` / `jax.grad` / `jax.vmap` so only the nuclear positions are traced.
- Choose the functional with `functional="lda"` / `"pbe"`, or pass a trained
  `neural_functional` to use the learned XC path.

## Related modules

- [Bayesian neural networks](../bayesian/README.md) — posterior frameworks,
  calibration helpers, and probabilistic PINNs.
- [Neural operators](../operators/README.md) — Fourier and DeepONet
  operators that share the NNX interface with `NeuralXCFunctional`.
- [Core framework](../../core/README.md) — physics losses, conservation
  utilities, and shared mathematical primitives.
