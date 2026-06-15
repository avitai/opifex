# Opifex Quantum Neural Networks

This module hosts three complementary families of differentiable molecular
electronic-structure methods, all native-JAX and JAX/NNX compatible:

1. **Kohn-Sham DFT** — a restricted Kohn-Sham (RKS) self-consistent-field
   solver and a trainable neural exchange-correlation functional.
2. **Neural-wavefunction VMC** — a FermiNet-core variational Monte Carlo stack
   (`vmc/`) that minimises the variational energy of a deep-network ansatz with
   no dependence on the Gaussian-integral engine.
3. **Equivariant Hamiltonian prediction** — a QHNet-style SE(3)-equivariant
   model (`hamiltonian/`) that predicts the DFT Fock and overlap matrices from
   geometry, reusing the shared equivariant kit.

Every public symbol below is exported and JAX/NNX compatible.

## Module map

| File | Public symbols |
|------|----------------|
| `dft/` | `SCFSolver`, `SCFResult`, `Functional`, `SolverMode`, the molecular grid and LDA/PBE XC primitives |
| `neural_xc.py` | `NeuralXCFunctional` |
| `vmc/` | `FermiNet`, `VMCDriver`, `VMCConfig`, `VMCResult`, `MetropolisHastingsSampler`, `local_energy`, `forward_laplacian`, `jvp_grad_laplacian`, `minsr_update`, `spring_update`, `SpringState` |
| `hamiltonian/` | `HamiltonianPredictor`, `block_from_irreps`, `PairExpansion` |

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

## Neural-wavefunction variational Monte Carlo

The `vmc/` subpackage minimises the variational energy
`E[θ] = ⟨E_loc⟩_{|ψ_θ|²}` of a FermiNet-core generalized-Slater ansatz. It is
integral-free (no dependency on the Gaussian-integral backend) and every
component is `jit` / `grad` / `vmap` clean.

```python
import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.quantum.vmc import (
    FermiNet, MetropolisHastingsSampler, VMCConfig, VMCDriver,
)

with jax.enable_x64(True):
    atoms = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])   # H2, Bohr
    charges = jnp.array([1.0, 1.0])

    ansatz = FermiNet(
        nspins=(1, 1), atoms=atoms, charges=charges,
        hidden_one=(32, 32), hidden_two=(16, 16),
        determinants=4, full_det=True, rngs=nnx.Rngs(0),
    )
    sampler = MetropolisHastingsSampler(atoms=atoms, steps=10, step_size=0.4)
    config = VMCConfig(batch_size=1024, iterations=600, optimizer="spring")

    result = VMCDriver(ansatz=ansatz, sampler=sampler, config=config).run(
        jax.random.PRNGKey(0)
    )
    print(result.energy, "±", result.energy_error)   # ≈ -1.1745 Ha
```

Reference energies recovered to chemical accuracy (1024 walkers, SPRING):

| System | E_VMC (Ha) | exact | error |
|--------|-----------:|------:|------:|
| H | -0.4996 | -0.5000 | 0.4 mHa |
| H₂ (R=1.4) | -1.1745 | -1.1745 | 0.0 mHa |
| He | -2.9037 | -2.9037 | 0.0 mHa |

Design highlights:

- **Ansatz** (`wavefunctions/ferminet.py`): permutation-equivariant one- and
  two-electron streams, per-orbital isotropic exponential envelopes, and a sum
  of generalized Slater determinants evaluated in the log domain
  (`logdet_matmul`). A single-walker `log|ψ|` / sign function is `vmap`-ed over
  walkers and over the determinant axis; a PsiFormer backbone can swap in.
- **Kinetic energy** (`laplacian.py`): a `jvp`-over-`grad` reference oracle and
  a native forward-Laplacian (two stacked JVPs over an orthonormal basis, the
  LapNet scheme) that agree to ~1e-14 on the real ansatz.
- **Optimizers** (`optimizers.py`): Adam bootstrap, then MinSR (sample-space /
  NTK Gram natural gradient) and SPRING (MinSR + Nesterov momentum); pure JAX
  linear algebra, no K-FAC.
- **Sampler** (`sampler.py`): FermiNet harmonic-mean Metropolis-Hastings with
  the MCMC sweeps fused into one `lax.scan`.

## Practical guidance

- VMC energies are tight (≤1 mHa); run under `jax.enable_x64(True)`.
- The natural-gradient path (`optimizer="spring"`/`"minsr"`) converges far
  faster per iteration than Adam — prefer it once the ansatz is initialised.
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
