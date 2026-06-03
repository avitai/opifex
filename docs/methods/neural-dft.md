# Differentiable Kohn-Sham DFT

## Overview

Opifex provides a native-JAX molecular Kohn-Sham density-functional theory (DFT)
solver and a trainable neural exchange-correlation (XC) functional. The
restricted Kohn-Sham (RKS) self-consistent-field (SCF) driver is built on the
McMurchie-Davidson Gaussian-integral backend; the converged total energy is a
pure, differentiable function of the nuclear coordinates, so analytic forces
come from differentiating the SCF fixed point.

Key features:

- **Real Kohn-Sham SCF**: LDA (Slater + VWN5) and PBE GGA functionals with DIIS
  acceleration and a direct-minimisation mode.
- **Analytic forces**: Implicit differentiation of the converged SCF fixed point
  (the PySCFAD rationale) gives exact, memory-cheap `F = -dE/dR`.
- **Trainable XC**: A constrained `NeuralXCFunctional` can replace the analytic
  XC inside the same SCF, with exact `dE/dtheta` for end-to-end learning.
- **Flax NNX / JAX**: Fully compatible with `jit`, `grad`, and `vmap`.

## Core Components

### SCF solver

`SCFSolver` is the entry point. It assembles the integrals and molecular grid
from a `MolecularSystem`, runs the RKS SCF, and exposes the differentiable
energy and analytic forces. The closed-shell RKS solver requires an even
electron count, and the bundled STO-3G minimal basis covers H, C, N and O.

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

    result = solver.solve()                  # SCFResult
    energy = solver.energy()                 # converged total energy (Hartree)
    energy, forces = solver.energy_and_forces()
```

The LDA energies are validated against PySCF in the test suite; for example,
H2 LDA/STO-3G agrees with `pyscf.dft.RKS` to about 1e-7 Hartree.

### Neural XC functional

`NeuralXCFunctional` is a constrained, attention-based exchange-correlation
functional that drives the same real SCF through the `neural_functional`
argument. `jax.grad` of `SCFSolver.energy_from_state` gives an exact
`dE/dtheta` through the implicit-diff SCF, so the learned-XC training loop is
end to end.

```python
import flax.nnx as nnx

from opifex.neural.quantum import NeuralXCFunctional
from opifex.neural.quantum.dft import SCFSolver

functional = NeuralXCFunctional(
    hidden_sizes=(256, 256, 128),
    use_attention=True,
    num_attention_heads=4,
    rngs=nnx.Rngs(0),
)

solver = SCFSolver(system, neural_functional=functional)
graphdef, state = nnx.split(functional)
gradient = jax.grad(solver.energy_from_state)(state)  # exact dE/dtheta
```

## Usage Examples

### Energy and forces from the problem API

The `ElectronicStructureProblem` wraps the SCF behind the unified problem
interface. Its energy and forces are the real Kohn-Sham quantities.

```python
import jax

from opifex.core.problems import create_molecular_system, create_neural_dft_problem

with jax.enable_x64(True):
    h2 = create_molecular_system([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))])
    problem = create_neural_dft_problem(molecular_system=h2)  # functional_type -> LDA/PBE

    energy = problem.compute_energy()   # ~ -1.12 Hartree (LDA/STO-3G)
    forces = problem.compute_forces()   # analytic -dE/dR
```

### JAX transforms

The differentiable energy is `jit` / `grad` / `vmap` compatible. Build the
solver eagerly first (its AO basis and grid are static structural metadata) so
the transform only traces the nuclear positions.

```python
with jax.enable_x64(True):
    _ = problem.scf_solver  # eager build before tracing
    positions = problem.molecular_system.positions
    energy = jax.jit(problem._energy_from_positions)(positions)
```

## Physics Constraints

The neural functional enforces exact constraints so it generalises:

- **Positivity**: the exchange-correlation enhancement keeps the energy density
  physical.
- **Symmetry**: invariant density / gradient features respect rotational and
  translational symmetry.
- **LDA limit**: the network initialises to the analytic LDA functional.

## API Reference

For detailed API documentation, see
[Neural Quantum API](../api/neural.md#opifex.neural.quantum).
