# Variational Monte Carlo

Variational Monte Carlo (VMC) solves the many-electron Schrödinger equation by
*optimising* a trial wavefunction rather than diagonalising a Hamiltonian. For a
parameterised wavefunction `psi_theta`, the variational energy

```
E[theta] = < E_loc >_{|psi_theta|^2},   E_loc(r) = (H psi_theta)(r) / psi_theta(r)
```

is an expectation of the *local energy* over walkers sampled from the Born density
`|psi_theta|^2`. The variational principle guarantees `E[theta] >= E_0`, so
minimising `E[theta]` drives `psi_theta` toward the exact ground state and the
recovered energy is always an upper bound — there is no label set and no
overfitting.

opifex builds a scalable, `jit` / `grad` / `vmap`-clean VMC stack entirely on JAX,
Flax NNX and optax (no external QMC dependency), assembled through a small set of
protocols so a new ansatz, sampler or optimiser plugs in without touching the
driver. It lives in `opifex.neural.quantum.vmc`.

## Design: backbone → local energy → optimiser

A VMC run is a composition of three swappable pieces, decoupled by the `Protocol`
definitions in `opifex.neural.quantum.vmc.protocols`:

- a **wavefunction** (the `Wavefunction` protocol) — a neural ansatz evaluated one
  walker at a time, returning `(sign, log|psi|)` in the log domain for numerical
  stability;
- a **sampler** (the `Sampler` protocol) — a Metropolis sampler of the Born
  density `|psi|^2`;
- an **optimiser** — the natural-gradient update that preconditions the energy
  gradient (Adam bootstrap, or the MinSR / SPRING natural gradient).

The `VMCDriver` wires them together: each iteration samples walkers from the
current `|psi_theta|^2`, evaluates the per-walker **local energy** (kinetic +
Coulomb), forms the energy gradient, and applies the optimiser update. The whole
step is one `jit`-compiled kernel.

## The wavefunction: FermiNet generalized-Slater ansatz

`FermiNet` is a Flax-NNX port of the Fermionic Neural Network (Pfau, Spencer,
Matthews & Foulkes 2020). The wavefunction is a weighted sum of generalized
Slater determinants

```
psi(r) = sum_k w_k det[ phi^k_i(r_j) ],
```

where each orbital `phi^k_i` is a **permutation-equivariant function of all
electron coordinates** (not a single-particle orbital), multiplied by an isotropic
exponential envelope that enforces the correct asymptotic decay. The equivariant
backbone interleaves a one-electron stream `h_one` and a two-electron stream
`h_two` with FermiNet symmetric feature pooling; the antisymmetry required of a
fermionic wavefunction comes from the determinant, evaluated in the log domain via
the log-sum-exp `logdet_matmul` so the magnitude never overflows. Spin
`(n_up, n_down)` is a *static* attribute, so the network is jit-clean.

## The local energy: forward-Laplacian kinetic term

The local energy of a real, log-domain wavefunction is

```
E_loc(r) = -1/2 ( nabla^2 log|psi| + |nabla log|psi||^2 ) + V(r),
```

with the molecular Coulomb potential `V = V_ee - V_eN + V_NN`
(`opifex.neural.quantum.vmc.hamiltonian`). The kinetic term needs the Laplacian
and squared gradient of `log|psi|`. opifex provides two interchangeable Laplacian
operators (`opifex.neural.quantum.vmc.laplacian`):

| Operator | Mechanism | Use |
|----------|-----------|-----|
| `jvp_grad_laplacian` | linearise `grad` once and read the Hessian diagonal one coordinate at a time (`O(n)` passes) | reference *oracle*, gates the fast path in tests |
| `forward_laplacian` | propagate value, full Jacobian and Laplacian through a **single** forward pass via stacked JVPs (`O(1)` passes) | default — the LapNet / `fwdlap` speed-up (Chen 2023) |

The forward-Laplacian obtains the Hessian diagonal in `O(1)` forward passes
(vectorised over an orthonormal basis of the input space) rather than `O(n)`
reverse-then-forward passes, expressed purely with `jax.jvp` and `jax.vmap` — no
`folx` / `fwdlap` dependency. The potential is cusp-safe: the electron-electron
distance diagonal is masked and only strict upper triangles are summed.

## Sampling: harmonic-mean Metropolis-Hastings

`MetropolisHastingsSampler` draws walkers from the Born density `log p(r) = 2
log|psi(r)|` with the FermiNet **asymmetric, harmonic-mean** proposal: the
per-electron proposal width scales with the harmonic mean of that electron's
distances to the nuclei, so electrons near a nucleus take small steps and valence
electrons take large ones. The accept/reject ratio includes the forward/reverse
proposal densities to preserve detailed balance, and a fixed number of MCMC sweeps
is fused into a single `jax.lax.scan` — one `jit`-compiled GPU kernel for the whole
sampling phase.

## Optimisation: MinSR / SPRING natural gradient

VMC is dramatically accelerated by the *natural* gradient (stochastic
reconfiguration / quantum natural gradient), which preconditions the energy
gradient by the inverse Fisher / quantum geometric tensor. For deep ansätze the
parameter count far exceeds the sample count, so the Fisher is inverted in the
**sample space** instead — the MinSR / SRT trick (Rende et al. 2023; Chen & Heyl,
Nat. Phys. 2024):

```
delta_theta = O_L^T (O_L O_L^T + lambda I)^{-1} dv,
```

where `O_L` is the centred, `1/sqrt(N)`-scaled per-sample Jacobian of `log|psi|`
and `dv = 2 (E_loc - <E_loc>) / sqrt(N)` is the centred energy-gradient signal. The
Gram matrix `O_L O_L^T` is only `N x N`. `spring_update` adds the **SPRING**
momentum scheme (Goldshlager, Abrahamsen & Lin 2024, arXiv:2401.10190): a
Nesterov-style accumulation of past updates plus a `proj_reg / N` projection
regulariser on the Gram matrix; with zero momentum and zero `proj_reg` it reduces
exactly to MinSR. The driver feeds these the FermiNet score-function gradient
`grad E = 2 < (E_loc - <E_loc>) grad log|psi| >`, with the local energy as a
stop-gradient baseline and outlier-robust median-absolute-deviation clipping. Adam
is available as a first-order bootstrap directly through optax.

## Example

```python
import jax
import jax.numpy as jnp
from flax import nnx

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

from opifex.neural.quantum.vmc import (
    FermiNet, MetropolisHastingsSampler, VMCConfig, VMCDriver,
)

# Helium atom: two electrons, one nucleus of charge 2 (positions in bohr).
atoms = jnp.array([[0.0, 0.0, 0.0]])
charges = jnp.array([2.0])

ansatz = FermiNet(
    nspins=(1, 1), atoms=atoms, charges=charges,
    hidden_one=(32, 32), hidden_two=(16, 16),
    determinants=4, full_det=True, rngs=nnx.Rngs(0),
)
sampler = MetropolisHastingsSampler(atoms=atoms, steps=10, step_size=0.4)
config = VMCConfig(
    batch_size=1024, iterations=800, optimizer="spring",
    learning_rate=0.04, equilibration_steps=200,
)
driver = VMCDriver(ansatz=ansatz, sampler=sampler, config=config)
result = driver.run(jax.random.PRNGKey(0))

print(result.energy, "+/-", result.energy_error)  # ~ -2.9037 Ha (exact)
```

The energy is a variational upper bound on the exact ground state; the same
composition recovers the H atom (`-0.5`), He (`-2.9037`) and H2 at R=1.4 bohr
(`-1.1745`) to chemical accuracy.

For an end-to-end example — building the ansatz, sampler and optimiser, recovering
all three energies, and plotting the energy convergence and the sampled walker
density — see
[Variational Monte Carlo: H, He and H2 ground states](../examples/quantum-chemistry/vmc-atoms.md).

## Extending

- **A new ansatz** (e.g. a PsiFormer attention backbone, von Glehn, Spencer &
  Pfau 2022, arXiv:2211.13672) — implement the `Wavefunction` protocol returning
  `(sign, log|psi|)`. The orbital / envelope / determinant machinery is agnostic
  to how the per-electron features are produced, so a PsiFormer self-attention
  backbone swaps in by replacing the equivariant feature stage alone.
- **A new sampler** (e.g. MALA / Langevin) — implement the `Sampler` protocol; a
  MALA sampler adds a `grad log|psi|` drift term to the harmonic-mean proposal
  mean.
- **A new optimiser** — the MinSR / SPRING updates return the raw parameter-update
  vector, applied with an external learning rate; a K-FAC or alternative
  natural-gradient preconditioner plugs into the same driver hook.

The shared primitives — the log-domain determinant building blocks
(`logdet_matmul`, `slogdet`), the forward-Laplacian, the molecular local energy,
and the MinSR / SPRING solves — live in `opifex.neural.quantum.vmc` and are reused
across the variational-Monte-Carlo family.

## References

- Pfau, Spencer, Matthews & Foulkes 2020, *Ab initio solution of the
  many-electron Schrödinger equation with deep neural networks*, Phys. Rev.
  Research 2, 033429 ([arXiv:1909.02487](https://arxiv.org/abs/1909.02487)) —
  FermiNet.
- von Glehn, Spencer & Pfau 2022, *A Self-Attention Ansatz for Ab-initio Quantum
  Chemistry* ([arXiv:2211.13672](https://arxiv.org/abs/2211.13672)) — PsiFormer.
- Goldshlager, Abrahamsen & Lin 2024, *A Kaczmarz-inspired approach to accelerate
  the optimization of neural network wavefunctions*
  ([arXiv:2401.10190](https://arxiv.org/abs/2401.10190)) — SPRING.
- Rende, Viteritti, Bardone, Becca & Goldt 2023, *A simple linear algebra
  identity to optimize large-scale neural network quantum states*
  ([arXiv:2310.05715](https://arxiv.org/abs/2310.05715)); Chen & Heyl 2024,
  Nat. Phys. — MinSR / stochastic-reconfiguration-by-sampling.
- Li, Chen et al. 2023, *A computational framework for neural network-based
  variational Monte Carlo with forward Laplacian*
  ([arXiv:2307.08214](https://arxiv.org/abs/2307.08214)) — the forward-Laplacian
  (LapNet).
```
