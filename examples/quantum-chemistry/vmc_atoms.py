# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.6
# ---

# %% [markdown]
"""
# Variational Monte Carlo: H, He and H2 ground states

| Property      | Value                                                       |
|---------------|-------------------------------------------------------------|
| Level         | Advanced                                                    |
| Runtime       | ~2 min (GPU, jit + lax.scan sampler)                        |
| Memory        | ~2 GB                                                       |
| Prerequisites | JAX, Flax NNX, variational Monte Carlo, neural wavefunctions |

## Overview

Recover the ground-state energies of the **hydrogen atom**, the **helium atom**,
and the **hydrogen molecule** from first principles with variational Monte Carlo
(VMC), using a FermiNet neural-network wavefunction. VMC minimises the
variational energy

```
E[theta] = < E_loc >_{|psi_theta|^2},   E_loc(r) = (H psi_theta)(r) / psi_theta(r)
```

of a trial wavefunction `psi_theta` by stochastic gradient descent on its
parameters. The expectation is taken over walkers sampled from the Born density
`|psi_theta|^2`. By the variational principle the energy is an upper bound on the
true ground state, so *lower is always better* and the exact small-system
energies are hard targets to recover -- there is no overfitting to a label set.

These three systems have essentially exact references (H `-0.5`, He `-2.9037`
(Pekeris), H2 at R=1.4 bohr `-1.1745` Hartree), so the goal here is faithful
recovery of known physics to **chemical accuracy** (better than 1 mHa), not
beating a benchmark. The same ansatz, sampler and optimiser scale unchanged to
larger molecules where no exact answer exists.

This example is deliberately **thin**: it composes opifex's committed VMC stack
and changes no library internals.

- `FermiNet` is the generalized-Slater equivariant ansatz, evaluated in the log
  domain one walker at a time and `vmap`-ed over the batch.
- `MetropolisHastingsSampler` draws walkers from `|psi|^2` with the FermiNet
  harmonic-mean proposal, fusing all MCMC sweeps into one `jax.lax.scan`.
- `VMCDriver` runs the energy-minimisation loop: each jitted step samples
  walkers, evaluates the per-walker local energy (forward-Laplacian kinetic
  term), and applies a SPRING natural-gradient update.
- `VMCConfig` selects the SPRING optimiser, the batch size and the iteration
  budget.

## Learning Goals

1. Build a `FermiNet` wavefunction for an atom or molecule
2. Sample the Born density `|psi|^2` with the harmonic-mean Metropolis sampler
3. Minimise the variational energy with the SPRING natural-gradient optimiser
4. Recover H, He and H2 ground-state energies to chemical accuracy
5. Read the energy-convergence curve and the sampled walker density
"""

# %% [markdown]
"""
## Imports and Setup

VMC energies are validated against *exact* references to ~1 mHa, so the example
runs in **float64** with the matmul precision pinned to `high`. On GPU a float32
matmul otherwise falls back to TF32 (~1e-3 relative error), two orders of
magnitude larger than the milli-Hartree tolerances here; `high` is the
error-corrected 3xTF32 path -- full fp32 accuracy at tensor-core speed.
"""

# %%
import time
import warnings
from dataclasses import dataclass
from pathlib import Path


warnings.filterwarnings("ignore")

import jax


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from flax import nnx


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.neural.quantum.vmc import (
    FermiNet,
    MetropolisHastingsSampler,
    VMCConfig,
    VMCDriver,
    VMCResult,
)


# %% [markdown]
"""
## Configuration

Each system is described by its nuclear geometry (`atoms`, in bohr), nuclear
charges, spin partition `(n_up, n_down)`, exact reference energy, and a per-system
optimisation budget. The ansatz, sampler and optimiser hyper-parameters are
shared across all three systems and follow the FermiNet recipe for small atoms
(reference implementation `../ferminet`):

- a small **two-layer** equivariant backbone (one-electron widths `(32, 32)`,
  two-electron widths `(16, 16)`) with **four** generalized-Slater determinants
  -- ample for one- and two-electron systems;
- the **SPRING** natural-gradient optimiser (Goldshlager et al.,
  arXiv:2401.10190), which preconditions the energy gradient by the inverse
  Fisher matrix solved in the (cheap) sample space and adds Nesterov momentum, so
  it converges in far fewer iterations than plain Adam;
- the harmonic-mean Metropolis sampler with **10** MCMC sweeps per step and a
  `0.4` bohr base step size, after a 200-sweep burn-in.

The batch is **1024 walkers**. The heavier systems (He, H2) get a larger
iteration budget; the learning rate is slightly lower for He, whose two
correlated electrons need a gentler step.
"""


# %%
@dataclass(frozen=True, slots=True, kw_only=True)
class System:
    """A molecular system and its exact reference energy for VMC recovery.

    Args:
        name: Short label for the system.
        atoms: Nuclear coordinates of shape ``(natom, 3)`` in bohr.
        charges: Nuclear charges of shape ``(natom,)``.
        nspins: ``(n_up, n_down)`` electron counts.
        exact_energy: Essentially-exact reference total energy in Hartree.
        iterations: Number of VMC optimisation steps.
        learning_rate: SPRING step size.
        seed: PRNG / parameter-initialisation seed.
    """

    name: str
    atoms: jax.Array
    charges: jax.Array
    nspins: tuple[int, int]
    exact_energy: float
    iterations: int
    learning_rate: float
    seed: int


# Shared ansatz / sampler / optimiser hyper-parameters (FermiNet small-atom recipe).
HIDDEN_ONE = (32, 32)  # one-electron stream widths (two equivariant layers)
HIDDEN_TWO = (16, 16)  # two-electron stream widths (same depth)
DETERMINANTS = 4  # generalized-Slater determinants in the sum
BATCH_SIZE = 1024  # number of MCMC walkers
SAMPLER_STEPS = 10  # MCMC sweeps fused into one lax.scan per VMC step
STEP_SIZE = 0.4  # base Metropolis proposal width (bohr), scaled per electron
EQUILIBRATION_STEPS = 200  # burn-in sweeps before optimisation
OPTIMIZER = "spring"  # SPRING natural-gradient (MinSR + Nesterov momentum)

# Per-system geometry, references and budgets. Energies in Hartree; H2 at the
# equilibrium bond length R = 1.4 bohr. References: H -1/2 (exact), He -2.9037
# (Pekeris 1958, essentially exact), H2 -1.1745 (exact Born-Oppenheimer).
SYSTEMS = (
    System(
        name="H",
        atoms=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([1.0]),
        nspins=(1, 0),
        exact_energy=-0.5,
        iterations=400,
        learning_rate=0.05,
        seed=0,
    ),
    System(
        name="He",
        atoms=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([2.0]),
        nspins=(1, 1),
        exact_energy=-2.9037,
        iterations=800,
        learning_rate=0.04,
        seed=2,
    ),
    System(
        name="H2",
        atoms=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        charges=jnp.array([1.0, 1.0]),
        nspins=(1, 1),
        exact_energy=-1.1745,
        iterations=600,
        learning_rate=0.05,
        seed=1,
    ),
)

OUTPUT_DIR = Path("docs/assets/examples/vmc_atoms")

CHEMICAL_ACCURACY_MHA = 1.594  # 1 kcal/mol in milli-Hartree

# %% [markdown]
"""
## Building one VMC run

A VMC run for any system is the same three-object composition: a `FermiNet`
ansatz, a `MetropolisHastingsSampler`, and a `VMCDriver` carrying the
`VMCConfig`. The ansatz exposes a single-walker `positions -> (sign, log|psi|)`
call (log domain for stability); the driver `vmap`s it over the walker batch for
sampling and for the per-walker local energy, and `grad`s it for both the
score-function energy gradient and the per-sample Jacobian the SPRING solve
needs. Spin `(n_up, n_down)` is a *static* attribute, so the whole step is one
`jit`-compiled GPU kernel.
"""


# %%
def build_driver(system: System) -> VMCDriver:
    """Compose the FermiNet ansatz, sampler and driver for one system."""
    ansatz = FermiNet(
        nspins=system.nspins,
        atoms=system.atoms,
        charges=system.charges,
        hidden_one=HIDDEN_ONE,
        hidden_two=HIDDEN_TWO,
        determinants=DETERMINANTS,
        full_det=True,
        rngs=nnx.Rngs(system.seed),
    )
    sampler = MetropolisHastingsSampler(
        atoms=system.atoms, steps=SAMPLER_STEPS, step_size=STEP_SIZE
    )
    config = VMCConfig(
        batch_size=BATCH_SIZE,
        iterations=system.iterations,
        optimizer=OPTIMIZER,
        learning_rate=system.learning_rate,
        equilibration_steps=EQUILIBRATION_STEPS,
    )
    return VMCDriver(ansatz=ansatz, sampler=sampler, config=config)


# %% [markdown]
"""
## Training

`VMCDriver.run` equilibrates the walkers under the freshly initialised
wavefunction, then runs `iterations` jitted optimisation steps. Each step:

1. advances the walkers by `SAMPLER_STEPS` Metropolis-Hastings sweeps (one fused
   `lax.scan`) so they track the *current* `|psi_theta|^2`;
2. evaluates the per-walker local energy with the native **forward-Laplacian**
   kinetic term (the Hessian diagonal of `log|psi|` via stacked JVPs, the LapNet
   speed-up), with outlier-robust median-absolute-deviation clipping;
3. forms the FermiNet score-function energy gradient
   `2 < (E_loc - <E_loc>) grad log|psi| >`, preconditions it with the SPRING
   natural-gradient solve (Fisher inverse in sample space + Nesterov momentum),
   and updates the parameters.

The reported energy is the mean local energy over the final batch with its Monte
Carlo standard error `sigma / sqrt(N)`. We run all three systems and keep each
result for the table and the convergence plot.
"""


# %%
def main() -> dict[str, float | int]:
    """Optimise all systems, render diagnostics, and return per-system energy errors."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Opifex Example: Variational Monte Carlo (H, He, H2)")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}")
    print(f"Systems: {', '.join(s.name for s in SYSTEMS)}")
    print(
        f"Ansatz: FermiNet, hidden_one={HIDDEN_ONE}, hidden_two={HIDDEN_TWO}, dets={DETERMINANTS}"
    )
    print(
        f"Sampler: harmonic-mean Metropolis, {SAMPLER_STEPS} sweeps/step, "
        f"step_size={STEP_SIZE} bohr"
    )
    print(f"Optimizer: {OPTIMIZER.upper()} natural gradient, {BATCH_SIZE} walkers")

    print()
    print("Starting VMC optimisation (jit compiles on the first step of each system)...")
    results: dict[str, VMCResult] = {}
    run_times: dict[str, float] = {}
    for system in SYSTEMS:
        driver = build_driver(system)
        start_time = time.time()
        result = driver.run(jax.random.PRNGKey(system.seed))
        elapsed = time.time() - start_time
        results[system.name] = result
        run_times[system.name] = elapsed
        error_mha = (float(result.energy) - system.exact_energy) * 1000.0
        print(
            f"{system.name:>3s} | {system.iterations:4d} iters | "
            f"E = {float(result.energy):+.5f} +/- {float(result.energy_error):.5f} Ha | "
            f"exact {system.exact_energy:+.4f} | err {error_mha:+6.2f} mHa | "
            f"t {elapsed:4.0f}s"
        )
    total_time = sum(run_times.values())
    print(f"All systems optimised in {total_time:.0f}s")

    # Results table: recovered energy vs exact reference.
    print("=" * 70)
    print("VMC ground-state energies vs exact references")
    print("=" * 70)
    header = (
        f"{'System':>6s} | {'E_VMC (Ha)':>20s} | {'Exact (Ha)':>11s} | "
        f"{'Error':>10s} | {'Status':>8s}"
    )
    print(header)
    print("-" * len(header))
    energy_errors_mha: dict[str, float] = {}
    for system in SYSTEMS:
        result = results[system.name]
        energy = float(result.energy)
        stderr = float(result.energy_error)
        error_mha = (energy - system.exact_energy) * 1000.0
        energy_errors_mha[system.name] = error_mha
        within = abs(error_mha) < CHEMICAL_ACCURACY_MHA
        status = "chem-acc" if within else "outside"
        print(
            f"{system.name:>6s} | {energy:+.5f} +/- {stderr:.5f} | "
            f"{system.exact_energy:+11.4f} | {error_mha:+7.2f} mHa | {status:>8s}"
        )
    print("-" * len(header))
    print(f"Chemical accuracy threshold: |error| < {CHEMICAL_ACCURACY_MHA} mHa (1 kcal/mol)")
    print()
    print("References: H -0.5 (exact), He -2.9037 (Pekeris 1958), H2@R=1.4bohr -1.1745")
    print("(exact Born-Oppenheimer). These are recoveries of known physics, not a")
    print("benchmark: the variational principle bounds every energy from above, so")
    print("recovering them to <1 mHa demonstrates the ansatz + sampler + optimiser are")
    print("correct and accurate -- the same stack scales unchanged to larger molecules.")

    # Visualization.
    colors = {"H": "#1f77b4", "He": "#d62728", "H2": "#2ca02c"}

    # Energy-convergence curves: error above exact (Ha) per iteration, symmetric-log.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for system in SYSTEMS:
        history = np.asarray(results[system.name].energy_history)
        error = history - system.exact_energy
        ax.plot(
            np.arange(1, history.shape[0] + 1),
            error,
            color=colors[system.name],
            lw=1.4,
            label=f"{system.name} (final {error[-1] * 1000:+.2f} mHa)",
        )
    ax.axhspan(
        -CHEMICAL_ACCURACY_MHA / 1000.0,
        CHEMICAL_ACCURACY_MHA / 1000.0,
        color="#7f7f7f",
        alpha=0.15,
        label="chemical accuracy (+/-1 kcal/mol)",
    )
    ax.axhline(0.0, color="#7f7f7f", ls="--", lw=1)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("VMC iteration")
    ax.set_ylabel("Energy error above exact (Ha)")
    ax.set_title("VMC energy convergence (H, He, H2)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "energy_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Hydrogen walker density vs the exact 1s radial distribution 4 r^2 |psi_1s|^2.
    h_walkers = np.asarray(results["H"].walkers)  # (batch, 1, 3)
    radii = np.linalg.norm(h_walkers[:, 0, :], axis=-1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        radii,
        bins=60,
        density=True,
        color=colors["H"],
        alpha=0.55,
        label="sampled walkers |psi|^2",
    )
    r_grid = np.linspace(0.0, radii.max(), 400)
    exact_radial = 4.0 * r_grid**2 * (np.exp(-r_grid) ** 2 / np.pi) * np.pi  # 4 r^2 |psi_1s|^2
    ax.plot(r_grid, exact_radial, color="#000000", lw=2, label="exact 1s: $4 r^2 |\\psi_{1s}|^2$")
    ax.set_xlabel("Electron-nucleus distance r (bohr)")
    ax.set_ylabel("Radial probability density")
    ax.set_title("Hydrogen atom: VMC walker density vs exact 1s")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hydrogen_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plots to {OUTPUT_DIR}/")
    print("  energy_convergence.png, hydrogen_density.png")

    return {f"{name}_energy_error_mha": error for name, error in energy_errors_mha.items()}


# %% [markdown]
"""
## Summary

A thin composition of opifex's VMC stack -- a `FermiNet` generalized-Slater
ansatz, the harmonic-mean `MetropolisHastingsSampler`, and the `VMCDriver` with
the SPRING natural-gradient optimiser -- recovers the ground-state energies of
the hydrogen atom, the helium atom, and the hydrogen molecule to **chemical
accuracy** (better than 1 mHa) in about two minutes on a single GPU. Each
optimisation step is one `jit`-compiled kernel: a fused `lax.scan` Metropolis
sweep, a forward-Laplacian local energy, and a sample-space SPRING update.

Because the variational principle bounds every recovered energy from above,
matching the exact references is a faithful recovery of known physics rather than
a fit to labels -- and the identical ansatz, sampler and optimiser scale, with
larger backbones and walker counts, to molecules where no exact answer exists.
See [Variational Monte Carlo](../../methods/variational-monte-carlo.md) for the
backbone -> local-energy -> optimiser design and how to swap in a PsiFormer
attention backbone or a MALA sampler.
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
