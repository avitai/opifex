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
# Equivariant DFT Hamiltonian Prediction (QHNet-style)

| Property      | Value                                                       |
|---------------|-------------------------------------------------------------|
| Level         | Advanced                                                    |
| Runtime       | ~2 min (GPU)                                                |
| Memory        | ~2 GB                                                       |
| Prerequisites | JAX, Flax NNX, SE(3)-equivariant networks, DFT/Hartree-Fock |

## Overview

Predict the **dense atomic-orbital DFT/Hartree-Fock Hamiltonian (Fock) matrix
`H`** and the **overlap matrix `S`** of a molecule directly from its geometry,
with the QHNet-style equivariant predictor (Yu et al. 2023, arXiv:2306.04922).
The predicted matrix is SE(3)-equivariant *by construction* — rotating the
molecule rotates `H` by the block-diagonal Wigner-D matrix,
`H(R x) = D(R) H(x) D(R)^T` — so a single converged self-consistent-field (SCF)
solution is enough geometric supervision to learn the operator.

Ground truth comes from **PySCF**: a restricted Hartree-Fock (RHF) calculation in
the STO-3G minimal basis gives the converged Fock matrix and the AO overlap, in
opifex's exact shell/AO ordering (`cart=True`: atom-major, `s` shells then `p` in
`(x, y, z)`). The predictor is fit to that matrix, and the eigenvalues of the
fitted `H` against the true `S` recover the molecular-orbital energies — the
quantity a downstream calculation actually consumes.

This example is deliberately **thin**: it composes opifex's committed
electronic-structure stack and changes no library internals.

- `AtomicOrbitalBasis.from_molecular_system` builds the STO-3G shell/AO layout.
- `HamiltonianPredictor` (registered `@register_property_head("hamiltonian")`) is
  the QHNet-style predictor: a NequIP steerable trunk feeds per-shell-pair
  `PairExpansion` blocks (node blocks on the diagonal, edge blocks off-diagonal),
  scattered into a dense matrix and symmetrized `H = H~ + H~^T`.
- PySCF supplies the converged RHF Fock `H` and overlap `S`.
- `optax` Adam with a jitted step overfits the predictor to one geometry.
- `wigner_d` assembles the block-diagonal AO rotation for the equivariance check.

## Learning Goals

1. Generate a converged RHF Fock `H` and overlap `S` from PySCF in opifex's AO
   ordering
2. Build the equivariant `HamiltonianPredictor` from the library
3. Fit it to the Fock matrix with a jitted Adam step
4. Report element-wise MAE on `H` and `S`, and recover the orbital energies
5. Verify the block-wise rotational equivariance `H(R x) = D(R) H(x) D(R)^T`
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import time
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np


mpl.use("Agg")
import matplotlib.pyplot as plt
import optax
import scipy.linalg
from flax import nnx
from pyscf import gto, scf
from scipy.spatial.transform import Rotation

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.quantum.hamiltonian import (
    HamiltonianPredictor,
    HamiltonianPredictorConfig,
)


# Tighten GPU float32 matmuls (3x TF32 passes) so the SE(3)-equivariance residual
# is dominated by real model error, not by reduced-precision matmul on the GPU.
jax.config.update("jax_default_matmul_precision", "high")

print("=" * 70)
print("Opifex Example: Equivariant DFT Hamiltonian Prediction (QHNet-style)")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Configuration

The molecules are **water** (the main target, exercising O `1s/2s/2p` and H `1s`
shells, hence the `s`/`p` cross blocks) and **H2** (a two-atom `s`-only sanity
target). Both use the STO-3G minimal basis in atomic units (positions in Bohr).

The predictor carries steerable hidden features up to `l_max = 2` (so the trunk
can represent every degree the `s`/`p` shell-pair blocks need: `0e`, `1o`, `2e`),
two tensor-product convolution layers, an 8-function Bessel radial basis, and a
generous cutoff so the small molecules form a complete graph. A single geometry
is fit, so a few hundred Adam steps converge the overfit; the predictor's
guaranteed equivariance means one geometry is enough to test that the
architecture can represent a real Fock/overlap matrix in opifex's AO ordering.

STO-3G for H/C/N/O is `s` and `p` only, so the Cartesian AO components already
coincide with the real spherical-harmonic (irrep) components in the same
`(x, y, z)` order, and **no Cartesian-to-spherical transform is needed**. Bases
with `d` orbitals (def2-SVP and up) need that transform on the predicted blocks;
that is a documented extension, not part of this example.
"""

# %%
BASIS = "sto-3g"
SEED = 0
NUM_STEPS = 600  # Adam steps to overfit a single geometry
LEARNING_RATE = 3e-3

# Predictor hyper-parameters. The hidden irreps must carry every degree the
# s/p shell-pair blocks reach (0e, 1o, 2e); the cutoff is large enough that the
# small molecules form a complete graph.
HIDDEN_IRREPS = "32x0e + 24x1o + 16x2e"
SH_LMAX = 2
NUM_INTERACTIONS = 2
NUM_RADIAL_BASIS = 8
CUTOFF = 8.0  # Bohr

# Water (O, H, H) and H2, in Bohr.
WATER_ATOMIC_NUMBERS = jnp.array([8, 1, 1])
WATER_POSITIONS = jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])
H2_ATOMIC_NUMBERS = jnp.array([1, 1])
H2_POSITIONS = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])

OUTPUT_DIR = Path("docs/assets/examples/hamiltonian_prediction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Basis: {BASIS} (s,p only -> Cartesian AOs equal the irreps; no transform)")
print(f"Predictor: irreps={HIDDEN_IRREPS}, layers={NUM_INTERACTIONS}, cutoff={CUTOFF} Bohr")
print(f"Fit: Adam (lr={LEARNING_RATE}), {NUM_STEPS} steps per geometry")

# %% [markdown]
"""
## Ground Truth from PySCF

A restricted Hartree-Fock calculation gives the converged Fock matrix `H` and the
AO overlap `S`. PySCF is run with `cart=True` so its Cartesian-AO ordering matches
opifex's STO-3G shell/AO layout exactly (atom-major; `s` shells then `p` in
`(x, y, z)`), so the matrices need no reordering before they become fit targets.
"""


# %%
def pyscf_targets(atomic_numbers: jax.Array, positions: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    """Return the converged RHF Fock ``H`` and overlap ``S`` from PySCF (cart order).

    Args:
        atomic_numbers: Nuclear charges [Shape: (n_atoms,)].
        positions: Atomic positions in Bohr [Shape: (n_atoms, 3)].

    Returns:
        A pair ``(fock, overlap)`` of ``(n_ao, n_ao)`` NumPy matrices in opifex's
        AO ordering.
    """
    atoms = [
        (int(z), tuple(float(c) for c in pos))
        for z, pos in zip(np.asarray(atomic_numbers), np.asarray(positions), strict=True)
    ]
    molecule = gto.M(atom=atoms, basis=BASIS, unit="Bohr", cart=True)
    mean_field = scf.RHF(molecule)
    mean_field.kernel()
    overlap = np.asarray(molecule.intor("int1e_ovlp"))
    fock = np.asarray(mean_field.get_fock())
    return fock, overlap


print()
print("Running PySCF RHF/STO-3G for water and H2...")
water_system = MolecularSystem(
    atomic_numbers=WATER_ATOMIC_NUMBERS, positions=WATER_POSITIONS, basis_set=BASIS
)
h2_system = MolecularSystem(
    atomic_numbers=H2_ATOMIC_NUMBERS, positions=H2_POSITIONS, basis_set=BASIS
)
water_fock, water_overlap = pyscf_targets(WATER_ATOMIC_NUMBERS, WATER_POSITIONS)
h2_fock, h2_overlap = pyscf_targets(H2_ATOMIC_NUMBERS, H2_POSITIONS)
print(f"Water ({water_system.molecular_formula}): {water_fock.shape[0]} AOs")
print(f"H2    ({h2_system.molecular_formula}): {h2_fock.shape[0]} AOs")
print(f"Fock scale (|H|max): water {np.abs(water_fock).max():.3f}, H2 {np.abs(h2_fock).max():.3f}")

# %% [markdown]
"""
## Building the Predictor

`HamiltonianPredictor` reads its block layout from the molecule's
`AtomicOrbitalBasis`: every intra-atom shell pair `(l_i, l_j)` becomes a diagonal
block driven by a node feature, and every directed atom pair an off-diagonal block
driven by an edge feature. A shared `PairExpansion` per angular-momentum pair type
turns the steerable feature into the dense `(2 l_i + 1) x (2 l_j + 1)` block via a
Clebsch-Gordan (Wigner-Eckart) contraction, so the assembled matrix is symmetric
and equivariant regardless of the weights.

A separate `property_name="overlap"` predictor fits `S` with the identical
mechanism — the overlap obeys the same transformation law as the Fock.
"""


# %%
def build_predictor(
    system: MolecularSystem, *, property_name: str, seed: int
) -> HamiltonianPredictor:
    """Build a `HamiltonianPredictor` bound to ``system``'s STO-3G basis.

    Args:
        system: The molecular system fixing the AO/shell-pair block layout.
        property_name: The emitted matrix property (`"hamiltonian"` or `"overlap"`).
        seed: Seed for all learnable weights.

    Returns:
        A freshly initialised predictor for ``system``.
    """
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name=BASIS)
    config = HamiltonianPredictorConfig(
        hidden_irreps=HIDDEN_IRREPS,
        sh_lmax=SH_LMAX,
        num_interactions=NUM_INTERACTIONS,
        num_radial_basis=NUM_RADIAL_BASIS,
        cutoff=CUTOFF,
        property_name=property_name,
    )
    return HamiltonianPredictor(basis=basis, config=config, rngs=nnx.Rngs(seed))


hamiltonian_predictor = build_predictor(water_system, property_name="hamiltonian", seed=SEED)
num_params = sum(
    int(np.prod(leaf.shape))
    for leaf in jax.tree_util.tree_leaves(nnx.state(hamiltonian_predictor, nnx.Param))
)
print(f"Trainable parameters: {num_params}")

# %% [markdown]
"""
## Fitting the Fock Matrix

The fit minimises the mean-squared error between the predicted and the PySCF Fock
matrix with Adam. The step is jitted with `nnx.jit`; because the predictor is
`jit`/`grad`/`vmap` clean over geometry, the whole fit runs on the GPU at the
jitted path. A single geometry is overfit — the predictor's guaranteed
equivariance means this directly tests whether the architecture can *represent* a
real Fock matrix in opifex's AO ordering, not just memorise scalars.
"""


# %%
def fit_matrix(
    predictor: HamiltonianPredictor,
    system: MolecularSystem,
    target: np.ndarray,
    *,
    property_name: str,
) -> tuple[HamiltonianPredictor, list[float]]:
    """Overfit ``predictor`` to a single ground-truth matrix.

    Args:
        predictor: The predictor to train (mutated in place).
        system: The molecular system the matrix belongs to.
        target: The ground-truth ``(n_ao, n_ao)`` matrix to fit.
        property_name: The predictor's emitted matrix key.

    Returns:
        A pair ``(predictor, loss_history)`` with the per-step MSE loss.
    """
    optimizer = nnx.Optimizer(predictor, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    target_array = jnp.asarray(target)

    def loss_fn(module: HamiltonianPredictor) -> jax.Array:
        prediction = module(system)[property_name]
        return jnp.mean((prediction - target_array) ** 2)

    @nnx.jit
    def step(module: HamiltonianPredictor, opt: nnx.Optimizer) -> jax.Array:
        loss, grads = nnx.value_and_grad(loss_fn)(module)
        opt.update(module, grads)
        return loss

    history: list[float] = []
    for _ in range(NUM_STEPS):
        history.append(float(step(predictor, optimizer)))
    return predictor, history


print()
print("Fitting the water Fock matrix H...")
start_time = time.time()
hamiltonian_predictor, h_loss_history = fit_matrix(
    hamiltonian_predictor, water_system, water_fock, property_name="hamiltonian"
)
print(f"Fit complete in {time.time() - start_time:.1f}s (final MSE {h_loss_history[-1]:.3e})")

print("Fitting the water overlap matrix S...")
overlap_predictor = build_predictor(water_system, property_name="overlap", seed=SEED + 1)
overlap_predictor, s_loss_history = fit_matrix(
    overlap_predictor, water_system, water_overlap, property_name="overlap"
)
print(f"Fit complete (final MSE {s_loss_history[-1]:.3e})")

# %% [markdown]
"""
## Evaluation

Element-wise MAE on the Fock `H` and overlap `S`, reported both absolutely (in
Hartree for `H`, dimensionless for `S`) and relative to each matrix's scale. The
physics-level check solves the generalised eigenvalue problem `H C = S C eps` for
the *fitted* `H` and the *true* `S`, recovering the converged molecular-orbital
energies — the quantity a downstream calculation consumes.
"""

# %%
water_h_pred = np.asarray(hamiltonian_predictor(water_system)["hamiltonian"])
water_s_pred = np.asarray(overlap_predictor(water_system)["overlap"])

h_mae = float(np.mean(np.abs(water_h_pred - water_fock)))
s_mae = float(np.mean(np.abs(water_s_pred - water_overlap)))
h_scale = float(np.abs(water_fock).max())
s_scale = float(np.abs(water_overlap).max())

# Also fit H2 (s-only) as a second, simpler target.
h2_predictor = build_predictor(h2_system, property_name="hamiltonian", seed=SEED + 2)
h2_predictor, _ = fit_matrix(h2_predictor, h2_system, h2_fock, property_name="hamiltonian")
h2_h_pred = np.asarray(h2_predictor(h2_system)["hamiltonian"])
h2_mae = float(np.mean(np.abs(h2_h_pred - h2_fock)))

# Orbital energies from the predicted H with the true S (generalised eigenproblem).
reference_energies = np.sort(scipy.linalg.eigh(water_fock, water_overlap, eigvals_only=True))
predicted_energies = np.sort(scipy.linalg.eigh(water_h_pred, water_overlap, eigvals_only=True))
energy_mae = float(np.mean(np.abs(predicted_energies - reference_energies)))

print("=" * 70)
print("Fit quality (single geometry, STO-3G)")
print("=" * 70)
print(
    f"Water Fock  H MAE: {h_mae:.4e} Hartree  ({100 * h_mae / h_scale:.2f}% of |H|max={h_scale:.3f})"
)
print(
    f"Water overlap S MAE: {s_mae:.4e}        ({100 * s_mae / s_scale:.2f}% of |S|max={s_scale:.3f})"
)
print(f"H2    Fock  H MAE: {h2_mae:.4e} Hartree")
print()
print("Molecular-orbital energies of the *predicted* H with the *true* S (Hartree):")
print(f"  PySCF RHF:  {np.array2string(reference_energies, precision=4, floatmode='fixed')}")
print(f"  Predicted:  {np.array2string(predicted_energies, precision=4, floatmode='fixed')}")
print(f"  MO-energy MAE: {energy_mae:.4e} Hartree")

# %% [markdown]
"""
## Equivariance Check

The defining property: under a random proper rotation `R` of the geometry the
predicted matrix must transform as `H(R x) = D(R) H(x) D(R)^T`, where `D(R)` is
the **block-diagonal Wigner-D matrix** assembled per shell — one `wigner_d(l, R)`
block per shell of degree `l`. This holds for *any* weights (the predictor is
equivariant by construction), and is measured here on the *fitted* water
predictor across several random rotations.
"""


# %%
def block_diagonal_wigner(basis: AtomicOrbitalBasis, rotation: jax.Array) -> jax.Array:
    """Assemble the per-shell block-diagonal Wigner-D rotation of the AO basis.

    Args:
        basis: The AO basis whose shells fix the block-diagonal layout.
        rotation: A ``3x3`` proper rotation matrix.

    Returns:
        The ``(n_ao, n_ao)`` block-diagonal Wigner-D matrix ``D(R)``.
    """
    blocks = [wigner_d(shell.angular_momentum, rotation) for shell in basis.shells]
    return jax.scipy.linalg.block_diag(*blocks)


water_basis = AtomicOrbitalBasis.from_molecular_system(water_system, basis_name=BASIS)
base_matrix = hamiltonian_predictor(water_system)["hamiltonian"]

print()
print("Block-wise equivariance H(R x) = D(R) H(x) D(R)^T under random rotations:")
equivariance_errors: list[float] = []
for seed in range(5):
    rotation = jnp.asarray(Rotation.random(rng=np.random.default_rng(seed)).as_matrix())
    rotated_system = MolecularSystem(
        atomic_numbers=water_system.atomic_numbers,
        positions=water_system.positions @ rotation.T,
        basis_set=BASIS,
    )
    rotated_matrix = hamiltonian_predictor(rotated_system)["hamiltonian"]
    wigner = block_diagonal_wigner(water_basis, rotation)
    expected = wigner @ base_matrix @ wigner.T
    error = float(jnp.max(jnp.abs(rotated_matrix - expected)))
    equivariance_errors.append(error)
    print(f"  rotation {seed}: max |H(Rx) - D H D^T| = {error:.3e}")

max_equivariance_error = max(equivariance_errors)
print(f"Worst-case equivariance error: {max_equivariance_error:.3e}")

# %% [markdown]
"""
## Visualization

Three diagnostics: the predicted-vs-PySCF Fock heatmaps (with the signed
residual), the MSE fit-loss curves for `H` and `S`, and the per-rotation
equivariance error.
"""

# %%
# Predicted vs PySCF Fock heatmaps + signed residual.
vmax = float(np.abs(water_fock).max())
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, data, title in (
    (axes[0], water_fock, "PySCF RHF Fock H"),
    (axes[1], water_h_pred, "Predicted Fock H"),
):
    image = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("AO index")
    ax.set_ylabel("AO index")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
residual = water_h_pred - water_fock
rmax = float(np.abs(residual).max())
image = axes[2].imshow(residual, cmap="RdBu_r", vmin=-rmax, vmax=rmax)
axes[2].set_title(f"Residual (predicted - PySCF), MAE {h_mae:.2e}")
axes[2].set_xlabel("AO index")
axes[2].set_ylabel("AO index")
fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
fig.suptitle("Equivariant Fock prediction for water (STO-3G)", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fock_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Fit-loss curves for H and S.
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, NUM_STEPS + 1), h_loss_history, color="#1f77b4", label="Fock H")
ax.plot(range(1, NUM_STEPS + 1), s_loss_history, color="#ff7f0e", label="Overlap S")
ax.set_yscale("log")
ax.set_xlabel("Adam step")
ax.set_ylabel("Mean-squared error")
ax.set_title("Fit loss (water, STO-3G)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Equivariance error per rotation.
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(equivariance_errors)), equivariance_errors, color="#2ca02c")
ax.set_yscale("log")
ax.set_xlabel("Random rotation")
ax.set_ylabel("max |H(Rx) - D(R) H(x) D(R)^T|")
ax.set_title("Block-wise SE(3) equivariance error")
ax.grid(True, which="both", axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "equivariance_error.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print()
print(f"Saved plots to {OUTPUT_DIR}/")
print("  fock_heatmaps.png, loss_curve.png, equivariance_error.png")

# %% [markdown]
"""
## Summary

A thin composition of opifex's electronic-structure stack —
`AtomicOrbitalBasis`, the registered `HamiltonianPredictor`, and a jitted Adam
fit — predicts the dense DFT/Hartree-Fock Fock matrix `H` and overlap `S` of a
molecule from its geometry, fit against PySCF ground truth in opifex's exact AO
ordering. The fitted matrix reproduces the PySCF matrices to a small element-wise
MAE, and the molecular-orbital energies of the predicted `H` (with the true `S`)
match the converged RHF eigenvalues.

The defining guarantee is **SE(3) equivariance by construction**:
`H(R x) = D(R) H(x) D(R)^T` holds to matmul precision for *any* weights, because
the predictor assembles each block from a Clebsch-Gordan (Wigner-Eckart)
contraction of a steerable feature and symmetrizes `H = H~ + H~^T`. One converged
geometry is therefore enough geometric supervision to learn an equivariant
operator.

This example uses STO-3G (H/C/N/O `s`/`p` only), where the Cartesian AO components
already equal the irrep components, so no Cartesian-to-spherical transform is
needed. Extending to larger bases with `d` orbitals (def2-SVP), the SO(2)-frame
convolution for scalability (QHNetV2), and the QH9 benchmark are documented in
[Hamiltonian Prediction](../../methods/hamiltonian-prediction.md).
"""
