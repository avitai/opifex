# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Neural DFT: H2 Molecule Ground State Energy

This example demonstrates computing the ground-state energy of an H2 molecule
using Opifex's Neural Density Functional Theory framework. Neural DFT combines
traditional DFT methodology with neural network-enhanced exchange-correlation
functionals and SCF solvers.

**Key Concepts:**
- Neural exchange-correlation (XC) functional
- Self-consistent field (SCF) iteration with neural mixing
- Chemical accuracy assessment
- Molecular system representation
"""

# %%
# Configuration
SEED = 42
GRID_SIZE = 100  # Electron density grid size
MAX_SCF_ITERATIONS = 50
CONVERGENCE_THRESHOLD = 1e-6
H2_BOND_LENGTH_ANGSTROM = 0.74  # Equilibrium H-H bond length

# Reference energy for H2 at equilibrium (Hartree)
# From high-level ab initio calculations
H2_REFERENCE_ENERGY = -1.174  # Approximate HF/DFT reference

# Output directory
OUTPUT_DIR = "docs/assets/examples/neural_dft"

# %%
print("=" * 70)
print("Opifex Example: Neural DFT on H2 Molecule")
print("=" * 70)

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx


print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %%
from opifex.core.quantum.molecular_system import (
    ANGSTROM_TO_BOHR,
    create_molecular_system,
)
from opifex.neural.quantum import NeuralDFT


# %% [markdown]
"""
## Step 1: Create H2 Molecular System

We define an H2 molecule with atoms along the z-axis. Positions are specified
in Angstrom and converted to Bohr (atomic units) internally.
"""

# %%
print()
print("Creating H2 molecular system...")

# Create H2 molecule using the helper function
# Positions in Angstrom, converted internally to Bohr
h2_molecule = create_molecular_system(
    atoms=[
        ("H", (0.0, 0.0, -H2_BOND_LENGTH_ANGSTROM / 2)),
        ("H", (0.0, 0.0, H2_BOND_LENGTH_ANGSTROM / 2)),
    ],
    charge=0,
    multiplicity=1,  # Singlet ground state
    basis_set="sto-3g",
)

# Print system information
print(f"  Molecular formula: {h2_molecule.molecular_formula}")
print(f"  Number of atoms: {h2_molecule.n_atoms}")
print(f"  Number of electrons: {h2_molecule.n_electrons}")
print(f"  Charge: {h2_molecule.charge}")
print(f"  Multiplicity: {h2_molecule.multiplicity}")
print(f"  Bond length: {H2_BOND_LENGTH_ANGSTROM:.2f} Angstrom")
print(f"  Bond length: {H2_BOND_LENGTH_ANGSTROM * ANGSTROM_TO_BOHR:.4f} Bohr")
print(f"  Quantum valid: {h2_molecule.validate_quantum_system()}")

# %%
# Also create alternative representation directly
print()
print("Positions (Bohr):")
for symbol, pos in zip(h2_molecule.symbols, h2_molecule.positions, strict=False):
    print(f"  {symbol}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")

# %% [markdown]
"""
## Step 2: Initialize Neural DFT Framework

The Neural DFT framework integrates:
1. **Neural XC Functional**: Learns exchange-correlation energy from density
2. **Neural SCF Solver**: Accelerates self-consistent field convergence
3. **Precision Control**: High-precision mode for chemical accuracy
"""

# %%
print()
print("Initializing Neural DFT framework...")

rngs = nnx.Rngs(SEED)

neural_dft = NeuralDFT(
    grid_size=GRID_SIZE,
    convergence_threshold=CONVERGENCE_THRESHOLD,
    max_scf_iterations=MAX_SCF_ITERATIONS,
    xc_functional_type="neural",  # Use neural XC functional
    mixing_strategy="neural",  # Neural density mixing
    use_neural_scf=True,
    chemical_accuracy_target=0.043,  # 1 kcal/mol in Hartree
    enable_high_precision=True,
    rngs=rngs,
)

print(f"  Grid size: {neural_dft.grid_size}")
print(f"  Convergence threshold: {neural_dft.convergence_threshold}")
print(f"  Max SCF iterations: {neural_dft.max_scf_iterations}")
print(f"  XC functional type: {neural_dft.xc_functional_type}")
print(f"  Mixing strategy: {neural_dft.mixing_strategy}")
print(f"  Chemical accuracy target: {neural_dft.chemical_accuracy_target} Ha")

# %% [markdown]
"""
## Step 3: Compute Ground State Energy

The `compute_energy` method performs:
1. Initial density guess (atomic density superposition)
2. SCF iterations with neural mixing
3. Energy decomposition (electronic, nuclear repulsion, XC)
4. Convergence and accuracy assessment
"""

# %%
print()
print("Computing H2 ground state energy...")
print("-" * 50)

# Compute DFT energy
result = neural_dft.compute_energy(h2_molecule, deterministic=True)

print()
print("SCF Convergence:")
print(f"  Converged: {result.converged}")
print(f"  Iterations: {result.iterations}")
print()
print("Energy Components (Hartree):")
print(f"  Total Energy:             {result.total_energy:.6f}")
print(f"  Electronic Energy:        {result.electronic_energy:.6f}")
print(f"  Nuclear Repulsion Energy: {result.nuclear_repulsion_energy:.6f}")
print(f"  XC Energy:                {result.xc_energy:.6f}")

# %% [markdown]
"""
## Step 4: Analyze Accuracy

Compare computed energy with reference values and assess whether
chemical accuracy (1 kcal/mol = 0.0016 Ha) was achieved.
"""

# %%
print()
print("Accuracy Analysis:")
print("-" * 50)

error_hartree = abs(result.total_energy - H2_REFERENCE_ENERGY)
error_kcal_mol = error_hartree * 627.5  # Convert to kcal/mol

print(f"  Reference Energy:         {H2_REFERENCE_ENERGY:.6f} Ha")
print(f"  Computed Energy:          {result.total_energy:.6f} Ha")
print(f"  Absolute Error:           {error_hartree:.6f} Ha")
print(f"  Error (kcal/mol):         {error_kcal_mol:.2f} kcal/mol")
print()
print("  Chemical Accuracy Target: 1.0 kcal/mol (0.0016 Ha)")
print(f"  Chemical Accuracy Met:    {result.chemical_accuracy_achieved}")

# Precision metrics
if result.precision_metrics:
    print()
    print("Precision Metrics:")
    for key, value in result.precision_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6e}")
        else:
            print(f"  {key}: {value}")

# %% [markdown]
"""
## Step 5: Potential Energy Curve

Scan over H-H bond lengths to compute the potential energy curve (PEC).
This demonstrates how the energy varies with molecular geometry.
"""

# %%
print()
print("Computing Potential Energy Curve...")
print("-" * 50)

# Bond lengths to scan (Angstrom)
bond_lengths = jnp.linspace(0.5, 2.0, 16)
energies = []
convergence_status = []

for i, bond_length in enumerate(bond_lengths):
    # Create H2 at this bond length
    h2 = create_molecular_system(
        atoms=[
            ("H", (0.0, 0.0, -float(bond_length) / 2)),
            ("H", (0.0, 0.0, float(bond_length) / 2)),
        ],
        charge=0,
        multiplicity=1,
    )

    # Compute energy
    res = neural_dft.compute_energy(h2, deterministic=True)
    energies.append(res.total_energy)
    convergence_status.append(res.converged)

    if (i + 1) % 4 == 0:
        print(f"  Computed {i + 1}/{len(bond_lengths)} points...")

energies = jnp.array(energies)
print()
print("  PEC computation complete!")
print(f"  Converged points: {sum(convergence_status)}/{len(convergence_status)}")

# Find equilibrium
min_idx = int(jnp.argmin(energies))
equilibrium_bond_length = float(bond_lengths[min_idx])
equilibrium_energy = float(energies[min_idx])

print()
print(f"  Equilibrium bond length: {equilibrium_bond_length:.3f} Angstrom")
print(f"  Equilibrium energy:      {equilibrium_energy:.6f} Ha")
print(f"  Literature value:        {H2_BOND_LENGTH_ANGSTROM:.2f} Angstrom")

# %% [markdown]
"""
## Step 6: Visualization
"""

# %%
print()
print("Generating visualizations...")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# %%
# Figure 1: Potential Energy Curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Full PEC
ax1 = axes[0]
ax1.plot(bond_lengths, energies, "b-o", linewidth=2, markersize=6, label="Neural DFT")
ax1.axvline(
    equilibrium_bond_length,
    color="r",
    linestyle="--",
    alpha=0.7,
    label=f"Min: {equilibrium_bond_length:.2f} A",
)
ax1.axvline(
    H2_BOND_LENGTH_ANGSTROM,
    color="g",
    linestyle=":",
    alpha=0.7,
    label=f"Exp: {H2_BOND_LENGTH_ANGSTROM:.2f} A",
)
ax1.set_xlabel("H-H Bond Length (Angstrom)", fontsize=12)
ax1.set_ylabel("Total Energy (Hartree)", fontsize=12)
ax1.set_title("H2 Potential Energy Curve", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Energy near equilibrium
ax2 = axes[1]
near_eq_mask = (bond_lengths >= 0.6) & (bond_lengths <= 1.2)
ax2.plot(
    bond_lengths[near_eq_mask],
    energies[near_eq_mask],
    "b-o",
    linewidth=2,
    markersize=8,
)
ax2.axvline(
    equilibrium_bond_length,
    color="r",
    linestyle="--",
    alpha=0.7,
    label=f"Computed: {equilibrium_bond_length:.2f} A",
)
ax2.axvline(
    H2_BOND_LENGTH_ANGSTROM,
    color="g",
    linestyle=":",
    alpha=0.7,
    label=f"Literature: {H2_BOND_LENGTH_ANGSTROM:.2f} A",
)
ax2.axhline(H2_REFERENCE_ENERGY, color="orange", linestyle="-.", alpha=0.7)
ax2.set_xlabel("H-H Bond Length (Angstrom)", fontsize=12)
ax2.set_ylabel("Total Energy (Hartree)", fontsize=12)
ax2.set_title("Near Equilibrium Region", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/potential_energy_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR}/potential_energy_curve.png")

# %%
# Figure 2: SCF Convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Convergence history
ax1 = axes[0]
if result.convergence_history is not None and len(result.convergence_history) > 1:
    iterations = jnp.arange(1, len(result.convergence_history) + 1)
    ax1.semilogy(iterations, jnp.abs(result.convergence_history), "b-o", linewidth=2)
    ax1.axhline(
        CONVERGENCE_THRESHOLD,
        color="r",
        linestyle="--",
        label=f"Threshold: {CONVERGENCE_THRESHOLD:.0e}",
    )
    ax1.set_xlabel("SCF Iteration", fontsize=12)
    ax1.set_ylabel("Energy Change (Hartree)", fontsize=12)
    ax1.set_title("SCF Convergence", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
else:
    ax1.text(
        0.5,
        0.5,
        "Single iteration or\nno convergence history",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax1.set_title("SCF Convergence", fontsize=14)

# Right: Final electron density
ax2 = axes[1]
if result.final_density is not None:
    grid_points = jnp.linspace(-10, 10, len(result.final_density))
    ax2.plot(grid_points, result.final_density, "b-", linewidth=2)
    ax2.fill_between(grid_points, result.final_density, alpha=0.3)
    ax2.set_xlabel("Position (Bohr)", fontsize=12)
    ax2.set_ylabel("Electron Density", fontsize=12)
    ax2.set_title("Final Electron Density", fontsize=14)
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, "No density available", ha="center", va="center", fontsize=12)
    ax2.set_title("Final Electron Density", fontsize=14)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/scf_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR}/scf_analysis.png")

# %% [markdown]
"""
## Step 7: Predict Chemical Accuracy

Use the built-in accuracy prediction to estimate error bounds.
"""

# %%
print()
print("Chemical Accuracy Prediction:")
print("-" * 50)

accuracy_prediction = neural_dft.predict_chemical_accuracy(
    h2_molecule, reference_energy=H2_REFERENCE_ENERGY
)

print(f"  Total Energy:              {accuracy_prediction['total_energy']:.6f} Ha")
print(f"  Converged:                 {accuracy_prediction['converged']}")
print(f"  Iterations:                {accuracy_prediction['iterations']}")

if "predicted_error_hartree" in accuracy_prediction:
    print()
    print(
        "  Predicted Error (Hartree): "
        f"{accuracy_prediction['predicted_error_hartree']:.6e}"
    )
    print(
        f"  Predicted Error (kcal/mol): {accuracy_prediction['predicted_error_kcal_mol']:.3f}"
    )
    print(
        f"  Within Chemical Accuracy:  {accuracy_prediction['within_chemical_accuracy_prediction']}"
    )

if "actual_error_hartree" in accuracy_prediction:
    print()
    print(
        f"  Actual Error (Hartree):    {accuracy_prediction['actual_error_hartree']:.6e}"
    )
    print(
        f"  Actual Error (kcal/mol):   {accuracy_prediction['actual_error_kcal_mol']:.3f}"
    )
    print(
        f"  Within Chemical Accuracy:  {accuracy_prediction['within_chemical_accuracy_actual']}"
    )

# %% [markdown]
"""
## Results Summary
"""

# %%
print()
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()
print(f"Molecule:                    {h2_molecule.molecular_formula}")
print(f"Number of electrons:         {h2_molecule.n_electrons}")
print(f"Grid size:                   {GRID_SIZE}")
print(f"SCF converged:               {result.converged}")
print(f"SCF iterations:              {result.iterations}")
print()
print(f"Total Energy:                {result.total_energy:.6f} Ha")
print(f"Reference Energy:            {H2_REFERENCE_ENERGY:.6f} Ha")
print(
    f"Error:                       {error_hartree:.6f} Ha ({error_kcal_mol:.2f} kcal/mol)"
)
print()
print(f"Equilibrium bond length:     {equilibrium_bond_length:.3f} A")
print(f"Literature bond length:      {H2_BOND_LENGTH_ANGSTROM:.2f} A")
print()
print(f"Chemical accuracy achieved:  {result.chemical_accuracy_achieved}")
print("=" * 70)

# %%
print()
print("Neural DFT H2 example completed successfully!")
