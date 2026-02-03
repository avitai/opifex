# Quantum Chemistry Examples

## Molecular Energy Calculation

```python
import jax.numpy as jnp
from opifex.neural.quantum import NeuralDFT
from opifex.core.quantum import MolecularSystem

# Define water molecule
water = MolecularSystem(
    atomic_numbers=jnp.array([8, 1, 1]),  # O, H, H
    positions=jnp.array([
        [0.0, 0.0, 0.0],      # O
        [0.0, 0.757, 0.587],  # H
        [0.0, -0.757, 0.587]  # H
    ]),
    charge=0,
    multiplicity=1  # Singlet state
)

# Neural DFT calculation
neural_dft = NeuralDFT(
    molecular_system=water,
    functional_type='neural_xc'
)
energy = neural_dft.compute_energy()
print(f"Water molecule energy: {energy:.6f} Ha")
```

## Reaction Pathway

```python
# Scan reaction coordinate
reaction_path = neural_dft.scan_reaction_coordinate(
    reactants=[reactant_geometry],
    products=[product_geometry],
    n_points=20
)

# Plot energy profile
import matplotlib.pyplot as plt
plt.plot(reaction_path.coordinates, reaction_path.energies)
plt.xlabel('Reaction Coordinate')
plt.ylabel('Energy (Ha)')
plt.title('Reaction Energy Profile')
```
