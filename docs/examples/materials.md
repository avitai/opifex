# Materials Science Examples

## Crystal Structure Prediction

```python
from opifex.neural.quantum import CrystalDFT
from ase.build import bulk

# Create crystal structure
silicon = bulk('Si', 'diamond', a=5.43)

# Neural DFT for crystals
crystal_dft = CrystalDFT()
energy = crystal_dft.calculate_energy(silicon)
forces = crystal_dft.calculate_forces(silicon)

print(f"Silicon crystal energy: {energy:.6f} eV/atom")
```

## Band Structure Calculation

```python
# Calculate electronic band structure
bands = crystal_dft.calculate_bands(
    crystal=silicon,
    k_path='GXL',
    n_bands=8
)

# Plot band structure
import matplotlib.pyplot as plt
plt.plot(bands.k_points, bands.eigenvalues)
plt.ylabel('Energy (eV)')
plt.title('Silicon Band Structure')
```

## Phase Diagram

```python
# Study phase transitions
temperatures = jnp.linspace(300, 1000, 20)
phase_energies = []

for T in temperatures:
    energy = neural_dft.calculate_free_energy(
        crystal=material,
        temperature=T
    )
    phase_energies.append(energy)

# Plot phase diagram
plt.plot(temperatures, phase_energies)
plt.xlabel('Temperature (K)')
plt.ylabel('Free Energy (eV)')
```
