# Neural Density Functional Theory

## Overview

The **Neural Density Functional Theory (Neural DFT)** framework in Opifex combines traditional DFT methodology with neural network enhancements to achieve chemical accuracy with improved efficiency. It integrates neural exchange-correlation (XC) functionals and neural-enhanced Self-Consistent Field (SCF) solvers into a unified, high-precision framework.

Key features include:

- **Neural XC Functionals**: Deep learning models that capture non-local electron correlations using attention mechanisms.
- **Neural SCF Solver**: Accelerated convergence using intelligent density mixing and convergence prediction.
- **Chemical Accuracy**: Built-in diagnostics and optimization targets for achieving 1 kcal/mol accuracy.
- **Flax NNX Integration**: Fully compatible with JAX transformations and modern neural network patterns.

## Core Components

### Neural DFT Driver

The `NeuralDFT` class is the main entry point for performing calculations. It orchestrates the interaction between the molecular system, the XC functional, and the SCF solver.

```python
import jax
import jax.numpy as jnp
from flax import nnx
from opifex.neural.quantum import NeuralDFT

# Initialize RNGs
rngs = nnx.Rngs(0)

# Create Neural DFT driver
dft_driver = NeuralDFT(
    grid_size=1000,
    convergence_threshold=1e-8,
    max_scf_iterations=100,
    xc_functional_type="neural",  # Use neural XC functional
    mixing_strategy="neural",     # Use neural density mixing
    chemical_accuracy_target=1e-6, # ~1 kcal/mol
    rngs=rngs
)
```

### Neural XC Functional

The `NeuralXCFunctional` replaces traditional approximations (like LDA or GGA) with a neural network that learns the exchange-correlation energy density from electron density and its gradients. It uses:

- **Density Feature Extraction**: Captures local and semi-local physics.
- **Multi-Head Attention**: Models long-range non-local interactions.
- **Physics Constraints**: Enforces exact conditions and bounds.

### Neural SCF Solver

The `NeuralSCFSolver` accelerates the iterative solution of the Kohn-Sham equations:

- **Density Mixing Network**: Predicts optimal mixing of densities between iterations to suppress charge sloshing.
- **Convergence Predictor**: Estimates the probability of convergence and remaining iterations.

## Usage Examples

### 1. Basic Energy Calculation

Calculate the ground state energy of a molecular system.

```python
from opifex.core.quantum.molecular_system import create_molecular_system

# Define a molecule (e.g., H2)
h2_system = create_molecular_system(
    atoms=[('H', (0.0, 0.0, 0.0)), ('H', (0.74, 0.0, 0.0))],
    charge=0,
    multiplicity=1
)

# Compute energy
result = dft_driver.compute_energy(h2_system)

print(f"Total Energy: {result.total_energy:.6f} Ha")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
```

### 2. Customizing Components

You can customize the neural components for specific research needs.

```python
from opifex.neural.quantum import NeuralXCFunctional, NeuralSCFSolver

# Custom XC Functional
custom_xc = NeuralXCFunctional(
    hidden_sizes=[256, 256, 128],
    use_attention=True,
    num_attention_heads=4,
    rngs=rngs
)

# Custom SCF Solver
custom_scf = NeuralSCFSolver(
    convergence_threshold=1e-9,
    mixing_strategy="neural",
    rngs=rngs
)

# Inject into driver (if supported by API or subclassing)
# Note: Currently NeuralDFT initializes its own components based on config.
# To use custom components, you would typically modify the driver or
# use the components directly in a custom loop.
```

### 3. Chemical Accuracy Prediction

The framework provides tools to assess the reliability of the results.

```python
# Predict accuracy
accuracy_metrics = dft_driver.predict_chemical_accuracy(h2_system)

print(f"Predicted Error: {accuracy_metrics['predicted_error_kcal_mol']:.2f} kcal/mol")
print(f"Within Chemical Accuracy: {accuracy_metrics['within_chemical_accuracy_prediction']}")
```

## Advanced Configuration

### Precision Settings

For quantum chemistry, numerical precision is critical. `NeuralDFT` supports high-precision modes.

```python
dft_high_prec = NeuralDFT(
    enable_high_precision=True,  # Use float64 where critical
    convergence_threshold=1e-10,
    rngs=rngs
)
```

### Physics Constraints

The neural functional enforces physics constraints to ensure generalizability.

- **Positivity**: Electron density is strictly non-negative.
- **Symmetry**: Respects rotational and translational symmetries (via invariant features).
- **Asymptotic Behavior**: Correct long-range decay of potentials.

## API Reference

For detailed API documentation, see [Neural Quantum API](../api/neural.md#opifex.neural.quantum).
