# Neural Density Functional Theory (Neural DFT) Tutorial

## Introduction

Neural Density Functional Theory (Neural DFT) represents a significant advancement in quantum chemistry, combining the theoretical rigor of traditional Density Functional Theory with the power of modern neural networks. In the Opifex framework, Neural DFT is implemented as a **first-class paradigm**, peer to Physics-Informed Neural Networks (PINNs) and Neural Operators.

## Key Features

### Chemical Accuracy

- **Energy Accuracy**: <1 kcal/mol for molecular energies
- **Force Accuracy**: <0.1 eV/Å for atomic forces
- **Density Accuracy**: High-fidelity electron density predictions

### Physics Constraints

- **Particle Number Conservation**: Exact electron count preservation
- **Density Positivity**: Enforced positive electron density
- **Quantum Mechanical Principles**: Built-in conservation laws

### Neural Components

- **Neural Exchange-Correlation Functionals**: DM21-style equivariant functionals
- **ML-Accelerated SCF Methods**: Neural convergence acceleration
- **Hybrid Classical-Neural Approaches**: Multi-fidelity quantum mechanical models

## Architecture Integration

Neural DFT is distributed across all 6 layers of the Opifex framework:

### Layer 1: FLAX-NNX/JAX Core

```python
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.core.quantum import QuantumOperator

# Core neural DFT primitives
class NeuralXCFunctional(nnx.Module):
    """Neural exchange-correlation functional with equivariance."""

    def __init__(self, features: int = 64):
        self.dense_layers = [
            nnx.Linear(4, features),  # rho, grad_rho, tau, lapl_rho
            nnx.Linear(features, features),
            nnx.Linear(features, 1)   # XC energy density
        ]

    def __call__(self, density_features):
        x = density_features
        for layer in self.dense_layers[:-1]:
            x = nnx.gelu(layer(x))
        return self.dense_layers[-1](x)
```

### Layer 2: Mathematical Abstractions

```python
from opifex.core import ElectronicStructureProblem, MolecularSystem

# Quantum mechanical problem definition
class H2MoleculeProblem(ElectronicStructureProblem):
    """Hydrogen molecule electronic structure problem."""

    def __init__(self, bond_length: float = 0.74):
        atoms = jnp.array([[0.0, 0.0, 0.0], [bond_length, 0.0, 0.0]])
        charges = jnp.array([1, 1])  # Hydrogen atoms
        super().__init__(atoms=atoms, charges=charges, n_electrons=2)
```

### Layer 3: Opifex Primitives

```python
from opifex.neural.quantum import NeuralDFT

# Neural DFT as peer to PINNs and Neural Operators
neural_dft = NeuralDFT(
    molecular_system=h2_system,
    xc_functional=NeuralXCFunctional(),
    basis_set="cc-pVDZ",
    chemical_accuracy_target=1.0  # kcal/mol
)

# Unified training interface
from opifex.neural import UnifiedTrainer

trainer = UnifiedTrainer()
trainer.add_paradigm("neural_dft", neural_dft)
trainer.add_paradigm("pinn", physics_informed_net)
trainer.add_paradigm("fno", fourier_neural_operator)
```

### Layer 4: Composed Models

```python
from opifex.optimization import HybridDFTModel

# Hybrid classical-neural DFT
hybrid_model = HybridDFTModel(
    classical_functional="PBE",
    neural_correction=neural_xc_functional,
    mixing_parameter=0.25
)

# Multi-fidelity approach
multifidelity_dft = MultiFidelityDFT(
    low_fidelity="HF",      # Hartree-Fock
    high_fidelity="CCSD(T)", # Coupled Cluster
    neural_bridge=neural_dft
)
```

### Layer 5: Application Interfaces

```python
from opifex.applications import MolecularPropertyPredictor

# High-level molecular property prediction
predictor = MolecularPropertyPredictor(
    backend="neural_dft",
    properties=["energy", "forces", "dipole_moment"],
    chemical_accuracy=True
)

# Predict properties for new molecule
results = predictor.predict(molecule="CCO")  # Ethanol
print(f"Energy: {results.energy:.6f} Hartree")
print(f"Forces: {results.forces}")
```

### Layer 6: Production Ecosystem

```python
from opifex.deployment import NeuralDFTService

# Cloud deployment
service = NeuralDFTService(
    model=neural_dft,
    scaling="auto",
    chemical_accuracy_validation=True
)

# Community neural functional registry
from opifex.community import FunctionalRegistry

registry = FunctionalRegistry()
registry.register_functional(
    name="MyNeuralXC",
    functional=my_neural_xc,
    validation_score=0.95,
    chemical_accuracy=0.8  # kcal/mol
)
```

## Quick Start Example

### 1. Define Molecular System

```python
import jax.numpy as jnp
from opifex.core import MolecularSystem

# Water molecule
water = MolecularSystem(
    atoms=jnp.array([
        [0.0000,  0.0000,  0.1173],  # O
        [0.0000,  0.7572, -0.4692],  # H
        [0.0000, -0.7572, -0.4692]   # H
    ]),
    charges=jnp.array([8, 1, 1]),  # O, H, H
    n_electrons=10
)
```

### 2. Create Neural DFT Model

```python
from opifex.neural import NeuralDFT, NeuralXCFunctional

# Neural exchange-correlation functional
neural_xc = NeuralXCFunctional(
    features=128,
    equivariant=True,
    physics_constraints=True
)

# Neural DFT model
neural_dft = NeuralDFT(
    molecular_system=water,
    xc_functional=neural_xc,
    basis_set="cc-pVTZ",
    scf_acceleration=True
)
```

### 3. Train with Chemical Accuracy

```python
from opifex.neural import ChemicalAccuracyTrainer

trainer = ChemicalAccuracyTrainer(
    target_accuracy=1.0,  # kcal/mol
    validation_datasets=["G2", "W4-11"],
    physics_constraints=True
)

# Train the model
trained_model = trainer.train(
    model=neural_dft,
    epochs=1000,
    batch_size=32
)
```

### 4. Validate Chemical Accuracy

```python
from opifex.benchmarking import ChemicalAccuracyValidator

validator = ChemicalAccuracyValidator()
results = validator.validate(
    model=trained_model,
    test_set="G2-1",
    reference="CCSD(T)/CBS"
)

print(f"Mean Absolute Error: {results.mae:.3f} kcal/mol")
print(f"Chemical Accuracy: {results.chemical_accuracy:.1%}")
```

## Features

### Physics-Informed Constraints

```python
from opifex.neural import PhysicsConstraints

constraints = PhysicsConstraints([
    "particle_number_conservation",
    "density_positivity",
    "cusp_conditions",
    "asymptotic_behavior"
])

neural_dft = NeuralDFT(
    molecular_system=system,
    xc_functional=neural_xc,
    physics_constraints=constraints
)
```

### Multi-Fidelity Training

```python
from opifex.neural import MultiFidelityTrainer

trainer = MultiFidelityTrainer(
    low_fidelity="HF/STO-3G",
    medium_fidelity="PBE/cc-pVDZ",
    high_fidelity="CCSD(T)/cc-pVTZ"
)

model = trainer.train(neural_dft, fidelity_schedule="adaptive")
```

### Cross-Paradigm Composition

```python
from opifex.neural import CrossParadigmComposer

# Combine Neural DFT with PINNs for dynamics
composer = CrossParadigmComposer()
dynamics_model = composer.compose([
    ("neural_dft", electronic_structure_model),
    ("pinn", molecular_dynamics_pinn)
])
```

## Benchmarking and Validation

### Chemical Accuracy Benchmarks

```python
from opifex.benchmarking import NeuralDFTBenchmark

benchmark = NeuralDFTBenchmark(
    datasets=["G2", "W4-11", "S22", "A24"],
    reference_methods=["CCSD(T)/CBS", "FCI"],
    accuracy_threshold=1.0  # kcal/mol
)

results = benchmark.evaluate(neural_dft)
benchmark.generate_report(results, format="publication")
```

### Performance Comparison

```python
from opifex.benchmarking import PerformanceComparison

comparison = PerformanceComparison()
comparison.add_method("Neural DFT", neural_dft)
comparison.add_method("PBE", classical_pbe)
comparison.add_method("B3LYP", hybrid_functional)

results = comparison.run(test_molecules=["H2O", "NH3", "CH4"])
comparison.plot_accuracy_vs_speed(results)
```

## Production Deployment

### Container Deployment

```python
from opifex.deployment import ContainerDeploy

deploy = ContainerDeploy(
    model=neural_dft,
    container_type="neural_dft",
    scaling_policy="chemical_accuracy_aware"
)

endpoint = deploy.create_endpoint(
    name="neural-dft-api",
    validation_required=True
)
```

### High-Performance Computing

```python
from opifex.deployment import HPCOptimizer

hpc = HPCOptimizer(
    target_platform="GPU_cluster",
    optimization_level="chemical_accuracy"
)

optimized_model = hpc.optimize(neural_dft)
```

## Educational Resources

### Interactive Examples

- **Hydrogen Molecule**: Basic two-electron system
- **Water Molecule**: Polar molecule with lone pairs
- **Benzene**: Aromatic system with π-electrons
- **Transition Metal Complexes**: d-orbital interactions

### Research Integration

- **Materials Discovery**: High-throughput screening
- **Drug Discovery**: Molecular property prediction
- **Catalysis**: Reaction pathway optimization
- **Energy Storage**: Battery material design

## Best Practices

### Model Design

1. **Start Simple**: Begin with small molecules and basic functionals
2. **Physics First**: Always enforce quantum mechanical constraints
3. **Validate Rigorously**: Use multiple benchmark datasets
4. **Chemical Accuracy**: Target <1 kcal/mol for production use

### Training Strategy

1. **Multi-Fidelity**: Use hierarchical training approach
2. **Transfer Learning**: Leverage pre-trained functionals
3. **Active Learning**: Focus on chemically relevant regions
4. **Uncertainty Quantification**: Include Bayesian components

### Production Deployment

1. **Validation Pipeline**: Automated chemical accuracy checking
2. **Monitoring**: Real-time performance tracking
3. **Fallback**: Classical methods for edge cases
4. **Documentation**: Comprehensive model provenance

## Conclusion

Neural DFT in the Opifex framework represents advanced capabilities in quantum chemistry, providing:

- **Chemical Accuracy**: <1 kcal/mol energy predictions
- **Unified Framework**: Seamless integration with other Opifex paradigms
- **Production Ready**: Enterprise-grade deployment capabilities
- **Community Driven**: Open ecosystem for functional development

The integration of Neural DFT as a first-class paradigm enables applications in materials science, drug discovery, and quantum chemistry research.
