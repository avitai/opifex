# Neural Operators

## Overview

Neural operators represent an advanced paradigm in scientific machine learning that learns mappings between function spaces rather than finite-dimensional vectors. Unlike traditional neural networks that map between fixed-size inputs and outputs, neural operators can generalize across different discretizations, resolutions, and problem parameters, making them ideal for solving families of partial differential equations (PDEs) and other function-to-function mappings.

The Opifex neural operators framework provides thorough implementations of Fourier Neural Operators (FNO), DeepONet, Graph Neural Operators (GNO), and other advanced architectures, enabling efficient solution of complex scientific computing problems with excellent generalization capabilities.

## Theoretical Foundation

### Function Space Learning

Traditional neural networks learn mappings $f: \mathbb{R}^n \to \mathbb{R}^m$ between finite-dimensional spaces. Neural operators learn mappings between infinite-dimensional function spaces:

$$\mathcal{G}: \mathcal{A} \to \mathcal{U}$$

where $\mathcal{A}$ and $\mathcal{U}$ are function spaces. For example, in PDE solving:

- $\mathcal{A}$: space of input functions (initial conditions, boundary conditions, coefficients)
- $\mathcal{U}$: space of solution functions

### Universal Approximation for Operators

Neural operators satisfy universal approximation theorems for operators, meaning they can approximate any continuous operator between function spaces to arbitrary accuracy given sufficient capacity.

### Discretization Invariance

A key advantage of neural operators is discretization invariance: once trained, they can evaluate functions at any resolution without retraining, enabling:

- **Super-resolution**: Evaluate at higher resolution than training data
- **Multi-resolution**: Handle varying discretizations in the same model
- **Mesh-free evaluation**: Evaluate at arbitrary points in the domain

## Core Neural Operator Architectures

### 1. Fourier Neural Operators (FNO)

FNO leverages the Fourier transform to capture global dependencies efficiently:

```python
from opifex.neural.operators import FourierNeuralOperator
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from opifex.core.training.trainer import Trainer
from opifex.core.training.config import TrainingConfig

# Create 2D FNO for PDEs
fno_2d = FourierNeuralOperator(
    in_channels=1,   # Input field dimension
    out_channels=1,  # Solution field dimension
    hidden_channels=64,
    modes=16,        # Fourier modes
    num_layers=4,
    activation=nnx.gelu,
    rngs=nnx.Rngs(42)
)

# Example: Darcy flow problem
# Input: permeability field a(x,y)
# Output: pressure field u(x,y) solving -∇·(a∇u) = f

# Generate synthetic training data (Self-contained example)
def generate_dummy_darcy_data(n_samples=100, resolution=64):
    """Generate synthetic Darcy flow data for demonstration."""
    key = jax.random.PRNGKey(42)

    # Generate random input fields (permeability)
    key1, key2 = jax.random.split(key)
    inputs = jax.random.normal(key1, (n_samples, resolution, resolution, 1))

    # Generate corresponding output fields (pressure) - dummy mapping
    # In a real scenario, this would be the solution from a numerical solver
    outputs = jnp.sin(inputs * jnp.pi) + 0.1 * jax.random.normal(key2, (n_samples, resolution, resolution, 1))

    return inputs, outputs

# Training data
train_inputs, train_outputs = generate_dummy_darcy_data(n_samples=100)
val_inputs, val_outputs = generate_dummy_darcy_data(n_samples=20)

# Train FNO
training_config = TrainingConfig(
    num_epochs=10,  # Reduced for demonstration
    batch_size=10,
    learning_rate=1e-3,
)

trainer = Trainer(model=fno_2d, config=training_config)
trained_fno, history = trainer.fit(
    train_data=(train_inputs, train_outputs),
    val_data=(val_inputs, val_outputs)
)

print(f"FNO training completed. Final validation loss: {history['final_val_loss']:.6f}")

# Test prediction
test_input = val_inputs[0:1]
prediction = trained_fno(test_input)
print(f"Prediction shape: {prediction.shape}")
```

### 2. DeepONet (Deep Operator Networks)

DeepONet uses a branch-trunk architecture to learn operators:

```python
from opifex.neural.operators import DeepONet

# Create DeepONet
deeponet = DeepONet(
    branch_sizes=[100, 128, 128, 128],  # [sensors, hidden..., output]
    trunk_sizes=[2, 128, 128, 128],     # [coords, hidden..., output]
    activation="relu",
    use_bias=True,
    rngs=nnx.Rngs(42)
)

# Example: Antiderivative operator
# Input function: f(x)
# Output function: F(x) = ∫₀ˣ f(s) ds

def generate_antiderivative_data(n_samples=1000):
    """Generate antiderivative training data."""
    key = jax.random.PRNGKey(42)

    # Generate random input functions
    input_functions = []
    output_functions = []

    for i in range(n_samples):
        # Random polynomial coefficients
        key, subkey = jax.random.split(key)
        coeffs = jax.random.normal(subkey, (5,))

        # Input function: polynomial
        def input_fn(x):
            return jnp.polyval(coeffs, x)

        # Output function: antiderivative
        antiderivative_coeffs = jnp.concatenate([coeffs / jnp.arange(1, 6), jnp.array([0])])
        def output_fn(x):
            return jnp.polyval(antiderivative_coeffs[::-1], x)

        # Sample functions at sensor/query locations
        x_sensors = jnp.linspace(0, 1, 100)
        input_samples = jax.vmap(input_fn)(x_sensors)

        x_query = jax.random.uniform(subkey, (50,), minval=0, maxval=1)
        output_samples = jax.vmap(output_fn)(x_query)

        input_functions.append((input_samples, x_query))
        output_functions.append(output_samples)

    return input_functions, output_functions

# Train DeepONet
antiderivative_inputs, antiderivative_outputs = generate_antiderivative_data()

deeponet_trainer = Trainer(model=deeponet, config=training_config)
trained_deeponet, deeponet_history = deeponet_trainer.fit(
    train_data=(antiderivative_inputs, antiderivative_outputs)
)

print(f"DeepONet training completed. Final loss: {deeponet_history['final_train_loss']:.6f}")
```

### 3. Graph Neural Operators (GNO)

GNO handles irregular geometries and unstructured meshes:

```python
from opifex.neural.operators import GraphNeuralOperator

# Create GNO
gno = GraphNeuralOperator(
    node_dim=3,      # Node features (x, y, boundary_flag)
    hidden_dim=64,   # Hidden dimension
    num_layers=6,    # Number of message passing layers
    edge_dim=2,      # Edge features (distance, angle)
    activation=nnx.gelu,
    rngs=nnx.Rngs(42)
)

# Example: Heat equation on irregular domains
def generate_irregular_mesh_data(n_samples=500):
    """Generate heat equation data on irregular meshes."""
    meshes = []
    solutions = []

    for i in range(n_samples):
        # Generate random irregular domain
        domain = generate_random_polygon(num_vertices=jax.random.randint(key, (), 6, 12))

        # Create unstructured mesh
        mesh = create_unstructured_mesh(domain, max_area=0.01)

        # Random boundary conditions and material properties
        boundary_temp = jax.random.uniform(key, (), minval=0, maxval=100)
        thermal_conductivity = jax.random.uniform(key, (), minval=0.1, maxval=2.0)

        # Solve heat equation
        solution = solve_heat_equation_fem(
            mesh=mesh,
            boundary_conditions={"dirichlet": boundary_temp},
            thermal_conductivity=thermal_conductivity
        )

        meshes.append(mesh)
        solutions.append(solution)

    return meshes, solutions

# Train GNO on irregular meshes
mesh_data, solution_data = generate_irregular_mesh_data()

gno_trainer = Trainer(model=gno, config=training_config)
trained_gno, gno_history = gno_trainer.fit(
    train_data=(mesh_data, solution_data)
)

print(f"GNO training completed on irregular meshes")
```


## Integration with Opifex Ecosystem

### 1. Physics-Informed Integration

Combine neural operators with physics-informed training:

```python
from opifex.core.physics.losses import PhysicsInformedLoss, PhysicsLossConfig

# Configure physics-informed loss
physics_config = PhysicsLossConfig(
    physics_loss_weight=0.1,
    boundary_loss_weight=10.0,
)

physics_loss = PhysicsInformedLoss(
    config=physics_config,
    equation_type="navier_stokes",
    domain_type="rectangular",
)
```

### 2. Optimization Integration

Use advanced optimization for neural operator training:

```python
from opifex.core.training.config import MetaOptimizerConfig
from opifex.optimization.meta_optimization import MetaOptimizer

# Meta-optimizer for neural operators
meta_config = MetaOptimizerConfig(
    meta_algorithm="l2o",
    base_optimizer="adam",
    meta_learning_rate=1e-3,
    adaptation_steps=10,
)

meta_optimizer = MetaOptimizer(config=meta_config, rngs=nnx.Rngs(42))

# Use step-by-step optimization
opt_state = meta_optimizer.init_optimizer_state(params)
for step_idx in range(100):
    params, opt_state, meta_info = meta_optimizer.step(
        loss_fn=loss_function,
        params=params,
        opt_state=opt_state,
        step=step_idx,
    )

print(f"Meta-optimization for neural operators completed")
```

## Best Practices

### 1. Architecture Selection

Guidelines for choosing the right neural operator:

- **Regular grids**: Use FNO for efficiency and global receptive field
- **Irregular meshes**: Use GNO for flexibility with unstructured data
- **Function-to-function**: Use DeepONet for explicit function space mapping
- **Multi-physics**: Use Multi-Physics operators with coupling
- **Long-range interactions**: Use Attention-based operators
- **Multi-scale**: Use Hierarchical operators

### 2. Training Strategies

- **Data preparation**: Normalize inputs/outputs, use diverse parameter ranges
- **Training strategy**: Start coarse resolution, progressively increase
- **Hyperparameter tuning**: Use Bayesian optimization for search
- **Validation**: Test on out-of-distribution parameters, validate super-resolution

### 3. Performance Optimization

- **Memory efficiency**: Use gradient checkpointing, parameter factorization
- **Computational efficiency**: Apply JIT compilation, GPU acceleration
- **Distributed training**: Scale to large datasets with data parallelism

## Future Directions

### 1. Emerging Architectures

- **Quantum Neural Operators**: Leverage quantum computing for exponential speedups
- **Neuromorphic Operators**: Deploy on neuromorphic hardware for energy efficiency
- **Hybrid Symbolic-Neural**: Combine symbolic reasoning with neural learning
- **Causal Neural Operators**: Enforce causality for time-dependent problems

### 2. Advanced Applications

- **Multi-Scale Materials**: From atoms to continuum
- **Biological Systems**: Protein folding, drug discovery
- **Financial Modeling**: Risk assessment, portfolio optimization
- **Autonomous Systems**: Real-time control and planning

## References

1. Li, Z., et al. "Fourier Neural Operator for Parametric Partial Differential Equations." ICLR 2021.
2. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators." Nature Machine Intelligence 3, 218-229 (2021).
3. Li, Z., et al. "Neural Operator: Graph Kernel Network for Partial Differential Equations." ICLR 2020 Workshop.
4. Kovachki, N., et al. "Neural operator: Learning maps between function spaces with applications to PDEs." Journal of Machine Learning Research 24, 1-97 (2023).
5. Cao, S. "Choose a Transformer: Fourier or Galerkin." NeurIPS 2021.

## See Also

- [Neural Networks Guide](../user-guide/neural-networks.md) - Neural network architectures
- [Training Guide](../user-guide/training.md) - Training strategies and optimization
- [Physics-Informed Methods](pinns.md) - Physics-informed neural networks
- [API Reference](../api/neural.md) - Complete neural operators API documentation
