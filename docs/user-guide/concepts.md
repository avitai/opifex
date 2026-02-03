# Core Concepts

## Overview

Opifex provides a unified framework for scientific machine learning, combining traditional numerical methods with modern deep learning approaches. Built on JAX and FLAX NNX, it offers high-performance, differentiable computing for scientific applications with comprehensive physics-informed capabilities.

## Framework Architecture

### JAX Ecosystem Foundation

Opifex is built entirely on the JAX ecosystem for maximum performance and scientific computing capabilities:

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Configure JAX for scientific computing
jax.config.update("jax_enable_x64", True)
print(f"Available devices: {[str(d) for d in jax.devices()]}")
print(f"Backend: {jax.default_backend()}")
print(f"64-bit precision: {jax.config.read('jax_enable_x64')}")
```

**Key Benefits:**

- **Automatic Differentiation**: Forward and reverse mode AD for gradients
- **JIT Compilation**: XLA optimization for high-performance execution
- **Multi-Device Support**: Seamless CPU/GPU/TPU execution
- **Functional Programming**: Pure functions for reproducible computations
- **64-bit Precision**: Scientific accuracy with configurable precision

### FLAX NNX Integration

Modern neural network framework with stateful transforms:

```python
from flax import nnx
from opifex.neural.base import StandardMLP

# Create RNG for reproducible initialization
rngs = nnx.Rngs(jax.random.PRNGKey(42))

# Build neural network with modern FLAX NNX
model = StandardMLP(
    layer_sizes=[2, 64, 64, 1],
    activation="swish",
    use_bias=True,
    rngs=rngs
)

# Forward pass
x = jax.random.normal(jax.random.PRNGKey(0), (32, 2))
output = model(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
```

## Key Components

### 1. Problems (`opifex.core.problems`)

Define scientific problems with comprehensive specification capabilities:

```python
from opifex.core.problems import create_pde_problem

# Define heat equation problem
def heat_equation(x, y, t, u, u_derivatives, params):
    """Heat equation: du/dt - alpha * (d2u/dx2 + d2u/dy2) = 0"""
    alpha = params.get('diffusivity', 0.01)
    u_t = u_derivatives['t']
    u_xx = u_derivatives['xx']
    u_yy = u_derivatives['yy']
    return u_t - alpha * (u_xx + u_yy)

problem = create_pde_problem(
    domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)},
    equation=heat_equation,
    boundary_conditions=[
        {"type": "dirichlet", "boundary": "left", "value": 0.0},
        {"type": "dirichlet", "boundary": "right", "value": 1.0}
    ],
    initial_conditions={"u": lambda x, y: 0.5 * (x + y)},
    parameters={"diffusivity": 0.01}
)
```

### 2. Neural Networks (`opifex.neural`)

Specialized architectures for scientific computing:

**Available Architectures:**

- **StandardMLP**: Multi-layer perceptrons with scientific activations
- **QuantumMLP**: Quantum-aware networks for molecular systems
- **FourierNeuralOperator**: Learn mappings between function spaces
- **DeepONet**: Deep operator networks for operator learning
- **PhysicsInformedOperator**: Physics-aware neural operators

```python
from opifex.neural.operators import FourierNeuralOperator

# Create Fourier Neural Operator
fno = FourierNeuralOperator(
    in_channels=2,
    out_channels=1,
    hidden_channels=64,
    modes=16,
    num_layers=4,
    rngs=rngs
)

# Process spatial data
spatial_data = jax.random.normal(jax.random.PRNGKey(1), (8, 64, 64, 2))
operator_output = fno(spatial_data)
print(f"Operator: {spatial_data.shape} -> {operator_output.shape}")
```

### 3. Training (`opifex.training`)

Physics-aware training procedures with advanced optimization:

```python
from opifex.training.basic_trainer import ModularTrainer
from opifex.core.training.config import TrainingConfig

# Configure comprehensive training
config = TrainingConfig(
    num_epochs=5000,
    batch_size=128,
    learning_rate=1e-3,
    validation_frequency=100,
    checkpoint_frequency=500
)

# Create modular trainer with error recovery
trainer = ModularTrainer(
    model=model,
    config=config,
    rngs=rngs
)

# Train with automatic error recovery and optimization
trained_model, history = trainer.train(
    train_data=(x_train, y_train),
    val_data=(x_val, y_val)
)
```

### 4. Geometry (`opifex.geometry`)

Comprehensive geometric modeling with CSG operations:

```python
from opifex.geometry import Rectangle, Circle, union, intersection
from opifex.geometry.manifolds import SphericalManifold

# Create 2D shapes
rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=1.5)
circle = Circle(center=jnp.array([1.0, 0.5]), radius=0.8)

# CSG operations
combined = union(rect, circle)
overlap = intersection(rect, circle)

# Sample boundary points
key = jax.random.PRNGKey(42)
boundary_points = rect.sample_boundary(n_points=100, key=key)

# Work with manifolds
sphere = SphericalManifold(dimension=2)
manifold_points = sphere.sample_points(n_points=50, key=key)
```

## Scientific ML Paradigms

### Physics-Informed Neural Networks (PINNs)

Neural networks that incorporate physical laws as soft constraints during training:

```python
from opifex.neural.pinns import MultiScalePINN
from opifex.core.physics.losses import PhysicsInformedLoss, PhysicsLossConfig

# Create multi-scale PINN
pinn = MultiScalePINN(
    layers=[50, 50, 50, 1],
    activation='tanh',
    physics_loss_weight=1.0,
    rngs=rngs
)

# Configure physics-informed loss
physics_config = PhysicsLossConfig(
    pde_weight=1.0,
    boundary_weight=10.0,
    initial_weight=1.0
)
physics_loss = PhysicsInformedLoss(config=physics_config)
```

**Key Features:**

- **Residual Computation**: Automatic PDE residual calculation
- **Boundary Enforcement**: Strong and weak boundary condition enforcement
- **Multi-Scale Training**: Handle problems across different scales
- **Adaptive Weighting**: Dynamic loss weight adjustment

### Neural Operators

Learn mappings between function spaces, enabling generalization across different problem parameters:

```python
from opifex.neural.operators import (
    FourierNeuralOperator,
    DeepONet,
    AdaptiveDeepONet,
    OperatorNetwork
)

# Fourier Neural Operator for PDEs
fno = FourierNeuralOperator(
    in_channels=2, out_channels=1,
    hidden_channels=64, modes=16,
    rngs=rngs
)

# Deep Operator Network
deeponet = DeepONet(
    branch_layers=[100, 128, 128],
    trunk_layers=[2, 128, 128],
    output_dim=1,
    rngs=rngs
)

# Unified operator interface
operator = OperatorNetwork(
    operator_type="fno",
    config={
        "in_channels": 2,
        "out_channels": 1,
        "hidden_channels": 64,
        "modes": 16
    },
    rngs=rngs
)
```

**Operator Types Available:**

- **FNO**: Fourier Neural Operators with spectral convolutions
- **DeepONet**: Deep operator networks with branch-trunk architecture
- **U-NO**: U-Net style neural operators
- **GINO**: Graph-informed neural operators
- **DISCO**: Discrete-continuous convolutions

### Neural Density Functional Theory

Quantum mechanical calculations using neural networks:

```python
from opifex.neural.base import QuantumMLP

# Quantum-aware neural network
quantum_net = QuantumMLP(
    layer_sizes=[3, 128, 128, 1],  # 3D coordinates -> energy
    activation="swish",
    enable_symmetry=True,
    precision="float64",  # Chemical accuracy
    rngs=rngs
)

# Molecular energy calculation
coordinates = jax.random.normal(jax.random.PRNGKey(2), (10, 3))  # 10 atoms
energy = quantum_net(coordinates)
print(f"Molecular energy: {energy}")
```

### Probabilistic Numerics

Uncertainty quantification in scientific computations:

```python
from opifex.neural.bayesian import UncertaintyQuantifier, VariationalConfig

# Bayesian neural network
bnn = UncertaintyQuantifier(
    layers=[32, 32, 1],
    variational_config=VariationalConfig(
        prior_std=0.1,
        likelihood_std=0.05,
        method="mean_field"
    ),
    rngs=rngs
)

# Prediction with uncertainty
x_test = jax.random.normal(jax.random.PRNGKey(3), (100, 2))
mean, std = bnn.predict_with_uncertainty(x_test, n_samples=100)
print(f"Prediction uncertainty: mean±std = {jnp.mean(mean):.3f}±{jnp.mean(std):.3f}")
```

## Advanced Features

### Multi-Device Support

Seamless scaling across hardware:

```python
# Check available devices
devices = jax.devices()
print(f"Available devices: {[str(d) for d in devices]}")

# Automatic device placement
if len(jax.devices('gpu')) > 0:
    print("🎮 GPU acceleration enabled")
else:
    print("💻 Running on CPU")
```

### Checkpointing and Persistence

Robust model saving and loading:

```python
from opifex.training.basic_trainer import TrainingConfig

config = TrainingConfig(
    checkpoint_frequency=100,
    checkpoint_config={
        "save_directory": "./checkpoints",
        "max_to_keep": 5,
        "save_best_only": True
    }
)
```

### Performance Optimization

Built-in performance monitoring and optimization:

```python
# JIT compilation for performance
@jax.jit
def optimized_forward(model, x):
    return model(x)

# Vectorized operations
batch_output = jax.vmap(optimized_forward, in_axes=(None, 0))(model, batch_data)
```

## Design Principles

### 1. Composability

Modular components that can be combined flexibly:

```python
# Compose different components
from opifex.training.basic_trainer import ModularTrainer
from opifex.training.recovery import ErrorRecoveryManager

trainer = ModularTrainer(
    model=model,
    config=config,
    components={
        "error_recovery": ErrorRecoveryManager(config={}),
        "custom_component": CustomTrainingComponent()
    }
)
```

### 2. Performance

Optimized for scientific computing workloads:

- JAX transformations (jit, vmap, pmap)
- XLA compilation for optimal performance
- Memory-efficient implementations
- GPU/TPU acceleration

### 3. Extensibility

Easy to add new methods and approaches:

- Protocol-based interfaces
- Modular architecture
- Plugin system for custom components
- Clear extension points

### 4. Reproducibility

Deterministic computations with proper seeding:

```python
# Reproducible random number generation
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(key)

# Deterministic model initialization
model = StandardMLP(layer_sizes=[2, 64, 1], rngs=rngs)

# Reproducible training
trainer = ModularTrainer(model=model, config=config, rngs=rngs)
```

## Getting Started

### Quick Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
from opifex.neural.base import StandardMLP
from opifex.training.basic_trainer import ModularTrainer
from opifex.core.training.config import TrainingConfig

# 1. Setup
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(key)

# 2. Create model
model = StandardMLP(layer_sizes=[2, 64, 64, 1], activation="swish", rngs=rngs)

# 3. Generate data
x = jax.random.uniform(key, (1000, 2), minval=-2, maxval=2)
y = jnp.sin(jnp.pi * x[:, 0]) * jnp.cos(jnp.pi * x[:, 1])

# 4. Configure training
config = TrainingConfig(num_epochs=1000, learning_rate=1e-3)

# 5. Train
trainer = ModularTrainer(model=model, config=config, rngs=rngs)
trained_model, history = trainer.train(train_data=(x, y))

print("✅ Opifex training complete!")
```

This complete framework enables researchers and practitioners to tackle complex scientific machine learning problems with advanced methods and high-performance computing capabilities.
