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
    scheduler="cosine_annealing",
    early_stopping_patience=5
)

trainer = Trainer(model=fno_2d, config=training_config)
trained_fno, history = trainer.train(
    train_data=(train_inputs, train_outputs),
    val_data=(val_inputs, val_outputs)
)

print(f"FNO training completed. Final validation loss: {history.val_loss[-1]:.6f}")

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
trained_deeponet, deeponet_history = deeponet_trainer.train(
    train_data=(antiderivative_inputs, antiderivative_outputs)
)

print(f"DeepONet training completed. Final loss: {deeponet_history.train_loss[-1]:.6f}")
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
trained_gno, gno_history = gno_trainer.train(
    train_data=(mesh_data, solution_data)
)

print(f"GNO training completed on irregular meshes")
```


## Scientific Applications

### 1. Computational Fluid Dynamics

Neural operators for fluid flow problems:

```python
from opifex.neural.operators import FluidDynamicsOperator

# Navier-Stokes operator
ns_config = {
    "reynolds_number_range": [100, 10000],
    "geometry_types": ["cylinder", "airfoil", "backward_step"],
    "boundary_conditions": ["no_slip", "slip", "periodic"],
    "compressibility": "incompressible"
}

ns_operator = FluidDynamicsOperator(
    base_operator=fno_2d,
    config=ns_config,
    rngs=nnx.Rngs(42)
)

# Generate CFD training data
def generate_cfd_data(n_samples=1000):
    """Generate CFD training data with varying Reynolds numbers and geometries."""
    geometries = []
    flow_fields = []

    for i in range(n_samples):
        # Random geometry
        if i % 3 == 0:
            geometry = generate_cylinder_geometry(radius=jax.random.uniform(key, (), 0.1, 0.3))
        elif i % 3 == 1:
            geometry = generate_airfoil_geometry(angle_of_attack=jax.random.uniform(key, (), -10, 10))
        else:
            geometry = generate_backward_step_geometry(step_height=jax.random.uniform(key, (), 0.1, 0.5))

        # Random Reynolds number
        reynolds = jax.random.uniform(key, (), 100, 10000)

        # Solve Navier-Stokes
        velocity, pressure = solve_navier_stokes(
            geometry=geometry,
            reynolds_number=reynolds,
            inlet_velocity=1.0
        )

        geometries.append(geometry)
        flow_fields.append(jnp.stack([velocity[..., 0], velocity[..., 1], pressure], axis=-1))

    return geometries, flow_fields

# Train CFD operator
cfd_inputs, cfd_outputs = generate_cfd_data()
cfd_trainer = Trainer(model=ns_operator, config=training_config)
trained_cfd_operator, cfd_history = cfd_trainer.train(
    train_data=(cfd_inputs, cfd_outputs)
)

print(f"CFD operator training completed")

# Test on new geometry
new_geometry = generate_custom_geometry()
predicted_flow = trained_cfd_operator(new_geometry[None, ...])[0]
print(f"Flow prediction shape: {predicted_flow.shape}")
```

### 2. Climate and Weather Modeling

Large-scale atmospheric and oceanic modeling:

```python
from opifex.neural.operators import ClimateOperator, AtmosphericModel

# Configure climate operator
climate_config = {
    "variables": ["temperature", "pressure", "humidity", "wind_u", "wind_v"],
    "vertical_levels": 20,
    "time_step_hours": 6,
    "spatial_resolution": "1deg",
    "physics_parameterizations": ["convection", "radiation", "boundary_layer"]
}

climate_operator = ClimateOperator(
    base_operator=attention_operator,  # Good for global patterns
    config=climate_config,
    rngs=nnx.Rngs(42)
)

# Generate climate training data
def generate_climate_data(n_samples=500):
    """Generate climate reanalysis training data."""
    atmospheric_states = []
    future_states = []

    # Load historical reanalysis data
    reanalysis_data = load_era5_data(years=range(1980, 2020))

    for i in range(n_samples):
        # Random time window
        start_time = jax.random.randint(key, (), 0, len(reanalysis_data) - 24)

        # Current atmospheric state
        current_state = reanalysis_data[start_time]

        # Future state (24 hours later)
        future_state = reanalysis_data[start_time + 4]  # 4 * 6 hours = 24 hours

        atmospheric_states.append(current_state)
        future_states.append(future_state)

    return atmospheric_states, future_states

# Train climate operator
climate_inputs, climate_outputs = generate_climate_data()
climate_trainer = Trainer(model=climate_operator, config=training_config)
trained_climate_operator, climate_history = climate_trainer.train(
    train_data=(climate_inputs, climate_outputs)
)

print(f"Climate operator training completed")

# Weather forecasting
current_weather = get_current_atmospheric_state()
weather_forecast = trained_climate_operator(current_weather[None, ...])[0]
print(f"24-hour weather forecast generated")
```

## Integration with Opifex Ecosystem

### 1. Physics-Informed Integration

Combine neural operators with physics-informed training:

```python
from opifex.core.physics.losses import PhysicsInformedLoss
from opifex.neural.operators import PhysicsInformedOperatorTraining

# Define PDE residual for Navier-Stokes
def navier_stokes_residual(u, v, p, x, y, t, reynolds):
    """Compute Navier-Stokes residual."""
    # Velocity derivatives
    u_t = jax.grad(u, argnums=2)(x, y, t)
    u_x = jax.grad(u, argnums=0)(x, y, t)
    u_y = jax.grad(u, argnums=1)(x, y, t)
    u_xx = jax.grad(jax.grad(u, argnums=0), argnums=0)(x, y, t)
    u_yy = jax.grad(jax.grad(u, argnums=1), argnums=1)(x, y, t)

    v_t = jax.grad(v, argnums=2)(x, y, t)
    v_x = jax.grad(v, argnums=0)(x, y, t)
    v_y = jax.grad(v, argnums=1)(x, y, t)
    v_xx = jax.grad(jax.grad(v, argnums=0), argnums=0)(x, y, t)
    v_yy = jax.grad(jax.grad(v, argnums=1), argnums=1)(x, y, t)

    # Pressure derivatives
    p_x = jax.grad(p, argnums=0)(x, y, t)
    p_y = jax.grad(p, argnums=1)(x, y, t)

    # Navier-Stokes equations
    continuity = u_x + v_y
    momentum_x = u_t + u * u_x + v * u_y + p_x - (1/reynolds) * (u_xx + u_yy)
    momentum_y = v_t + u * v_x + v * v_y + p_y - (1/reynolds) * (v_xx + v_yy)

    return continuity, momentum_x, momentum_y

# Physics-informed operator training
pi_operator_trainer = PhysicsInformedOperatorTraining(
    model=ns_operator,
    pde_residual_fn=navier_stokes_residual,
    physics_weight=0.1,
    boundary_weight=10.0
)

# Train with physics constraints
pi_training_result = pi_operator_trainer.train(
    data_points=cfd_data,
    physics_points=physics_collocation_points,
    boundary_points=boundary_points,
    num_epochs=500
)

print(f"Physics-informed operator training completed")
print(f"PDE residual: {pi_training_result.final_pde_residual:.6f}")
```

### 2. Optimization Integration

Use advanced optimization for neural operator training:

```python
from opifex.optimization.meta_optimizers import MetaOptimizer

# Meta-optimizer for neural operators
meta_config = MetaOptimizerConfig(
    operator_aware=True,
    spectral_regularization=True,
    multi_scale_optimization=True,
    meta_learning_rate=1e-3
)

meta_optimizer = MetaOptimizer(config=meta_config, rngs=nnx.Rngs(42))

# Optimize neural operator training
optimized_operator = meta_optimizer.optimize_operator(
    operator=fno_2d,
    training_data=operator_training_data,
    validation_data=operator_validation_data,
    num_meta_epochs=100
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
