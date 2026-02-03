# Neural Networks

## Overview

Opifex provides a comprehensive collection of specialized neural network architectures designed for scientific computing applications. Built with FLAX NNX, all networks support automatic differentiation, JIT compilation, and multi-device execution for high-performance scientific machine learning.

## Core Neural Network Architectures

### Standard Multi-Layer Perceptrons

The foundation of scientific neural networks with enhanced capabilities:

```python
import jax
import jax.numpy as jnp
from flax import nnx
from opifex.neural.base import StandardMLP, QuantumMLP

# Create RNG for reproducible initialization
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(key)

# Standard MLP for general scientific computing
model = StandardMLP(
    layer_sizes=[2, 64, 64, 1],
    activation="swish",
    use_bias=True,
    dropout_rate=0.1,
    rngs=rngs
)

# Test forward pass
x = jax.random.normal(jax.random.PRNGKey(0), (32, 2))
output = model(x)
print(f"Standard MLP: {x.shape} -> {output.shape}")

# Quantum-aware MLP for molecular systems
quantum_model = QuantumMLP(
    layer_sizes=[3, 128, 128, 1],  # 3D coordinates -> energy
    activation="swish",
    enforce_symmetry=True,
    precision="float64",  # Chemical accuracy
    rngs=rngs
)

# Molecular coordinates (10 atoms in 3D)
coords = jax.random.normal(jax.random.PRNGKey(1), (10, 3))
energy = quantum_model(coords)
print(f"Molecular energy: {energy}")
```

**Available Activations (27 functions):**

```python
from opifex.neural.activations import get_activation, list_activations

# List all available activations
print("Available activations:", list_activations())

# Get specific activation function
swish = get_activation("swish")
gelu = get_activation("gelu")
tanh = get_activation("tanh")

# Scientific activations
sin = get_activation("sin")
gaussian = get_activation("gaussian")
```

## Neural Operators

### Fourier Neural Operators (FNO)

Learn mappings between function spaces using spectral methods:

```python
from opifex.neural.operators import FourierNeuralOperator, FourierLayer

# Standard FNO for PDE operator learning
fno = FourierNeuralOperator(
    in_channels=2,      # Input function channels
    out_channels=1,     # Output function channels
    hidden_channels=64, # Hidden dimension
    modes=16,          # Fourier modes to keep
    num_layers=4,      # Number of Fourier layers
    rngs=rngs
)

# Process 2D spatial data (batch, height, width, channels)
spatial_data = jax.random.normal(jax.random.PRNGKey(2), (8, 64, 64, 2))
fno_output = fno(spatial_data)
print(f"FNO: {spatial_data.shape} -> {fno_output.shape}")

# Individual Fourier layer for custom architectures
fourier_layer = FourierLayer(
    in_channels=32,
    out_channels=32,
    modes=12,
    rngs=rngs
)
```

### Deep Operator Networks (DeepONet)

Branch-trunk architecture for operator learning:

```python
from opifex.neural.operators import DeepONet, AdaptiveDeepONet, FourierEnhancedDeepONet

# Standard DeepONet
deeponet = DeepONet(
    branch_layers=[100, 128, 128],  # Branch network (input functions)
    trunk_layers=[2, 128, 128],     # Trunk network (query points)
    output_dim=1,                   # Output dimension
    rngs=rngs
)

# Test with function data and query points
function_data = jax.random.normal(jax.random.PRNGKey(3), (32, 100))  # 32 functions, 100 points each
query_points = jax.random.uniform(jax.random.PRNGKey(4), (32, 50, 2))  # 32 batches, 50 queries, 2D points

deeponet_output = deeponet(function_data, query_points)
print(f"DeepONet: functions {function_data.shape} + queries {query_points.shape} -> {deeponet_output.shape}")

# Adaptive DeepONet with dynamic architecture
adaptive_deeponet = AdaptiveDeepONet(
    base_branch_layers=[50, 64],
    base_trunk_layers=[2, 64],
    adaptation_layers=[32, 16],
    output_dim=1,
    rngs=rngs
)

# Fourier-enhanced DeepONet
fourier_deeponet = FourierEnhancedDeepONet(
    branch_layers=[100, 128],
    trunk_layers=[2, 128],
    fourier_modes=8,
    output_dim=1,
    rngs=rngs
)
```

### Specialized Neural Operators

Advanced operator architectures for specific applications:

```python
from opifex.neural.operators.specialized import (
    OperatorNetwork,
    UNeuralOperator,
    WaveletNeuralOperator,
    LatentNeuralOperator
)

# Unified operator interface
operator = OperatorNetwork(
    operator_type="fno",
    config={
        "in_channels": 2,
        "out_channels": 1,
        "hidden_channels": 64,
        "modes": 16,
        "activation": "gelu"
    },
    rngs=rngs
)

# U-Net style neural operator
uno = UNeuralOperator(
    in_channels=3,
    out_channels=1,
    hidden_channels=32,
    num_layers=4,
    rngs=rngs
)

# Wavelet-based neural operator
wavelet_no = WaveletNeuralOperator(
    in_channels=2,
    out_channels=1,
    wavelet_type="db4",
    levels=3,
    rngs=rngs
)

# Latent space neural operator
latent_no = LatentNeuralOperator(
    input_dim=64,
    latent_dim=16,
    output_dim=1,
    encoder_layers=[64, 32, 16],
    decoder_layers=[16, 32, 64],
    rngs=rngs
)
```

### DISCO Convolutions

Advanced discrete-continuous convolutions for irregular data:

```python
from opifex.neural.operators.specialized import (
    DiscreteContinuousConv2d,
    EquidistantDiscreteContinuousConv2d,
    create_disco_encoder,
    create_disco_decoder
)

# DISCO convolution for irregular grids
disco_conv = DiscreteContinuousConv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=5,
    activation=nnx.gelu,
    rngs=rngs
)

# Optimized DISCO for regular grids (10x+ speedup)
equi_disco = EquidistantDiscreteContinuousConv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=5,
    grid_spacing=0.1,
    rngs=rngs
)

# Test with 2D spatial data
x = jax.random.normal(jax.random.PRNGKey(5), (8, 64, 64, 3))
disco_output = disco_conv(x)
equi_output = equi_disco(x)

print(f"DISCO: {x.shape} -> {disco_output.shape}")
print(f"Equidistant DISCO: {x.shape} -> {equi_output.shape}")

# Encoder-decoder with DISCO
encoder = create_disco_encoder(
    in_channels=3,
    hidden_channels=[32, 64, 128],
    rngs=rngs
)

decoder = create_disco_decoder(
    in_channels=128,
    hidden_channels=[64, 32],
    out_channels=1,
    rngs=rngs
)
```

## Physics-Informed Neural Networks (PINNs)

### Multi-Scale PINNs

Neural networks that incorporate physical laws as constraints:

```python
from opifex.neural.pinns import MultiScalePINN, create_heat_equation_pinn

# Multi-scale PINN for complex PDEs
pinn = MultiScalePINN(
    layers=[50, 50, 50, 1],
    activation='tanh',
    physics_loss_weight=1.0,
    scales=[1, 2, 4],  # Multiple scales
    rngs=rngs
)

# Specialized PINN constructors
heat_pinn = create_heat_equation_pinn(
    layers=[50, 50, 50, 1],
    domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)},
    diffusivity=0.01,
    rngs=rngs
)

# Test PINN with spatiotemporal data
x = jax.random.uniform(jax.random.PRNGKey(6), (100, 3))  # (x, y, t)
pinn_output = pinn(x)
print(f"PINN: {x.shape} -> {pinn_output.shape}")
```

### Physics-Aware Components

```python
from opifex.neural.operators.physics import (
    PhysicsInformedOperator,
    PhysicsAwareAttention,
    PhysicsCrossAttention
)

# Physics-informed neural operator
physics_operator = PhysicsInformedOperator(
    base_operator="fno",
    physics_constraints=["conservation", "symmetry"],
    constraint_weights={"conservation": 1.0, "symmetry": 0.5},
    rngs=rngs
)

# Physics-aware attention mechanism
physics_attention = PhysicsAwareAttention(
    embed_dim=64,
    num_heads=8,
    physics_bias=True,
    rngs=rngs
)

# Cross-attention with physics constraints
cross_attention = PhysicsCrossAttention(
    query_dim=64,
    key_dim=64,
    value_dim=64,
    num_heads=4,
    rngs=rngs
)
```

## Graph Neural Networks

### Graph Neural Operators

For irregular domains and network structures:

```python
from opifex.neural.operators.graph import GraphNeuralOperator, MessagePassingLayer
from opifex.geometry.topology import GraphTopology

# Create graph topology
num_nodes = 100
edges = jax.random.randint(jax.random.PRNGKey(7), (200, 2), 0, num_nodes)
node_features = jax.random.normal(jax.random.PRNGKey(8), (num_nodes, 16))
edge_features = jax.random.normal(jax.random.PRNGKey(9), (200, 8))

# Graph neural operator
gno = GraphNeuralOperator(
    node_features=16,
    edge_features=8,
    hidden_dim=32,
    output_dim=1,
    num_layers=3,
    rngs=rngs
)

# Message passing layer
mp_layer = MessagePassingLayer(
    node_dim=16,
    edge_dim=8,
    message_dim=32,
    rngs=rngs
)

# Process graph data
graph_output = gno(node_features, edges, edge_features)
print(f"Graph Neural Operator: nodes {node_features.shape} -> {graph_output.shape}")
```

## Bayesian Neural Networks

### Uncertainty Quantification

```python
from opifex.neural.bayesian import UncertaintyQuantifier, VariationalConfig

# Bayesian neural network configuration
variational_config = VariationalConfig(
    prior_std=0.1,
    likelihood_std=0.05,
    method="mean_field",
    kl_weight=1e-3
)

# Uncertainty quantifier
bnn = UncertaintyQuantifier(
    layers=[32, 32, 1],
    variational_config=variational_config,
    rngs=rngs
)

# Prediction with uncertainty
x_test = jax.random.normal(jax.random.PRNGKey(10), (100, 2))
mean, std = bnn.predict_with_uncertainty(x_test, n_samples=100)

print(f"Bayesian prediction: mean shape {mean.shape}, std shape {std.shape}")
print(f"Average uncertainty: {jnp.mean(std):.4f}")
```

## Custom Architecture Development

### Building Custom Networks

```python
import flax.nnx as nnx

class PhysicsInformedMLP(nnx.Module):
    """Custom physics-informed neural network."""

    def __init__(self, features: list[int], physics_weight: float = 1.0, rngs: nnx.Rngs = None):
        self.features = features
        self.physics_weight = physics_weight

        # Create layers
        self.layers = []
        for i in range(len(features) - 1):
            self.layers.append(
                nnx.Linear(features[i], features[i + 1], rngs=rngs)
            )

    def __call__(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = nnx.tanh(x)  # Physics-friendly activation

        # Final layer (no activation)
        x = self.layers[-1](x)
        return x

    def physics_loss(self, x, u, derivatives):
        """Compute physics-informed loss."""
        # Example: Heat equation residual
        u_t = derivatives['t']
        u_xx = derivatives['xx']
        residual = u_t - 0.01 * u_xx  # Heat equation
        return jnp.mean(residual**2)

# Create custom network
custom_pinn = PhysicsInformedMLP(
    features=[3, 50, 50, 1],  # (x, y, t) -> u
    physics_weight=1.0,
    rngs=rngs
)

# Test custom network
spatiotemporal_input = jax.random.uniform(jax.random.PRNGKey(11), (64, 3))
custom_output = custom_pinn(spatiotemporal_input)
print(f"Custom PINN: {spatiotemporal_input.shape} -> {custom_output.shape}")
```

### Physics-Aware Layers

Specialized layers that enforce physical constraints:

```python
class ConservationLayer(nnx.Module):
    """Layer that enforces conservation laws."""

    def __init__(self, features: int, rngs: nnx.Rngs = None):
        self.linear = nnx.Linear(features, features, rngs=rngs)
        self.conservation_weight = nnx.Param(jnp.ones(1))

    def __call__(self, x):
        # Standard transformation
        y = self.linear(x)

        # Enforce conservation (sum preservation)
        x_sum = jnp.sum(x, axis=-1, keepdims=True)
        y_sum = jnp.sum(y, axis=-1, keepdims=True)
        conservation_correction = (x_sum - y_sum) / x.shape[-1]

        # Apply conservation constraint
        y_conserved = y + conservation_correction * self.conservation_weight
        return y_conserved

class SymplecticLayer(nnx.Module):
    """Layer that preserves symplectic structure."""

    def __init__(self, features: int, rngs: nnx.Rngs = None):
        assert features % 2 == 0, "Symplectic layer requires even number of features"
        self.features = features
        self.linear_q = nnx.Linear(features // 2, features // 2, rngs=rngs)
        self.linear_p = nnx.Linear(features // 2, features // 2, rngs=rngs)

    def __call__(self, x):
        # Split into position and momentum
        q, p = jnp.split(x, 2, axis=-1)

        # Symplectic transformation
        q_new = q + self.linear_p(p)
        p_new = p - self.linear_q(q_new)

        return jnp.concatenate([q_new, p_new], axis=-1)

# Use physics-aware layers
conservation_layer = ConservationLayer(features=32, rngs=rngs)
symplectic_layer = SymplecticLayer(features=32, rngs=rngs)

# Test layers
test_input = jax.random.normal(jax.random.PRNGKey(12), (16, 32))
conserved_output = conservation_layer(test_input)
symplectic_output = symplectic_layer(test_input)

print(f"Conservation layer: {test_input.shape} -> {conserved_output.shape}")
print(f"Symplectic layer: {test_input.shape} -> {symplectic_output.shape}")
```

## Training Strategies

### Multi-Objective Training

Balance between data fitting and physics constraints:

```python
from opifex.core.physics.losses import PhysicsInformedLoss, PhysicsLossConfig

# Configure multi-objective loss
physics_config = PhysicsLossConfig(
    pde_weight=1.0,
    boundary_weight=10.0,
    initial_weight=1.0,
    data_weight=1.0
)

physics_loss = PhysicsInformedLoss(config=physics_config)

# Custom loss function
def multi_objective_loss(model, params, x_data, y_data, x_physics):
    # Data loss
    y_pred = model(x_data)
    data_loss = jnp.mean((y_pred - y_data)**2)

    # Physics loss
    physics_residual = physics_loss.compute_residual(model, x_physics)
    physics_loss_value = jnp.mean(physics_residual**2)

    # Combined loss
    total_loss = data_loss + physics_config.pde_weight * physics_loss_value
    return total_loss
```

### Adaptive Weighting

Dynamically adjust loss weights during training:

```python
class AdaptiveWeightScheduler:
    """Adaptive weight scheduler for multi-objective training."""

    def __init__(self, initial_weights: dict[str, float]):
        self.weights = initial_weights
        self.loss_history = {key: [] for key in initial_weights}

    def update_weights(self, current_losses: dict[str, float], epoch: int):
        """Update weights based on loss magnitudes and trends."""
        for key, loss_value in current_losses.items():
            self.loss_history[key].append(loss_value)

            # Adaptive weighting based on loss magnitude
            if len(self.loss_history[key]) > 10:
                recent_trend = jnp.mean(jnp.array(self.loss_history[key][-5:]))
                if recent_trend > jnp.mean(jnp.array(self.loss_history[key][-10:-5])):
                    self.weights[key] *= 1.1  # Increase weight if loss is increasing
                else:
                    self.weights[key] *= 0.99  # Slightly decrease if improving

        return self.weights

# Use adaptive weighting
scheduler = AdaptiveWeightScheduler({
    "data": 1.0,
    "physics": 1.0,
    "boundary": 10.0
})
```

### Curriculum Learning

Progressively increase problem complexity:

```python
class CurriculumScheduler:
    """Curriculum learning for physics-informed neural networks."""

    def __init__(self, stages: list[dict]):
        self.stages = stages
        self.current_stage = 0

    def get_current_config(self, epoch: int) -> dict:
        """Get current training configuration based on epoch."""
        # Simple epoch-based curriculum
        stage_length = 1000  # epochs per stage
        stage_idx = min(epoch // stage_length, len(self.stages) - 1)
        return self.stages[stage_idx]

# Define curriculum stages
curriculum_stages = [
    {"domain_complexity": 0.1, "physics_weight": 0.1},  # Simple domain, low physics
    {"domain_complexity": 0.5, "physics_weight": 0.5},  # Medium complexity
    {"domain_complexity": 1.0, "physics_weight": 1.0},  # Full complexity
]

curriculum = CurriculumScheduler(curriculum_stages)
```

## Best Practices

### 1. Initialization Strategies

Physics-informed initialization for better convergence:

```python
def physics_informed_init(key, shape, physics_scale=1e-3):
    """Initialize weights with physics-informed scaling."""
    # Xavier initialization with physics scaling
    fan_in = shape[0] if len(shape) > 1 else 1
    std = jnp.sqrt(2.0 / fan_in) * physics_scale
    return jax.random.normal(key, shape) * std

# Apply to model initialization
def init_physics_model(model_class, config, rngs):
    """Initialize model with physics-informed weights."""
    # Custom initialization logic here
    return model_class(**config, rngs=rngs)
```

### 2. Activation Function Selection

Choose appropriate activations for different physics problems:

```python
# Recommended activations by problem type
ACTIVATION_RECOMMENDATIONS = {
    "heat_equation": "tanh",      # Smooth, bounded
    "wave_equation": "sin",       # Periodic solutions
    "navier_stokes": "swish",     # Smooth, unbounded
    "quantum_systems": "gelu",    # Smooth, good gradients
    "optimization": "relu",       # Simple, fast
}

def get_recommended_activation(problem_type: str) -> str:
    """Get recommended activation for physics problem."""
    return ACTIVATION_RECOMMENDATIONS.get(problem_type, "swish")
```

### 3. Architecture Sizing Guidelines

Balance expressivity with computational cost:

```python
def estimate_model_size(layer_sizes: list[int]) -> dict:
    """Estimate model parameters and memory usage."""
    total_params = 0
    for i in range(len(layer_sizes) - 1):
        total_params += layer_sizes[i] * layer_sizes[i + 1]  # Weights
        total_params += layer_sizes[i + 1]  # Biases

    # Rough memory estimate (bytes)
    memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32

    return {
        "total_parameters": total_params,
        "memory_mb": memory_mb,
        "recommended_batch_size": max(1, int(1000 / jnp.sqrt(total_params)))
    }

# Example usage
model_stats = estimate_model_size([2, 64, 64, 1])
print(f"Model statistics: {model_stats}")
```

### 4. Regularization Techniques

Physics-based regularization for better generalization:

```python
def physics_regularization(model, x, lambda_reg=1e-4):
    """Apply physics-based regularization."""
    # Gradient penalty for smoothness
    def model_fn(x_single):
        return model(x_single.reshape(1, -1)).squeeze()

    # Compute gradients
    grad_fn = jax.grad(model_fn)
    gradients = jax.vmap(grad_fn)(x)

    # Gradient penalty (encourage smoothness)
    gradient_penalty = jnp.mean(jnp.sum(gradients**2, axis=-1))

    return lambda_reg * gradient_penalty

# Apply in training loop
def regularized_loss(model, x_data, y_data, x_physics):
    # Standard loss
    y_pred = model(x_data)
    data_loss = jnp.mean((y_pred - y_data)**2)

    # Add physics regularization
    reg_loss = physics_regularization(model, x_physics)

    return data_loss + reg_loss
```

This comprehensive neural network guide provides everything needed to build, train, and deploy sophisticated scientific machine learning models with Opifex's extensive architecture collection.
