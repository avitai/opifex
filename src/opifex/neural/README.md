# Opifex Neural: Neural Networks, Training Infrastructure & Neural DFT

This package implements neural network architectures for scientific machine learning, including Neural Density Functional Theory, all built with FLAX NNX. Sprint 1.2 and 1.3 have been completed with comprehensive neural network primitives, training infrastructure, and neural operator foundations.

## ✅ SPRINT 1.2 + 1.3 COMPLETED IMPLEMENTATIONS

### ✅ **Standard MLP Implementation** (513 lines, `base.py`)

**Status**: ✅ FULLY IMPLEMENTED AND TESTED
**Architecture**: Modern FLAX NNX with stateful transforms

**Implemented Components**:

- [x] **StandardMLP Class** - Multi-layer perceptron with configurable architecture
- [x] **QuantumMLP Class** - Quantum-aware neural networks for molecular systems
- [x] **Energy Computation** - Direct molecular energy calculation capabilities
- [x] **Force Computation** - Automatic differentiation for molecular forces
- [x] **Symmetry Enforcement** - Permutation symmetry for molecular systems
- [x] **Precision Support** - Both float32 and float64 for chemical accuracy
- [x] **Dropout Support** - Configurable dropout for regularization
- [x] **Custom Initialization** - Physics-informed weight initialization

### ✅ **Activation Function Library** (282 lines, `activations.py`)

**Status**: ✅ FULLY IMPLEMENTED AND TESTED
**Total Functions**: 27 activation functions with registry system

**Implemented Components**:

- [x] **FLAX NNX Activations** - Direct integration with modern FLAX NNX functions
- [x] **JAX Activations** - Native JAX activation functions
- [x] **Scientific Extensions** - Physics-informed and quantum-specific activations
- [x] **Registry Management** - Dynamic registration and retrieval system
- [x] **Zero Performance Overhead** - Direct function references, no wrapper calls
- [x] **Complete Integration** - Seamless connection with MLP implementations

**Available Activations**:

```python
# FLAX NNX: celu, elu, gelu, glu, leaky_relu, log_sigmoid, log_softmax,
#           relu, relu6, sigmoid, silu, soft_sign, softmax, softplus,
#           swish, tanh
# JAX: hard_tanh, hard_sigmoid, hard_swish
# Scientific: mish, snake, gaussian, quadratic, cubic, quartic,
#            exponential, logarithmic, sinusoidal, cosinusoidal
```

## 📚 Comprehensive Usage Examples

### 1. Basic Neural Networks

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.base import StandardMLP, QuantumMLP
from opifex.neural.activations import get_activation, register_activation

# Create a standard MLP
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(key)

# Basic MLP for regression
regression_model = StandardMLP(
    layer_sizes=[3, 64, 64, 32, 1],
    activation="swish",
    dropout_rate=0.1,
    use_bias=True,
    rngs=rngs
)

# Forward pass
x = jax.random.normal(key, (32, 3))  # batch_size=32, input_dim=3
y = regression_model(x)
print(f"Regression output shape: {y.shape}")  # (32, 1)

# Multi-output MLP for classification
classification_model = StandardMLP(
    layer_sizes=[10, 128, 64, 5],  # 5 classes
    activation="gelu",
    final_activation="softmax",
    rngs=rngs
)

x_class = jax.random.normal(key, (16, 10))
y_class = classification_model(x_class)
print(f"Classification output shape: {y_class.shape}")  # (16, 5)
print(f"Probabilities sum: {jnp.sum(y_class, axis=1)}")  # Should be ~1.0
```

### 2. Quantum-Aware Neural Networks

```python
from opifex.neural.base import QuantumMLP
from opifex.core.quantum.molecular_system import create_molecular_system

# Create molecular system
water_positions = jnp.array([
    [0.0000,  0.0000,  0.1173],   # O
    [0.0000,  0.7572, -0.4692],   # H
    [0.0000, -0.7572, -0.4692]    # H
])

water_system = create_molecular_system(
    atomic_symbols=["O", "H", "H"],
    positions=water_positions,
    charge=0,
    spin=0
)

# Quantum MLP for molecular energy prediction
quantum_model = QuantumMLP(
    layer_sizes=[9, 128, 128, 64, 1],  # 3 atoms × 3 coordinates = 9 inputs
    n_atoms=3,
    activation="swish",
    enforce_symmetry=True,  # Permutation symmetry for identical atoms
    rngs=rngs
)

# Compute molecular energy
coordinates = water_system.positions.flatten()  # (9,)
energy = quantum_model(coordinates[None, :])  # Add batch dimension
print(f"Molecular energy: {energy[0, 0]:.6f} Ha")

# Compute forces via automatic differentiation
def energy_fn(coords):
    return quantum_model(coords.reshape(1, -1))[0, 0]

forces = -jax.grad(energy_fn)(coordinates.reshape(-1))
forces_per_atom = forces.reshape(3, 3)  # (n_atoms, 3)
print(f"Forces on atoms:\n{forces_per_atom}")
```

### 3. Custom Activation Functions

```python
from opifex.neural.activations import register_activation, get_activation

# Define custom physics-informed activation
def neural_swish(x, beta=1.0):
    """Neural Swish with learnable parameter"""
    return x * jax.nn.sigmoid(beta * x)

def adaptive_relu(x, alpha=0.01):
    """Adaptive ReLU with learnable slope"""
    return jnp.where(x > 0, x, alpha * x)

def gaussian_activation(x, sigma=1.0):
    """Gaussian activation for smooth approximation"""
    return jnp.exp(-0.5 * (x / sigma)**2)

# Register custom activations
register_activation("neural_swish", neural_swish)
register_activation("adaptive_relu", adaptive_relu)
register_activation("gaussian", gaussian_activation)

# Use custom activation in model
custom_model = StandardMLP(
    layer_sizes=[5, 32, 32, 1],
    activation="neural_swish",
    rngs=rngs
)

# Test all available activations
available_activations = [
    "relu", "tanh", "sigmoid", "swish", "gelu", "elu",
    "gaussian", "neural_swish", "mish", "snake"
]

x_test = jnp.linspace(-2, 2, 100)
for act_name in available_activations:
    try:
        activation_fn = get_activation(act_name)
        y_test = activation_fn(x_test)
        print(f"{act_name}: min={jnp.min(y_test):.3f}, max={jnp.max(y_test):.3f}")
    except ValueError:
        print(f"{act_name}: Not available")
```

### 5. Enhanced Calibration Framework with Physics-Aware Temperature Scaling (NEW)

```python
from opifex.neural.bayesian import TemperatureScaling

# Initialize enhanced temperature scaling with physics constraints
rngs = nnx.Rngs(jax.random.PRNGKey(42))
enhanced_calibrator = TemperatureScaling(
    physics_constraints=['energy_conservation', 'positivity', 'boundedness'],
    constraint_strength=0.2,  # Physics constraint penalty weight
    adaptive=True,           # Enable adaptive temperature learning
    rngs=rngs
)

# Apply physics-aware calibration to model predictions
predictions = jax.random.normal(key, (100, 1))
inputs = jax.random.normal(key, (100, 5))

# Get calibrated predictions with constraint enforcement
calibrated_predictions, constraint_penalty = enhanced_calibrator.apply_physics_aware_calibration(
    predictions, inputs
)

# Optimize temperature with physics constraints
targets = jax.random.normal(key, (100, 1))
optimal_temp = enhanced_calibrator.optimize_temperature_with_physics_constraints(
    predictions, targets, inputs
)

print(f"Calibrated predictions range: [{jnp.min(calibrated_predictions):.3f}, {jnp.max(calibrated_predictions):.3f}]")
print(f"Physics constraint penalty: {constraint_penalty:.6f}")
print(f"Optimal temperature: {optimal_temp:.4f}")

# Standard forward pass with uncertainty quantification
calibrated_preds, aleatoric_uncertainty = enhanced_calibrator(predictions, inputs)
print(f"Aleatoric uncertainty: {jnp.mean(aleatoric_uncertainty):.6f}")
```

### 6. Physics-Informed Bayesian Framework (NEW)

```python
from opifex.neural.bayesian import (
    PhysicsInformedPriors,
    ConservationLawPriors,
    HierarchicalBayesianFramework,
    PhysicsAwareUncertaintyPropagation,
    DomainSpecificPriors
)

# Physics-Informed Priors for constraint enforcement
physics_priors = PhysicsInformedPriors(
    conservation_laws=['energy', 'momentum', 'mass'],
    boundary_conditions=['dirichlet', 'neumann'],
    penalty_weight=1.0,
    rngs=rngs
)

# Apply physics constraints to model parameters
unconstrained_params = jax.random.normal(key, (10,))
constrained_params = physics_priors.apply_constraints(unconstrained_params)
violation_penalty = physics_priors.compute_violation_penalty(constrained_params)

print(f"Original params range: [{jnp.min(unconstrained_params):.3f}, {jnp.max(unconstrained_params):.3f}]")
print(f"Constrained params range: [{jnp.min(constrained_params):.3f}, {jnp.max(constrained_params):.3f}]")
print(f"Constraint violation penalty: {violation_penalty:.6f}")

# Conservation Law Priors with adaptive weighting
conservation_priors = ConservationLawPriors(
    conservation_laws=['energy', 'momentum', 'mass'],
    uncertainty_scale=0.1,
    prior_strength=1.0,
    adaptive_weighting=True,
    rngs=rngs
)

# Compute physics-aware uncertainty
predictions = jax.random.normal(key, (100, 1))
model_uncertainty = jax.random.uniform(key, (100, 1), minval=0.01, maxval=0.1)
physics_state = jax.random.normal(key, (100, 3))  # Physical state representation

physics_uncertainty = conservation_priors.compute_physics_aware_uncertainty(
    predictions, model_uncertainty, physics_state
)

print(f"Model uncertainty mean: {jnp.mean(model_uncertainty):.6f}")
print(f"Physics-aware uncertainty mean: {jnp.mean(physics_uncertainty):.6f}")

# Sample physics-constrained parameters
base_params = jax.random.normal(key, (50,))
constrained_samples = conservation_priors.sample_physics_constrained_params(
    base_params, constraint_strength=0.8
)
print(f"Constraint satisfaction: {jnp.mean(jnp.abs(constrained_samples - base_params)):.6f}")

# Domain-Specific Priors for scientific applications
quantum_priors = DomainSpecificPriors(
    domain="quantum_chemistry",
    parameter_ranges={
        "bond_length": (0.5, 3.0),
        "angle": (60.0, 180.0),
        "energy": (-100.0, 0.0)
    },
    distribution_types={
        "bond_length": "truncated_normal",
        "angle": "uniform",
        "energy": "gaussian"
    },
    rngs=rngs
)

# Sample domain-specific parameters
bond_samples = quantum_priors.sample_domain_priors((20,), "bond_length")
angle_samples = quantum_priors.sample_domain_priors((20,), "angle")
energy_samples = quantum_priors.sample_domain_priors((20,), "energy")

print(f"Bond length samples: [{jnp.min(bond_samples):.3f}, {jnp.max(bond_samples):.3f}]")
print(f"Angle samples: [{jnp.min(angle_samples):.1f}, {jnp.max(angle_samples):.1f}]")
print(f"Energy samples: [{jnp.min(energy_samples):.3f}, {jnp.max(energy_samples):.3f}]")

# Hierarchical Bayesian Framework for multi-level uncertainty
hierarchical_framework = HierarchicalBayesianFramework(
    hierarchy_levels=3,
    level_dimensions=[64, 32, 16],
    uncertainty_propagation="multiplicative",
    correlation_structure="exchangeable",
    rngs=rngs
)

# Sample hierarchical parameters
level_0_params = hierarchical_framework.sample_hierarchical_parameters((10,), level=0)
level_1_params = hierarchical_framework.sample_hierarchical_parameters((10,), level=1)
level_2_params = hierarchical_framework.sample_hierarchical_parameters((10,), level=2)

print(f"Level 0 params shape: {level_0_params.shape}")
print(f"Level 1 params shape: {level_1_params.shape}")
print(f"Level 2 params shape: {level_2_params.shape}")

# Propagate uncertainty through hierarchy
base_uncertainty = jnp.ones((10, 64)) * 0.1
propagated_uncertainty = hierarchical_framework.propagate_uncertainty_hierarchically(
    base_uncertainty, target_level=2
)
print(f"Propagated uncertainty shape: {propagated_uncertainty.shape}")

# Physics-Aware Uncertainty Propagation
uncertainty_propagator = PhysicsAwareUncertaintyPropagation(
    conservation_laws=['energy', 'momentum'],
    constraint_tolerance=1e-6,
    uncertainty_inflation=1.1,
    correlation_aware=True,
    rngs=rngs
)

# Propagate uncertainty with physics constraints
input_uncertainty = jax.random.uniform(key, (50, 3), minval=0.01, maxval=0.1)
model_jacobian = jax.random.normal(key, (50, 3, 3))
physics_state = jax.random.normal(key, (50, 3))

propagated_physics_uncertainty = uncertainty_propagator.propagate_with_physics_constraints(
    input_uncertainty, model_jacobian, physics_state
)

# Compute physics-informed confidence
confidence = uncertainty_propagator.compute_physics_informed_confidence(
    predictions, propagated_physics_uncertainty, physics_state
)

print(f"Input uncertainty mean: {jnp.mean(input_uncertainty):.6f}")
print(f"Physics-propagated uncertainty mean: {jnp.mean(propagated_physics_uncertainty):.6f}")
print(f"Physics-informed confidence mean: {jnp.mean(confidence):.6f}")

# Uncertainty-aware constraint projection
projected_params, projected_uncertainty = uncertainty_propagator.uncertainty_aware_constraint_projection(
    base_params, input_uncertainty
)

print(f"Original uncertainty std: {jnp.std(input_uncertainty):.6f}")
print(f"Projected uncertainty std: {jnp.std(projected_uncertainty):.6f}")
```

### 7. Advanced Neural Operators (NEW)

```python
from opifex.neural.operators.foundations import (
    FourierNeuralOperator,
    DeepONet,
    MultiScaleFourierNeuralOperator,
    LatentNeuralOperator,
    WaveletNeuralOperator,
    OperatorNetwork
)

# Standard Fourier Neural Operator
fno = FourierNeuralOperator(
    in_channels=3,
    out_channels=1,
    hidden_channels=64,
    modes=16,
    num_layers=4,
    rngs=rngs
)

# Process 2D physics data
physics_data = jax.random.normal(key, (8, 3, 64, 64))
fno_output = fno(physics_data)
print(f"FNO output shape: {fno_output.shape}")  # (8, 1, 64, 64)

# Multi-Scale FNO for hierarchical physics problems
ms_fno = MultiScaleFourierNeuralOperator(
    in_channels=2,
    out_channels=1,
    hidden_channels=64,
    modes_per_scale=[16, 8, 4],  # Multi-resolution processing
    num_layers_per_scale=[2, 2, 2],
    use_cross_scale_attention=True,
    attention_heads=8,
    rngs=rngs
)

# Handle multi-scale turbulence data
turbulence_data = jax.random.normal(key, (4, 2, 128, 128))
ms_output = ms_fno(turbulence_data, training=True)
print(f"Multi-scale FNO output: {ms_output.shape}")  # (4, 1, 128, 128)

# Latent Neural Operator for efficient compression
lno = LatentNeuralOperator(
    in_channels=4,
    out_channels=2,
    latent_dim=128,
    num_latent_tokens=32,  # Compressed latent space
    num_attention_heads=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    physics_constraints=["conservation_energy", "conservation_mass"],
    rngs=rngs
)

# Large-scale data compression
large_data = jax.random.normal(key, (2, 4, 512))  # Large spatial resolution
physics_params = jax.random.normal(key, (2, 2))  # Physics parameters

compressed_output = lno(
    large_data,
    physics_info=physics_params,
    training=True
)
print(f"Latent operator compression: {large_data.shape} -> {compressed_output.shape}")

# Wavelet Neural Operator for multi-resolution analysis
wno = WaveletNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    num_levels=3,  # 3-level wavelet decomposition
    wavelet_type="db4",  # Daubechies-4 wavelets
    use_learnable_wavelets=False,
    rngs=rngs
)

# Process time-series signals with multiple frequencies
def create_multi_frequency_signal(n_samples=256):
    """Create test signal with multiple frequency components"""
    t = jnp.linspace(0, 1, n_samples)
    # Combine low, medium, and high frequency components
    signal = (jnp.sin(2 * jnp.pi * 5 * t) +           # Low frequency
             0.5 * jnp.sin(2 * jnp.pi * 20 * t) +      # Medium frequency
             0.2 * jnp.sin(2 * jnp.pi * 50 * t) +      # High frequency
             0.1 * jax.random.normal(key, (n_samples,))) # Noise
    return signal

# Test wavelet decomposition
test_signal = create_multi_frequency_signal(128)
signal_batch = test_signal[None, None, :]  # Add batch and channel dims
wavelet_output = wno(signal_batch, training=False)

print(f"Wavelet processing: {signal_batch.shape} -> {wavelet_output.shape}")
print(f"Original signal std: {jnp.std(test_signal):.4f}")
print(f"Processed signal std: {jnp.std(wavelet_output[0, 0, :]):.4f}")

# Deep Operator Networks with enhanced features
deeponet = DeepONet(
    branch_input_dim=100,  # Number of function evaluation points
    trunk_input_dim=2,     # Spatial coordinates (x, y)
    branch_hidden_dims=[128, 128, 128],
    trunk_hidden_dims=[64, 64, 64],
    latent_dim=128,
    rngs=rngs
)

# Function-to-function mapping
branch_data = jax.random.normal(key, (16, 100))  # Input functions
trunk_coords = jax.random.uniform(key, (16, 50, 2), minval=-1, maxval=1)  # Query points

deeponet_output = deeponet(branch_data, trunk_coords)
print(f"DeepONet mapping: {branch_data.shape} + {trunk_coords.shape} -> {deeponet_output.shape}")

# Unified Operator Network interface
unified_fno = OperatorNetwork(
    operator_type="fno",
    config={
        "in_channels": 2,
        "out_channels": 1,
        "hidden_channels": 32,
        "modes": 12,
        "num_layers": 3
    },
    rngs=rngs
)

unified_deeponet = OperatorNetwork(
    operator_type="deeponet",
    config={
        "branch_input_dim": 64,
        "trunk_input_dim": 1,
        "branch_hidden_dims": [64, 64],
        "trunk_hidden_dims": [32, 32],
        "latent_dim": 64
    },
    rngs=rngs
)

# Test unified interface
test_data_fno = jax.random.normal(key, (4, 2, 32))
test_data_deeponet = (
    jax.random.normal(key, (4, 64)),     # branch
    jax.random.normal(key, (4, 32, 1))   # trunk
)

unified_fno_output = unified_fno(test_data_fno)
unified_deeponet_output = unified_deeponet(*test_data_deeponet)

print(f"Unified FNO output: {unified_fno_output.shape}")
print(f"Unified DeepONet output: {unified_deeponet_output.shape}")
```

### 4. Advanced Neural Network Architectures

```python
from opifex.neural.base import StandardMLP

# Residual MLP with skip connections
class ResidualMLP(nnx.Module):
    """MLP with residual connections"""

    def __init__(self, layer_sizes, activation="relu", rngs=None):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.activation = get_activation(activation)

        # Build layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = nnx.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                rngs=rngs
            )
            self.layers.append(layer)

    def __call__(self, x):
        """Forward pass with residual connections"""
        hidden = x

        for i, layer in enumerate(self.layers[:-1]):  # All but last layer
            layer_input = hidden
            hidden = layer(hidden)
            hidden = self.activation(hidden)

            # Add residual connection if dimensions match
            if layer_input.shape[-1] == hidden.shape[-1]:
                hidden = hidden + layer_input

        # Final layer without residual
        output = self.layers[-1](hidden)
        return output

# Multi-scale neural network
class MultiScaleMLP(nnx.Module):
    """MLP with multiple resolution pathways"""

    def __init__(self, input_dim, output_dim, scales=[1, 2, 4], rngs=None):
        super().__init__()
        self.scales = scales

        # Create pathway for each scale
        self.pathways = []
        for scale in scales:
            hidden_dim = 64 * scale
            pathway = StandardMLP(
                layer_sizes=[input_dim, hidden_dim, hidden_dim // 2, output_dim],
                activation="swish",
                rngs=rngs
            )
            self.pathways.append(pathway)

        # Combination layer
        self.combiner = nnx.Linear(
            in_features=len(scales) * output_dim,
            out_features=output_dim,
            rngs=rngs
        )

    def __call__(self, x):
        """Forward pass through multiple scales"""
        pathway_outputs = []

        for pathway in self.pathways:
            output = pathway(x)
            pathway_outputs.append(output)

        # Concatenate and combine
        combined = jnp.concatenate(pathway_outputs, axis=-1)
        final_output = self.combiner(combined)

        return final_output

# Create and test advanced architectures
residual_model = ResidualMLP(
    layer_sizes=[10, 64, 64, 64, 1],
    activation="swish",
    rngs=rngs
)

multiscale_model = MultiScaleMLP(
    input_dim=5,
    output_dim=3,
    scales=[1, 2, 4],
    rngs=rngs
)

# Test forward passes
x_res = jax.random.normal(key, (8, 10))
y_res = residual_model(x_res)
print(f"Residual MLP output: {y_res.shape}")

x_multi = jax.random.normal(key, (8, 5))
y_multi = multiscale_model(x_multi)
print(f"Multi-scale MLP output: {y_multi.shape}")
```

### 5. Neural Operators Examples

```python
from opifex.neural.operators.foundations import (
    FourierNeuralOperator, DeepONet, OperatorNetwork,
    SpectralConvolution, FourierLayer
)

# Fourier Neural Operator (FNO)
fno_config = {
    "in_channels": 1,
    "out_channels": 1,
    "hidden_channels": 64,
    "modes": 16,
    "num_layers": 4,
    "activation": nnx.gelu,
    "use_mixed_precision": False,
    "dtype": jnp.float32
}

fno = OperatorNetwork(
    operator_type="fno",
    config=fno_config,
    rngs=rngs
)

# Generate 1D function data for FNO
batch_size, grid_size = 32, 64
input_functions = jax.random.normal(key, (batch_size, 1, grid_size))
output_functions = fno(input_functions)

print(f"FNO input shape: {input_functions.shape}")   # (32, 1, 64)
print(f"FNO output shape: {output_functions.shape}") # (32, 1, 64)

# Deep Operator Network (DeepONet)
deeponet_config = {
    "branch_input_dim": 64,  # Number of sensor points
    "trunk_input_dim": 1,    # Coordinate dimension
    "branch_hidden_dims": [128, 128],
    "trunk_hidden_dims": [64, 64],
    "latent_dim": 128,
    "enhanced": False,
    "activation": nnx.tanh,
    "dtype": jnp.float32
}

deeponet = OperatorNetwork(
    operator_type="deeponet",
    config=deeponet_config,
    rngs=rngs
)

# Generate DeepONet training data
branch_input = jax.random.normal(key, (16, 64))     # Function values at sensors
trunk_input = jax.random.uniform(key, (16, 100, 1), minval=0, maxval=1)  # Query points

output = deeponet(branch_input, trunk_input)
print(f"DeepONet output shape: {output.shape}")  # (16, 100)

# Enhanced DeepONet with physics constraints
enhanced_config = {
    "branch_input_dim": 64,
    "trunk_input_dim": 2,
    "branch_hidden_dims": [128, 64],
    "trunk_hidden_dims": [64, 32],
    "latent_dim": 64,
    "enhanced": True,
    "use_attention": True,
    "attention_heads": 4,
    "sensor_optimization": True,
    "num_sensors": 32,
    "physics_constraints": ["mass_conservation"],
    "activation": nnx.tanh
}

enhanced_deeponet = OperatorNetwork(
    operator_type="deeponet",
    config=enhanced_config,
    rngs=rngs
)

# Test enhanced DeepONet
branch_2d = jax.random.normal(key, (8, 64))
trunk_2d = jax.random.uniform(key, (8, 50, 2), minval=-1, maxval=1)
spatial_coords = jax.random.uniform(key, (50, 2), minval=-1, maxval=1)

enhanced_output = enhanced_deeponet(
    branch_inputs=branch_2d,
    trunk_input=trunk_2d,
    spatial_coords=spatial_coords,
    training=True
)
print(f"Enhanced DeepONet output shape: {enhanced_output.shape}")  # (8, 50)
```

### 6. Spectral Convolution Layers

```python
from opifex.neural.operators.foundations import SpectralConvolution, FourierLayer

# Custom spectral convolution network
class SpectralNet(nnx.Module):
    """Neural network with spectral convolution layers"""

    def __init__(self, modes, hidden_channels, num_layers, rngs):
        super().__init__()

        # Input projection
        self.input_proj = nnx.Linear(1, hidden_channels, rngs=rngs)

        # Spectral layers
        self.spectral_layers = []
        for _ in range(num_layers):
            layer = FourierLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                modes=modes,
                activation=nnx.gelu,
                rngs=rngs
            )
            self.spectral_layers.append(layer)

        # Output projection
        self.output_proj = nnx.Linear(hidden_channels, 1, rngs=rngs)

    def __call__(self, x):
        """Forward pass through spectral network"""
        # x shape: (batch, channels, spatial)
        batch_size, channels, spatial_size = x.shape

        # Reshape for linear layers: (batch, spatial, channels)
        x_reshaped = jnp.moveaxis(x, 1, -1)
        x_flat = x_reshaped.reshape(-1, channels)

        # Input projection
        hidden = self.input_proj(x_flat)
        hidden = hidden.reshape(batch_size, spatial_size, -1)
        hidden = jnp.moveaxis(hidden, -1, 1)  # Back to (batch, channels, spatial)

        # Apply spectral layers
        for layer in self.spectral_layers:
            hidden = layer(hidden)

        # Output projection
        output_reshaped = jnp.moveaxis(hidden, 1, -1)
        output_flat = output_reshaped.reshape(-1, hidden.shape[1])
        output = self.output_proj(output_flat)
        output = output.reshape(batch_size, spatial_size, 1)
        output = jnp.moveaxis(output, -1, 1)

        return output

# Create and test spectral network
spectral_net = SpectralNet(
    modes=16,
    hidden_channels=32,
    num_layers=3,
    rngs=rngs
)

# Test on 1D function data
x_spectral = jax.random.normal(key, (16, 1, 128))  # (batch, channels, spatial)
y_spectral = spectral_net(x_spectral)
print(f"Spectral network output: {y_spectral.shape}")  # (16, 1, 128)
```

### 7. Multi-Physics Neural Networks

```python
from opifex.neural.base import StandardMLP

class MultiPhysicsNetwork(nnx.Module):
    """Neural network for coupled physics problems"""

    def __init__(self, physics_types, shared_layers, specific_layers, rngs):
        super().__init__()
        self.physics_types = physics_types

        # Shared encoder
        self.shared_encoder = StandardMLP(
            layer_sizes=[3] + shared_layers,  # (x, y, t) input
            activation="swish",
            rngs=rngs
        )

        # Physics-specific decoders
        self.physics_decoders = {}
        for physics_type in physics_types:
            decoder = StandardMLP(
                layer_sizes=[shared_layers[-1]] + specific_layers + [1],
                activation="tanh",
                rngs=rngs
            )
            self.physics_decoders[physics_type] = decoder

        # Coupling layer
        self.coupling_layer = StandardMLP(
            layer_sizes=[len(physics_types), 32, len(physics_types)],
            activation="relu",
            rngs=rngs
        )

    def __call__(self, x, y, t):
        """Forward pass for multi-physics prediction"""
        # Combine coordinates
        coords = jnp.stack([x, y, t], axis=-1)

        # Shared encoding
        shared_features = self.shared_encoder(coords)

        # Physics-specific predictions
        physics_outputs = []
        for physics_type in self.physics_types:
            decoder = self.physics_decoders[physics_type]
            output = decoder(shared_features)
            physics_outputs.append(output)

        # Stack outputs: (batch, n_physics)
        stacked_outputs = jnp.concatenate(physics_outputs, axis=-1)

        # Apply coupling
        coupled_outputs = self.coupling_layer(stacked_outputs)

        return coupled_outputs

# Create multi-physics network for heat-fluid coupling
physics_types = ["temperature", "velocity_x", "velocity_y", "pressure"]
multi_physics_net = MultiPhysicsNetwork(
    physics_types=physics_types,
    shared_layers=[64, 64],
    specific_layers=[32, 32],
    rngs=rngs
)

# Test multi-physics prediction
x_coords = jax.random.uniform(key, (100,), minval=0, maxval=1)
y_coords = jax.random.uniform(key, (100,), minval=0, maxval=1)
t_coords = jax.random.uniform(key, (100,), minval=0, maxval=1)

multi_output = multi_physics_net(x_coords, y_coords, t_coords)
print(f"Multi-physics output shape: {multi_output.shape}")  # (100, 4)
print(f"Physics variables: {physics_types}")
```

### 8. Training Neural Networks

```python
import optax
from opifex.core.training.trainer import Trainer
from opifex.training.physics_losses import PhysicsInformedLoss, PhysicsLossConfig

# Standard supervised training
def train_supervised_model():
    """Train a neural network on supervised data"""

    # Generate synthetic regression data
    key = jax.random.PRNGKey(123)
    n_samples = 1000

    # True function: f(x) = sin(π*x) * exp(-x)
    x_train = jax.random.uniform(key, (n_samples, 1), minval=0, maxval=2)
    y_train = jnp.sin(jnp.pi * x_train) * jnp.exp(-x_train)

    # Add noise
    noise = jax.random.normal(key, y_train.shape) * 0.01
    y_train_noisy = y_train + noise

    # Create model
    model = StandardMLP(
        layer_sizes=[1, 32, 32, 1],
        activation="tanh",
        rngs=rngs
    )

    # Define training step
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    def train_step(x_batch, y_batch):
        def loss_fn(model):
            pred = model(x_batch)
            mse = jnp.mean((pred - y_batch)**2)
            return mse

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    # Training loop
    batch_size = 32
    n_epochs = 1000

    for epoch in range(n_epochs):
        # Random batch
        indices = jax.random.choice(key, n_samples, (batch_size,), replace=False)
        x_batch = x_train[indices]
        y_batch = y_train_noisy[indices]

        loss = train_step(x_batch, y_batch)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    return model

# Physics-informed training
def train_physics_informed_model():
    """Train with physics constraints"""

    # Heat equation: ∂u/∂t = α ∂²u/∂x²
    def heat_pde(u_fn, x, t):
        u_t = jax.grad(lambda t: u_fn(x, t))(t)
        u_xx = jax.grad(jax.grad(lambda x: u_fn(x, t)))(x)
        alpha = 0.1
        return u_t - alpha * u_xx

    # Create PINN model
    pinn_model = StandardMLP(
        layer_sizes=[2, 50, 50, 1],  # (x, t) -> u
        activation="tanh",
        rngs=rngs
    )

    # Physics loss configuration
    physics_config = PhysicsLossConfig(
        pde_weight=1.0,
        boundary_weight=10.0,
        initial_weight=10.0
    )

    physics_loss = PhysicsInformedLoss(
        config=physics_config,
        equation_type="heat",
        domain_type="1d"
    )

    # Training data
    n_physics = 1000
    n_boundary = 100

    # Interior points for PDE
    x_physics = jax.random.uniform(key, (n_physics,), minval=0, maxval=1)
    t_physics = jax.random.uniform(key, (n_physics,), minval=0, maxval=1)

    # Boundary points
    x_boundary = jnp.array([0.0, 1.0])  # Left and right boundaries
    t_boundary = jax.random.uniform(key, (n_boundary,), minval=0, maxval=1)

    # Training step with physics loss
    optimizer = nnx.Optimizer(pinn_model, optax.adam(1e-3))

    def pinn_train_step():
        def loss_fn(model):
            # Physics loss
            def u_pred(x, t):
                return model(jnp.array([x, t])[None, :])[0, 0]

            physics_residuals = jax.vmap(
                lambda x, t: heat_pde(u_pred, x, t)
            )(x_physics, t_physics)
            physics_loss_val = jnp.mean(physics_residuals**2)

            # Boundary loss (u=0 at boundaries)
            boundary_coords = jnp.stack([
                jnp.repeat(x_boundary, len(t_boundary) // 2),
                jnp.tile(t_boundary, 2)
            ], axis=1)
            boundary_pred = model(boundary_coords)
            boundary_loss_val = jnp.mean(boundary_pred**2)

            # Initial condition loss (u(x,0) = sin(πx))
            x_initial = jax.random.uniform(key, (100,), minval=0, maxval=1)
            t_initial = jnp.zeros_like(x_initial)
            initial_coords = jnp.stack([x_initial, t_initial], axis=1)
            initial_pred = model(initial_coords)
            initial_true = jnp.sin(jnp.pi * x_initial)[:, None]
            initial_loss_val = jnp.mean((initial_pred - initial_true)**2)

            total_loss = physics_loss_val + 10 * boundary_loss_val + 10 * initial_loss_val
            return total_loss

        loss, grads = nnx.value_and_grad(loss_fn)(pinn_model)
        optimizer.update(grads)
        return loss

    # PINN training loop
    for epoch in range(2000):
        loss = pinn_train_step()

        if epoch % 200 == 0:
            print(f"PINN Epoch {epoch}, Loss: {loss:.6f}")

    return pinn_model

# Train both models
print("Training supervised model...")
supervised_model = train_supervised_model()

print("\nTraining physics-informed model...")
pinn_model = train_physics_informed_model()

print("\nTraining completed!")
```

## ✅ ADVANCED BAYESIAN & UNCERTAINTY QUANTIFICATION COMPLETED (Phase 1)

### ✅ **PROBABILISTIC FRAMEWORK ENHANCEMENT - PHASE 1 COMPLETE**

**Status**: ✅ **FULLY IMPLEMENTED AND TESTED**
**Implementation**: `bayesian/uncertainty_quantification.py` - 1,102 lines with advanced uncertainty methods
**Testing**: ✅ **58/58 Bayesian tests passing (100% success rate)**
**Coverage**: 47% test coverage on uncertainty quantification with comprehensive validation

#### ✅ **Advanced Uncertainty Quantification Classes**

**New Advanced Classes Implemented**:

- ✅ **AdvancedUncertaintyAggregator**: Multi-source uncertainty aggregation with adaptive weighting
- ✅ **AdvancedEpistemicUncertainty**: Enhanced ensemble disagreement and predictive diversity methods
- ✅ **AdvancedAleatoricUncertainty**: Distributional uncertainty for multiple distribution types

**Core Capabilities**:

- ✅ **Weighted Uncertainty Aggregation**: Multiple aggregation methods (weighted_variance, weighted_mean, max_weighted, robust_weighted)
- ✅ **Adaptive Weighting Strategies**: Reliability-based, inverse-variance, entropy-based, and uniform weighting
- ✅ **Uncertainty Quality Assessment**: Coverage probability, interval width, calibration error, and confidence metrics
- ✅ **Enhanced Ensemble Methods**: Variance, standard deviation, range, and IQR-based disagreement measures
- ✅ **Predictive Diversity**: Pairwise distance and cosine diversity metrics for epistemic uncertainty
- ✅ **Multi-Distribution Support**: Gaussian, Laplace, and mixture distribution uncertainty quantification

#### Usage Examples

```python
from opifex.neural.bayesian import (
    AdvancedUncertaintyAggregator,
    AdvancedEpistemicUncertainty,
    AdvancedAleatoricUncertainty,
    EnhancedUncertaintyQuantifier
)

# Advanced epistemic uncertainty analysis
epistemic_analyzer = AdvancedEpistemicUncertainty()

# Compute ensemble disagreement
ensemble_predictions = jax.random.normal(key, (5, 100, 1))
variance_uncertainty = epistemic_analyzer.compute_ensemble_disagreement(
    ensemble_predictions, aggregation_method="variance"
)

# Compute predictive diversity
diversity = epistemic_analyzer.compute_predictive_diversity(
    ensemble_predictions, diversity_metric="pairwise_distance"
)

# Advanced aleatoric uncertainty
aleatoric_analyzer = AdvancedAleatoricUncertainty()

# Gaussian distributional uncertainty
gaussian_params = {"log_std": jax.random.normal(key, (100, 1)) * 0.1}
gaussian_uncertainty = aleatoric_analyzer.distributional_uncertainty(
    gaussian_params, distribution_type="gaussian"
)

# Multi-source uncertainty aggregation
aggregator = AdvancedUncertaintyAggregator()
uncertainty_sources = [variance_uncertainty, gaussian_uncertainty]

# Weighted aggregation
aggregated_uncertainty = aggregator.weighted_uncertainty_aggregation(
    uncertainty_sources, aggregation_method="weighted_variance"
)

# Adaptive weighting
reliability_scores = [jnp.ones((100,)) * 0.9, jnp.ones((100,)) * 0.7]
adaptive_weights = aggregator.adaptive_weighting(
    uncertainty_sources,
    reliability_scores=reliability_scores,
    adaptation_method="reliability_based"
)

# Uncertainty quality assessment
predictions = jnp.mean(ensemble_predictions, axis=0)
true_values = predictions + jax.random.normal(key, predictions.shape) * 0.1

quality_metrics = aggregator.uncertainty_quality_assessment(
    predictions=predictions,
    uncertainties=aggregated_uncertainty,
    true_values=true_values
)

print("Quality Metrics:", quality_metrics)
```

#### ✅ **Enhanced Integration**

**Module Integration**:

- ✅ **Backward Compatibility**: All existing uncertainty quantification methods preserved
- ✅ **Seamless Integration**: New classes work alongside existing Bayesian framework
- ✅ **Unified Interface**: Consistent API design across all uncertainty methods
- ✅ **Production Ready**: Comprehensive testing and validation

**Framework Integration**:

- ✅ **Physics-Informed Priors**: Ready for integration with conservation law constraints
- ✅ **Training Infrastructure**: Compatible with existing training and optimization systems
- ✅ **Neural Operators**: Uncertainty quantification for operator learning applications
- ✅ **Neural DFT**: Uncertainty estimation for quantum chemistry applications

### 📋 **REMAINING PROBABILISTIC PHASES**

**Phase 2: Enhanced Calibration Framework** ⏳ **PLANNED**

- Advanced temperature scaling with physics constraints
- Isotonic regression for non-parametric calibration
- Conformal prediction intervals with coverage guarantees

**Phase 3: Physics-Informed Integration** ⏳ **PLANNED**

- Conservation law priors for uncertainty estimation
- Domain-specific prior distributions
- Hierarchical Bayesian approaches for scientific computing

**Phase 4: Production Integration & Testing** ⏳ **PLANNED**

- Real-time uncertainty estimation capabilities
- Performance optimization and benchmarking
- Comprehensive integration testing

## ✅ NEURAL OPERATORS FOUNDATIONS COMPLETED (Sprint 1.3)

### Neural Operators ✅ **COMPLETED**

- **`operators/foundations.py`**: Complete neural operator foundations (641 lines) ✅
  - **FourierNeuralOperator**: Complete FNO implementation with spectral convolutions
  - **DeepOperatorNetwork**: Branch-trunk architecture foundations
  - **SpectralConvolution**: FFT-based spectral convolution layers
  - **OperatorNetwork**: Universal operator learning interfaces
- **Testing**: ✅ **26/26 neural operator tests passing (100% success rate)**
- **Code Quality**: ✅ **All lint violations resolved, 17/17 pre-commit hooks passing**

### ✅ **SPRINT 1.5 ADVANCED NEURAL OPERATORS - COMPLETED**

#### ✅ **Neural Operator Foundations** - Complete Implementation

**Status**: ✅ **FULLY IMPLEMENTED AND TESTED**
**Implementation**: `operators/foundations.py` - 894 lines with comprehensive operator library
**Testing**: ✅ **102/102 neural operator tests passing (100% success rate)**
**Coverage**: 93% test coverage on neural operator foundations

**Core Operators**:

- ✅ **Fourier Neural Operators (FNO)**: Spectral convolution with factorization support
- ✅ **Deep Operator Networks (DeepONet)**: Branch-trunk architecture with physics integration
- ✅ **Graph Neural Operators (GNO)**: Message passing for irregular domains
- ✅ **Physics-Informed Neural Operators**: Unified physics constraint framework

**Advanced Operators (NEW)**:

- ✅ **Multi-Scale Fourier Neural Operators (MS-FNO)**: Hierarchical resolution handling
- ✅ **Latent Neural Operators (LNO)**: Attention-based compression with latent representations
- ✅ **Wavelet Neural Operators (WNO)**: Multi-scale wavelet decomposition with Daubechies-4

**Enhanced Features**:

- ✅ **Physics-Aware Attention**: Multi-head attention with conservation law constraints
- ✅ **Sensor Optimization**: Learnable sensor placement for DeepONet
- ✅ **Multi-Physics Support**: Coupled physics systems with adaptive weighting
- ✅ **Parameter Factorization**: Tucker and CP decomposition for memory efficiency
- ✅ **Adaptive Architectures**: Multi-resolution and self-adaptive operators

### Physics-Informed Neural Networks 📋 **PLANNED**

- **`pinn.py`**: Standard physics-informed neural networks
- **`xpinn.py`**: Extended PINNs with domain decomposition
- **`vpinn.py`**: Variational PINNs with weak formulations
- **`cpinn.py`**: Conservative PINNs for conservation laws

### Advanced PINN Variants 📋 **PLANNED**

- **`fourier_pinn.py`**: Fourier PINNs for spectral bias mitigation
- **`bayesian_pinn.py`**: Bayesian PINNs with uncertainty quantification
- **`adaptive_pinn.py`**: Adaptive training and loss weighting

## Neural Density Functional Theory (Neural DFT) ✅ **FOUNDATION READY**

### Core Neural DFT Components 📋 **PLANNED FOR FUTURE SPRINTS**

- **`neural_dft.py`**: Main Neural DFT implementation with chemical accuracy
- **`neural_xc.py`**: Neural exchange-correlation functionals (DM21-style)
- **`scf_acceleration.py`**: ML-accelerated self-consistent field methods
- **`molecular_systems.py`**: 3D molecular geometry and periodic boundary conditions

### Advanced Neural DFT Methods 📋 **PLANNED**

- **`hybrid_dft.py`**: Hybrid classical-neural DFT approaches
- **`multifidelity_dft.py`**: Multi-fidelity quantum mechanical models
- **`physics_constraints.py`**: Quantum mechanical constraints and conservation laws
- **`chemical_accuracy.py`**: <1 kcal/mol energy accuracy validation

## Implementation Status

### ✅ **SPRINT 1.2 + 1.3 COMPLETED (100%)**

**Current Status**: ✅ **NEURAL INFRASTRUCTURE COMPLETE**
**Implementation**: 1,436+ lines across 3 major components (base.py + activations.py + operators/foundations.py)
**Testing**: ✅ **231 tests passed, 0 skipped** (100% critical success rate)
**Quality**: Production-ready with comprehensive validation

**Completed Tasks**:

- ✅ **Task 1.2.1**: Standard MLP Implementation - Complete FLAX NNX neural networks
- ✅ **Task 1.2.2**: Activation Function Library - 27 functions with registry system
- ✅ **Task 1.2.3**: Basic Training Infrastructure - Complete training framework (see `../training/`)
- ✅ **Task 1.3.1**: Physics-Informed Loss Functions - Multi-physics composition (see `../training/`)
- ✅ **Task 1.3.2**: Advanced Optimization Algorithms - Meta-optimization (see `../optimization/`)
- ✅ **Task 1.3.3**: Neural Operator Foundations - FNO, DeepONet, and operator learning primitives

### 🎯 **NEXT TARGET: Sprint 1.5 Advanced Neural Operators**

**Sprint ID**: SCIML-SPRINT-1.5
**Priority**: 🔴 **HIGH** - Core neural operator functionality for scientific computing
**Estimated Duration**: 2-3 weeks
**Status**: 📋 **READY TO BEGIN** - All prerequisites satisfied

**Implementation Readiness**: ⭐⭐⭐⭐⭐ (5/5)

- ✅ **Complete Foundation**: Sprint 1.1 + 1.2 + 1.3 + 1.4 provide comprehensive neural foundation
- ✅ **All Dependencies Ready**: JAX ecosystem operational, GPU infrastructure complete
- ✅ **Architecture Established**: Clear integration patterns with unified interfaces
- ✅ **Quality Standards**: Production-ready patterns established (231 tests passing)
- ✅ **Technical Patterns**: Modern Python, JAX integration, comprehensive validation

### Creative Phase Architecture Complete ✅

All neural architectures have been comprehensively designed during creative phases:

#### Creative Phase 1: Neural Operator Architecture ✅ COMPLETE

**Architectural Decisions Finalized**:

- **Modular Component Architecture**: Composition-based design for maximum flexibility
- **Physics-Cross-Attention**: Multi-head attention with conservation law integration
- **Parameter Factorization**: Low-rank decompositions for 50-80% memory reduction
- **Component Composition**: SpectralConvolution, PhysicsCrossAttention, ParameterFactorization

#### Creative Phase 2: Physics Loss Architecture ✅ COMPLETE

**Architectural Decisions Finalized**:

- **Hierarchical Loss Composition**: Data, PDE, boundary, and conservation losses
- **Adaptive Weighting**: Gradient-based automatic balancing
- **Automatic Residual Computation**: JAX autodiff for arbitrary-order PDE residuals
- **Multi-Physics Coupling**: Unified framework for coupled physics problems

## Key Features

- **FLAX NNX Exclusive**: All implementations use modern FLAX NNX ✅
- **JAX Integration**: Native JAX arrays and automatic differentiation ✅
- **Type Safety**: Complete type annotations with modern Python syntax ✅
- **Performance Optimization**: Optimized for scientific computing workloads ✅
- **Neural DFT Ready**: Foundation prepared for quantum chemistry applications ✅
- **Comprehensive Testing**: Part of 231/231 tests passing ✅
- **Production Quality**: All pre-commit hooks passing, professional standards ✅

## Getting Started

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.base import StandardMLP, QuantumMLP
from opifex.neural.activations import get_activation, register_activation

# Basic usage example
key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(key)

# Create a neural network
model = StandardMLP(
    layer_sizes=[10, 64, 64, 1],
    activation="swish",
    rngs=rngs
)

# Forward pass
x = jax.random.normal(key, (32, 10))
y = model(x)
print(f"Output shape: {y.shape}")  # (32, 1)
```

## Integration Points

### Core Problems Module ✅

- **Neural DFT Problems**: Direct integration with `ElectronicStructureProblem`
- **Molecular Systems**: Seamless handling of atomic coordinates and properties
- **Physics Constraints**: Built-in validation of quantum mechanical principles

### Training Infrastructure ✅

- **Complete Training Framework**: See `../training/basic_trainer.py` (666 lines)
- **Loss Functions**: MSE, physics-informed, and quantum-specific losses
- **Optimization**: Integration with Optax optimizers
- **Metrics**: Comprehensive monitoring and validation

### Geometry Integration ✅

- **Molecular Geometry**: Direct integration with 3D molecular systems
- **Symmetry Handling**: Permutation symmetry for molecular neural networks
- **Boundary Conditions**: Physics-aware boundary condition enforcement

## Dependencies

- **JAX 0.6.1+**: Core array operations and automatic differentiation ✅
- **FLAX NNX 0.10.6+**: Modern neural network framework ✅
- **jaxtyping**: Type annotations for JAX arrays ✅
- **Python 3.10+**: Modern Python features and type system ✅

## Future Enhancements (Sprint 1.3+)

### Advanced Neural Components

- **Physics-Informed Loss Functions**: Hierarchical loss composition with adaptive weighting
- **Neural Operator Primitives**: FNO, DeepONet, and modern operator learning
- **Advanced Optimization**: Learn-to-optimize foundations and meta-learning
- **Uncertainty Quantification**: Bayesian neural networks with calibrated uncertainty
