# Opifex Neural: Neural Networks, Operators & Quantum Models

This package implements neural network architectures for scientific machine
learning, built with FLAX NNX. It spans standard MLPs and activations, neural
operators, machine-learning interatomic potentials, an equivariant core, and a
differentiable quantum-chemistry stack.

## Subpackages

- **`base` / `activations`** — `StandardMLP` and a registry of activation
  functions (FLAX NNX, JAX, and scientific extensions such as `mish`, `snake`,
  `gaussian`).
- **`atomistic`** — E(3)-aware machine-learning interatomic potentials
  (SchNet / PaiNN / NequIP backbones) assembled as an `AtomisticModel` with
  energy / forces / stress and charge / spin / dipole heads, plus fine-tuning
  utilities (LoRA, parameter EMA, element remapping) and ASE integration.
- **`equivariant`** — irreps algebra (`Irreps`, `IrrepsArray`), tensor products
  (`FullyConnectedTensorProduct`, `ChannelwiseTensorProduct`), spherical
  harmonics, radial bases, gates, and scatter / radius-graph primitives.
- **`operators`** — neural operators for PDE and function-space learning: FNO and
  variants (TFNO, UFNO, SFNO, Local/AM/MS-FNO), DeepONet variants, graph / mesh
  operators (`GraphNeuralOperator`, `MeshGraphNet`), and uncertainty-aware
  operators. `operators.foundations` re-exports the common entry points.
- **`pinns`** — physics-informed networks (`SimplePINN`, `MultiScalePINN`) with
  factory helpers for heat, Poisson, and Navier–Stokes problems.
- **`bayesian`** — variational layers, calibration tools (`TemperatureScaling`,
  `PlattScaling`, `IsotonicRegression`), and probabilistic / multi-fidelity PINNs.
- **`quantum`** — differentiable Kohn–Sham DFT (`SCFSolver`), a trainable neural
  exchange–correlation functional (`NeuralXCFunctional`), and Hamiltonian /
  variational-Monte-Carlo models (see [`quantum/README.md`](quantum/README.md)).
- **`kan` / `clifford`** — Kolmogorov–Arnold and Clifford-algebra building blocks.

## Activation Library

A registry maps names to activation functions and supports dynamic
registration:

```python
# FLAX NNX: celu, elu, gelu, glu, leaky_relu, log_sigmoid, log_softmax,
#           relu, relu6, sigmoid, silu, soft_sign, softmax, softplus,
#           swish, tanh
# JAX: hard_tanh, hard_sigmoid, hard_swish
# Scientific: mish, snake, gaussian, quadratic, cubic, quartic,
#            exponential, logarithmic, sinusoidal, cosinusoidal
```

## Usage Examples

### 1. Basic Neural Networks

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.base import StandardMLP
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

### 2. Atomistic Machine-Learning Potentials

Molecular and materials property prediction uses the `opifex.neural.atomistic`
machine-learning interatomic potentials: a permutation- and E(3)-aware backbone
produces per-atom embeddings, and typed heads read them out into energy, forces
and stress. See the [Atomistic Potentials guide](../../../docs/methods/atomistic-potentials.md).

```python
from flax import nnx

from opifex.core.quantum.molecular_system import create_water_molecule
from opifex.core.quantum.protocols import RadiusNeighborList
from opifex.core.quantum.registry import BackboneRegistry
from opifex.neural.atomistic import AtomisticModel
from opifex.neural.atomistic.heads import EnergyHead, ForcesHead

# Importing the backbones package registers "schnet" / "painn" / "nequip".
import opifex.neural.atomistic.backbones  # noqa: F401

rngs = nnx.Rngs(0)

# Build a SchNet-backed potential with energy and (conservative) force heads.
backbone = BackboneRegistry().require("schnet")(rngs=rngs)
model = AtomisticModel(
    backbone=backbone,
    heads={"energy": EnergyHead(feature_dim=64, rngs=rngs), "forces": ForcesHead()},
    neighbor_list=RadiusNeighborList(cutoff=5.0),
    max_edges=64,
)

# Predict every configured property in one call.
prediction = model(create_water_molecule())
print(f"Energy: {prediction['energy']}")            # scalar invariant energy
print(f"Forces: {prediction['forces'].shape}")      # (n_atoms, 3), -dE/dR
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

### 4. Calibration with Physics-Aware Temperature Scaling

```python
from opifex.neural.bayesian import TemperatureScaling

# Initialize temperature scaling with physics constraints
rngs = nnx.Rngs(jax.random.PRNGKey(42))
calibrator = TemperatureScaling(
    physics_constraints=['energy_conservation', 'positivity', 'boundedness'],
    constraint_strength=0.2,  # Physics constraint penalty weight
    adaptive=True,           # Enable adaptive temperature learning
    rngs=rngs
)

# Apply physics-aware calibration to model predictions
predictions = jax.random.normal(key, (100, 1))
inputs = jax.random.normal(key, (100, 5))

# Get calibrated predictions with constraint enforcement
calibrated_predictions, constraint_penalty = calibrator.apply_physics_aware_calibration(
    predictions, inputs
)

# Optimize temperature with physics constraints
targets = jax.random.normal(key, (100, 1))
optimal_temp = calibrator.optimize_temperature_with_physics_constraints(
    predictions, targets, inputs
)

print(f"Physics constraint penalty: {constraint_penalty:.6f}")
print(f"Optimal temperature: {optimal_temp:.4f}")
```

### 5. Neural Operators

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

test_signal = jax.random.normal(key, (1, 1, 128))
wavelet_output = wno(test_signal, training=False)
print(f"Wavelet processing: {test_signal.shape} -> {wavelet_output.shape}")

# Deep Operator Networks
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

test_data_fno = jax.random.normal(key, (4, 2, 32))
print(f"Unified FNO output: {unified_fno(test_data_fno).shape}")
```

### 6. Spectral Convolution Layers

`FourierLayer` is the reusable spectral block underlying the FNO models and can
be composed into custom networks:

```python
from opifex.neural.operators.foundations import FourierLayer

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

### 7. Training Neural Networks

```python
import optax
from opifex.core.training.trainer import Trainer
from opifex.core.physics.losses import PhysicsInformedLoss, PhysicsLossConfig

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

    return pinn_model

supervised_model = train_supervised_model()
pinn_model = train_physics_informed_model()
```

## Differentiable Kohn-Sham DFT

The `quantum/` subpackage provides a native-JAX molecular Kohn-Sham DFT solver
and a trainable neural exchange-correlation functional (see
[`quantum/README.md`](quantum/README.md)).

- **`quantum/dft/`**: Restricted Kohn-Sham SCF (`SCFSolver`) on the
  McMurchie-Davidson Gaussian-integral backend, with LDA / PBE functionals,
  DIIS and direct-minimisation modes, and implicit-diff analytic forces.
- **`quantum/neural_xc.py`**: Constrained, attention-based neural
  exchange-correlation functional (`NeuralXCFunctional`) wired into the SCF
  with exact `dE/dtheta` for end-to-end learned-XC training.

## Key Features

- **FLAX NNX**: All implementations use modern FLAX NNX.
- **JAX Integration**: Native JAX arrays and automatic differentiation.
- **Type Safety**: Complete type annotations with modern Python syntax.
- **Quantum-Chemistry Ready**: Foundation for atomistic and electronic-structure
  applications.

## Integration Points

- **Core quantum module**: integrates with `ElectronicStructureProblem`,
  molecular systems, and physics constraints.
- **Training infrastructure**: see `../training/` for trainers, loss functions
  (MSE, physics-informed, quantum-specific), and Optax-based optimization.
- **Geometry**: molecular geometry, permutation symmetry, and boundary
  conditions integrate with `../geometry/`.

## Dependencies

- **JAX**: Core array operations and automatic differentiation.
- **FLAX NNX**: Modern neural network framework.
- **jaxtyping**: Type annotations for JAX arrays.
- **Python 3.11+**: Modern Python features and type system.
</content>
</invoke>
