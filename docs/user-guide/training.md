# Training Infrastructure Guide

## Overview

The Opifex training framework provides comprehensive, production-ready training infrastructure for scientific machine learning models. Built on JAX and FLAX NNX, it supports physics-informed neural networks (PINNs), neural operators, quantum neural networks, and traditional supervised learning with advanced optimization algorithms and physics-aware loss functions.

The training system is designed with modularity and extensibility in mind, featuring component-based architecture, advanced error recovery, and sophisticated metrics collection for scientific computing applications.

## Core Training Components

### Unified Trainer Architecture ⭐ **RECOMMENDED**
```python
from opifex.core.training.trainer import Trainer
from opifex.core.training.config import TrainingConfig
from opifex.core.training.config import QuantumTrainingConfig
from opifex.core.training.physics_configs import ConservationConfig
from opifex.neural import StandardMLP
import jax.numpy as jnp
import jax

# Create model
model = StandardMLP(
    features=[50, 50, 50, 1],
    activation="tanh",
    use_bias=True
)

# Configure physics-aware training with composable configs
conservation_config = ConservationConfig(
    laws=["energy", "momentum"],
    energy_tolerance=1e-6,
    momentum_tolerance=1e-6,
)

quantum_config = QuantumTrainingConfig(
    chemical_accuracy_target=1e-3,
    scf_max_iterations=100,
    enable_symmetry_enforcement=True,
)

config = TrainingConfig(
    num_epochs=1000,
    batch_size=256,
    learning_rate=1e-3,
    validation_frequency=100,
    checkpoint_frequency=100,
    conservation_config=conservation_config,
    quantum_config=quantum_config,
)

# Initialize trainer
trainer = Trainer(model, config)

# Prepare training data
key = jax.random.PRNGKey(42)
x_train = jax.random.uniform(key, (1000, 2), minval=-1.0, maxval=1.0)
y_train = jnp.sin(jnp.pi * x_train[:, 0]) * jnp.cos(jnp.pi * x_train[:, 1])

# Train the model
history = trainer.train(
    train_data=(x_train, y_train),
    validation_data=(x_train[:200], y_train[:200])
)

print(f"Training completed in {len(history['train_losses'])} epochs")
print(f"Final training loss: {history['train_losses'][-1]:.6f}")
```

**Key Advantages:**

- **Composable**: Mix and match physics configurations without modifying trainer code
- **Type-Safe**: Full IDE support with comprehensive type hints
- **Zero Runtime Overhead**: All configuration at initialization
- **Extensible**: Add new configs without changing existing code
- **Well-Tested**: 88 comprehensive tests covering all functionality

**Available Physics Configurations:**

- `ConservationConfig`: Energy, momentum, mass, and symmetry conservation
- `MultiScaleConfig`: Multi-scale physics with adaptive coupling
- `QuantumTrainingConfig`: Quantum chemistry and electronic structure
- `BoundaryConfig`: Boundary condition enforcement
- `DFTConfig`: Density functional theory workflows
- `SCFConfig`: Self-consistent field convergence
- `MetricsTrackingConfig`: Custom metrics tracking
- `LoggingConfig`: Advanced logging and alerting
- `PerformanceConfig`: Performance optimization settings



### Basic Training Infrastructure

The `BasicTrainer` class provides a complete training framework with physics-informed capabilities:

```python
from opifex.training.basic_trainer import BasicTrainer
from opifex.core.training.config import TrainingConfig
from opifex.neural import StandardMLP
import jax.numpy as jnp
import jax

# Create a neural network model
model = StandardMLP(
    features=[50, 50, 50, 1],
    activation="tanh",
    use_bias=True
)

# Configure training parameters
config = TrainingConfig(
    optimizer="adam",
    learning_rate=1e-3,
    num_epochs=1000,
    batch_size=256,
    validation_frequency=100,
    early_stopping_patience=50,
    checkpoint_frequency=100
)

# Initialize trainer
trainer = BasicTrainer(
    model=model,
    training_config=config
)

# Prepare training data
key = jax.random.PRNGKey(42)
x_train = jax.random.uniform(key, (1000, 2), minval=-1.0, maxval=1.0)
y_train = jnp.sin(jnp.pi * x_train[:, 0]) * jnp.cos(jnp.pi * x_train[:, 1])

# Train the model
history = trainer.train(
    train_data=(x_train, y_train),
    validation_data=(x_train[:200], y_train[:200])
)

print(f"Training completed in {len(history.train_losses)} epochs")
print(f"Final training loss: {history.train_losses[-1]:.6f}")
print(f"Final validation loss: {history.val_losses[-1]:.6f}")
```

### Advanced Modular Training Architecture

For complex scientific applications, the `ModularTrainer` provides a component-based architecture:

```python
from opifex.training.basic_trainer import ModularTrainer
from opifex.training.recovery import ErrorRecoveryManager
from opifex.training.components import FlexibleOptimizerFactory
from opifex.training.metrics import AdvancedMetricsCollector

# Configure advanced training components
error_recovery = ErrorRecoveryManager(
    config={
        "max_retries": 3,
        "checkpoint_on_error": True,
        "gradient_clip_threshold": 10.0,
        "loss_explosion_threshold": 1e6,
        "learning_rate": 1e-3
    }
)

optimizer_factory = FlexibleOptimizerFactory(
    config={
        "optimizer_type": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "use_schedule": True,
        "schedule_type": "cosine",
        "total_steps": 10000
    }
)

metrics_collector = AdvancedMetricsCollector()

# Create modular trainer with custom components
modular_trainer = ModularTrainer(
    model=model,
    config=config,
    components={
        "error_recovery": error_recovery,
        "optimizer_factory": optimizer_factory,
        "metrics_collector": metrics_collector
    }
)

# Train with advanced error handling and metrics
advanced_history = modular_trainer.train(
    train_data=(x_train, y_train),
    validation_data=(x_train[:200], y_train[:200])
)

print("Advanced modular training completed with enhanced error recovery")
```

## Physics-Informed Neural Networks (PINNs)

### Basic PINN Training

Physics-informed training incorporates physical laws directly into the loss function:

```python
from opifex.core.physics.losses import PhysicsInformedLoss, PhysicsLossConfig
from opifex.core.problems import PDEProblem
from opifex.core.conditions import DirichletBC

# Define a PDE problem (2D Poisson equation)
class PoissonProblem(PDEProblem):
    def __init__(self):
        domain = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        boundary_conditions = [
            DirichletBC(boundary="left", value=0.0),
            DirichletBC(boundary="right", value=0.0),
            DirichletBC(boundary="top", value=0.0),
            DirichletBC(boundary="bottom", value=0.0)
        ]

        super().__init__(
            domain=domain,
            equation=self._poisson_equation,
            boundary_conditions=boundary_conditions
        )

    def residual(self, x, u, u_derivatives):
        """Poisson equation: ∇²u = f(x,y)"""
        u_xx = u_derivatives["xx"]
        u_yy = u_derivatives["yy"]

        # Source term
        x_coord, y_coord = x[..., 0], x[..., 1]
        source = -2 * jnp.pi**2 * jnp.sin(jnp.pi * x_coord) * jnp.sin(jnp.pi * y_coord)

        return u_xx + u_yy - source

# Configure physics-informed loss
physics_config = PhysicsLossConfig(
    pde_weight=1.0,
    boundary_weight=10.0,
    initial_weight=1.0,
    adaptive_weighting=True
)

physics_loss = PhysicsInformedLoss(config=physics_config)

# Create PINN trainer
pinn_trainer = BasicTrainer(
    model=model,
    training_config=config,
    physics_loss=physics_loss
)

# Generate collocation points for physics loss
poisson_problem = PoissonProblem()
key = jax.random.PRNGKey(123)

# Interior collocation points
x_physics = jax.random.uniform(key, (2000, 2), minval=0.0, maxval=1.0)

# Boundary points
x_boundary = jnp.concatenate([
    jnp.column_stack([jnp.zeros(100), jnp.linspace(0, 1, 100)]),  # Left
    jnp.column_stack([jnp.ones(100), jnp.linspace(0, 1, 100)]),   # Right
    jnp.column_stack([jnp.linspace(0, 1, 100), jnp.zeros(100)]),  # Bottom
    jnp.column_stack([jnp.linspace(0, 1, 100), jnp.ones(100)])    # Top
])
u_boundary = jnp.zeros(len(x_boundary))

# Train PINN
pinn_history = pinn_trainer.train(
    collocation_points=x_physics,
    boundary_data=(x_boundary, u_boundary),
    problem=poisson_problem
)

print(f"PINN training completed")
print(f"Final physics loss: {pinn_history.physics_losses[-1]:.6f}")
print(f"Final boundary loss: {pinn_history.boundary_losses[-1]:.6f}")
```

## Neural Operator Training

### Fourier Neural Operator (FNO) Training

```python
from opifex.neural import FNO
from opifex.training.basic_trainer import BasicTrainer
from opifex.core.training.config import TrainingConfig

# Create FNO model for operator learning
fno_model = FNO(
    modes=[16, 16],  # Fourier modes in each dimension
    width=64,        # Channel width
    n_layers=4,      # Number of Fourier layers
    input_dim=2,     # Input function dimension
    output_dim=1     # Output function dimension
)

# Generate operator training data (input-output function pairs)
def generate_operator_data(n_samples=1000, resolution=64):
    """Generate training data for operator learning."""
    key = jax.random.PRNGKey(456)

    # Input functions (random Gaussian random fields)
    x = jnp.linspace(0, 1, resolution)
    y = jnp.linspace(0, 1, resolution)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    input_functions = []
    output_functions = []

    for i in range(n_samples):
        # Random input function
        key, subkey = jax.random.split(key)
        coeffs = jax.random.normal(subkey, (8, 8))

        input_func = jnp.zeros((resolution, resolution))
        for kx in range(8):
            for ky in range(8):
                input_func += coeffs[kx, ky] * jnp.sin(
                    2 * jnp.pi * kx * X
                ) * jnp.sin(2 * jnp.pi * ky * Y)

        # Corresponding output function (solve PDE)
        output_func = solve_pde_with_input(input_func, X, Y)

        input_functions.append(input_func)
        output_functions.append(output_func)

    return jnp.stack(input_functions), jnp.stack(output_functions)

def solve_pde_with_input(input_func, X, Y):
    """Solve PDE with given input function (simplified)."""
    # This would typically involve a numerical PDE solver
    # For demonstration, we use a simple transformation
    return jnp.fft.fft2(input_func).real

# Generate training data
input_funcs, output_funcs = generate_operator_data(n_samples=500)

# Configure FNO training
fno_config = TrainingConfig(
    optimizer="adam",
    learning_rate=1e-3,
    num_epochs=200,
    batch_size=16,  # Smaller batch size for function data
    validation_frequency=20
)

# Train FNO
fno_trainer = Trainer(model=fno_model, config=fno_config) # Changed from BasicTrainer to Trainer and fixed config argument

fno_history = fno_trainer.train(
    train_data=(input_funcs[:400], output_funcs[:400]),
    validation_data=(input_funcs[400:], output_funcs[400:])
)

print(f"FNO training completed")
print(f"Final training loss: {fno_history.train_losses[-1]:.6f}")
```

### DeepONet Training

```python
from opifex.neural import DeepONet

# Create DeepONet model
deeponet_model = DeepONet(
    branch_net=[100, 100, 100],      # Branch network architecture
    trunk_net=[2, 100, 100, 100],    # Trunk network architecture (2D input)
    output_dim=1                      # Scalar output
)

# Generate DeepONet training data
def generate_deeponet_data(n_samples=1000, n_sensors=100):
    """Generate training data for DeepONet."""
    key = jax.random.PRNGKey(789)

    # Sensor locations (fixed)
    sensor_locations = jnp.linspace(0, 1, n_sensors)

    # Query locations (variable)
    query_locations = jax.random.uniform(key, (n_samples, 2))

    branch_inputs = []  # Function values at sensors
    trunk_inputs = []   # Query coordinates
    outputs = []        # Function values at query points

    for i in range(n_samples):
        # Random function (polynomial)
        key, subkey = jax.random.split(key)
        coeffs = jax.random.normal(subkey, (5,))

        # Function values at sensor locations
        sensor_values = jnp.sum(
            coeffs[:, None] * sensor_locations[None, :]**jnp.arange(5)[:, None],
            axis=0
        )

        # Function value at query location
        query_x, query_y = query_locations[i]
        query_value = jnp.sum(coeffs * query_x**jnp.arange(5)) * jnp.sin(jnp.pi * query_y)

        branch_inputs.append(sensor_values)
        trunk_inputs.append(query_locations[i])
        outputs.append(query_value)

    return (
        jnp.stack(branch_inputs),
        jnp.stack(trunk_inputs),
        jnp.array(outputs)
    )

# Generate DeepONet training data
branch_data, trunk_data, target_data = generate_deeponet_data(n_samples=2000)

# Train DeepONet
deeponet_trainer = BasicTrainer(model=deeponet_model, training_config=fno_config)

deeponet_history = deeponet_trainer.train(
    train_data=((branch_data[:1600], trunk_data[:1600]), target_data[:1600]),
    validation_data=((branch_data[1600:], trunk_data[1600:]), target_data[1600:])
)

print(f"DeepONet training completed")
print(f"Final training loss: {deeponet_history.train_losses[-1]:.6f}")
```

## Advanced Optimization Strategies

### Learning Rate Scheduling

```python
import optax

def create_advanced_scheduler(base_lr=1e-3, total_steps=10000):
    """Create sophisticated learning rate schedule."""

    # Warmup phase
    warmup_steps = int(0.1 * total_steps)
    warmup_schedule = optax.linear_schedule(
        init_value=1e-6,
        end_value=base_lr,
        transition_steps=warmup_steps
    )

    # Cosine annealing with restarts
    cosine_steps = total_steps - warmup_steps
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=cosine_steps,
        alpha=0.1  # Minimum learning rate factor
    )

    # Combine schedules
    combined_schedule = optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps]
    )

    return combined_schedule

# Use advanced scheduling in training
advanced_config = TrainingConfig(
    optimizer="adamw",
    learning_rate=create_advanced_scheduler(base_lr=1e-3, total_steps=5000),
    weight_decay=1e-4,
    num_epochs=100,
    batch_size=64
)
```

### Gradient Clipping and Regularization

```python
class RegularizedTrainer(BasicTrainer):
    """Trainer with advanced regularization techniques."""

    def __init__(self, model, config, regularization_config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        self.reg_config = regularization_config or {}

        # Configure gradient clipping
        self.gradient_clip_value = self.reg_config.get("gradient_clip", 1.0)

        # Regularization weights
        self.l1_weight = self.reg_config.get("l1_weight", 0.0)
        self.l2_weight = self.reg_config.get("l2_weight", 1e-4)
        self.spectral_norm_weight = self.reg_config.get("spectral_norm", 0.0)

    def compute_regularization_loss(self, params):
        """Compute various regularization terms."""
        reg_loss = 0.0

        # L1 regularization
        if self.l1_weight > 0:
            l1_loss = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_leaves(params))
            reg_loss += self.l1_weight * l1_loss

        # L2 regularization
        if self.l2_weight > 0:
            l2_loss = sum(jnp.sum(p**2) for p in jax.tree_leaves(params))
            reg_loss += self.l2_weight * l2_loss

        return reg_loss

# Use regularized training
reg_config = {
    "gradient_clip": 1.0,
    "l1_weight": 1e-5,
    "l2_weight": 1e-4,
    "spectral_norm": 1e-3
}

regularized_trainer = RegularizedTrainer(
    model=model,
    config=config,
    regularization_config=reg_config
)
```

## Monitoring, Visualization, and Checkpointing

### Advanced Metrics Collection

```python
from opifex.training.metrics import AdvancedMetricsCollector
import matplotlib.pyplot as plt

class ComprehensiveMetricsCollector(AdvancedMetricsCollector):
    """Enhanced metrics collection with physics-aware diagnostics."""

    def __init__(self):
        super().__init__()
        self.physics_metrics = {}
        self.convergence_metrics = {}
        self.gradient_metrics = {}

    def collect_physics_metrics(self, params, batch, model, problem=None):
        """Collect physics-specific metrics."""
        if problem is None:
            return

        # Physics residual statistics
        x_physics = batch[0] if len(batch) > 0 else None
        if x_physics is not None:
            def network_fn(x):
                return model.apply(params, x)

            u_pred = network_fn(x_physics)
            u_derivatives = self._compute_derivatives(network_fn, x_physics)
            residuals = problem.residual(x_physics, u_pred, u_derivatives)

            self.physics_metrics.update({
                "residual_mean": jnp.mean(jnp.abs(residuals)),
                "residual_max": jnp.max(jnp.abs(residuals)),
                "residual_std": jnp.std(residuals)
            })

    def collect_gradient_metrics(self, gradients):
        """Collect gradient-based metrics."""
        grad_norms = [jnp.linalg.norm(g) for g in jax.tree_leaves(gradients)]

        self.gradient_metrics.update({
            "grad_norm_mean": jnp.mean(jnp.array(grad_norms)),
            "grad_norm_max": jnp.max(jnp.array(grad_norms)),
            "grad_norm_total": jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_leaves(gradients)))
        })

# Use comprehensive metrics
comprehensive_metrics = ComprehensiveMetricsCollector()
```

### Real-Time Visualization

```python
class TrainingVisualizer:
    """Real-time training visualization."""

    def __init__(self, update_frequency=10):
        self.update_frequency = update_frequency
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.loss_history = {"train": [], "val": [], "physics": [], "boundary": []}
        self.metrics_history = {}

        plt.ion()  # Interactive mode

    def update_plots(self, epoch, current_losses, current_metrics):
        """Update all visualization plots."""
        # Update loss history
        for key, value in current_losses.items():
            if key in self.loss_history:
                self.loss_history[key].append(value)

        # Update metrics history
        for key, value in current_metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        if epoch % self.update_frequency == 0:
            self._redraw_plots(epoch)

# Use visualization during training
visualizer = TrainingVisualizer(update_frequency=5)
```

### Robust Checkpointing System

```python
import orbax.checkpoint as ocp
from pathlib import Path
import time

class AdvancedCheckpointManager:
    """Advanced checkpointing with metadata and recovery."""

    def __init__(self, checkpoint_dir, max_to_keep=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

        # Initialize Orbax checkpoint manager
        self.manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            max_to_keep=max_to_keep,
            item_names=("model_state", "optimizer_state", "metadata")
        )

    def save_checkpoint(self, epoch, model_state, optimizer_state,
                       training_metrics, physics_metrics=None):
        """Save comprehensive checkpoint with metadata."""

        # Prepare metadata
        metadata = {
            "epoch": epoch,
            "training_metrics": training_metrics,
            "physics_metrics": physics_metrics or {},
            "timestamp": time.time(),
            "model_info": {
                "architecture": type(model_state).__name__,
                "parameter_count": sum(
                    p.size for p in jax.tree_leaves(model_state) if hasattr(p, 'size')
                )
            }
        }

        # Save checkpoint
        checkpoint_data = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metadata": metadata
        }

        self.manager.save(epoch, checkpoint_data)

        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, epoch=None):
        """Load checkpoint with automatic recovery."""
        try:
            if epoch is None:
                # Load latest checkpoint
                latest_step = self.manager.latest_step()
                if latest_step is None:
                    return None
                epoch = latest_step

            checkpoint_data = self.manager.restore(epoch)

            print(f"Checkpoint loaded from epoch {epoch}")
            return checkpoint_data

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None

# Use advanced checkpointing
checkpoint_manager = AdvancedCheckpointManager(
    checkpoint_dir="./checkpoints/advanced_training",
    max_to_keep=10
)

print("Comprehensive training infrastructure guide completed")
```

This thorough training guide provides the complete infrastructure for advanced scientific machine learning training. The modular, component-based architecture enables researchers to build sophisticated training workflows while maintaining the flexibility needed for modern scientific applications.

## Advanced Training Techniques

### Multilevel Training

Multilevel training accelerates convergence by training from coarse to fine representations, leveraging multigrid insights for neural network optimization.

```python
from opifex.training.multilevel import CascadeTrainer, MultilevelConfig

# Configure coarse-to-fine training
config = MultilevelConfig(
    num_levels=3,
    coarsening_factor=0.5,
    level_epochs=[100, 200, 500],
)

trainer = CascadeTrainer(
    input_dim=2,
    output_dim=1,
    base_hidden_dims=[64, 64],
    config=config,
    rngs=nnx.Rngs(0),
)

# Train through hierarchy
while not trainer.is_at_finest():
    model = trainer.get_current_model()
    # ... train current level ...
    trainer.advance_level()
```

**Key Benefits:**

- Faster convergence through hierarchical initialization
- Better optimization landscape via progressive capacity
- Natural curriculum from simple to complex representations

For comprehensive details on MLP and FNO hierarchies, see the [Multilevel Training Guide](../methods/multilevel-training.md).

### Adaptive Sampling

Adaptive sampling focuses computational resources on high-residual regions, improving training efficiency for PINNs:

```python
from opifex.training.adaptive_sampling import RADSampler, RADConfig

# Configure residual-based sampling
config = RADConfig(
    beta=1.0,               # Residual exponent
    resample_frequency=100,  # Steps between resampling
)

sampler = RADSampler(config)

# During training
residuals = compute_pde_residual(model, all_points)
batch = sampler.sample(all_points, residuals, batch_size=256, key=key)
```

**Strategies Available:**

- **RAD**: Samples with probability proportional to residual magnitude
- **RAR-D**: Progressively adds points near high-residual regions

For detailed algorithms and best practices, see the [Adaptive Sampling Guide](../methods/adaptive-sampling.md).

### GradNorm Loss Balancing

For multi-task learning with multiple loss terms, GradNorm automatically balances gradient magnitudes:

```python
from opifex.core.physics.gradnorm import GradNormBalancer, GradNormConfig

config = GradNormConfig(
    alpha=1.5,           # Asymmetry parameter
    learning_rate=0.01,  # Weight update rate
)

balancer = GradNormBalancer(num_losses=3, config=config, rngs=nnx.Rngs(0))

# Compute weighted loss
losses = jnp.array([pde_loss, bc_loss, data_loss])
weighted_loss = balancer.compute_weighted_loss(losses)
```

**Benefits:**

- Prevents any single loss from dominating training
- Encourages uniform convergence across all objectives
- Adapts weights dynamically based on training progress

For the complete algorithm and configuration options, see the [GradNorm Guide](../methods/gradnorm.md).

## See Also

- [Multilevel Training](../methods/multilevel-training.md) - Coarse-to-fine training hierarchies
- [Adaptive Sampling](../methods/adaptive-sampling.md) - RAD and RAR-D strategies
- [GradNorm](../methods/gradnorm.md) - Multi-task loss balancing
- [NTK Analysis](../methods/ntk-analysis.md) - Training diagnostics via spectral analysis
- [Second-Order Optimization](../methods/second-order-optimization.md) - L-BFGS and hybrid optimizers
