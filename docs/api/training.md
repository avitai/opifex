# Training API Documentation

## Overview

The `opifex.training` module provides comprehensive training infrastructure for scientific machine learning, including physics-informed neural networks, optimization algorithms, and quantum-aware training workflows.

**Module Structure:**

- `opifex.core.training.trainer` - **Unified Trainer (Recommended)**
- `opifex.core.training.config` - Training configuration classes
- `opifex.core.training.physics_configs` - Physics-specific configurations
- `opifex.training.basic_trainer` - Core trainer implementations
- `opifex.training.metrics` - Metrics tracking and state management
- `opifex.training.recovery` - Error recovery and stability handling
- `opifex.training.components` - Modular training components
- `opifex.training.utils` - Utility functions for safe model operations

## Core Classes

### Trainer ⭐ **RECOMMENDED**

The unified, composable trainer architecture for all training workflows.

```python
from opifex.core.training.trainer import Trainer
from opifex.core.training.config import TrainingConfig
from opifex.core.training.physics_configs import ConservationConfig, MultiScaleConfig

# Configure physics-aware training
conservation_config = ConservationConfig(
    laws=["energy", "momentum"],
    energy_tolerance=1e-6,
)

multiscale_config = MultiScaleConfig(
    scales=["molecular", "atomic"],
    weights={"molecular": 0.5, "atomic": 0.5},
)

config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    conservation_config=conservation_config,
    multiscale_config=multiscale_config,
)

# Create and use trainer
# Create a dummy model for demonstration
class SimpleModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(10, 1, rngs=rngs)
    def __call__(self, x):
        return self.linear(x)

model = SimpleModel(rngs=nnx.Rngs(0))
trainer = Trainer(model, config)
trained_model, history = trainer.train(train_data, val_data)
```

**Key Features:**

- **Composable Architecture**: Mix and match physics configurations
- **Type-Safe**: Full type hints and IDE support
- **Zero Runtime Overhead**: Configuration at initialization only
- **Extensible**: Add custom configs without modifying trainer
- **Production-Ready**: Comprehensive testing and error handling

**Supported Physics Configurations:**

- `ConservationConfig`: Energy, momentum, mass, and symmetry conservation
- `MultiScaleConfig`: Multi-scale physics with coupling
- `QuantumTrainingConfig`: Quantum chemistry and electronic structure
- `BoundaryConfig`: Boundary condition enforcement
- `DFTConfig`: Density functional theory workflows
- `SCFConfig`: Self-consistent field convergence
- `MetricsTrackingConfig`: Custom metrics tracking
- `LoggingConfig`: Advanced logging and alerting



### BasicTrainer

Standard training workflow with physics-informed capabilities.

```python
from opifex.training.basic_trainer import BasicTrainer
from opifex.core.training.config import TrainingConfig

trainer = BasicTrainer(model, config)
trained_model, history = trainer.train(train_data, val_data)
```

**Key Features:**

- Physics-informed neural network (PINN) training
- Orbax-compatible checkpointing
- JAX Array and automatic differentiation support
- Type-safe with jaxtyping annotations

### ModularTrainer ✅ **NEW**

Component-based training architecture with production-grade capabilities.

```python
from opifex.training.basic_trainer import ModularTrainer
from opifex.core.training.config import TrainingConfig
from opifex.training.recovery import ErrorRecoveryManager
from opifex.training.components import FlexibleOptimizerFactory

trainer = ModularTrainer(
    model=model,
    config=config,
    rngs=rngs,
    components={
        "error_recovery": ErrorRecoveryManager(),
        "optimizer_factory": FlexibleOptimizerFactory()
    }
)
```

**Key Features:**

- Component composition architecture
- Pluggable training components
- Production-grade error handling
- Optimization strategies
- Physics-aware metrics collection

## Components

### ErrorRecoveryManager ✅ **NEW**

Production-grade error handling with gradient stability and automatic recovery.

```python
from opifex.training.recovery import ErrorRecoveryManager

error_manager = ErrorRecoveryManager(
    config={
        "max_retries": 5,
        "gradient_clip_threshold": 1.0,
        "loss_explosion_threshold": 100.0,
        "checkpoint_on_error": True
    }
)
```

**Features:**

- Gradient clipping with automatic threshold adaptation
- NaN detection and recovery mechanisms
- Loss explosion detection and mitigation
- Multiple recovery strategies (gradient clipping, learning rate reduction, parameter reinitialization)
- Comprehensive error logging and analytics

### FlexibleOptimizerFactory ✅ **NEW**

Optimizer creation with scheduling support.

```python
from opifex.training.components import FlexibleOptimizerFactory

optimizer_factory = FlexibleOptimizerFactory(
    config={
        "optimizer_type": "adamw",  # "adam", "adamw", "sgd"
        "learning_rate": 1e-3,
        "schedule_type": "cosine",  # "cosine", "exponential", "linear"
        "total_steps": 1000,
        "cosine_alpha": 0.0
    }
)
```

**Supported Optimizers:**

- **Adam**: Adaptive moment estimation
- **AdamW**: Adam with weight decay
- **SGD**: Stochastic gradient descent with momentum

**Supported Schedules:**

- **Cosine**: Cosine annealing learning rate decay
- **Exponential**: Exponential decay
- **Linear**: Linear decay

### MetricsCollector ✅ **NEW**

Physics-aware metrics collection with convergence tracking.

```python
from opifex.training.metrics import AdvancedMetricsCollector

collector = AdvancedMetricsCollector()
collector.start_training()
metrics = collector.collect_physics_metrics(model, x, y_true)
```

**Collected Metrics:**

- Training loss and validation metrics
- Gradient norms and stability indicators
- Physics-specific metrics (energy conservation, mass conservation)
- Convergence rates and training diagnostics
- Real-time performance analytics

### TrainingComponentBase ✅ **NEW**

Base class for creating custom training components.

```python
from opifex.training.components import TrainingComponentBase

class CustomComponent(TrainingComponentBase):
    def initialize(self, **kwargs):
        # Initialize component
        pass

    def update(self, **kwargs):
        # Update component state
        pass
```

**Purpose:**

- Enables modular component development
- Provides common interface for training components
- Supports pluggable architecture patterns

## Configuration Classes

### TrainingConfig

Training configuration with comprehensive parameter control.

```python
from opifex.core.training.config import TrainingConfig

config = TrainingConfig(
    num_epochs=1000,
    batch_size=64,
    learning_rate=1e-3,
    validation_frequency=100,
    checkpoint_frequency=500,
    early_stopping=True,
    patience=50
)
```

**Parameters:**

- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate (can be overridden by optimizer factory)
- `validation_frequency`: Validation evaluation frequency
- `checkpoint_frequency`: Model checkpointing frequency
- `early_stopping`: Enable early stopping
- `patience`: Early stopping patience

### TrainingState

Enhanced training state with comprehensive tracking.

```python
from opifex.training.metrics import TrainingState

# Automatically managed by trainers
state = trainer.training_state
print(f"Current epoch: {state.epoch}")
print(f"Best validation loss: {state.best_val_loss}")
```

**Tracked Information:**

- Current epoch and step counters
- Best validation metrics
- Model and optimizer states
- Recovery attempt history
- Training diagnostics

## Physics-Informed Training

### PhysicsInformedLoss

Hierarchical multi-physics loss composition with adaptive weighting.

```python
from opifex.training import PhysicsInformedLoss, PhysicsLossConfig

physics_loss = PhysicsInformedLoss(
    config=PhysicsLossConfig(
        physics_weight=1.0,
        boundary_weight=1.0,
        data_weight=1.0,
        adaptive_weighting=True
    )
)

# Use with BasicTrainer
trainer.set_physics_loss(physics_loss)
```

**Supported Physics:**

- Partial differential equations (PDEs)
- Conservation laws (mass, momentum, energy)
- Quantum mechanical constraints
- Boundary condition enforcement

## Usage Examples

### Basic Training Workflow

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.base import StandardMLP
from opifex.training.basic_trainer import BasicTrainer
from opifex.core.training.config import TrainingConfig

# Create model
model = StandardMLP([1, 32, 32, 1], activation="tanh", rngs=nnx.Rngs(42))

# Configure training
config = TrainingConfig(num_epochs=1000, batch_size=64, learning_rate=1e-3)

# Create trainer and train
trainer = BasicTrainer(model, config)
trained_model, history = trainer.train(train_data, val_data)
```

### Modular Training

```python
from opifex.training.basic_trainer import ModularTrainer
from opifex.core.training.config import TrainingConfig
from opifex.training.recovery import ErrorRecoveryManager
from opifex.training.components import FlexibleOptimizerFactory

# Configure components
error_recovery = ErrorRecoveryManager(config={"max_retries": 5, "gradient_clip_threshold": 1.0})
optimizer_factory = FlexibleOptimizerFactory(config={"optimizer_type": "adamw", "schedule_type": "cosine"})

# Create modular trainer
# Note: AdvancedMetricsCollector is automatically created by ModularTrainer
trainer = ModularTrainer(
    model=model,
    config=config,
    rngs=rngs,
    components={
        "error_recovery": error_recovery,
        "optimizer_factory": optimizer_factory
    }
)

# Train with capabilities
trained_model, history = trainer.train(train_data, val_data)
```

### Physics-Informed Training

```python
from opifex.training.basic_trainer import BasicTrainer
from opifex.core.physics.losses import PhysicsInformedLoss

# Define PDE residual
def pde_residual(model_fn, x, t):
    u = model_fn(jnp.array([x, t]).reshape(1, -1))
    # Compute PDE residual (example: heat equation)
    u_t = jax.grad(lambda t: model_fn(jnp.array([x, t]).reshape(1, -1)))(t)
    u_xx = jax.grad(jax.grad(lambda x: model_fn(jnp.array([x, t]).reshape(1, -1))))(x)
    return u_t - 0.1 * u_xx

# Configure physics loss
physics_loss = PhysicsInformedLoss(pde_residual=pde_residual)

# Set up PINN training
trainer = BasicTrainer(model, config)
trainer.set_physics_loss(physics_loss)

# Train with physics constraints
trained_model, history = trainer.train(
    train_data=(domain_points, None),  # No target data for domain points
    boundary_data=(boundary_points, boundary_values)
)
```

## Integration

### With Neural Networks

```python
from opifex.neural.base import StandardMLP
from opifex.neural.quantum import QuantumMLP

# Standard networks
standard_model = StandardMLP([3, 64, 64, 1], activation="swish", rngs=rngs)

# Quantum networks
quantum_model = QuantumMLP(features=[128, 128, 1], n_atoms=3, rngs=rngs)
```

### With Optimization

```python
from opifex.optimization import MetaOptimizer

# Use with learn-to-optimize
meta_optimizer = MetaOptimizer()
trainer = BasicTrainer(model, config, meta_optimizer=meta_optimizer)
```

### With Geometry

```python
from opifex.geometry import ComplexDomain
from opifex.core.conditions import DirichletBC

# Complex domain training
domain = ComplexDomain(boundaries=["left", "right", "top", "bottom"])
boundary_conditions = [DirichletBC(boundary="left", value=0.0)]
```

## Best Practices

### Production Training

1. **Use ModularTrainer** for production workflows with error recovery
2. **Configure appropriate error recovery** strategies for your problem
3. **Monitor training metrics** with MetricsCollector
4. **Use learning rate scheduling** for better convergence
5. **Enable checkpointing** for long training runs

### Physics-Informed Training

1. **Balance loss weights** between physics, boundary, and data terms
2. **Use adaptive weighting** for complex multi-physics problems
3. **Monitor conservation** laws during training
4. **Validate physics** constraints on test data

### Performance Optimization

1. **Choose appropriate batch sizes** for your hardware
2. **Use JAX transformations** (vmap, jit) for efficiency
3. **Profile training** with JAX profiling tools
4. **Monitor gradient health** and stability

## Troubleshooting

### Common Issues

- **NaN losses**: Enable NaN detection in ErrorRecoveryManager
- **Gradient explosions**: Use gradient clipping with appropriate thresholds
- **Slow convergence**: Try different optimizers and learning rate schedules
- **Physics constraint violations**: Increase physics loss weights or improve residual computation

### Debug Features

- **Comprehensive logging** of training metrics and errors
- **Recovery attempt tracking** for debugging stability issues
- **Gradient norm monitoring** for optimization health
- **Physics constraint validation** for PINN problems

## Multilevel Training {: #multilevel }

Coarse-to-fine training hierarchies for accelerated convergence.

### Width-Based Hierarchy (MLPs)

::: opifex.training.multilevel.coarse_to_fine
    options:
        show_root_heading: true
        show_source: false
        members:
            - CascadeTrainer
            - MultilevelConfig
            - create_network_hierarchy
            - prolongate
            - restrict

### Mode-Based Hierarchy (FNOs)

::: opifex.training.multilevel.multilevel_fno
    options:
        show_root_heading: true
        show_source: false
        members:
            - MultilevelFNOTrainer
            - MultilevelFNOConfig
            - create_fno_hierarchy
            - create_mode_hierarchy
            - prolongate_fno_modes
            - restrict_fno_modes

For usage examples and best practices, see the [Multilevel Training Guide](../methods/multilevel-training.md).

## Adaptive Sampling {: #adaptive-sampling }

Residual-based sampling strategies for efficient PINN training.

::: opifex.training.adaptive_sampling
    options:
        show_root_heading: true
        show_source: false
        members:
            - RADSampler
            - RADConfig
            - RARDRefiner
            - RARDConfig
            - compute_sampling_distribution

For detailed algorithms and best practices, see the [Adaptive Sampling Guide](../methods/adaptive-sampling.md).
