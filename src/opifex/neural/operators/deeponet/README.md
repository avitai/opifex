# DeepONet: Deep Operator Networks

This module provides comprehensive DeepONet implementations for learning operators between function spaces using the branch-trunk architecture.

## Available Implementations

### Core Components

- **`base.py`**: Core DeepONet architecture with `BranchNet`, `TrunkNet`, and `DeepONet` classes
- **`__init__.py`**: Module exports for easy imports

### Enhanced Variants

- **`enhanced.py`**: `FourierEnhancedDeepONet` with spectral processing for improved frequency domain representation
- **`adaptive.py`**: `AdaptiveDeepONet` with learnable sensor placement and adaptive attention mechanisms
- **`multiphysics.py`**: `MultiPhysicsDeepONet` for coupled physics systems with physics-aware attention

## Quick Usage

```python
from flax import nnx
import jax

# Basic DeepONet
from opifex.neural.operators.deeponet import DeepONet

rngs = nnx.Rngs(jax.random.PRNGKey(42))
deeponet = DeepONet(
    branch_input_dim=100,
    trunk_input_dim=2,
    branch_hidden_dims=[128, 128],
    trunk_hidden_dims=[64, 64],
    latent_dim=128,
    rngs=rngs
)

# Multi-physics enhanced variant
from opifex.neural.operators.deeponet import MultiPhysicsDeepONet

multi_deeponet = MultiPhysicsDeepONet(
    branch_input_dim=100,
    trunk_input_dim=3,
    branch_hidden_dims=[128, 128],
    trunk_hidden_dims=[64, 64],
    latent_dim=128,
    num_physics_systems=3,
    use_attention=True,
    num_sensors=20,  # Required when sensor_optimization=True (default)
    rngs=rngs
)
```

For detailed documentation and examples, see the main [Neural Operators README](../README.md).
