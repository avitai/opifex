# Physics-Aware Neural Operators

This module provides physics-aware components for neural operator learning, incorporating physics constraints and domain knowledge into operator learning architectures.

## Available Components

### Attention Mechanisms

- **`attention.py`**: Physics-aware attention mechanisms with `PhysicsAwareAttention` and `PhysicsCrossAttention` classes
- **`__init__.py`**: Module exports for easy imports

### Physics-Informed Operators

- **`informed.py`**: `PhysicsInformedNeuralOperator` with built-in physics constraint enforcement

## Quick Usage

```python
from flax import nnx
import jax

# Physics-aware attention
from opifex.neural.operators.physics import PhysicsAwareAttention

rngs = nnx.Rngs(jax.random.PRNGKey(42))
physics_attention = PhysicsAwareAttention(
    embed_dim=128,
    num_heads=8,
    physics_constraints=['energy_conservation', 'momentum_conservation'],
    rngs=rngs
)

# Physics-informed neural operator
from opifex.neural.operators.physics import PhysicsInformedNeuralOperator

pino = PhysicsInformedNeuralOperator(
    in_channels=2,
    out_channels=1,
    hidden_channels=64,
    modes=16,
    physics_constraints=['conservation_laws'],
    rngs=rngs
)
```

For detailed documentation and examples, see the main [Neural Operators README](../README.md).
