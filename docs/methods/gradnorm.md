# GradNorm: Multi-Task Loss Balancing

GradNorm automatically balances the contribution of different loss terms in multi-task learning by normalizing gradient magnitudes. This prevents any single loss from dominating training and ensures balanced convergence across all objectives.

## Overview

Physics-informed neural networks often combine multiple loss terms:

- **PDE residual loss** (physics enforcement)
- **Boundary condition loss** (spatial constraints)
- **Initial condition loss** (temporal constraints)
- **Data loss** (observed measurements)

These losses can have vastly different magnitudes and gradient norms, causing training imbalances. GradNorm addresses this automatically.

!!! tip "Survey Reference"
    This implementation follows the methodology described in Section 2.2.2 of the PINN survey (arXiv:2601.10222v1).

## Theoretical Foundation

### The Gradient Dominance Problem

When training with multiple losses $L = \sum_i w_i L_i$:

- Losses with larger gradients $\|\nabla_\theta L_i\|$ dominate parameter updates
- Smaller-gradient losses converge slowly or not at all
- Manual weight tuning is tedious and problem-specific

### GradNorm Algorithm

GradNorm adjusts weights to equalize gradient contributions:

$$L_{grad} = \sum_i \left| \|w_i \nabla_\theta L_i\| - \bar{G} \cdot r_i^\alpha \right|$$

where:

- $\bar{G} = \frac{1}{n}\sum_j \|w_j \nabla_\theta L_j\|$: Average weighted gradient norm
- $r_i = \frac{L_i(t)}{L_i(0)}$: Relative inverse training rate
- $\alpha$: Asymmetry parameter controlling task prioritization

### Training Rate Balancing

- Tasks training **slower** (higher $r_i$) get **larger** target gradients
- Tasks training **faster** (lower $r_i$) get **smaller** target gradients
- This encourages all tasks to converge at similar rates

## GradNormBalancer

### Basic Usage

```python
from flax import nnx
from opifex.core.physics.gradnorm import GradNormBalancer, GradNormConfig

# Configure GradNorm
config = GradNormConfig(
    alpha=1.5,           # Asymmetry parameter
    learning_rate=0.01,  # Weight update rate
    update_frequency=1,  # Update weights every step
    min_weight=0.01,     # Minimum allowed weight
    max_weight=100.0,    # Maximum allowed weight
)

# Create balancer for 3 losses
balancer = GradNormBalancer(
    num_losses=3,
    config=config,
    rngs=nnx.Rngs(0),
)

# Compute individual losses
losses = jnp.array([pde_loss, bc_loss, data_loss])

# Get weighted total loss
weighted_loss = balancer.compute_weighted_loss(losses)

# Access current weights
print(f"Weights: {balancer.weights}")
```

### Training Loop Integration

```python
import optax

model = create_pinn_model()
balancer = GradNormBalancer(num_losses=3, rngs=nnx.Rngs(0))

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(nnx.state(model))

# Define individual loss functions
def compute_losses(model, x_pde, x_bc, x_data, y_data):
    pde_loss = compute_pde_residual_loss(model, x_pde)
    bc_loss = compute_boundary_loss(model, x_bc)
    data_loss = compute_data_loss(model, x_data, y_data)
    return jnp.array([pde_loss, bc_loss, data_loss])

# Training loop
for step in range(num_steps):
    # Compute individual losses
    losses = compute_losses(model, x_pde, x_bc, x_data, y_data)

    # Initialize on first step
    if step == 0:
        balancer.set_initial_losses(losses)

    # Get weighted loss for gradient computation
    def total_loss_fn(model):
        losses = compute_losses(model, x_pde, x_bc, x_data, y_data)
        return balancer.compute_weighted_loss(losses)

    # Training step
    loss, grads = nnx.value_and_grad(total_loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    nnx.update(model, updates)

    # Update GradNorm weights
    if step % balancer.config.update_frequency == 0:
        grad_norms = compute_gradient_norms_manual(model, x_pde, x_bc, x_data, y_data)
        balancer.update_weights(grad_norms, losses, balancer.get_initial_losses())

    if step % 100 == 0:
        print(f"Step {step}: loss={loss:.4e}, weights={balancer.weights}")
```

### Computing Gradient Norms

```python
from opifex.core.physics.gradnorm import compute_gradient_norms

# Define loss functions (each takes model, returns scalar)
loss_fns = [
    lambda m: compute_pde_loss(m, x_pde),
    lambda m: compute_bc_loss(m, x_bc),
    lambda m: compute_data_loss(m, x_data, y_data),
]

# Compute gradient norms for each loss
grad_norms = compute_gradient_norms(model, loss_fns)
# Shape: (3,) - one norm per loss
```

## Configuration

### GradNormConfig

```python
@dataclass(frozen=True)
class GradNormConfig:
    alpha: float = 1.5           # Asymmetry parameter
    learning_rate: float = 0.01  # Weight update learning rate
    update_frequency: int = 1    # Steps between weight updates
    min_weight: float = 0.01     # Minimum weight bound
    max_weight: float = 100.0    # Maximum weight bound
```

### Alpha Parameter

The asymmetry parameter $\alpha$ controls task prioritization:

| Alpha Value | Behavior |
|-------------|----------|
| $\alpha = 0$ | Equal target gradients for all tasks |
| $\alpha = 1$ | Linear scaling with training rate |
| $\alpha = 1.5$ | Moderate prioritization (default) |
| $\alpha = 2$ | Strong prioritization of slow tasks |

```python
# Equal treatment of all losses
config = GradNormConfig(alpha=0.0)

# Moderate prioritization (recommended starting point)
config = GradNormConfig(alpha=1.5)

# Strong emphasis on slow-converging losses
config = GradNormConfig(alpha=2.0)
```

## Utility Functions

### Inverse Training Rates

```python
from opifex.core.physics.gradnorm import compute_inverse_training_rates

# Current and initial losses
current_losses = jnp.array([1e-3, 1e-2, 1e-4])
initial_losses = jnp.array([1.0, 0.5, 0.1])

# Compute relative rates (normalized to mean 1)
rates = compute_inverse_training_rates(current_losses, initial_losses)
# rates[i] > 1: task i is training slower than average
# rates[i] < 1: task i is training faster than average
```

### Manual GradNorm Loss Computation

```python
from opifex.core.physics.gradnorm import GradNormBalancer

# For custom training loops
def compute_gradnorm_loss_manual(balancer, grad_norms, losses, initial_losses):
    return balancer.compute_gradnorm_loss(grad_norms, losses, initial_losses)
```

## Best Practices

### Initial Loss Storage

Always set initial losses at the start of training:

```python
# At step 0
if step == 0:
    losses = compute_losses(model)
    balancer.set_initial_losses(losses)
```

### Weight Clamping

GradNorm includes automatic weight clamping:

```python
# Prevent extreme weights
config = GradNormConfig(
    min_weight=0.1,   # Don't let any loss become negligible
    max_weight=10.0,  # Don't let any loss dominate
)
```

### Update Frequency

```python
# Every step (most responsive)
config = GradNormConfig(update_frequency=1)

# Every 10 steps (smoother, less overhead)
config = GradNormConfig(update_frequency=10)

# Recommendations:
# - Use frequency=1 for small batches
# - Use frequency=10-50 for large batches
```

### Monitoring Weights

```python
# Track weight evolution
weight_history = []

for step in range(num_steps):
    # ... training ...
    weight_history.append(balancer.weights.copy())

# Analyze
import matplotlib.pyplot as plt

weights = jnp.array(weight_history)
for i, name in enumerate(['PDE', 'BC', 'Data']):
    plt.plot(weights[:, i], label=name)
plt.legend()
plt.xlabel('Step')
plt.ylabel('Weight')
plt.title('GradNorm Weight Evolution')
```

## Combining with Other Techniques

### With Adaptive Sampling

```python
from opifex.training.adaptive_sampling import RADSampler

sampler = RADSampler()
balancer = GradNormBalancer(num_losses=3, rngs=nnx.Rngs(0))

for step in range(num_steps):
    # Adaptive sampling for PDE points
    residuals = compute_pde_residual(model, all_points)
    pde_batch = sampler.sample(all_points, residuals, batch_size, key)

    # Compute losses with adaptively sampled PDE points
    pde_loss = jnp.mean(compute_pde_residual(model, pde_batch) ** 2)
    bc_loss = compute_bc_loss(model, x_bc)
    data_loss = compute_data_loss(model, x_data, y_data)

    losses = jnp.array([pde_loss, bc_loss, data_loss])
    weighted_loss = balancer.compute_weighted_loss(losses)
    # ...
```

### With Multilevel Training

```python
from opifex.training.multilevel import CascadeTrainer

trainer = CascadeTrainer(...)

while not trainer.is_at_finest():
    model = trainer.get_current_model()

    # Reset balancer for each level
    balancer = GradNormBalancer(num_losses=3, rngs=nnx.Rngs(0))

    for epoch in range(trainer.get_epochs_for_current_level()):
        losses = compute_losses(model)

        if epoch == 0:
            balancer.set_initial_losses(losses)

        weighted_loss = balancer.compute_weighted_loss(losses)
        # ... train ...

    trainer.advance_level()
```

### With Domain Decomposition

```python
from opifex.neural.pinns.domain_decomposition import XPINN

model = XPINN(...)

# Balance subdomain losses + interface losses
num_losses = len(model.subdomains) + 2  # subdomains + continuity + flux
balancer = GradNormBalancer(num_losses=num_losses, rngs=nnx.Rngs(0))

def compute_all_losses(model):
    # Per-subdomain PDE losses
    subdomain_losses = [
        model.compute_subdomain_residual(i, pde_residual_fn, points[i])
        for i in range(len(model.subdomains))
    ]

    # Interface losses
    continuity_loss = model.compute_continuity_loss()
    flux_loss = model.compute_flux_loss()

    return jnp.array([*subdomain_losses, continuity_loss, flux_loss])
```

## Complete Training Example

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from opifex.core.physics.gradnorm import GradNormBalancer, GradNormConfig

# Model
class PINN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.layers = nnx.List([
            nnx.Linear(2, 64, rngs=rngs),
            nnx.Linear(64, 64, rngs=rngs),
            nnx.Linear(64, 1, rngs=rngs),
        ])

    def __call__(self, x):
        for layer in list(self.layers)[:-1]:
            x = nnx.tanh(layer(x))
        return list(self.layers)[-1](x)

model = PINN(rngs=nnx.Rngs(0))

# Loss functions
def pde_loss_fn(model, x):
    def u_fn(xi):
        return model(xi.reshape(1, -1)).squeeze()
    laplacian = jax.vmap(lambda xi: jnp.trace(jax.hessian(u_fn)(xi)))(x)
    return jnp.mean(laplacian ** 2)

def bc_loss_fn(model, x_bc, u_bc):
    pred = model(x_bc)
    return jnp.mean((pred - u_bc) ** 2)

def data_loss_fn(model, x_data, u_data):
    pred = model(x_data)
    return jnp.mean((pred - u_data) ** 2)

# Setup
config = GradNormConfig(alpha=1.5, learning_rate=0.01)
balancer = GradNormBalancer(num_losses=3, config=config, rngs=nnx.Rngs(0))

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(nnx.state(model))

# Training
for step in range(5000):
    # Compute individual losses
    losses = jnp.array([
        pde_loss_fn(model, x_pde),
        bc_loss_fn(model, x_bc, u_bc),
        data_loss_fn(model, x_data, u_data),
    ])

    if step == 0:
        balancer.set_initial_losses(losses)

    # Weighted total loss
    def total_loss(m):
        l = jnp.array([
            pde_loss_fn(m, x_pde),
            bc_loss_fn(m, x_bc, u_bc),
            data_loss_fn(m, x_data, u_data),
        ])
        return balancer.compute_weighted_loss(l)

    # Training step
    loss, grads = nnx.value_and_grad(total_loss)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    nnx.update(model, updates)

    # Update GradNorm weights
    loss_fns = [
        lambda m: pde_loss_fn(m, x_pde),
        lambda m: bc_loss_fn(m, x_bc, u_bc),
        lambda m: data_loss_fn(m, x_data, u_data),
    ]
    grad_norms = jnp.array([
        jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(
            nnx.value_and_grad(fn)(model)[1]
        )))
        for fn in loss_fns
    ])
    balancer.update_weights(grad_norms, losses, balancer.get_initial_losses())

    if step % 500 == 0:
        print(f"Step {step}")
        print(f"  Losses: PDE={losses[0]:.4e}, BC={losses[1]:.4e}, Data={losses[2]:.4e}")
        print(f"  Weights: {balancer.weights}")
        print(f"  Total: {loss:.4e}")
```

## See Also

- [Training Guide](../user-guide/training.md) - General training procedures
- [NTK Analysis](ntk-analysis.md) - Training diagnostics
- [Second-Order Optimization](second-order-optimization.md) - Advanced optimizers
- [API Reference](../api/physics.md#gradnorm) - Complete API documentation
