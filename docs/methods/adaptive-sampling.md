# Adaptive Sampling for PINNs

Adaptive sampling strategies concentrate collocation points in regions where the PDE residual is high, improving training efficiency by focusing computational resources where they're most needed.

## Overview

Adaptive sampling addresses a fundamental challenge in PINN training:

- **Uniform sampling** wastes resources on well-approximated regions
- **Residual-based sampling** focuses on difficult regions
- **Dynamic refinement** adapts as training progresses

!!! tip "Survey Reference"
    This implementation follows the methodology described in Section 5.2 of the PINN survey (arXiv:2601.10222v1).

## RAD (Residual-based Adaptive Distribution)

RAD samples collocation points with probability proportional to the PDE residual magnitude.

### Sampling Distribution

The sampling probability for each candidate point is:

$$p_j = \frac{|r_j|^\beta}{\sum_k |r_k|^\beta}$$

where:

- $r_j$: PDE residual at point $j$
- $\beta$: Concentration exponent (higher = more focused)

### RADSampler

```python
import jax
import jax.numpy as jnp
from opifex.training.adaptive_sampling import RADSampler, RADConfig

# Configure RAD sampling
config = RADConfig(
    beta=1.0,               # Residual exponent
    resample_frequency=100,  # Steps between resampling
    min_probability=1e-6,    # Minimum sampling probability
    temperature=1.0,         # Probability smoothing
)

sampler = RADSampler(config)

# Domain points (full candidate set)
domain_points = jnp.linspace(0, 1, 1000).reshape(-1, 1)

# Compute PDE residuals
residuals = compute_pde_residual(model, domain_points)

# Sample collocation points
key = jax.random.key(0)
batch = sampler.sample(
    domain_points=domain_points,
    residuals=residuals,
    batch_size=128,
    key=key,
)  # Shape: (128, 1)
```

### Beta Parameter Effect

| Beta Value | Behavior |
|------------|----------|
| $\beta = 0$ | Uniform sampling |
| $\beta = 0.5$ | Mild concentration |
| $\beta = 1.0$ | Linear concentration (default) |
| $\beta = 2.0$ | Strong concentration |
| $\beta > 2$ | Very aggressive focusing |

```python
# Mild concentration (good for smooth problems)
config = RADConfig(beta=0.5)

# Strong concentration (good for sharp features)
config = RADConfig(beta=2.0)
```

### Computing Importance Weights

Instead of resampling, you can weight the loss function:

```python
# Compute importance weights
weights = sampler.compute_weights(residuals)

# Use in loss function
def weighted_loss_fn(model, x, weights):
    residuals = compute_pde_residual(model, x)
    return jnp.sum(weights * residuals ** 2)
```

### Training with RAD

```python
import optax
from flax import nnx

# Setup
model = create_model()
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(nnx.state(model))
sampler = RADSampler(RADConfig(beta=1.0))

# Full domain for residual computation
all_points = generate_domain_points(5000)

for step in range(num_steps):
    key = jax.random.fold_in(jax.random.key(0), step)

    # Periodically update sampling distribution
    if step % sampler.config.resample_frequency == 0:
        residuals = compute_pde_residual(model, all_points)

    # Sample batch based on residuals
    batch = sampler.sample(all_points, residuals, batch_size=256, key=key)

    # Training step
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    nnx.update(model, updates)
```

## RAR-D (Residual-based Adaptive Refinement)

RAR-D progressively adds new collocation points near high-residual regions, increasing resolution where needed.

### RARDRefiner

```python
from opifex.training.adaptive_sampling import RARDRefiner, RARDConfig

# Configure refinement
config = RARDConfig(
    num_new_points=10,          # Points to add per refinement
    percentile_threshold=90.0,  # Focus on top 10% residuals
    noise_scale=0.1,            # Perturbation scale
)

refiner = RARDRefiner(config)

# Initial collocation points
current_points = jnp.linspace(0, 1, 100).reshape(-1, 1)

# Domain bounds
bounds = jnp.array([[0.0, 1.0]])  # Shape: (dim, 2)

# Compute residuals
residuals = compute_pde_residual(model, current_points)

# Refine: add new points near high-residual regions
key = jax.random.key(0)
refined_points = refiner.refine(
    current_points=current_points,
    residuals=residuals,
    bounds=bounds,
    key=key,
)  # Shape: (110, 1) - added 10 new points
```

### Refinement Algorithm

1. **Identify high-residual regions** (above percentile threshold)
2. **Sample base points** from high-residual regions
3. **Add random perturbation** to create new points
4. **Clip to domain bounds**
5. **Concatenate** with existing points

### Training with RAR-D

```python
# Setup
refiner = RARDRefiner(RARDConfig(num_new_points=20))
current_points = generate_initial_points(200)
bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])  # 2D domain

for epoch in range(num_epochs):
    key = jax.random.fold_in(jax.random.key(0), epoch)

    # Train for some steps with current points
    for step in range(steps_per_epoch):
        loss, grads = nnx.value_and_grad(loss_fn)(model, current_points)
        # ... update ...

    # Periodically refine
    if epoch % refine_frequency == 0 and epoch > 0:
        residuals = compute_pde_residual(model, current_points)
        current_points = refiner.refine(
            current_points, residuals, bounds, key
        )
        print(f"Epoch {epoch}: {len(current_points)} points")
```

### Identifying Refinement Regions

```python
# Check which points are in refinement regions
refinement_mask = refiner.identify_refinement_regions(residuals)

# Visualize refinement regions (for debugging)
import matplotlib.pyplot as plt

plt.scatter(
    current_points[~refinement_mask, 0],
    current_points[~refinement_mask, 1],
    c='blue', alpha=0.5, label='Regular'
)
plt.scatter(
    current_points[refinement_mask, 0],
    current_points[refinement_mask, 1],
    c='red', alpha=0.8, label='High residual'
)
plt.legend()
```

## Utility Functions

### Computing Sampling Distribution

```python
from opifex.training.adaptive_sampling import compute_sampling_distribution

residuals = compute_pde_residual(model, points)

# Compute probabilities
probs = compute_sampling_distribution(
    residuals=residuals,
    beta=1.0,
    min_probability=1e-6,
)

# Verify it's a valid distribution
assert jnp.allclose(probs.sum(), 1.0)
assert (probs >= 0).all()
```

## Configuration Reference

### RADConfig

```python
@dataclass(frozen=True)
class RADConfig:
    beta: float = 1.0              # Residual exponent
    resample_frequency: int = 100  # Steps between resampling
    min_probability: float = 1e-6  # Minimum sampling probability
    temperature: float = 1.0       # Probability smoothing
```

### RARDConfig

```python
@dataclass(frozen=True)
class RARDConfig:
    num_new_points: int = 10          # Points to add per refinement
    percentile_threshold: float = 90.0 # Refinement threshold percentile
    noise_scale: float = 0.1          # Perturbation scale (relative to domain)
```

## Best Practices

### RAD vs RAR-D

| Method | Best For | Considerations |
|--------|----------|----------------|
| **RAD** | Continuous refinement, batch training | Fixed point count, resamples existing |
| **RAR-D** | Growing resolution, localized features | Point count grows, may need pruning |

### Choosing Beta

```python
# Start moderate, increase if needed
config = RADConfig(beta=1.0)

# For problems with sharp features (shocks, discontinuities)
config = RADConfig(beta=2.0)

# For smooth problems (prevent over-focusing)
config = RADConfig(beta=0.5)
```

### Resample Frequency

```python
# Frequent resampling (responsive, more overhead)
config = RADConfig(resample_frequency=50)

# Infrequent resampling (stable, less overhead)
config = RADConfig(resample_frequency=500)

# Adaptive: decrease frequency as training progresses
resample_freq = max(50, 500 - step // 10)
```

### Memory Management for RAR-D

```python
# Limit maximum points to control memory
max_points = 5000

if len(current_points) > max_points:
    # Option 1: Stop refining
    pass

    # Option 2: Remove low-residual points
    residuals = compute_pde_residual(model, current_points)
    keep_mask = residuals > jnp.percentile(residuals, 10)
    current_points = current_points[keep_mask]

    # Option 3: Uniform subsampling
    indices = jax.random.choice(key, len(current_points), (max_points,))
    current_points = current_points[indices]
```

## Combining with Other Techniques

### With Domain Decomposition

```python
from opifex.neural.pinns.domain_decomposition import XPINN
from opifex.training.adaptive_sampling import RADSampler

model = XPINN(...)
sampler = RADSampler()

# Sample separately for each subdomain
for subdomain_id in range(len(model.subdomains)):
    subdomain = model.subdomains[subdomain_id]
    subdomain_points = points_in_subdomain(all_points, subdomain)

    # Compute residual for this subdomain's network
    residuals = compute_subdomain_residual(
        model.networks[subdomain_id], subdomain_points
    )

    # Sample for this subdomain
    batch = sampler.sample(subdomain_points, residuals, batch_size, key)
```

### With Multilevel Training

```python
from opifex.training.multilevel import CascadeTrainer

trainer = CascadeTrainer(...)
sampler = RADSampler()

while not trainer.is_at_finest():
    model = trainer.get_current_model()

    # Use adaptive sampling at each level
    for epoch in range(trainer.get_epochs_for_current_level()):
        residuals = compute_pde_residual(model, all_points)
        batch = sampler.sample(all_points, residuals, batch_size, key)

        loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
        # ...

    trainer.advance_level()
```

## Complete Training Example

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from opifex.training.adaptive_sampling import RADSampler, RARDRefiner, RADConfig, RARDConfig

# Create model
class PINN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.net = nnx.List([
            nnx.Linear(2, 64, rngs=rngs),
            nnx.Linear(64, 64, rngs=rngs),
            nnx.Linear(64, 1, rngs=rngs),
        ])

    def __call__(self, x):
        for layer in list(self.net)[:-1]:
            x = nnx.tanh(layer(x))
        return list(self.net)[-1](x)

model = PINN(rngs=nnx.Rngs(0))

# Setup adaptive sampling
rad_sampler = RADSampler(RADConfig(beta=1.0, resample_frequency=100))
rar_refiner = RARDRefiner(RARDConfig(num_new_points=50))

# Domain
bounds = jnp.array([[0.0, 1.0], [0.0, 1.0]])
current_points = jax.random.uniform(jax.random.key(0), (500, 2))

# Optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(nnx.state(model))

# Training
for epoch in range(100):
    key = jax.random.fold_in(jax.random.key(42), epoch)

    # Compute residuals on current point set
    residuals = compute_pde_residual(model, current_points)

    # RAD sampling for this epoch's training
    batch_size = 256
    num_batches = len(current_points) // batch_size

    for batch_idx in range(num_batches):
        batch_key = jax.random.fold_in(key, batch_idx)
        batch = rad_sampler.sample(current_points, residuals, batch_size, batch_key)

        loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        nnx.update(model, updates)

    # Periodic refinement with RAR-D
    if epoch % 20 == 0 and epoch > 0:
        residuals = compute_pde_residual(model, current_points)
        current_points = rar_refiner.refine(
            current_points, residuals, bounds, key
        )
        print(f"Epoch {epoch}: {len(current_points)} points, loss={loss:.4e}")
```

## See Also

- [Training Guide](../user-guide/training.md) - General training procedures
- [Domain Decomposition PINNs](domain-decomposition-pinns.md) - DD-PINN methods
- [Multilevel Training](multilevel-training.md) - Coarse-to-fine training
- [API Reference](../api/training.md#adaptive-sampling) - Complete API documentation
