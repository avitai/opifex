# Multilevel Training

Multilevel training leverages multigrid insights to accelerate neural network convergence by training from coarse to fine representations. This approach captures low-frequency features quickly on coarse networks, then refines high-frequency details on finer networks.

## Overview

Multilevel training offers significant benefits:

- **Faster convergence** through hierarchical initialization
- **Better optimization landscape** via coarse-to-fine progression
- **Reduced overfitting** risk from progressive capacity
- **Natural curriculum** from simple to complex representations

!!! tip "Survey Reference"
    This implementation follows the methodology described in Section 8.2 of the PINN survey (arXiv:2601.10222v1).

## Width-Based Hierarchy (MLPs)

For standard MLPs, the hierarchy is based on network width (number of neurons per layer).

### CascadeTrainer

The `CascadeTrainer` provides a generic framework for multilevel training, supporting any model hierarchy and optimizer.

```python
from flax import nnx
import optax
from opifex.training.multilevel import (
    CascadeTrainer,
    create_network_hierarchy,
    prolongate,
    MultilevelAdam,
)

# 1. Create hierarchy (List of models from coarse to fine)
hierarchy = create_network_hierarchy(
    input_dim=2,
    output_dim=1,
    base_hidden_dims=[64, 64],
    num_levels=3,
    coarsening_factor=0.5,
    rngs=nnx.Rngs(0),
)

# 2. Create Multilevel Optimizer
# MultilevelAdam automatically handles state resizing during level transitions
optimizer = MultilevelAdam(learning_rate=1e-3)

# 3. Create Trainer
trainer = CascadeTrainer(
    hierarchy=hierarchy,
    optimizer=optimizer,
    prolongate_fn=prolongate,
)

# Current model (Level 0 - Coarsest)
model = trainer.model
```

### Multilevel Optimization

Standard optimizers (like Adam) maintain state (momentum, variance) that corresponds to specific parameters. When moving from a coarse model to a fine model, this state must be **prolongated** to match the new parameter shapes.

Opifex provides `MultilevelAdam`, a specialized optimizer that wraps `optax.adam` and handles this transition automatically.

```python
# During training
optimizer.update(model, grads)

# When advancing level:
# trainer.advance_level() automatically calls:
# optimizer.resize_state(new_model, transition_fn)
```

### Training Loop

```python
# Iterate until finest level is completed
while True:
    model = trainer.model
    level = trainer.current_level_index

    print(f"Training Level {level}")

    # Train for some epochs
    for epoch in range(100):
        grads = nnx.grad(loss_fn)(model, batch)
        # Update model and optimizer state
        trainer.step(grads)

    if trainer.is_at_finest:
        break

    # Advance to next level (automatically prolongates model and optimizer state)
    trainer.advance_level()
```

### Transfer Operators

Transfer operators move parameters between hierarchy levels.

```python
from opifex.training.multilevel import prolongate, restrict

# Prolongate: coarse -> fine (copy and pad)
fine_model = prolongate(coarse_model, fine_model)

# Restrict: fine -> coarse (truncate)
coarse_model = restrict(fine_model, coarse_model)
```

**Prolongation:** Copies coarse parameters to corresponding fine parameters, leaving additional fine parameters at initialization.

**Restriction:** Extracts a subset of fine parameters for the coarse model.

### Creating Custom Hierarchies

You can use `create_network_hierarchy` or manually create a list of models.

```python
from opifex.training.multilevel import create_network_hierarchy

hierarchy = create_network_hierarchy(
    input_dim=2,
    output_dim=1,
    base_hidden_dims=[128, 128],
    num_levels=4,
    coarsening_factor=0.5,
    activation=nnx.gelu,  # Custom activation
    rngs=nnx.Rngs(0),
)

# hierarchy[0]: smallest network (coarsest)
# hierarchy[-1]: largest network (finest)
```

## Mode-Based Hierarchy (FNOs)

For Fourier Neural Operators, the hierarchy is based on the number of Fourier modes retained.

### FNO Training Example

```python
from opifex.training.multilevel import (
    create_fno_hierarchy,
    prolongate_fno_modes,
)

# 1. Create FNO hierarchy
fno_hierarchy = create_fno_hierarchy(
    base_modes=16,
    width=64,
    num_levels=3,
    reduction_factor=2,
    rngs=nnx.Rngs(0),
    # ... other args
)

# 2. Use generic CascadeTrainer with FNO-specific transfer
trainer = CascadeTrainer(
    hierarchy=fno_hierarchy,
    optimizer=MultilevelAdam(1e-3),
    prolongate_fn=prolongate_fno_modes,
)
```


## Best Practices

### Choosing Number of Levels

| Problem Complexity | Recommended Levels |
|-------------------|-------------------|
| Simple (smooth solutions) | 2-3 |
| Moderate | 3-4 |
| Complex (multi-scale) | 4-5 |

```python
# Simple problem: few levels, aggressive coarsening
config = MultilevelConfig(
    num_levels=2,
    coarsening_factor=0.5,
)

# Complex problem: more levels, gradual refinement
config = MultilevelConfig(
    num_levels=5,
    coarsening_factor=0.7,  # Less aggressive
)
```

### Epoch Distribution

More epochs at finer levels capture more detail:

```python
# Standard: increasing epochs
config = MultilevelConfig(
    level_epochs=[50, 100, 200],
)

# Fast warmup: emphasize fine level
config = MultilevelConfig(
    level_epochs=[20, 50, 100],
    warmup_epochs=100,  # Extra at finest
)
```

### Combining with Other Techniques

**With Adaptive Sampling:**

```python
from opifex.training.adaptive_sampling import RADSampler

sampler = RADSampler()

while not trainer.is_at_finest():
    model = trainer.get_current_model()

    for epoch in range(trainer.get_epochs_for_current_level()):
        # Compute residuals
        residuals = compute_residual(model, all_points)

        # Adaptive sampling
        batch = sampler.sample(all_points, residuals, batch_size, key)

        # Training step
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
        # ...

    trainer.advance_level()
```

**With GradNorm:**

```python
from opifex.core.physics.gradnorm import GradNormBalancer

balancer = GradNormBalancer(num_losses=3, rngs=nnx.Rngs(0))

while not trainer.is_at_finest():
    model = trainer.get_current_model()

    # Reset balancer for each level
    balancer._initial_losses = None

    for epoch in range(trainer.get_epochs_for_current_level()):
        losses = compute_losses(model)

        if epoch == 0:
            balancer.set_initial_losses(losses)

        weighted_loss = balancer.compute_weighted_loss(losses)
        # ...

    trainer.advance_level()
```

**With Second-Order Optimization:**

```python
from opifex.optimization.second_order import (
    HybridOptimizer,
    HybridOptimizerConfig,
)

while not trainer.is_at_finest():
    model = trainer.get_current_model()

    # Use Adam at coarse levels, hybrid at finest
    if trainer.is_at_finest():
        optimizer = HybridOptimizer(HybridOptimizerConfig())
    else:
        optimizer = optax.adam(1e-3)

    # ... training ...
    trainer.advance_level()
```

### Monitoring Progress

```python
# Track loss at each level
level_losses = []

while not trainer.is_at_finest():
    model = trainer.get_current_model()
    level = trainer.current_level

    # Training
    losses = []
    for epoch in range(trainer.get_epochs_for_current_level()):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        losses.append(float(loss))
        # ... update ...

    level_losses.append({
        'level': level,
        'final_loss': losses[-1],
        'improvement': losses[0] / losses[-1],
    })

    trainer.advance_level()

# Analyze progression
for info in level_losses:
    print(f"Level {info['level']}: loss={info['final_loss']:.4e}, "
          f"improvement={info['improvement']:.1f}x")
```

## Complete Training Example

```python
import jax.numpy as jnp
import optax
from flax import nnx
from opifex.training.multilevel import CascadeTrainer, MultilevelConfig

# Problem setup
def pde_residual(model, x):
    """Compute PDE residual for Poisson equation."""
    def u_scalar(xi):
        return model(xi.reshape(1, -1)).squeeze()

    laplacian = jax.vmap(lambda xi: jnp.trace(jax.hessian(u_scalar)(xi)))(x)
    f = jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1])
    return laplacian + f

def loss_fn(model, x_interior, x_boundary):
    residual = pde_residual(model, x_interior)
    pde_loss = jnp.mean(residual ** 2)

    boundary_pred = model(x_boundary)
    bc_loss = jnp.mean(boundary_pred ** 2)

    return pde_loss + 10.0 * bc_loss

# Create multilevel trainer
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
    rngs=nnx.Rngs(42),
)

# Training data
x_interior = jax.random.uniform(jax.random.key(0), (1000, 2))
x_boundary = generate_boundary_points(100)

# Multilevel training loop
for level in range(config.num_levels):
    model = trainer.get_current_model()
    epochs = trainer.get_epochs_for_current_level()

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(nnx.state(model))

    print(f"\n--- Level {level} ---")
    for epoch in range(epochs):
        loss, grads = nnx.value_and_grad(
            lambda m: loss_fn(m, x_interior, x_boundary)
        )(model)

        updates, opt_state = optimizer.update(grads, opt_state)
        nnx.update(model, updates)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4e}")

    if not trainer.is_at_finest():
        trainer.advance_level()

# Final model
final_model = trainer.get_current_model()
```

## See Also

- [Training Guide](../user-guide/training.md) - General training procedures
- [Adaptive Sampling](adaptive-sampling.md) - Residual-based sampling
- [GradNorm](gradnorm.md) - Multi-task loss balancing
- [API Reference](../api/training.md#multilevel) - Complete API documentation
