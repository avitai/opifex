# Advanced Training Techniques Tutorial

## Introduction

This tutorial demonstrates how to combine multiple advanced training techniques for physics-informed neural networks. We'll integrate:

- **Multilevel Training**: Coarse-to-fine network hierarchy
- **Adaptive Sampling**: Focus on high-residual regions
- **GradNorm**: Balance multiple loss terms
- **Second-Order Optimization**: L-BFGS refinement

Together, these techniques dramatically improve convergence speed and solution quality.

## Problem Setup

We'll solve the Allen-Cahn equation, a challenging nonlinear PDE with sharp interfaces:

$$\frac{\partial u}{\partial t} = \epsilon^2 \nabla^2 u + u - u^3$$

with $\epsilon = 0.01$ (sharp interface), periodic boundary conditions, and initial condition:

$$u(x, 0) = \sin(\pi x)^2 \cos(\pi y)^2 - 0.5$$

## Prerequisites

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt

# Imports for advanced techniques
from opifex.training.multilevel import CascadeTrainer, MultilevelConfig
from opifex.training.adaptive_sampling import RADSampler, RARDRefiner, RADConfig, RARDConfig
from opifex.core.physics.gradnorm import GradNormBalancer, GradNormConfig
from opifex.optimization.second_order import HybridOptimizer, HybridOptimizerConfig, LBFGSConfig

key = jax.random.key(42)
epsilon = 0.01  # Sharp interface parameter
```

## Step 1: Define the PDE

```python
def allen_cahn_residual(model, x_t):
    """
    Compute Allen-Cahn residual: u_t - ε²∇²u - u + u³ = 0

    Args:
        model: Neural network u(x, y, t)
        x_t: Points of shape (N, 3) with [x, y, t]
    """
    def u_scalar(xt):
        return model(xt.reshape(1, -1)).squeeze()

    def compute_derivatives(xt):
        # First derivatives
        grad_u = jax.grad(u_scalar)(xt)
        u_x, u_y, u_t = grad_u[0], grad_u[1], grad_u[2]

        # Second derivatives (Laplacian)
        hess = jax.hessian(u_scalar)(xt)
        u_xx = hess[0, 0]
        u_yy = hess[1, 1]

        return u_t, u_xx, u_yy

    # Vectorize over all points
    u_t, u_xx, u_yy = jax.vmap(compute_derivatives)(x_t)
    u = model(x_t).squeeze()

    # Allen-Cahn: u_t = ε²(u_xx + u_yy) + u - u³
    residual = u_t - epsilon**2 * (u_xx + u_yy) - u + u**3
    return residual

def initial_condition(x_y):
    """Initial condition: u(x, y, 0) = sin²(πx)cos²(πy) - 0.5"""
    x, y = x_y[:, 0], x_y[:, 1]
    return jnp.sin(jnp.pi * x)**2 * jnp.cos(jnp.pi * y)**2 - 0.5

def generate_training_points(n_interior=5000, n_initial=1000, n_boundary=500, key=None):
    """Generate training points for space-time domain [0,1]² × [0,1]."""
    if key is None:
        key = jax.random.key(0)

    keys = jax.random.split(key, 3)

    # Interior points (x, y, t) ∈ [0,1]³
    x_interior = jax.random.uniform(keys[0], (n_interior, 3))

    # Initial condition points (t = 0)
    x_y_initial = jax.random.uniform(keys[1], (n_initial, 2))
    x_initial = jnp.column_stack([x_y_initial, jnp.zeros(n_initial)])
    u_initial = initial_condition(x_y_initial)

    # Boundary points (periodic, so we need matching pairs)
    t_bc = jax.random.uniform(keys[2], (n_boundary,))

    return x_interior, x_initial, u_initial, t_bc

x_interior, x_initial, u_initial, t_bc = generate_training_points(key=key)
```

## Step 2: Define Loss Functions

We have three loss components that need balancing:

```python
def pde_loss_fn(model, x_interior):
    """PDE residual loss."""
    residual = allen_cahn_residual(model, x_interior)
    return jnp.mean(residual ** 2)

def ic_loss_fn(model, x_initial, u_initial):
    """Initial condition loss."""
    pred = model(x_initial).squeeze()
    return jnp.mean((pred - u_initial) ** 2)

def periodic_bc_loss_fn(model, t_bc):
    """Periodic boundary condition loss: u(0,y,t) = u(1,y,t), etc."""
    n = len(t_bc)
    y_vals = jnp.linspace(0, 1, n)

    # x-direction periodicity
    left = jnp.column_stack([jnp.zeros(n), y_vals, t_bc])
    right = jnp.column_stack([jnp.ones(n), y_vals, t_bc])
    x_periodic = jnp.mean((model(left) - model(right)) ** 2)

    # y-direction periodicity
    x_vals = jnp.linspace(0, 1, n)
    bottom = jnp.column_stack([x_vals, jnp.zeros(n), t_bc])
    top = jnp.column_stack([x_vals, jnp.ones(n), t_bc])
    y_periodic = jnp.mean((model(bottom) - model(top)) ** 2)

    return x_periodic + y_periodic

def compute_all_losses(model, x_interior, x_initial, u_initial, t_bc):
    """Compute all three loss components."""
    pde = pde_loss_fn(model, x_interior)
    ic = ic_loss_fn(model, x_initial, u_initial)
    bc = periodic_bc_loss_fn(model, t_bc)
    return jnp.array([pde, ic, bc])
```

## Step 3: Multilevel Training Setup

Configure coarse-to-fine training hierarchy:

```python
# Configure multilevel training
multilevel_config = MultilevelConfig(
    num_levels=3,                    # 3 hierarchy levels
    coarsening_factor=0.5,           # Width halves at each coarser level
    level_epochs=[500, 1000, 2000],  # More epochs at finer levels
    warmup_epochs=500,               # Extra epochs at finest level
)

# Create cascade trainer
# Networks: [16,16,16] -> [32,32,32] -> [64,64,64]
trainer = CascadeTrainer(
    input_dim=3,                     # (x, y, t)
    output_dim=1,                    # u
    base_hidden_dims=[64, 64, 64],   # Finest level architecture
    config=multilevel_config,
    rngs=nnx.Rngs(0),
)

print(f"Created {multilevel_config.num_levels}-level hierarchy")
print(f"Level 0 (coarsest): {trainer.get_current_model()}")
```

## Step 4: Configure GradNorm Balancing

```python
# GradNorm automatically balances PDE, IC, and BC losses
gradnorm_config = GradNormConfig(
    alpha=1.5,           # Asymmetry parameter (prioritize slow tasks)
    learning_rate=0.025, # Weight update rate
    update_frequency=10, # Update every 10 steps
    min_weight=0.1,      # Minimum weight bound
    max_weight=10.0,     # Maximum weight bound
)

# We'll create a new balancer for each level
def create_gradnorm_balancer():
    return GradNormBalancer(
        num_losses=3,  # PDE, IC, BC
        config=gradnorm_config,
        rngs=nnx.Rngs(0),
    )
```

## Step 5: Configure Adaptive Sampling

```python
# RAD: Residual-based Adaptive Distribution
rad_config = RADConfig(
    beta=1.0,                # Residual exponent
    resample_frequency=50,   # Update distribution every 50 steps
    min_probability=1e-6,    # Numerical stability
)

sampler = RADSampler(rad_config)

# RAR-D: For growing point set (optional, for refinement)
rard_config = RARDConfig(
    num_new_points=100,       # Points to add per refinement
    percentile_threshold=90,  # Focus on top 10% residuals
    noise_scale=0.05,         # Perturbation scale
)

refiner = RARDRefiner(rard_config)
```

## Step 6: Configure Hybrid Optimizer

```python
# Hybrid Adam -> L-BFGS for final refinement
hybrid_config = HybridOptimizerConfig(
    adam_lr=1e-3,
    switch_threshold=1e-4,    # Switch when loss < threshold
    max_adam_steps=None,      # No step limit, use threshold
    lbfgs_config=LBFGSConfig(
        memory_size=20,
        max_iterations=100,
        tolerance=1e-8,
    ),
)
```

## Step 7: Full Training Loop

Now let's combine everything:

```python
def advanced_training_loop(
    trainer,
    x_interior,
    x_initial,
    u_initial,
    t_bc,
    use_adaptive_sampling=True,
    use_gradnorm=True,
):
    """
    Complete training loop combining:
    - Multilevel coarse-to-fine
    - GradNorm loss balancing
    - Adaptive sampling
    """
    all_history = []

    # Full point set for adaptive sampling
    all_interior_points = x_interior.copy()
    bounds = jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])  # 3D domain

    # Iterate through hierarchy levels
    level = 0
    while not trainer.is_at_finest():
        print(f"\n{'='*50}")
        print(f"Level {level}: Training...")
        print(f"{'='*50}")

        model = trainer.get_current_model()
        epochs = trainer.get_epochs_for_current_level()

        # Create fresh GradNorm balancer for this level
        balancer = create_gradnorm_balancer() if use_gradnorm else None

        # Standard Adam optimizer for this level
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(nnx.state(model))

        level_history = {"losses": [], "weights": []}
        current_points = all_interior_points

        for epoch in range(epochs):
            key = jax.random.fold_in(jax.random.key(level), epoch)

            # Adaptive sampling: resample based on residuals
            if use_adaptive_sampling and epoch % rad_config.resample_frequency == 0:
                residuals = allen_cahn_residual(model, all_interior_points)
                current_points = sampler.sample(
                    all_interior_points,
                    jnp.abs(residuals),
                    batch_size=min(2000, len(all_interior_points)),
                    key=key,
                )

            # Compute individual losses
            losses = compute_all_losses(model, current_points, x_initial, u_initial, t_bc)

            # Initialize GradNorm on first step
            if use_gradnorm and epoch == 0:
                balancer.set_initial_losses(losses)

            # Define total loss function
            def total_loss_fn(m):
                l = compute_all_losses(m, current_points, x_initial, u_initial, t_bc)
                if use_gradnorm:
                    return balancer.compute_weighted_loss(l)
                else:
                    return l[0] + 10.0 * l[1] + 10.0 * l[2]  # Fixed weights

            # Gradient step
            loss, grads = nnx.value_and_grad(total_loss_fn)(model)
            updates, opt_state = optimizer.update(grads, opt_state)
            nnx.update(model, updates)

            # Update GradNorm weights
            if use_gradnorm and epoch % gradnorm_config.update_frequency == 0:
                # Compute gradient norms for weight update
                def get_grad_norm(loss_fn):
                    _, g = nnx.value_and_grad(loss_fn)(model)
                    return jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(g)))

                grad_norms = jnp.array([
                    get_grad_norm(lambda m: pde_loss_fn(m, current_points)),
                    get_grad_norm(lambda m: ic_loss_fn(m, x_initial, u_initial)),
                    get_grad_norm(lambda m: periodic_bc_loss_fn(m, t_bc)),
                ])
                balancer.update_weights(grad_norms, losses, balancer.get_initial_losses())

            # Logging
            if epoch % 100 == 0:
                weights = balancer.weights if use_gradnorm else jnp.array([1.0, 10.0, 10.0])
                level_history["losses"].append(float(loss))
                level_history["weights"].append(weights.tolist())

                print(f"  Epoch {epoch}: loss={loss:.4e}, "
                      f"PDE={losses[0]:.4e}, IC={losses[1]:.4e}, BC={losses[2]:.4e}")
                if use_gradnorm:
                    print(f"    GradNorm weights: {balancer.weights}")

        all_history.append(level_history)

        # Advance to finer level (prolongation happens automatically)
        if not trainer.is_at_finest():
            trainer.advance_level()
            level += 1

    # Final refinement with L-BFGS (optional)
    print(f"\n{'='*50}")
    print("Final L-BFGS Refinement")
    print(f"{'='*50}")

    final_model = trainer.get_current_model()

    # Use hybrid optimizer for final polish
    hybrid = HybridOptimizer(hybrid_config)

    for step in range(500):
        losses = compute_all_losses(final_model, all_interior_points, x_initial, u_initial, t_bc)
        total = losses[0] + 10.0 * losses[1] + 10.0 * losses[2]

        def loss_fn(m):
            l = compute_all_losses(m, all_interior_points, x_initial, u_initial, t_bc)
            return l[0] + 10.0 * l[1] + 10.0 * l[2]

        _, grads = nnx.value_and_grad(loss_fn)(final_model)
        hybrid.step(final_model, grads, total)

        if step % 100 == 0:
            print(f"  Step {step}: loss={total:.4e}")
            if hybrid.has_switched:
                print("    (Using L-BFGS)")

    return trainer.get_current_model(), all_history

# Run training!
trained_model, history = advanced_training_loop(
    trainer,
    x_interior,
    x_initial,
    u_initial,
    t_bc,
    use_adaptive_sampling=True,
    use_gradnorm=True,
)
```

## Step 8: Comparison Study

Let's compare with simpler approaches:

```python
def train_baseline(x_interior, x_initial, u_initial, t_bc, epochs=5000):
    """Train standard PINN without advanced techniques."""
    from opifex.neural.base import StandardMLP

    class SimplePINN(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.net = StandardMLP(
                features=[3, 64, 64, 64, 1],
                activation="tanh",
                rngs=rngs
            )

        def __call__(self, x):
            return self.net(x)

    model = SimplePINN(rngs=nnx.Rngs(0))
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(nnx.state(model))

    losses = []
    for epoch in range(epochs):
        def loss_fn(m):
            l = compute_all_losses(m, x_interior, x_initial, u_initial, t_bc)
            return l[0] + 10.0 * l[1] + 10.0 * l[2]

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        nnx.update(model, updates)

        if epoch % 500 == 0:
            losses.append(float(loss))
            print(f"Baseline Epoch {epoch}: loss={loss:.4e}")

    return model, losses

print("\n--- Training Baseline (no advanced techniques) ---")
baseline_model, baseline_losses = train_baseline(
    x_interior, x_initial, u_initial, t_bc, epochs=5000
)
```

## Step 9: Visualize Results

```python
def visualize_solution(model, t_values=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """Visualize solution at different time snapshots."""
    fig, axes = plt.subplots(1, len(t_values), figsize=(4*len(t_values), 4))

    x = jnp.linspace(0, 1, 100)
    y = jnp.linspace(0, 1, 100)
    X, Y = jnp.meshgrid(x, y)

    for idx, t in enumerate(t_values):
        # Create evaluation points
        T = jnp.full_like(X, t)
        points = jnp.column_stack([X.ravel(), Y.ravel(), T.ravel()])

        # Predict
        u = model(points).reshape(100, 100)

        # Plot
        im = axes[idx].contourf(X, Y, u, levels=50, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[idx].set_title(f't = {t:.2f}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    plt.savefig("allen_cahn_solution.png", dpi=150)
    plt.show()

# Visualize trained solution
visualize_solution(trained_model)

# Plot training comparison
plt.figure(figsize=(10, 6))

# Baseline losses
epochs_baseline = range(0, len(baseline_losses)*500, 500)
plt.semilogy(epochs_baseline, baseline_losses, 'b--', label='Baseline PINN', linewidth=2)

# Advanced training losses (concatenate all levels)
advanced_losses = []
epoch_offset = 0
for level_idx, level_hist in enumerate(history):
    level_losses = level_hist["losses"]
    epochs = range(epoch_offset, epoch_offset + len(level_losses)*100, 100)
    plt.semilogy(epochs, level_losses, label=f'Level {level_idx}')
    advanced_losses.extend(level_losses)
    epoch_offset += len(level_losses) * 100

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Convergence: Advanced vs Baseline")
plt.legend()
plt.grid(True)
plt.savefig("training_comparison.png", dpi=150)
plt.show()
```

## Step 10: Analyze GradNorm Weight Evolution

```python
def plot_weight_evolution(history):
    """Plot how GradNorm weights evolve during training."""
    fig, axes = plt.subplots(1, len(history), figsize=(5*len(history), 4))

    if len(history) == 1:
        axes = [axes]

    for level_idx, level_hist in enumerate(history):
        weights = jnp.array(level_hist["weights"])
        steps = range(len(weights))

        axes[level_idx].plot(steps, weights[:, 0], label='PDE')
        axes[level_idx].plot(steps, weights[:, 1], label='IC')
        axes[level_idx].plot(steps, weights[:, 2], label='BC')
        axes[level_idx].set_xlabel("Update Step")
        axes[level_idx].set_ylabel("Weight")
        axes[level_idx].set_title(f"Level {level_idx} GradNorm Weights")
        axes[level_idx].legend()
        axes[level_idx].grid(True)

    plt.tight_layout()
    plt.savefig("gradnorm_weights.png", dpi=150)
    plt.show()

plot_weight_evolution(history)
```

## Best Practices Summary

### When to Use Each Technique

| Technique | Use When | Typical Benefit |
|-----------|----------|-----------------|
| **Multilevel** | Multi-scale solutions, slow convergence | 2-5x faster |
| **GradNorm** | Multiple competing losses | More balanced accuracy |
| **Adaptive Sampling** | Sharp features, localized errors | 2-10x accuracy improvement |
| **L-BFGS Refinement** | Near convergence, need final polish | Lower final error |

### Hyperparameter Guidelines

**Multilevel Training:**
```python
# Simple problems
config = MultilevelConfig(num_levels=2, level_epochs=[500, 1000])

# Complex multi-scale problems
config = MultilevelConfig(num_levels=4, level_epochs=[200, 400, 800, 1600])
```

**GradNorm:**
```python
# Default starting point
config = GradNormConfig(alpha=1.5, learning_rate=0.01)

# Strong emphasis on lagging losses
config = GradNormConfig(alpha=2.0, learning_rate=0.025)
```

**Adaptive Sampling:**
```python
# Moderate concentration
config = RADConfig(beta=1.0, resample_frequency=100)

# Aggressive focus on hard regions
config = RADConfig(beta=2.0, resample_frequency=50)
```

### Debugging Tips

1. **GradNorm weights explode**: Reduce `learning_rate`, increase `min_weight`
2. **Adaptive sampling too aggressive**: Reduce `beta`, increase `min_probability`
3. **Multilevel transfer poor**: Increase overlap in network widths, more epochs per level
4. **L-BFGS not improving**: May already be at local minimum, increase `memory_size`

## Summary

This tutorial demonstrated combining:

1. **Multilevel Training** - Train small networks first, transfer to larger
2. **GradNorm** - Automatically balance PDE, IC, and BC losses
3. **Adaptive Sampling** - Focus training on high-residual regions
4. **Hybrid Optimization** - Adam for exploration, L-BFGS for refinement

These techniques are complementary and can provide significant improvements over standard PINN training, especially for challenging problems with sharp features or multiple scales.

## See Also

- [Multilevel Training Guide](../methods/multilevel-training.md) - Detailed configuration
- [Adaptive Sampling Guide](../methods/adaptive-sampling.md) - RAD and RAR-D algorithms
- [GradNorm Guide](../methods/gradnorm.md) - Loss balancing theory
- [Second-Order Optimization](../methods/second-order-optimization.md) - L-BFGS details
- [Domain Decomposition Tutorial](domain-decomposition-tutorial.md) - Combine with DD-PINNs
