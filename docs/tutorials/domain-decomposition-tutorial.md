# Domain Decomposition PINN Tutorial

## Introduction

This tutorial walks through implementing physics-informed neural networks using domain decomposition methods. We'll solve a 2D Poisson equation on a complex domain by decomposing it into simpler subdomains, demonstrating all four DD-PINN variants available in Opifex.

## Problem Setup

We'll solve the Poisson equation:

$$-\nabla^2 u = f(x, y) \quad \text{on } \Omega = [0, 1]^2$$

with Dirichlet boundary conditions $u = 0$ on $\partial\Omega$ and source term:

$$f(x, y) = 2\pi^2 \sin(\pi x) \sin(\pi y)$$

The exact solution is $u(x, y) = \sin(\pi x) \sin(\pi y)$.

## Prerequisites

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt

# Set random seed for reproducibility
key = jax.random.key(42)
```

## Step 1: Define the Problem

First, let's define our PDE residual and boundary conditions:

```python
def pde_residual(model, x):
    """Compute Poisson equation residual: -∇²u - f = 0."""
    def u_scalar(xi):
        return model(xi.reshape(1, -1)).squeeze()

    # Compute Laplacian using JAX autodiff
    def laplacian(xi):
        hess = jax.hessian(u_scalar)(xi)
        return jnp.trace(hess)

    laplacians = jax.vmap(laplacian)(x)

    # Source term
    f = 2 * jnp.pi**2 * jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1])

    return -laplacians - f

def boundary_loss(model, x_bc):
    """Compute boundary condition loss (Dirichlet: u = 0)."""
    pred = model(x_bc)
    return jnp.mean(pred ** 2)

def exact_solution(x):
    """Exact solution for validation."""
    return jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1])
```

## Step 2: Generate Training Points

```python
def generate_points(n_interior=2000, n_boundary=400, key=None):
    """Generate interior and boundary points."""
    if key is None:
        key = jax.random.key(0)

    key1, key2 = jax.random.split(key)

    # Interior points
    x_interior = jax.random.uniform(key1, (n_interior, 2))

    # Boundary points (all four edges)
    n_per_edge = n_boundary // 4

    # Left (x=0)
    left = jnp.column_stack([jnp.zeros(n_per_edge), jnp.linspace(0, 1, n_per_edge)])
    # Right (x=1)
    right = jnp.column_stack([jnp.ones(n_per_edge), jnp.linspace(0, 1, n_per_edge)])
    # Bottom (y=0)
    bottom = jnp.column_stack([jnp.linspace(0, 1, n_per_edge), jnp.zeros(n_per_edge)])
    # Top (y=1)
    top = jnp.column_stack([jnp.linspace(0, 1, n_per_edge), jnp.ones(n_per_edge)])

    x_boundary = jnp.concatenate([left, right, bottom, top])

    return x_interior, x_boundary

x_interior, x_boundary = generate_points(key=key)
```

## Step 3: Standard PINN Baseline

Let's first establish a baseline with a standard single-network PINN:

```python
from opifex.neural.base import StandardMLP

# Create standard PINN
class StandardPINN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.net = StandardMLP(
            features=[2, 64, 64, 64, 1],
            activation="tanh",
            rngs=rngs
        )

    def __call__(self, x):
        return self.net(x)

# Training function
def train_pinn(model, x_interior, x_boundary, num_epochs=5000, lr=1e-3):
    """Train a standard PINN."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(nnx.state(model))

    @jax.jit
    def loss_fn(model):
        residual = pde_residual(model, x_interior)
        pde_loss = jnp.mean(residual ** 2)
        bc_loss = boundary_loss(model, x_boundary)
        return pde_loss + 10.0 * bc_loss

    losses = []
    for epoch in range(num_epochs):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        nnx.update(model, updates)

        if epoch % 500 == 0:
            losses.append(float(loss))
            print(f"Epoch {epoch}: loss = {loss:.4e}")

    return losses

# Train baseline
standard_pinn = StandardPINN(rngs=nnx.Rngs(0))
baseline_losses = train_pinn(standard_pinn, x_interior, x_boundary)
```

## Step 4: XPINN Implementation

Now let's use XPINN (Extended PINN) with explicit interface conditions:

```python
from opifex.neural.pinns.domain_decomposition import (
    XPINN, XPINNConfig, Subdomain, uniform_partition
)

# Create 2x2 domain decomposition
subdomains, interfaces = uniform_partition(
    bounds=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
    num_partitions=[2, 2],  # 2x2 grid
    overlap=0.1,            # 10% overlap
)

print(f"Created {len(subdomains)} subdomains and {len(interfaces)} interfaces")

# Configure XPINN
xpinn_config = XPINNConfig(
    continuity_weight=10.0,    # Interface continuity
    gradient_weight=1.0,       # Gradient matching
    hidden_dims=[64, 64],
)

# Create XPINN model
xpinn = XPINN(
    subdomains=subdomains,
    interfaces=interfaces,
    input_dim=2,
    output_dim=1,
    config=xpinn_config,
    rngs=nnx.Rngs(0),
)

# Train XPINN
def train_xpinn(model, x_interior, x_boundary, num_epochs=5000, lr=1e-3):
    """Train XPINN with interface conditions."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(nnx.state(model))

    @jax.jit
    def loss_fn(model):
        # PDE residual in each subdomain
        residual = pde_residual(model, x_interior)
        pde_loss = jnp.mean(residual ** 2)

        # Boundary conditions
        bc_loss = boundary_loss(model, x_boundary)

        # Interface conditions (continuity + gradient matching)
        interface_loss = model.compute_interface_loss()

        return pde_loss + 10.0 * bc_loss + interface_loss

    losses = []
    for epoch in range(num_epochs):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        nnx.update(model, updates)

        if epoch % 500 == 0:
            losses.append(float(loss))
            print(f"Epoch {epoch}: loss = {loss:.4e}")

    return losses

xpinn_losses = train_xpinn(xpinn, x_interior, x_boundary)
```

## Step 5: FBPINN Implementation

FBPINN uses smooth window functions for seamless blending:

```python
from opifex.neural.pinns.domain_decomposition import (
    FBPINN, FBPINNConfig, CosineWindow
)

# Configure FBPINN
fbpinn_config = FBPINNConfig(
    window_type="cosine",      # Smooth cosine windows
    hidden_dims=[64, 64],
    overlap_factor=1.5,        # Increased overlap for smooth blending
)

# Create FBPINN model
fbpinn = FBPINN(
    subdomains=subdomains,
    input_dim=2,
    output_dim=1,
    config=fbpinn_config,
    rngs=nnx.Rngs(0),
)

# Training is similar - no explicit interface loss needed
def train_fbpinn(model, x_interior, x_boundary, num_epochs=5000, lr=1e-3):
    """Train FBPINN with window function blending."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(nnx.state(model))

    @jax.jit
    def loss_fn(model):
        residual = pde_residual(model, x_interior)
        pde_loss = jnp.mean(residual ** 2)
        bc_loss = boundary_loss(model, x_boundary)
        return pde_loss + 10.0 * bc_loss

    losses = []
    for epoch in range(num_epochs):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        nnx.update(model, updates)

        if epoch % 500 == 0:
            losses.append(float(loss))
            print(f"Epoch {epoch}: loss = {loss:.4e}")

    return losses

fbpinn_losses = train_fbpinn(fbpinn, x_interior, x_boundary)
```

## Step 6: CPINN Implementation

CPINN enforces flux conservation at interfaces, crucial for conservation laws:

```python
from opifex.neural.pinns.domain_decomposition import (
    CPINN, CPINNConfig
)

# Configure CPINN
cpinn_config = CPINNConfig(
    continuity_weight=10.0,
    flux_weight=5.0,           # Explicit flux conservation
    hidden_dims=[64, 64],
)

# Create CPINN model
cpinn = CPINN(
    subdomains=subdomains,
    interfaces=interfaces,
    input_dim=2,
    output_dim=1,
    config=cpinn_config,
    rngs=nnx.Rngs(0),
)

def train_cpinn(model, x_interior, x_boundary, num_epochs=5000, lr=1e-3):
    """Train CPINN with flux conservation."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(nnx.state(model))

    @jax.jit
    def loss_fn(model):
        residual = pde_residual(model, x_interior)
        pde_loss = jnp.mean(residual ** 2)
        bc_loss = boundary_loss(model, x_boundary)

        # Combined interface loss (continuity + flux conservation)
        continuity_loss = model.compute_continuity_loss()
        flux_loss = model.compute_flux_loss()

        return pde_loss + 10.0 * bc_loss + continuity_loss + flux_loss

    losses = []
    for epoch in range(num_epochs):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        nnx.update(model, updates)

        if epoch % 500 == 0:
            losses.append(float(loss))
            print(f"Epoch {epoch}: loss = {loss:.4e}")

    return losses

cpinn_losses = train_cpinn(cpinn, x_interior, x_boundary)
```

## Step 7: APINN Implementation

APINN uses a learnable gating network for adaptive blending:

```python
from opifex.neural.pinns.domain_decomposition import (
    APINN, APINNConfig
)

# Configure APINN
apinn_config = APINNConfig(
    gating_hidden_dims=[32, 32],  # Gating network architecture
    temperature=1.0,              # Softmax temperature
    continuity_weight=5.0,        # Optional continuity regularization
    hidden_dims=[64, 64],
)

# Create APINN model
apinn = APINN(
    subdomains=subdomains,
    interfaces=interfaces,
    input_dim=2,
    output_dim=1,
    config=apinn_config,
    rngs=nnx.Rngs(0),
)

def train_apinn(model, x_interior, x_boundary, num_epochs=5000, lr=1e-3):
    """Train APINN with learnable gating."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(nnx.state(model))

    @jax.jit
    def loss_fn(model):
        residual = pde_residual(model, x_interior)
        pde_loss = jnp.mean(residual ** 2)
        bc_loss = boundary_loss(model, x_boundary)

        # Optional interface regularization
        interface_loss = model.compute_interface_loss()

        return pde_loss + 10.0 * bc_loss + interface_loss

    losses = []
    for epoch in range(num_epochs):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        nnx.update(model, updates)

        if epoch % 500 == 0:
            losses.append(float(loss))
            print(f"Epoch {epoch}: loss = {loss:.4e}")

    return losses

apinn_losses = train_apinn(apinn, x_interior, x_boundary)
```

## Step 8: Comparison and Visualization

```python
def evaluate_model(model, n_test=50):
    """Evaluate model against exact solution."""
    x = jnp.linspace(0, 1, n_test)
    y = jnp.linspace(0, 1, n_test)
    X, Y = jnp.meshgrid(x, y)
    test_points = jnp.column_stack([X.ravel(), Y.ravel()])

    pred = model(test_points).reshape(n_test, n_test)
    exact = exact_solution(test_points).reshape(n_test, n_test)

    l2_error = jnp.sqrt(jnp.mean((pred - exact) ** 2))
    max_error = jnp.max(jnp.abs(pred - exact))

    return pred, exact, l2_error, max_error

# Evaluate all models
models = {
    "Standard PINN": standard_pinn,
    "XPINN": xpinn,
    "FBPINN": fbpinn,
    "CPINN": cpinn,
    "APINN": apinn,
}

print("\n--- Error Comparison ---")
for name, model in models.items():
    pred, exact, l2_error, max_error = evaluate_model(model)
    print(f"{name}: L2 error = {l2_error:.4e}, Max error = {max_error:.4e}")

# Plot training curves
plt.figure(figsize=(10, 6))
all_losses = {
    "Standard PINN": baseline_losses,
    "XPINN": xpinn_losses,
    "FBPINN": fbpinn_losses,
    "CPINN": cpinn_losses,
    "APINN": apinn_losses,
}

for name, losses in all_losses.items():
    plt.semilogy(range(0, len(losses)*500, 500), losses, label=name)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Convergence Comparison")
plt.legend()
plt.grid(True)
plt.savefig("dd_pinn_comparison.png", dpi=150)
plt.show()
```

## Step 9: Visualize Solutions

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot each model's solution
x = jnp.linspace(0, 1, 50)
y = jnp.linspace(0, 1, 50)
X, Y = jnp.meshgrid(x, y)
test_points = jnp.column_stack([X.ravel(), Y.ravel()])

# Exact solution
exact = exact_solution(test_points).reshape(50, 50)
im = axes[0, 0].contourf(X, Y, exact, levels=50, cmap='viridis')
axes[0, 0].set_title("Exact Solution")
plt.colorbar(im, ax=axes[0, 0])

# Model predictions
for idx, (name, model) in enumerate(list(models.items())[:5]):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    pred = model(test_points).reshape(50, 50)
    im = axes[row, col].contourf(X, Y, pred, levels=50, cmap='viridis')
    axes[row, col].set_title(name)
    plt.colorbar(im, ax=axes[row, col])

plt.tight_layout()
plt.savefig("dd_pinn_solutions.png", dpi=150)
plt.show()
```

## Best Practices

### Choosing the Right Method

| Method | Best For | Key Consideration |
|--------|----------|-------------------|
| **XPINN** | Sharp interfaces, discontinuities | Explicit interface control |
| **FBPINN** | Smooth solutions, easy setup | No interface hyperparameters |
| **CPINN** | Conservation laws (fluids, heat) | Flux conservation critical |
| **APINN** | Unknown optimal decomposition | Learns blending adaptively |

### Hyperparameter Guidelines

1. **Interface weights**: Start with 1-10x the PDE loss weight
2. **Overlap**: 5-15% of subdomain size for XPINN/CPINN
3. **Window overlap**: 1.2-2.0x for FBPINN
4. **Gating temperature**: 0.1-2.0 for APINN (lower = sharper transitions)

### Scaling to Complex Domains

```python
# For complex geometries, increase partitions
subdomains, interfaces = uniform_partition(
    bounds=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
    num_partitions=[4, 4],  # 4x4 = 16 subdomains
    overlap=0.1,
)

# Increase network capacity per subdomain
config = XPINNConfig(
    hidden_dims=[128, 128, 128],
    continuity_weight=10.0,
)
```

## Summary

In this tutorial, we implemented:

1. **Standard PINN** baseline for comparison
2. **XPINN** with explicit interface continuity and gradient matching
3. **FBPINN** with smooth window function blending
4. **CPINN** with flux conservation enforcement
5. **APINN** with learnable gating networks

Domain decomposition enables:
- **Parallelization**: Independent subdomain networks
- **Scalability**: Handle complex, large-scale domains
- **Accuracy**: Capture local features with dedicated networks

## See Also

- [Domain Decomposition PINNs Guide](../methods/domain-decomposition-pinns.md) - Complete API reference
- [Physics-Informed Neural Networks](../methods/pinns.md) - PINN fundamentals
- [Advanced Training Tutorial](advanced-training-tutorial.md) - Combine with GradNorm, adaptive sampling
