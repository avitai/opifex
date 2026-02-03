# Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) incorporate physical laws, described by differential equations, directly into the neural network training process. This enables learning from both data and physics, reducing data requirements and ensuring physically consistent solutions.

## Overview

PINNs leverage automatic differentiation to embed PDEs into the loss function:

- **Data-driven learning** from observed measurements
- **Physics enforcement** through PDE residual minimization
- **Boundary/initial conditions** as soft or hard constraints
- **No mesh required** - operates on collocation points

!!! tip "Survey Reference"
    This framework implements methodologies from the comprehensive PINN survey (arXiv:2601.10222v1).

## Theoretical Foundation

### Problem Formulation

Consider a PDE of the form:

$$\mathcal{L}[u](x) = f(x), \quad x \in \Omega$$

with boundary conditions:

$$\mathcal{B}[u](x) = g(x), \quad x \in \partial\Omega$$

A neural network $u_\theta(x)$ approximates the solution by minimizing:

$$\mathcal{L}_{total} = \lambda_{pde} \mathcal{L}_{pde} + \lambda_{bc} \mathcal{L}_{bc} + \lambda_{data} \mathcal{L}_{data}$$

### Loss Components

**PDE Residual Loss:**
$$\mathcal{L}_{pde} = \frac{1}{N_r} \sum_{i=1}^{N_r} \left| \mathcal{L}[u_\theta](x_i) - f(x_i) \right|^2$$

**Boundary Condition Loss:**
$$\mathcal{L}_{bc} = \frac{1}{N_b} \sum_{i=1}^{N_b} \left| \mathcal{B}[u_\theta](x_i) - g(x_i) \right|^2$$

**Data Loss:**
$$\mathcal{L}_{data} = \frac{1}{N_d} \sum_{i=1}^{N_d} \left| u_\theta(x_i) - u^{obs}_i \right|^2$$

## Multi-Scale PINNs

The `opifex` library provides a specialized `MultiScalePINN` architecture designed to capture physics phenomena across multiple scales.

::: opifex.neural.pinns.multi_scale.MultiScalePINN
    options:
        show_root_heading: true
        show_source: true

### Factory Functions

::: opifex.neural.pinns.multi_scale.create_heat_equation_pinn
::: opifex.neural.pinns.multi_scale.create_navier_stokes_pinn

## Building Custom PINNs

You can build custom PINNs by combining `opifex.neural.base.StandardMLP` with `opifex.core.problems.PDEProblem`.

### Basic Example

```python
import jax
import jax.numpy as jnp
from flax import nnx
from opifex.neural.base import StandardMLP
from opifex.core.problems import create_pde_problem

# 1. Define the PDE (e.g., 1D Poisson equation: u_xx = -f)
def poisson_residual(model, x):
    """Compute PDE residual for Poisson equation."""
    def u_scalar(xi):
        return model(xi.reshape(1, -1)).squeeze()

    # Compute second derivative
    u_xx = jax.vmap(lambda xi: jax.hessian(u_scalar)(xi).squeeze())(x)
    f = jnp.sin(jnp.pi * x[:, 0])
    return u_xx + f

# 2. Create the Neural Network
rngs = nnx.Rngs(0)
model = StandardMLP(
    layer_sizes=[1, 64, 64, 1],
    activation='tanh',
    rngs=rngs
)

# 3. Define loss function
def loss_fn(model, x_interior, x_boundary):
    # PDE residual
    residual = poisson_residual(model, x_interior)
    pde_loss = jnp.mean(residual ** 2)

    # Boundary conditions (u(0) = u(1) = 0)
    bc_pred = model(x_boundary)
    bc_loss = jnp.mean(bc_pred ** 2)

    return pde_loss + 10.0 * bc_loss

# 4. Training
import optax

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(nnx.state(model))

# Generate training points
x_interior = jax.random.uniform(jax.random.key(0), (1000, 1))
x_boundary = jnp.array([[0.0], [1.0]])

for step in range(5000):
    loss, grads = nnx.value_and_grad(
        lambda m: loss_fn(m, x_interior, x_boundary)
    )(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    nnx.update(model, updates)

    if step % 500 == 0:
        print(f"Step {step}: loss = {loss:.4e}")
```

### 2D Laplace Equation Example

```python
import jax
import jax.numpy as jnp
from flax import nnx

class LaplacePINN(nnx.Module):
    """PINN for solving the Laplace equation."""

    def __init__(self, hidden_dims: list[int], rngs: nnx.Rngs):
        layers = []
        dims = [2, *hidden_dims, 1]  # 2D input, scalar output
        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i+1], rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, x):
        """Forward pass."""
        h = x
        for layer in list(self.layers)[:-1]:
            h = nnx.tanh(layer(h))
        return list(self.layers)[-1](h)

    def compute_residual(self, x):
        """Compute Laplace equation residual: u_xx + u_yy = 0."""
        def u_scalar(xi):
            return self(xi.reshape(1, -1)).squeeze()

        def laplacian(xi):
            hess = jax.hessian(u_scalar)(xi)
            return hess[0, 0] + hess[1, 1]  # u_xx + u_yy

        return jax.vmap(laplacian)(x)

# Create and train
model = LaplacePINN(hidden_dims=[64, 64, 64], rngs=nnx.Rngs(0))

# Domain: unit square [0, 1]^2
x_interior = jax.random.uniform(jax.random.key(0), (1000, 2))

# Boundary: known Dirichlet conditions
# (simplified - in practice, sample all four boundaries)
x_boundary = jnp.vstack([
    jnp.column_stack([jnp.zeros(25), jnp.linspace(0, 1, 25)]),
    jnp.column_stack([jnp.ones(25), jnp.linspace(0, 1, 25)]),
])
u_boundary = jnp.sin(jnp.pi * x_boundary[:, 1])  # Example BC
```

## Training Enhancements

Opifex provides several techniques to improve PINN training:

### Loss Balancing

Use [GradNorm](gradnorm.md) for automatic multi-task loss balancing:

```python
from opifex.core.physics.gradnorm import GradNormBalancer

balancer = GradNormBalancer(num_losses=3, rngs=nnx.Rngs(0))
losses = jnp.array([pde_loss, bc_loss, data_loss])
weighted_loss = balancer.compute_weighted_loss(losses)
```

### Adaptive Sampling

Use [RAD sampling](adaptive-sampling.md) to focus on high-residual regions:

```python
from opifex.training.adaptive_sampling import RADSampler

sampler = RADSampler()
residuals = model.compute_residual(all_points)
batch = sampler.sample(all_points, residuals, batch_size=256, key=key)
```

### Second-Order Optimization

Use [hybrid optimizers](second-order-optimization.md) for faster convergence:

```python
from opifex.optimization.second_order import HybridOptimizer

optimizer = HybridOptimizer(HybridOptimizerConfig(
    first_order_steps=1000,
    switch_criterion=SwitchCriterion.LOSS_VARIANCE,
))
```

### Multilevel Training

Use [multilevel training](multilevel-training.md) for hierarchical convergence:

```python
from opifex.training.multilevel import CascadeTrainer

trainer = CascadeTrainer(
    input_dim=2, output_dim=1,
    base_hidden_dims=[64, 64],
    config=MultilevelConfig(num_levels=3),
    rngs=nnx.Rngs(0),
)
```

### NTK Diagnostics

Use [NTK analysis](ntk-analysis.md) to diagnose training issues:

```python
from opifex.core.physics.ntk import NTKSpectralAnalyzer

analyzer = NTKSpectralAnalyzer(model)
diagnostics = analyzer.analyze(x_train, learning_rate=1e-3)
print(f"Condition number: {diagnostics.condition_number}")
```

## Advanced Methods

### Domain Decomposition

For large or complex domains, use [domain decomposition methods](domain-decomposition-pinns.md):

| Method | Description | Best For |
|--------|-------------|----------|
| **XPINN** | Explicit interface conditions | Non-overlapping domains |
| **FBPINN** | Window function blending | Smooth solutions |
| **CPINN** | Conservation enforcement | Conservation laws |
| **APINN** | Learned gating | Unknown optimal decomposition |

```python
from opifex.neural.pinns.domain_decomposition import XPINN, Subdomain, Interface

model = XPINN(
    input_dim=2, output_dim=1,
    subdomains=subdomains,
    interfaces=interfaces,
    hidden_dims=[32, 32],
    rngs=nnx.Rngs(0),
)
```

## Common PDEs

### Heat Equation

$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$

```python
def heat_residual(model, x, t, alpha=0.01):
    """x: spatial coords, t: time."""
    xt = jnp.column_stack([x, t])

    def u_scalar(xi):
        return model(xi.reshape(1, -1)).squeeze()

    # Compute derivatives
    def compute_derivs(xi):
        grad_u = jax.grad(u_scalar)(xi)
        hess_u = jax.hessian(u_scalar)(xi)
        u_t = grad_u[-1]  # Time derivative
        laplacian = jnp.sum(jnp.diag(hess_u)[:-1])  # Spatial Laplacian
        return u_t - alpha * laplacian

    return jax.vmap(compute_derivs)(xt)
```

### Burgers' Equation

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

```python
def burgers_residual(model, x, t, nu=0.01):
    xt = jnp.column_stack([x, t])

    def u_scalar(xi):
        return model(xi.reshape(1, -1)).squeeze()

    def compute_derivs(xi):
        u = u_scalar(xi)
        grad_u = jax.grad(u_scalar)(xi)
        hess_u = jax.hessian(u_scalar)(xi)
        u_x, u_t = grad_u[0], grad_u[1]
        u_xx = hess_u[0, 0]
        return u_t + u * u_x - nu * u_xx

    return jax.vmap(compute_derivs)(xt)
```

### Navier-Stokes (2D Incompressible)

$$\frac{\partial \vec{u}}{\partial t} + (\vec{u} \cdot \nabla)\vec{u} = -\nabla p + \nu \nabla^2 \vec{u}$$
$$\nabla \cdot \vec{u} = 0$$

```python
def navier_stokes_residual(model, xy, t, nu=0.01):
    """Model outputs [u, v, p]."""
    xyt = jnp.column_stack([xy, t])

    def field(xi):
        return model(xi.reshape(1, -1)).squeeze()  # [u, v, p]

    def compute_residuals(xi):
        # Get field values and derivatives
        uvp = field(xi)
        u, v, p = uvp[0], uvp[1], uvp[2]

        jac = jax.jacfwd(field)(xi)  # Shape: (3, 3) for [u,v,p] x [x,y,t]
        u_x, u_y, u_t = jac[0, 0], jac[0, 1], jac[0, 2]
        v_x, v_y, v_t = jac[1, 0], jac[1, 1], jac[1, 2]
        p_x, p_y = jac[2, 0], jac[2, 1]

        hess = jax.hessian(lambda xi: field(xi)[0])(xi)
        u_xx, u_yy = hess[0, 0], hess[1, 1]

        hess_v = jax.hessian(lambda xi: field(xi)[1])(xi)
        v_xx, v_yy = hess_v[0, 0], hess_v[1, 1]

        # Momentum equations
        res_u = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
        res_v = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)

        # Continuity
        res_cont = u_x + v_y

        return jnp.array([res_u, res_v, res_cont])

    return jax.vmap(compute_residuals)(xyt)
```

## Best Practices

### Network Architecture

- **Activation:** `tanh` for smooth solutions, `gelu` for faster training
- **Depth:** 3-5 layers for most problems
- **Width:** 32-128 neurons per layer
- **Input normalization:** Scale inputs to [-1, 1] or [0, 1]

### Collocation Point Selection

- **Interior:** 1000-10000 points (problem-dependent)
- **Boundary:** 100-1000 points per boundary segment
- **Distribution:** Use adaptive sampling for efficiency

### Loss Weighting

- Start with equal weights
- Use GradNorm for automatic balancing
- Increase BC weights if constraints are violated
- Monitor individual loss components

### Training Strategy

1. **Warmup:** Use Adam with learning rate warmup
2. **Main training:** Continue with Adam or switch to hybrid
3. **Fine-tuning:** Use L-BFGS for final convergence

## See Also

- [Domain Decomposition PINNs](domain-decomposition-pinns.md) - Large-scale problems
- [NTK Analysis](ntk-analysis.md) - Training diagnostics
- [Adaptive Sampling](adaptive-sampling.md) - Efficient collocation
- [GradNorm](gradnorm.md) - Loss balancing
- [Second-Order Optimization](second-order-optimization.md) - Fast convergence
- [Multilevel Training](multilevel-training.md) - Hierarchical training
- [Training Guide](../user-guide/training.md) - General training procedures
- [API Reference](../api/neural.md) - Complete API documentation
