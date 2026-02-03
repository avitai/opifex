# Domain Decomposition Physics-Informed Neural Networks

Domain decomposition approaches divide complex computational domains into smaller subdomains, training separate neural networks for each region while enforcing interface conditions. This enables scalable solutions to large-scale PDE problems.

## Overview

Domain decomposition PINNs address the scalability challenges of standard PINNs by:

- **Decomposing the domain** into manageable subdomains
- **Training separate networks** for each subdomain
- **Enforcing interface conditions** to ensure global solution consistency
- **Enabling parallelization** across subdomains

!!! tip "Survey Reference"
    This implementation follows the methodologies described in Section 8.3 of the PINN survey (arXiv:2601.10222v1).

## Theoretical Foundation

### Domain Decomposition Principles

Given a domain $\Omega$ and a PDE $\mathcal{L}[u] = f$, domain decomposition divides $\Omega$ into non-overlapping (or overlapping) subdomains $\Omega_1, \Omega_2, \ldots, \Omega_k$ such that:

$$\Omega = \bigcup_{i=1}^{k} \Omega_i$$

Each subdomain $\Omega_i$ has its own neural network $u_i(x; \theta_i)$ approximating the solution.

### Interface Conditions

At the interface $\Gamma_{ij}$ between subdomains $\Omega_i$ and $\Omega_j$, two conditions must be enforced:

**Continuity Condition:**
$$u^{(i)}(x) = u^{(j)}(x), \quad x \in \Gamma_{ij}$$

**Flux Continuity Condition:**
$$\nabla u^{(i)} \cdot \vec{n} = \nabla u^{(j)} \cdot \vec{n}, \quad x \in \Gamma_{ij}$$

where $\vec{n}$ is the interface normal vector.

## Methods

### XPINN (Extended PINN)

XPINN decomposes the domain into non-overlapping subdomains with explicit interface conditions.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.neural.pinns.domain_decomposition import (
    XPINN,
    XPINNConfig,
    Subdomain,
    Interface,
)

# Define subdomains
subdomains = [
    Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
    Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
]

# Define interface between subdomains
interfaces = [
    Interface(
        subdomain_ids=(0, 1),
        points=jnp.linspace(0.5, 0.5, 10).reshape(-1, 1),
        normal=jnp.array([1.0]),
    )
]

# Create XPINN model
config = XPINNConfig(
    continuity_weight=1.0,  # Weight for u_left = u_right
    flux_weight=1.0,        # Weight for du/dn_left = du/dn_right
    residual_weight=1.0,    # Weight for PDE residual
)

model = XPINN(
    input_dim=1,
    output_dim=1,
    subdomains=subdomains,
    interfaces=interfaces,
    hidden_dims=[32, 32],
    config=config,
    rngs=nnx.Rngs(0),
)

# Compute interface losses
continuity_loss = model.compute_continuity_loss()
flux_loss = model.compute_flux_loss()
total_interface_loss = model.compute_interface_loss()
```

**Configuration Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `continuity_weight` | 1.0 | Weight for interface continuity loss |
| `flux_weight` | 1.0 | Weight for flux matching loss |
| `residual_weight` | 1.0 | Weight for PDE residual loss |
| `average_residual_weight` | 0.0 | Weight for residual averaging at interfaces |

### FBPINN (Finite Basis PINN)

FBPINN uses smooth window functions to blend subdomain solutions, eliminating the need for explicit interface conditions.

```python
from opifex.neural.pinns.domain_decomposition import (
    FBPINN,
    FBPINNConfig,
    Subdomain,
)

# Define overlapping subdomains
subdomains = [
    Subdomain(id=0, bounds=jnp.array([[0.0, 0.6]])),  # Overlap region: [0.4, 0.6]
    Subdomain(id=1, bounds=jnp.array([[0.4, 1.0]])),
]

# Configure window functions
config = FBPINNConfig(
    window_type="cosine",     # Options: "cosine", "gaussian"
    normalize_windows=True,   # Ensure partition of unity
    overlap_factor=0.2,       # Overlap fraction
    gaussian_sigma=0.25,      # Sigma for Gaussian windows
)

model = FBPINN(
    input_dim=1,
    output_dim=1,
    subdomains=subdomains,
    interfaces=[],  # No explicit interfaces needed
    hidden_dims=[32, 32],
    config=config,
    rngs=nnx.Rngs(0),
)

# Forward pass automatically blends using window functions
x = jnp.linspace(0, 1, 100).reshape(-1, 1)
u = model(x)
```

**Window Functions:**

The output is computed as a partition of unity:

$$u(x) = \frac{\sum_i w_i(x) \cdot u_i(x)}{\sum_j w_j(x)}$$

Available window types:

- **Cosine Window:** $w(x) = 0.5(1 + \cos(\pi r))$ for $r < 1$
- **Gaussian Window:** $w(x) = \exp(-\|x - c\|^2 / 2\sigma^2)$

### CPINN (Conservative PINN)

CPINN extends XPINN with explicit flux conservation for problems governed by conservation laws.

```python
from opifex.neural.pinns.domain_decomposition import (
    CPINN,
    CPINNConfig,
    Subdomain,
    Interface,
)

# Define subdomains and interfaces
subdomains = [
    Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
    Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
]

interfaces = [
    Interface(
        subdomain_ids=(0, 1),
        points=jnp.array([[0.5]] * 10),
        normal=jnp.array([1.0]),
    )
]

config = CPINNConfig(
    flux_weight=1.0,         # Weight for flux conservation
    continuity_weight=1.0,   # Weight for solution continuity
    conservation_weight=0.1, # Weight for global conservation
)

model = CPINN(
    input_dim=1,
    output_dim=1,
    subdomains=subdomains,
    interfaces=interfaces,
    hidden_dims=[32, 32],
    config=config,
    rngs=nnx.Rngs(0),
)

# Compute conservation-specific losses
continuity_loss = model.compute_continuity_loss()
flux_conservation_loss = model.compute_flux_conservation_loss()
```

**Conservation Enforcement:**

For conservation laws, the normal flux must be continuous:

$$F^{(i)} \cdot \vec{n} = F^{(j)} \cdot \vec{n}$$

where $F = \nabla u$ is the flux vector.

### APINN (Augmented PINN)

APINN uses a learnable gating network to automatically determine subdomain blending weights.

```python
from opifex.neural.pinns.domain_decomposition import (
    APINN,
    APINNConfig,
    Subdomain,
    Interface,
)

# Define subdomains
subdomains = [
    Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
    Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
]

interfaces = [
    Interface(
        subdomain_ids=(0, 1),
        points=jnp.array([[0.5]] * 10),
        normal=jnp.array([1.0]),
    )
]

config = APINNConfig(
    temperature=1.0,              # Softmax temperature
    gating_hidden_dims=[16, 16],  # Gating network architecture
    continuity_weight=1.0,        # Interface continuity weight
)

model = APINN(
    input_dim=1,
    output_dim=1,
    subdomains=subdomains,
    interfaces=interfaces,
    hidden_dims=[32, 32],
    config=config,
    rngs=nnx.Rngs(0),
)

# Get learned gating weights
x = jnp.linspace(0, 1, 100).reshape(-1, 1)
gating_weights = model.get_gating_weights(x)  # Shape: (100, 2)
```

**Gating Mechanism:**

The gating network outputs weights through temperature-controlled softmax:

$$g_i(x) = \frac{\exp(z_i(x) / T)}{\sum_j \exp(z_j(x) / T)}$$

- **Lower temperature ($T < 1$):** Sharper, more discrete selection
- **Higher temperature ($T > 1$):** Smoother, more uniform blending

## Base Classes and Utilities

### Subdomain Class

```python
from opifex.neural.pinns.domain_decomposition import Subdomain

# Create a 2D subdomain
subdomain = Subdomain(
    id=0,
    bounds=jnp.array([
        [0.0, 0.5],  # x bounds: [0, 0.5]
        [0.0, 1.0],  # y bounds: [0, 1]
    ]),
    overlap=0.0,  # No overlap (for Schwarz methods)
)

# Check if point is inside
point = jnp.array([0.25, 0.5])
is_inside = subdomain.contains(point)

# Get subdomain properties
center = subdomain.center  # Centroid
volume = subdomain.volume  # Area in 2D
```

### Interface Class

```python
from opifex.neural.pinns.domain_decomposition import Interface

# Create interface between subdomains 0 and 1
interface = Interface(
    subdomain_ids=(0, 1),
    points=jnp.array([
        [0.5, 0.0],
        [0.5, 0.5],
        [0.5, 1.0],
    ]),  # Sample points on interface
    normal=jnp.array([1.0, 0.0]),  # Normal pointing from 0 to 1
)
```

### Automatic Domain Partitioning

```python
from opifex.neural.pinns.domain_decomposition import uniform_partition

# Create uniform partition of a 2D domain
bounds = jnp.array([
    [0.0, 1.0],  # x: [0, 1]
    [0.0, 1.0],  # y: [0, 1]
])

subdomains, interfaces = uniform_partition(
    bounds=bounds,
    num_partitions=(2, 2),   # 2x2 grid = 4 subdomains
    interface_points=20,      # Points per interface
)

# Creates:
# - 4 subdomains in a grid
# - 4 interfaces (2 vertical, 2 horizontal)
```

## Best Practices

### Choosing the Right Method

| Method | Best For | Advantages | Disadvantages |
|--------|----------|------------|---------------|
| **XPINN** | Sharp interfaces, discontinuous solutions | Explicit control, clear separation | Requires interface point tuning |
| **FBPINN** | Smooth solutions, overlapping domains | No explicit interface conditions | Window functions need overlap |
| **CPINN** | Conservation laws, flux-dominated problems | Strong conservation guarantees | More complex loss computation |
| **APINN** | Unknown optimal decomposition | Learns optimal blending | Additional network to train |

### Interface Point Selection

1. **Density:** Use enough points to capture interface behavior (typically 10-50)
2. **Distribution:** Uniform distribution along interface works well for most cases
3. **Normals:** Ensure consistent normal orientation (outward from first subdomain)

### Loss Weighting

```python
# Start with equal weights and adjust based on convergence
config = XPINNConfig(
    continuity_weight=1.0,
    flux_weight=1.0,
    residual_weight=1.0,
)

# If continuity violations persist, increase weight
config = XPINNConfig(
    continuity_weight=10.0,  # Increased
    flux_weight=1.0,
    residual_weight=1.0,
)
```

### Network Architecture

- **Subdomain networks:** Similar architecture to standard PINNs
- **Hidden dimensions:** Typically [32, 32] to [64, 64, 64]
- **Activation:** `tanh` for smooth solutions, `gelu` for faster training

## Training Example

```python
import optax
from flax import nnx

# Create model
model = XPINN(
    input_dim=2,
    output_dim=1,
    subdomains=subdomains,
    interfaces=interfaces,
    hidden_dims=[64, 64],
    rngs=nnx.Rngs(0),
)

# Define PDE residual
def pde_residual(network, x):
    """Laplace equation: u_xx + u_yy = 0"""
    def u_fn(xi):
        return network(xi.reshape(1, -1)).squeeze()

    # Compute Hessian
    hess = jax.hessian(u_fn)
    laplacian = jax.vmap(lambda xi: jnp.trace(hess(xi)))(x)
    return laplacian

# Training step
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(nnx.state(model))

def loss_fn(model):
    # PDE residual for each subdomain
    residual_loss = model.compute_total_residual(
        pde_residual,
        collocation_points_per_subdomain,
    )

    # Interface losses
    interface_loss = model.compute_interface_loss()

    return residual_loss + interface_loss

# Training loop
for step in range(num_steps):
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    nnx.update(model, updates)
```

## See Also

- [Physics-Informed Neural Networks](pinns.md) - Base PINN methods
- [Adaptive Sampling](adaptive-sampling.md) - Residual-based sampling strategies
- [Training Guide](../user-guide/training.md) - General training procedures
- [API Reference](../api/neural.md#domain-decomposition) - Complete API documentation
