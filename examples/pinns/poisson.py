# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Poisson Equation PINN
#
# This example demonstrates solving the 2D Poisson equation using a
# Physics-Informed Neural Network (PINN). The Poisson equation is
# fundamental to electrostatics, heat conduction, and potential flow.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx


# %%
# Configuration
print("=" * 70)
print("Opifex Example: Poisson Equation PINN")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Problem configuration
N_INTERIOR = 2000  # Collocation points in interior
N_BOUNDARY = 500  # Points on boundary
EPOCHS = 5000
LEARNING_RATE = 1e-3

# Network configuration
HIDDEN_DIMS = [64, 64, 64]

print(f"Interior points: {N_INTERIOR}, Boundary points: {N_BOUNDARY}")
print(f"Epochs: {EPOCHS}, Learning rate: {LEARNING_RATE}")
print(f"Network: {HIDDEN_DIMS}")

# %% [markdown]
# ## Problem Definition
#
# We solve the Poisson equation on the unit square $[0, 1]^2$:
#
# $$-\nabla^2 u = f(x, y)$$
#
# where $f(x, y) = 2\pi^2 \sin(\pi x) \sin(\pi y)$
#
# with homogeneous Dirichlet boundary conditions: $u = 0$ on $\partial\Omega$.
#
# **Analytical solution**: $u(x, y) = \sin(\pi x) \sin(\pi y)$


# %%
# Define the source term and analytical solution
def source_term(x, y):
    """Source term f(x, y) for the Poisson equation."""
    return 2.0 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def analytical_solution(x, y):
    """Analytical solution u(x, y)."""
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


print()
print("Problem: -∇²u = f(x,y) on [0,1]²")
print("Source term: f(x,y) = 2π² sin(πx) sin(πy)")
print("Boundary: u = 0 (Dirichlet)")
print("Analytical solution: u(x,y) = sin(πx) sin(πy)")

# %% [markdown]
# ## PINN Architecture
#
# Create a simple MLP that takes (x, y) coordinates and outputs u(x, y).


# %%
class PoissonPINN(nnx.Module):
    """PINN for the Poisson equation.

    Simple MLP architecture with tanh activation, suitable for smooth solutions.
    """

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs):
        """Initialize PINN.

        Args:
            hidden_dims: List of hidden layer dimensions
            rngs: Random number generators
        """
        super().__init__()

        layers = []
        in_features = 2  # (x, y)

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        layers.append(nnx.Linear(in_features, 1, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, xy: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            xy: Coordinates [batch, 2]

        Returns:
            Solution values [batch, 1]
        """
        h = xy
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)


# %%
print()
print("Creating PINN model...")

pinn = PoissonPINN(hidden_dims=HIDDEN_DIMS, rngs=nnx.Rngs(42))

# Count parameters
n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
print(f"PINN parameters: {n_params:,}")

# %% [markdown]
# ## Collocation Points
#
# Sample interior points for PDE residual and boundary points for BC enforcement.

# %%
print()
print("Generating collocation points...")

key = jax.random.PRNGKey(42)
key_interior, key_boundary = jax.random.split(key)

# Interior points (uniform in [0, 1]^2)
x_interior = jax.random.uniform(key_interior, (N_INTERIOR, 2))

# Boundary points (sample all 4 edges)
n_per_edge = N_BOUNDARY // 4
keys = jax.random.split(key_boundary, 4)

# Bottom edge (y=0)
bottom = jnp.column_stack(
    [jax.random.uniform(keys[0], (n_per_edge,)), jnp.zeros(n_per_edge)]
)
# Top edge (y=1)
top = jnp.column_stack(
    [jax.random.uniform(keys[1], (n_per_edge,)), jnp.ones(n_per_edge)]
)
# Left edge (x=0)
left = jnp.column_stack(
    [jnp.zeros(n_per_edge), jax.random.uniform(keys[2], (n_per_edge,))]
)
# Right edge (x=1)
right = jnp.column_stack(
    [jnp.ones(n_per_edge), jax.random.uniform(keys[3], (n_per_edge,))]
)

x_boundary = jnp.concatenate([bottom, top, left, right], axis=0)

print(f"Interior points: {x_interior.shape}")
print(f"Boundary points: {x_boundary.shape}")

# %% [markdown]
# ## Physics-Informed Loss
#
# The loss function combines:
# 1. **PDE residual**: $\mathcal{L}_{pde} = \frac{1}{N}\sum_i |{-\nabla^2 u(x_i) - f(x_i)}|^2$
# 2. **Boundary loss**: $\mathcal{L}_{bc} = \frac{1}{M}\sum_j |u(x_j)|^2$


# %%
def compute_laplacian(pinn, xy):
    """Compute Laplacian of PINN output using autodiff.

    Args:
        pinn: The PINN model
        xy: Coordinates [batch, 2]

    Returns:
        Laplacian values [batch, 1]
    """

    def u_scalar(xy_single):
        """Scalar output for single point."""
        return pinn(xy_single.reshape(1, 2)).squeeze()

    def laplacian_single(xy_single):
        """Compute Laplacian for single point using Hessian trace."""
        hessian = jax.hessian(u_scalar)(xy_single)
        return jnp.trace(hessian)

    # Vectorize over batch
    return jax.vmap(laplacian_single)(xy)


def pde_residual_loss(pinn, xy):
    """Compute PDE residual loss: -∇²u - f = 0."""
    laplacian = compute_laplacian(pinn, xy)
    f = source_term(xy[:, 0], xy[:, 1])
    residual = -laplacian - f
    return jnp.mean(residual**2)


def boundary_loss(pinn, xy):
    """Compute boundary loss: u = 0 on boundary."""
    u = pinn(xy).squeeze()
    return jnp.mean(u**2)


def total_loss(pinn, x_int, x_bc, lambda_bc=10.0):
    """Total physics-informed loss."""
    loss_pde = pde_residual_loss(pinn, x_int)
    loss_bc = boundary_loss(pinn, x_bc)
    return loss_pde + lambda_bc * loss_bc


# %% [markdown]
# ## Training

# %%
print()
print("Training PINN...")

opt = nnx.Optimizer(pinn, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(pinn, opt, x_int, x_bc):
    """Single training step."""

    def loss_fn(model):
        return total_loss(model, x_int, x_bc)

    loss, grads = nnx.value_and_grad(loss_fn)(pinn)
    opt.update(pinn, grads)
    return loss


losses = []
for epoch in range(EPOCHS):
    loss = train_step(pinn, opt, x_interior, x_boundary)
    losses.append(float(loss))

    if (epoch + 1) % 1000 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:5d}/{EPOCHS}: loss={loss:.6e}")

print(f"Final loss: {losses[-1]:.6e}")

# %% [markdown]
# ## Evaluation
#
# Compare PINN solution against the analytical solution.

# %%
print()
print("Evaluating PINN...")

# Create evaluation grid
nx, ny = 50, 50
x_eval = jnp.linspace(0, 1, nx)
y_eval = jnp.linspace(0, 1, ny)
xx, yy = jnp.meshgrid(x_eval, y_eval)
xy_eval = jnp.column_stack([xx.ravel(), yy.ravel()])

# PINN prediction
u_pred = pinn(xy_eval).squeeze()
u_pred_grid = u_pred.reshape(ny, nx)

# Analytical solution
u_exact = analytical_solution(xx, yy)

# Compute errors
error = jnp.abs(u_pred_grid - u_exact)
l2_error = jnp.sqrt(jnp.sum((u_pred_grid - u_exact) ** 2) / jnp.sum(u_exact**2))
max_error = jnp.max(error)
mean_error = jnp.mean(error)

print(f"Relative L2 error:   {l2_error:.6e}")
print(f"Maximum point error: {max_error:.6e}")
print(f"Mean point error:    {mean_error:.6e}")

# %% [markdown]
# ## Visualization

# %%
# Create output directory
output_dir = Path("docs/assets/examples/poisson_pinn")
output_dir.mkdir(parents=True, exist_ok=True)

mpl.use("Agg")

# Plot solution comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# PINN solution
im0 = axes[0].imshow(
    np.array(u_pred_grid), extent=[0, 1, 0, 1], origin="lower", cmap="viridis"
)
axes[0].set_title("PINN Solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Analytical solution
im1 = axes[1].imshow(
    np.array(u_exact), extent=[0, 1, 0, 1], origin="lower", cmap="viridis"
)
axes[1].set_title("Analytical Solution")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Point-wise error
im2 = axes[2].imshow(np.array(error), extent=[0, 1, 0, 1], origin="lower", cmap="hot")
axes[2].set_title(f"Error (max={float(max_error):.2e})")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

# Training loss
axes[3].semilogy(losses, linewidth=1)
axes[3].set_xlabel("Epoch")
axes[3].set_ylabel("Loss")
axes[3].set_title("Training Loss")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "solution.png", dpi=150, bbox_inches="tight")
plt.close()
print()
print(f"Solution saved to {output_dir / 'solution.png'}")

# %%
# Cross-section plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# y = 0.5 cross-section
idx_y = ny // 2
axes[0].plot(
    np.array(x_eval), np.array(u_pred_grid[idx_y, :]), "b-", label="PINN", linewidth=2
)
axes[0].plot(
    np.array(x_eval), np.array(u_exact[idx_y, :]), "r--", label="Exact", linewidth=2
)
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x, 0.5)")
axes[0].set_title("Cross-section at y = 0.5")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# x = 0.5 cross-section
idx_x = nx // 2
axes[1].plot(
    np.array(y_eval), np.array(u_pred_grid[:, idx_x]), "b-", label="PINN", linewidth=2
)
axes[1].plot(
    np.array(y_eval), np.array(u_exact[:, idx_x]), "r--", label="Exact", linewidth=2
)
axes[1].set_xlabel("y")
axes[1].set_ylabel("u(0.5, y)")
axes[1].set_title("Cross-section at x = 0.5")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "cross_sections.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Cross-sections saved to {output_dir / 'cross_sections.png'}")

# %%
# Summary
print()
print("=" * 70)
print("Poisson Equation PINN example completed")
print("=" * 70)
print()
print("Results Summary:")
print(f"  Final loss:        {losses[-1]:.6e}")
print(f"  Relative L2 error: {float(l2_error):.6e}")
print(f"  Maximum error:     {float(max_error):.6e}")
print(f"  Mean error:        {float(mean_error):.6e}")
print(f"  Parameters:        {n_params:,}")
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
