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
# # Helmholtz Equation PINN
#
# This example demonstrates solving the 2D Helmholtz equation using a
# Physics-Informed Neural Network (PINN). The Helmholtz equation arises
# in acoustics, electromagnetics, and quantum mechanics (time-independent
# Schrodinger equation).
#
# Reference: DeepXDE's Helmholtz_Dirichlet_2d.py example

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
# Configuration - matching DeepXDE setup
print("=" * 70)
print("Opifex Example: Helmholtz Equation PINN")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Problem configuration (from DeepXDE: n=2)
N_MODES = 2  # Number of wavelengths in each direction
K0 = 2.0 * jnp.pi * N_MODES  # Wave number k0 = 4*pi

# Domain bounds
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0

# Collocation points (from DeepXDE)
N_DOMAIN = 2500  # Interior collocation points
N_BOUNDARY = 400  # Boundary points

# Network configuration (from DeepXDE: [2] + [150]*3 + [1])
HIDDEN_DIMS = [150, 150, 150]

# Training configuration (from DeepXDE: Adam 5000 iter @ lr=1e-3)
EPOCHS = 5000
LEARNING_RATE = 1e-3

# Use hard constraint for BCs
USE_HARD_CONSTRAINT = True

print(f"Wave number: k0 = {float(K0):.4f} (n={N_MODES} modes)")
print(f"Wavelength: {1.0 / N_MODES:.4f}")
print("Domain: [0, 1] x [0, 1]")
print(f"Collocation: {N_DOMAIN} domain, {N_BOUNDARY} boundary")
print(f"Network: [2] + {HIDDEN_DIMS} + [1]")
print(f"Hard BC constraint: {USE_HARD_CONSTRAINT}")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Problem Definition
#
# The Helmholtz equation:
#
# $$-\nabla^2 u - k_0^2 u = f(x, y)$$
#
# with:
# - **Domain**: $[0, 1] \times [0, 1]$
# - **Source term**: $f(x, y) = k_0^2 \sin(k_0 x) \sin(k_0 y)$
# - **Boundary conditions**: $u = 0$ on $\partial\Omega$ (Dirichlet)
# - **Analytical solution**: $u(x, y) = \sin(k_0 x) \sin(k_0 y)$


# %%
# Analytical solution
def exact_solution(xy):
    """Exact solution: u = sin(k0*x) * sin(k0*y)."""
    x, y = xy[:, 0], xy[:, 1]
    return jnp.sin(K0 * x) * jnp.sin(K0 * y)


def source_term(xy):
    """Source term: f = k0^2 * sin(k0*x) * sin(k0*y)."""
    x, y = xy[:, 0], xy[:, 1]
    return K0**2 * jnp.sin(K0 * x) * jnp.sin(K0 * y)


print()
print("Helmholtz equation: -nabla^2(u) - k0^2 * u = f(x,y)")
print(f"  Wave number: k0 = 2*pi*{N_MODES} = {float(K0):.4f}")
print("  Source term: f = k0^2 * sin(k0*x) * sin(k0*y)")
print("  Boundary: u = 0 (Dirichlet)")
print("  Analytical solution: u = sin(k0*x) * sin(k0*y)")

# %% [markdown]
# ## PINN Architecture with Hard Constraint
#
# To exactly satisfy homogeneous Dirichlet BCs, we use a hard constraint:
#
# $$u_{net}(x, y) = x(1-x) \cdot y(1-y) \cdot \hat{u}(x, y)$$
#
# This ensures $u = 0$ on all boundaries automatically.


# %%
class HelmholtzPINN(nnx.Module):
    """PINN for the Helmholtz equation with hard BC constraint.

    Architecture matches DeepXDE: [2, 150, 150, 150, 1] with sin activation.
    Hard constraint: u = x*(1-x)*y*(1-y) * network_output
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
        """Forward pass with hard BC constraint.

        Args:
            xy: Coordinates [batch, 2] where columns are (x, y)

        Returns:
            Solution values [batch, 1]
        """
        # Neural network output
        h = xy
        for layer in self.layers[:-1]:
            h = jnp.sin(layer(h))  # sin activation (from DeepXDE)
        u_hat = self.layers[-1](h)

        if USE_HARD_CONSTRAINT:
            # Hard constraint: u = x*(1-x) * y*(1-y) * u_hat
            # This enforces u = 0 on all boundaries exactly
            x, y = xy[:, 0:1], xy[:, 1:2]
            bc_mask = x * (1 - x) * y * (1 - y)
            return bc_mask * u_hat
        return u_hat


# %%
print()
print("Creating PINN model...")

pinn = HelmholtzPINN(hidden_dims=HIDDEN_DIMS, rngs=nnx.Rngs(42))

# Count parameters
n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
print(f"PINN parameters: {n_params:,}")

# %% [markdown]
# ## Collocation Points

# %%
print()
print("Generating collocation points...")

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 6)

# Domain interior points
x_domain = jax.random.uniform(keys[0], (N_DOMAIN,), minval=X_MIN, maxval=X_MAX)
y_domain = jax.random.uniform(keys[1], (N_DOMAIN,), minval=Y_MIN, maxval=Y_MAX)
xy_domain = jnp.column_stack([x_domain, y_domain])

# Boundary points (for soft constraint if needed, or evaluation)
n_per_edge = N_BOUNDARY // 4

# Bottom edge (y = 0)
x_bottom = jax.random.uniform(keys[2], (n_per_edge,), minval=X_MIN, maxval=X_MAX)
xy_bottom = jnp.column_stack([x_bottom, jnp.zeros(n_per_edge)])

# Top edge (y = 1)
x_top = jax.random.uniform(keys[3], (n_per_edge,), minval=X_MIN, maxval=X_MAX)
xy_top = jnp.column_stack([x_top, jnp.ones(n_per_edge)])

# Left edge (x = 0)
y_left = jax.random.uniform(keys[4], (n_per_edge,), minval=Y_MIN, maxval=Y_MAX)
xy_left = jnp.column_stack([jnp.zeros(n_per_edge), y_left])

# Right edge (x = 1)
y_right = jax.random.uniform(keys[5], (n_per_edge,), minval=Y_MIN, maxval=Y_MAX)
xy_right = jnp.column_stack([jnp.ones(n_per_edge), y_right])

xy_boundary = jnp.concatenate([xy_bottom, xy_top, xy_left, xy_right], axis=0)

print(f"Domain points:   {xy_domain.shape}")
print(f"Boundary points: {xy_boundary.shape}")

# %% [markdown]
# ## Physics-Informed Loss
#
# The Helmholtz equation residual:
# $$\mathcal{L} = |-\nabla^2 u - k_0^2 u - f|^2$$
#
# With hard constraint, no boundary loss is needed!


# %%
def compute_pde_residual(pinn, xy):
    """Compute Helmholtz PDE residual: -nabla^2(u) - k0^2*u - f = 0.

    Args:
        pinn: The PINN model
        xy: Coordinates [batch, 2]

    Returns:
        Residual values [batch]
    """

    def u_scalar(xy_single):
        """Scalar output for single point."""
        return pinn(xy_single.reshape(1, 2)).squeeze()

    def residual_single(xy_single):
        """Compute residual for single point."""
        # Second derivatives using Hessian
        hess = jax.hessian(u_scalar)(xy_single)
        u_xx = hess[0, 0]  # d^2u/dx^2
        u_yy = hess[1, 1]  # d^2u/dy^2
        laplacian = u_xx + u_yy

        # Get u value
        u = u_scalar(xy_single)

        # Source term
        f = K0**2 * jnp.sin(K0 * xy_single[0]) * jnp.sin(K0 * xy_single[1])

        # Helmholtz equation: -laplacian - k0^2*u - f = 0
        return -laplacian - K0**2 * u - f

    return jax.vmap(residual_single)(xy)


def pde_loss(pinn, xy):
    """Compute PDE residual loss."""
    residual = compute_pde_residual(pinn, xy)
    return jnp.mean(residual**2)


def boundary_loss(pinn, xy):
    """Compute boundary loss: u = 0 on boundary (for soft constraint)."""
    u = pinn(xy).squeeze()
    return jnp.mean(u**2)


def total_loss(pinn, xy_dom, xy_bc, lambda_bc=100.0):
    """Total physics-informed loss."""
    loss_pde = pde_loss(pinn, xy_dom)
    if USE_HARD_CONSTRAINT:
        # No boundary loss needed with hard constraint
        return loss_pde
    loss_bc = boundary_loss(pinn, xy_bc)
    return loss_pde + lambda_bc * loss_bc


# %% [markdown]
# ## Training

# %%
print()
print("Training PINN...")

opt = nnx.Optimizer(pinn, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(pinn, opt, xy_dom, xy_bc):
    """Single training step."""

    def loss_fn(model):
        return total_loss(model, xy_dom, xy_bc)

    loss, grads = nnx.value_and_grad(loss_fn)(pinn)
    opt.update(pinn, grads)
    return loss


losses = []
for epoch in range(EPOCHS):
    loss = train_step(pinn, opt, xy_domain, xy_boundary)
    losses.append(float(loss))

    if (epoch + 1) % 1000 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:5d}/{EPOCHS}: loss={loss:.6e}")

print(f"Final loss: {losses[-1]:.6e}")

# %% [markdown]
# ## Evaluation

# %%
print()
print("Evaluating PINN...")

# Create evaluation grid
nx, ny = 100, 100
x_eval = jnp.linspace(X_MIN, X_MAX, nx)
y_eval = jnp.linspace(Y_MIN, Y_MAX, ny)
xx, yy = jnp.meshgrid(x_eval, y_eval)
xy_eval = jnp.column_stack([xx.ravel(), yy.ravel()])

# PINN prediction
u_pred = pinn(xy_eval).squeeze()
u_pred_grid = u_pred.reshape(ny, nx)

# Exact solution
u_exact_grid = exact_solution(xy_eval).reshape(ny, nx)

# Compute errors
error = jnp.abs(u_pred_grid - u_exact_grid)
l2_error = float(
    jnp.sqrt(jnp.sum((u_pred_grid - u_exact_grid) ** 2) / jnp.sum(u_exact_grid**2))
)
max_error = float(jnp.max(error))
mean_error = float(jnp.mean(error))

# Mean PDE residual
residual = compute_pde_residual(pinn, xy_eval)
mean_residual = float(jnp.mean(jnp.abs(residual)))

# Boundary error (should be ~0 with hard constraint)
bc_error = float(jnp.mean(jnp.abs(pinn(xy_boundary).squeeze())))

print(f"Relative L2 error:   {l2_error:.6e}")
print(f"Maximum point error: {max_error:.6e}")
print(f"Mean point error:    {mean_error:.6e}")
print(f"Mean PDE residual:   {mean_residual:.6e}")
print(f"Boundary error:      {bc_error:.6e}")

# %% [markdown]
# ## Visualization

# %%
# Create output directory
output_dir = Path("docs/assets/examples/helmholtz_pinn")
output_dir.mkdir(parents=True, exist_ok=True)

mpl.use("Agg")

# Plot solution comparison
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# PINN solution
im0 = axes[0].imshow(
    np.array(u_pred_grid),
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower",
    cmap="RdBu_r",
)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("PINN Solution")
plt.colorbar(im0, ax=axes[0])

# Exact solution
im1 = axes[1].imshow(
    np.array(u_exact_grid),
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower",
    cmap="RdBu_r",
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Exact Solution")
plt.colorbar(im1, ax=axes[1])

# Error
im2 = axes[2].imshow(
    np.array(error),
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower",
    cmap="hot",
)
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_title(f"Error (L2={l2_error:.2e})")
plt.colorbar(im2, ax=axes[2])

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
# Cross-sections
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# y = 0.5 cross-section
y_idx = ny // 2
axes[0].plot(
    np.array(x_eval), np.array(u_pred_grid[y_idx, :]), "b-", label="PINN", linewidth=2
)
axes[0].plot(
    np.array(x_eval),
    np.array(u_exact_grid[y_idx, :]),
    "r--",
    label="Exact",
    linewidth=2,
)
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x, 0.5)")
axes[0].set_title(f"Cross-section at y = {float(y_eval[y_idx]):.2f}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# x = 0.5 cross-section
x_idx = nx // 2
axes[1].plot(
    np.array(y_eval), np.array(u_pred_grid[:, x_idx]), "b-", label="PINN", linewidth=2
)
axes[1].plot(
    np.array(y_eval),
    np.array(u_exact_grid[:, x_idx]),
    "r--",
    label="Exact",
    linewidth=2,
)
axes[1].set_xlabel("y")
axes[1].set_ylabel("u(0.5, y)")
axes[1].set_title(f"Cross-section at x = {float(x_eval[x_idx]):.2f}")
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
print("Helmholtz Equation PINN example completed")
print("=" * 70)
print()
print("Results Summary:")
print(f"  Final loss:          {losses[-1]:.6e}")
print(f"  Relative L2 error:   {l2_error:.6e}")
print(f"  Maximum error:       {max_error:.6e}")
print(f"  Mean error:          {mean_error:.6e}")
print(f"  Mean PDE residual:   {mean_residual:.6e}")
print(f"  Boundary error:      {bc_error:.6e}")
print(f"  Parameters:          {n_params:,}")
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
