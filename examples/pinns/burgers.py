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
# # Burgers Equation PINN
#
# This example demonstrates solving the viscous Burgers equation using a
# Physics-Informed Neural Network (PINN). The Burgers equation is a
# fundamental nonlinear PDE used in fluid mechanics and shock wave theory.
#
# Reference: DeepXDE's Burgers example (pinn_forward/Burgers.py)

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
print("Opifex Example: Burgers Equation PINN")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Problem configuration (from DeepXDE)
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 0.99
NU = 0.01 / jnp.pi  # Viscosity coefficient

# Collocation points (from DeepXDE: num_domain=2540, num_boundary=80, num_initial=160)
N_DOMAIN = 2540  # Interior collocation points
N_BOUNDARY = 80  # Boundary points per edge
N_INITIAL = 160  # Initial condition points

# Network configuration (from DeepXDE: [2] + [20]*3 + [1])
HIDDEN_DIMS = [20, 20, 20]

# Training configuration (from DeepXDE: Adam 15000 iter @ lr=1e-3)
EPOCHS = 15000
LEARNING_RATE = 1e-3

print(f"Domain: x ∈ [{X_MIN}, {X_MAX}], t ∈ [{T_MIN}, {T_MAX}]")
print(f"Viscosity: nu = {float(NU):.6f}")
print(f"Collocation: {N_DOMAIN} domain, {N_BOUNDARY} boundary, {N_INITIAL} initial")
print(f"Network: [2] + {HIDDEN_DIMS} + [1]")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Problem Definition
#
# The viscous Burgers equation:
#
# $$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$
#
# with:
# - **Domain**: $x \in [-1, 1]$, $t \in [0, 0.99]$
# - **Initial condition**: $u(x, 0) = -\sin(\pi x)$
# - **Boundary conditions**: $u(-1, t) = u(1, t) = 0$


# %%
# Define initial condition (from DeepXDE: -sin(πx))
def initial_condition(x):
    """Initial condition: u(x, 0) = -sin(πx)."""
    return -jnp.sin(jnp.pi * x)


# Boundary conditions: u = 0 at x = ±1
def boundary_condition(t):
    """Boundary condition: u(±1, t) = 0."""
    return jnp.zeros_like(t)


print()
print("Burgers equation: du/dt + u*du/dx = nu*d2u/dx2")
print("Initial condition: u(x, 0) = -sin(πx)")
print("Boundary conditions: u(±1, t) = 0")

# %% [markdown]
# ## PINN Architecture
#
# Network from DeepXDE: [2] + [20]*3 + [1] with tanh activation.


# %%
class BurgersPINN(nnx.Module):
    """PINN for the Burgers equation.

    Architecture matches DeepXDE: [2, 20, 20, 20, 1] with tanh activation.
    """

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs):
        """Initialize PINN.

        Args:
            hidden_dims: List of hidden layer dimensions
            rngs: Random number generators
        """
        super().__init__()

        layers = []
        in_features = 2  # (x, t)

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        layers.append(nnx.Linear(in_features, 1, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, xt: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            xt: Coordinates [batch, 2] where columns are (x, t)

        Returns:
            Solution values [batch, 1]
        """
        h = xt
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)


# %%
print()
print("Creating PINN model...")

pinn = BurgersPINN(hidden_dims=HIDDEN_DIMS, rngs=nnx.Rngs(42))

# Count parameters
n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
print(f"PINN parameters: {n_params:,}")

# %% [markdown]
# ## Collocation Points
#
# Sample points following DeepXDE's distribution:
# - Domain: 2540 random points in [-1,1] x [0,0.99]
# - Boundary: 80 points at x = ±1
# - Initial: 160 points at t = 0

# %%
print()
print("Generating collocation points...")

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 5)

# Domain interior points (matching DeepXDE's num_domain=2540)
x_domain = jax.random.uniform(keys[0], (N_DOMAIN,), minval=X_MIN, maxval=X_MAX)
t_domain = jax.random.uniform(keys[1], (N_DOMAIN,), minval=T_MIN, maxval=T_MAX)
xt_domain = jnp.column_stack([x_domain, t_domain])

# Boundary points (matching DeepXDE's num_boundary=80)
# x = -1 boundary
t_left = jax.random.uniform(keys[2], (N_BOUNDARY // 2,), minval=T_MIN, maxval=T_MAX)
xt_left = jnp.column_stack([jnp.full(N_BOUNDARY // 2, X_MIN), t_left])

# x = +1 boundary
t_right = jax.random.uniform(keys[3], (N_BOUNDARY // 2,), minval=T_MIN, maxval=T_MAX)
xt_right = jnp.column_stack([jnp.full(N_BOUNDARY // 2, X_MAX), t_right])

xt_boundary = jnp.concatenate([xt_left, xt_right], axis=0)

# Initial condition points (matching DeepXDE's num_initial=160)
x_initial = jax.random.uniform(keys[4], (N_INITIAL,), minval=X_MIN, maxval=X_MAX)
xt_initial = jnp.column_stack([x_initial, jnp.zeros(N_INITIAL)])
u_initial = initial_condition(x_initial)

print(f"Domain points:   {xt_domain.shape}")
print(f"Boundary points: {xt_boundary.shape}")
print(f"Initial points:  {xt_initial.shape}")

# %% [markdown]
# ## Physics-Informed Loss
#
# The loss function combines:
# 1. **PDE residual**: $|\partial_t u + u \cdot \partial_x u - \nu \cdot \partial_{xx} u|^2$
# 2. **Boundary loss**: $|u(±1, t)|^2$
# 3. **Initial condition loss**: $|u(x, 0) - u_0(x)|^2$


# %%
def compute_pde_residual(pinn, xt):
    """Compute Burgers PDE residual: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0.

    Args:
        pinn: The PINN model
        xt: Coordinates [batch, 2] where columns are (x, t)

    Returns:
        Residual values [batch]
    """

    def u_scalar(xt_single):
        """Scalar output for single point."""
        return pinn(xt_single.reshape(1, 2)).squeeze()

    def residual_single(xt_single):
        """Compute residual for single point."""
        # First derivatives
        grad_u = jax.grad(u_scalar)(xt_single)
        du_dx = grad_u[0]  # ∂u/∂x
        du_dt = grad_u[1]  # ∂u/∂t

        # Second derivative (∂²u/∂x²)
        def du_dx_fn(xt_s):
            return jax.grad(u_scalar)(xt_s)[0]

        d2u_dx2 = jax.grad(du_dx_fn)(xt_single)[0]

        # Get u value
        u = u_scalar(xt_single)

        # Burgers equation: du/dt + u*du/dx - nu*d2u/dx2 = 0
        return du_dt + u * du_dx - NU * d2u_dx2

    return jax.vmap(residual_single)(xt)


def pde_loss(pinn, xt):
    """Compute PDE residual loss."""
    residual = compute_pde_residual(pinn, xt)
    return jnp.mean(residual**2)


def boundary_loss(pinn, xt):
    """Compute boundary loss: u(±1, t) = 0."""
    u = pinn(xt).squeeze()
    return jnp.mean(u**2)


def initial_loss(pinn, xt, u0):
    """Compute initial condition loss: u(x, 0) = u0(x)."""
    u = pinn(xt).squeeze()
    return jnp.mean((u - u0) ** 2)


def total_loss(pinn, xt_dom, xt_bc, xt_ic, u_ic):
    """Total physics-informed loss."""
    loss_pde = pde_loss(pinn, xt_dom)
    loss_bc = boundary_loss(pinn, xt_bc)
    loss_ic = initial_loss(pinn, xt_ic, u_ic)
    return loss_pde + loss_bc + loss_ic


# %% [markdown]
# ## Training
#
# Following DeepXDE: Adam optimizer with lr=1e-3 for 15000 iterations.

# %%
print()
print("Training PINN...")

opt = nnx.Optimizer(pinn, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(pinn, opt, xt_dom, xt_bc, xt_ic, u_ic):
    """Single training step."""

    def loss_fn(model):
        return total_loss(model, xt_dom, xt_bc, xt_ic, u_ic)

    loss, grads = nnx.value_and_grad(loss_fn)(pinn)
    opt.update(pinn, grads)
    return loss


losses = []
for epoch in range(EPOCHS):
    loss = train_step(pinn, opt, xt_domain, xt_boundary, xt_initial, u_initial)
    losses.append(float(loss))

    if (epoch + 1) % 3000 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:5d}/{EPOCHS}: loss={loss:.6e}")

print(f"Final loss: {losses[-1]:.6e}")

# %% [markdown]
# ## Evaluation
#
# Evaluate the PINN on a regular grid and compute the PDE residual.

# %%
print()
print("Evaluating PINN...")

# Create evaluation grid
nx, nt = 256, 100
x_eval = jnp.linspace(X_MIN, X_MAX, nx)
t_eval = jnp.linspace(T_MIN, T_MAX, nt)
xx, tt = jnp.meshgrid(x_eval, t_eval)
xt_eval = jnp.column_stack([xx.ravel(), tt.ravel()])

# PINN prediction
u_pred = pinn(xt_eval).squeeze()
u_pred_grid = u_pred.reshape(nt, nx)

# Compute mean PDE residual
residual = compute_pde_residual(pinn, xt_eval)
mean_residual = float(jnp.mean(jnp.abs(residual)))

print(f"Mean PDE residual: {mean_residual:.6e}")

# Check initial condition
u_initial_pred = pinn(jnp.column_stack([x_eval, jnp.zeros(nx)])).squeeze()
u_initial_exact = initial_condition(x_eval)
ic_error = float(jnp.mean(jnp.abs(u_initial_pred - u_initial_exact)))
print(f"Initial condition error: {ic_error:.6e}")

# Check boundary conditions
u_bc_left = pinn(jnp.column_stack([jnp.full(nt, X_MIN), t_eval])).squeeze()
u_bc_right = pinn(jnp.column_stack([jnp.full(nt, X_MAX), t_eval])).squeeze()
bc_error = float(jnp.mean(jnp.abs(u_bc_left)) + jnp.mean(jnp.abs(u_bc_right))) / 2
print(f"Boundary condition error: {bc_error:.6e}")

# %% [markdown]
# ## Visualization

# %%
# Create output directory
output_dir = Path("docs/assets/examples/burgers_pinn")
output_dir.mkdir(parents=True, exist_ok=True)

mpl.use("Agg")

# Plot solution evolution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Solution heatmap
im0 = axes[0].imshow(
    np.array(u_pred_grid),
    extent=[X_MIN, X_MAX, T_MIN, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
)
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
axes[0].set_title("PINN Solution u(x, t)")
plt.colorbar(im0, ax=axes[0])

# Time snapshots
t_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]
colors = plt.cm.viridis(np.linspace(0, 1, len(t_indices)))
for i, t_idx in enumerate(t_indices):
    t_val = float(t_eval[t_idx])
    axes[1].plot(
        np.array(x_eval),
        np.array(u_pred_grid[t_idx, :]),
        color=colors[i],
        label=f"t={t_val:.2f}",
        linewidth=1.5,
    )
axes[1].set_xlabel("x")
axes[1].set_ylabel("u(x, t)")
axes[1].set_title("Solution at Different Times")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Training loss
axes[2].semilogy(losses, linewidth=1)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")
axes[2].set_title("Training Loss")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "solution.png", dpi=150, bbox_inches="tight")
plt.close()
print()
print(f"Solution saved to {output_dir / 'solution.png'}")

# %%
# Initial condition comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Initial condition
axes[0].plot(
    np.array(x_eval), np.array(u_initial_exact), "b-", label="Exact", linewidth=2
)
axes[0].plot(
    np.array(x_eval), np.array(u_initial_pred), "r--", label="PINN", linewidth=2
)
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x, 0)")
axes[0].set_title("Initial Condition Comparison")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PDE residual
residual_grid = np.array(jnp.abs(residual).reshape(nt, nx))
im1 = axes[1].imshow(
    residual_grid,
    extent=[X_MIN, X_MAX, T_MIN, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="hot",
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
axes[1].set_title(f"PDE Residual (mean={mean_residual:.2e})")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig(output_dir / "analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis saved to {output_dir / 'analysis.png'}")

# %%
# Summary
print()
print("=" * 70)
print("Burgers Equation PINN example completed")
print("=" * 70)
print()
print("Results Summary:")
print(f"  Final loss:          {losses[-1]:.6e}")
print(f"  Mean PDE residual:   {mean_residual:.6e}")
print(f"  IC error:            {ic_error:.6e}")
print(f"  BC error:            {bc_error:.6e}")
print(f"  Parameters:          {n_params:,}")
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
