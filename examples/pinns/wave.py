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
# # Wave Equation PINN
#
# This example demonstrates solving the 1D wave equation using a Physics-Informed
# Neural Network (PINN). The wave equation describes propagation of waves in
# strings, acoustics, and electromagnetic fields.
#
# Reference: DeepXDE's wave_1d.py example

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
print("Opifex Example: Wave Equation PINN")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Problem configuration (from DeepXDE)
C = 1.0  # Wave speed (simplified from DeepXDE's C=10 for easier training)

# Domain bounds
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# Collocation points (from DeepXDE: 360 each)
N_DOMAIN = 2000  # Interior collocation points
N_BOUNDARY = 200  # Boundary points
N_INITIAL = 200  # Initial condition points

# Network configuration
HIDDEN_DIMS = [50, 50, 50]

# Training configuration
EPOCHS = 15000
LEARNING_RATE = 1e-3

print(f"Wave speed: c = {C}")
print(f"Domain: x in [{X_MIN}, {X_MAX}], t in [{T_MIN}, {T_MAX}]")
print(f"Collocation: {N_DOMAIN} domain, {N_BOUNDARY} boundary, {N_INITIAL} initial")
print(f"Network: [2] + {HIDDEN_DIMS} + [1]")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Problem Definition
#
# The 1D wave equation:
#
# $$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$
#
# with:
# - **Domain**: $x \in [0, 1]$, $t \in [0, 1]$
# - **Initial condition**: $u(x, 0) = \sin(\pi x)$ (standing wave)
# - **Initial velocity**: $\frac{\partial u}{\partial t}(x, 0) = 0$
# - **Boundary conditions**: $u(0, t) = u(1, t) = 0$ (fixed ends)
#
# **Analytical solution**: $u(x, t) = \sin(\pi x) \cos(c \pi t)$


# %%
# Analytical solution
def exact_solution(x, t):
    """Exact solution for standing wave: u = sin(pi*x) * cos(c*pi*t)."""
    return jnp.sin(jnp.pi * x) * jnp.cos(C * jnp.pi * t)


def initial_condition(x):
    """Initial condition: u(x, 0) = sin(pi*x)."""
    return jnp.sin(jnp.pi * x)


def initial_velocity(x):
    """Initial velocity: u_t(x, 0) = 0."""
    return jnp.zeros_like(x)


print()
print("Wave equation: u_tt = c^2 * u_xx")
print("Initial condition: u(x, 0) = sin(pi*x)")
print("Initial velocity: u_t(x, 0) = 0")
print("Boundary conditions: u(0, t) = u(1, t) = 0")
print("Analytical solution: u(x, t) = sin(pi*x) * cos(c*pi*t)")

# %% [markdown]
# ## PINN Architecture


# %%
class WavePINN(nnx.Module):
    """PINN for the wave equation.

    Simple MLP architecture with tanh activation.
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

pinn = WavePINN(hidden_dims=HIDDEN_DIMS, rngs=nnx.Rngs(42))

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
t_domain = jax.random.uniform(keys[1], (N_DOMAIN,), minval=T_MIN, maxval=T_MAX)
xt_domain = jnp.column_stack([x_domain, t_domain])

# Boundary points (x = 0 and x = 1)
t_left = jax.random.uniform(keys[2], (N_BOUNDARY // 2,), minval=T_MIN, maxval=T_MAX)
xt_left = jnp.column_stack([jnp.zeros(N_BOUNDARY // 2), t_left])

t_right = jax.random.uniform(keys[3], (N_BOUNDARY // 2,), minval=T_MIN, maxval=T_MAX)
xt_right = jnp.column_stack([jnp.ones(N_BOUNDARY // 2), t_right])

xt_boundary = jnp.concatenate([xt_left, xt_right], axis=0)

# Initial condition points (t = 0)
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
# 1. **PDE residual**: $|u_{tt} - c^2 u_{xx}|^2$
# 2. **Boundary loss**: $|u(0, t)|^2 + |u(1, t)|^2$
# 3. **Initial condition loss**: $|u(x, 0) - u_0(x)|^2$
# 4. **Initial velocity loss**: $|u_t(x, 0)|^2$


# %%
def compute_pde_residual(pinn, xt):
    """Compute wave equation PDE residual: u_tt - c^2 * u_xx = 0.

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
        # Second derivatives using Hessian
        hess = jax.hessian(u_scalar)(xt_single)
        u_xx = hess[0, 0]  # d^2u/dx^2
        u_tt = hess[1, 1]  # d^2u/dt^2

        # Wave equation: u_tt - c^2 * u_xx = 0
        return u_tt - C**2 * u_xx

    return jax.vmap(residual_single)(xt)


def compute_time_derivative(pinn, xt):
    """Compute du/dt for initial velocity condition."""

    def u_scalar(xt_single):
        return pinn(xt_single.reshape(1, 2)).squeeze()

    def du_dt_single(xt_single):
        grad = jax.grad(u_scalar)(xt_single)
        return grad[1]  # du/dt

    return jax.vmap(du_dt_single)(xt)


def pde_loss(pinn, xt):
    """Compute PDE residual loss."""
    residual = compute_pde_residual(pinn, xt)
    return jnp.mean(residual**2)


def boundary_loss(pinn, xt):
    """Compute boundary loss: u(0, t) = u(1, t) = 0."""
    u = pinn(xt).squeeze()
    return jnp.mean(u**2)


def initial_loss(pinn, xt, u0):
    """Compute initial condition loss: u(x, 0) = u0(x)."""
    u = pinn(xt).squeeze()
    return jnp.mean((u - u0) ** 2)


def initial_velocity_loss(pinn, xt):
    """Compute initial velocity loss: u_t(x, 0) = 0."""
    u_t = compute_time_derivative(pinn, xt)
    return jnp.mean(u_t**2)


def total_loss(pinn, xt_dom, xt_bc, xt_ic, u_ic, lambda_bc=10.0, lambda_ic=10.0):
    """Total physics-informed loss."""
    loss_pde = pde_loss(pinn, xt_dom)
    loss_bc = boundary_loss(pinn, xt_bc)
    loss_ic = initial_loss(pinn, xt_ic, u_ic)
    loss_vel = initial_velocity_loss(pinn, xt_ic)
    return loss_pde + lambda_bc * loss_bc + lambda_ic * (loss_ic + loss_vel)


# %% [markdown]
# ## Training

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

# %%
print()
print("Evaluating PINN...")

# Create evaluation grid
nx, nt = 100, 100
x_eval = jnp.linspace(X_MIN, X_MAX, nx)
t_eval = jnp.linspace(T_MIN, T_MAX, nt)
xx, tt = jnp.meshgrid(x_eval, t_eval)
xt_eval = jnp.column_stack([xx.ravel(), tt.ravel()])

# PINN prediction
u_pred = pinn(xt_eval).squeeze()
u_pred_grid = u_pred.reshape(nt, nx)

# Exact solution
u_exact_grid = exact_solution(xx, tt)

# Compute errors
error = jnp.abs(u_pred_grid - u_exact_grid)
l2_error = float(
    jnp.sqrt(jnp.sum((u_pred_grid - u_exact_grid) ** 2) / jnp.sum(u_exact_grid**2))
)
max_error = float(jnp.max(error))
mean_error = float(jnp.mean(error))

# Mean PDE residual
residual = compute_pde_residual(pinn, xt_eval)
mean_residual = float(jnp.mean(jnp.abs(residual)))

print(f"Relative L2 error:   {l2_error:.6e}")
print(f"Maximum point error: {max_error:.6e}")
print(f"Mean point error:    {mean_error:.6e}")
print(f"Mean PDE residual:   {mean_residual:.6e}")

# %% [markdown]
# ## Visualization

# %%
# Create output directory
output_dir = Path("docs/assets/examples/wave_pinn")
output_dir.mkdir(parents=True, exist_ok=True)

mpl.use("Agg")

# Plot solution evolution
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# PINN solution heatmap
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

# Exact solution
im1 = axes[1].imshow(
    np.array(u_exact_grid),
    extent=[X_MIN, X_MAX, T_MIN, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
axes[1].set_title("Exact Solution")
plt.colorbar(im1, ax=axes[1])

# Error
im2 = axes[2].imshow(
    np.array(error),
    extent=[X_MIN, X_MAX, T_MIN, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="hot",
)
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
axes[2].set_title(f"Error (max={max_error:.2e})")
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
# Time snapshots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Snapshots at different times
t_indices = [0, nt // 4, nt // 2, 3 * nt // 4]
colors = ["b", "g", "r", "m"]
for t_idx, color in zip(t_indices, colors, strict=True):
    t_val = float(t_eval[t_idx])
    axes[0].plot(
        np.array(x_eval),
        np.array(u_pred_grid[t_idx, :]),
        f"{color}-",
        label=f"PINN t={t_val:.2f}",
        linewidth=2,
    )
    axes[0].plot(
        np.array(x_eval),
        np.array(u_exact_grid[t_idx, :]),
        f"{color}--",
        label=f"Exact t={t_val:.2f}",
        linewidth=1,
        alpha=0.7,
    )
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x, t)")
axes[0].set_title("Solution at Different Times")
axes[0].legend(fontsize=8, ncol=2)
axes[0].grid(True, alpha=0.3)

# Initial condition comparison
axes[1].plot(
    np.array(x_eval), np.array(u_pred_grid[0, :]), "b-", label="PINN", linewidth=2
)
axes[1].plot(
    np.array(x_eval), np.array(u_exact_grid[0, :]), "r--", label="Exact", linewidth=2
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("u(x, 0)")
axes[1].set_title("Initial Condition")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis saved to {output_dir / 'analysis.png'}")

# %%
# Summary
print()
print("=" * 70)
print("Wave Equation PINN example completed")
print("=" * 70)
print()
print("Results Summary:")
print(f"  Final loss:          {losses[-1]:.6e}")
print(f"  Relative L2 error:   {l2_error:.6e}")
print(f"  Maximum error:       {max_error:.6e}")
print(f"  Mean error:          {mean_error:.6e}")
print(f"  Mean PDE residual:   {mean_residual:.6e}")
print(f"  Parameters:          {n_params:,}")
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
