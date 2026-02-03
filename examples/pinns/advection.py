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
# # Advection Equation PINN
#
# This example demonstrates solving the 1D linear advection equation using a
# Physics-Informed Neural Network (PINN). The advection equation describes
# transport of a quantity by a flow field without diffusion.

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
print("Opifex Example: Advection Equation PINN")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Problem configuration
C = 1.0  # Advection velocity

# Domain bounds
X_MIN, X_MAX = 0.0, 2.0
T_MIN, T_MAX = 0.0, 1.0

# Collocation points
N_DOMAIN = 5000  # Interior collocation points
N_BOUNDARY = 200  # Boundary points (inflow)
N_INITIAL = 400  # Initial condition points

# Network configuration
HIDDEN_DIMS = [40, 40, 40]

# Training configuration
EPOCHS = 10000
LEARNING_RATE = 1e-3

print(f"Advection velocity: c = {C}")
print(f"Domain: x in [{X_MIN}, {X_MAX}], t in [{T_MIN}, {T_MAX}]")
print(f"Collocation: {N_DOMAIN} domain, {N_BOUNDARY} inflow, {N_INITIAL} initial")
print(f"Network: [2] + {HIDDEN_DIMS} + [1]")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Problem Definition
#
# The 1D linear advection equation:
#
# $$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0$$
#
# with:
# - **Domain**: $x \in [0, 2]$, $t \in [0, 1]$
# - **Initial condition**: $u(x, 0) = \exp(-(x-0.5)^2 / 0.1)$ (Gaussian pulse)
# - **Inflow BC**: $u(0, t) = \exp(-(0-ct-0.5)^2 / 0.1)$ (matches analytical)
#
# **Analytical solution**: $u(x, t) = u_0(x - ct)$ (translating wave)


# %%
# Analytical solution (translating Gaussian)
def exact_solution(x, t):
    """Exact solution: Gaussian pulse traveling with speed c."""
    x0 = 0.5  # Initial center
    sigma2 = 0.1  # Variance
    return jnp.exp(-((x - C * t - x0) ** 2) / sigma2)


def initial_condition(x):
    """Initial condition: Gaussian pulse centered at x=0.5."""
    x0 = 0.5
    sigma2 = 0.1
    return jnp.exp(-((x - x0) ** 2) / sigma2)


def inflow_condition(t):
    """Inflow BC at x=0: u(0, t) matches analytical solution."""
    x0 = 0.5
    sigma2 = 0.1
    return jnp.exp(-((-C * t - x0) ** 2) / sigma2)


print()
print("Advection equation: du/dt + c*du/dx = 0")
print(f"  Velocity: c = {C}")
print("  IC: u(x, 0) = exp(-(x-0.5)^2 / 0.1)")
print("  BC: u(0, t) = exact solution at inflow")
print("  Solution: u(x, t) = u0(x - c*t)")

# %% [markdown]
# ## PINN Architecture


# %%
class AdvectionPINN(nnx.Module):
    """PINN for the advection equation."""

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs):
        super().__init__()

        layers = []
        in_features = 2  # (x, t)

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        layers.append(nnx.Linear(in_features, 1, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, xt: jax.Array) -> jax.Array:
        """Forward pass through the MLP network."""
        h = xt
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)


# %%
print()
print("Creating PINN model...")

pinn = AdvectionPINN(hidden_dims=HIDDEN_DIMS, rngs=nnx.Rngs(42))

n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
print(f"PINN parameters: {n_params:,}")

# %% [markdown]
# ## Collocation Points

# %%
print()
print("Generating collocation points...")

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 4)

# Domain interior points
x_domain = jax.random.uniform(keys[0], (N_DOMAIN,), minval=X_MIN, maxval=X_MAX)
t_domain = jax.random.uniform(keys[1], (N_DOMAIN,), minval=T_MIN, maxval=T_MAX)
xt_domain = jnp.column_stack([x_domain, t_domain])

# Inflow boundary (x = 0)
t_inflow = jax.random.uniform(keys[2], (N_BOUNDARY,), minval=T_MIN, maxval=T_MAX)
xt_inflow = jnp.column_stack([jnp.zeros(N_BOUNDARY), t_inflow])
u_inflow = inflow_condition(t_inflow)

# Initial condition (t = 0)
x_initial = jax.random.uniform(keys[3], (N_INITIAL,), minval=X_MIN, maxval=X_MAX)
xt_initial = jnp.column_stack([x_initial, jnp.zeros(N_INITIAL)])
u_initial = initial_condition(x_initial)

print(f"Domain points:   {xt_domain.shape}")
print(f"Inflow points:   {xt_inflow.shape}")
print(f"Initial points:  {xt_initial.shape}")

# %% [markdown]
# ## Physics-Informed Loss


# %%
def compute_pde_residual(pinn, xt):
    """Compute advection PDE residual: u_t + c*u_x = 0."""

    def u_scalar(xt_single):
        return pinn(xt_single.reshape(1, 2)).squeeze()

    def residual_single(xt_single):
        grad_u = jax.grad(u_scalar)(xt_single)
        u_x = grad_u[0]
        u_t = grad_u[1]
        return u_t + C * u_x

    return jax.vmap(residual_single)(xt)


def pde_loss(pinn, xt):
    """Compute mean squared PDE residual loss."""
    residual = compute_pde_residual(pinn, xt)
    return jnp.mean(residual**2)


def initial_loss(pinn, xt, u0):
    """Compute mean squared initial condition loss."""
    u = pinn(xt).squeeze()
    return jnp.mean((u - u0) ** 2)


def inflow_loss(pinn, xt, u_in):
    """Compute mean squared inflow boundary loss."""
    u = pinn(xt).squeeze()
    return jnp.mean((u - u_in) ** 2)


def total_loss(pinn, xt_dom, xt_ic, u_ic, xt_in, u_in, lambda_bc=10.0):
    """Compute weighted total loss: PDE + lambda_bc * (IC + inflow)."""
    loss_pde = pde_loss(pinn, xt_dom)
    loss_ic = initial_loss(pinn, xt_ic, u_ic)
    loss_in = inflow_loss(pinn, xt_in, u_in)
    return loss_pde + lambda_bc * (loss_ic + loss_in)


# %% [markdown]
# ## Training

# %%
print()
print("Training PINN...")

opt = nnx.Optimizer(pinn, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(pinn, opt, xt_dom, xt_ic, u_ic, xt_in, u_in):
    """Execute single training step with gradient update."""

    def loss_fn(model):
        return total_loss(model, xt_dom, xt_ic, u_ic, xt_in, u_in)

    loss, grads = nnx.value_and_grad(loss_fn)(pinn)
    opt.update(pinn, grads)
    return loss


losses = []
for epoch in range(EPOCHS):
    loss = train_step(pinn, opt, xt_domain, xt_initial, u_initial, xt_inflow, u_inflow)
    losses.append(float(loss))

    if (epoch + 1) % 2000 == 0 or epoch == 0:
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

# Errors
error = jnp.abs(u_pred_grid - u_exact_grid)
l2_error = float(
    jnp.sqrt(
        jnp.sum((u_pred_grid - u_exact_grid) ** 2) / jnp.sum(u_exact_grid**2 + 1e-10)
    )
)
max_error = float(jnp.max(error))
mean_error = float(jnp.mean(error))

# PDE residual
residual = compute_pde_residual(pinn, xt_eval)
mean_residual = float(jnp.mean(jnp.abs(residual)))

print(f"Relative L2 error:   {l2_error:.6e}")
print(f"Maximum point error: {max_error:.6e}")
print(f"Mean point error:    {mean_error:.6e}")
print(f"Mean PDE residual:   {mean_residual:.6e}")

# %% [markdown]
# ## Visualization

# %%
output_dir = Path("docs/assets/examples/advection_pinn")
output_dir.mkdir(parents=True, exist_ok=True)

mpl.use("Agg")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# PINN solution
im0 = axes[0].imshow(
    np.array(u_pred_grid),
    extent=[X_MIN, X_MAX, T_MIN, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
axes[0].set_title("PINN Solution")
plt.colorbar(im0, ax=axes[0])

# Exact solution
im1 = axes[1].imshow(
    np.array(u_exact_grid),
    extent=[X_MIN, X_MAX, T_MIN, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="viridis",
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
# Time snapshots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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
axes[0].legend(fontsize=7, ncol=2)
axes[0].grid(True, alpha=0.3)

# Characteristic line
axes[1].plot(
    np.array(t_eval),
    np.array(u_pred_grid[:, nx // 4]),
    "b-",
    label="PINN x=0.5",
    linewidth=2,
)
axes[1].plot(
    np.array(t_eval),
    np.array(u_exact_grid[:, nx // 4]),
    "r--",
    label="Exact x=0.5",
    linewidth=2,
)
axes[1].set_xlabel("t")
axes[1].set_ylabel("u(0.5, t)")
axes[1].set_title("Evolution at x=0.5")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis saved to {output_dir / 'analysis.png'}")

# %%
print()
print("=" * 70)
print("Advection Equation PINN example completed")
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
