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
# # Allen-Cahn Equation PINN
#
# This example demonstrates solving the Allen-Cahn equation using a
# Physics-Informed Neural Network (PINN). The Allen-Cahn equation is a
# reaction-diffusion PDE that models phase separation and interface dynamics
# in materials science.
#
# Reference: DeepXDE's Allen_Cahn.py example

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
print("Opifex Example: Allen-Cahn Equation PINN")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Problem configuration (from DeepXDE)
D = 0.001  # Diffusion coefficient

# Domain bounds
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# Collocation points (from DeepXDE)
N_DOMAIN = 8000  # Interior collocation points
N_BOUNDARY = 400  # Boundary points
N_INITIAL = 800  # Initial condition points

# Network configuration (from DeepXDE: [2] + [20]*3 + [1])
HIDDEN_DIMS = [20, 20, 20]

# Training configuration
EPOCHS = 20000  # Reduced from 40000 for faster demo
LEARNING_RATE = 1e-3

print(f"Diffusion coefficient: D = {D}")
print(f"Domain: x in [{X_MIN}, {X_MAX}], t in [{T_MIN}, {T_MAX}]")
print(f"Collocation: {N_DOMAIN} domain, {N_BOUNDARY} boundary, {N_INITIAL} initial")
print(f"Network: [2] + {HIDDEN_DIMS} + [1]")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Problem Definition
#
# The Allen-Cahn equation is a reaction-diffusion PDE:
#
# $$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + 5(u - u^3)$$
#
# with:
# - **Domain**: $x \in [-1, 1]$, $t \in [0, 1]$
# - **Initial condition**: $u(x, 0) = x^2 \cos(\pi x)$
# - **Boundary conditions**: $u(-1, t) = u(1, t) = -1$
#
# The reaction term $5(u - u^3)$ has stable equilibria at $u = \pm 1$ and
# unstable equilibrium at $u = 0$. This creates phase separation dynamics.


# %%
# Initial and boundary conditions
def initial_condition(x):
    """Initial condition: u(x, 0) = x^2 * cos(pi*x)."""
    return x**2 * jnp.cos(jnp.pi * x)


def boundary_value():
    """Boundary condition: u(+-1, t) = -1."""
    return -1.0


print()
print("Allen-Cahn equation: du/dt = D*d2u/dx2 + 5*(u - u^3)")
print(f"  Diffusion: D = {D}")
print("  Reaction: 5*(u - u^3) with equilibria at u = -1, 0, +1")
print("  IC: u(x, 0) = x^2 * cos(pi*x)")
print("  BC: u(-1, t) = u(1, t) = -1")

# %% [markdown]
# ## PINN Architecture with Hard Constraint
#
# To satisfy IC and BC exactly, we use output transform:
#
# $$u(x, t) = x^2 \cos(\pi x) + t(1-x^2) \hat{u}(x, t)$$
#
# At $t=0$: $u = x^2 \cos(\pi x)$ (IC satisfied)
# At $x=\pm 1$: $u = \cos(\pm\pi) = -1$ (BC satisfied)


# %%
class AllenCahnPINN(nnx.Module):
    """PINN for the Allen-Cahn equation with hard constraints.

    Architecture matches DeepXDE: [2, 20, 20, 20, 1] with tanh activation.
    Output transform enforces IC and BC exactly.
    """

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs):
        """Initialize PINN."""
        super().__init__()

        layers = []
        in_features = 2  # (x, t)

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        layers.append(nnx.Linear(in_features, 1, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, xt: jax.Array) -> jax.Array:
        """Forward pass with hard constraint.

        Output transform: u = x^2*cos(pi*x) + t*(1-x^2)*u_hat
        This enforces:
          - IC: u(x, 0) = x^2*cos(pi*x)
          - BC: u(+-1, t) = cos(+-pi) = -1
        """
        # Neural network output
        h = xt
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        u_hat = self.layers[-1](h)

        # Hard constraint via output transform
        x, t = xt[:, 0:1], xt[:, 1:2]
        ic_term = x**2 * jnp.cos(jnp.pi * x)
        bc_mask = t * (1 - x**2)
        return ic_term + bc_mask * u_hat


# %%
print()
print("Creating PINN model...")

pinn = AllenCahnPINN(hidden_dims=HIDDEN_DIMS, rngs=nnx.Rngs(42))

# Count parameters
n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
print(f"PINN parameters: {n_params:,}")

# %% [markdown]
# ## Collocation Points

# %%
print()
print("Generating collocation points...")

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 5)

# Domain interior points
x_domain = jax.random.uniform(keys[0], (N_DOMAIN,), minval=X_MIN, maxval=X_MAX)
t_domain = jax.random.uniform(keys[1], (N_DOMAIN,), minval=T_MIN, maxval=T_MAX)
xt_domain = jnp.column_stack([x_domain, t_domain])

# Boundary points (x = -1 and x = 1)
t_left = jax.random.uniform(keys[2], (N_BOUNDARY // 2,), minval=T_MIN, maxval=T_MAX)
xt_left = jnp.column_stack([jnp.full(N_BOUNDARY // 2, X_MIN), t_left])

t_right = jax.random.uniform(keys[3], (N_BOUNDARY // 2,), minval=T_MIN, maxval=T_MAX)
xt_right = jnp.column_stack([jnp.full(N_BOUNDARY // 2, X_MAX), t_right])

xt_boundary = jnp.concatenate([xt_left, xt_right], axis=0)

# Initial condition points (t = 0)
x_initial = jax.random.uniform(keys[4], (N_INITIAL,), minval=X_MIN, maxval=X_MAX)
xt_initial = jnp.column_stack([x_initial, jnp.zeros(N_INITIAL)])

print(f"Domain points:   {xt_domain.shape}")
print(f"Boundary points: {xt_boundary.shape}")
print(f"Initial points:  {xt_initial.shape}")

# %% [markdown]
# ## Physics-Informed Loss
#
# The Allen-Cahn equation residual:
# $$\mathcal{L} = |u_t - D \cdot u_{xx} - 5(u - u^3)|^2$$
#
# With hard constraints, no explicit IC/BC loss is needed!


# %%
def compute_pde_residual(pinn, xt):
    """Compute Allen-Cahn PDE residual: u_t - D*u_xx - 5*(u - u^3) = 0."""

    def u_scalar(xt_single):
        """Scalar output for single point."""
        return pinn(xt_single.reshape(1, 2)).squeeze()

    def residual_single(xt_single):
        """Compute residual for single point."""
        # Get u value
        u = u_scalar(xt_single)

        # First derivative (time)
        grad_u = jax.grad(u_scalar)(xt_single)
        u_t = grad_u[1]  # du/dt

        # Second derivative (space)
        def du_dx(xt_s):
            return jax.grad(u_scalar)(xt_s)[0]

        u_xx = jax.grad(du_dx)(xt_single)[0]

        # Allen-Cahn: u_t = D*u_xx + 5*(u - u^3)
        return u_t - D * u_xx - 5.0 * (u - u**3)

    return jax.vmap(residual_single)(xt)


def pde_loss(pinn, xt):
    """Compute PDE residual loss."""
    residual = compute_pde_residual(pinn, xt)
    return jnp.mean(residual**2)


def total_loss(pinn, xt_dom):
    """Total loss (PDE only with hard constraints)."""
    return pde_loss(pinn, xt_dom)


# %% [markdown]
# ## Training

# %%
print()
print("Training PINN...")

opt = nnx.Optimizer(pinn, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(pinn, opt, xt_dom):
    """Single training step."""

    def loss_fn(model):
        return total_loss(model, xt_dom)

    loss, grads = nnx.value_and_grad(loss_fn)(pinn)
    opt.update(pinn, grads)
    return loss


losses = []
for epoch in range(EPOCHS):
    loss = train_step(pinn, opt, xt_domain)
    losses.append(float(loss))

    if (epoch + 1) % 4000 == 0 or epoch == 0:
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

# Check IC satisfaction
u_ic_pred = pinn(xt_initial).squeeze()
u_ic_exact = initial_condition(x_initial)
ic_error = float(jnp.mean(jnp.abs(u_ic_pred - u_ic_exact)))

# Check BC satisfaction
u_bc_pred = pinn(xt_boundary).squeeze()
u_bc_exact = jnp.full(N_BOUNDARY, boundary_value())
bc_error = float(jnp.mean(jnp.abs(u_bc_pred - u_bc_exact)))

# Mean PDE residual
residual = compute_pde_residual(pinn, xt_eval)
mean_residual = float(jnp.mean(jnp.abs(residual)))

print(f"IC error (should be ~0):  {ic_error:.6e}")
print(f"BC error (should be ~0):  {bc_error:.6e}")
print(f"Mean PDE residual:        {mean_residual:.6e}")

# %% [markdown]
# ## Visualization

# %%
# Create output directory
output_dir = Path("docs/assets/examples/allen_cahn_pinn")
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
colors = ["b", "g", "r", "m", "k"]
for t_idx, color in zip(t_indices, colors, strict=True):
    t_val = float(t_eval[t_idx])
    axes[1].plot(
        np.array(x_eval),
        np.array(u_pred_grid[t_idx, :]),
        f"{color}-",
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
# Initial condition and phase evolution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Initial condition comparison
axes[0].plot(
    np.array(x_eval),
    np.array(u_pred_grid[0, :]),
    "b-",
    label="PINN (t=0)",
    linewidth=2,
)
x_fine = jnp.linspace(X_MIN, X_MAX, 200)
axes[0].plot(
    np.array(x_fine),
    np.array(initial_condition(x_fine)),
    "r--",
    label="IC: x^2*cos(pi*x)",
    linewidth=2,
)
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x, 0)")
axes[0].set_title("Initial Condition")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Solution at x=0 over time
x_mid_idx = nx // 2
axes[1].plot(
    np.array(t_eval),
    np.array(u_pred_grid[:, x_mid_idx]),
    "b-",
    label="u(0, t)",
    linewidth=2,
)
axes[1].axhline(y=1, color="g", linestyle="--", alpha=0.5, label="Equilibrium +1")
axes[1].axhline(y=-1, color="r", linestyle="--", alpha=0.5, label="Equilibrium -1")
axes[1].set_xlabel("t")
axes[1].set_ylabel("u(0, t)")
axes[1].set_title("Evolution at x=0")
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
print("Allen-Cahn Equation PINN example completed")
print("=" * 70)
print()
print("Results Summary:")
print(f"  Final loss:          {losses[-1]:.6e}")
print(f"  IC error:            {ic_error:.6e}")
print(f"  BC error:            {bc_error:.6e}")
print(f"  Mean PDE residual:   {mean_residual:.6e}")
print(f"  Parameters:          {n_params:,}")
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
