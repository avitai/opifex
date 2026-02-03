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
# # Diffusion-Reaction Equation PINN
#
# This example demonstrates solving a diffusion-reaction equation using a PINN.
# The problem features multiple frequency components that the network must learn.
#
# **Reference**: DeepXDE `examples/pinn_forward/diffusion_reaction.py`

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
print("Opifex Example: Diffusion-Reaction Equation PINN")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Problem configuration
D = 1.0  # Diffusion coefficient

# Domain bounds (matching DeepXDE)
X_MIN, X_MAX = -jnp.pi, jnp.pi
T_MIN, T_MAX = 0.0, 1.0

# Collocation points
N_DOMAIN = 2000
N_BOUNDARY = 100
N_INITIAL = 200

# Network configuration (matching DeepXDE: [2] + [30]*6 + [1])
HIDDEN_DIMS = [30, 30, 30, 30, 30, 30]

# Training configuration
EPOCHS = 15000
LEARNING_RATE = 1e-3

print()
print(f"Diffusion coefficient: D = {D}")
print(f"Domain: x in [{float(X_MIN):.4f}, {float(X_MAX):.4f}], t in [{T_MIN}, {T_MAX}]")
print(f"Collocation: {N_DOMAIN} domain, {N_BOUNDARY} boundary, {N_INITIAL} initial")
print(f"Network: [2] + {HIDDEN_DIMS} + [1]")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Problem Definition
#
# Diffusion-reaction equation:
#
# $$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + f(x, t)$$
#
# where the source term $f$ is chosen so the exact solution is a sum of sine waves:
#
# $$u(x, t) = e^{-t} \left( \sin(x) + \frac{\sin(2x)}{2} + \frac{\sin(3x)}{3} +
#                          \frac{\sin(4x)}{4} + \frac{\sin(8x)}{8} \right)$$


# %%
def exact_solution(x, t):
    """Exact solution: sum of sine waves with exponential decay."""
    return jnp.exp(-t) * (
        jnp.sin(x)
        + jnp.sin(2 * x) / 2
        + jnp.sin(3 * x) / 3
        + jnp.sin(4 * x) / 4
        + jnp.sin(8 * x) / 8
    )


def source_term(x, t):
    """Source term f(x, t) for the manufactured solution.

    Computed from: f = u_t - D*u_xx
    """
    return jnp.exp(-t) * (
        3 * jnp.sin(2 * x) / 2
        + 8 * jnp.sin(3 * x) / 3
        + 15 * jnp.sin(4 * x) / 4
        + 63 * jnp.sin(8 * x) / 8
    )


def initial_condition(x):
    """Initial condition: u(x, 0)."""
    return exact_solution(x, 0.0)


print()
print("Diffusion-reaction: du/dt = D*d^2u/dx^2 + f(x,t)")
print(f"  Diffusion: D = {D}")
print("  Solution: sum of sin(kx)/k terms with exp(-t) decay")
print("  BC: u(-pi, t) = u(pi, t) = 0 (periodic-like)")
print("  IC: u(x, 0) = sin(x) + sin(2x)/2 + ...")


# %% [markdown]
# ## PINN with Hard Constraint


# %%
class DiffusionReactionPINN(nnx.Module):
    """PINN for diffusion-reaction with hard IC and BC constraint."""

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs):
        """Initialize the PINN."""
        super().__init__()

        layers = []
        in_features = 2  # (x, t)

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        layers.append(nnx.Linear(in_features, 1, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, xt: jax.Array) -> jax.Array:
        """Forward pass with hard constraint for IC and BC."""
        x, t = xt[:, 0:1], xt[:, 1:2]

        # Network output
        h = xt
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        u_hat = self.layers[-1](h)

        # Hard constraint (matching DeepXDE output_transform):
        # u = t * (pi^2 - x^2) * u_hat + IC(x)
        # This enforces:
        # - At t=0: u = IC(x)
        # - At x=+/-pi: u = IC(+/-pi) = 0 (since sin(k*pi) = 0)
        ic_term = (
            jnp.sin(x)
            + jnp.sin(2 * x) / 2
            + jnp.sin(3 * x) / 3
            + jnp.sin(4 * x) / 4
            + jnp.sin(8 * x) / 8
        )
        bc_mask = t * (jnp.pi**2 - x**2)

        return bc_mask * u_hat + ic_term


# %%
print()
print("Creating PINN model...")

pinn = DiffusionReactionPINN(hidden_dims=HIDDEN_DIMS, rngs=nnx.Rngs(42))

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

print(f"Domain points: {xt_domain.shape}")

# %% [markdown]
# ## Physics-Informed Loss


# %%
def compute_pde_residual(pinn, xt):
    """Compute diffusion-reaction PDE residual."""

    def u_scalar(xt_single):
        """Scalar version for differentiation."""
        return pinn(xt_single.reshape(1, 2)).squeeze()

    def residual_single(xt_single):
        """Compute residual at single point."""
        x, t = xt_single[0], xt_single[1]

        # Derivatives
        grad_u = jax.grad(u_scalar)(xt_single)
        u_t = grad_u[1]

        hess = jax.hessian(u_scalar)(xt_single)
        u_xx = hess[0, 0]

        # Source term
        f = source_term(x, t)

        # Residual: u_t - D*u_xx - f = 0
        return u_t - D * u_xx - f

    return jax.vmap(residual_single)(xt)


def pde_loss(pinn, xt):
    """Compute mean squared PDE residual."""
    residual = compute_pde_residual(pinn, xt)
    return jnp.mean(residual**2)


# %% [markdown]
# ## Training

# %%
print()
print("Training PINN...")

opt = nnx.Optimizer(pinn, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(pinn, opt, xt_dom):
    """Perform one training step."""

    def loss_fn(model):
        return pde_loss(model, xt_dom)

    loss, grads = nnx.value_and_grad(loss_fn)(pinn)
    opt.update(pinn, grads)
    return loss


losses = []
for epoch in range(EPOCHS):
    loss = train_step(pinn, opt, xt_domain)
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
mean_residual = float(jnp.mean(jnp.abs(compute_pde_residual(pinn, xt_eval))))

# IC/BC errors (should be ~0 due to hard constraint)
x_ic = jnp.linspace(X_MIN, X_MAX, 100)
xt_ic = jnp.column_stack([x_ic, jnp.zeros(100)])
u_ic_pred = pinn(xt_ic).squeeze()
u_ic_exact = initial_condition(x_ic)
ic_error = float(jnp.mean(jnp.abs(u_ic_pred - u_ic_exact)))

print(f"Relative L2 error:   {l2_error:.6e}")
print(f"Maximum point error: {max_error:.6e}")
print(f"Mean point error:    {mean_error:.6e}")
print(f"Mean PDE residual:   {mean_residual:.6e}")
print(f"IC error (hard):     {ic_error:.6e}")

# %% [markdown]
# ## Visualization

# %%
output_dir = Path("docs/assets/examples/diffusion_reaction_pinn")
output_dir.mkdir(parents=True, exist_ok=True)

mpl.use("Agg")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# PINN solution
im0 = axes[0].imshow(
    np.array(u_pred_grid),
    extent=[float(X_MIN), float(X_MAX), T_MIN, T_MAX],
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
    extent=[float(X_MIN), float(X_MAX), T_MIN, T_MAX],
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
    extent=[float(X_MIN), float(X_MAX), T_MIN, T_MAX],
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

# Frequency content comparison at t=0
axes[1].plot(np.array(x_eval), np.array(u_ic_pred), "b-", label="PINN IC", linewidth=2)
axes[1].plot(
    np.array(x_eval),
    np.array(u_ic_exact),
    "r--",
    label="Exact IC",
    linewidth=2,
    alpha=0.7,
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("u(x, 0)")
axes[1].set_title("Initial Condition (Hard Constraint)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis saved to {output_dir / 'analysis.png'}")

# %%
print()
print("=" * 70)
print("Diffusion-Reaction Equation PINN example completed")
print("=" * 70)
print()
print("Results Summary:")
print(f"  Final loss:          {losses[-1]:.6e}")
print(f"  Relative L2 error:   {l2_error:.6e}")
print(f"  Maximum error:       {max_error:.6e}")
print(f"  Mean PDE residual:   {mean_residual:.6e}")
print(f"  IC error (hard):     {ic_error:.6e}")
print(f"  Parameters:          {n_params:,}")
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
