# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
"""
# GradNorm: Automatic Loss Balancing for PINNs

This example demonstrates how to use GradNorm for automatic loss weight
balancing in multi-objective PINN training. GradNorm dynamically adjusts
weights to equalize gradient contributions across loss components.

**Key Concepts:**
- Multi-objective PINN training (PDE + BC + IC losses)
- Gradient magnitude monitoring
- Automatic weight adaptation via GradNorm
- Training rate balancing across loss components

**SciML Context:**
PINNs with multiple loss terms (PDE residual, boundary conditions, initial
conditions) often suffer from gradient imbalance - one loss dominates and
prevents others from decreasing. GradNorm solves this automatically.

**Key Result:**
GradNorm achieves balanced loss reduction across all components, avoiding
the common failure mode of boundary/initial conditions being poorly satisfied.
"""

# %%
# Configuration
SEED = 42
N_COLLOCATION = 500
N_BOUNDARY = 100
N_INITIAL = 100
LEARNING_RATE = 1e-3
TRAINING_STEPS = 1000
GRADNORM_ALPHA = 1.5  # Asymmetry parameter (0 = equal, higher = more balancing)

# Output directory
OUTPUT_DIR = "docs/assets/examples/gradnorm"

# %%
print("=" * 70)
print("Opifex Example: GradNorm Loss Balancing for PINNs")
print("=" * 70)

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx


print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %%
from opifex.core.physics.gradnorm import (
    compute_gradient_norms,
    GradNormBalancer,
    GradNormConfig,
)


# %% [markdown]
"""
## Step 1: Define the Problem

We solve the heat equation with Dirichlet boundary conditions:
    u_t = alpha * u_xx on [0, 1] x [0, T]
    u(x, 0) = sin(pi*x)
    u(0, t) = u(1, t) = 0

Exact solution: u(x, t) = sin(pi*x) * exp(-pi^2*alpha*t)
"""


# %%
class HeatEquationPINN(nnx.Module):
    """PINN for 1D heat equation."""

    def __init__(self, hidden_dims: list[int] | None = None, *, rngs: nnx.Rngs):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32, 32]
        layers = []
        in_dim = 2  # x, t

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        layers.append(nnx.Linear(in_dim, 1, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, xt: jax.Array) -> jax.Array:
        """Forward pass through the PINN."""
        h = xt
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)


# %%
print()
print("Creating PINN model...")

pinn = HeatEquationPINN(hidden_dims=[32, 32, 32], rngs=nnx.Rngs(SEED))
n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
print("  Architecture: [2] -> [32] -> [32] -> [32] -> [1]")
print(f"  Parameters: {n_params:,}")

# %% [markdown]
"""
## Step 2: Generate Training Data
"""

# %%
print()
print("Generating training data...")

key = jax.random.PRNGKey(SEED)
ALPHA = 0.01  # Thermal diffusivity
T_MAX = 0.5

# Domain points for PDE residual
key, subkey = jax.random.split(key)
x_domain = jax.random.uniform(subkey, (N_COLLOCATION, 1), minval=0.0, maxval=1.0)
key, subkey = jax.random.split(key)
t_domain = jax.random.uniform(subkey, (N_COLLOCATION, 1), minval=0.0, maxval=T_MAX)
xt_domain = jnp.concatenate([x_domain, t_domain], axis=1)

# Boundary points (x=0 and x=1)
key, subkey = jax.random.split(key)
t_boundary = jax.random.uniform(subkey, (N_BOUNDARY, 1), minval=0.0, maxval=T_MAX)
xt_left = jnp.concatenate([jnp.zeros((N_BOUNDARY, 1)), t_boundary], axis=1)
xt_right = jnp.concatenate([jnp.ones((N_BOUNDARY, 1)), t_boundary], axis=1)
xt_boundary = jnp.concatenate([xt_left, xt_right], axis=0)

# Initial condition points (t=0)
key, subkey = jax.random.split(key)
x_initial = jax.random.uniform(subkey, (N_INITIAL, 1), minval=0.0, maxval=1.0)
xt_initial = jnp.concatenate([x_initial, jnp.zeros((N_INITIAL, 1))], axis=1)
u_initial = jnp.sin(jnp.pi * x_initial)  # sin(πx) at t=0

print(f"  Domain points: {xt_domain.shape}")
print(f"  Boundary points: {xt_boundary.shape}")
print(f"  Initial points: {xt_initial.shape}")


# %% [markdown]
"""
## Step 3: Define Loss Functions

We have three loss components:
1. PDE residual: u_t - alpha * u_xx = 0
2. Boundary condition: u(0, t) = u(1, t) = 0
3. Initial condition: u(x, 0) = sin(pi*x)
"""


# %%
def compute_pde_residual(pinn, xt_domain, alpha):
    """Compute heat equation residual: u_t - α * u_xx = 0."""

    def u_scalar(xt_single):
        return pinn(xt_single.reshape(1, 2)).squeeze()

    def residual_single(xt_single):
        # First derivatives
        du = jax.grad(u_scalar)(xt_single)
        du_dt = du[1]

        # Second derivative in x
        def du_dx_fn(xt):
            return jax.grad(u_scalar)(xt)[0]

        d2u_dx2 = jax.grad(du_dx_fn)(xt_single)[0]

        # PDE: u_t - alpha * u_xx = 0
        return du_dt - alpha * d2u_dx2

    return jax.vmap(residual_single)(xt_domain)


def pde_loss_fn(pinn):
    """PDE residual loss."""
    residual = compute_pde_residual(pinn, xt_domain, ALPHA)
    return jnp.mean(residual**2)


def bc_loss_fn(pinn):
    """Boundary condition loss: u(0, t) = u(1, t) = 0."""
    u_bc = pinn(xt_boundary).squeeze()
    return jnp.mean(u_bc**2)


def ic_loss_fn(pinn):
    """Initial condition loss: u(x, 0) = sin(πx)."""
    u_ic = pinn(xt_initial).squeeze()
    u_target = u_initial.squeeze()
    return jnp.mean((u_ic - u_target) ** 2)


# %% [markdown]
"""
## Step 4: Setup GradNorm Balancer

GradNorm automatically balances the three loss components based on their
gradient magnitudes and training rates.
"""

# %%
print()
print("Setting up GradNorm balancer...")

config = GradNormConfig(
    alpha=GRADNORM_ALPHA,  # Asymmetry parameter
    learning_rate=0.01,  # Learning rate for weight updates
    update_frequency=1,  # Update weights every step
)

balancer = GradNormBalancer(
    num_losses=3,  # PDE, BC, IC
    config=config,
    rngs=nnx.Rngs(SEED),
)

print(f"  GradNorm alpha: {config.alpha}")
print(f"  Weight learning rate: {config.learning_rate}")
print(f"  Initial weights: {balancer.weights}")


# %% [markdown]
"""
## Step 5: Train with GradNorm

We train the PINN with automatic loss weight balancing and compare to fixed weights.
"""

# %%
print()
print("Training PINN with GradNorm...")
print("-" * 50)

# Create optimizer
opt = nnx.Optimizer(pinn, optax.adam(LEARNING_RATE), wrt=nnx.Param)

# Loss functions as list
loss_fns = [pde_loss_fn, bc_loss_fn, ic_loss_fn]
loss_names = ["PDE", "BC", "IC"]

# Compute initial losses
initial_losses = jnp.array([fn(pinn) for fn in loss_fns])
balancer.set_initial_losses(initial_losses)
print(
    f"Initial losses: PDE={float(initial_losses[0]):.4f}, "
    f"BC={float(initial_losses[1]):.4f}, IC={float(initial_losses[2]):.4f}"
)

# Training history
history = {
    "step": [],
    "total_loss": [],
    "pde_loss": [],
    "bc_loss": [],
    "ic_loss": [],
    "weight_pde": [],
    "weight_bc": [],
    "weight_ic": [],
}


@nnx.jit
def compute_losses(pinn):
    """Compute all individual losses."""
    return jnp.array([pde_loss_fn(pinn), bc_loss_fn(pinn), ic_loss_fn(pinn)])


def train_step_gradnorm(pinn, opt, balancer):
    """Training step with GradNorm."""
    # Compute individual losses
    losses = compute_losses(pinn)

    # Compute gradient norms
    grad_norms = compute_gradient_norms(pinn, loss_fns)

    # Update GradNorm weights
    initial = balancer.get_initial_losses()
    if initial is not None:
        balancer.update_weights(grad_norms, losses, initial)

    # Compute weighted loss and gradients
    def total_loss_fn(model):
        ls = jnp.array([pde_loss_fn(model), bc_loss_fn(model), ic_loss_fn(model)])
        return balancer.compute_weighted_loss(ls)

    total_loss, grads = nnx.value_and_grad(total_loss_fn)(pinn)
    opt.update(pinn, grads)

    return total_loss, losses


for step in range(TRAINING_STEPS):
    total_loss, losses = train_step_gradnorm(pinn, opt, balancer)

    if step % 100 == 0:
        weights = balancer.weights
        history["step"].append(step)
        history["total_loss"].append(float(total_loss))
        history["pde_loss"].append(float(losses[0]))
        history["bc_loss"].append(float(losses[1]))
        history["ic_loss"].append(float(losses[2]))
        history["weight_pde"].append(float(weights[0]))
        history["weight_bc"].append(float(weights[1]))
        history["weight_ic"].append(float(weights[2]))

        print(
            f"  Step {step:4d}: loss={total_loss:.6e}, "
            f"PDE={losses[0]:.4e}, BC={losses[1]:.4e}, IC={losses[2]:.4e}"
        )
        print(
            f"           weights: PDE={weights[0]:.3f}, "
            f"BC={weights[1]:.3f}, IC={weights[2]:.3f}"
        )

# Final step
weights = balancer.weights
history["step"].append(TRAINING_STEPS)
history["total_loss"].append(float(total_loss))
history["pde_loss"].append(float(losses[0]))
history["bc_loss"].append(float(losses[1]))
history["ic_loss"].append(float(losses[2]))
history["weight_pde"].append(float(weights[0]))
history["weight_bc"].append(float(weights[1]))
history["weight_ic"].append(float(weights[2]))

print(
    f"  Step {TRAINING_STEPS:4d}: loss={total_loss:.6e}, "
    f"PDE={losses[0]:.4e}, BC={losses[1]:.4e}, IC={losses[2]:.4e}"
)


# %% [markdown]
"""
## Step 6: Train with Fixed Weights (Baseline)

For comparison, we train another PINN with fixed equal weights.
"""

# %%
print()
print("Training PINN with fixed weights (baseline)...")
print("-" * 50)

# Create fresh model
pinn_fixed = HeatEquationPINN(hidden_dims=[32, 32, 32], rngs=nnx.Rngs(SEED))
opt_fixed = nnx.Optimizer(pinn_fixed, optax.adam(LEARNING_RATE), wrt=nnx.Param)

fixed_history = {
    "step": [],
    "total_loss": [],
    "pde_loss": [],
    "bc_loss": [],
    "ic_loss": [],
}

FIXED_WEIGHT = 1.0  # Equal weights for all


def make_fixed_pde_loss():
    """Create PDE loss function for fixed weights baseline."""

    def fn(model):
        residual = compute_pde_residual(model, xt_domain, ALPHA)
        return jnp.mean(residual**2)

    return fn


def make_fixed_bc_loss():
    """Create BC loss function for fixed weights baseline."""

    def fn(model):
        u_bc = model(xt_boundary).squeeze()
        return jnp.mean(u_bc**2)

    return fn


def make_fixed_ic_loss():
    """Create IC loss function for fixed weights baseline."""

    def fn(model):
        u_ic = model(xt_initial).squeeze()
        u_target = u_initial.squeeze()
        return jnp.mean((u_ic - u_target) ** 2)

    return fn


@nnx.jit
def train_step_fixed(pinn, opt):
    """Training step with fixed weights."""

    def total_loss_fn(model):
        pde = compute_pde_residual(model, xt_domain, ALPHA)
        pde_loss = jnp.mean(pde**2)
        bc_loss = jnp.mean(model(xt_boundary).squeeze() ** 2)
        ic_loss = jnp.mean((model(xt_initial).squeeze() - u_initial.squeeze()) ** 2)
        return FIXED_WEIGHT * (pde_loss + bc_loss + ic_loss), (
            pde_loss,
            bc_loss,
            ic_loss,
        )

    (total_loss, losses), grads = nnx.value_and_grad(total_loss_fn, has_aux=True)(pinn)
    opt.update(pinn, grads)

    return total_loss, losses


for step in range(TRAINING_STEPS):
    total_loss, losses = train_step_fixed(pinn_fixed, opt_fixed)
    pde_l, bc_l, ic_l = float(losses[0]), float(losses[1]), float(losses[2])

    if step % 100 == 0:
        fixed_history["step"].append(step)
        fixed_history["total_loss"].append(float(total_loss))
        fixed_history["pde_loss"].append(pde_l)
        fixed_history["bc_loss"].append(bc_l)
        fixed_history["ic_loss"].append(ic_l)

        print(
            f"  Step {step:4d}: loss={total_loss:.6e}, "
            f"PDE={pde_l:.4e}, BC={bc_l:.4e}, IC={ic_l:.4e}"
        )

# Final step
fixed_history["step"].append(TRAINING_STEPS)
fixed_history["total_loss"].append(float(total_loss))
fixed_history["pde_loss"].append(pde_l)
fixed_history["bc_loss"].append(bc_l)
fixed_history["ic_loss"].append(ic_l)


# %% [markdown]
"""
## Step 7: Evaluate Solutions
"""

# %%
print()
print("Evaluating solutions...")

# Evaluation grid
x_eval = jnp.linspace(0, 1, 50)
t_eval = jnp.linspace(0, T_MAX, 50)
X, T = jnp.meshgrid(x_eval, t_eval)
xt_eval = jnp.stack([X.ravel(), T.ravel()], axis=1)

# Exact solution
U_exact = jnp.sin(jnp.pi * X) * jnp.exp(-(jnp.pi**2) * ALPHA * T)

# GradNorm solution
U_gradnorm = pinn(xt_eval).squeeze().reshape(50, 50)
l2_gradnorm = float(jnp.sqrt(jnp.mean((U_gradnorm - U_exact) ** 2)))

# Fixed weights solution
U_fixed = pinn_fixed(xt_eval).squeeze().reshape(50, 50)
l2_fixed = float(jnp.sqrt(jnp.mean((U_fixed - U_exact) ** 2)))

print(f"  GradNorm L2 error: {l2_gradnorm:.6e}")
print(f"  Fixed weights L2 error: {l2_fixed:.6e}")
print(f"  Improvement: {(l2_fixed - l2_gradnorm) / l2_fixed * 100:.1f}%")


# %% [markdown]
"""
## Step 8: Visualization
"""

# %%
print()
print("Generating visualizations...")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
mpl.use("Agg")

# %%
# Figure 1: Loss comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Total loss
ax1 = axes[0, 0]
ax1.semilogy(
    history["step"], history["total_loss"], "b-", label="GradNorm", linewidth=2
)
ax1.semilogy(
    fixed_history["step"],
    fixed_history["total_loss"],
    "r--",
    label="Fixed",
    linewidth=2,
)
ax1.set_xlabel("Training Step", fontsize=12)
ax1.set_ylabel("Total Loss (log scale)", fontsize=12)
ax1.set_title("Total Loss Comparison", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Individual losses - GradNorm
ax2 = axes[0, 1]
ax2.semilogy(history["step"], history["pde_loss"], "b-", label="PDE", linewidth=2)
ax2.semilogy(history["step"], history["bc_loss"], "g-", label="BC", linewidth=2)
ax2.semilogy(history["step"], history["ic_loss"], "r-", label="IC", linewidth=2)
ax2.set_xlabel("Training Step", fontsize=12)
ax2.set_ylabel("Loss (log scale)", fontsize=12)
ax2.set_title("GradNorm: Individual Losses", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Individual losses - Fixed weights
ax3 = axes[1, 0]
ax3.semilogy(
    fixed_history["step"], fixed_history["pde_loss"], "b-", label="PDE", linewidth=2
)
ax3.semilogy(
    fixed_history["step"], fixed_history["bc_loss"], "g-", label="BC", linewidth=2
)
ax3.semilogy(
    fixed_history["step"], fixed_history["ic_loss"], "r-", label="IC", linewidth=2
)
ax3.set_xlabel("Training Step", fontsize=12)
ax3.set_ylabel("Loss (log scale)", fontsize=12)
ax3.set_title("Fixed Weights: Individual Losses", fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Weight evolution
ax4 = axes[1, 1]
ax4.plot(history["step"], history["weight_pde"], "b-", label="w_PDE", linewidth=2)
ax4.plot(history["step"], history["weight_bc"], "g-", label="w_BC", linewidth=2)
ax4.plot(history["step"], history["weight_ic"], "r-", label="w_IC", linewidth=2)
ax4.set_xlabel("Training Step", fontsize=12)
ax4.set_ylabel("Weight", fontsize=12)
ax4.set_title("GradNorm Weight Evolution", fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/training_comparison.png")

# %%
# Figure 2: Solution comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Exact solution
ax1 = axes[0]
im1 = ax1.imshow(
    np.array(U_exact),
    extent=[0, 1, 0, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("t", fontsize=12)
ax1.set_title("Exact Solution", fontsize=14)
plt.colorbar(im1, ax=ax1)

# GradNorm solution
ax2 = axes[1]
im2 = ax2.imshow(
    np.array(U_gradnorm),
    extent=[0, 1, 0, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("t", fontsize=12)
ax2.set_title(f"GradNorm (L2={l2_gradnorm:.2e})", fontsize=14)
plt.colorbar(im2, ax=ax2)

# Error
ax3 = axes[2]
error = np.abs(np.array(U_gradnorm - U_exact))
im3 = ax3.imshow(
    error, extent=[0, 1, 0, T_MAX], origin="lower", aspect="auto", cmap="hot"
)
ax3.set_xlabel("x", fontsize=12)
ax3.set_ylabel("t", fontsize=12)
ax3.set_title("Absolute Error", fontsize=14)
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/solution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/solution.png")


# %% [markdown]
"""
## Results Summary
"""

# %%
print()
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()
print("Final Losses:")
print(
    f"  GradNorm: PDE={history['pde_loss'][-1]:.4e}, "
    f"BC={history['bc_loss'][-1]:.4e}, IC={history['ic_loss'][-1]:.4e}"
)
print(
    f"  Fixed:    PDE={fixed_history['pde_loss'][-1]:.4e}, "
    f"BC={fixed_history['bc_loss'][-1]:.4e}, IC={fixed_history['ic_loss'][-1]:.4e}"
)
print()
print("Final Weights (GradNorm):")
print(
    f"  w_PDE={history['weight_pde'][-1]:.3f}, "
    f"w_BC={history['weight_bc'][-1]:.3f}, w_IC={history['weight_ic'][-1]:.3f}"
)
print()
print("Solution Quality:")
print(f"  GradNorm L2 error: {l2_gradnorm:.6e}")
print(f"  Fixed L2 error:    {l2_fixed:.6e}")
print(f"  Improvement:       {(l2_fixed - l2_gradnorm) / l2_fixed * 100:.1f}%")
print()
print("Key Insights:")
print("  1. GradNorm automatically balances loss components")
print("  2. Weights adapt based on gradient magnitudes and training rates")
print("  3. All components (PDE, BC, IC) decrease together with GradNorm")
print("  4. Fixed weights often lead to one component dominating")
print("=" * 70)

# %%
print()
print("GradNorm example completed successfully!")
print(f"Results saved to: {OUTPUT_DIR}")
