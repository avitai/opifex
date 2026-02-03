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
# Residual-based Adaptive Sampling for PINNs

This example demonstrates how to use Residual-based Adaptive Distribution (RAD)
sampling for more efficient PINN training. RAD concentrates collocation points
in regions with high PDE residual.

**Key Concepts:**
- Residual-weighted sampling distribution
- Adaptive collocation point refinement (RAR-D)
- Comparison with uniform sampling

**SciML Context:**
PINNs with uniform collocation point distributions often struggle with
solutions that have localized features (sharp gradients, boundary layers).
Adaptive sampling focuses computational effort where it's needed most.

**Reference Implementation:**
Based on DeepXDE's Residual-based Adaptive Refinement (RAR) algorithm.
"""

# %%
# Configuration
SEED = 42
N_INITIAL_POINTS = 200
N_UNIFORM_POINTS = 400  # Total for uniform baseline
REFINE_FREQUENCY = 200
N_REFINE_POINTS = 50
LEARNING_RATE = 1e-3
TRAINING_STEPS = 1000

# Output directory
OUTPUT_DIR = "docs/assets/examples/adaptive_sampling"

# %%
print("=" * 70)
print("Opifex Example: Residual-based Adaptive Sampling")
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
from opifex.core.training.components.adaptive_sampling import (
    RADConfig,
    RADSampler,
    RARDConfig,
    RARDRefiner,
)


# %% [markdown]
"""
## Step 1: Define the Problem

We solve the Burgers equation with a shock-like solution:
    u_t + u * u_x = nu * u_xx on [0, 2*pi] x [0, 1]
    u(x, 0) = -sin(x)
    u(0, t) = u(2*pi, t) = 0 (periodic-like)

The solution develops a steep gradient (quasi-shock) that requires
high resolution to capture accurately.
"""


# %%
class BurgersPINN(nnx.Module):
    """PINN for 1D Burgers equation."""

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
print("Creating PINN models...")

pinn_adaptive = BurgersPINN(hidden_dims=[32, 32, 32], rngs=nnx.Rngs(SEED))
pinn_uniform = BurgersPINN(hidden_dims=[32, 32, 32], rngs=nnx.Rngs(SEED))

n_params = sum(
    x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn_adaptive, nnx.Param))
)
print("  Architecture: [2] -> [32] -> [32] -> [32] -> [1]")
print(f"  Parameters: {n_params:,}")


# %% [markdown]
"""
## Step 2: Generate Initial Training Data
"""

# %%
print()
print("Generating initial training data...")

key = jax.random.PRNGKey(SEED)
NU = 0.01  # Viscosity (low = sharp gradients)
X_MIN, X_MAX = 0.0, 2.0 * jnp.pi
T_MAX = 0.5

# Initial uniform domain points for adaptive method
key, subkey = jax.random.split(key)
x_domain = jax.random.uniform(subkey, (N_INITIAL_POINTS, 1), minval=X_MIN, maxval=X_MAX)
key, subkey = jax.random.split(key)
t_domain = jax.random.uniform(subkey, (N_INITIAL_POINTS, 1), minval=0.0, maxval=T_MAX)
xt_adaptive = jnp.concatenate([x_domain, t_domain], axis=1)

# Fixed uniform points for baseline
key, subkey = jax.random.split(key)
x_uniform = jax.random.uniform(
    subkey, (N_UNIFORM_POINTS, 1), minval=X_MIN, maxval=X_MAX
)
key, subkey = jax.random.split(key)
t_uniform = jax.random.uniform(subkey, (N_UNIFORM_POINTS, 1), minval=0.0, maxval=T_MAX)
xt_uniform = jnp.concatenate([x_uniform, t_uniform], axis=1)

# Initial condition points
key, subkey = jax.random.split(key)
x_initial = jax.random.uniform(subkey, (100, 1), minval=X_MIN, maxval=X_MAX)
xt_initial = jnp.concatenate([x_initial, jnp.zeros((100, 1))], axis=1)
u_initial = -jnp.sin(x_initial)

# Boundary points
key, subkey = jax.random.split(key)
t_boundary = jax.random.uniform(subkey, (50, 1), minval=0.0, maxval=T_MAX)
xt_left = jnp.concatenate([jnp.full((50, 1), X_MIN), t_boundary], axis=1)
xt_right = jnp.concatenate([jnp.full((50, 1), X_MAX), t_boundary], axis=1)

# Domain bounds for refinement
bounds = jnp.array([[X_MIN, X_MAX], [0.0, T_MAX]])

print(f"  Initial adaptive points: {xt_adaptive.shape}")
print(f"  Uniform baseline points: {xt_uniform.shape}")


# %% [markdown]
"""
## Step 3: Define Loss Functions
"""


# %%
def compute_burgers_residual(pinn, xt, nu):
    """Compute Burgers equation residual: u_t + u*u_x - ν*u_xx = 0."""

    def u_scalar(xt_single):
        return pinn(xt_single.reshape(1, 2)).squeeze()

    def residual_single(xt_single):
        # Forward pass
        u = u_scalar(xt_single)

        # First derivatives
        du = jax.grad(u_scalar)(xt_single)
        du_dx = du[0]
        du_dt = du[1]

        # Second derivative in x
        def du_dx_fn(xt):
            return jax.grad(u_scalar)(xt)[0]

        d2u_dx2 = jax.grad(du_dx_fn)(xt_single)[0]

        # Burgers: u_t + u*u_x - nu*u_xx = 0
        return du_dt + u * du_dx - nu * d2u_dx2

    return jax.vmap(residual_single)(xt)


def pinn_loss(pinn, xt_domain, xt_initial, u_initial, xt_left, xt_right, nu):
    """Total PINN loss."""
    # PDE residual
    residual = compute_burgers_residual(pinn, xt_domain, nu)
    loss_pde = jnp.mean(residual**2)

    # Initial condition
    u_ic = pinn(xt_initial).squeeze()
    u_target = u_initial.squeeze()
    loss_ic = jnp.mean((u_ic - u_target) ** 2)

    # Periodic-like boundary (u at left ≈ u at right)
    u_left = pinn(xt_left).squeeze()
    u_right = pinn(xt_right).squeeze()
    loss_bc = jnp.mean((u_left - u_right) ** 2)

    return loss_pde + 10.0 * loss_ic + loss_bc


# %% [markdown]
"""
## Step 4: Setup Adaptive Sampling

We use RAR-D (Residual-based Adaptive Refinement with Distribution) to
add new collocation points near high-residual regions during training.
"""

# %%
print()
print("Setting up adaptive sampling...")

rad_config = RADConfig(beta=1.0)
rard_config = RARDConfig(
    num_new_points=N_REFINE_POINTS,
    percentile_threshold=90.0,  # Focus on top 10% residual regions
    noise_scale=0.1,
)

sampler = RADSampler(rad_config)
refiner = RARDRefiner(rard_config)

print(f"  RAD beta: {rad_config.beta}")
print(f"  Refinement points per step: {rard_config.num_new_points}")
print(f"  Refinement frequency: {REFINE_FREQUENCY} steps")


# %% [markdown]
"""
## Step 5: Train with Adaptive Sampling
"""

# %%
print()
print("Training PINN with adaptive sampling...")
print("-" * 50)

opt_adaptive = nnx.Optimizer(pinn_adaptive, optax.adam(LEARNING_RATE), wrt=nnx.Param)

adaptive_history = {
    "step": [],
    "loss": [],
    "n_points": [],
    "max_residual": [],
}

# Current collocation points (will grow)
xt_current = xt_adaptive.copy()


def make_loss_fn(xt_domain):
    """Create loss function factory to avoid loop variable capture."""

    def loss_fn(model):
        return pinn_loss(model, xt_domain, xt_initial, u_initial, xt_left, xt_right, NU)

    return loss_fn


for step in range(TRAINING_STEPS):
    # Training step - use factory to avoid B023 loop variable capture
    loss_fn = make_loss_fn(xt_current)

    loss, grads = nnx.value_and_grad(loss_fn)(pinn_adaptive)
    opt_adaptive.update(pinn_adaptive, grads)

    # Periodic refinement
    if step > 0 and step % REFINE_FREQUENCY == 0:
        # Compute residuals at current points
        residuals = compute_burgers_residual(pinn_adaptive, xt_current, NU)

        # Add new points near high-residual regions
        key, subkey = jax.random.split(key)
        xt_current = refiner.refine(xt_current, residuals, bounds, subkey)

        max_res = float(jnp.max(jnp.abs(residuals)))
        print(
            f"  Step {step:4d}: loss={loss:.6e}, points={len(xt_current)}, max_res={max_res:.4e}"
        )

    if step % 100 == 0:
        residuals = compute_burgers_residual(pinn_adaptive, xt_current, NU)
        adaptive_history["step"].append(step)
        adaptive_history["loss"].append(float(loss))
        adaptive_history["n_points"].append(len(xt_current))
        adaptive_history["max_residual"].append(float(jnp.max(jnp.abs(residuals))))

# Final
residuals = compute_burgers_residual(pinn_adaptive, xt_current, NU)
adaptive_history["step"].append(TRAINING_STEPS)
adaptive_history["loss"].append(float(loss))
adaptive_history["n_points"].append(len(xt_current))
adaptive_history["max_residual"].append(float(jnp.max(jnp.abs(residuals))))

print(f"  Final: loss={loss:.6e}, points={len(xt_current)}")


# %% [markdown]
"""
## Step 6: Train with Uniform Sampling (Baseline)
"""

# %%
print()
print("Training PINN with uniform sampling (baseline)...")
print("-" * 50)

opt_uniform = nnx.Optimizer(pinn_uniform, optax.adam(LEARNING_RATE), wrt=nnx.Param)

uniform_history = {
    "step": [],
    "loss": [],
    "max_residual": [],
}


@nnx.jit
def train_step_uniform(pinn, opt):
    """Single training step with uniform sampling."""

    def loss_fn(model):
        return pinn_loss(
            model, xt_uniform, xt_initial, u_initial, xt_left, xt_right, NU
        )

    loss, grads = nnx.value_and_grad(loss_fn)(pinn)
    opt.update(pinn, grads)
    return loss


for step in range(TRAINING_STEPS):
    loss = train_step_uniform(pinn_uniform, opt_uniform)

    if step % 100 == 0:
        residuals = compute_burgers_residual(pinn_uniform, xt_uniform, NU)
        uniform_history["step"].append(step)
        uniform_history["loss"].append(float(loss))
        uniform_history["max_residual"].append(float(jnp.max(jnp.abs(residuals))))

        if step % 200 == 0:
            print(f"  Step {step:4d}: loss={loss:.6e}")

# Final
residuals = compute_burgers_residual(pinn_uniform, xt_uniform, NU)
uniform_history["step"].append(TRAINING_STEPS)
uniform_history["loss"].append(float(loss))
uniform_history["max_residual"].append(float(jnp.max(jnp.abs(residuals))))

print(f"  Final: loss={loss:.6e}")


# %% [markdown]
"""
## Step 7: Evaluate and Visualize
"""

# %%
print()
print("Generating visualizations...")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
mpl.use("Agg")

# Evaluation grid
x_eval = jnp.linspace(X_MIN, X_MAX, 100)
t_eval_final = 0.3  # Evaluate at mid-time where gradients are sharp
xt_eval = jnp.stack([x_eval, jnp.full(100, t_eval_final)], axis=1)

u_adaptive = pinn_adaptive(xt_eval).squeeze()
u_uniform = pinn_uniform(xt_eval).squeeze()

# %%
# Figure 1: Training comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss curves
ax1 = axes[0, 0]
ax1.semilogy(
    adaptive_history["step"],
    adaptive_history["loss"],
    "b-",
    label="Adaptive",
    linewidth=2,
)
ax1.semilogy(
    uniform_history["step"],
    uniform_history["loss"],
    "r--",
    label="Uniform",
    linewidth=2,
)
ax1.set_xlabel("Training Step", fontsize=12)
ax1.set_ylabel("Loss (log scale)", fontsize=12)
ax1.set_title("Training Loss Comparison", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Max residual
ax2 = axes[0, 1]
ax2.semilogy(
    adaptive_history["step"],
    adaptive_history["max_residual"],
    "b-",
    label="Adaptive",
    linewidth=2,
)
ax2.semilogy(
    uniform_history["step"],
    uniform_history["max_residual"],
    "r--",
    label="Uniform",
    linewidth=2,
)
ax2.set_xlabel("Training Step", fontsize=12)
ax2.set_ylabel("Max Residual (log scale)", fontsize=12)
ax2.set_title("Maximum PDE Residual", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Point count growth
ax3 = axes[1, 0]
ax3.plot(
    adaptive_history["step"],
    adaptive_history["n_points"],
    "b-o",
    linewidth=2,
    markersize=4,
)
ax3.axhline(
    y=N_UNIFORM_POINTS, color="r", linestyle="--", label=f"Uniform ({N_UNIFORM_POINTS})"
)
ax3.set_xlabel("Training Step", fontsize=12)
ax3.set_ylabel("Number of Points", fontsize=12)
ax3.set_title("Collocation Point Count", fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Solution comparison
ax4 = axes[1, 1]
ax4.plot(np.array(x_eval), np.array(u_adaptive), "b-", label="Adaptive", linewidth=2)
ax4.plot(np.array(x_eval), np.array(u_uniform), "r--", label="Uniform", linewidth=2)
ax4.set_xlabel("x", fontsize=12)
ax4.set_ylabel(f"u(x, t={t_eval_final})", fontsize=12)
ax4.set_title("Solution at t=0.3", fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/training_comparison.png")

# %%
# Figure 2: Collocation point distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Final adaptive points
ax1 = axes[0]
ax1.scatter(
    np.array(xt_current[:, 0]), np.array(xt_current[:, 1]), s=1, alpha=0.5, c="blue"
)
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("t", fontsize=12)
ax1.set_title(f"Adaptive Points (n={len(xt_current)})", fontsize=14)
ax1.set_xlim(X_MIN, X_MAX)
ax1.set_ylim(0, T_MAX)

# Uniform points
ax2 = axes[1]
ax2.scatter(
    np.array(xt_uniform[:, 0]), np.array(xt_uniform[:, 1]), s=1, alpha=0.5, c="red"
)
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("t", fontsize=12)
ax2.set_title(f"Uniform Points (n={N_UNIFORM_POINTS})", fontsize=14)
ax2.set_xlim(X_MIN, X_MAX)
ax2.set_ylim(0, T_MAX)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/point_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/point_distribution.png")


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
print("Training Results:")
print(f"  Adaptive final loss:    {adaptive_history['loss'][-1]:.6e}")
print(f"  Uniform final loss:     {uniform_history['loss'][-1]:.6e}")
print(f"  Adaptive final points:  {adaptive_history['n_points'][-1]}")
print(f"  Uniform points:         {N_UNIFORM_POINTS}")
print()
print("Max PDE Residual:")
print(f"  Adaptive: {adaptive_history['max_residual'][-1]:.6e}")
print(f"  Uniform:  {uniform_history['max_residual'][-1]:.6e}")
print()
print("Key Insights:")
print("  1. Adaptive sampling concentrates points near high-residual regions")
print("  2. RAR-D adds points where the PDE is least satisfied")
print("  3. Better residual distribution often leads to lower overall error")
print("  4. Adaptive methods are especially useful for solutions with sharp features")
print("=" * 70)

# %%
print()
print("Adaptive sampling example completed successfully!")
print(f"Results saved to: {OUTPUT_DIR}")
