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
# # XPINN: Extended PINN on Viscous Burgers Equation
#
# This example demonstrates solving the 1D viscous Burgers equation using XPINN
# (Extended Physics-Informed Neural Network). XPINNs use non-overlapping
# subdomains with explicit interface conditions for continuity and flux matching.
#
# **Reference:** Ameya D. Jagtap, George Em Karniadakis.
# "Extended Physics-Informed Neural Networks (XPINNs):
# A Generalized Space-Time Domain Decomposition Based Deep Learning Framework
# for Nonlinear Partial Differential Equations" (2020)
# https://github.com/AmeyaJagtap/XPINNs
#
# **Problem Reference:** FBPINNs/fbpinns/problems.py BurgersEquation2D

# %% [markdown]
# ## Setup and Imports

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

from opifex.neural.pinns.domain_decomposition import (
    Interface,
    Subdomain,
    XPINN,
    XPINNConfig,
)


print("=" * 70)
print("Opifex Example: XPINN on 1D Viscous Burgers Equation")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## Configuration
#
# Following the FBPINNs/DeepXDE Burgers equation setup:
# - Domain: x in [-1, 1], t in [0, 1]
# - Viscosity: nu = 0.01/pi
# - IC: u(x, 0) = -sin(pi*x)
# - BC: u(-1, t) = u(1, t) = 0

# %%
# Problem configuration (from FBPINNs reference)
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
NU = 0.01 / jnp.pi  # Viscosity coefficient

# Training configuration
N_DOMAIN = 500  # Points per subdomain
N_BOUNDARY = 100  # Boundary points per edge
N_INITIAL = 200  # Initial condition points
N_INTERFACE = 50  # Points per interface
EPOCHS = 15000
LEARNING_RATE = 0.001

# XPINN configuration (2 subdomains in x-direction)
NUM_SUBDOMAINS = 2
HIDDEN_DIMS = [40, 40, 40]

print()
print("Viscous Burgers: du/dt + u*du/dx = nu*d^2u/dx^2")
print(f"  Viscosity: nu = 0.01/pi ≈ {float(NU):.6f}")
print(f"Domain: x in [{X_MIN}, {X_MAX}], t in [{T_MIN}, {T_MAX}]")
print(f"Subdomains: {NUM_SUBDOMAINS}")
print(f"Collocation: {NUM_SUBDOMAINS * N_DOMAIN} domain, {2 * N_BOUNDARY} boundary")
print(f"Network per subdomain: [2] + {HIDDEN_DIMS} + [1]")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Define the Burgers Problem
#
# The 1D viscous Burgers equation:
# $$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$
#
# With initial condition $u(x, 0) = -\sin(\pi x)$ and Dirichlet BCs $u(\pm 1, t) = 0$.


# %%
def initial_condition(x):
    """Initial condition: u(x, 0) = -sin(pi*x)."""
    return -jnp.sin(jnp.pi * x)


def boundary_condition(t):
    """Boundary condition: u(+-1, t) = 0."""
    return jnp.zeros_like(t)


print()
print("Burgers equation: du/dt + u*du/dx = nu*d^2u/dx^2")
print(f"  Viscosity: nu = 0.01/pi ≈ {float(NU):.6f}")
print("  IC: u(x, 0) = -sin(pi*x)")
print("  BC: u(-1, t) = u(1, t) = 0 (Dirichlet)")

# %% [markdown]
# ## Create XPINN Model with 2 Subdomains
#
# We decompose the domain at x = 0 into left [-1, 0] and right [0, 1] subdomains.

# %%
# Define non-overlapping subdomains in (x, t) space
subdomains = [
    Subdomain(id=0, bounds=jnp.array([[X_MIN, 0.0], [T_MIN, T_MAX]])),  # Left
    Subdomain(id=1, bounds=jnp.array([[0.0, X_MAX], [T_MIN, T_MAX]])),  # Right
]

# Define interface at x = 0
t_interface = jnp.linspace(T_MIN, T_MAX, N_INTERFACE)
interface_points = jnp.column_stack(
    [
        jnp.zeros(N_INTERFACE),  # x = 0
        t_interface,
    ]
)

interfaces = [
    Interface(
        subdomain_ids=(0, 1),
        points=interface_points,
        normal=jnp.array([1.0, 0.0]),  # Normal pointing from left to right
    )
]

# XPINN configuration
xpinn_config = XPINNConfig(
    continuity_weight=10.0,  # Weight for u_left = u_right at interface
    flux_weight=10.0,  # Weight for du/dx_left = du/dx_right at interface
    residual_weight=1.0,  # Weight for PDE residual
)

# Create XPINN model
print()
print("Creating XPINN model...")
model = XPINN(
    input_dim=2,  # (x, t)
    output_dim=1,
    subdomains=subdomains,
    interfaces=interfaces,
    hidden_dims=HIDDEN_DIMS,
    config=xpinn_config,
    rngs=nnx.Rngs(42),
)


# Count parameters
def count_params(m):
    """Count total parameters in model."""
    return sum(p.size for p in jax.tree.leaves(nnx.state(m)))


total_params = count_params(model)
print(f"Total XPINN parameters: {total_params}")
print(f"Parameters per subdomain: ~{total_params // len(subdomains)}")

# %% [markdown]
# ## Generate Collocation Points

# %%
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 10)

# Domain interior points for each subdomain
# Left subdomain [-1, 0] x [0, 1]
x_left = jax.random.uniform(keys[0], (N_DOMAIN,), minval=X_MIN, maxval=0.0)
t_left = jax.random.uniform(keys[1], (N_DOMAIN,), minval=T_MIN, maxval=T_MAX)
xt_left = jnp.column_stack([x_left, t_left])

# Right subdomain [0, 1] x [0, 1]
x_right = jax.random.uniform(keys[2], (N_DOMAIN,), minval=0.0, maxval=X_MAX)
t_right = jax.random.uniform(keys[3], (N_DOMAIN,), minval=T_MIN, maxval=T_MAX)
xt_right = jnp.column_stack([x_right, t_right])

collocation_points = [xt_left, xt_right]

# Boundary conditions at x=-1 and x=1
t_bc = jax.random.uniform(keys[4], (N_BOUNDARY,), minval=T_MIN, maxval=T_MAX)
xt_bc_left = jnp.column_stack([jnp.full(N_BOUNDARY, X_MIN), t_bc])
xt_bc_right = jnp.column_stack([jnp.full(N_BOUNDARY, X_MAX), t_bc])
xt_bc = jnp.vstack([xt_bc_left, xt_bc_right])
u_bc = jnp.zeros(xt_bc.shape[0])  # u = 0 at boundaries

# Initial condition at t = 0
x_ic = jax.random.uniform(keys[5], (N_INITIAL,), minval=X_MIN, maxval=X_MAX)
xt_ic = jnp.column_stack([x_ic, jnp.zeros(N_INITIAL)])
u_ic = initial_condition(x_ic)

print()
print("Generating collocation points...")
print(f"Left subdomain points:  {xt_left.shape}")
print(f"Right subdomain points: {xt_right.shape}")
print(f"Boundary points:        {xt_bc.shape}")
print(f"Initial points:         {xt_ic.shape}")
print(f"Interface points:       {interface_points.shape}")

# %% [markdown]
# ## Define Physics-Informed Loss
#
# Following the FBPINNs hard boundary constraint approach using tanh masking.


# %%
def compute_pde_residual(network, xt, nu=NU):
    """Compute Burgers PDE residual: u_t + u*u_x - nu*u_xx = 0."""

    def u_scalar(xt_single):
        return network(xt_single.reshape(1, 2)).squeeze()

    def residual_single(xt_single):
        u = u_scalar(xt_single)

        # First derivatives
        grad_u = jax.grad(u_scalar)(xt_single)
        u_x = grad_u[0]
        u_t = grad_u[1]

        # Second derivative u_xx
        def u_x_fn(xts):
            return jax.grad(u_scalar)(xts)[0]

        u_xx = jax.grad(u_x_fn)(xt_single)[0]

        # Residual: u_t + u*u_x - nu*u_xx = 0
        return u_t + u * u_x - nu * u_xx

    return jax.vmap(residual_single)(xt)


def subdomain_pde_loss(model, subdomain_id, xt):
    """Compute PDE residual loss for a specific subdomain."""
    network = list(model.networks)[subdomain_id]
    residuals = compute_pde_residual(network, xt)
    return jnp.mean(residuals**2)


def bc_loss(model, xt_bc, u_bc):
    """Boundary condition loss."""
    # Left BC (x=-1) -> subdomain 0
    # Right BC (x=1) -> subdomain 1
    n_per_bc = xt_bc.shape[0] // 2

    networks_list = list(model.networks)
    u_pred_left = networks_list[0](xt_bc[:n_per_bc]).squeeze()
    u_pred_right = networks_list[1](xt_bc[n_per_bc:]).squeeze()

    loss_left = jnp.mean((u_pred_left - u_bc[:n_per_bc]) ** 2)
    loss_right = jnp.mean((u_pred_right - u_bc[n_per_bc:]) ** 2)

    return (loss_left + loss_right) / 2.0


def ic_loss(model, xt_ic, u_ic):
    """Initial condition loss (JAX-compatible, no boolean indexing)."""
    # Use weighted loss instead of masking to be JIT-compatible
    # Weight points by their subdomain membership (soft weighting)
    x_vals = xt_ic[:, 0]

    # Left subdomain: x < 0
    weight_left = jnp.where(x_vals < 0, 1.0, 0.0)
    networks_list = list(model.networks)
    u_pred_left = networks_list[0](xt_ic).squeeze()
    loss_left = jnp.sum(weight_left * (u_pred_left - u_ic) ** 2) / (
        jnp.sum(weight_left) + 1e-8
    )

    # Right subdomain: x >= 0
    weight_right = jnp.where(x_vals >= 0, 1.0, 0.0)
    u_pred_right = networks_list[1](xt_ic).squeeze()
    loss_right = jnp.sum(weight_right * (u_pred_right - u_ic) ** 2) / (
        jnp.sum(weight_right) + 1e-8
    )

    return (loss_left + loss_right) / 2.0


def total_loss(model, colloc_pts, xt_bc, u_bc, xt_ic, u_ic, config):
    """Total XPINN loss: PDE + BC + IC + interface conditions."""
    # PDE residual in each subdomain
    loss_pde = jnp.array(0.0)
    for i, xt in enumerate(colloc_pts):
        loss_pde = loss_pde + subdomain_pde_loss(model, i, xt)
    loss_pde = loss_pde / len(colloc_pts)

    # Boundary and initial conditions
    loss_bc = bc_loss(model, xt_bc, u_bc)
    loss_ic = ic_loss(model, xt_ic, u_ic)

    # Interface conditions (continuity + flux)
    loss_continuity = model.compute_continuity_loss()
    loss_flux = model.compute_flux_loss()

    return (
        config.residual_weight * loss_pde
        + 10.0 * loss_bc
        + 10.0 * loss_ic
        + config.continuity_weight * loss_continuity
        + config.flux_weight * loss_flux
    )


# %% [markdown]
# ## Training

# %%
print()
print("Training XPINN...")

opt = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(model, opt, colloc_pts, xt_bc, u_bc, xt_ic, u_ic):
    """Single training step with gradient update."""

    def loss_fn(m):
        return total_loss(m, colloc_pts, xt_bc, u_bc, xt_ic, u_ic, m.config)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


losses = []
for epoch in range(EPOCHS):
    loss = train_step(model, opt, collocation_points, xt_bc, u_bc, xt_ic, u_ic)
    losses.append(float(loss))

    if (epoch + 1) % 3000 == 0 or epoch == 0:
        cont_loss = float(model.compute_continuity_loss())
        flux_loss = float(model.compute_flux_loss())
        print(
            f"  Epoch {epoch + 1:5d}/{EPOCHS}: loss={loss:.6e}, "
            f"continuity={cont_loss:.6e}, flux={flux_loss:.6e}"
        )

print(f"Final loss: {losses[-1]:.6e}")

# %% [markdown]
# ## Evaluation

# %%
print()
print("Evaluating XPINN...")

# Create evaluation grid
nx, nt = 100, 100
x_eval = jnp.linspace(X_MIN, X_MAX, nx)
t_eval = jnp.linspace(T_MIN, T_MAX, nt)
X, T = jnp.meshgrid(x_eval, t_eval)
xt_eval = jnp.column_stack([X.ravel(), T.ravel()])

# Compute predictions using the full XPINN model
u_pred = model(xt_eval).squeeze().reshape(nt, nx)

# Compute IC error (we know the exact IC)
u_ic_pred = model(jnp.column_stack([x_eval, jnp.zeros(nx)])).squeeze()
u_ic_exact = initial_condition(x_eval)
ic_error = jnp.mean(jnp.abs(u_ic_pred - u_ic_exact))

# Compute interface continuity
networks_list = list(model.networks)
u_left_interface = networks_list[0](interface_points).squeeze()
u_right_interface = networks_list[1](interface_points).squeeze()
interface_jump = jnp.mean(jnp.abs(u_left_interface - u_right_interface))

# BC error
bc_pred_left = model(jnp.column_stack([jnp.full(nt, X_MIN), t_eval])).squeeze()
bc_pred_right = model(jnp.column_stack([jnp.full(nt, X_MAX), t_eval])).squeeze()
bc_error = (jnp.mean(jnp.abs(bc_pred_left)) + jnp.mean(jnp.abs(bc_pred_right))) / 2.0

print(f"IC error (mean abs):     {ic_error:.6e}")
print(f"BC error (mean abs):     {bc_error:.6e}")
print(f"Interface jump:          {interface_jump:.6e}")

# %% [markdown]
# ## Visualization

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Predicted solution
im0 = axes[0, 0].contourf(X, T, u_pred, levels=50, cmap="RdBu_r")
axes[0, 0].axvline(x=0.0, color="white", linestyle="--", linewidth=2, label="Interface")
axes[0, 0].set_title("XPINN Prediction")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("t")
plt.colorbar(im0, ax=axes[0, 0])

# Left subdomain prediction
networks_list_viz = list(model.networks)
u_left_pred = networks_list_viz[0](xt_eval).squeeze().reshape(nt, nx)
im1 = axes[0, 1].contourf(X, T, u_left_pred, levels=50, cmap="RdBu_r")
axes[0, 1].axvline(x=0.0, color="white", linestyle="--", linewidth=2)
axes[0, 1].set_title("Left Subdomain Network")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("t")
plt.colorbar(im1, ax=axes[0, 1])

# Right subdomain prediction
u_right_pred = networks_list_viz[1](xt_eval).squeeze().reshape(nt, nx)
im2 = axes[0, 2].contourf(X, T, u_right_pred, levels=50, cmap="RdBu_r")
axes[0, 2].axvline(x=0.0, color="white", linestyle="--", linewidth=2)
axes[0, 2].set_title("Right Subdomain Network")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("t")
plt.colorbar(im2, ax=axes[0, 2])

# Solution slices at different times
t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
for t_val in t_slices:
    t_idx = int(t_val * (nt - 1))
    axes[1, 0].plot(x_eval, u_pred[t_idx, :], label=f"t={t_val}")
axes[1, 0].plot(x_eval, u_ic_exact, "k--", linewidth=1.5, label="IC (exact)")
axes[1, 0].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("u")
axes[1, 0].set_title("Solution at Different Times")
axes[1, 0].legend(loc="best", fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Interface continuity check
axes[1, 1].plot(t_interface, u_left_interface, "b-", linewidth=2, label="Left at x=0")
axes[1, 1].plot(
    t_interface, u_right_interface, "r--", linewidth=2, label="Right at x=0"
)
axes[1, 1].set_xlabel("t")
axes[1, 1].set_ylabel("u at interface")
axes[1, 1].set_title(f"Interface Continuity (jump={interface_jump:.4e})")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Training history
axes[1, 2].semilogy(losses)
axes[1, 2].set_xlabel("Epoch")
axes[1, 2].set_ylabel("Total Loss")
axes[1, 2].set_title("Training History")
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "docs/assets/examples/xpinn_helmholtz/solution.png", dpi=150, bbox_inches="tight"
)
print()
print("Saved: docs/assets/examples/xpinn_helmholtz/solution.png")
plt.show()

# %%
# Analysis plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# IC comparison
axes[0].plot(x_eval, u_ic_exact, "b-", linewidth=2, label="Exact IC")
axes[0].plot(x_eval, u_ic_pred, "r--", linewidth=2, label="XPINN at t=0")
axes[0].axvline(x=0, color="gray", linestyle=":", alpha=0.5, label="Interface")
axes[0].set_xlabel("x")
axes[0].set_ylabel("u")
axes[0].set_title(f"Initial Condition (error={ic_error:.4e})")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Shock formation visualization (cross-section at x=0)
u_at_interface = u_pred[:, nx // 2]  # At x=0
axes[1].plot(t_eval, u_at_interface, "b-", linewidth=2)
axes[1].set_xlabel("t")
axes[1].set_ylabel("u(0, t)")
axes[1].set_title("Solution at Interface (x=0)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "docs/assets/examples/xpinn_helmholtz/analysis.png", dpi=150, bbox_inches="tight"
)
print("Saved: docs/assets/examples/xpinn_helmholtz/analysis.png")
plt.show()

# %% [markdown]
# ## Results Summary

# %%
print()
print("=" * 70)
print("Results Summary")
print("=" * 70)
print(f"Final Loss:          {losses[-1]:.6e}")
print(f"IC Error (mean abs): {ic_error:.6e}")
print(f"BC Error (mean abs): {bc_error:.6e}")
print(f"Interface Jump:      {interface_jump:.6e}")
print(f"Total Parameters:    {total_params}")
print(f"Training Epochs:     {EPOCHS}")
print(f"Number of Subdomains:{len(subdomains)}")
