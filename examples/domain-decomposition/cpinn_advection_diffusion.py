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
# # CPINN: Conservative PINN on Advection-Diffusion Equation
#
# This example demonstrates solving the 1D advection-diffusion equation using CPINN
# (Conservative Physics-Informed Neural Network). CPINNs enforce strong flux
# conservation at subdomain interfaces, critical for conservation laws.

# %% [markdown]
# ## Setup and Imports

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

from opifex.neural.pinns.domain_decomposition import (
    CPINN,
    CPINNConfig,
    Interface,
    Subdomain,
)


print("=" * 70)
print("Opifex Example: CPINN on 1D Advection-Diffusion Equation")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## Configuration

# %%
# Problem configuration
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 0.5
C = 1.0  # Advection velocity
D = 0.01  # Diffusion coefficient

# Training configuration
N_DOMAIN = 500  # Points per subdomain
N_BOUNDARY = 100  # Boundary points
N_INITIAL = 200  # Initial condition points
N_INTERFACE = 30  # Points on each interface
EPOCHS = 15000
LEARNING_RATE = 0.001

# CPINN configuration (3 subdomains in x-direction)
NUM_SUBDOMAINS = 3
HIDDEN_DIMS = [32, 32]

print()
print(f"Domain: x in [{X_MIN}, {X_MAX}], t in [{T_MIN}, {T_MAX}]")
print(f"Advection velocity: c = {C}")
print(f"Diffusion coefficient: D = {D}")
print(f"Subdomains: {NUM_SUBDOMAINS}")
print(
    f"Collocation: {NUM_SUBDOMAINS * N_DOMAIN} domain, {N_BOUNDARY} boundary, {N_INITIAL} initial"
)
print(f"Network per subdomain: [2] + {HIDDEN_DIMS} + [1]")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Define the Advection-Diffusion Problem
#
# We solve the 1D advection-diffusion equation:
# $$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = D \frac{\partial^2 u}{\partial x^2}$$
#
# With initial condition $u(x, 0) = \sin(\pi x)$ and homogeneous Dirichlet BCs.


# %%
def exact_solution(x, t, c=C, d=D):
    """Exact solution for advection-diffusion with sin(pi*x) IC.

    u(x, t) = exp(-D*pi^2*t) * sin(pi*(x - c*t))
    """
    return jnp.exp(-d * jnp.pi**2 * t) * jnp.sin(jnp.pi * (x - c * t))


def initial_condition(x):
    """Initial condition: u(x, 0) = sin(pi*x)."""
    return jnp.sin(jnp.pi * x)


print()
print("Advection-diffusion: du/dt + c*du/dx = D*d^2u/dx^2")
print(f"  Advection: c = {C}")
print(f"  Diffusion: D = {D}")
print("  IC: u(x, 0) = sin(pi*x)")
print("  BC: u(0, t) = u(1, t) = 0 (Dirichlet)")

# %% [markdown]
# ## Create CPINN Model with 3 Subdomains
#
# We decompose the domain into 3 subdomains along x:
# [0, 1/3], [1/3, 2/3], [2/3, 1]

# %%
# Define non-overlapping subdomains in (x, t) space
x_boundaries = jnp.linspace(X_MIN, X_MAX, NUM_SUBDOMAINS + 1)

subdomains = []
for i in range(NUM_SUBDOMAINS):
    bounds = jnp.array(
        [
            [x_boundaries[i], x_boundaries[i + 1]],  # x bounds
            [T_MIN, T_MAX],  # t bounds (full time range)
        ]
    )
    subdomains.append(Subdomain(id=i, bounds=bounds))

# Define interfaces at x = 1/3 and x = 2/3
interfaces = []
for i in range(NUM_SUBDOMAINS - 1):
    x_interface = x_boundaries[i + 1]
    t_interface = jnp.linspace(T_MIN, T_MAX, N_INTERFACE)

    interface_points = jnp.column_stack(
        [
            jnp.full(N_INTERFACE, x_interface),
            t_interface,
        ]
    )

    interfaces.append(
        Interface(
            subdomain_ids=(i, i + 1),
            points=interface_points,
            normal=jnp.array([1.0, 0.0]),  # Normal in x-direction
        )
    )

# CPINN configuration with emphasis on flux conservation
cpinn_config = CPINNConfig(
    continuity_weight=10.0,  # Weight for solution continuity
    flux_weight=10.0,  # Weight for flux conservation
    conservation_weight=0.1,  # Additional conservation enforcement
)

# Create CPINN model
print()
print("Creating CPINN model...")
model = CPINN(
    input_dim=2,  # (x, t)
    output_dim=1,
    subdomains=subdomains,
    interfaces=interfaces,
    hidden_dims=HIDDEN_DIMS,
    config=cpinn_config,
    rngs=nnx.Rngs(42),
)


# Count parameters
def count_params(m):
    """Count total parameters in model."""
    return sum(p.size for p in jax.tree.leaves(nnx.state(m)))


total_params = count_params(model)
print(f"Total CPINN parameters: {total_params}")
print(f"Parameters per subdomain: ~{total_params // len(subdomains)}")
print(f"Number of interfaces: {len(interfaces)}")

# %% [markdown]
# ## Generate Collocation Points

# %%
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 10)

# Domain interior points for each subdomain
collocation_points_per_subdomain = []
for i, subdomain in enumerate(subdomains):
    x_lo, x_hi = subdomain.bounds[0]
    x_pts = jax.random.uniform(keys[i], (N_DOMAIN,), minval=x_lo, maxval=x_hi)
    t_pts = jax.random.uniform(
        keys[i + NUM_SUBDOMAINS], (N_DOMAIN,), minval=T_MIN, maxval=T_MAX
    )
    xt_pts = jnp.column_stack([x_pts, t_pts])
    collocation_points_per_subdomain.append(xt_pts)

# Combine all domain points
xt_domain = jnp.vstack(collocation_points_per_subdomain)

# Boundary conditions at x=0 and x=1
t_bc = jax.random.uniform(keys[7], (N_BOUNDARY,), minval=T_MIN, maxval=T_MAX)
xt_bc_left = jnp.column_stack([jnp.zeros(N_BOUNDARY), t_bc])
xt_bc_right = jnp.column_stack([jnp.ones(N_BOUNDARY), t_bc])
xt_bc = jnp.vstack([xt_bc_left, xt_bc_right])
u_bc = jnp.zeros(xt_bc.shape[0])  # Homogeneous Dirichlet

# Initial condition at t=0
x_ic = jax.random.uniform(keys[8], (N_INITIAL,), minval=X_MIN, maxval=X_MAX)
xt_ic = jnp.column_stack([x_ic, jnp.zeros(N_INITIAL)])
u_ic = initial_condition(x_ic)

print()
print("Generating collocation points...")
for i, pts in enumerate(collocation_points_per_subdomain):
    print(f"Subdomain {i} points: {pts.shape}")
print(f"Boundary points:      {xt_bc.shape}")
print(f"Initial points:       {xt_ic.shape}")
print(f"Interface points:     {interfaces[0].points.shape} each")

# %% [markdown]
# ## Define Physics-Informed Loss


# %%
def compute_pde_residual(network, xt, c=C, d=D):
    """Compute advection-diffusion PDE residual: u_t + c*u_x - D*u_xx = 0."""

    def u_scalar(xt_single):
        return network(xt_single.reshape(1, 2)).squeeze()

    def residual_single(xt_single):
        # First derivatives
        grad_u = jax.grad(u_scalar)(xt_single)
        u_x = grad_u[0]
        u_t = grad_u[1]

        # Second derivative u_xx
        def u_x_fn(xt_s):
            return jax.grad(u_scalar)(xt_s)[0]

        u_xx = jax.grad(u_x_fn)(xt_single)[0]

        # Residual: u_t + c*u_x - D*u_xx = 0
        return u_t + c * u_x - d * u_xx

    return jax.vmap(residual_single)(xt)


def subdomain_residual_loss(model, subdomain_id, xt):
    """Compute PDE residual loss for a specific subdomain."""
    network = list(model.networks)[subdomain_id]
    residuals = compute_pde_residual(network, xt)
    return jnp.mean(residuals**2)


def bc_loss(model, xt_bc, u_bc):
    """Boundary condition loss (applied to boundary subdomains)."""
    # For x=0, use subdomain 0
    # For x=1, use subdomain NUM_SUBDOMAINS-1
    n_per_bc = xt_bc.shape[0] // 2

    # Left BC (x=0) -> subdomain 0
    networks_list = list(model.networks)
    u_pred_left = networks_list[0](xt_bc[:n_per_bc]).squeeze()
    loss_left = jnp.mean((u_pred_left - u_bc[:n_per_bc]) ** 2)

    # Right BC (x=1) -> last subdomain
    u_pred_right = networks_list[-1](xt_bc[n_per_bc:]).squeeze()
    loss_right = jnp.mean((u_pred_right - u_bc[n_per_bc:]) ** 2)

    return (loss_left + loss_right) / 2.0


def ic_loss(model, xt_ic, u_ic):
    """Initial condition loss (JAX-compatible, no boolean indexing)."""
    # Use weighted loss instead of masking to be JIT-compatible
    total_loss = jnp.array(0.0)
    x_vals = xt_ic[:, 0]

    networks_list = list(model.networks)
    for i, subdomain in enumerate(model.subdomains):
        x_lo, x_hi = subdomain.bounds[0]
        # Create weight mask (1 if point in subdomain, 0 otherwise)
        weight = jnp.where((x_vals >= x_lo) & (x_vals <= x_hi), 1.0, 0.0)
        weight_sum = jnp.sum(weight)

        # Evaluate network on all points, weight by subdomain membership
        u_pred = networks_list[i](xt_ic).squeeze()
        loss = jnp.sum(weight * (u_pred - u_ic) ** 2) / (weight_sum + 1e-8)
        total_loss = total_loss + loss

    return total_loss / len(model.subdomains)


def total_loss(model, collocation_points, xt_bc, u_bc, xt_ic, u_ic, config):
    """Total CPINN loss: PDE residuals + BC + IC + interface conditions."""
    # PDE residual in each subdomain
    loss_pde = jnp.array(0.0)
    for i, xt in enumerate(collocation_points):
        loss_pde = loss_pde + subdomain_residual_loss(model, i, xt)
    loss_pde = loss_pde / len(collocation_points)

    # Boundary and initial conditions
    loss_bc = bc_loss(model, xt_bc, u_bc)
    loss_ic = ic_loss(model, xt_ic, u_ic)

    # Interface conditions (continuity + flux conservation)
    loss_continuity = model.compute_continuity_loss()
    loss_flux = model.compute_flux_conservation_loss()

    return (
        loss_pde
        + 10.0 * loss_bc
        + 10.0 * loss_ic
        + config.continuity_weight * loss_continuity
        + config.flux_weight * loss_flux
    )


# %% [markdown]
# ## Training

# %%
print()
print("Training CPINN...")

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
    loss = train_step(
        model, opt, collocation_points_per_subdomain, xt_bc, u_bc, xt_ic, u_ic
    )
    losses.append(float(loss))

    if (epoch + 1) % 3000 == 0 or epoch == 0:
        cont_loss = float(model.compute_continuity_loss())
        flux_loss = float(model.compute_flux_conservation_loss())
        print(
            f"  Epoch {epoch + 1:5d}/{EPOCHS}: loss={loss:.6e}, "
            f"continuity={cont_loss:.6e}, flux={flux_loss:.6e}"
        )

print(f"Final loss: {losses[-1]:.6e}")

# %% [markdown]
# ## Evaluation

# %%
print()
print("Evaluating CPINN...")

# Create evaluation grid
nx, nt = 100, 50
x_eval = jnp.linspace(X_MIN, X_MAX, nx)
t_eval = jnp.linspace(T_MIN, T_MAX, nt)
X, T = jnp.meshgrid(x_eval, t_eval)
xt_eval = jnp.column_stack([X.ravel(), T.ravel()])

# Compute predictions using the full CPINN model
u_pred = model(xt_eval).squeeze().reshape(nt, nx)
u_exact = exact_solution(X, T)

# Compute errors
error = jnp.abs(u_pred - u_exact)
l2_error = jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2))
rel_l2_error = l2_error / jnp.sqrt(jnp.mean(u_exact**2))
max_error = jnp.max(error)
mean_error = jnp.mean(error)

# Compute interface flux conservation
interface_flux_errors = []
for interface in interfaces:
    left_id, right_id = interface.subdomain_ids
    points = interface.points
    normal = interface.normal

    # Compute gradients (flux) at interface
    def get_grad(network, pt):
        """Compute gradient of network output at point."""

        def u_fn(p):
            return network(p.reshape(1, 2)).squeeze()

        return jax.grad(u_fn)(pt)

    networks_list = list(model.networks)
    left_net = networks_list[left_id]
    right_net = networks_list[right_id]
    grad_left = jax.vmap(lambda p, net=left_net: get_grad(net, p))(points)
    grad_right = jax.vmap(lambda p, net=right_net: get_grad(net, p))(points)

    # Normal flux
    flux_left = jnp.sum(grad_left * normal, axis=-1)
    flux_right = jnp.sum(grad_right * normal, axis=-1)
    flux_jump = jnp.mean(jnp.abs(flux_left - flux_right))
    interface_flux_errors.append(float(flux_jump))

print(f"Relative L2 error:   {rel_l2_error:.6e}")
print(f"Maximum point error: {max_error:.6e}")
print(f"Mean point error:    {mean_error:.6e}")
for i, flux_err in enumerate(interface_flux_errors):
    print(f"Interface {i} flux jump: {flux_err:.6e}")

# %% [markdown]
# ## Visualization

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# Predicted solution
im0 = axes[0, 0].contourf(X, T, u_pred, levels=50, cmap="viridis")
# Draw interface lines
for interface in interfaces:
    x_int = interface.points[0, 0]
    axes[0, 0].axvline(x=x_int, color="white", linestyle="--", linewidth=1.5)
axes[0, 0].set_title("CPINN Prediction")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("t")
plt.colorbar(im0, ax=axes[0, 0])

# Exact solution
im1 = axes[0, 1].contourf(X, T, u_exact, levels=50, cmap="viridis")
axes[0, 1].set_title("Exact Solution")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("t")
plt.colorbar(im1, ax=axes[0, 1])

# Point-wise error
im2 = axes[0, 2].contourf(X, T, error, levels=50, cmap="hot")
for interface in interfaces:
    x_int = interface.points[0, 0]
    axes[0, 2].axvline(x=x_int, color="white", linestyle="--", linewidth=1.5)
axes[0, 2].set_title(f"Absolute Error (max={max_error:.4e})")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("t")
plt.colorbar(im2, ax=axes[0, 2])

# Solution at different times
t_slices = [0.0, 0.25, 0.5]
for t_val in t_slices:
    t_idx = int(t_val / T_MAX * (nt - 1))
    axes[1, 0].plot(x_eval, u_pred[t_idx, :], label=f"CPINN t={t_val}")
    axes[1, 0].plot(x_eval, u_exact[t_idx, :], "--", label=f"Exact t={t_val}")
# Draw interface lines
for interface in interfaces:
    x_int = interface.points[0, 0]
    axes[1, 0].axvline(x=x_int, color="gray", linestyle=":", alpha=0.7)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("u")
axes[1, 0].set_title("Solution at Different Times")
axes[1, 0].legend(fontsize=8, ncol=2)
axes[1, 0].grid(True, alpha=0.3)

# IC comparison
axes[1, 1].plot(x_eval, u_pred[0, :], "b-", linewidth=2, label="CPINN t=0")
axes[1, 1].plot(x_eval, u_exact[0, :], "r--", linewidth=2, label="Exact IC")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("u")
axes[1, 1].set_title("Initial Condition")
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
    "docs/assets/examples/cpinn_advection_diffusion/solution.png",
    dpi=150,
    bbox_inches="tight",
)
print()
print("Saved: docs/assets/examples/cpinn_advection_diffusion/solution.png")
plt.show()

# %%
# Analysis: Interface flux conservation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Subdomain predictions comparison
colors = ["blue", "green", "red"]
networks_list_viz = list(model.networks)
for i in range(NUM_SUBDOMAINS):
    # Evaluate subdomain network across full domain for comparison
    u_sub = networks_list_viz[i](xt_eval).squeeze().reshape(nt, nx)
    # Show final time slice
    axes[0].plot(
        x_eval,
        u_sub[-1, :],
        color=colors[i],
        linestyle="-" if i == 0 else "--" if i == 1 else ":",
        linewidth=2,
        label=f"Subdomain {i}",
    )

axes[0].plot(x_eval, u_exact[-1, :], "k-", linewidth=1.5, label="Exact")
for interface in interfaces:
    x_int = interface.points[0, 0]
    axes[0].axvline(x=x_int, color="gray", linestyle=":", alpha=0.7)
axes[0].set_xlabel("x")
axes[0].set_ylabel("u")
axes[0].set_title(f"Subdomain Networks at t={T_MAX}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Interface profiles
networks_list_int = list(model.networks)
for i, interface in enumerate(interfaces):
    points = interface.points
    t_vals = points[:, 1]

    u_left = networks_list_int[i](points).squeeze()
    u_right = networks_list_int[i + 1](points).squeeze()

    axes[1].plot(t_vals, u_left, "-", label=f"Left of interface {i}")
    axes[1].plot(t_vals, u_right, "--", label=f"Right of interface {i}")

axes[1].set_xlabel("t")
axes[1].set_ylabel("u at interface")
axes[1].set_title("Interface Continuity Check")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "docs/assets/examples/cpinn_advection_diffusion/analysis.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved: docs/assets/examples/cpinn_advection_diffusion/analysis.png")
plt.show()

# %% [markdown]
# ## Results Summary

# %%
print()
print("=" * 70)
print("Results Summary")
print("=" * 70)
print(f"Final Loss:          {losses[-1]:.6e}")
print(f"Relative L2 Error:   {rel_l2_error:.6e}")
print(f"Maximum Point Error: {max_error:.6e}")
print(f"Mean Point Error:    {mean_error:.6e}")
for i, flux_err in enumerate(interface_flux_errors):
    print(f"Interface {i} Flux Jump: {flux_err:.6e}")
print(f"Total Parameters:    {total_params}")
print(f"Training Epochs:     {EPOCHS}")
print(f"Number of Subdomains:{len(subdomains)}")
