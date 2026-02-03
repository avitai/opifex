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
# # Inverse Diffusion Equation PINN
#
# This example demonstrates solving an inverse problem: discovering the unknown
# diffusion coefficient in the heat/diffusion equation from sparse observations.
#
# **Reference**: DeepXDE `examples/pinn_inverse/diffusion_1d_inverse.py`
#
# This is a fundamental inverse problem in PDE parameter identification.

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
print("Opifex Example: Inverse Diffusion Equation PINN")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# True diffusion coefficient (to be discovered)
# DeepXDE reference uses C=1.0 but starts from C=2.0 initial guess
C_TRUE = 1.0

# Domain bounds (matching DeepXDE)
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# Collocation points (matching DeepXDE: 40 domain, 20 boundary, 10 initial)
N_DOMAIN = 400  # Increased from 40 for better coverage
N_BOUNDARY = 100  # 50 per boundary
N_INITIAL = 100
N_OBSERVE = 10  # 10 observation points at t=1 (matching DeepXDE)

# Network configuration (matching DeepXDE: [2] + [32]*3 + [1])
HIDDEN_DIMS = [32, 32, 32]

# Training configuration
EPOCHS = 20000
LEARNING_RATE = 1e-3

print()
print(f"True diffusion coefficient: C = {C_TRUE}")
print(f"Domain: x in [{X_MIN}, {X_MAX}], t in [{T_MIN}, {T_MAX}]")
print(f"Collocation: {N_DOMAIN} domain, {N_BOUNDARY} boundary, {N_INITIAL} initial")
print(f"Observation points: {N_OBSERVE} at t=1 (for parameter discovery)")
print(f"Network: [2] + {HIDDEN_DIMS} + [1]")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Problem Definition
#
# **PDE**: Diffusion equation with source term
#
# $$\frac{\partial u}{\partial t} - C \frac{\partial^2 u}{\partial x^2} = f(x, t)$$
#
# where the source term is chosen so the exact solution is:
#
# $$u(x, t) = \sin(\pi x) \cdot e^{-t}$$
#
# Substituting this into the PDE gives us:
#
# $$f(x, t) = e^{-t}(\sin(\pi x) - C \cdot \pi^2 \sin(\pi x))$$
#
# **Inverse Problem**: Given observations of u(x,t), discover the unknown C.


# %%
# Exact solution and source term
def exact_solution(x, t):
    """Exact solution: u(x, t) = sin(pi*x) * exp(-t)."""
    return jnp.sin(jnp.pi * x) * jnp.exp(-t)


def source_term(x, t):
    """Source term f(x, t) so that u = sin(pi*x)*exp(-t) is the exact solution.

    From the PDE: u_t - C*u_xx = f
    u_t = -sin(pi*x)*exp(-t)
    u_xx = -pi^2*sin(pi*x)*exp(-t)
    f = -sin(pi*x)*exp(-t) - C*(-pi^2*sin(pi*x)*exp(-t))
    f = exp(-t)*(sin(pi*x)*(-1 + C*pi^2))

    For C=1 (the true value we want to discover):
    f = exp(-t)*(sin(pi*x) - pi^2*sin(pi*x))
    """
    return jnp.exp(-t) * (jnp.sin(jnp.pi * x) - jnp.pi**2 * jnp.sin(jnp.pi * x))


print()
print("Diffusion equation: du/dt - C * d^2u/dx^2 = f(x, t)")
print(f"  True coefficient: C = {C_TRUE}")
print("  Exact solution: u(x, t) = sin(pi*x) * exp(-t)")
print("  BC: u(-1, t) = u(1, t) = sin(pi*(-1))*exp(-t) = 0")
print("  IC: u(x, 0) = sin(pi*x)")
print("  Goal: Discover C from sparse observations at t=1")


# %% [markdown]
# ## PINN with Trainable Parameter


# %%
class InverseDiffusionPINN(nnx.Module):
    """PINN for the inverse diffusion equation with trainable coefficient."""

    def __init__(self, hidden_dims: list[int], C_init: float, *, rngs: nnx.Rngs):
        super().__init__()

        # Trainable diffusion coefficient (to be discovered)
        # Use log transform to ensure positivity
        self.log_C = nnx.Param(jnp.log(jnp.array(C_init)))

        layers = []
        in_features = 2  # (x, t)

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        layers.append(nnx.Linear(in_features, 1, rngs=rngs))
        self.layers = nnx.List(layers)

    @property
    def coef(self) -> jax.Array:
        """Return positive diffusion coefficient via exp transform."""
        return jnp.exp(self.log_C.value)

    def __call__(self, xt: jax.Array) -> jax.Array:
        """Forward pass through the network."""
        h = xt
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)


# %%
print()
print("Creating PINN model...")

# Initialize with incorrect guess (DeepXDE uses 2.0)
C_INIT = 2.0
pinn = InverseDiffusionPINN(hidden_dims=HIDDEN_DIMS, C_init=C_INIT, rngs=nnx.Rngs(42))

n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
print(f"PINN parameters: {n_params:,}")
print(f"Initial C guess: {float(pinn.coef):.6f}")
print(f"True C:          {C_TRUE:.6f}")

# %% [markdown]
# ## Collocation Points and Observation Data

# %%
print()
print("Generating collocation points and observations...")

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 6)

# Domain interior points
x_domain = jax.random.uniform(keys[0], (N_DOMAIN,), minval=X_MIN, maxval=X_MAX)
t_domain = jax.random.uniform(keys[1], (N_DOMAIN,), minval=T_MIN, maxval=T_MAX)
xt_domain = jnp.column_stack([x_domain, t_domain])

# Boundary points (x = -1 and x = 1)
t_bc = jax.random.uniform(keys[2], (N_BOUNDARY,), minval=T_MIN, maxval=T_MAX)
xt_bc_left = jnp.column_stack(
    [jnp.full(N_BOUNDARY // 2, X_MIN), t_bc[: N_BOUNDARY // 2]]
)
xt_bc_right = jnp.column_stack(
    [jnp.full(N_BOUNDARY // 2, X_MAX), t_bc[N_BOUNDARY // 2 :]]
)
xt_bc = jnp.vstack([xt_bc_left, xt_bc_right])
# BC values: sin(pi * (+/-1)) * exp(-t) = 0
u_bc = exact_solution(xt_bc[:, 0], xt_bc[:, 1])

# Initial condition points (t = 0)
x_ic = jax.random.uniform(keys[3], (N_INITIAL,), minval=X_MIN, maxval=X_MAX)
xt_ic = jnp.column_stack([x_ic, jnp.zeros(N_INITIAL)])
u_ic = exact_solution(x_ic, jnp.zeros(N_INITIAL))

# Observation points at t=1 (matching DeepXDE: linspace from -1 to 1, 10 points)
x_obs = jnp.linspace(X_MIN, X_MAX, N_OBSERVE)
t_obs = jnp.ones(N_OBSERVE)  # All at t=1
xt_obs = jnp.column_stack([x_obs, t_obs])
u_obs = exact_solution(x_obs, t_obs)

print(f"Domain points:      {xt_domain.shape}")
print(f"Boundary points:    {xt_bc.shape}")
print(f"Initial points:     {xt_ic.shape}")
print(f"Observation points: {xt_obs.shape} (at t=1)")

# %% [markdown]
# ## Physics-Informed Loss with Parameter Discovery


# %%
def compute_pde_residual(pinn, xt):
    """Compute diffusion PDE residual: u_t - C*u_xx + f = 0.

    Note: DeepXDE formulation uses u_t - C*u_xx + f = 0 (plus sign for f).
    """

    def u_scalar(xt_single):
        return pinn(xt_single.reshape(1, 2)).squeeze()

    def residual_single(xt_single):
        x, t = xt_single[0], xt_single[1]

        # Get u and its derivatives
        grad_u = jax.grad(u_scalar)(xt_single)
        u_t = grad_u[1]

        # Second derivative
        hess = jax.hessian(u_scalar)(xt_single)
        u_xx = hess[0, 0]

        # Source term (fixed for true C=1)
        f = source_term(x, t)

        # Diffusion: u_t - C*u_xx + f = 0 (matching DeepXDE)
        return u_t - pinn.coef * u_xx + f

    return jax.vmap(residual_single)(xt)


def pde_loss(pinn, xt):
    """Compute mean squared PDE residual."""
    residual = compute_pde_residual(pinn, xt)
    return jnp.mean(residual**2)


def bc_loss(pinn, xt, u_target):
    """Compute mean squared boundary condition error."""
    u = pinn(xt).squeeze()
    return jnp.mean((u - u_target) ** 2)


def ic_loss(pinn, xt, u_target):
    """Compute mean squared initial condition error."""
    u = pinn(xt).squeeze()
    return jnp.mean((u - u_target) ** 2)


def data_loss(pinn, xt, u_target):
    """Loss from observation data - key for parameter discovery."""
    u = pinn(xt).squeeze()
    return jnp.mean((u - u_target) ** 2)


def total_loss(pinn, xt_dom, xt_bc, u_bc, xt_ic, u_ic, xt_obs, u_obs):
    """Total loss = PDE + BC + IC + Data fitting.

    All losses weighted equally (following DeepXDE default).
    """
    loss_pde = pde_loss(pinn, xt_dom)
    loss_bc = bc_loss(pinn, xt_bc, u_bc)
    loss_ic = ic_loss(pinn, xt_ic, u_ic)
    loss_data = data_loss(pinn, xt_obs, u_obs)

    return loss_pde + loss_bc + loss_ic + loss_data


# %% [markdown]
# ## Training

# %%
print()
print("Training PINN (discovering diffusion coefficient)...")

opt = nnx.Optimizer(pinn, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(pinn, opt, xt_dom, xt_bc, u_bc, xt_ic, u_ic, xt_obs, u_obs):
    """Perform one training step with gradient update."""

    def loss_fn(model):
        return total_loss(model, xt_dom, xt_bc, u_bc, xt_ic, u_ic, xt_obs, u_obs)

    loss, grads = nnx.value_and_grad(loss_fn)(pinn)
    opt.update(pinn, grads)
    return loss


losses = []
C_history = []

for epoch in range(EPOCHS):
    loss = train_step(pinn, opt, xt_domain, xt_bc, u_bc, xt_ic, u_ic, xt_obs, u_obs)
    losses.append(float(loss))
    C_history.append(float(pinn.coef))

    if (epoch + 1) % 4000 == 0 or epoch == 0:
        print(
            f"  Epoch {epoch + 1:5d}/{EPOCHS}: loss={loss:.6e}, C={float(pinn.coef):.6f}"
        )

print(f"Final loss: {losses[-1]:.6e}")
print()
print(f"Discovered C: {float(pinn.coef):.6f}")
print(f"True C:       {C_TRUE:.6f}")
print(f"Relative error: {abs(float(pinn.coef) - C_TRUE) / C_TRUE * 100:.2f}%")

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

print(f"Relative L2 error:   {l2_error:.6e}")
print(f"Maximum point error: {max_error:.6e}")
print(f"Mean point error:    {mean_error:.6e}")
print(f"Mean PDE residual:   {mean_residual:.6e}")

# %% [markdown]
# ## Visualization

# %%
output_dir = Path("docs/assets/examples/inverse_diffusion_pinn")
output_dir.mkdir(parents=True, exist_ok=True)

mpl.use("Agg")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Solution comparison
im0 = axes[0, 0].imshow(
    np.array(u_pred_grid),
    extent=[X_MIN, X_MAX, T_MIN, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
axes[0, 0].scatter(
    np.array(x_obs), np.array(t_obs), c="r", s=50, marker="x", label="Observations"
)
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("t")
axes[0, 0].set_title("PINN Solution (with observations)")
axes[0, 0].legend()
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(
    np.array(u_exact_grid),
    extent=[X_MIN, X_MAX, T_MIN, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("t")
axes[0, 1].set_title("Exact Solution")
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(
    np.array(error),
    extent=[X_MIN, X_MAX, T_MIN, T_MAX],
    origin="lower",
    aspect="auto",
    cmap="hot",
)
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("t")
axes[0, 2].set_title(f"Error (L2={l2_error:.2e})")
plt.colorbar(im2, ax=axes[0, 2])

# Row 2: Parameter discovery and training
axes[1, 0].semilogy(losses, linewidth=1)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].set_title("Training Loss")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(C_history, "b-", linewidth=2, label="Discovered C")
axes[1, 1].axhline(
    y=C_TRUE, color="r", linestyle="--", linewidth=2, label=f"True C = {C_TRUE}"
)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Diffusion Coefficient (C)")
axes[1, 1].set_title("Parameter Discovery")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Time slices comparison
t_indices = [0, nt // 2, nt - 1]
colors = ["b", "g", "r"]
for t_idx, color in zip(t_indices, colors, strict=True):
    t_val = float(t_eval[t_idx])
    axes[1, 2].plot(
        np.array(x_eval),
        np.array(u_pred_grid[t_idx, :]),
        f"{color}-",
        label=f"PINN t={t_val:.2f}",
        linewidth=2,
    )
    axes[1, 2].plot(
        np.array(x_eval),
        np.array(u_exact_grid[t_idx, :]),
        f"{color}--",
        label=f"Exact t={t_val:.2f}",
        linewidth=1,
        alpha=0.7,
    )
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("u(x, t)")
axes[1, 2].set_title("Solution Comparison")
axes[1, 2].legend(fontsize=7, ncol=2)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "solution.png", dpi=150, bbox_inches="tight")
plt.close()
print()
print(f"Solution saved to {output_dir / 'solution.png'}")

# %%
# Parameter convergence analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Convergence plot
axes[0].plot(C_history, "b-", linewidth=2, label="Discovered C")
axes[0].axhline(
    y=C_TRUE, color="r", linestyle="--", linewidth=2, label=f"True C = {C_TRUE}"
)
axes[0].fill_between(
    range(len(C_history)),
    C_TRUE * 0.95,
    C_TRUE * 1.05,
    alpha=0.2,
    color="r",
    label="5% error band",
)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Diffusion Coefficient (C)")
axes[0].set_title("Parameter Convergence")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Relative error over time
C_errors = [abs(C - C_TRUE) / C_TRUE * 100 for C in C_history]
axes[1].semilogy(C_errors, "b-", linewidth=2)
axes[1].axhline(y=5, color="r", linestyle="--", linewidth=1, label="5% error")
axes[1].axhline(y=1, color="g", linestyle="--", linewidth=1, label="1% error")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Relative Error (%)")
axes[1].set_title("Parameter Discovery Error")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis saved to {output_dir / 'analysis.png'}")

# %%
print()
print("=" * 70)
print("Inverse Diffusion Equation PINN example completed")
print("=" * 70)
print()
print("Results Summary:")
print(f"  Final loss:        {losses[-1]:.6e}")
print(f"  Discovered C:      {float(pinn.coef):.6f}")
print(f"  True C:            {C_TRUE:.6f}")
print(f"  Parameter error:   {abs(float(pinn.coef) - C_TRUE) / C_TRUE * 100:.2f}%")
print(f"  Relative L2 error: {l2_error:.6e}")
print(f"  Parameters:        {n_params:,}")
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
