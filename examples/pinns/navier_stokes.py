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
# # Navier-Stokes PINN: Kovasznay Flow
#
# This example demonstrates solving the 2D steady Navier-Stokes equations
# using a Physics-Informed Neural Network (PINN). The Kovasznay flow is an
# exact solution that serves as an important benchmark for incompressible
# flow solvers.
#
# Reference: DeepXDE's Kovasznay_flow.py example

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
print("Opifex Example: Navier-Stokes PINN (Kovasznay Flow)")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Problem configuration (from DeepXDE)
RE = 20  # Reynolds number
NU = 1.0 / RE  # Kinematic viscosity

# Domain bounds (from DeepXDE: Rectangle([-0.5, -0.5], [1, 1.5]))
X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 1.5

# Kovasznay flow parameter (analytical solution coefficient)
LAMBDA = 1.0 / (2.0 * NU) - jnp.sqrt(1.0 / (4.0 * NU**2) + 4.0 * jnp.pi**2)

# Collocation points (from DeepXDE: num_domain=2601, num_boundary=400)
N_DOMAIN = 2601  # Interior collocation points
N_BOUNDARY = 400  # Boundary points

# Network configuration (from DeepXDE: [2] + [50]*4 + [3])
HIDDEN_DIMS = [50, 50, 50, 50]

# Training configuration (from DeepXDE: Adam 30000 iter @ lr=1e-3)
EPOCHS = 30000
LEARNING_RATE = 1e-3

print(f"Reynolds number: Re = {RE}")
print(f"Domain: x in [{X_MIN}, {X_MAX}], y in [{Y_MIN}, {Y_MAX}]")
print(f"Collocation: {N_DOMAIN} domain, {N_BOUNDARY} boundary")
print(f"Network: [2] + {HIDDEN_DIMS} + [3]")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Kovasznay Flow
#
# The Kovasznay flow is an exact solution to the steady Navier-Stokes equations:
#
# **Momentum equations:**
# $$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}
#   = -\frac{\partial p}{\partial x} + \frac{1}{Re}
#     \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$
#
# $$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}
#   = -\frac{\partial p}{\partial y} + \frac{1}{Re}
#     \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$
#
# **Continuity equation:**
# $$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$
#
# **Analytical solution:**
# $$u(x, y) = 1 - e^{\lambda x} \cos(2\pi y)$$
# $$v(x, y) = \frac{\lambda}{2\pi} e^{\lambda x} \sin(2\pi y)$$
# $$p(x, y) = \frac{1}{2}(1 - e^{2\lambda x})$$


# %%
# Analytical solution functions
def u_exact(xy):
    """Exact x-velocity: u = 1 - exp(lambda*x) * cos(2*pi*y)."""
    x, y = xy[:, 0], xy[:, 1]
    return 1.0 - jnp.exp(LAMBDA * x) * jnp.cos(2.0 * jnp.pi * y)


def v_exact(xy):
    """Exact y-velocity: v = (lambda / 2*pi) * exp(lambda*x) * sin(2*pi*y)."""
    x, y = xy[:, 0], xy[:, 1]
    return LAMBDA / (2.0 * jnp.pi) * jnp.exp(LAMBDA * x) * jnp.sin(2.0 * jnp.pi * y)


def p_exact(xy):
    """Exact pressure: p = 0.5 * (1 - exp(2*lambda*x))."""
    x = xy[:, 0]
    return 0.5 * (1.0 - jnp.exp(2.0 * LAMBDA * x))


print()
print("Kovasznay Flow: Steady 2D incompressible Navier-Stokes")
print(f"  Lambda = {float(LAMBDA):.6f}")
print("  u(x,y) = 1 - exp(lambda*x) * cos(2*pi*y)")
print("  v(x,y) = (lambda/2*pi) * exp(lambda*x) * sin(2*pi*y)")
print("  p(x,y) = 0.5 * (1 - exp(2*lambda*x))")

# %% [markdown]
# ## PINN Architecture
#
# Network from DeepXDE: [2] + [50]*4 + [3] with tanh activation.
# Output: [u, v, p] (velocity components and pressure).


# %%
class NavierStokesPINN(nnx.Module):
    """PINN for the Navier-Stokes equations (Kovasznay flow).

    Architecture matches DeepXDE: [2, 50, 50, 50, 50, 3] with tanh activation.
    Output: [u, v, p] - velocity components and pressure.
    """

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs):
        """Initialize PINN.

        Args:
            hidden_dims: List of hidden layer dimensions
            rngs: Random number generators
        """
        super().__init__()

        layers = []
        in_features = 2  # (x, y)

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        layers.append(nnx.Linear(in_features, 3, rngs=rngs))  # Output: [u, v, p]
        self.layers = nnx.List(layers)

    def __call__(self, xy: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            xy: Coordinates [batch, 2] where columns are (x, y)

        Returns:
            Solution values [batch, 3] where columns are (u, v, p)
        """
        h = xy
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)


# %%
print()
print("Creating PINN model...")

pinn = NavierStokesPINN(hidden_dims=HIDDEN_DIMS, rngs=nnx.Rngs(42))

# Count parameters
n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
print(f"PINN parameters: {n_params:,}")

# %% [markdown]
# ## Collocation Points
#
# Sample points following DeepXDE's distribution:
# - Domain: 2601 random points in the rectangle
# - Boundary: 400 points on all 4 edges

# %%
print()
print("Generating collocation points...")

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 6)

# Domain interior points (matching DeepXDE's num_domain=2601)
x_domain = jax.random.uniform(keys[0], (N_DOMAIN,), minval=X_MIN, maxval=X_MAX)
y_domain = jax.random.uniform(keys[1], (N_DOMAIN,), minval=Y_MIN, maxval=Y_MAX)
xy_domain = jnp.column_stack([x_domain, y_domain])

# Boundary points (matching DeepXDE's num_boundary=400)
n_per_edge = N_BOUNDARY // 4

# Bottom edge (y = Y_MIN)
x_bottom = jax.random.uniform(keys[2], (n_per_edge,), minval=X_MIN, maxval=X_MAX)
xy_bottom = jnp.column_stack([x_bottom, jnp.full(n_per_edge, Y_MIN)])

# Top edge (y = Y_MAX)
x_top = jax.random.uniform(keys[3], (n_per_edge,), minval=X_MIN, maxval=X_MAX)
xy_top = jnp.column_stack([x_top, jnp.full(n_per_edge, Y_MAX)])

# Left edge (x = X_MIN)
y_left = jax.random.uniform(keys[4], (n_per_edge,), minval=Y_MIN, maxval=Y_MAX)
xy_left = jnp.column_stack([jnp.full(n_per_edge, X_MIN), y_left])

# Right edge (x = X_MAX)
y_right = jax.random.uniform(keys[5], (n_per_edge,), minval=Y_MIN, maxval=Y_MAX)
xy_right = jnp.column_stack([jnp.full(n_per_edge, X_MAX), y_right])

xy_boundary = jnp.concatenate([xy_bottom, xy_top, xy_left, xy_right], axis=0)

# Exact boundary values
u_bc = u_exact(xy_boundary)
v_bc = v_exact(xy_boundary)
p_bc = p_exact(xy_boundary)

print(f"Domain points:   {xy_domain.shape}")
print(f"Boundary points: {xy_boundary.shape}")

# %% [markdown]
# ## Physics-Informed Loss
#
# The loss function combines:
# 1. **Momentum-x**: $u \cdot u_x + v \cdot u_y + p_x - (1/Re)(u_{xx} + u_{yy}) = 0$
# 2. **Momentum-y**: $u \cdot v_x + v \cdot v_y + p_y - (1/Re)(v_{xx} + v_{yy}) = 0$
# 3. **Continuity**: $u_x + v_y = 0$
# 4. **Boundary conditions**: Dirichlet for u, v, p


# %%
def compute_pde_residuals(pinn, xy):
    """Compute Navier-Stokes PDE residuals.

    Returns:
        Tuple of (momentum_x, momentum_y, continuity) residuals, each [batch]
    """

    def uvp_scalar(xy_single):
        """Scalar output for single point."""
        return pinn(xy_single.reshape(1, 2)).squeeze()

    def residuals_single(xy_single):
        """Compute all residuals for single point."""
        # Get u, v (p not needed directly, we use p_x, p_y)
        uvp = uvp_scalar(xy_single)
        u, v = uvp[0], uvp[1]

        # First derivatives via Jacobian
        jac = jax.jacobian(uvp_scalar)(xy_single)
        u_x, u_y = jac[0, 0], jac[0, 1]
        v_x, v_y = jac[1, 0], jac[1, 1]
        p_x, p_y = jac[2, 0], jac[2, 1]

        # Second derivatives (Hessian diagonals)
        def u_fn(xy_s):
            return uvp_scalar(xy_s)[0]

        def v_fn(xy_s):
            return uvp_scalar(xy_s)[1]

        hess_u = jax.hessian(u_fn)(xy_single)
        hess_v = jax.hessian(v_fn)(xy_single)

        u_xx, u_yy = hess_u[0, 0], hess_u[1, 1]
        v_xx, v_yy = hess_v[0, 0], hess_v[1, 1]

        # Momentum equations
        momentum_x = u * u_x + v * u_y + p_x - (1.0 / RE) * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - (1.0 / RE) * (v_xx + v_yy)

        # Continuity equation
        continuity = u_x + v_y

        return jnp.array([momentum_x, momentum_y, continuity])

    residuals = jax.vmap(residuals_single)(xy)
    return residuals[:, 0], residuals[:, 1], residuals[:, 2]


def pde_loss(pinn, xy):
    """Compute PDE residual loss."""
    mom_x, mom_y, cont = compute_pde_residuals(pinn, xy)
    return jnp.mean(mom_x**2) + jnp.mean(mom_y**2) + jnp.mean(cont**2)


def boundary_loss(pinn, xy, u_target, v_target, p_target):
    """Compute boundary condition loss."""
    uvp = pinn(xy)
    u_pred, v_pred, p_pred = uvp[:, 0], uvp[:, 1], uvp[:, 2]

    loss_u = jnp.mean((u_pred - u_target) ** 2)
    loss_v = jnp.mean((v_pred - v_target) ** 2)
    loss_p = jnp.mean((p_pred - p_target) ** 2)

    return loss_u + loss_v + loss_p


def total_loss(pinn, xy_dom, xy_bc, u_bc, v_bc, p_bc, lambda_bc=1.0):
    """Total physics-informed loss."""
    loss_pde = pde_loss(pinn, xy_dom)
    loss_bc = boundary_loss(pinn, xy_bc, u_bc, v_bc, p_bc)
    return loss_pde + lambda_bc * loss_bc


# %% [markdown]
# ## Training
#
# Following DeepXDE: Adam optimizer with lr=1e-3 for 30000 iterations.

# %%
print()
print("Training PINN...")

opt = nnx.Optimizer(pinn, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(pinn, opt, xy_dom, xy_bc, u_bc_vals, v_bc_vals, p_bc_vals):
    """Single training step."""

    def loss_fn(model):
        return total_loss(model, xy_dom, xy_bc, u_bc_vals, v_bc_vals, p_bc_vals)

    loss, grads = nnx.value_and_grad(loss_fn)(pinn)
    opt.update(pinn, grads)
    return loss


losses = []
for epoch in range(EPOCHS):
    loss = train_step(pinn, opt, xy_domain, xy_boundary, u_bc, v_bc, p_bc)
    losses.append(float(loss))

    if (epoch + 1) % 5000 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:5d}/{EPOCHS}: loss={loss:.6e}")

print(f"Final loss: {losses[-1]:.6e}")

# %% [markdown]
# ## Evaluation
#
# Compare PINN solution against the analytical Kovasznay flow solution.

# %%
print()
print("Evaluating PINN...")

# Create evaluation grid
nx, ny = 100, 100
x_eval = jnp.linspace(X_MIN, X_MAX, nx)
y_eval = jnp.linspace(Y_MIN, Y_MAX, ny)
xx, yy = jnp.meshgrid(x_eval, y_eval)
xy_eval = jnp.column_stack([xx.ravel(), yy.ravel()])

# PINN prediction
uvp_pred = pinn(xy_eval)
u_pred = uvp_pred[:, 0].reshape(ny, nx)
v_pred = uvp_pred[:, 1].reshape(ny, nx)
p_pred = uvp_pred[:, 2].reshape(ny, nx)

# Exact solution
u_true = u_exact(xy_eval).reshape(ny, nx)
v_true = v_exact(xy_eval).reshape(ny, nx)
p_true = p_exact(xy_eval).reshape(ny, nx)

# Compute errors
l2_error_u = float(jnp.sqrt(jnp.sum((u_pred - u_true) ** 2) / jnp.sum(u_true**2)))
l2_error_v = float(jnp.sqrt(jnp.sum((v_pred - v_true) ** 2) / jnp.sum(v_true**2)))
# Avoid division by zero for pressure (can be near zero)
p_norm = jnp.sqrt(jnp.sum(p_true**2))
if p_norm > 1e-10:
    l2_error_p = float(jnp.sqrt(jnp.sum((p_pred - p_true) ** 2) / p_norm))
else:
    l2_error_p = float(jnp.sqrt(jnp.mean((p_pred - p_true) ** 2)))

# Mean PDE residual
mom_x, mom_y, cont = compute_pde_residuals(pinn, xy_eval)
mean_residual = (
    float(jnp.mean(jnp.abs(mom_x)) + jnp.mean(jnp.abs(mom_y)) + jnp.mean(jnp.abs(cont)))
    / 3.0
)

print(f"L2 relative error (u): {l2_error_u:.6e}")
print(f"L2 relative error (v): {l2_error_v:.6e}")
print(f"L2 relative error (p): {l2_error_p:.6e}")
print(f"Mean PDE residual:     {mean_residual:.6e}")

# %% [markdown]
# ## Visualization

# %%
# Create output directory
output_dir = Path("docs/assets/examples/navier_stokes_pinn")
output_dir.mkdir(parents=True, exist_ok=True)

mpl.use("Agg")

# Plot velocity and pressure fields
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# PINN solutions
im0 = axes[0, 0].imshow(
    np.array(u_pred),
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower",
    cmap="RdBu_r",
)
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
axes[0, 0].set_title("PINN: u (x-velocity)")
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(
    np.array(v_pred),
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower",
    cmap="RdBu_r",
)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
axes[0, 1].set_title("PINN: v (y-velocity)")
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(
    np.array(p_pred),
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower",
    cmap="RdBu_r",
)
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")
axes[0, 2].set_title("PINN: p (pressure)")
plt.colorbar(im2, ax=axes[0, 2])

# Errors
error_u = np.abs(np.array(u_pred) - np.array(u_true))
error_v = np.abs(np.array(v_pred) - np.array(v_true))
error_p = np.abs(np.array(p_pred) - np.array(p_true))

im3 = axes[1, 0].imshow(
    error_u,
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower",
    cmap="hot",
)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
axes[1, 0].set_title(f"Error: u (L2={l2_error_u:.2e})")
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(
    error_v,
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower",
    cmap="hot",
)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
axes[1, 1].set_title(f"Error: v (L2={l2_error_v:.2e})")
plt.colorbar(im4, ax=axes[1, 1])

im5 = axes[1, 2].imshow(
    error_p,
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    origin="lower",
    cmap="hot",
)
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("y")
axes[1, 2].set_title(f"Error: p (L2={l2_error_p:.2e})")
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.savefig(output_dir / "solution.png", dpi=150, bbox_inches="tight")
plt.close()
print()
print(f"Solution saved to {output_dir / 'solution.png'}")

# %%
# Training loss and cross-sections
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Training loss
axes[0].semilogy(losses, linewidth=1)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)

# Cross-section at y = 0.5
y_idx = ny // 2
axes[1].plot(
    np.array(x_eval), np.array(u_pred[y_idx, :]), "b-", label="PINN u", linewidth=2
)
axes[1].plot(
    np.array(x_eval), np.array(u_true[y_idx, :]), "b--", label="Exact u", linewidth=2
)
axes[1].plot(
    np.array(x_eval), np.array(v_pred[y_idx, :]), "r-", label="PINN v", linewidth=2
)
axes[1].plot(
    np.array(x_eval), np.array(v_true[y_idx, :]), "r--", label="Exact v", linewidth=2
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("velocity")
axes[1].set_title(f"Velocity at y = {float(y_eval[y_idx]):.2f}")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Cross-section at x = 0.25
x_idx = int(0.5 * nx)  # Approximately x = 0.25
axes[2].plot(
    np.array(y_eval), np.array(p_pred[:, x_idx]), "g-", label="PINN p", linewidth=2
)
axes[2].plot(
    np.array(y_eval), np.array(p_true[:, x_idx]), "g--", label="Exact p", linewidth=2
)
axes[2].set_xlabel("y")
axes[2].set_ylabel("pressure")
axes[2].set_title(f"Pressure at x = {float(x_eval[x_idx]):.2f}")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis saved to {output_dir / 'analysis.png'}")

# %%
# Summary
print()
print("=" * 70)
print("Navier-Stokes PINN (Kovasznay Flow) example completed")
print("=" * 70)
print()
print("Results Summary:")
print(f"  Final loss:          {losses[-1]:.6e}")
print(f"  L2 error (u):        {l2_error_u:.6e}")
print(f"  L2 error (v):        {l2_error_v:.6e}")
print(f"  L2 error (p):        {l2_error_p:.6e}")
print(f"  Mean PDE residual:   {mean_residual:.6e}")
print(f"  Parameters:          {n_params:,}")
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
