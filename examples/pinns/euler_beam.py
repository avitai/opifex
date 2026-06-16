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
# # Euler-Bernoulli Beam PINN
#
# This example demonstrates solving the Euler-Bernoulli beam equation using a PINN.
# This is a fourth-order ODE from structural mechanics describing beam deflection.
#
# **Reference**: DeepXDE `examples/pinn_forward/Euler_beam.py`

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
mpl.use("Agg")

# Problem configuration
# EI * d^4w/dx^4 = q, we set EI=1, q=-1 for simplicity
Q = -1.0  # Distributed load (negative = downward)

# Domain bounds
X_MIN, X_MAX = 0.0, 1.0

# Collocation points (DeepXDE uses only 10 domain + 2 boundary)
N_DOMAIN = 100
N_BOUNDARY = 10  # Points at x=0 and x=1

# Network configuration (matching DeepXDE: [1] + [20]*3 + [1])
HIDDEN_DIMS = [20, 20, 20]

# Training configuration
EPOCHS = 15000
LEARNING_RATE = 1e-3

# %% [markdown]
# ## Problem Definition
#
# **Euler-Bernoulli Beam Equation (4th order ODE):**
#
# $$EI \frac{d^4 w}{dx^4} = q(x)$$
#
# where:
# - $w(x)$ = deflection
# - $EI$ = flexural rigidity (set to 1)
# - $q(x)$ = distributed load (set to -1)
#
# **Cantilever Beam Boundary Conditions:**
# - Fixed end (x=0): $w(0) = 0$, $w'(0) = 0$
# - Free end (x=1): $w''(1) = 0$ (moment), $w'''(1) = 0$ (shear)
#
# **Exact Solution:**
# $$w(x) = -\frac{x^4}{24} + \frac{x^3}{6} - \frac{x^2}{4}$$


# %%
def exact_solution(x):
    """Exact solution for cantilever beam with uniform load."""
    return -(x**4) / 24 + (x**3) / 6 - (x**2) / 4


def exact_derivative(x):
    """First derivative: w'(x)."""
    return -(x**3) / 6 + (x**2) / 2 - x / 2


def exact_second_derivative(x):
    """Second derivative: w''(x)."""
    return -(x**2) / 2 + x - 0.5


def exact_third_derivative(x):
    """Third derivative: w'''(x)."""
    return -x + 1


# %% [markdown]
# ## PINN Architecture


# %%
class EulerBeamPINN(nnx.Module):
    """PINN for the Euler-Bernoulli beam equation."""

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs):
        """Initialize the PINN."""
        super().__init__()

        layers = []
        in_features = 1  # x only (no time)

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))
            in_features = hidden_dim

        layers.append(nnx.Linear(in_features, 1, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the network."""
        h = x
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)


# %% [markdown]
# ## Physics-Informed Loss
#
# For 4th-order derivatives, we need to compute $w''''$ using nested differentiation.


# %%
def compute_derivatives(pinn, x):
    """Compute w, w', w'', w''', w'''' at given points."""

    def w_scalar(x_single):
        """Scalar output for differentiation."""
        return pinn(x_single.reshape(1, 1)).squeeze()

    def derivatives_single(x_single):
        """Compute all derivatives at a single point."""
        # w
        w = w_scalar(x_single)

        # w' = dw/dx
        w_x = jax.grad(w_scalar)(x_single)[0]

        # w'' = d^2w/dx^2
        def w_x_fn(xs):
            return jax.grad(w_scalar)(xs)[0]

        w_xx = jax.grad(w_x_fn)(x_single)[0]

        # w''' = d^3w/dx^3
        def w_xx_fn(xs):
            def w_x_inner(xs2):
                return jax.grad(w_scalar)(xs2)[0]

            return jax.grad(w_x_inner)(xs)[0]

        w_xxx = jax.grad(w_xx_fn)(x_single)[0]

        # w'''' = d^4w/dx^4
        def w_xxx_fn(xs):
            def w_xx_inner(xs2):
                def w_x_inner2(xs3):
                    return jax.grad(w_scalar)(xs3)[0]

                return jax.grad(w_x_inner2)(xs2)[0]

            return jax.grad(w_xx_inner)(xs)[0]

        w_xxxx = jax.grad(w_xxx_fn)(x_single)[0]

        return w, w_x, w_xx, w_xxx, w_xxxx

    # Vectorize
    return jax.vmap(derivatives_single)(x)


def pde_loss(pinn, x):
    """PDE loss: w'''' + 1 = 0 (since q=-1 and EI=1)."""
    _, _, _, _, w_xxxx = compute_derivatives(pinn, x)
    residual = w_xxxx - Q  # w'''' = q = -1
    return jnp.mean(residual**2)


def bc_loss(pinn, x_left, x_right):
    """Boundary condition losses."""
    # Left BC: w(0) = 0, w'(0) = 0
    w_l, w_x_l, _, _, _ = compute_derivatives(pinn, x_left)
    loss_w0 = jnp.mean(w_l**2)
    loss_wx0 = jnp.mean(w_x_l**2)

    # Right BC: w''(1) = 0, w'''(1) = 0
    _, _, w_xx_r, w_xxx_r, _ = compute_derivatives(pinn, x_right)
    loss_wxx1 = jnp.mean(w_xx_r**2)
    loss_wxxx1 = jnp.mean(w_xxx_r**2)

    return loss_w0 + loss_wx0 + loss_wxx1 + loss_wxxx1


def total_loss(pinn, x_dom, x_left, x_right, lambda_bc=100.0):
    """Total loss = PDE + weighted BC."""
    loss_pde = pde_loss(pinn, x_dom)
    loss_bc = bc_loss(pinn, x_left, x_right)
    return loss_pde + lambda_bc * loss_bc


# %% [markdown]
# ## Training


# %%
@nnx.jit
def train_step(pinn, opt, x_dom, x_left, x_right):
    """Perform one training step."""

    def loss_fn(model):
        return total_loss(model, x_dom, x_left, x_right)

    loss, grads = nnx.value_and_grad(loss_fn)(pinn)
    opt.update(pinn, grads)
    return loss


# %% [markdown]
# ## Run the example
#
# Builds collocation points, trains the beam PINN, evaluates against the exact
# solution and boundary conditions, and saves the visualizations.


# %%
def main() -> dict[str, float | int]:
    """Train an Euler-Bernoulli beam PINN, evaluate against the exact solution, and plot."""
    print("=" * 70)
    print("Opifex Example: Euler-Bernoulli Beam PINN")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print("Euler-Bernoulli beam: EI * d^4w/dx^4 = q")
    print(f"  Load: q = {Q}")
    print(f"Domain: x in [{X_MIN}, {X_MAX}]")
    print(f"Collocation: {N_DOMAIN} domain, {N_BOUNDARY} boundary")
    print(f"Network: [1] + {HIDDEN_DIMS} + [1]")
    print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

    pinn = EulerBeamPINN(hidden_dims=HIDDEN_DIMS, rngs=nnx.Rngs(42))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
    print(f"PINN parameters: {n_params:,}")

    key = jax.random.PRNGKey(42)
    x_domain = jax.random.uniform(key, (N_DOMAIN,), minval=X_MIN, maxval=X_MAX).reshape(-1, 1)
    x_left = jnp.zeros((N_BOUNDARY // 2, 1))  # x = 0
    x_right = jnp.ones((N_BOUNDARY // 2, 1))  # x = 1
    print(f"Domain points: {x_domain.shape}")
    print(f"Left BC points: {x_left.shape}")
    print(f"Right BC points: {x_right.shape}")

    print("Training PINN...")
    opt = nnx.Optimizer(pinn, optax.adam(LEARNING_RATE), wrt=nnx.Param)
    losses = []
    for epoch in range(EPOCHS):
        loss = train_step(pinn, opt, x_domain, x_left, x_right)
        losses.append(float(loss))
        if (epoch + 1) % 3000 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:5d}/{EPOCHS}: loss={loss:.6e}")
    final_loss = losses[-1]
    print(f"Final loss: {final_loss:.6e}")

    print("Evaluating PINN...")
    nx = 200
    x_eval = jnp.linspace(X_MIN, X_MAX, nx).reshape(-1, 1)

    w_pred = pinn(x_eval).squeeze()
    w_exact = exact_solution(x_eval.squeeze())

    error = jnp.abs(w_pred - w_exact)
    l2_error = float(jnp.sqrt(jnp.sum((w_pred - w_exact) ** 2) / jnp.sum(w_exact**2 + 1e-10)))
    max_error = float(jnp.max(error))
    mean_error = float(jnp.mean(error))

    w_0, w_x_0, _, _, _ = compute_derivatives(pinn, jnp.array([[0.0]]))
    _, _, w_xx_1, w_xxx_1, _ = compute_derivatives(pinn, jnp.array([[1.0]]))

    print(f"Relative L2 error:   {l2_error:.6e}")
    print(f"Maximum point error: {max_error:.6e}")
    print(f"Mean point error:    {mean_error:.6e}")
    print("Boundary condition errors:")
    print(f"  w(0) = {float(w_0[0]):.6e} (should be 0)")
    print(f"  w'(0) = {float(w_x_0[0]):.6e} (should be 0)")
    print(f"  w''(1) = {float(w_xx_1[0]):.6e} (should be 0)")
    print(f"  w'''(1) = {float(w_xxx_1[0]):.6e} (should be 0)")

    output_dir = Path("docs/assets/examples/euler_beam_pinn")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(np.array(x_eval), np.array(w_pred), "b-", label="PINN", linewidth=2)
    axes[0].plot(np.array(x_eval), np.array(w_exact), "r--", label="Exact", linewidth=2, alpha=0.7)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("w(x)")
    axes[0].set_title("Beam Deflection")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()  # Deflection is typically shown downward

    axes[1].plot(np.array(x_eval), np.array(error), "g-", linewidth=2)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|Error|")
    axes[1].set_title(f"Point-wise Error (L2={l2_error:.2e})")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(losses, linewidth=1)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Training Loss")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "solution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Solution saved to {output_dir / 'solution.png'}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _, w_x_all, w_xx_all, w_xxx_all, w_xxxx_all = compute_derivatives(pinn, x_eval)
    w_x_exact = exact_derivative(x_eval.squeeze())
    w_xx_exact = exact_second_derivative(x_eval.squeeze())
    w_xxx_exact = exact_third_derivative(x_eval.squeeze())
    w_xxxx_exact = jnp.full_like(x_eval.squeeze(), -1.0)  # w'''' = q = -1

    axes[0, 0].plot(np.array(x_eval), np.array(w_x_all), "b-", label="PINN", linewidth=2)
    axes[0, 0].plot(
        np.array(x_eval), np.array(w_x_exact), "r--", label="Exact", linewidth=2, alpha=0.7
    )
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("w'(x)")
    axes[0, 0].set_title("Slope")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(np.array(x_eval), np.array(w_xx_all), "b-", label="PINN", linewidth=2)
    axes[0, 1].plot(
        np.array(x_eval), np.array(w_xx_exact), "r--", label="Exact", linewidth=2, alpha=0.7
    )
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("w''(x)")
    axes[0, 1].set_title("Curvature (Moment)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(np.array(x_eval), np.array(w_xxx_all), "b-", label="PINN", linewidth=2)
    axes[1, 0].plot(
        np.array(x_eval),
        np.array(w_xxx_exact),
        "r--",
        label="Exact",
        linewidth=2,
        alpha=0.7,
    )
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("w'''(x)")
    axes[1, 0].set_title("Shear")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(np.array(x_eval), np.array(w_xxxx_all), "b-", label="PINN", linewidth=2)
    axes[1, 1].plot(
        np.array(x_eval),
        np.array(w_xxxx_exact),
        "r--",
        label="Exact (q=-1)",
        linewidth=2,
        alpha=0.7,
    )
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("w''''(x)")
    axes[1, 1].set_title("Load (4th derivative)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Analysis saved to {output_dir / 'analysis.png'}")

    print("=" * 70)
    print("Euler-Bernoulli Beam PINN example completed")
    print(f"  Final loss:          {final_loss:.6e}")
    print(f"  Relative L2 error:   {l2_error:.6e}")
    print(f"  Maximum error:       {max_error:.6e}")
    print(f"  BC w(0):             {float(w_0[0]):.6e}")
    print(f"  BC w'(0):            {float(w_x_0[0]):.6e}")
    print(f"  BC w''(1):           {float(w_xx_1[0]):.6e}")
    print(f"  BC w'''(1):          {float(w_xxx_1[0]):.6e}")
    print(f"  Parameters:          {n_params:,}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return {
        "final_loss": final_loss,
        "l2_relative_error": l2_error,
        "max_error": max_error,
        "mean_error": mean_error,
        "bc_error": float(
            (abs(float(w_0[0])) + abs(float(w_x_0[0])) + abs(float(w_xx_1[0])) + abs(float(w_xxx_1[0])))
            / 4
        ),
        "param_count": int(n_params),
    }


# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
