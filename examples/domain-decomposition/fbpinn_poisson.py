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
# # FBPINN: Finite Basis PINN on Damped Harmonic Oscillator
#
# This example demonstrates solving the damped harmonic oscillator ODE using FBPINN
# (Finite Basis Physics-Informed Neural Network). This is the canonical FBPINN
# benchmark problem from Moseley et al. (2023).
#
# **Reference:** Ben Moseley, Andrew Markham, Tarje Nissen-Meyer.
# "Finite Basis Physics-Informed Neural Networks (FBPINNs): a scalable domain decomposition
# approach for solving differential equations" (2023)
# https://github.com/benmoseley/FBPINNs
#
# **Problem:** HarmonicOscillator1DHardBC from `fbpinns/problems.py`

# %% [markdown]
# ## Setup and Imports

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

from opifex.neural.pinns.domain_decomposition import (
    FBPINN,
    FBPINNConfig,
    Subdomain,
)


print("=" * 70)
print("Opifex Example: FBPINN on Damped Harmonic Oscillator")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
# ## Configuration
#
# Following the FBPINNs reference implementation (HarmonicOscillator1DHardBC):
# - Domain: t in [0, 1]
# - Damped harmonic oscillator: m*u'' + mu*u' + k*u = 0
# - Parameters: d=2, w0=20 (gives mu=4, k=400)
# - Hard BC: u = 1 + tanh(t/sd)^2 * u_network (enforces u(0)=1, u'(0)=0)
# - sd=0.1 for the hard constraint smoothness

# %%
# Problem configuration (from FBPINNs reference)
T_MIN, T_MAX = 0.0, 1.0
D = 2.0  # Damping ratio
W0 = 20.0  # Natural frequency
MU = 2 * D  # Damping coefficient (=4)
K = W0**2  # Spring constant (=400)
SD = 0.1  # Smoothness parameter for hard BC

# Training configuration
N_DOMAIN = 2000  # Collocation points
EPOCHS = 20000
LEARNING_RATE = 0.001

# FBPINN configuration (4 overlapping subdomains)
NUM_SUBDOMAINS = 4
HIDDEN_DIMS = [32, 32]

print()
print(f"Damped harmonic oscillator: u'' + {MU}*u' + {K}*u = 0")
print(f"Domain: t in [{T_MIN}, {T_MAX}]")
print(f"Subdomains: {NUM_SUBDOMAINS} (overlapping)")
print(f"Hard BC: u = 1 + tanh(t/{SD})^2 * u_network")
print(f"Network per subdomain: [1] + {HIDDEN_DIMS} + [1]")
print(f"Training: {EPOCHS} epochs @ lr={LEARNING_RATE}")

# %% [markdown]
# ## Define the Problem
#
# The damped harmonic oscillator ODE:
# $$m \frac{d^2 u}{dt^2} + \mu \frac{du}{dt} + k u = 0$$
#
# With hard boundary constraint following FBPINNs:
# $$u = 1 + \tanh^2(t/\sigma) \cdot u_{network}$$
#
# At $t=0$: $\tanh(0)=0$, so $u=1$ (satisfies $u(0)=1$)
# The derivative also satisfies $u'(0)=0$ by construction.


# %%
def exact_solution(t, d=D, w0=W0):
    """Exact solution for damped harmonic oscillator.

    Reference: FBPINNs/fbpinns/problems.py HarmonicOscillator1D
    """
    w = jnp.sqrt(w0**2 - d**2)
    phi = jnp.arctan(-d / w)
    A = 1.0 / (2.0 * jnp.cos(phi))
    return jnp.exp(-d * t) * 2 * A * jnp.cos(phi + w * t)


def hard_bc_constraint(t, u_network, sd=SD):
    """Hard boundary constraint from FBPINNs.

    u = 1 + tanh(t/sd)^2 * u_network

    This enforces:
    - u(0) = 1 (since tanh(0) = 0)
    - u'(0) = 0 (by construction, derivative of tanh^2 is 0 at t=0)
    """
    return 1.0 + jnp.tanh(t / sd) ** 2 * u_network


# Verify initial conditions
print()
print("Damped harmonic oscillator: u'' + mu*u' + k*u = 0")
print(f"  Damping coefficient: mu = {MU}")
print(f"  Spring constant: k = {K}")
print(f"  IC: u(0) = {float(exact_solution(0.0)):.6f} (should be 1.0)")
print(f"  Hard BC: u = 1 + tanh(t/{SD})^2 * u_network")


# %% [markdown]
# ## Create Custom FBPINN with Hard BC
#
# We subclass FBPINN to apply the hard boundary constraint directly in the forward pass.
# This follows the standard Opifex pattern for implementing hard constraints (see AllenCahnPINN).


# %%
class HarmonicOscillatorFBPINN(FBPINN):
    """FBPINN subclass with hard boundary constraint for harmonic oscillator.

    This extends Opifex's FBPINN by overriding __call__ to apply the hard BC.
    The constraint u = 1 + tanh(t/sd)^2 * u_network enforces u(0)=1, u'(0)=0.
    """

    def __init__(
        self,
        subdomains,
        interfaces,
        hidden_dims,
        *,
        sd: float = 0.1,
        config=None,
        rngs,
    ):
        """Initialize FBPINN with hard BC constraint.

        Args:
            subdomains: List of subdomain definitions
            interfaces: List of interface definitions (empty for FBPINN)
            hidden_dims: Hidden layer dimensions for subdomain networks
            sd: Smoothness parameter for hard BC (default 0.1)
            config: FBPINN configuration
            rngs: Random number generators
        """
        super().__init__(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=hidden_dims,
            config=config,
            rngs=rngs,
        )
        self.sd = sd

    def get_raw_output(self, t):
        """Get raw network output without hard BC constraint.

        Useful for visualization and analysis of the hard BC effect.
        """
        return super().__call__(t)

    def __call__(self, t):
        """Forward pass with hard BC constraint.

        Output transform: u = 1 + tanh(t/sd)^2 * u_network
        This enforces:
          - u(0) = 1 (since tanh(0) = 0)
          - u'(0) = 0 (derivative of tanh^2 is 0 at t=0)
        """
        # Get window-blended output from parent FBPINN
        u_network = self.get_raw_output(t)
        # Apply hard BC: u = 1 + tanh(t/sd)^2 * u_network
        return 1.0 + jnp.tanh(t / self.sd) ** 2 * u_network


# Create overlapping 1D subdomains following FBPINNs convention
# Subdomains should overlap ~50% for smooth blending
# Extend bounds slightly beyond domain to ensure full coverage at boundaries
subdomain_width = (T_MAX - T_MIN) / NUM_SUBDOMAINS * 1.5
extension = 0.05  # Small extension beyond domain bounds

subdomains = []
for i in range(NUM_SUBDOMAINS):
    center = T_MIN + (i + 0.5) * (T_MAX - T_MIN) / NUM_SUBDOMAINS
    t_lo = center - subdomain_width / 2
    t_hi = center + subdomain_width / 2
    # Extend first and last subdomains beyond domain boundaries
    if i == 0:
        t_lo = T_MIN - extension
    if i == NUM_SUBDOMAINS - 1:
        t_hi = T_MAX + extension
    bounds = jnp.array([[t_lo, t_hi]])
    subdomains.append(Subdomain(id=i, bounds=bounds))

# FBPINN configuration
fbpinn_config = FBPINNConfig(
    window_type="cosine",
    normalize_windows=True,
)

# Create FBPINN model with hard BC (subclass approach)
print()
print("Creating FBPINN model...")
model = HarmonicOscillatorFBPINN(
    subdomains=subdomains,
    interfaces=[],
    hidden_dims=HIDDEN_DIMS,
    sd=SD,
    config=fbpinn_config,
    rngs=nnx.Rngs(42),
)


# Count parameters
def count_params(m):
    """Count total parameters in model."""
    return sum(p.size for p in jax.tree.leaves(nnx.state(m)))


total_params = count_params(model)
print(f"Total FBPINN parameters: {total_params}")
print(f"Parameters per subdomain: ~{total_params // len(subdomains)}")
print()
print("Subdomain bounds:")
for i, sd in enumerate(subdomains):
    print(
        f"  Subdomain {i}: [{float(sd.bounds[0, 0]):.3f}, {float(sd.bounds[0, 1]):.3f}]"
    )

# %% [markdown]
# ## Generate Collocation Points

# %%
key = jax.random.PRNGKey(42)

# Domain interior points (uniform random)
t_domain = jax.random.uniform(key, (N_DOMAIN,), minval=T_MIN, maxval=T_MAX)
t_domain = t_domain.reshape(-1, 1)

print()
print("Generating collocation points...")
print(f"Domain points: {t_domain.shape}")
print("(No explicit BC points needed - hard constraint enforces BC)")

# %% [markdown]
# ## Define Physics-Informed Loss
#
# With hard BC constraint, we only need the PDE residual loss.
# The boundary conditions are automatically satisfied.


# %%
def compute_pde_residual(model, t, mu=MU, k=K):
    """Compute harmonic oscillator PDE residual: u'' + mu*u' + k*u = 0."""

    def u_scalar(t_single):
        return model(t_single.reshape(1, 1)).squeeze()

    def residual_single(t_single):
        u = u_scalar(t_single)

        # First derivative
        u_t = jax.grad(u_scalar)(t_single)[0]

        # Second derivative
        def u_t_fn(ts):
            return jax.grad(u_scalar)(ts)[0]

        u_tt = jax.grad(u_t_fn)(t_single)[0]

        # Residual: u'' + mu*u' + k*u = 0
        return u_tt + mu * u_t + k * u

    return jax.vmap(residual_single)(t)


def pde_loss(model, t):
    """Mean squared PDE residual (only loss needed with hard BC)."""
    residual = compute_pde_residual(model, t)
    return jnp.mean(residual**2)


# %% [markdown]
# ## Training

# %%
print()
print("Training FBPINN...")

opt = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(model, opt, t_dom):
    """Single training step with gradient update."""

    def loss_fn(m):
        return pde_loss(m, t_dom)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss


losses = []
for epoch in range(EPOCHS):
    loss = train_step(model, opt, t_domain)
    losses.append(float(loss))

    if (epoch + 1) % 4000 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:5d}/{EPOCHS}: loss={loss:.6e}")

print(f"Final loss: {losses[-1]:.6e}")

# %% [markdown]
# ## Evaluation

# %%
print()
print("Evaluating FBPINN...")

# Create evaluation grid
n_eval = 500
t_eval = jnp.linspace(T_MIN, T_MAX, n_eval).reshape(-1, 1)

# Compute predictions
u_pred = model(t_eval).squeeze()
u_exact = exact_solution(t_eval.squeeze())

# Compute errors
error = jnp.abs(u_pred - u_exact)
l2_error = jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2))
rel_l2_error = l2_error / jnp.sqrt(jnp.mean(u_exact**2))
max_error = jnp.max(error)
mean_error = jnp.mean(error)

# Compute PDE residual on grid
pde_res = compute_pde_residual(model, t_eval)
mean_pde_residual = jnp.mean(jnp.abs(pde_res))

# Check boundary conditions (should be exactly satisfied by hard BC)
u_pred_0 = float(model(jnp.array([[0.0]])).squeeze())
print(f"Relative L2 error:   {rel_l2_error:.6e}")
print(f"Maximum point error: {max_error:.6e}")
print(f"Mean point error:    {mean_error:.6e}")
print(f"Mean PDE residual:   {mean_pde_residual:.6e}")
print(f"u(0) predicted:      {u_pred_0:.6f} (exact: 1.0, hard BC)")

# %% [markdown]
# ## Visualization

# %%
# Verify window weights sum to 1 (partition of unity)
weights = model.compute_window_weights(t_eval)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Solution comparison
axes[0, 0].plot(t_eval, u_exact, "b-", linewidth=2, label="Exact")
axes[0, 0].plot(t_eval, u_pred, "r--", linewidth=2, label="FBPINN")
axes[0, 0].set_xlabel("t")
axes[0, 0].set_ylabel("u")
axes[0, 0].set_title("Damped Harmonic Oscillator Solution")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Point-wise error
axes[0, 1].semilogy(t_eval, error)
axes[0, 1].set_xlabel("t")
axes[0, 1].set_ylabel("Absolute Error")
axes[0, 1].set_title(f"Error (max={max_error:.4e})")
axes[0, 1].grid(True, alpha=0.3)

# Window functions
cmap = plt.colormaps["tab10"]
colors = [cmap(i) for i in range(NUM_SUBDOMAINS)]
for i in range(NUM_SUBDOMAINS):
    axes[1, 0].plot(t_eval, weights[:, i], color=colors[i], label=f"Subdomain {i}")
    # Mark subdomain bounds
    sd_bounds = subdomains[i]
    axes[1, 0].axvline(
        sd_bounds.bounds[0, 0], color=colors[i], linestyle=":", alpha=0.5
    )
    axes[1, 0].axvline(
        sd_bounds.bounds[0, 1], color=colors[i], linestyle=":", alpha=0.5
    )
axes[1, 0].plot(t_eval, jnp.sum(weights, axis=-1), "k--", linewidth=2, label="Sum")
axes[1, 0].set_xlabel("t")
axes[1, 0].set_ylabel("Window Weight")
axes[1, 0].set_title("FBPINN Window Functions (Partition of Unity)")
axes[1, 0].legend(loc="center right")
axes[1, 0].grid(True, alpha=0.3)

# Training history
axes[1, 1].semilogy(losses)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Total Loss")
axes[1, 1].set_title("Training History")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "docs/assets/examples/fbpinn_poisson/solution.png", dpi=150, bbox_inches="tight"
)
print()
print("Saved: docs/assets/examples/fbpinn_poisson/solution.png")
plt.show()

# %%
# Analysis: Individual subdomain networks and hard BC effect
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Show the hard BC constraint effect
t_test = jnp.linspace(0, 1, 100).reshape(-1, 1)
u_network_raw = model.get_raw_output(t_test).squeeze()
u_with_bc = model(t_test).squeeze()

axes[0].plot(t_test, u_network_raw, "g-", linewidth=1.5, label="Raw network output")
axes[0].plot(t_test, u_with_bc, "b-", linewidth=2, label="With hard BC")
axes[0].plot(
    t_test, exact_solution(t_test.squeeze()), "r--", linewidth=2, label="Exact"
)
axes[0].set_xlabel("t")
axes[0].set_ylabel("u")
axes[0].set_title("Hard BC Effect: u = 1 + tanh^2(t/sigma) * u_network")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PDE residual
axes[1].plot(t_eval, jnp.abs(pde_res), "b-", linewidth=1)
axes[1].set_xlabel("t")
axes[1].set_ylabel("|PDE Residual|")
axes[1].set_title(f"PDE Residual (mean={mean_pde_residual:.4e})")
axes[1].set_yscale("log")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "docs/assets/examples/fbpinn_poisson/analysis.png", dpi=150, bbox_inches="tight"
)
print("Saved: docs/assets/examples/fbpinn_poisson/analysis.png")
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
print(f"Mean PDE Residual:   {mean_pde_residual:.6e}")
print(f"u(0) predicted:      {u_pred_0:.6f} (exact: 1.0)")
print(f"Total Parameters:    {total_params}")
print(f"Training Epochs:     {EPOCHS}")
print(f"Number of Subdomains:{len(subdomains)}")
print()
print("Window weights sum check:")
weight_sum = jnp.sum(weights, axis=-1)
print(f"  Min sum: {jnp.min(weight_sum):.6f}")
print(f"  Max sum: {jnp.max(weight_sum):.6f}")
print("  (Should be close to 1.0 for partition of unity)")
