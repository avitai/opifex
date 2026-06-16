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
# Neural Tangent Kernel (NTK) Analysis for PINNs

This example demonstrates how to use the Neural Tangent Kernel to diagnose
and understand PINN training dynamics. NTK analysis reveals spectral bias,
predicts convergence rates, and identifies problematic modes.

**Key Concepts:**
- Empirical NTK computation for PINNs
- Eigenvalue analysis and condition number
- Spectral bias detection
- Convergence rate prediction from NTK spectrum
- Mode-wise error decay analysis

**SciML Context:**
Understanding why PINNs struggle with certain problems (high-frequency solutions,
stiff PDEs) can be explained through NTK theory. The eigenvalue spectrum
determines which solution modes are learned quickly vs slowly.
"""

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
from opifex.core.physics.ntk.diagnostics import (
    detect_spectral_bias,
    estimate_epochs_to_convergence,
    identify_slow_modes,
)
from opifex.core.physics.ntk.spectral_analysis import (
    compute_condition_number,
    compute_effective_rank,
)
from opifex.core.physics.ntk.wrapper import NTKWrapper


# %% [markdown]
"""
## Step 1: Define a Simple PINN

We use a Poisson equation PINN to demonstrate NTK analysis:
-Δu = f(x) on [0, 1]
u(0) = u(1) = 0
"""


# %%
class PoissonPINN(nnx.Module):
    """Simple PINN for 1D Poisson equation."""

    def __init__(self, hidden_dims: list[int] | None = None, *, rngs: nnx.Rngs):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32]
        layers = []
        in_dim = 1

        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        layers.append(nnx.Linear(in_dim, 1, rngs=rngs))
        self.layers = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the PINN."""
        h = x
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)


# %% [markdown]
"""
## Step 4 helpers: PDE residual and PINN loss

We'll track how the NTK and its spectrum evolve during training. These
helpers define the Poisson residual and the total PINN loss.
"""


# %%
def compute_pde_residual(pinn, x):
    """Compute Poisson PDE residual: -d2u/dx2 - f(x) = 0."""

    def u_scalar(x_single):
        return pinn(x_single.reshape(1, 1)).squeeze()

    def residual_single(x_single):
        # Second derivative
        d2u_dx2 = jax.grad(jax.grad(u_scalar))(x_single)

        # Source term: f(x) = pi^2 * sin(pi*x)
        f = jnp.pi**2 * jnp.sin(jnp.pi * x_single)

        return -d2u_dx2 - f

    def residual_single_wrapper(x_val):
        return residual_single(x_val.squeeze())

    return jax.vmap(residual_single_wrapper)(x)


def pinn_loss(pinn, x_domain, x_bc):
    """Total PINN loss."""
    # PDE residual
    residual = compute_pde_residual(pinn, x_domain)
    loss_pde = jnp.mean(residual**2)

    # Boundary conditions
    u_bc = pinn(x_bc).squeeze()
    loss_bc = jnp.mean(u_bc**2)

    return loss_pde + 10.0 * loss_bc


# %% [markdown]
"""
## Run the example

Steps 2-6 run inside ``main()``: generate collocation points, compute the
initial NTK and diagnostics, train the PINN while tracking NTK evolution,
evaluate the solution, and save visualizations.
"""


# %%
def main() -> dict[str, float | int]:
    """Run NTK analysis for a 1D Poisson PINN and return scalar metrics."""
    # Configuration
    seed = 42
    n_collocation = 100  # Points for NTK computation
    learning_rate = 1e-3
    training_steps = 500
    ntk_compute_frequency = 100
    output_dir = "docs/assets/examples/ntk_analysis"

    print("=" * 70)
    print("Opifex Example: NTK Analysis for PINNs")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    # Step 1: Create the PINN model
    print()
    print("Creating PINN model...")

    pinn = PoissonPINN(hidden_dims=[32, 32], rngs=nnx.Rngs(seed))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(pinn, nnx.Param)))
    print("  Architecture: [1] -> [32] -> [32] -> [1]")
    print(f"  Parameters: {n_params:,}")

    # Step 2: Generate collocation points
    print()
    print("Generating collocation points...")

    key = jax.random.PRNGKey(seed)

    # Domain points for NTK computation
    x_ntk = jax.random.uniform(key, (n_collocation, 1), minval=0.0, maxval=1.0)

    # Training points (more for actual training)
    key, subkey = jax.random.split(key)
    x_train = jax.random.uniform(subkey, (500, 1), minval=0.0, maxval=1.0)

    # Boundary points
    x_bc = jnp.array([[0.0], [1.0]])

    print(f"  NTK computation points: {x_ntk.shape}")
    print(f"  Training points: {x_train.shape}")

    # Step 3: Compute initial NTK and analyze spectrum
    print()
    print("Computing initial NTK...")
    print("-" * 50)

    # Create NTK wrapper
    ntk_wrapper = NTKWrapper(pinn)

    # Compute NTK matrix
    ntk_matrix = ntk_wrapper.compute_ntk(x_ntk)
    print(f"  NTK matrix shape: {ntk_matrix.shape}")

    # Compute eigenvalues (clip small negatives from numerical noise)
    eigenvalues_raw = ntk_wrapper.compute_eigenvalues(x_ntk)
    eigenvalues = jnp.maximum(eigenvalues_raw, 1e-10)  # Ensure positive for stability
    print(f"  Eigenvalues range: [{float(eigenvalues[-1]):.6e}, {float(eigenvalues[0]):.6e}]")

    # Compute diagnostics
    cond_number = compute_condition_number(eigenvalues)
    eff_rank = compute_effective_rank(eigenvalues)
    spectral_bias = detect_spectral_bias(eigenvalues)

    print()
    print("Initial NTK Diagnostics:")
    print(f"  Condition number: {float(cond_number):.2e}")
    print(f"  Effective rank: {float(eff_rank):.2f}")
    print(f"  Spectral bias indicator: {float(spectral_bias):.2f}")

    # Identify slow modes
    slow_modes = identify_slow_modes(eigenvalues, learning_rate, threshold=0.999)
    n_slow = int(jnp.sum(slow_modes))
    print(f"  Slow-converging modes: {n_slow}/{len(eigenvalues)}")

    # Estimate epochs to convergence
    est_epochs = estimate_epochs_to_convergence(
        eigenvalues, learning_rate, target_reduction=0.01
    )
    est_epochs_int = int(min(float(est_epochs), 1e9))  # Cap at 1 billion for display
    print(f"  Estimated epochs to 99% convergence: {est_epochs_int:,}")

    # Step 4: Train PINN and track NTK evolution
    print()
    print("Training PINN with NTK tracking...")
    print("-" * 50)

    opt = nnx.Optimizer(pinn, optax.adam(learning_rate), wrt=nnx.Param)

    losses = []
    ntk_history = {
        "step": [],
        "condition_number": [],
        "effective_rank": [],
        "max_eigenvalue": [],
        "min_eigenvalue": [],
        "eigenvalues": [],
    }

    @nnx.jit
    def train_step(pinn, opt, x_domain, x_bc):
        """Single training step for PINN."""

        def loss_fn(model):
            return pinn_loss(model, x_domain, x_bc)

        loss, grads = nnx.value_and_grad(loss_fn)(pinn)
        opt.update(pinn, grads)
        return loss

    for step in range(training_steps):
        loss = train_step(pinn, opt, x_train, x_bc)
        losses.append(float(loss))

        # Track NTK at specified frequency
        if step % ntk_compute_frequency == 0:
            eigenvalues_raw = ntk_wrapper.compute_eigenvalues(x_ntk)
            eigenvalues = jnp.maximum(eigenvalues_raw, 1e-10)  # Clip for stability

            ntk_history["step"].append(step)
            ntk_history["condition_number"].append(float(compute_condition_number(eigenvalues)))
            ntk_history["effective_rank"].append(float(compute_effective_rank(eigenvalues)))
            ntk_history["max_eigenvalue"].append(float(eigenvalues[0]))
            ntk_history["min_eigenvalue"].append(float(jnp.maximum(eigenvalues[-1], 1e-10)))
            ntk_history["eigenvalues"].append(np.array(jnp.maximum(eigenvalues, 1e-10)))

            print(
                f"  Step {step:4d}: loss={loss:.6e}, "
                f"cond={ntk_history['condition_number'][-1]:.2e}"
            )

    # Final NTK computation
    eigenvalues_final_raw = ntk_wrapper.compute_eigenvalues(x_ntk)
    eigenvalues_final = jnp.maximum(eigenvalues_final_raw, 1e-10)
    ntk_history["step"].append(training_steps)
    ntk_history["condition_number"].append(float(compute_condition_number(eigenvalues_final)))
    ntk_history["effective_rank"].append(float(compute_effective_rank(eigenvalues_final)))
    ntk_history["max_eigenvalue"].append(float(eigenvalues_final[0]))
    ntk_history["min_eigenvalue"].append(float(jnp.maximum(eigenvalues_final[-1], 1e-10)))
    ntk_history["eigenvalues"].append(np.array(jnp.maximum(eigenvalues_final, 1e-10)))

    print(
        f"  Step {training_steps:4d}: loss={losses[-1]:.6e}, "
        f"cond={ntk_history['condition_number'][-1]:.2e}"
    )

    # Step 5: Evaluate solution quality
    print()
    print("Evaluating trained PINN...")

    # Evaluation grid
    x_eval = jnp.linspace(0, 1, 100).reshape(-1, 1)

    # PINN prediction
    u_pred = pinn(x_eval).squeeze()

    # Exact solution: u(x) = sin(pi*x) for f(x) = pi^2*sin(pi*x)
    u_exact = jnp.sin(jnp.pi * x_eval.squeeze())

    # L2 error
    l2_error = float(jnp.sqrt(jnp.mean((u_pred - u_exact) ** 2)))
    max_error = float(jnp.max(jnp.abs(u_pred - u_exact)))

    print(f"  L2 error: {l2_error:.6e}")
    print(f"  Max error: {max_error:.6e}")

    # Step 6: Visualization
    print()
    print("Generating visualizations...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mpl.use("Agg")

    # Figure 1: NTK Eigenvalue Spectrum Evolution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Eigenvalue spectrum at different training stages
    ax1 = axes[0, 0]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(ntk_history["eigenvalues"])))
    for i, (history_step, eigs) in enumerate(
        zip(ntk_history["step"], ntk_history["eigenvalues"], strict=False)
    ):
        ax1.semilogy(eigs, color=colors[i], label=f"Step {history_step}", alpha=0.8)
    ax1.set_xlabel("Eigenvalue Index", fontsize=12)
    ax1.set_ylabel("Eigenvalue (log scale)", fontsize=12)
    ax1.set_title("NTK Eigenvalue Spectrum Evolution", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Condition number over training
    ax2 = axes[0, 1]
    ax2.semilogy(
        ntk_history["step"],
        ntk_history["condition_number"],
        "b-o",
        linewidth=2,
        markersize=6,
    )
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Condition Number (log scale)", fontsize=12)
    ax2.set_title("NTK Condition Number During Training", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Training loss
    ax3 = axes[1, 0]
    ax3.semilogy(losses, linewidth=1)
    ax3.set_xlabel("Training Step", fontsize=12)
    ax3.set_ylabel("Loss (log scale)", fontsize=12)
    ax3.set_title("Training Loss", fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Eigenvalue range
    ax4 = axes[1, 1]
    ax4.semilogy(
        ntk_history["step"],
        ntk_history["max_eigenvalue"],
        "b-o",
        label="Max eigenvalue",
        linewidth=2,
    )
    ax4.semilogy(
        ntk_history["step"],
        ntk_history["min_eigenvalue"],
        "r-s",
        label="Min eigenvalue",
        linewidth=2,
    )
    ax4.set_xlabel("Training Step", fontsize=12)
    ax4.set_ylabel("Eigenvalue (log scale)", fontsize=12)
    ax4.set_title("NTK Eigenvalue Range", fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ntk_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/ntk_evolution.png")

    # Figure 2: Solution and Error
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Solution comparison
    ax1 = axes[0]
    ax1.plot(np.array(x_eval), np.array(u_exact), "b-", label="Exact", linewidth=2)
    ax1.plot(np.array(x_eval), np.array(u_pred), "r--", label="PINN", linewidth=2)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("u(x)", fontsize=12)
    ax1.set_title("Solution Comparison", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pointwise error
    ax2 = axes[1]
    error = np.abs(np.array(u_pred - u_exact))
    ax2.semilogy(np.array(x_eval), error, "k-", linewidth=1.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("|u_pred - u_exact| (log scale)", fontsize=12)
    ax2.set_title(f"Pointwise Error (L2 = {l2_error:.2e})", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/solution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/solution.png")

    # Results summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print("NTK Analysis:")
    print(f"  Initial condition number: {ntk_history['condition_number'][0]:.2e}")
    print(f"  Final condition number:   {ntk_history['condition_number'][-1]:.2e}")
    print(f"  Initial effective rank:   {ntk_history['effective_rank'][0]:.2f}")
    print(f"  Final effective rank:     {ntk_history['effective_rank'][-1]:.2f}")
    print()
    print("Training Results:")
    print(f"  Initial loss: {losses[0]:.6e}")
    print(f"  Final loss:   {losses[-1]:.6e}")
    print(f"  L2 error:     {l2_error:.6e}")
    print(f"  Max error:    {max_error:.6e}")
    print()
    print("Key Insights:")
    print("  1. NTK condition number reflects training difficulty")
    print("  2. Large eigenvalue gaps indicate spectral bias")
    print("  3. Effective rank shows how many modes are actively learned")
    print("  4. NTK evolves during training (finite-width effects)")
    print("=" * 70)

    print()
    print("NTK analysis example completed successfully!")
    print(f"Results saved to: {output_dir}")

    return {
        "param_count": int(n_params),
        "final_loss": float(losses[-1]),
        "l2_error": float(l2_error),
        "max_error": float(max_error),
        "initial_condition_number": float(ntk_history["condition_number"][0]),
        "final_condition_number": float(ntk_history["condition_number"][-1]),
    }


# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
