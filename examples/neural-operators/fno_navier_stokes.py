# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # FNO on 2D Navier-Stokes Equations
#
# | Property      | Value                                    |
# |---------------|------------------------------------------|
# | Level         | Intermediate                             |
# | Runtime       | ~3 min (CPU) / ~30 sec (GPU)             |
# | Memory        | ~3 GB                                    |
# | Prerequisites | JAX, Flax NNX, FNO basics, CFD concepts  |
#
# ## Overview
#
# Train a Fourier Neural Operator (FNO) to learn the solution operator for the
# 2D incompressible Navier-Stokes equations:
#
#     du/dt + (u*nabla)u = -nabla(p)/rho + nu*laplacian(u)
#     div(u) = 0  (incompressibility)
#
# where u = (u, v) is the velocity field, p is pressure, rho is density, and nu is
# kinematic viscosity. The operator we learn maps the **initial velocity field** to
# the **velocity field at the final time**, with the flow parameterized by viscosity.
#
# This example demonstrates:
#
# - **FNO for CFD** — mapping the initial velocity field to the final-time solution
# - **Taylor-Green vortex** — analytically incompressible initial conditions
# - **Viscosity variation** — training across different viscosity regimes
# - **Velocity field prediction** — 2-channel input/output for (u, v) components
#
# ## Learning Goals
#
# 1. Understand FNO as an initial-condition -> final-state solution operator
# 2. Work with multi-channel velocity fields
# 3. Use the uniform Navier-Stokes ``PDELoaders`` contract
# 4. Evaluate final-time prediction accuracy (MSE / relative L2)
# 5. Visualize 2D velocity fields and vorticity

# %% [markdown]
# ## Imports and Setup

# %%
import time
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import optax
from flax import nnx


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.data.loaders import create_navier_stokes_loader
from opifex.neural.operators.fno.base import FourierNeuralOperator


# %% [markdown]
# ## Run the example
#
# All run logic (data loading, model creation, training, evaluation, plotting)
# lives in ``main`` so nothing heavy executes at import time.


# %%
def main() -> dict[str, float | int]:
    """Train an FNO on 2D Navier-Stokes data and return finite scalar metrics."""
    # --- Configuration ---
    resolution = 32  # Spatial grid resolution
    n_train = 100  # Training samples
    n_test = 30  # Test samples
    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-3
    modes = 12  # Fourier modes
    hidden_width = 32
    num_layers = 4
    viscosity_range = (0.001, 0.01)  # Kinematic viscosity range
    time_range = (0.0, 1.0)  # Time interval (operator maps IC -> final time)
    seed = 42

    output_dir = Path("docs/assets/examples/fno_navier_stokes")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Opifex Example: FNO on 2D Navier-Stokes Equations")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Viscosity range: {viscosity_range}")
    print(f"Time range: {time_range} (predict velocity at final time)")
    print(f"Training samples: {n_train}, Test samples: {n_test}")
    print(f"FNO config: modes={modes}, width={hidden_width}, layers={num_layers}")

    # --- Data Loading ---
    print()
    print("Generating Navier-Stokes data via datarax...")
    print("  (Using Taylor-Green vortex initial conditions)")

    n_samples = n_train + n_test
    loaders = create_navier_stokes_loader(
        n_samples=n_samples,
        batch_size=batch_size,
        resolution=resolution,
        viscosity_range=(0.001, 0.01),
        time_range=time_range,
        val_fraction=n_test / n_samples,
        seed=seed,
    )

    def _collect(pipeline):
        """Materialize a datarax pipeline into (inputs, outputs) numpy arrays."""
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    x_train, y_train = _collect(loaders.train)
    x_test, y_test = _collect(loaders.val)

    print(f"Training: X={x_train.shape}, Y={y_train.shape}")
    print(f"Test:     X={x_test.shape}, Y={y_test.shape}")
    print("  X = (batch, 2=[u,v], res, res) = initial velocity")
    print("  Y = (batch, 2=[u,v], res, res) = velocity at final time")

    # --- Model Creation ---
    print()
    print("Creating FNO model...")

    in_channels = 2  # (u, v) initial velocity components
    out_channels = 2  # (u, v) final-time velocity components

    model = FourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_width,
        modes=modes,
        num_layers=num_layers,
        rngs=nnx.Rngs(seed),
    )

    params = nnx.state(model, nnx.Param)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")

    # --- Training setup ---
    print()
    print("Setting up training...")

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    def loss_fn(model, x, y):
        """MSE loss for final-time velocity field prediction."""
        y_pred = model(x)  # (batch, 2, res, res)
        return jnp.mean((y_pred - y) ** 2)

    @nnx.jit
    def train_step(model, optimizer, x_batch, y_batch):
        """Single training step."""
        loss, grads = nnx.value_and_grad(loss_fn)(model, x_batch, y_batch)
        optimizer.update(model, grads)
        return loss

    # --- Training loop ---
    print("Starting training...")
    print()

    x_train_jnp = jnp.array(x_train)
    y_train_jnp = jnp.array(y_train)
    x_test_jnp = jnp.array(x_test)
    y_test_jnp = jnp.array(y_test)

    n_samples = x_train_jnp.shape[0]
    n_batches = n_samples // batch_size

    train_losses = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        perm = jax.random.permutation(jax.random.PRNGKey(epoch), n_samples)
        x_shuffled = x_train_jnp[perm]
        y_shuffled = y_train_jnp[perm]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            loss = train_step(model, optimizer, x_batch, y_batch)
            epoch_loss += float(loss)

        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{num_epochs}: Loss = {avg_loss:.6f}")

    training_time = time.time() - start_time
    final_loss = train_losses[-1]
    print()
    print(f"Training completed in {training_time:.1f}s")

    # --- Evaluation ---
    print()
    print("Running evaluation...")

    # Predict final-time velocity on test set
    predictions = model(x_test_jnp)  # (batch, 2, res, res)

    # Compute metrics
    test_mse = float(jnp.mean((predictions - y_test_jnp) ** 2))

    # Relative L2 error per sample (over both velocity components)
    pred_flat = predictions.reshape(predictions.shape[0], -1)
    true_flat = y_test_jnp.reshape(y_test_jnp.shape[0], -1)
    rel_l2 = jnp.linalg.norm(pred_flat - true_flat, axis=1) / jnp.linalg.norm(true_flat, axis=1)
    mean_rel_l2 = float(jnp.mean(rel_l2))

    print(f"Test MSE:         {test_mse:.6f}")
    print(f"Test Relative L2: {mean_rel_l2:.6f}")

    # Per-component final-time errors
    u_mse = float(jnp.mean((predictions[:, 0] - y_test_jnp[:, 0]) ** 2))
    v_mse = float(jnp.mean((predictions[:, 1] - y_test_jnp[:, 1]) ** 2))
    print()
    print("Per-component final-time MSE:")
    print(f"  u-MSE={u_mse:.6f}, v-MSE={v_mse:.6f}")

    # --- Visualization ---
    print()
    print("Generating visualizations...")

    predictions_np = np.asarray(predictions)

    # --- Final-time prediction comparison ---
    # For a few test samples, show input velocity magnitude alongside the predicted
    # and true final-time velocity magnitude and the absolute error.
    def _velocity_magnitude(field: np.ndarray) -> np.ndarray:
        """Speed |u| = sqrt(u^2 + v^2) from a (2, H, W) velocity field."""
        return np.sqrt(field[0] ** 2 + field[1] ** 2)

    n_show = min(3, x_test.shape[0])
    col_titles = ["Input |u| (t=0)", "Predicted |u| (final)", "True |u| (final)", "Abs. error"]
    fig, axes = plt.subplots(n_show, 4, figsize=(14, 3.5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "FNO 2D Navier-Stokes: Initial -> Final-Time Velocity (Opifex)",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )

    for row in range(n_show):
        input_mag = _velocity_magnitude(x_test[row])
        pred_mag = _velocity_magnitude(predictions_np[row])
        true_mag = _velocity_magnitude(y_test[row])
        error_mag = np.abs(pred_mag - true_mag)

        speed_max = float(max(input_mag.max(), pred_mag.max(), true_mag.max()))
        panels = [
            (input_mag, "viridis", 0.0, speed_max),
            (pred_mag, "viridis", 0.0, speed_max),
            (true_mag, "viridis", 0.0, speed_max),
            (error_mag, "magma", 0.0, float(error_mag.max()) or 1e-8),
        ]

        for col, (data, cmap, vmin, vmax) in enumerate(panels):
            im = axes[row, col].imshow(data.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            if row == 0:
                axes[row, col].set_title(col_titles[col])
            if col == 0:
                axes[row, col].set_ylabel(f"sample {row}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_dir / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions saved to {output_dir / 'predictions.png'}")

    # --- Training history ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("FNO Navier-Stokes Training", fontsize=14, fontweight="bold")

    axes[0].semilogy(range(1, num_epochs + 1), train_losses, "b-", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (log scale)")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    axes[1].hist(np.array(rel_l2), bins=15, alpha=0.7, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Relative L2 Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Test Error Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training analysis saved to {output_dir / 'training_analysis.png'}")

    print()
    print("=" * 70)
    print(f"FNO Navier-Stokes example completed in {training_time:.1f}s")
    print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return {
        "l2_relative_error": mean_rel_l2,
        "test_mse": test_mse,
        "final_loss": final_loss,
        "param_count": int(param_count),
    }


# %% [markdown]
# ## Results Summary
#
# The FNO successfully learns the Navier-Stokes solution operator, mapping the
# initial velocity field to the velocity field at the final time. Key observations:
#
# - **Multi-channel handling**: FNO naturally handles (u, v) velocity components
# - **Viscosity variation**: Model generalizes across different viscosity regimes
# - **Flow structure**: Predicted final-time fields preserve physical features
#
# ## Next Steps
#
# - Increase resolution and training data for better accuracy
# - Add physics-informed loss for PINO on Navier-Stokes
# - Predict at multiple final times by varying ``time_range``
# - Use spectral convergence analysis for solution quality
#
# ### Related Examples
#
# - [FNO on Darcy Flow](fno-darcy.md) — Elliptic PDE baseline
# - [FNO on Burgers Equation](fno-burgers.md) — 1D time-dependent PDE
# - [PINO on Burgers](pino-burgers.md) — Physics-informed operator learning
# - [Navier-Stokes PINN](../pinns/navier-stokes.md) — Physics-only approach

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
