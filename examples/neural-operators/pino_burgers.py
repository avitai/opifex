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
# # Physics-Informed Neural Operator (PINO) on Burgers Equation
#
# | Property      | Value                                    |
# |---------------|------------------------------------------|
# | Level         | Advanced                                 |
# | Runtime       | ~3 min (CPU) / ~1-2 min (GPU)           |
# | Memory        | ~2 GB                                    |
# | Prerequisites | JAX, Flax NNX, FNO, PDEs basics          |
#
# ## Overview
#
# Train a Physics-Informed Neural Operator (PINO) on the 1D viscous Burgers
# equation. PINO pairs the FNO architecture with a physics-informed loss so that
# the learned operator both fits the data *and* satisfies the governing PDE.
#
# The Burgers equation is
#
#     u_t + u * u_x = nu * u_xx ,
#
# where ``u`` is velocity, ``nu`` is viscosity, and subscripts denote partial
# derivatives. The spatial domain is the periodic interval ``[0, 1)``.
#
# ## The genuine PINO setup
#
# Following Li et al. (2021), *Physics-Informed Neural Operator for Learning
# Partial Differential Equations*, and the reference implementation in
# ``neuraloperator`` (``scripts/train_burgers_pino.py``), the operator maps the
# initial condition ``u(x, 0)`` — broadcast/repeated across the time axis — to
# the **full space-time solution** ``u(t, x)``, a 2D field over ``(time, space)``.
#
# Training minimises three terms (cf. ``neuralop.losses.equation_losses``
# ``BurgersEqnLoss`` + ``ICLoss``):
#
# - **data loss**: mean relative L2 between the predicted and the
#   ground-truth space-time trajectory,
# - **IC loss**: MSE between the predicted ``t = 0`` slice and the true initial
#   condition,
# - **equation loss**: the mean-squared Burgers PDE residual
#   ``u_t + u * u_x - nu * u_xx`` evaluated by finite differences over the whole
#   predicted field.
#
# The viscosity ``nu = 0.01`` is shared between the data-generating solver and
# the equation loss, so the supervised physics is self-consistent.
#
# This example demonstrates:
#
# - **2D FNO backbone** over the ``(time, space)`` grid for operator learning
# - **Physics loss** via a finite-difference Burgers residual
# - **Multi-objective training** balancing data, IC, and equation losses
# - **On-device data generation** with the pseudo-spectral ETDRK4 solver
#
# Equivalent to ``neuraloperator/scripts/train_burgers_pino.py``,
# reimplemented using Opifex APIs.
#
# ## Learning Goals
#
# 1. Understand the PINO operator: IC tiled over time -> full ``u(t, x)`` field
# 2. Implement the Burgers PDE residual with finite differences
# 3. Combine data + IC + equation losses with fixed weights
# 4. Generate self-consistent Burgers trajectories with the spectral solver

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

from opifex.core.metrics import per_sample_relative_l2, relative_l2_error
from opifex.data.sources.pde_generation import _burgers_ic
from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.physics.spectral.steppers import solve_burgers_spectral


# %% [markdown]
# ## Configuration
#
# The model maps the IC, tiled over ``NUM_TIME`` frames, to the full space-time
# field. ``VISCOSITY`` is used both by the solver and by the equation loss so the
# physics is self-consistent. The loss weights follow the reference ordering
# (data, IC, equation); fixed weights are used here for a clean, reproducible
# result (the reference uses a Relobralo adaptive aggregator).

# %%
NUM_SPACE = 128  # spatial resolution (nx)
NUM_TIME = 11  # time frames including t=0 (nt = num_snapshots + 1)
N_TRAIN = 1000
N_TEST = 200
BATCH_SIZE = 50
NUM_EPOCHS = 800
LEARNING_RATE = 1e-3
MODES = 16
HIDDEN_WIDTH = 32
NUM_LAYERS = 4
VISCOSITY = 0.01  # shared by the solver and the equation loss
DOMAIN_LENGTH = 1.0  # x in [0, 1), periodic
TIME_FINAL = 1.0  # t in [0, 1]
SOLVER_STEPS = 250  # ETDRK4 steps used to build the trajectories

# Loss weights (data, ic, equation), cf. neuralop training_loss = [equation, ic, l2].
DATA_WEIGHT = 1.0
IC_WEIGHT = 5.0
EQUATION_WEIGHT = 0.5

SEED = 42

# Grid spacings for the finite-difference equation loss.
DX = DOMAIN_LENGTH / NUM_SPACE  # periodic, so dx = L / nx
DT = TIME_FINAL / (NUM_TIME - 1)

OUTPUT_DIR = Path("docs/assets/examples/pino_burgers")


# %% [markdown]
# ## Data Generation
#
# Initial conditions are opifex's canonical periodic spectral Gaussian-random-field
# Burgers ICs (``opifex.data.sources.pde_generation._burgers_ic`` — the same
# generator behind ``create_burgers_loader``, reused here rather than reimplemented).
# Each IC is evolved to the full trajectory ``u(t, x)`` with opifex's pseudo-spectral
# ETDRK4 Burgers solver, vmapped over the batch on device.
# ``solve_burgers_spectral`` returns ``(num_snapshots + 1, nx)`` real snapshots
# including the initial condition, i.e. exactly the ``nt = NUM_TIME`` time frames.


# %%
def generate_trajectories(n_samples: int, seed: int) -> jax.Array:
    """Generate ``n_samples`` Burgers space-time trajectories ``u(t, x)``.

    Each sample draws a GRF initial condition and integrates the viscous Burgers
    equation with the pseudo-spectral ETDRK4 solver. The whole batch is generated
    with a single ``jit(vmap(...))`` call on device.

    Args:
        n_samples: Number of trajectories to generate.
        seed: Base seed; sample ``i`` uses ``PRNGKey(seed + i)``.

    Returns:
        Trajectories of shape ``(n_samples, NUM_TIME, NUM_SPACE)`` including the
        initial condition as the first time frame.
    """

    def per_sample(key: jax.Array) -> jax.Array:
        ic = _burgers_ic(key, NUM_SPACE)
        return solve_burgers_spectral(
            ic,
            VISCOSITY,
            domain_extent=DOMAIN_LENGTH,
            time_final=TIME_FINAL,
            num_steps=SOLVER_STEPS,
            num_snapshots=NUM_TIME - 1,
        )

    keys = jax.vmap(jax.random.PRNGKey)(seed + jnp.arange(n_samples))
    return jax.jit(jax.vmap(per_sample))(keys)


# %% [markdown]
# ## Physics Loss: Burgers Equation Residual
#
# The residual is computed by finite differences over the predicted field of
# shape ``(batch, nt, nx)``: a forward difference in time, central differences in
# space with periodic wrap-around. A perfect solution has zero residual. This
# mirrors ``neuralop.losses.equation_losses.BurgersEqnLoss`` (``method="fdm"``),
# which takes ``MSE(u_t, -u*u_x + nu*u_xx)``.


# %%
def compute_burgers_residual(u: jax.Array, dx: float, dt: float, nu: float) -> jax.Array:
    """Compute the Burgers PDE residual ``u_t + u*u_x - nu*u_xx`` by finite differences.

    Time uses a forward difference; space uses periodic central differences. The
    fields are aligned at the earlier time level so every returned residual point
    corresponds to a real grid location.

    Args:
        u: Predicted field of shape ``(batch, nt, nx)``.
        dx: Spatial step size.
        dt: Temporal step size.
        nu: Viscosity coefficient.

    Returns:
        Residual of shape ``(batch, nt - 1, nx)``.
    """
    # Forward difference in time -> (batch, nt-1, nx).
    u_t = (u[:, 1:, :] - u[:, :-1, :]) / dt

    # Evaluate spatial derivatives at the earlier time level so shapes align.
    u_level = u[:, :-1, :]
    u_right = jnp.roll(u_level, shift=-1, axis=-1)
    u_left = jnp.roll(u_level, shift=1, axis=-1)
    u_x = (u_right - u_left) / (2.0 * dx)
    u_xx = (u_right - 2.0 * u_level + u_left) / (dx**2)

    return u_t + u_level * u_x - nu * u_xx


def equation_loss(pred: jax.Array, dx: float, dt: float, nu: float) -> jax.Array:
    """Mean-squared Burgers PDE residual over the predicted field."""
    residual = compute_burgers_residual(pred, dx, dt, nu)
    return jnp.mean(residual**2)


def pino_loss_fn(
    model: FourierNeuralOperator,
    ic_tiled: jax.Array,
    trajectory: jax.Array,
    ic: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Combined data + IC + equation loss for PINO training.

    The model maps the tiled initial condition ``(batch, 1, nt, nx)`` to the full
    predicted space-time field. The data term is the relative L2 against the
    ground-truth trajectory, the IC term anchors the predicted ``t=0`` slice to
    the true initial condition, and the equation term enforces the Burgers PDE
    residual over the whole field (cf. neuralop ``BurgersEqnLoss`` + ``ICLoss``).

    Args:
        model: FNO backbone.
        ic_tiled: Tiled initial condition, shape ``(batch, 1, nt, nx)``.
        trajectory: Ground-truth field, shape ``(batch, nt, nx)``.
        ic: True initial condition, shape ``(batch, nx)``.

    Returns:
        Total loss and a dict of the three component losses.
    """
    pred = model(ic_tiled)[:, 0]  # (batch, nt, nx)
    data_loss = relative_l2_error(pred, trajectory)
    ic_loss = jnp.mean((pred[:, 0, :] - ic) ** 2)
    eqn_loss = equation_loss(pred, DX, DT, VISCOSITY)
    total = DATA_WEIGHT * data_loss + IC_WEIGHT * ic_loss + EQUATION_WEIGHT * eqn_loss
    return total, {"data": data_loss, "ic": ic_loss, "equation": eqn_loss}


# %% [markdown]
# ## Run the Example
#
# ``main()`` generates the Burgers trajectories, builds the 2D FNO backbone,
# trains with the combined data + IC + equation loss, evaluates on the held-out
# test trajectories, saves figures, and returns a small dict of finite metrics.


# %%
def main() -> dict[str, float | int]:
    """Train and evaluate a PINO on the 1D Burgers equation."""
    print("=" * 70)
    print("Opifex Example: PINO on 1D Burgers Equation")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Grid: nx={NUM_SPACE}, nt={NUM_TIME}, viscosity={VISCOSITY}")
    print(f"Trajectories: train={N_TRAIN}, test={N_TEST}")
    print(f"FNO config: modes={MODES}, width={HIDDEN_WIDTH}, layers={NUM_LAYERS}")
    print(f"Loss weights: data={DATA_WEIGHT}, ic={IC_WEIGHT}, equation={EQUATION_WEIGHT}")
    print(f"Spacings: dx={DX:.5f}, dt={DT:.5f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data generation ---
    print()
    print("Generating Burgers space-time trajectories (jit+vmap spectral solver)...")
    train_traj = np.asarray(generate_trajectories(N_TRAIN, SEED))
    test_traj = np.asarray(generate_trajectories(N_TEST, SEED + N_TRAIN))
    print(f"Train trajectories: {train_traj.shape}")
    print(f"Test trajectories:  {test_traj.shape}")

    train_traj_jnp = jnp.asarray(train_traj)
    test_traj_jnp = jnp.asarray(test_traj)

    def tile_ic(trajectory: jax.Array) -> jax.Array:
        """Tile the t=0 frame over the time axis -> ``(batch, 1, nt, nx)``."""
        ic = trajectory[:, 0:1, :]  # (batch, 1, nx)
        tiled = jnp.repeat(ic, NUM_TIME, axis=1)  # (batch, nt, nx)
        return tiled[:, None, :, :]  # (batch, 1, nt, nx)

    train_input = tile_ic(train_traj_jnp)
    test_input = tile_ic(test_traj_jnp)
    train_ic = train_traj_jnp[:, 0, :]
    test_ic = test_traj_jnp[:, 0, :]

    # --- Model creation ---
    print()
    print("Creating PINO model (2D FNO backbone over (time, space))...")
    model = FourierNeuralOperator(
        in_channels=1,
        out_channels=1,
        hidden_channels=HIDDEN_WIDTH,
        modes=MODES,
        num_layers=NUM_LAYERS,
        spatial_dims=2,
        positional_embedding=True,
        domain_padding=0.0,  # space is periodic; time axis is short, no padding
        rngs=nnx.Rngs(SEED),
    )
    params = nnx.state(model, nnx.Param)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")

    # --- Training setup ---
    # The schedule is stepped once per optimizer update (one mini-batch), so the
    # cosine decay must run over the total number of updates, not epochs.
    n_batches = N_TRAIN // BATCH_SIZE
    schedule = optax.cosine_decay_schedule(LEARNING_RATE, NUM_EPOCHS * n_batches)
    optimizer = nnx.Optimizer(model, optax.adamw(schedule), wrt=nnx.Param)

    @nnx.jit
    def train_step(
        model: FourierNeuralOperator,
        optimizer: nnx.Optimizer,
        ic_tiled: jax.Array,
        trajectory: jax.Array,
        ic: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Single PINO training step (data + IC + equation loss)."""

        def loss_fn(
            model: FourierNeuralOperator,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            return pino_loss_fn(model, ic_tiled, trajectory, ic)

        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return loss, aux

    # --- Training loop ---
    print()
    print("Starting PINO training...")
    print(f"Optimizer: AdamW (cosine-decayed lr from {LEARNING_RATE})")

    history: dict[str, list[float]] = {"total": [], "data": [], "ic": [], "equation": []}

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), N_TRAIN)
        epoch_totals = {"total": 0.0, "data": 0.0, "ic": 0.0, "equation": 0.0}

        for i in range(n_batches):
            idx = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            loss, aux = train_step(
                model,
                optimizer,
                train_input[idx],
                train_traj_jnp[idx],
                train_ic[idx],
            )
            epoch_totals["total"] += float(loss)
            epoch_totals["data"] += float(aux["data"])
            epoch_totals["ic"] += float(aux["ic"])
            epoch_totals["equation"] += float(aux["equation"])

        for key, value in epoch_totals.items():
            history[key].append(value / n_batches)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:4d}/{NUM_EPOCHS}: "
                f"Total={history['total'][-1]:.6f}, "
                f"Data={history['data'][-1]:.6f}, "
                f"IC={history['ic'][-1]:.6e}, "
                f"Equation={history['equation'][-1]:.6e}"
            )

    training_time = time.time() - start_time
    print()
    print(f"Training completed in {training_time:.1f}s")

    # --- Evaluation ---
    print()
    print("Running evaluation...")
    predictions = model(test_input)[:, 0]  # (N, nt, nx)

    test_rel_l2 = float(relative_l2_error(predictions, test_traj_jnp))
    test_ic_error = float(jnp.sqrt(jnp.mean((predictions[:, 0, :] - test_ic) ** 2)))
    test_residual = float(equation_loss(predictions, DX, DT, VISCOSITY))

    print(f"Test relative L2 (full trajectory): {test_rel_l2:.6f}")
    print(f"Test IC RMS error (t=0 slice):      {test_ic_error:.6e}")
    print(f"Test mean PDE residual (MSE):        {test_residual:.6e}")

    per_sample_rel_l2 = np.asarray(per_sample_relative_l2(predictions, test_traj_jnp))

    # --- Visualization: sample space-time fields ---
    print()
    print("Generating visualizations...")
    n_vis = 3
    fig, axes = plt.subplots(n_vis, 3, figsize=(13, 3.2 * n_vis))
    fig.suptitle("PINO 1D Burgers: Space-Time Solution (Opifex)", fontsize=14, fontweight="bold")
    extent = (0.0, DOMAIN_LENGTH, TIME_FINAL, 0.0)
    for i in range(n_vis):
        truth = test_traj[i]
        pred = np.asarray(predictions[i])
        vmin, vmax = float(truth.min()), float(truth.max())
        axes[i, 0].imshow(truth, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
        axes[i, 0].set_ylabel(f"Sample {i}\ntime t")
        axes[i, 1].imshow(pred, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap="viridis")
        err = axes[i, 2].imshow(np.abs(pred - truth), aspect="auto", extent=extent, cmap="magma")
        fig.colorbar(err, ax=axes[i, 2], fraction=0.046)
        if i == 0:
            axes[i, 0].set_title("Ground truth u(t, x)")
            axes[i, 1].set_title("PINO prediction")
            axes[i, 2].set_title("Absolute error")
        for j in range(3):
            axes[i, j].set_xlabel("space x")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions saved to {OUTPUT_DIR / 'predictions.png'}")

    # --- Visualization: training history + error distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("PINO Training Dynamics", fontsize=14, fontweight="bold")

    epochs_arr = np.arange(1, NUM_EPOCHS + 1)
    axes[0].semilogy(epochs_arr, history["total"], "k-", linewidth=2, label="Total")
    axes[0].semilogy(epochs_arr, history["data"], "b--", linewidth=1.5, label="Data (rel L2)")
    axes[0].semilogy(epochs_arr, history["ic"], "g--", linewidth=1.5, label="IC (MSE)")
    axes[0].semilogy(epochs_arr, history["equation"], "r--", linewidth=1.5, label="Equation (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Training Loss Components")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(per_sample_rel_l2, bins=25, alpha=0.7, color="steelblue", edgecolor="black")
    axes[1].axvline(test_rel_l2, color="red", linestyle="--", label=f"mean = {test_rel_l2:.4f}")
    axes[1].set_xlabel("Relative L2 Error (full trajectory)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Test Error Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training analysis saved to {OUTPUT_DIR / 'training_analysis.png'}")

    print()
    print("=" * 70)
    print(f"PINO Burgers example completed in {training_time:.1f}s")
    print(f"Test relative L2: {test_rel_l2:.6f}, IC error: {test_ic_error:.2e}")
    print(f"Test mean PDE residual: {test_residual:.2e}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return {
        "test_rel_l2": test_rel_l2,
        "test_ic_error": test_ic_error,
        "test_pde_residual": test_residual,
        "final_total_loss": history["total"][-1],
        "num_parameters": int(param_count),
        "training_time_s": training_time,
    }


# %% [markdown]
# ## Results Summary
#
# The PINO learns the full space-time Burgers solution operator from the tiled
# initial condition. The data term drives accuracy, the IC term pins the
# ``t = 0`` slice to the true initial condition, and the equation term keeps the
# predicted field consistent with the Burgers PDE — the mean PDE residual is
# reported on the held-out test set.
#
# ## Next Steps
#
# - Vary the loss weights (data / IC / equation) and study the trade-off
# - Swap fixed weights for an adaptive aggregator (SoftAdapt, ReLoBRaLo)
# - Increase ``NUM_TIME`` for a denser temporal grid
# - Use spectral differentiation instead of finite differences for the residual
# - Extend the operator to varying viscosity (add ``nu`` as an input channel)
#
# ### Related Examples
#
# - [FNO on Burgers Equation](fno-burgers.md) — Data-only FNO baseline
# - [FNO on Darcy Flow](fno-darcy.md) — 2D elliptic PDE
# - [Burgers PINN](../pinns/burgers.md) — Physics-only neural network
# - [TFNO on Darcy Flow](tfno-darcy.md) — Tensorized FNO with compression

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
