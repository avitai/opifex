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
"""
# FNO on Burgers Equation

| Property      | Value                                    |
|---------------|------------------------------------------|
| Level         | Intermediate                             |
| Runtime       | ~3 min (CPU)                             |
| Memory        | ~1 GB                                    |
| Prerequisites | JAX, Flax NNX, Neural Operators basics   |

## Overview

Train a Fourier Neural Operator (FNO) on the 1D Burgers equation, a nonlinear
PDE that develops shocks and is a standard benchmark for operator learning.
Given an initial condition u(x, 0), the FNO learns the solution operator that
maps it to the solution u(x, T) at the final time.

This example demonstrates:

- **1D FNO** operating on `(batch, channels, resolution)` tensors
- **Canonical Burgers data** from `create_burgers_loader`: Gaussian-random-field
  initial conditions evolved by the pseudo-spectral ETDRK4 solver
- **Operator learning** mapping the initial condition to the final-time solution
- **Trainer.fit()** for end-to-end training with validation

Equivalent to `neuraloperator/examples/` Burgers examples,
reimplemented using Opifex APIs.

## Learning Goals

1. Load canonical 1D Burgers data with `create_burgers_loader`
2. Configure `FourierNeuralOperator` for 1D spatial data
3. Map an initial condition to the final-time solution
4. Evaluate with the relative L2 error
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import time
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from flax import nnx


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.core.evaluation import predict_in_batches
from opifex.core.metrics import per_sample_relative_l2
from opifex.core.normalization import GaussianNormalizer
from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.config import LossConfig, OptimizationConfig
from opifex.data.loaders import create_burgers_loader
from opifex.neural.operators.fno.base import FourierNeuralOperator


# %% [markdown]
"""
## Configuration

The Burgers equation u_t + u * u_x = nu * u_xx develops shocks whose
steepness depends on viscosity nu. We sample viscosity from a range so
the FNO learns to generalize across different shock profiles.
"""

# %% [markdown]
"""
## Data Loading via datarax

`create_burgers_loader` generates the canonical FNO-benchmark Burgers data:
initial conditions are sampled from a Gaussian random field on the periodic
domain and evolved to the final time with the pseudo-spectral ETDRK4 solver
(`opifex.physics.spectral`). Each sample maps u(x, 0) to u(x, T); the loader
serves it through datarax pipelines.
"""

# %% [markdown]
"""
## Model Creation

For 1D Burgers, the FNO maps 1 input channel (the initial condition) to 1
output channel (the final-time solution), with the grid coordinate appended as a
second input channel. It handles 1D spatial data given
`(batch, channels, resolution)` tensors.
"""

# %% [markdown]
"""
## Training with Opifex Trainer

The `Trainer.fit()` method handles the full training loop: JIT compilation,
batching, validation, and progress logging. The loss is the relative L2 error
between the predicted and true final-time solution.
"""

# %% [markdown]
"""
## Evaluation

Evaluate the trained FNO on the test set with per-sample and per-time-step
metrics.
"""

# %% [markdown]
"""
## Visualization

Create visualizations showing sample predictions and error analysis.
"""

# %% [markdown]
"""
## Run the example
"""


# %%
def main() -> dict[str, float | int]:
    """Train a 1D FNO on Burgers data and return finite scalar metrics."""
    # Canonical FNO-paper Burgers benchmark (Li et al. 2021; ref:
    # ../deeponet-fno/src/burgers/fourier_1d.py): map the initial condition to the
    # solution at a single final time, fixed viscosity, with the grid coordinate
    # appended as a second input channel (a(x), x). modes/width/lr/epochs/loss
    # match the reference configuration.
    resolution = 128
    time_steps = 1
    n_train = 1000
    n_test = 200
    batch_size = 20
    num_epochs = 750
    learning_rate = 1e-3
    weight_decay = 1e-4
    modes = 16
    hidden_width = 128
    num_layers = 4
    viscosity_range = (0.1, 0.1)
    seed = 42
    steps_per_epoch = max(1, n_train // batch_size)
    lr_transition_steps = 60 * steps_per_epoch
    lr_decay_rate = 0.5

    output_dir = Path("docs/assets/examples/fno_burgers")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Opifex Example: FNO on 1D Burgers Equation")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Resolution: {resolution}")
    print(f"Time steps: {time_steps}")
    print(f"Viscosity range: {viscosity_range}")
    print(f"Training samples: {n_train}, Test samples: {n_test}")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"FNO config: modes={modes}, width={hidden_width}, layers={num_layers}")

    print()
    print("Generating 1D Burgers data (jit+vmap) and serving via datarax...")
    n_samples = n_train + n_test
    loaders = create_burgers_loader(
        n_samples=n_samples,
        batch_size=batch_size,
        resolution=resolution,
        viscosity_range=viscosity_range,
        val_fraction=n_test / n_samples,
        seed=seed,
    )

    # Collect the datarax pipelines into arrays. Batches are already
    # channels-first: input u(x,0) and output u(x,T) are both (N, 1, resolution).
    def _collect(pipeline) -> tuple[np.ndarray, np.ndarray]:
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    x_train, y_train = _collect(loaders.train)
    x_test, y_test = _collect(loaders.val)

    # Gaussian normalization (the standard operator-learning recipe); evaluation
    # un-normalizes predictions back to physical units before measuring error.
    x_norm = GaussianNormalizer.fit(x_train)
    y_norm = GaussianNormalizer.fit(y_train)
    x_train_n = x_norm.normalize(x_train)
    y_train_n = y_norm.normalize(y_train)
    x_test_n = x_norm.normalize(x_test)
    y_test_n = y_norm.normalize(y_test)

    print(f"Training data: X={x_train.shape}, Y={y_train.shape}")
    print(f"Test data:     X={x_test.shape}, Y={y_test.shape}")
    print(f"Input:  initial condition u(x,0)  -> {x_train.shape[1]} channel(s)")
    print(f"Output: solution u(x,t_1..t_T)    -> {y_train.shape[1]} channel(s)")

    print()
    print("Creating FNO model...")
    model = FourierNeuralOperator(
        in_channels=1,
        out_channels=time_steps,
        hidden_channels=hidden_width,
        modes=modes,
        num_layers=num_layers,
        spatial_dims=1,
        positional_embedding=True,  # append grid coordinate -> input is (a(x), x)
        rngs=nnx.Rngs(seed),
    )

    params = nnx.state(model, nnx.Param)
    param_count = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print("Model: FourierNeuralOperator (1D)")
    print("  Input channels: 1 (initial condition)")
    print(f"  Output channels: {time_steps} (solution at each time step)")
    print(f"  Fourier modes: {modes}, Hidden width: {hidden_width}, Layers: {num_layers}")
    print(f"  Total parameters: {param_count:,}")

    print()
    print("Setting up Trainer...")
    config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_frequency=20,
        verbose=True,
        loss_config=LossConfig(loss_type="relative_l2"),
        optimization_config=OptimizationConfig(
            optimizer="adamw",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            schedule_type="exponential",
            transition_steps=lr_transition_steps,
            decay_rate=lr_decay_rate,
        ),
    )

    trainer = Trainer(
        model=model,
        config=config,
        rngs=nnx.Rngs(seed),
    )

    print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print("Loss: relative L2 (the standard operator-learning objective)")
    print()
    print("Starting training...")
    start_time = time.time()

    trained_model, metrics = trainer.fit(
        train_data=(jnp.array(x_train_n), jnp.array(y_train_n)),
        val_data=(jnp.array(x_test_n), jnp.array(y_test_n)),
    )

    training_time = time.time() - start_time
    final_loss = float(metrics.get("final_train_loss", float("nan")))
    print(f"Training completed in {training_time:.1f}s")
    print(f"Final train loss: {final_loss}")
    print(f"Final val loss:   {metrics.get('final_val_loss', 'N/A')}")

    print()
    print("Running evaluation...")
    y_test_jnp = jnp.array(y_test)

    # Un-normalize predictions back to physical units before measuring error.
    predictions = y_norm.denormalize(predict_in_batches(trained_model, jnp.array(x_test_n)))

    # Overall metrics
    test_mse = float(jnp.mean((predictions - y_test_jnp) ** 2))

    per_sample_rel_l2 = per_sample_relative_l2(predictions, y_test_jnp)
    mean_rel_l2 = float(jnp.mean(per_sample_rel_l2))

    # Per-time-step errors
    per_step_mse = []
    for t in range(time_steps):
        step_mse = float(jnp.mean((predictions[:, t, :] - y_test_jnp[:, t, :]) ** 2))
        per_step_mse.append(step_mse)

    print(f"Test MSE:         {test_mse:.6f}")
    print(f"Test Relative L2: {mean_rel_l2:.6f}")
    print(f"Min Relative L2:  {float(jnp.min(per_sample_rel_l2)):.6f}")
    print(f"Max Relative L2:  {float(jnp.max(per_sample_rel_l2)):.6f}")
    print()
    print("Per-time-step MSE:")
    for t, mse in enumerate(per_step_mse):
        print(f"  t_{t + 1}: {mse:.6f}")

    print()
    print("Generating visualizations...")

    x_grid = np.linspace(-1, 1, resolution)

    # --- Sample predictions ---
    n_vis = min(4, len(x_test))
    fig, axes = plt.subplots(n_vis, time_steps + 1, figsize=(3.5 * (time_steps + 1), 3 * n_vis))
    fig.suptitle("FNO 1D Burgers Predictions (Opifex)", fontsize=14, fontweight="bold")

    if n_vis == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_vis):
        # Initial condition
        axes[i, 0].plot(x_grid, x_test[i, 0], "k-", linewidth=1.5, label="u(x,0)")
        axes[i, 0].set_title("Initial Condition" if i == 0 else "")
        axes[i, 0].set_ylabel(f"Sample {i}")
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].legend(fontsize=8)

        # Predicted vs ground truth at each time step
        for t in range(time_steps):
            axes[i, t + 1].plot(x_grid, y_test[i, t], "b-", linewidth=1.5, alpha=0.8, label="Truth")
            axes[i, t + 1].plot(
                x_grid,
                np.array(predictions[i, t]),
                "r--",
                linewidth=1.5,
                alpha=0.8,
                label="FNO",
            )
            if i == 0:
                axes[i, t + 1].set_title(f"t = t_{t + 1}")
                axes[i, t + 1].legend(fontsize=8)
            axes[i, t + 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample predictions saved to {output_dir / 'predictions.png'}")

    # --- Error analysis ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FNO Burgers Error Analysis", fontsize=14, fontweight="bold")

    per_sample_errors = np.array(per_sample_rel_l2)

    axes[0].hist(per_sample_errors, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    axes[0].set_title("Relative L2 Error Distribution")
    axes[0].set_xlabel("Relative L2 Error")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(per_sample_errors, "o-", alpha=0.7, color="coral", markersize=3)
    axes[1].set_title("Relative L2 Error per Sample")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Relative L2 Error")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(
        range(1, time_steps + 1),
        per_step_mse,
        color="mediumpurple",
        edgecolor="black",
        alpha=0.7,
    )
    axes[2].set_title("MSE per Time Step")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("MSE")
    axes[2].set_xticks(range(1, time_steps + 1))
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Error analysis saved to {output_dir / 'error_analysis.png'}")

    print()
    print("=" * 70)
    print(f"FNO Burgers example completed in {training_time:.1f}s")
    print(f"Test MSE: {test_mse:.6f}, Relative L2: {mean_rel_l2:.6f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return {
        "l2_relative_error": mean_rel_l2,
        "test_mse": test_mse,
        "final_loss": final_loss,
        "param_count": param_count,
    }


# %% [markdown]
"""
## Results Summary

After running this example you should observe:
- Decreasing training and validation loss over 15 epochs
- Reasonable L2 relative error on test Burgers solutions
- Prediction quality degrades slightly for later time steps (error accumulation)
- Shock structures in Burgers solutions are captured by the FNO

## Next Steps

- Increase resolution and training epochs for sharper shock resolution
- Try 2D Burgers with `dimension="2d"` and `GridEmbedding2D`
- Compare against PINO (physics-informed neural operator) which adds PDE loss
- Experiment with different viscosity ranges to test generalization
- Try `TensorizedFourierNeuralOperator` for parameter-efficient training

### Related Examples

- [FNO on Darcy Flow](fno-darcy.md) — 2D elliptic PDE with grid embedding
- [PINO on Navier-Stokes](pino-navier-stokes.md) — Physics-informed operator
- [Burgers PINN](../pinns/burgers.md) — Solve Burgers with physics-informed neural networks
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
