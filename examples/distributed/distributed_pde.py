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
# Distributed Data-Parallel Operator Learning

| Property      | Value                                        |
|---------------|----------------------------------------------|
| Level         | Intermediate                                 |
| Runtime       | ~1-2 min (1 GPU); faster per-step on 2+ GPUs |
| Memory        | ~2 GB                                        |
| Prerequisites | `source activate.sh`                         |

## Overview

Train a **Fourier Neural Operator (FNO)** on **Darcy flow** — a 2D elliptic PDE
mapping a permeability coefficient field to the pressure solution — using
**SPMD data-parallel training** across all available JAX devices.

The point of this example is that going distributed is a *zero-code-change*
operation in Opifex. You take the exact same FNO + `Trainer` recipe used in the
[FNO on Darcy Flow](../neural-operators/fno-darcy.md) example and add **one
argument**: `distributed_config` on `TrainingConfig`. The `Trainer` then builds
the JAX device mesh and shards every mini-batch across the `"data"` axis behind
the scenes.

This example demonstrates:
- **DistributedConfig**: declarative device-mesh topology configuration
- **Trainer integration**: zero-code-change data-parallel training
- **Automatic batch sharding**: batches are partitioned across the `"data"` axis
- **A real PDE metric**: physical-space relative L2 error, not a regression loss

**How it works:**
1. `DistributedConfig` describes the device mesh shape and axis names.
2. `Trainer.__init__` creates a `DistributedManager` from this config.
3. `Trainer.fit` shards each mini-batch across the `"data"` mesh axis before
   feeding it to the JIT-compiled training step.

It runs correctly on a single GPU (`mesh_shape=(1,)`) and scales to more devices
by simply reporting a larger `jax.device_count()`.
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import warnings


warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.core.evaluation import predict_in_batches
from opifex.core.metrics import per_sample_relative_l2
from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.config import LossConfig, OptimizationConfig
from opifex.data.loaders import create_darcy_loader
from opifex.distributed.config import DistributedConfig
from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.base import FourierNeuralOperator


# %% [markdown]
"""
## Step 1: Define the Model

A grid-embedded FNO — exactly the architecture used in the standalone
[FNO on Darcy Flow](../neural-operators/fno-darcy.md) example. `GridEmbedding2D`
injects spatial coordinates as extra input channels, and
`FourierNeuralOperator` performs the spectral operator learning. The distributed
machinery is completely orthogonal to the model definition.
"""


# %%
class FNOWithGrid(nnx.Module):
    """FNO with a built-in grid embedding for positional encoding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        hidden_channels: int,
        num_layers: int,
        grid_boundaries: list[list[float]],
        *,
        domain_padding: float,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the grid embedding and the underlying FNO.

        Args:
            in_channels: Number of physical input channels (before the grid).
            out_channels: Number of output channels.
            modes: Number of Fourier modes per spatial dimension.
            hidden_channels: Number of FNO hidden channels.
            num_layers: Number of spectral layers.
            grid_boundaries: Per-axis ``[min, max]`` grid extents.
            domain_padding: Fraction of each spatial axis to pad
                (resolution-invariant) to reduce the Gibbs phenomenon for the
                non-periodic Darcy problem.
            rngs: Random number generators.
        """
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=in_channels,
            grid_boundaries=grid_boundaries,
        )
        self.fno = FourierNeuralOperator(
            in_channels=self.grid_embedding.out_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            domain_padding=domain_padding,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: grid embedding -> FNO.

        Args:
            x: Input of shape ``(batch, channels, H, W)``.

        Returns:
            Output of shape ``(batch, out_channels, H, W)``.
        """
        # (batch, channels, H, W) -> (batch, H, W, channels) for embedding
        x_hwc = jnp.moveaxis(x, 1, -1)
        x_embedded = self.grid_embedding(x_hwc)
        # (batch, H, W, channels) -> (batch, channels, H, W) for FNO
        x_chw = jnp.moveaxis(x_embedded, -1, 1)
        return self.fno(x_chw)


# %% [markdown]
"""
## Step 2: Configure Distributed Training

`DistributedConfig` is a frozen dataclass describing the mesh topology. We build
it from `jax.device_count()` so the same script runs on 1 GPU
(`mesh_shape=(1,)`) and scales to N GPUs (`mesh_shape=(N,)`). The `"data"` axis
name and `strategy="data"` select pure data parallelism.
"""

# %% [markdown]
"""
## Step 3: Create Trainer with Distributed Config

The **only** change compared to single-device training is passing
`distributed_config` into `TrainingConfig`. Everything else — the FNO model,
the relative-L2 loss, the AdamW optimizer, JIT compilation, and gradient
computation — is identical to the single-device example.
"""

# %% [markdown]
"""
## Step 4: Load Darcy Data, Train, and Evaluate

We load real Darcy flow data via `create_darcy_loader` (permeability ->
pressure), apply Gaussian normalization (fit on train, applied to all splits),
train with `Trainer.fit`, then un-normalize predictions back to physical
pressure units and report the **relative L2 error** — the standard
operator-learning quality metric.
"""

# %% [markdown]
"""
## Summary

This example showed how to enable data-parallel training for operator learning
with effectively one extra argument:

1. Create a `DistributedConfig` describing the mesh topology
2. Pass it to `TrainingConfig(distributed_config=...)`
3. Call `trainer.fit()` as usual

The `Trainer` automatically:
- Creates a `DistributedManager` to manage the JAX device mesh
- Shards each mini-batch across the `"data"` axis before the training step
- JIT-compiles the training step via `@nnx.jit`

The reported test relative L2 confirms this is a genuine PDE solve, not a toy
regression — the same accuracy you would get single-device, now data-parallel.

## Next Steps

- Explore FSDP by setting `strategy="fsdp"` with a 2D mesh
- Add model-parallel sharding with `nnx.with_partitioning()`
- Compare per-step throughput as you add devices
- See the [distributed module tests](../../tests/distributed/) for more patterns
"""


# %%
def main() -> dict[str, float | int]:
    """Run distributed data-parallel FNO training on Darcy flow.

    Returns:
        Finite scalar metrics: device count, parameter count, final train loss,
        and the physical-space test relative L2 error.
    """
    # --- Configuration (small but real Darcy FNO) ---
    resolution = 32
    n_train = 1000
    n_test = 100
    batch_size = 32
    num_epochs = 100
    learning_rate = 5e-3
    weight_decay = 1e-4
    modes = 12
    hidden_width = 32
    num_layers = 4
    domain_padding = 0.25  # resolution-invariant fraction (Gibbs padding)
    seed = 42

    steps_per_epoch = n_train // batch_size
    lr_decay_epochs = 30
    lr_transition_steps = lr_decay_epochs * steps_per_epoch
    lr_decay_rate = 0.5

    num_devices = jax.device_count()
    print("=" * 70)
    print("Distributed Data-Parallel Operator Learning (FNO on Darcy Flow)")
    print("=" * 70)
    print(f"JAX backend:  {jax.default_backend()}")
    print(f"JAX devices:  {jax.devices()}")
    print(f"Devices:      {num_devices}")
    print(f"Resolution:   {resolution}x{resolution}")
    print(f"Train/Test:   {n_train} / {n_test}")
    print(f"FNO config:   modes={modes}, width={hidden_width}, layers={num_layers}")

    # --- Distributed config: built from the live device count ---
    distributed_config = DistributedConfig(
        mesh_shape=(num_devices,),
        mesh_axis_names=("data",),
        strategy="data",
    )
    print()
    print("Distributed mesh configuration:")
    print(f"  Mesh shape:  {distributed_config.mesh_shape}")
    print(f"  Axis names:  {distributed_config.mesh_axis_names}")
    print(f"  Strategy:    {distributed_config.strategy}")
    print(f"  Num devices: {num_devices}")
    print("  (The ONLY change from single-device training is passing this config.)")

    # --- Data loading via datarax ---
    print()
    print("Loading Darcy flow data (permeability -> pressure)...")
    n_samples = n_train + n_test
    loaders = create_darcy_loader(
        n_samples=n_samples,
        batch_size=batch_size,
        resolution=resolution,
        val_fraction=n_test / n_samples,
        seed=seed,
    )

    def _collect(pipeline) -> tuple[np.ndarray, np.ndarray]:
        inputs, outputs = [], []
        for batch in pipeline:
            inputs.append(np.asarray(batch["input"]))
            outputs.append(np.asarray(batch["output"]))
        return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)

    x_train, y_train = _collect(loaders.train)
    x_test, y_test = _collect(loaders.val)
    print(f"Training data: X={x_train.shape}, Y={y_train.shape}")
    print(f"Test data:     X={x_test.shape}, Y={y_test.shape}")

    # --- Normalization (fit on train, applied to all splits) ---
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()
    x_train_n = (x_train - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    x_test_n = (x_test - x_mean) / x_std

    # --- Model ---
    print()
    print("Creating FNO model with grid embedding...")
    model = FNOWithGrid(
        in_channels=1,
        out_channels=1,
        modes=modes,
        hidden_channels=hidden_width,
        num_layers=num_layers,
        grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        domain_padding=domain_padding,
        rngs=nnx.Rngs(seed),
    )
    params = nnx.state(model, nnx.Param)
    param_count = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print(f"Model parameters: {param_count:,}")

    # --- Trainer: identical recipe, plus distributed_config ---
    print()
    print("Setting up distributed Trainer...")
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
        distributed_config=distributed_config,
    )
    trainer = Trainer(model=model, config=config, rngs=nnx.Rngs(seed))
    print("Trainer created with distributed config")

    print()
    print("Training (data-parallel across the device mesh)...")
    trained_model, metrics = trainer.fit(
        train_data=(jnp.array(x_train_n), jnp.array(y_train_n)),
        val_data=(jnp.array(x_test_n), jnp.array((y_test - y_mean) / y_std)),
    )
    initial_train_loss = float(metrics["initial_train_loss"])
    final_train_loss = float(metrics["final_train_loss"])

    # --- Evaluation in physical pressure units ---
    print()
    print("Running evaluation (un-normalized to physical pressure)...")
    predictions = predict_in_batches(trained_model, jnp.array(x_test_n)) * y_std + y_mean
    test_rel_l2 = float(jnp.mean(per_sample_relative_l2(predictions, jnp.array(y_test))))

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Devices used:      {num_devices}")
    print(f"  Model parameters:  {param_count:,}")
    print(f"  Initial train loss:{initial_train_loss:.6f}")
    print(f"  Final train loss:  {final_train_loss:.6f}")
    print(f"  Test Relative L2:  {test_rel_l2:.6f}")
    print("=" * 70)
    print()
    print("Distributed operator-learning complete!")

    return {
        "num_devices": int(num_devices),
        "model_parameters": param_count,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "test_rel_l2": test_rel_l2,
    }


# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
