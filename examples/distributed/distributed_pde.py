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
# Distributed Data-Parallel Training

| Property      | Value                                        |
|---------------|----------------------------------------------|
| Level         | Intermediate                                 |
| Runtime       | ~10 seconds (1 GPU), ~5 seconds (2+ GPUs)   |
| Memory        | ~500 MB                                      |
| Prerequisites | `source activate.sh`                         |

## Overview

Train a simple PDE solver network using **SPMD data-parallel training**
across all available JAX devices. Opifex wraps the mesh setup and batch
sharding behind `DistributedConfig` — just pass it to `TrainingConfig`
and the `Trainer` handles the rest.

This example demonstrates:
- **DistributedConfig**: Declarative mesh topology configuration
- **Trainer integration**: Zero-code-change distributed training
- **Automatic batch sharding**: Batches are partitioned across devices
- **JIT compilation**: Training step is compiled via `nnx.jit`

**How it works:**
1. `DistributedConfig` describes the device mesh shape and axis names.
2. `Trainer.__init__` creates a `DistributedManager` from this config.
3. `Trainer.fit` shards each mini-batch across the `"data"` mesh axis
   before feeding it to the JIT-compiled training step.
"""

# %%
import jax
import jax.numpy as jnp
from flax import nnx

from opifex.core.training.config import TrainingConfig
from opifex.core.training.trainer import Trainer
from opifex.distributed.config import DistributedConfig


num_devices = jax.device_count()
print("=" * 60)
print("Distributed Data-Parallel Training")
print("=" * 60)
print(f"JAX backend:  {jax.default_backend()}")
print(f"Devices:      {num_devices}")

# %% [markdown]
"""
## Step 1: Define the Model

A minimal feed-forward network standing in for a PDE solver.
Any `nnx.Module` works — the distributed machinery is orthogonal
to the model definition.
"""

# %%


class SimplePDEModel(nnx.Module):
    """Toy feed-forward surrogate for a PDE solution operator."""

    def __init__(self, features: int = 64, rngs: nnx.Rngs | None = None) -> None:
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.layer1 = nnx.Linear(4, features, rngs=rngs)
        self.layer2 = nnx.Linear(features, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the model."""
        return self.layer2(nnx.relu(self.layer1(x)))


model = SimplePDEModel(features=64, rngs=nnx.Rngs(42))
n_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
print(f"Model parameters: {n_params:,}")

# %% [markdown]
"""
## Step 2: Configure Distributed Training

`DistributedConfig` is a frozen dataclass describing the mesh topology.
Pass `-1` for a mesh axis size to use all available devices on that axis.
"""

# %%
distributed_config = DistributedConfig(
    mesh_shape=(num_devices,),
    mesh_axis_names=("data",),
    strategy="data",
)
print(f"Mesh shape:   {distributed_config.mesh_shape}")
print(f"Axis names:   {distributed_config.mesh_axis_names}")
print(f"Strategy:     {distributed_config.strategy}")

# %% [markdown]
"""
## Step 3: Create Trainer with Distributed Config

The only change compared to single-device training is passing
`distributed_config` into `TrainingConfig`. Everything else —
optimizer creation, JIT compilation, gradient computation — stays
the same.
"""

# %%
training_config = TrainingConfig(
    num_epochs=20,
    learning_rate=1e-3,
    batch_size=32,
    verbose=False,
    distributed_config=distributed_config,
)

trainer = Trainer(model, training_config)
print("Trainer created with distributed config")

# %% [markdown]
"""
## Step 4: Generate Data and Train

Synthetic regression data: predict the sum-of-squares from 4 input features.
"""

# %%
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (256, 4))
y = jnp.sum(x**2, axis=-1, keepdims=True)

print(f"Training data: x={x.shape}, y={y.shape}")
print()
print("Training...")
trained_model, metrics = trainer.fit(train_data=(x, y))

# %% [markdown]
"""
## Results
"""

# %%
print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Devices used:    {num_devices}")
print(f"  Initial loss:    {metrics['initial_train_loss']:.6f}")
print(f"  Final loss:      {metrics['final_train_loss']:.6f}")
print("=" * 60)
print()
print("Distributed training complete!")

# %% [markdown]
"""
## Summary

This example showed how to enable data-parallel training with three lines:

1. Create a `DistributedConfig` describing the mesh topology
2. Pass it to `TrainingConfig(distributed_config=...)`
3. Call `trainer.fit()` as usual

The `Trainer` automatically:
- Creates a `DistributedManager` to manage the JAX device mesh
- Shards each mini-batch across the `"data"` axis before the training step
- JIT-compiles the training step via `@nnx.jit`

## Next Steps

- Explore FSDP by setting `strategy="fsdp"` with a 2D mesh
- Add model-parallel sharding with `nnx.with_partitioning()`
- See the [distributed module tests](../../tests/distributed/) for more patterns
"""
