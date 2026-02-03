# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

#!/usr/bin/env python3

# %% [markdown]
"""
# PDEBench Dataset Loading with Opifex

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~30 sec (GPU) |
| **Prerequisites** | JAX, h5py, HDF5 file |
| **Format** | Python + Jupyter |

## Overview

PDEBench is a comprehensive benchmark suite for scientific machine learning,
providing HDF5-formatted simulation trajectories across 1D/2D/3D PDEs
(Burgers, Navier-Stokes, Darcy Flow, etc.).

Opifex's `PDEBenchSource` provides an eager-loading interface that converts
HDF5 data to JAX arrays at initialization, then offers pure-JAX iteration
for training neural operators.

## Learning Goals

1. **Create** a synthetic HDF5 file matching the PDEBench format
2. **Load** the dataset with `PDEBenchSource` — all I/O at init
3. **Inspect** shapes, sliding window pairs, and coordinate grids
4. **Batch** data for training with `get_batch()`
5. **Iterate** over the full dataset with epoch reset
"""

# %%
import tempfile
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource


# %% [markdown]
"""
## Step 1: Create a Synthetic PDEBench HDF5 File

PDEBench datasets follow a specific HDF5 structure:
- `/tensor`: shape `(N, T, X[, Y], C)` — simulation trajectories
- `/x`, `/y`, `/t`: optional coordinate grids

We create a small synthetic dataset for demonstration.
"""


# %%
def create_synthetic_pdebench_hdf5(
    file_path: Path,
    n_samples: int = 20,
    n_timesteps: int = 20,
    n_spatial: int = 64,
    n_channels: int = 1,
) -> None:
    """Create a synthetic 1D Burgers-like HDF5 file.

    The data simulates a simple diffusing step function to
    approximate the structure of real PDEBench Burgers data.
    """
    rng = np.random.default_rng(42)

    # Generate smooth initial conditions and evolve them
    x = np.linspace(0, 2 * np.pi, n_spatial)
    t = np.linspace(0, 1, n_timesteps)
    data = np.zeros((n_samples, n_timesteps, n_spatial, n_channels))

    for i in range(n_samples):
        # Random superposition of sinusoids as initial condition
        u0 = rng.normal(0, 0.5) * np.sin(x) + rng.normal(0, 0.3) * np.sin(2 * x)
        for j, tj in enumerate(t):
            # Simple diffusion-like evolution
            data[i, j, :, 0] = u0 * np.exp(-0.5 * tj)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("tensor", data=data.astype(np.float32))
        f.create_dataset("x", data=x.astype(np.float32))
        f.create_dataset("t", data=t.astype(np.float32))

    print(f"Created synthetic HDF5: {file_path}")
    print(f"  tensor shape: {data.shape}")
    print(f"  x shape:      {x.shape}")
    print(f"  t shape:      {t.shape}")


# %% [markdown]
"""
## Step 2: Load with PDEBenchSource

`PDEBenchSource` handles all file I/O at `__init__`:
1. Reads the HDF5 file
2. Applies train/test split along the sample axis
3. Creates sliding window input/target pairs over time
4. Optionally normalizes to [0, 1]
5. Converts everything to JAX arrays

After init, the source is pure JAX — no more file I/O.
"""

# %%
# Create temporary synthetic data
tmp_dir = tempfile.mkdtemp()
hdf5_path = Path(tmp_dir) / "1D_Burgers_synth.hdf5"
create_synthetic_pdebench_hdf5(hdf5_path, n_samples=20, n_timesteps=20)

# Configure and load
config = PDEBenchConfig(
    file_path=hdf5_path,
    dataset_name="1D_Burgers",
    train_split=0.8,
    split="train",
    input_steps=5,
    output_steps=5,
    normalize=True,
)
source = PDEBenchSource(config, rngs=nnx.Rngs(0))

print(f"\nDataset loaded: {len(source)} sliding window pairs")
print(f"  inputs shape:  {source.inputs.shape}")
print(f"  targets shape: {source.targets.shape}")
print(f"  coordinates:   {source.coordinates is not None}")
if source.coordinates is not None:
    for k, v in source.coordinates.items():
        print(f"    {k}: {v.shape}")

# %% [markdown]
"""
## Step 3: Inspect Individual Elements

Each element is a dict with:
- `"input"`: shape `(input_steps, *spatial, channels)`
- `"target"`: shape `(output_steps, *spatial, channels)`
- `"coordinates"`: dict of spatial/temporal grids (if available)
"""

# %%
element = source[0]
print("Element keys:", list(element.keys()))
print(f"  input shape:  {element['input'].shape}")
print(f"  target shape: {element['target'].shape}")

# Verify normalization
print(
    f"\n  input range:  [{float(jnp.min(element['input'])):.4f}, "
    f"{float(jnp.max(element['input'])):.4f}]"
)
print(
    f"  target range: [{float(jnp.min(element['target'])):.4f}, "
    f"{float(jnp.max(element['target'])):.4f}]"
)

# %% [markdown]
"""
## Step 4: Batch Retrieval for Training

`get_batch()` supports two modes:
1. **Stateful** (no key): sequential batches from current position
2. **Stateless** (with key): random batches for evaluation
"""

# %%
# Stateful sequential batch
batch = source.get_batch(batch_size=4)
print("Sequential batch:")
print(f"  input shape:  {batch['input'].shape}")
print(f"  target shape: {batch['target'].shape}")

# Stateless random batch
key = jax.random.key(42)
random_batch = source.get_batch(batch_size=8, key=key)
print("\nRandom batch:")
print(f"  input shape:  {random_batch['input'].shape}")
print(f"  target shape: {random_batch['target'].shape}")

# %% [markdown]
"""
## Step 5: Full Epoch Iteration

Iterate over all elements in the dataset.
Call `reset()` to start a new epoch.
"""

# %%
# Full iteration
count = 0
for _element in source:
    count += 1
print(f"Iterated over {count} elements (dataset has {len(source)})")

# Reset for another epoch
source.reset()
batch_after_reset = source.get_batch(batch_size=2)
print(f"After reset, batch input shape: {batch_after_reset['input'].shape}")

# %% [markdown]
"""
## Step 6: Train vs Test Split

Create separate sources for training and evaluation
by changing the `split` parameter.
"""

# %%
test_config = PDEBenchConfig(
    file_path=hdf5_path,
    dataset_name="1D_Burgers",
    train_split=0.8,
    split="test",
    input_steps=5,
    output_steps=5,
    normalize=True,
)
test_source = PDEBenchSource(test_config, rngs=nnx.Rngs(0))

print(f"Train samples: {len(source)}")
print(f"Test samples:  {len(test_source)}")

# %% [markdown]
"""
## Results Summary

| Metric | Value |
|--------|-------|
| I/O Strategy | Eager — all at init, pure JAX after |
| Window Pairing | Sliding window over time axis |
| Normalization | Per-channel min-max to [0, 1] |
| Batch Modes | Stateful (sequential) and stateless (random) |
| Split Support | Train/test via `split` parameter |

## Next Steps

- Use `PDEBenchSource` with real PDEBench `.hdf5` files from
  [PDEBench](https://github.com/pdebench/PDEBench)
- Train an FNO or DeepONet on the loaded data
- See [Darcy Flow Analysis](darcy_flow_analysis.md) for a related example
"""


# %%
def main():
    """Run the PDEBench loading example."""
    print("=" * 60)
    print("PDEBench Loading Example — Complete")
    print("=" * 60)
    print(f"Loaded {len(source)} train + {len(test_source)} test pairs")
    print(f"Input shape:  {source.inputs.shape}")
    print(f"Target shape: {source.targets.shape}")
    print(f"Backend:      {jax.default_backend()}")


if __name__ == "__main__":
    main()
