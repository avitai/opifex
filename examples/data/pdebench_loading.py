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

PDEBench is a full benchmark suite for scientific machine learning,
providing HDF5-formatted simulation trajectories across 1D/2D/3D PDEs
(Burgers, Navier-Stokes, Darcy Flow, etc.).

Opifex's `PDEBenchSource` is a **datarax `DataSourceModule`**: it reads the HDF5 file at init,
performs the PDE-specific input/target time-window pairing (the one step datarax has no operator
for), and then exposes the standard datarax contract (`get_batch_at` / `element_spec`) so it is
driven by a datarax **`Pipeline`**. Normalisation is a datarax **`MapOperator`** stage, not baked
into the arrays. `create_pdebench_loader` assembles the source + normalize stage into a Pipeline.

## Learning Goals

1. **Create** a synthetic HDF5 file matching the PDEBench format
2. **Build** a datarax `Pipeline` over the dataset with `create_pdebench_loader`
3. **Batch** data for training via the pipeline's `.step()` (JAX-traceable)
4. **Inspect** the source's element contract (`element_spec`, `get_batch_at`) and coordinate grids
5. **Normalize** with a composable `MapOperator` stage (train/test via the `split` config)
"""

# %%
import tempfile
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.data.sources.scientific import (
    create_pdebench_loader,
    PDEBenchConfig,
    PDEBenchSource,
)


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
## Step 2: The source — HDF5 read, split, window pairing, datarax contract

`PDEBenchSource` reads the file at `__init__`, splits train/test, and creates the input/target
time-window pairs. It then satisfies the datarax `DataSourceModule` contract — `element_spec()`
declares per-element shapes and `get_batch_at(start, size, key)` is a stateless, JAX-traceable
fetch — so a `Pipeline` can drive it. Coordinate grids are domain metadata on `source.coordinates`.
"""


# %%
def build_source(hdf5_path: Path) -> PDEBenchSource:
    """Construct a train-split PDEBenchSource over the synthetic 1D Burgers file."""
    config = PDEBenchConfig(
        file_path=hdf5_path,
        dataset_name="1D_Burgers",
        train_split=0.8,
        split="train",
        input_steps=5,
        output_steps=5,
        normalize=True,
    )
    return PDEBenchSource(config, rngs=nnx.Rngs(0))


# %% [markdown]
"""
## Step 3: Build a datarax Pipeline with `create_pdebench_loader`

The loader assembles the source and (because `normalize=True`) a per-channel min-max
`MapOperator` stage into a `Pipeline`. `pipeline.step()` fetches one batch through the source's
`get_batch_at` and runs the normalize stage — the whole call is JAX-traceable, so it composes with
`pipeline.scan(...)` for a GPU-fused training epoch.
"""

# %% [markdown]
"""
## Step 4: Train vs test splits

Two loaders differing only in the `split` config give normalized train/test pipelines.
"""


# %%
def main() -> dict[str, float | int]:
    """Build PDEBench train/test datarax pipelines and report the loaded contract + batch."""
    print("=" * 72)
    print("Opifex Example: PDEBench loading on datarax (Source + MapOperator + Pipeline)")
    print("=" * 72)
    print(f"JAX backend: {jax.default_backend()}")

    tmp_dir = tempfile.mkdtemp()
    hdf5_path = Path(tmp_dir) / "1D_Burgers_synth.hdf5"
    create_synthetic_pdebench_hdf5(hdf5_path, n_samples=20, n_timesteps=20)

    # --- Source contract ---
    source = build_source(hdf5_path)
    spec = source.element_spec()
    print(f"\nWindowed pairs: {len(source)}  (input_steps=5, output_steps=5)")
    print(f"  element_spec input:  {spec['input'].shape}")
    print(f"  element_spec target: {spec['target'].shape}")
    print(f"  coordinates: {None if source.coordinates is None else list(source.coordinates)}")

    # --- datarax Pipeline (train) ---
    batch_size = 4
    train_config = PDEBenchConfig(
        file_path=hdf5_path,
        dataset_name="1D_Burgers",
        train_split=0.8,
        split="train",
        input_steps=5,
        output_steps=5,
        normalize=True,
    )
    test_config = PDEBenchConfig(
        file_path=hdf5_path,
        dataset_name="1D_Burgers",
        train_split=0.8,
        split="test",
        input_steps=5,
        output_steps=5,
        normalize=True,
    )
    train_loader = create_pdebench_loader(train_config, batch_size=batch_size)
    test_loader = create_pdebench_loader(test_config, batch_size=batch_size)

    batch = train_loader.step()
    input_min, input_max = float(jnp.min(batch["input"])), float(jnp.max(batch["input"]))
    print(f"\nPipeline.step() batch: input {batch['input'].shape}, target {batch['target'].shape}")
    print(f"  normalized input range: [{input_min:.4f}, {input_max:.4f}]  (MapOperator stage)")

    print(f"\nTrain pairs: {len(train_loader.source)}   Test pairs: {len(test_loader.source)}")
    print("=" * 72)

    return {
        "train_pairs": len(train_loader.source),
        "test_pairs": len(test_loader.source),
        "batch_size": batch_size,
        "input_min": input_min,
        "input_max": input_max,
    }


# %% [markdown]
"""
## Results Summary

| Aspect | How it is built |
|--------|-----------------|
| Source | `PDEBenchSource` (datarax `DataSourceModule`): HDF5 read + split + window pairing |
| Contract | `element_spec()` + stateless, traceable `get_batch_at(start, size, key)` |
| Normalization | datarax `MapOperator` stage (per-channel min-max), not baked into arrays |
| Batching | datarax `Pipeline.step()` / `.scan()` |
| Splits | Train/test via the `split` config |

## Next Steps

- Use the loader with real PDEBench `.hdf5` files from
  [PDEBench](https://github.com/pdebench/PDEBench)
- Drive training with `pipeline.scan(step_fn, length=...)` for a GPU-fused epoch
- See [Darcy Flow Analysis](darcy_flow_analysis.md) for a related example
"""


# %%
if __name__ == "__main__":
    summary = main()
    for metric_name, metric_value in summary.items():
        print(f"{metric_name}: {metric_value}")
