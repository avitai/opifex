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
# PDEBench Dataset Loading with Opifex

| Metadata | Value |
| --- | --- |
| **Tier** | Tutorial |
| **Devices** | CPU-compatible |
| **Prerequisites** | JAX, h5py and the HDF5 format |

## Overview

PDEBench is a full benchmark suite for scientific machine learning,
providing HDF5-formatted simulation trajectories across 1D/2D/3D PDEs
(Burgers, Navier-Stokes, Darcy Flow, etc.).

Opifex reads a PDEBench file as PDEBench's own loader does (`FNODatasetSingle` in
`pdebench/models/fno/utils.py`): every field becomes `[N, x..., t, V]`, the test split is the
first `test_ratio` of the samples, and a record is the first `initial_step` steps of a trajectory
(the input) with the whole trajectory (the target). `load_pdebench` returns one split's arrays;
`create_pdebench_loader` serves them through a datarax `Pipeline`.

## Learning Goals

1. **Create** a synthetic HDF5 file in PDEBench's 1D Burgers layout
2. **Load** the train and test splits with `load_pdebench`
3. **Batch** records through a datarax `Pipeline` with `create_pdebench_loader`
4. **Normalize** with statistics fitted on the train split only
"""

# %%
import tempfile
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from opifex.core.normalization import GaussianNormalizer
from opifex.data.sources.pdebench import create_pdebench_loader, load_pdebench, PDEBenchConfig


# %% [markdown]
"""
## Step 1: Create a Synthetic PDEBench HDF5 File

PDEBench's 1D Burgers files (for example `1D_Burgers_Sols_Nu1.0.hdf5`) hold:
- `tensor`: shape `(N, T, X)`, one trajectory per sample
- `x-coordinate`: shape `(X,)`
- `t-coordinate`: shape `(T + 1,)`

We create a small synthetic dataset in that layout.
"""


# %%
def create_synthetic_pdebench_hdf5(
    file_path: Path,
    n_samples: int = 20,
    n_timesteps: int = 20,
    n_spatial: int = 64,
) -> None:
    """Create a synthetic file in PDEBench's 1D Burgers layout.

    The data is a decaying superposition of sinusoids, which has the shape of real PDEBench
    Burgers data, not its physics.
    """
    rng = np.random.default_rng(42)

    x = np.linspace(0, 2 * np.pi, n_spatial)
    t = np.linspace(0, 1, n_timesteps + 1)
    data = np.zeros((n_samples, n_timesteps, n_spatial))
    for i in range(n_samples):
        u0 = rng.normal(0, 0.5) * np.sin(x) + rng.normal(0, 0.3) * np.sin(2 * x)
        for j, tj in enumerate(t[:-1]):
            data[i, j] = u0 * np.exp(-0.5 * tj)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("tensor", data=data.astype(np.float32))
        f.create_dataset("x-coordinate", data=x.astype(np.float32))
        f.create_dataset("t-coordinate", data=t.astype(np.float32))

    print(f"Created synthetic HDF5: {file_path}")
    print(f"  tensor shape:       {data.shape}")
    print(f"  x-coordinate shape: {x.shape}")
    print(f"  t-coordinate shape: {t.shape}")


# %% [markdown]
"""
## Step 2: Load the splits

`load_pdebench` reads one split: the test split is the first `int(N * test_ratio)` samples and
the train split the rest. `inputs` are the first `initial_step` time steps of each `targets`
trajectory, and `grid` holds the coordinates every sample shares.
"""

# %% [markdown]
"""
## Step 3: Batch through a datarax Pipeline

`create_pdebench_loader` serves a split's `{"input", "target"}` records through a `MemorySource`
and a `Pipeline`; the train split is shuffled from the seed.
"""

# %% [markdown]
"""
## Step 4: Normalize with train statistics

PDEBench applies no normalisation. Fit a `GaussianNormalizer` on the train split and reuse it
for the test split, so both are scaled by the same, train-only statistics.
"""


# %%
def main() -> dict[str, float | int]:
    """Load PDEBench train/test splits, batch them, and normalize with train statistics."""
    print("=" * 72)
    print("Opifex Example: PDEBench loading")
    print("=" * 72)
    print(f"JAX backend: {jax.default_backend()}")

    tmp_dir = tempfile.mkdtemp()
    hdf5_path = Path(tmp_dir) / "1D_Burgers_synth.hdf5"
    create_synthetic_pdebench_hdf5(hdf5_path, n_samples=20, n_timesteps=20)

    train = load_pdebench(PDEBenchConfig(file_path=hdf5_path, split="train", initial_step=10))
    test = load_pdebench(PDEBenchConfig(file_path=hdf5_path, split="test", initial_step=10))
    print(f"\nTrain: inputs {train.inputs.shape}, targets {train.targets.shape}")
    print(f"Test:  inputs {test.inputs.shape}, targets {test.targets.shape}")
    print(f"Grid:  {train.grid.shape}")

    batch_size = 4
    loader = create_pdebench_loader(
        PDEBenchConfig(file_path=hdf5_path, split="train", initial_step=10),
        batch_size=batch_size,
        seed=0,
    )
    batch = next(iter(loader))
    print(f"\nPipeline batch: input {batch['input'].shape}, target {batch['target'].shape}")

    normalizer = GaussianNormalizer.fit(jnp.asarray(train.targets))
    test_targets = normalizer.normalize(jnp.asarray(test.targets))
    test_mean, test_std = float(jnp.mean(test_targets)), float(jnp.std(test_targets))
    print(f"Test targets with train statistics: mean {test_mean:.4f}, std {test_std:.4f}")
    print("=" * 72)

    summary = {
        "train_samples": int(train.targets.shape[0]),
        "test_samples": int(test.targets.shape[0]),
        "batch_size": batch_size,
        "test_mean_with_train_statistics": test_mean,
        "test_std_with_train_statistics": test_std,
    }
    for name, value in summary.items():
        print(f"{name}: {value}")
    return summary


# %% [markdown]
"""
## Results Summary

| Aspect | How it is built |
|--------|-----------------|
| Reading | `load_pdebench`: PDEBench's layouts, strides and split |
| Records | `(target[..., :initial_step, :], target)`, grid shared |
| Batching | datarax `MemorySource` + `Pipeline` (`create_pdebench_loader`) |
| Normalization | `GaussianNormalizer` fitted on the train split |

## Next Steps

- Use the loader with real PDEBench files from [PDEBench](https://github.com/pdebench/PDEBench)
  (Darcy flow takes `initial_step=1`)
- See [Darcy Flow Analysis](darcy_flow_analysis.md) for a related example
"""


# %%
if __name__ == "__main__":
    main()
