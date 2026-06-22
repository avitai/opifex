# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# VTK Mesh Loading: Batching Irregular Meshes on datarax

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~20 s (GPU) / ~40 s (CPU) |
| **Prerequisites** | JAX, meshio, unstructured meshes, GNN batching |
| **Format** | Python + Jupyter |
| **Memory** | ~0.5 GB |

## Overview

Unstructured-mesh simulation data (CFD, FEM) is **ragged**: every mesh has a different number of
nodes and edges, so the meshes cannot be stacked into a single dense tensor for batched training.
The standard fix in JAX graph learning is to **pad each mesh to the dataset maximum and carry a
boolean mask**, giving uniform arrays that JIT-compile and batch cleanly.

Opifex's `VTKMeshSource` is a datarax `DataSourceModule` that does exactly this at load: it reads
`.vtu`/`.vtp` files with `meshio`, converts cell connectivity to a COO `edge_index`, pads
`node_positions` / `node_features` to `max_nodes` (+ `node_mask`) and `edge_index` to `max_edges`
(+ `edge_mask`), and exposes the datarax contract (`element_spec`, stateless `get_batch_at`) so a
`Pipeline` can drive batched, JIT-traceable iteration. `create_vtk_mesh_loader` assembles it.

## What You'll Learn

1. Load irregular meshes with `VTKMeshSource` (ragged → padded + masked)
2. Read the source's datarax contract: `element_spec` and `get_batch_at`
3. Batch variable-size meshes through a datarax `Pipeline` and use the masks
"""

# %%
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.data.sources.scientific import (
    create_vtk_mesh_loader,
    VTKMeshConfig,
    VTKMeshSource,
)


# %% [markdown]
"""
## Step 1: Synthetic unstructured meshes of varying resolution

We triangulate square patches at several resolutions, so the meshes genuinely differ in node and
edge counts (the ragged case that motivates padding). Each node carries a 3-component "velocity"
field sampled from a smooth analytic flow, mimicking a CFD solution on an unstructured mesh.
"""


# %%
def triangulated_grid(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(points, triangles, velocity)`` for a ``resolution x resolution`` triangulated patch.

    ``points`` is ``(resolution**2, 3)`` (z = 0), ``triangles`` indexes two triangles per cell, and
    ``velocity`` is a smooth analytic field ``(u, v, 0)`` sampled at the nodes.
    """
    axis = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(axis, axis)
    points = np.stack([xx.ravel(), yy.ravel(), np.zeros(resolution * resolution)], axis=-1)

    triangles = []
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            a = row * resolution + col
            b = a + 1
            c = a + resolution
            d = c + 1
            triangles.append([a, b, c])
            triangles.append([b, d, c])

    # Smooth analytic velocity field (a simple vortex), per node.
    u = -np.sin(np.pi * points[:, 1])
    v = np.sin(np.pi * points[:, 0])
    velocity = np.stack([u, v, np.zeros_like(u)], axis=-1)

    return (
        points.astype(np.float32),
        np.array(triangles, dtype=np.int64),
        velocity.astype(np.float32),
    )


def write_synthetic_meshes(directory: Path, resolutions: tuple[int, ...]) -> None:
    """Write one ``.vtu`` file per resolution with a nodal velocity field."""
    import meshio

    for resolution in resolutions:
        points, triangles, velocity = triangulated_grid(resolution)
        mesh = meshio.Mesh(
            points=points,
            cells=[("triangle", triangles)],
            point_data={"velocity": velocity},
        )
        mesh.write(directory / f"patch_{resolution}.vtu")


# %% [markdown]
"""
## Step 2: Load + batch through a datarax Pipeline

`create_vtk_mesh_loader` builds a `VTKMeshSource` (which pads all meshes to the dataset maxima and
adds masks) and wraps it in a datarax `Pipeline`. `pipeline.step()` returns a batch of padded
meshes; `node_mask` tells the model which nodes are real so padded nodes contribute nothing.
"""


# %%
def main() -> dict[str, float | int]:
    """Load synthetic varying-size meshes and report the padded/masked datarax batch contract."""
    print("=" * 72)
    print("Opifex Example: VTK mesh loading on datarax (ragged -> padded + masked Pipeline)")
    print("=" * 72)
    print(f"JAX backend: {jax.default_backend()}")

    resolutions = (6, 8, 12)  # 36, 64, 144 nodes -> genuinely ragged
    tmp_dir = Path(tempfile.mkdtemp())
    write_synthetic_meshes(tmp_dir, resolutions)
    print(f"\nWrote {len(resolutions)} meshes with node counts {[r * r for r in resolutions]}")

    source = VTKMeshSource(
        VTKMeshConfig(directory=tmp_dir, node_features=("velocity",)),
        rngs=nnx.Rngs(0),
    )
    spec = source.element_spec()
    max_nodes = spec["node_positions"].shape[0]
    print(f"\nLoaded {len(source)} meshes; padded to max_nodes = {max_nodes}")
    print("  element_spec (per-mesh padded shapes):")
    for key, value in spec.items():
        print(f"    {key:<15} {tuple(value.shape)}")

    # Per-mesh masks record each mesh's true node count (the rest is padding).
    real_nodes = [int(source[i]["node_mask"].sum()) for i in range(len(source))]
    print(f"\n  real nodes per mesh (from node_mask): {real_nodes}")
    print(f"  (every mesh stored as ({max_nodes}, 3); the mask hides the padding)")

    # Batch all meshes through the datarax Pipeline.
    loader = create_vtk_mesh_loader(
        VTKMeshConfig(directory=tmp_dir, node_features=("velocity",)),
        batch_size=len(resolutions),
    )
    batch = loader.step()
    print("\nPipeline.step() batch shapes:")
    for key, value in batch.items():
        print(f"    {key:<15} {tuple(value.shape)}")

    # The masked node count summed over the batch equals the true total node count.
    masked_total = int(jnp.sum(batch["node_mask"]))
    true_total = sum(r * r for r in resolutions)
    print(
        f"\n  masked node total {masked_total} == true total {true_total}: {masked_total == true_total}"
    )
    print("=" * 72)

    return {
        "num_meshes": len(source),
        "max_nodes": int(max_nodes),
        "min_real_nodes": int(min(real_nodes)),
        "masked_node_total": masked_total,
        "true_node_total": true_total,
    }


# %% [markdown]
"""
## Results Summary

| Aspect | How it is built |
|--------|-----------------|
| Source | `VTKMeshSource` (datarax `DataSourceModule`): meshio read + COO edges + pad-to-max + masks |
| Contract | `element_spec()` + stateless, traceable `get_batch_at(start, size, key)` |
| Ragged handling | per-axis pad to dataset max; `node_mask`/`edge_mask` flag real vs padded entries |
| Batching | datarax `Pipeline.step()` / `.scan()` |

## Next Steps

- Use the loader with real `.vtu`/`.vtp` meshes (CFD/FEM exports)
- Feed `node_features` + `edge_index` (with `node_mask`/`edge_mask`) to a GNN; multiply messages by
  the masks so padded nodes/edges contribute nothing
- See [PDEBench Loading](pdebench-loading.md) for the structured-grid counterpart
"""


# %%
if __name__ == "__main__":
    summary = main()
    for metric_name, metric_value in summary.items():
        print(f"{metric_name}: {metric_value}")
