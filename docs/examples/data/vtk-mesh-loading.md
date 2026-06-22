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

## Files

- **Python Script**: [`examples/data/vtk_mesh_loading.py`](https://github.com/avitai/opifex/blob/main/examples/data/vtk_mesh_loading.py)
- **Jupyter Notebook**: [`examples/data/vtk_mesh_loading.ipynb`](https://github.com/avitai/opifex/blob/main/examples/data/vtk_mesh_loading.ipynb)

## Quick Start

```bash
source activate.sh && python examples/data/vtk_mesh_loading.py
```

## Core Concept

Three triangulated patches at resolutions 6/8/12 give meshes of **36, 64, and 144 nodes** — the
ragged case. `VTKMeshSource` pads every mesh to `max_nodes = 144` and stores a `node_mask` so the
padding is recoverable; a datarax `Pipeline` then batches them into uniform tensors.

```python
from opifex.data.sources.scientific import VTKMeshConfig, create_vtk_mesh_loader

loader = create_vtk_mesh_loader(
    VTKMeshConfig(directory=mesh_dir, node_features=("velocity",)),
    batch_size=3,
)
batch = loader.step()   # node_positions (3, 144, 3), node_mask (3, 144), edge_index (3, 2, 770) ...
```

## Results

**Terminal Output:**

```text
Wrote 3 meshes with node counts [36, 64, 144]

Loaded 3 meshes; padded to max_nodes = 144
  element_spec (per-mesh padded shapes):
    node_positions  (144, 3)
    node_mask       (144,)
    node_features   (144, 3)
    edge_index      (2, 770)
    edge_mask       (770,)

  real nodes per mesh (from node_mask): [144, 36, 64]
  (every mesh stored as (144, 3); the mask hides the padding)

Pipeline.step() batch shapes:
    edge_index      (3, 2, 770)
    edge_mask       (3, 770)
    node_features   (3, 144, 3)
    node_mask       (3, 144)
    node_positions  (3, 144, 3)

  masked node total 244 == true total 244: True
```

The masked node total (`244`) equals the sum of the meshes' real node counts (`36 + 64 + 144`),
confirming the masks exactly recover the unpadded data after batching.

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
