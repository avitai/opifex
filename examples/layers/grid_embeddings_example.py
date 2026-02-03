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
# Grid Embeddings for Neural Operators

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~2 min (CPU) |
| **Prerequisites** | JAX, NumPy, Neural Operators basics |
| **Format** | Python + Jupyter |

## Overview

Grid embeddings inject spatial coordinate information into neural operator inputs,
enabling the model to learn position-dependent features. This is essential for
operators like FNO that operate on spatially structured data.

## Learning Goals

1. **Create** spatial coordinate embeddings with `GridEmbedding2D`
2. **Generalize** embeddings to N dimensions with `GridEmbeddingND`
3. **Apply** frequency-based positional encoding with `SinusoidalEmbedding`
4. **Visualize** embedding coordinate grids and their effects
"""

# %%
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from opifex.neural.operators.common.embeddings import (
    GridEmbedding2D,
    GridEmbeddingND,
    regular_grid_2d,
    SinusoidalEmbedding,
)


# %% [markdown]
"""
# Grid Embeddings for Neural Operators

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~2 min (CPU) |
| **Prerequisites** | JAX, NumPy, Neural Operators basics |
| **Format** | Python + Jupyter |

## Overview

Grid embeddings inject spatial coordinate information into neural operator inputs,
enabling the model to learn position-dependent features. This is essential for
operators like FNO that operate on spatially structured data.

This example demonstrates three embedding methods available in Opifex:
`GridEmbedding2D` for standard 2D coordinate injection, `GridEmbeddingND` for
arbitrary dimensions, and `SinusoidalEmbedding` for frequency-based positional
encoding (Transformer-style).

## Learning Goals

1. Create spatial coordinate embeddings with `GridEmbedding2D`
2. Generalize embeddings to N dimensions with `GridEmbeddingND`
3. Apply frequency-based positional encoding with `SinusoidalEmbedding`
4. Visualize embedding coordinate grids
"""


# %%
def demonstrate_grid_embedding_2d(
    spatial_shape: tuple[int, int] = (32, 32),
    batch_size: int = 4,
    in_channels: int = 3,
    grid_boundaries: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Demonstrate 2D grid embedding functionality."""
    if grid_boundaries is None:
        grid_boundaries = [[0.0, 1.0], [0.0, 1.0]]

    print()
    print("Grid Embedding 2D Demonstration")
    print(f"   Spatial Shape: {spatial_shape}")
    print(f"   Grid Boundaries: {grid_boundaries}")

    # Create grid embedding
    embedding = GridEmbedding2D(
        in_channels=in_channels, grid_boundaries=grid_boundaries
    )

    # Generate sample input data
    rng_key = jax.random.PRNGKey(42)
    sample_input = jax.random.normal(rng_key, (batch_size, *spatial_shape, in_channels))

    # Apply embedding
    start_time = time.time()
    embedded_data = embedding(sample_input)
    embedding_time = time.time() - start_time

    # Get coordinate grids
    x_grid, y_grid = embedding.get_grid(spatial_shape)

    print(f"   Input Shape: {sample_input.shape}")
    print(f"   Output Shape: {embedded_data.shape}")
    print(f"   Output Channels: {embedding.out_channels}")
    print(f"   Embedding Time: {embedding_time * 1000:.2f} ms")

    return {
        "embedded_data": embedded_data,
        "coordinate_grids": (x_grid, y_grid),
        "sample_input": sample_input,
        "embedding": embedding,
        "embedding_info": {
            "type": "GridEmbedding2D",
            "in_channels": in_channels,
            "out_channels": embedding.out_channels,
            "grid_boundaries": grid_boundaries,
        },
    }


# %% [markdown]
"""
## 2. N-Dimensional Grid Embedding

Generalizing to arbitrary dimensions.
"""


# %%
def demonstrate_grid_embedding_nd(
    spatial_shape: tuple[int, ...],
    batch_size: int = 2,
    in_channels: int = 2,
    grid_boundaries: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Demonstrate N-dimensional grid embedding functionality."""
    dim = len(spatial_shape)
    if grid_boundaries is None:
        grid_boundaries = [[0.0, 1.0] for _ in range(dim)]

    print()
    print(f"Grid Embedding {dim}D Demonstration")
    print(f"   Spatial Shape: {spatial_shape}")
    print(f"   Dimensions: {dim}")

    # Create N-dimensional grid embedding
    embedding = GridEmbeddingND(
        in_channels=in_channels, dim=dim, grid_boundaries=grid_boundaries
    )

    # Generate sample input data
    rng_key = jax.random.PRNGKey(43)
    sample_input = jax.random.normal(rng_key, (batch_size, *spatial_shape, in_channels))

    # Apply embedding
    start_time = time.time()
    embedded_data = embedding(sample_input)
    embedding_time = time.time() - start_time

    print(f"   Input Shape: {sample_input.shape}")
    print(f"   Output Shape: {embedded_data.shape}")
    print(f"   Output Channels: {embedding.out_channels}")
    print(f"   Coordinate Channels: {dim}")
    print(f"   Embedding Time: {embedding_time * 1000:.2f} ms")

    # Get coordinate grids for each dimension
    coordinate_grids = embedding.get_grid(spatial_shape)

    return {
        "embedded_data": embedded_data,
        "coordinate_grids": coordinate_grids,
        "sample_input": sample_input,
        "embedding": embedding,
        "embedding_info": {
            "type": "GridEmbeddingND",
            "dim": dim,
            "in_channels": in_channels,
            "out_channels": embedding.out_channels,
            "grid_boundaries": grid_boundaries,
        },
    }


# %% [markdown]
"""
## 3. Sinusoidal Embedding

Frequency-based positional encoding (Transformer-style).
"""


# %%
def demonstrate_sinusoidal_embedding(
    spatial_shape: tuple[int, int] = (32, 32),
    batch_size: int = 4,
    in_channels: int = 3,
    num_frequencies: int = 8,
) -> dict[str, Any]:
    """Demonstrate sinusoidal embedding functionality."""
    print()
    print("Sinusoidal Embedding Demonstration")
    print(f"   Spatial Shape: {spatial_shape}")
    print(f"   Frequencies: {num_frequencies}")

    # Create sinusoidal embedding
    embedding = SinusoidalEmbedding(
        in_channels=in_channels,
        num_frequencies=num_frequencies,
        embedding_type="transformer",
    )

    # Generate sample input data (batch, n_points, channels)
    rng_key = jax.random.PRNGKey(44)
    h, w = spatial_shape
    n_points = h * w

    # Create coordinate data
    x_coords = jnp.linspace(0, 1, w)
    y_coords = jnp.linspace(0, 1, h)
    x_grid, y_grid = jnp.meshgrid(x_coords, y_coords, indexing="xy")

    # Flatten spatial coordinates to (n_points, 2)
    coord_flat = jnp.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)
    coord_batched = jnp.repeat(coord_flat[None, :, :], batch_size, axis=0)

    # Add input channels
    input_features = jax.random.normal(rng_key, (batch_size, n_points, in_channels))
    sample_input = input_features  # Using features directly as SinusoidalEmbedding usually expects features

    # Apply embedding
    start_time = time.time()
    embedded_data = embedding(sample_input)
    embedding_time = time.time() - start_time

    # Reshape back to spatial for display
    embedded_spatial = embedded_data.reshape(batch_size, h, w, -1)

    print(f"   Input Shape: {sample_input.shape}")
    print(f"   Output Shape: {embedded_data.shape}")
    print(f"   Output Channels: {embedding.out_channels}")
    print(f"   Embedding Time: {embedding_time * 1000:.2f} ms")

    # Compute frequency analysis (use float32 to avoid int32 overflow for large num_frequencies)
    frequencies = jnp.array([2**i for i in range(num_frequencies)], dtype=jnp.float32)

    return {
        "embedded_data": embedded_spatial,
        "sample_input": sample_input,
        "embedding": embedding,
        "embedding_info": {
            "type": "SinusoidalEmbedding",
            "in_channels": in_channels,
            "out_channels": embedding.out_channels,
            "num_frequencies": num_frequencies,
        },
        "frequency_analysis": {
            "frequencies": frequencies,
            "embedding_patterns": embedded_spatial[0, :, :, :num_frequencies],
        },
    }


# %% [markdown]
"""
## Visualization

Visualizing the generated coordinate grids.
"""


# %%
def visualize_coordinate_grids(
    spatial_shape: tuple[int, int] = (16, 16),
    grid_boundaries: list[list[float]] | None = None,
) -> None:
    """Visualize coordinate grids and embedding effects."""
    if grid_boundaries is None:
        grid_boundaries = [[-1.0, 1.0], [-1.0, 1.0]]

    print()
    print("Coordinate Grid Visualization")

    # Generate coordinate grids
    x_grid, y_grid = regular_grid_2d(spatial_shape, grid_boundaries)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Grid Embedding Coordinate Visualization", fontsize=16)

    # X coordinates
    im1 = axes[0, 0].imshow(x_grid, cmap="viridis", aspect="equal")
    axes[0, 0].set_title("X Coordinates")
    plt.colorbar(im1, ax=axes[0, 0])

    # Y coordinates
    im2 = axes[0, 1].imshow(y_grid, cmap="plasma", aspect="equal")
    axes[0, 1].set_title("Y Coordinates")
    plt.colorbar(im2, ax=axes[0, 1])

    # Magnitude
    coord_magnitude = jnp.sqrt(x_grid**2 + y_grid**2)
    im3 = axes[1, 0].imshow(coord_magnitude, cmap="coolwarm", aspect="equal")
    axes[1, 0].set_title("Coordinate Magnitude")
    plt.colorbar(im3, ax=axes[1, 0])

    # Contours
    axes[1, 1].contour(x_grid, levels=10, colors="blue", alpha=0.6)
    axes[1, 1].contour(y_grid, levels=10, colors="red", alpha=0.6)
    axes[1, 1].set_title("Coordinate Contours")
    axes[1, 1].legend(["X contours", "Y contours"])

    plt.tight_layout()
    output_dir = Path("docs/assets/examples/grid_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / "coordinate_grids.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved coordinate grid visualization")


# %% [markdown]
"""
## Results Summary

| Embedding Method | Input Channels | Output Channels | Added Dimensions |
|-----------------|----------------|-----------------|------------------|
| GridEmbedding2D | 3 | 5 | +2 (x, y coordinates) |
| GridEmbeddingND (3D) | 2 | 5 | +3 (x, y, z coordinates) |
| SinusoidalEmbedding | 3 | 51 | +48 (frequency encodings) |

## Next Steps

### Experiments to Try

1. Change `grid_boundaries` to non-standard ranges (e.g., `[[-1, 1], [0, 2*pi]]`)
2. Increase `num_frequencies` in sinusoidal embedding and observe channel growth
3. Apply embeddings to real Darcy flow data before FNO training

### Related Examples

- [DISCO Convolutions](disco_convolutions_example.md) - Convolutions on arbitrary grids
- [FNO Darcy Comprehensive](../models/fno_darcy_comprehensive.md) - FNO using grid embeddings
- [Fourier Continuation](fourier_continuation_example.md) - Boundary handling for spectral methods

### API Reference

- [`GridEmbedding2D`](../../api/neural.md) - 2D spatial coordinate injection
- [`GridEmbeddingND`](../../api/neural.md) - N-dimensional coordinate embedding
- [`SinusoidalEmbedding`](../../api/neural.md) - Frequency-based positional encoding
"""


# %%
def main():
    """Run all grid embedding demonstrations."""
    print("=" * 60)
    print("GRID EMBEDDINGS FOR NEURAL OPERATORS")
    print("=" * 60)

    grid_2d_results = demonstrate_grid_embedding_2d()
    grid_nd_results = demonstrate_grid_embedding_nd(spatial_shape=(16, 16, 16))
    sinusoidal_results = demonstrate_sinusoidal_embedding()
    visualize_coordinate_grids()

    print()
    print("=" * 60)
    print("Grid embedding demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
