"""Utilities for graph neural operators.

Functions for converting grid data to graph representations compatible with
GraphNeuralOperator.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp


def grid_to_graph_data(
    grid: jax.Array,
    connectivity: int | Literal["radius"] = 4,
    radius: float = 1.5,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert 2D grid data to graph representation for GraphNeuralOperator.

    Converts a batch of 2D grids into graph format with:
    - Node features: grid values + normalized (x, y) positions
    - Edge indices: connectivity between neighboring grid points
    - Edge features: relative position vectors between connected nodes

    Args:
        grid: Input grid of shape [batch, channels, H, W]
        connectivity: Connectivity pattern. Options:
            - 4: 4-neighbor connectivity (horizontal/vertical)
            - 8: 8-neighbor connectivity (includes diagonals)
            - "radius": radius-based connectivity
        radius: Radius for connectivity when connectivity="radius".
            Default 1.5 includes diagonal neighbors.

    Returns:
        Tuple of (node_features, edge_indices, edge_features):
            - node_features: [batch, H*W, channels+2] with grid values and positions
            - edge_indices: [batch, num_edges, 2] connectivity pairs
            - edge_features: [batch, num_edges, 2] relative position vectors
    """
    batch_size, channels, height, width = grid.shape
    num_nodes = height * width

    # Create node features: [batch, H*W, channels + 2]
    # Flatten spatial dimensions while keeping batch and channel dims
    values = grid.reshape(batch_size, channels, num_nodes)  # [B, C, N]
    values = jnp.transpose(values, (0, 2, 1))  # [B, N, C]

    # Create normalized grid positions
    y_coords, x_coords = jnp.meshgrid(
        jnp.linspace(0, 1, height),
        jnp.linspace(0, 1, width),
        indexing="ij",
    )
    positions = jnp.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)  # [N, 2]
    positions = jnp.broadcast_to(positions, (batch_size, num_nodes, 2))  # [B, N, 2]

    # Concatenate values and positions
    node_features = jnp.concatenate([values, positions], axis=-1)  # [B, N, C+2]

    # Create edge connectivity based on pattern
    edge_indices = _create_edge_indices(height, width, connectivity, radius)

    # Broadcast to batch dimension
    num_edges = edge_indices.shape[0]
    edge_indices = jnp.broadcast_to(
        edge_indices[None, :, :], (batch_size, num_edges, 2)
    )

    # Compute edge features (relative positions)
    # For each edge (src, dst), edge_feature = pos[dst] - pos[src]
    edge_features = _compute_edge_features(positions, edge_indices)

    return node_features, edge_indices, edge_features


def _create_edge_indices(
    height: int,
    width: int,
    connectivity: int | Literal["radius"],
    radius: float,
) -> jax.Array:
    """Create edge index array for grid connectivity.

    Args:
        height: Grid height
        width: Grid width
        connectivity: 4, 8, or "radius"
        radius: Radius for radius-based connectivity

    Returns:
        Edge indices of shape [num_edges, 2]
    """
    # Define neighbor offsets based on connectivity
    if connectivity == 4:
        # Right, Down only (bidirectional added later)
        offsets = [(0, 1), (1, 0)]
    elif connectivity == 8:
        # Right, Down, Right-Down, Left-Down
        offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    elif connectivity == "radius":
        # Compute all pairs within radius
        return _create_radius_edges(height, width, radius)
    else:
        msg = f"Invalid connectivity: {connectivity}. Use 4, 8, or 'radius'."
        raise ValueError(msg)

    # Build edge list
    edges_src = []
    edges_dst = []

    for dy, dx in offsets:
        for i in range(height):
            for j in range(width):
                ni, nj = i + dy, j + dx
                if 0 <= ni < height and 0 <= nj < width:
                    src = i * width + j
                    dst = ni * width + nj
                    # Add both directions for undirected graph
                    edges_src.extend([src, dst])
                    edges_dst.extend([dst, src])

    return jnp.array([edges_src, edges_dst]).T  # [num_edges, 2]


def _create_radius_edges(height: int, width: int, radius: float) -> jax.Array:
    """Create edges based on radius threshold.

    Args:
        height: Grid height
        width: Grid width
        radius: Distance threshold for edge creation

    Returns:
        Edge indices of shape [num_edges, 2]
    """
    # Create grid positions (using grid spacing of 1)
    y_coords, x_coords = jnp.meshgrid(
        jnp.arange(height), jnp.arange(width), indexing="ij"
    )
    positions = jnp.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)  # [N, 2]

    n_nodes = height * width
    edges_src = []
    edges_dst = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = jnp.linalg.norm(positions[i] - positions[j])
            if dist <= radius:
                edges_src.extend([i, j])
                edges_dst.extend([j, i])

    return jnp.array([edges_src, edges_dst]).T


def _compute_edge_features(
    positions: jax.Array,
    edge_indices: jax.Array,
) -> jax.Array:
    """Compute edge features as relative position vectors.

    Args:
        positions: Node positions [batch, num_nodes, 2]
        edge_indices: Edge connectivity [batch, num_edges, 2]

    Returns:
        Edge features [batch, num_edges, 2] as (dst_pos - src_pos)
    """

    def compute_single_batch(pos, edges):
        """Compute edge features for a single batch."""
        src_indices = edges[:, 0]
        dst_indices = edges[:, 1]

        src_pos = pos[src_indices]  # [num_edges, 2]
        dst_pos = pos[dst_indices]  # [num_edges, 2]

        return dst_pos - src_pos

    # Use vmap over batch dimension
    return jax.vmap(compute_single_batch)(positions, edge_indices)


def graph_to_grid(
    node_features: jax.Array,
    height: int,
    width: int,
    channels: int | None = None,
) -> jax.Array:
    """Convert graph node features back to 2D grid format.

    Args:
        node_features: Node features [batch, H*W, node_dim]
        height: Original grid height
        width: Original grid width
        channels: Number of value channels (default: node_dim - 2 for positions)

    Returns:
        Grid of shape [batch, channels, H, W]
    """
    batch_size, _, node_dim = node_features.shape

    if channels is None:
        channels = node_dim - 2  # Subtract x, y positions

    # Extract only value channels (exclude positions)
    values = node_features[:, :, :channels]  # [B, N, C]

    # Transpose and reshape to grid
    values = jnp.transpose(values, (0, 2, 1))  # [B, C, N]
    return values.reshape(batch_size, channels, height, width)
