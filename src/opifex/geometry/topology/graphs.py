"""Graph topology structures.

This module provides the :class:`GraphTopology` container used to describe
irregular graph structures (nodes, edges, adjacency/degree/Laplacian) common in
scientific applications. Learnable graph operators live in
``opifex.neural.operators.graph``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from jaxtyping import Float, Int


class GraphTopology:
    """Basic graph structure for neural operations.

    Represents graphs with nodes, edges, and optional features for
    graph neural network computations.
    """

    def __init__(
        self,
        nodes: Float[jax.Array, "n d"],
        edges: Int[jax.Array, "e 2"],
        edge_features: Float[jax.Array, "e f"] | None = None,
        adjacency_matrix: Float[jax.Array, "n n"] | None = None,
    ) -> None:
        """Initialize graph topology.

        Args:
            nodes: Node features [num_nodes, node_dim]
            edges: Edge connectivity [num_edges, 2]
            edge_features: Optional edge features [num_edges, edge_dim]
            adjacency_matrix: Optional adjacency matrix [num_nodes, num_nodes]
        """
        self.nodes = nodes
        self.edges = edges
        self.edge_features = edge_features

        # Compute adjacency matrix if not provided
        if adjacency_matrix is None:
            self.adjacency_matrix = self._compute_adjacency()
        else:
            self.adjacency_matrix = adjacency_matrix

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.nodes.shape[0]

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.edges.shape[0]

    @property
    def node_dim(self) -> int:
        """Dimension of node features."""
        return self.nodes.shape[1]

    def _compute_adjacency(self) -> Float[jax.Array, "n n"]:
        """Compute adjacency matrix from edge list."""
        n = self.num_nodes
        adj = jnp.zeros((n, n))

        # Set adjacency entries to 1 for connected nodes
        adj = adj.at[self.edges[:, 0], self.edges[:, 1]].set(1.0)
        return adj.at[self.edges[:, 1], self.edges[:, 0]].set(1.0)  # Undirected

    def get_neighbors(self, node_idx: int) -> Int[jax.Array, ...]:
        """Get neighbor nodes for a given node."""
        return jnp.where(self.adjacency_matrix[node_idx] > 0)[0]

    def degree_matrix(self) -> Float[jax.Array, ...]:
        """Compute degree matrix (diagonal matrix of node degrees)."""
        degrees = jnp.sum(self.adjacency_matrix, axis=1)
        return jnp.diag(degrees)

    def laplacian_matrix(self, normalized: bool = True) -> Float[jax.Array, "n n"]:
        """Compute graph Laplacian matrix.

        Args:
            normalized: Whether to compute normalized Laplacian

        Returns:
            Laplacian matrix
        """
        D = self.degree_matrix()
        laplacian = D - self.adjacency_matrix

        # Use JAX-compatible conditional
        def compute_normalized_laplacian():
            """Return the symmetric normalised Laplacian ``I - D^{-1/2} A D^{-1/2}``."""
            # Compute D^(-1/2)
            degrees = jnp.diag(D)
            D_inv_sqrt = jnp.diag(jnp.where(degrees > 0, 1.0 / jnp.sqrt(degrees), 0.0))
            return D_inv_sqrt @ laplacian @ D_inv_sqrt

        return jnp.where(normalized, compute_normalized_laplacian(), laplacian)

    @classmethod
    def from_molecular_system(
        cls, atomic_coords: Float[jax.Array, "n 3"], cutoff_radius: float = 5.0
    ) -> GraphTopology:
        """Create graph from molecular system with distance-based connectivity.

        Args:
            atomic_coords: Atomic coordinates [num_atoms, 3]
            cutoff_radius: Distance cutoff for edge creation

        Returns:
            GraphTopology with distance-based edges
        """
        # Compute pairwise distances
        diff = atomic_coords[:, None, :] - atomic_coords[None, :, :]
        distances = jnp.linalg.norm(diff, axis=-1)

        # Create adjacency matrix based on cutoff
        adjacency = (distances < cutoff_radius) & (distances > 0)

        # Convert to edge list
        edge_indices = jnp.where(adjacency)
        edges = jnp.column_stack([edge_indices[0], edge_indices[1]])

        return cls(
            nodes=atomic_coords,
            edges=edges,
            adjacency_matrix=jnp.asarray(adjacency),
        )
