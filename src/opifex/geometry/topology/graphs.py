"""Graph neural networks and topological structures.

This module implements graph-based neural networks for learning on
irregular topological structures common in scientific applications.
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
    ):
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


def linear_layer(x: jax.Array, weights: jax.Array, bias: jax.Array) -> jax.Array:
    """Apply linear transformation."""
    return x @ weights + bias


def gelu_activation(x: jax.Array) -> jax.Array:
    """Apply GELU activation function."""
    return jax.nn.gelu(x)


class GraphMessagePassing:
    """Message passing layer for graph neural networks.

    Pure JAX implementation for better type compatibility.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        output_dim: int,
        key: jax.Array,
    ):
        """Initialize message passing layer.

        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for message computation
            output_dim: Output dimension for node features
            key: JAX random key for parameter initialization
        """
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize parameters
        key1, key2, key3, key4 = jax.random.split(key, 4)

        # Message network parameters
        input_dim = 2 * node_dim + edge_dim
        self.msg_w1 = jax.random.normal(key1, (input_dim, hidden_dim)) * 0.1
        self.msg_b1 = jnp.zeros(hidden_dim)
        self.msg_w2 = jax.random.normal(key2, (hidden_dim, hidden_dim)) * 0.1
        self.msg_b2 = jnp.zeros(hidden_dim)

        # Update network parameters
        update_input_dim = node_dim + hidden_dim
        self.upd_w1 = jax.random.normal(key3, (update_input_dim, hidden_dim)) * 0.1
        self.upd_b1 = jnp.zeros(hidden_dim)
        self.upd_w2 = jax.random.normal(key4, (hidden_dim, output_dim)) * 0.1
        self.upd_b2 = jnp.zeros(output_dim)

    def __call__(
        self,
        nodes: Float[jax.Array, "n d"],
        edges: Int[jax.Array, "e 2"],
        edge_features: Float[jax.Array, "e f"] | None = None,
    ) -> Float[jax.Array, "n d"]:
        """Forward pass of message passing.

        Args:
            nodes: Node features [num_nodes, node_dim]
            edges: Edge connectivity [num_edges, 2]
            edge_features: Edge features [num_edges, edge_dim]

        Returns:
            Updated node features [num_nodes, node_dim]
        """
        num_nodes = nodes.shape[0]

        # Get source and target node features
        src_nodes = nodes[edges[:, 0]]  # [num_edges, node_dim]
        dst_nodes = nodes[edges[:, 1]]  # [num_edges, node_dim]

        # Prepare message input
        if edge_features is not None:
            message_input = jnp.concatenate(
                [src_nodes, dst_nodes, edge_features], axis=-1
            )
        else:
            # Use zero edge features if not provided
            zero_edges = jnp.zeros((edges.shape[0], self.edge_dim))
            message_input = jnp.concatenate([src_nodes, dst_nodes, zero_edges], axis=-1)

        # Compute messages using simple linear layers
        h1 = gelu_activation(linear_layer(message_input, self.msg_w1, self.msg_b1))
        messages = linear_layer(h1, self.msg_w2, self.msg_b2)

        # Aggregate messages at destination nodes
        aggregated = jnp.zeros((num_nodes, self.hidden_dim))
        aggregated = aggregated.at[edges[:, 1]].add(messages)

        # Update node features
        update_input = jnp.concatenate([nodes, aggregated], axis=-1)
        h2 = gelu_activation(linear_layer(update_input, self.upd_w1, self.upd_b1))
        return linear_layer(h2, self.upd_w2, self.upd_b2)


class GraphNeuralOperator:
    """Graph Neural Operator for function approximation on graphs.

    Pure JAX implementation combining multiple message passing layers.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        num_layers: int = 3,
        key: jax.Array | None = None,
    ):
        """Initialize graph neural operator.

        Args:
            node_dim: Input node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension for processing
            output_dim: Output dimension (defaults to node_dim)
            num_layers: Number of message passing layers
            key: JAX random key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else node_dim
        self.num_layers = num_layers

        # Split keys for different components
        keys = jax.random.split(key, num_layers + 2)

        # Initialize input projection
        self.input_w = jax.random.normal(keys[0], (node_dim, hidden_dim)) * 0.1
        self.input_b = jnp.zeros(hidden_dim)

        # Initialize message passing layers
        self.mp_layers = [
            GraphMessagePassing(
                hidden_dim, edge_dim, hidden_dim, hidden_dim, keys[i + 1]
            )
            for i in range(num_layers)
        ]

        # Initialize output projection
        self.output_w = jax.random.normal(keys[-1], (hidden_dim, self.output_dim)) * 0.1
        self.output_b = jnp.zeros(self.output_dim)

    def __call__(
        self,
        graph: GraphTopology,
        target_function: Float[jax.Array, "n out"] | None = None,
    ) -> Float[jax.Array, "n out"]:
        """Forward pass of graph neural operator.

        Args:
            graph: Input graph topology with node features
            target_function: Target function values (for supervised learning)

        Returns:
            Predicted function values at nodes
        """
        # Project input features
        node_features = linear_layer(graph.nodes, self.input_w, self.input_b)

        # Apply message passing layers sequentially
        # Note: Cannot use jax.lax.scan with Python objects, so use a simple loop
        for mp_layer in self.mp_layers:
            updated_features = mp_layer(node_features, graph.edges, graph.edge_features)
            # Residual connection
            node_features = updated_features + linear_layer(
                graph.nodes, self.input_w, self.input_b
            )

        # Output projection
        return linear_layer(node_features, self.output_w, self.output_b)


# JAX pytree registration for GraphTopology
def _graph_topology_tree_flatten(graph):
    children = (graph.nodes, graph.edges, graph.edge_features, graph.adjacency_matrix)
    aux_data = None
    return children, aux_data


def _graph_topology_tree_unflatten(aux_data, children):
    nodes, edges, edge_features, adjacency_matrix = children
    return GraphTopology(nodes, edges, edge_features, adjacency_matrix)


jax.tree_util.register_pytree_node(
    GraphTopology, _graph_topology_tree_flatten, _graph_topology_tree_unflatten
)
