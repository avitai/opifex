"""Graph neural operators for irregular domains.

This module implements graph neural operators for learning operators on
irregular geometries and unstructured meshes.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx


class GraphNeuralOperator(nnx.Module):
    """Graph Neural Operator for learning operators on irregular domains.

    Implements message passing neural networks with geometric awareness for
    learning operators on graph-structured data. Suitable for irregular meshes,
    molecular systems, and other graph-based scientific computing applications.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        edge_dim: int = 0,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ):
        """Initialize Graph Neural Operator.

        Args:
            node_dim: Dimension of node features
            hidden_dim: Hidden dimension for message passing
            num_layers: Number of message passing layers
            edge_dim: Dimension of edge features (0 for no edge features)
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.activation = activation

        # Input projection to hidden dimension
        self.input_projection = nnx.Linear(
            in_features=node_dim,
            out_features=hidden_dim,
            rngs=rngs,
        )

        # Message passing layers
        self.message_passing_layers = nnx.List(
            [
                MessagePassingLayer(
                    node_dim=hidden_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection back to node dimension
        self.output_projection = nnx.Linear(
            in_features=hidden_dim,
            out_features=node_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        node_features: jax.Array,
        edge_indices: jax.Array,
        edge_features: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass of Graph Neural Operator.

        Args:
            node_features: Node features [batch, num_nodes, node_dim]
            edge_indices: Edge connectivity [batch, num_edges, 2]
            edge_features: Edge features [batch, num_edges, edge_dim] (optional)

        Returns:
            Updated node features [batch, num_nodes, node_dim]
        """
        # Project to hidden dimension
        h = self.input_projection(node_features)

        # Apply message passing layers with residual connections
        for layer in self.message_passing_layers:
            h_new = layer(h, edge_indices, edge_features)
            h = h + h_new  # Residual connection

        # Project back to original node dimension
        return self.output_projection(h)


class MessagePassingLayer(nnx.Module):
    """Message passing layer for graph neural networks.

    Implements the message passing paradigm:
    1. Compute messages between connected nodes
    2. Aggregate messages at each node
    3. Update node features based on aggregated messages
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        *,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ):
        """Initialize message passing layer.

        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for message computation
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # Message network: processes [source_node, target_node, edge_features]
        message_input_dim = 2 * node_dim + edge_dim
        self.message_net = nnx.Sequential(
            nnx.Linear(
                in_features=message_input_dim,
                out_features=hidden_dim,
                rngs=rngs,
            ),
            activation,
            nnx.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
                rngs=rngs,
            ),
        )

        # Update network: processes [node_features, aggregated_messages]
        update_input_dim = node_dim + hidden_dim
        self.update_net = nnx.Sequential(
            nnx.Linear(
                in_features=update_input_dim,
                out_features=hidden_dim,
                rngs=rngs,
            ),
            activation,
            nnx.Linear(
                in_features=hidden_dim,
                out_features=node_dim,
                rngs=rngs,
            ),
        )

    def __call__(
        self,
        node_features: jax.Array,
        edge_indices: jax.Array,
        edge_features: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass of message passing layer.

        Args:
            node_features: Node features [batch, num_nodes, node_dim]
            edge_indices: Edge connectivity [batch, num_edges, 2]
            edge_features: Edge features [batch, num_edges, edge_dim] (optional)

        Returns:
            Updated node features [batch, num_nodes, node_dim]
        """
        batch_size, num_nodes, _ = node_features.shape
        _, num_edges, _ = edge_indices.shape

        # Vectorized message passing using advanced indexing
        def process_batch(batch_idx):
            nodes = node_features[batch_idx]  # [num_nodes, node_dim]
            edges = edge_indices[batch_idx]  # [num_edges, 2]

            # Get source and target node features
            src_nodes = nodes[edges[:, 0]]  # [num_edges, node_dim]
            dst_nodes = nodes[edges[:, 1]]  # [num_edges, node_dim]

            # Prepare message input
            if edge_features is not None:
                edge_feats = edge_features[batch_idx]  # [num_edges, edge_dim]
                message_input = jnp.concatenate(
                    [src_nodes, dst_nodes, edge_feats], axis=-1
                )
            else:
                # Use zero edge features if not provided
                zero_edges = jnp.zeros((num_edges, self.edge_dim))
                message_input = jnp.concatenate(
                    [src_nodes, dst_nodes, zero_edges], axis=-1
                )

            # Compute messages
            messages = self.message_net(message_input)  # [num_edges, hidden_dim]

            # Aggregate messages at destination nodes
            aggregated = jnp.zeros((num_nodes, self.hidden_dim))
            aggregated = aggregated.at[edges[:, 1]].add(messages)

            # Update node features
            update_input = jnp.concatenate([nodes, aggregated], axis=-1)
            return self.update_net(update_input)

        # Process all batches
        return jax.vmap(process_batch)(jnp.arange(batch_size))
