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


class MollifiedGNO(nnx.Module):
    """Mollified Graph Neural Operator (mGNO).

    Extends GraphNeuralOperator with a Gaussian mollifier (smoothing kernel)
    that averages neighbor contributions weighted by spatial proximity.
    Reduces error on irregular point clouds by smoothing high-frequency
    artifacts from message passing.

    The smoothing kernel is:
        w(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))

    where sigma = smoothing_radius. Applied as a post-processing step
    after the base GNO forward pass.

    Reference: Mollified Graph Neural Operators for irregular meshes.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        smoothing_radius: float = 0.1,
        *,
        edge_dim: int = 0,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ):
        """Initialize MollifiedGNO.

        Args:
            node_dim: Dimension of node features.
            hidden_dim: Hidden dimension for message passing.
            num_layers: Number of message passing layers.
            smoothing_radius: Gaussian kernel bandwidth (sigma). 0 = no smoothing.
            edge_dim: Dimension of edge features.
            activation: Activation function.
            rngs: Random number generators.
        """
        super().__init__()
        self.smoothing_radius = smoothing_radius

        # Base GNO for message passing
        self.gno = GraphNeuralOperator(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_dim=edge_dim,
            activation=activation,
            rngs=rngs,
        )

    def __call__(
        self,
        node_features: jax.Array,
        edge_indices: jax.Array,
        edge_features: jax.Array | None = None,
        *,
        positions: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass with mollified smoothing.

        Args:
            node_features: Node features [batch, num_nodes, node_dim].
            edge_indices: Edge connectivity [batch, num_edges, 2].
            edge_features: Edge features [batch, num_edges, edge_dim] (optional).
            positions: Node spatial positions [batch, num_nodes, spatial_dim].
                Required when smoothing_radius > 0.

        Returns:
            Smoothed node features [batch, num_nodes, node_dim].
        """
        # Base GNO forward pass
        h = self.gno(node_features, edge_indices, edge_features)

        # Apply mollifier if radius > 0 and positions are provided
        if self.smoothing_radius > 0 and positions is not None:
            h = self._apply_mollifier(h, positions)

        return h

    def _apply_mollifier(
        self,
        features: jax.Array,
        positions: jax.Array,
    ) -> jax.Array:
        """Apply Gaussian mollifier smoothing.

        Computes weighted average of features based on spatial proximity:
            h_i = sum_j w(x_i, x_j) * h_j / sum_j w(x_i, x_j)

        Args:
            features: Node features [batch, num_nodes, node_dim].
            positions: Node positions [batch, num_nodes, spatial_dim].

        Returns:
            Smoothed features [batch, num_nodes, node_dim].
        """
        sigma = self.smoothing_radius

        def smooth_batch(feats, pos):
            # Compute pairwise squared distances [num_nodes, num_nodes]
            diff = pos[:, None, :] - pos[None, :, :]  # [N, N, D]
            sq_dist = jnp.sum(diff**2, axis=-1)  # [N, N]

            # Gaussian kernel weights
            weights = jnp.exp(-sq_dist / (2.0 * sigma**2 + 1e-10))  # [N, N]

            # Normalize weights per node
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-10)

            # Weighted average of features
            return weights @ feats  # [N, node_dim]

        return jax.vmap(smooth_batch)(features, positions)
