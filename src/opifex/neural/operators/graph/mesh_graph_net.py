"""MeshGraphNet for mesh-based simulation.

Implements the encoder-processor-decoder architecture from DeepMind's
"Learning Mesh-Based Simulation with Graph Neural Networks" (Pfaff et al., 2021).

The architecture:
  - **Encoder**: Projects node and edge features to a shared hidden dimension.
  - **Processor**: A stack of MessagePassingLayers with residual connections.
  - **Decoder**: Projects processed node features to the desired output dimension.

Reuses ``MessagePassingLayer`` from ``gno.py`` for the processor to avoid
duplicating message-passing logic.
"""

from collections.abc import Callable

import jax
from flax import nnx

from opifex.neural.operators.graph.gno import MessagePassingLayer


class MeshGraphNet(nnx.Module):
    """MeshGraphNet for learning mesh-based simulations.

    Encoder-processor-decoder architecture operating on graph-structured
    mesh data.  The processor reuses ``MessagePassingLayer`` so that all
    message-passing logic lives in a single place.

    Args:
        node_input_dim: Dimension of input node features.
        edge_input_dim: Dimension of input edge features (0 for none).
        output_dim: Dimension of output node features.
        hidden_dim: Hidden dimension used throughout the network.
        num_layers: Number of message-passing layers in the processor.
        activation: Activation function.
        rngs: Flax NNX random number generators.

    Raises:
        ValueError: If any dimension or layer count is non-positive.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 6,
        *,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        _validate_config(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

        # --- Encoder ---
        self.node_encoder = nnx.Sequential(
            nnx.Linear(in_features=node_input_dim, out_features=hidden_dim, rngs=rngs),
            activation,
            nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs),
        )

        # Edge encoder maps raw edge features to the hidden dim used by
        # MessagePassingLayer.  When ``edge_input_dim == 0`` the encoder is
        # still created (with input size 0 is invalid, so we gate on it).
        if edge_input_dim > 0:
            self.edge_encoder: nnx.Sequential | None = nnx.Sequential(
                nnx.Linear(
                    in_features=edge_input_dim, out_features=hidden_dim, rngs=rngs
                ),
                activation,
                nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs),
            )
            processor_edge_dim = hidden_dim
        else:
            self.edge_encoder = None
            processor_edge_dim = 0

        # --- Processor (stack of MessagePassingLayers with residuals) ---
        self.processor_layers = nnx.List(
            [
                MessagePassingLayer(
                    node_dim=hidden_dim,
                    edge_dim=processor_edge_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )

        # --- Decoder ---
        self.decoder = nnx.Sequential(
            nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs),
            activation,
            nnx.Linear(in_features=hidden_dim, out_features=output_dim, rngs=rngs),
        )

    def __call__(
        self,
        node_features: jax.Array,
        edge_indices: jax.Array,
        edge_features: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass through encoder-processor-decoder.

        Args:
            node_features: Node features ``[batch, num_nodes, node_input_dim]``.
            edge_indices: Edge connectivity ``[batch, num_edges, 2]``.
            edge_features: Edge features ``[batch, num_edges, edge_input_dim]``
                (optional, ``None`` when ``edge_input_dim == 0``).

        Returns:
            Predicted node features ``[batch, num_nodes, output_dim]``.
        """
        # --- Encode ---
        h_nodes = self.node_encoder(node_features)

        h_edges: jax.Array | None = None
        if self.edge_encoder is not None and edge_features is not None:
            h_edges = self.edge_encoder(edge_features)

        # --- Process ---
        for layer in self.processor_layers:
            h_update = layer(h_nodes, edge_indices, h_edges)
            h_nodes = h_nodes + h_update  # Residual connection

        # --- Decode ---
        return self.decoder(h_nodes)


def _validate_config(
    *,
    node_input_dim: int,
    edge_input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
) -> None:
    """Validate MeshGraphNet constructor arguments.

    Args:
        node_input_dim: Must be > 0.
        edge_input_dim: Must be >= 0.
        output_dim: Must be > 0.
        hidden_dim: Must be > 0.
        num_layers: Must be > 0.

    Raises:
        ValueError: If any constraint is violated.
    """
    if node_input_dim <= 0:
        msg = f"node_input_dim must be positive, got {node_input_dim}"
        raise ValueError(msg)
    if edge_input_dim < 0:
        msg = f"edge_input_dim must be non-negative, got {edge_input_dim}"
        raise ValueError(msg)
    if output_dim <= 0:
        msg = f"output_dim must be positive, got {output_dim}"
        raise ValueError(msg)
    if hidden_dim <= 0:
        msg = f"hidden_dim must be positive, got {hidden_dim}"
        raise ValueError(msg)
    if num_layers <= 0:
        msg = f"num_layers must be positive, got {num_layers}"
        raise ValueError(msg)
