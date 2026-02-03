"""Latent Neural Operator implementation.

This module contains the LatentNeuralOperator class that uses attention-based
latent representations for efficient function space learning.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

# Import PhysicsAwareAttention from physics module
from opifex.neural.operators.physics.attention import PhysicsAwareAttention


class LatentNeuralOperator(nnx.Module):
    """Latent Neural Operator with attention-based latent representations.

    This operator learns compact latent representations of function spaces
    using attention mechanisms, enabling efficient learning of complex
    operator mappings with reduced computational overhead.

    Features:
    - Learnable latent space for function representation
    - Multi-head attention for function-to-latent and latent-to-function mappings
    - Physics-aware attention constraints
    - Efficient inference through latent space operations
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        num_latent_tokens: int,
        *,
        num_attention_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        physics_constraints: list[str] | None = None,
        dropout_rate: float = 0.0,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ):
        """Initialize Latent Neural Operator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            latent_dim: Dimension of latent space
            num_latent_tokens: Number of latent tokens
            num_attention_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            physics_constraints: List of physics constraints
            dropout_rate: Dropout rate
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.physics_constraints = physics_constraints or []
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Learnable latent tokens
        self.latent_tokens = nnx.Param(
            jax.random.normal(rngs.params(), (num_latent_tokens, latent_dim))
        )

        # Input projection
        self.input_proj = nnx.Linear(
            in_features=in_channels,
            out_features=latent_dim,
            rngs=rngs,
        )

        # Encoder: function to latent
        encoder_layers_temp = []
        for _ in range(num_encoder_layers):
            layer = PhysicsAwareAttention(
                embed_dim=latent_dim,
                num_heads=num_attention_heads,
                physics_constraints=physics_constraints,
                dropout_rate=dropout_rate,
                rngs=rngs,
            )
            encoder_layers_temp.append(layer)
            self.encoder_layers = nnx.List(encoder_layers_temp)

        # Latent processing layers
        self.latent_self_attention = PhysicsAwareAttention(
            embed_dim=latent_dim,
            num_heads=num_attention_heads,
            physics_constraints=physics_constraints,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

        # Decoder: latent to function
        decoder_layers_temp = []
        for _ in range(num_decoder_layers):
            layer = PhysicsAwareAttention(
                embed_dim=latent_dim,
                num_heads=num_attention_heads,
                physics_constraints=physics_constraints,
                dropout_rate=dropout_rate,
                rngs=rngs,
            )
            decoder_layers_temp.append(layer)
        self.decoder_layers = nnx.List(decoder_layers_temp)

        # Output projection
        self.output_proj = nnx.Linear(
            in_features=latent_dim,
            out_features=out_channels,
            rngs=rngs,
        )

        # Layer normalization
        self.encoder_norms = nnx.List(
            [
                nnx.LayerNorm(num_features=latent_dim, rngs=rngs)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_norms = nnx.List(
            [
                nnx.LayerNorm(num_features=latent_dim, rngs=rngs)
                for _ in range(num_decoder_layers)
            ]
        )
        self.latent_norm = nnx.LayerNorm(num_features=latent_dim, rngs=rngs)

        # Dropout
        self.dropout: nnx.Dropout | None
        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def _add_positional_encoding(self, x: jax.Array) -> jax.Array:
        """Add positional encoding to input."""
        _, seq_len, embed_dim = x.shape

        # Create positional encoding
        positions = jnp.arange(seq_len)
        div_term = jnp.exp(
            jnp.arange(0, embed_dim, 2) * (-jnp.log(10000.0) / embed_dim)
        )

        pe = jnp.zeros((seq_len, embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(positions[:, None] * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(positions[:, None] * div_term))

        return x + pe[None, :, :]

    def __call__(
        self,
        x: jax.Array,
        *,
        physics_info: jax.Array | None = None,
        training: bool = False,
    ) -> jax.Array:
        """Apply Latent Neural Operator.

        Args:
            x: Input tensor (batch, channels, *spatial_dims)
            physics_info: Optional physics information
            training: Whether in training mode

        Returns:
            Output tensor (batch, out_channels, *spatial_dims)
        """
        batch_size = x.shape[0]
        original_shape = x.shape

        # Flatten spatial dimensions
        x_flat = x.reshape(batch_size, self.in_channels, -1).transpose(0, 2, 1)

        # Input projection
        x_embedded = self.input_proj(x_flat)
        x_embedded = self._add_positional_encoding(x_embedded)

        # Prepare latent tokens
        latent_tokens = jnp.tile(self.latent_tokens[None, :, :], (batch_size, 1, 1))

        # Encoder: function to latent
        current_latent = latent_tokens
        for _i, (encoder_layer, norm) in enumerate(
            zip(self.encoder_layers, self.encoder_norms, strict=False)
        ):
            # Cross-attention: latent queries, function keys/values
            attended_latent = encoder_layer(
                jnp.concatenate([current_latent, x_embedded], axis=1),
                physics_info=physics_info,
                training=training,
            )[:, : self.num_latent_tokens, :]

            # Residual connection and normalization
            current_latent = norm(current_latent + attended_latent)

            if self.dropout is not None and training:
                current_latent = self.dropout(current_latent)

        # Latent self-attention
        processed_latent = self.latent_self_attention(
            current_latent, physics_info=physics_info, training=training
        )
        processed_latent = self.latent_norm(current_latent + processed_latent)

        if self.dropout is not None and training:
            processed_latent = self.dropout(processed_latent)

        # Decoder: latent to function
        current_output = x_embedded
        for _i, (decoder_layer, norm) in enumerate(
            zip(self.decoder_layers, self.decoder_norms, strict=False)
        ):
            # Cross-attention: function queries, latent keys/values
            attended_output = decoder_layer(
                jnp.concatenate([current_output, processed_latent], axis=1),
                physics_info=physics_info,
                training=training,
            )[:, : x_embedded.shape[1], :]

            # Residual connection and normalization
            current_output = norm(current_output + attended_output)

            if self.dropout is not None and training:
                current_output = self.dropout(current_output)

        # Output projection and reshape back to original spatial dimensions
        return (
            self.output_proj(current_output)
            .transpose(0, 2, 1)
            .reshape(batch_size, self.out_channels, *original_shape[2:])
        )
