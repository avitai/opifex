"""Adaptive DeepONet variants.

Adaptive Deep Operator Networks with dynamic architecture adjustment
and multi-resolution capabilities for complex operator learning tasks.

This module provides adaptive DeepONet architectures that can adjust
their behavior based on problem complexity following FLAX NNX patterns
and critical technical guidelines.
"""

import jax
import jax.numpy as jnp
from flax import nnx

# Import required components from other modules
from opifex.neural.base import StandardMLP


class AdaptiveDeepONet(nnx.Module):
    """Adaptive DeepONet with dynamic architecture adjustment.

    This variant can adapt its architecture based on problem complexity
    and provides multiple resolution levels for different accuracy requirements.
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        base_latent_dim: int,
        *,
        num_resolution_levels: int = 3,
        adaptive_latent_scaling: bool = True,
        use_residual_connections: bool = True,
        activation: str = "tanh",
        rngs: nnx.Rngs,
    ):
        """Initialize Adaptive DeepONet.

        Args:
            branch_input_dim: Branch network input dimension
            trunk_input_dim: Trunk network input dimension
            base_latent_dim: Base latent dimension (scaled for different levels)
            num_resolution_levels: Number of resolution levels
            adaptive_latent_scaling: Whether to scale latent dimensions adaptively
            use_residual_connections: Whether to use residual connections
            activation: Activation function name
            rngs: Random number generators
        """
        super().__init__()
        self.base_latent_dim = base_latent_dim
        self.num_resolution_levels = num_resolution_levels
        self.adaptive_latent_scaling = adaptive_latent_scaling
        self.use_residual_connections = use_residual_connections

        # Create multi-resolution branch networks
        branch_networks_temp = []
        trunk_networks_temp = []

        for level in range(num_resolution_levels):
            # Scale latent dimension based on resolution level
            if adaptive_latent_scaling:
                latent_dim = base_latent_dim * (2**level)
            else:
                latent_dim = base_latent_dim

            # Scale hidden dimensions for complexity
            branch_layer_sizes = [
                branch_input_dim,
                64 * (2**level),
                128 * (2**level),
                latent_dim,
            ]
            trunk_layer_sizes = [
                trunk_input_dim,
                64 * (2**level),
                128 * (2**level),
                latent_dim,
            ]

            # Create networks for this resolution level using StandardMLP
            branch_net = StandardMLP(
                layer_sizes=branch_layer_sizes,
                activation=activation,
                dropout_rate=0.0,
                use_bias=True,
                apply_final_dropout=False,
                rngs=rngs,
            )

            trunk_net = StandardMLP(
                layer_sizes=trunk_layer_sizes,
                activation=activation,
                dropout_rate=0.0,
                use_bias=True,
                apply_final_dropout=False,
                rngs=rngs,
            )

            branch_networks_temp.append(branch_net)
            trunk_networks_temp.append(trunk_net)

        self.branch_networks = nnx.List(branch_networks_temp)
        self.trunk_networks = nnx.List(trunk_networks_temp)

        # Adaptive weighting network
        self.weight_predictor = nnx.Sequential(
            nnx.Linear(branch_input_dim, 64, rngs=rngs),
            nnx.tanh,
            nnx.Linear(64, 32, rngs=rngs),
            nnx.tanh,
            nnx.Linear(32, num_resolution_levels, rngs=rngs),
            nnx.softmax,
        )

        # Optional residual connection networks
        if use_residual_connections:
            residual_networks_temp = []
            for level in range(num_resolution_levels - 1):
                # Connect lower resolution to higher resolution
                if adaptive_latent_scaling:
                    lower_dim = base_latent_dim * (2**level)
                    higher_dim = base_latent_dim * (2 ** (level + 1))
                else:
                    lower_dim = higher_dim = base_latent_dim

                residual_net = nnx.Linear(lower_dim, higher_dim, rngs=rngs)
                residual_networks_temp.append(residual_net)

            self.residual_networks = nnx.List(residual_networks_temp)

    def __call__(
        self,
        branch_input: jax.Array,
        trunk_input: jax.Array,
        *,
        resolution_level: int | None = None,
        adaptive_weights: bool = True,
        deterministic: bool = True,
    ) -> jax.Array:
        """Apply Adaptive DeepONet.

        Args:
            branch_input: Function values (batch, branch_input_dim)
            trunk_input: Query coordinates (batch, num_locations, trunk_input_dim)
            resolution_level: Specific resolution level to use (optional)
            adaptive_weights: Whether to use adaptive weighting
            deterministic: Whether to use deterministic mode

        Returns:
            Function values at query locations (batch, num_locations)
        """
        batch_size, num_locations, trunk_dim = trunk_input.shape
        trunk_input_flat = trunk_input.reshape(batch_size * num_locations, trunk_dim)

        # If specific resolution level is requested
        if resolution_level is not None:
            level = min(resolution_level, self.num_resolution_levels - 1)
            branch_encoding = self.branch_networks[level](
                branch_input, deterministic=deterministic
            )
            trunk_encoding_flat = self.trunk_networks[level](
                trunk_input_flat, deterministic=deterministic
            )

            trunk_encoding = trunk_encoding_flat.reshape(batch_size, num_locations, -1)

            return jnp.sum(branch_encoding[:, None, :] * trunk_encoding, axis=-1)

        # Multi-resolution computation with adaptive weighting
        outputs: list[jax.Array] = []
        branch_encodings: list[jax.Array] = []

        for level in range(self.num_resolution_levels):
            branch_encoding = self.branch_networks[level](
                branch_input, deterministic=deterministic
            )
            trunk_encoding_flat = self.trunk_networks[level](
                trunk_input_flat, deterministic=deterministic
            )

            # Apply residual connections
            if self.use_residual_connections and level > 0:
                residual = self.residual_networks[level - 1](branch_encodings[-1])
                branch_encoding = branch_encoding + residual

            branch_encodings.append(branch_encoding)

            trunk_encoding = trunk_encoding_flat.reshape(batch_size, num_locations, -1)

            # Compute output for this level
            level_output = jnp.sum(
                branch_encoding[:, None, :] * trunk_encoding, axis=-1
            )
            outputs.append(level_output)

        # Adaptive weighting
        if adaptive_weights:
            weights = self.weight_predictor(branch_input)  # (batch, num_levels)
            weights = weights[:, :, None]  # (batch, num_levels, 1)

            # Weighted combination of outputs
            stacked_outputs = jnp.stack(
                outputs, axis=1
            )  # (batch, num_levels, num_locations)
            final_output = jnp.sum(weights * stacked_outputs, axis=1)
        else:
            # Simple averaging
            stacked_outputs = jnp.stack(outputs, axis=1)
            final_output = jnp.mean(stacked_outputs, axis=1)

        return final_output
