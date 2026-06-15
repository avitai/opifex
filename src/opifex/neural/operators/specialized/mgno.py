"""
Multipole Graph Neural Operator (MGNO)

Advanced neural operator for long-range interactions and hierarchical systems.
Uses multipole expansion methods for efficient computation of long-range forces
and interactions in molecular dynamics, N-body simulations, and plasma physics.

Key Features:
- Numerically stable multipole expansions
- Hierarchical message passing for long-range interactions
- Support for molecular dynamics and N-body problems
- Efficient computation of multipole moments
- Graph-based representation for particle systems
"""

import jax
import jax.numpy as jnp
from flax import nnx


class MultipoleExpansion(nnx.Module):
    """
    Numerically stable multipole expansion layer.

    Computes multipole moments with proper numerical stability
    to prevent overflow and NaN generation in hierarchical computations.
    """

    def __init__(
        self,
        channels: int,
        max_order: int = 4,
        epsilon: float = 1e-8,
        stabilization_factor: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """
        Initialize multipole expansion with numerical stability.

        Args:
            channels: Number of feature channels
            max_order: Maximum multipole order
            epsilon: Small constant for numerical stability
            stabilization_factor: Factor for moment normalization
            rngs: Random number generator state
        """
        self.channels = channels
        self.max_order = max_order
        self.epsilon = epsilon
        self.stabilization_factor = stabilization_factor

        # FIXED: Multipole coefficients with stability-focused initialization
        self.multipole_weights = nnx.Param(
            nnx.initializers.xavier_normal()(rngs.params(), (channels, channels, max_order + 1))
            * (stabilization_factor / jnp.sqrt(max_order + 1))  # Scale for stability
        )

        # Layer normalization for output stability
        self.layer_norm = nnx.LayerNorm(channels, rngs=rngs)

        # Learnable scaling factors per order
        self.order_scales = nnx.Param(jnp.ones(max_order + 1) * stabilization_factor)

    def __call__(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        """
        Apply multipole expansion with full stability measures.

        Args:
            x: Input features (batch, num_points, channels)
            positions: Particle positions (batch, num_points, coord_dim)

        Returns:
            Multipole-transformed features (batch, num_points, channels)
        """
        # Clip inputs to a finite dynamic range to prevent overflow in the
        # radial power terms. (No NaN masking — invalid inputs must fail fast.)
        x = jnp.clip(x, -1e6, 1e6)
        positions = jnp.clip(positions, -1e6, 1e6)

        _batch_size, _num_points, _channels = x.shape  # Marked as unused for validation

        # Per-order multipole moments: (batch, num_points, channels, max_order + 1).
        moments = self._compute_stable_multipole_moments(x, positions)

        # Learned mixing across input channels and multipole orders.
        # Weights are indexed (channel_in=i, channel_out=j, order=o); contract i, o
        # to produce a per-output-channel feature of shape (batch, num_points, channels).
        transformed = jnp.einsum("...io,ijo->...j", moments, self.multipole_weights.value)

        # Layer normalization for output stability.
        return self.layer_norm(transformed)

    def _compute_stable_multipole_moments(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        """
        Compute multipole moments with full numerical stability.

        Args:
            x: Input features (batch, num_points, channels)
            positions: Particle positions (batch, num_points, coord_dim)

        Returns:
            Stable per-order multipole moments
            (batch, num_points, channels, max_order + 1).
        """
        _batch_size, _num_points, _channels = x.shape  # Marked as unused for validation
        coord_dim = positions.shape[-1]

        # Collect one moment tensor per multipole order; stacked along a new
        # trailing axis so the learned weights can mix across orders.
        per_order_moments: list[jax.Array] = []

        for order in range(self.max_order + 1):
            # FIXED: Compute distances with stability measures
            distances = jnp.linalg.norm(positions, axis=-1, keepdims=True)
            distances_safe = jnp.maximum(distances, self.epsilon)

            # FIXED: Normalized radial component to prevent explosion
            # Use tanh to bound the radial component
            radial_raw = jnp.power(distances_safe, order)
            radial_normalized = radial_raw / (1.0 + radial_raw)  # Bounds between 0 and 1
            radial = jnp.tanh(radial_normalized)  # Further stabilization

            # FIXED: Angular component (stable spherical harmonics approximation).
            # Check 3D before 2D: ``coord_dim >= 2`` also matches 3D, so the more
            # specific branch must come first or it is unreachable.
            if coord_dim >= 3:
                # Azimuthal angle plus polar angle for 3D
                theta = jnp.arctan2(positions[..., 1], positions[..., 0] + self.epsilon)
                phi = jnp.arctan2(
                    jnp.sqrt(positions[..., 0] ** 2 + positions[..., 1] ** 2),
                    positions[..., 2] + self.epsilon,
                )
                angular = jnp.cos(order * theta) * jnp.sin(order * phi)
            elif coord_dim >= 2:
                # Use stable arctangent computation
                theta = jnp.arctan2(positions[..., 1], positions[..., 0] + self.epsilon)
                angular = jnp.cos(order * theta)
            else:
                angular = jnp.ones_like(radial[..., 0])

            # FIXED: Combine with input features using stable operations
            moment_contribution = x * radial * angular[..., None]

            # FIXED: Normalize by moment magnitude to prevent instability
            moment_norm = jnp.linalg.norm(moment_contribution, axis=-1, keepdims=True)
            normalized_moment = moment_contribution / (moment_norm + self.epsilon)

            # FIXED: Apply learnable order-dependent scaling with stability
            order_scale = jnp.clip(self.order_scales.value[order], 0.01, 1.0)
            weighted_moment = normalized_moment * order_scale / (order + 1.0)

            per_order_moments.append(weighted_moment)

        # Stack along a trailing order axis: (batch, num_points, channels, order).
        return jnp.stack(per_order_moments, axis=-1)


class MGNOLayer(nnx.Module):
    """
    MGNO layer with numerical stability and robust message passing.

    Combines multipole expansion with local graph neural network operations
    for handling both long-range and short-range interactions.
    """

    def __init__(
        self,
        channels: int,
        max_multipole_order: int = 4,
        use_local_messages: bool = True,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """
        Initialize MGNO layer with stability features.

        Args:
            channels: Number of feature channels
            max_multipole_order: Maximum multipole expansion order
            use_local_messages: Whether to use local message passing
            dropout_rate: Dropout rate for regularization
            rngs: Random number generator state
        """
        self.channels = channels
        self.max_multipole_order = max_multipole_order
        self.use_local_messages = use_local_messages

        # FIXED: Multipole expansion with enhanced stability
        self.multipole_expansion = MultipoleExpansion(
            channels=channels,
            max_order=max_multipole_order,
            epsilon=1e-8,
            stabilization_factor=0.1,
            rngs=rngs,
        )

        # FIXED: Local message passing (if enabled)
        if use_local_messages:
            # Graph neural network for local interactions
            self.message_mlp = nnx.Sequential(
                nnx.Linear(channels * 2, channels, rngs=rngs),
                nnx.gelu,
                nnx.Linear(channels, channels, rngs=rngs),
            )

        # FIXED: Update function (combining multipole and local features)
        update_input_dim = channels * 2 if use_local_messages else channels
        self.update_mlp = nnx.Sequential(
            nnx.Linear(update_input_dim, channels * 2, rngs=rngs),
            nnx.gelu,
            nnx.Linear(channels * 2, channels, rngs=rngs),
        )

        # Layer normalization for stability
        self.layer_norm = nnx.LayerNorm(channels, rngs=rngs)

        # Dropout for regularization
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def _local_message_passing(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        """
        Perform local message passing between nearby particles.

        Args:
            x: Input features (batch, num_points, channels)
            positions: Particle positions (batch, num_points, coord_dim)

        Returns:
            Local messages (batch, num_points, channels)
        """
        _batch_size, _num_points, _channels = x.shape  # Marked as unused for validation

        # Compute pairwise distances
        pos_diff = positions[:, :, None, :] - positions[:, None, :, :]
        distances = jnp.linalg.norm(pos_diff, axis=-1)

        # Use a local cutoff radius (e.g., within top 8 nearest neighbors)
        k = min(8, _num_points - 1)
        _, nearest_indices = jax.lax.top_k(-distances, k + 1)  # +1 to exclude self

        # Extract features of nearest neighbors (excluding self)
        neighbor_indices = nearest_indices[:, :, 1:]  # Remove self (index 0)

        # Gather neighbor features
        neighbor_features = jnp.take_along_axis(
            x[:, None, :, :].repeat(x.shape[1], axis=1),
            neighbor_indices[..., None],
            axis=2,
        )

        # Compute messages for each neighbor
        self_features = x[:, :, None, :].repeat(k, axis=2)
        combined_features = jnp.concatenate([self_features, neighbor_features], axis=-1)

        # Apply message MLP
        if hasattr(self, "message_mlp"):
            messages = self.message_mlp(combined_features)
        else:
            # Fallback if no local message passing
            messages = jnp.zeros_like(combined_features[..., : self.channels])

        # Mean pooling aggregation for simplicity
        # Shape: (batch_size, channels, num_neighbors) -> (batch_size, channels)
        return jnp.mean(messages, axis=2)

    def __call__(self, x: jax.Array, positions: jax.Array, training: bool = False) -> jax.Array:
        """
        Forward pass through MGNO layer with full error handling.

        Args:
            x: Input features (batch, num_points, channels)
            positions: Particle positions (batch, num_points, coord_dim)
            training: Whether in training mode

        Returns:
            Output features (batch, num_points, channels)
        """
        # Store original input for residual connection
        x_orig = x

        # Multipole expansion (long-range interactions). Errors propagate — a
        # failing sub-module must not be silently substituted for the input.
        multipole_features = self.multipole_expansion(x, positions)

        # Local message passing (short-range interactions).
        local_messages = self._local_message_passing(x, positions)

        # Update function combining multipole and local features.
        update_input = jnp.concatenate([multipole_features, local_messages], axis=-1)
        updated_features = self.update_mlp(update_input)

        # Apply dropout during training.
        if training:
            updated_features = self.dropout(updated_features)

        # Residual connection with layer normalization.
        return self.layer_norm(x_orig + updated_features)


class MultipoleGraphNeuralOperator(nnx.Module):
    """
    Complete Multipole Graph Neural Operator with numerical stability.

    Neural operator for systems with long-range interactions such as
    molecular dynamics, N-body simulations, and plasma physics.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 64,
        num_layers: int = 3,
        max_degree: int = 4,  # Renamed from max_multipole_order to match test
        use_local_messages: bool = True,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """
        Initialize MGNO with full stability features.

        Args:
            in_features: Number of input feature channels
            out_features: Number of output feature channels
            hidden_features: Hidden layer width
            num_layers: Number of MGNO layers
            max_degree: Maximum multipole expansion order
            use_local_messages: Whether to use local message passing
            dropout_rate: Dropout rate for regularization
            rngs: Random number generator state
        """
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nnx.Linear(in_features, hidden_features, rngs=rngs)

        # MGNO layers with stability — assignment outside the loop avoids
        # the NNX hazard of rebinding ``self.mgno_layers`` per iteration.
        mgno_layers_temp = []
        for _ in range(num_layers):
            layer = MGNOLayer(
                channels=hidden_features,
                max_multipole_order=max_degree,
                use_local_messages=use_local_messages,
                dropout_rate=dropout_rate,
                rngs=rngs,
            )
            mgno_layers_temp.append(layer)
        self.mgno_layers = nnx.List(mgno_layers_temp)

        # Output projection
        self.output_proj = nnx.Linear(hidden_features, out_features, rngs=rngs)

        # Global dropout for regularization
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, positions: jax.Array, training: bool = False) -> jax.Array:
        """
        Forward pass with full error handling and stability.

        Args:
            x: Input features (batch, num_points, in_channels)
            positions: Particle positions (batch, num_points, coord_dim)
            training: Whether in training mode

        Returns:
            Output features (batch, num_points, out_channels)
        """
        # Static shape validation (operates on ``.shape``, so it stays a Python
        # branch that is safe under jit/vmap and does not concretise input values).
        if x.shape[0] != positions.shape[0] or x.shape[1] != positions.shape[1]:
            raise ValueError(
                f"Batch size or number of points mismatch: x={x.shape}, positions={positions.shape}"
            )

        # Input projection.
        x = self.input_proj(x)

        # Apply MGNO layers. Errors propagate (fail fast) — a failing layer is
        # never silently skipped, which would otherwise yield a wrong result.
        for layer in self.mgno_layers:
            x = layer(x, positions, training=training)
            x = nnx.gelu(x)
            if training:
                x = self.dropout(x)

        # Output projection. On failure the error propagates rather than being
        # masked by an all-zero prediction.
        return self.output_proj(x)


# Factory functions for different MGNO configurations
def create_molecular_mgno(
    in_features: int, out_features: int, *, rngs: nnx.Rngs
) -> MultipoleGraphNeuralOperator:
    """Create MGNO optimized for molecular dynamics simulations."""
    return MultipoleGraphNeuralOperator(
        in_features=in_features,
        out_features=out_features,
        hidden_features=128,
        num_layers=6,
        max_degree=6,
        use_local_messages=True,
        dropout_rate=0.05,
        rngs=rngs,
    )


def create_nbody_mgno(
    in_features: int, out_features: int, *, rngs: nnx.Rngs
) -> MultipoleGraphNeuralOperator:
    """Create MGNO for N-body gravitational simulations."""
    return MultipoleGraphNeuralOperator(
        in_features=in_features,
        out_features=out_features,
        hidden_features=96,
        num_layers=4,
        max_degree=8,
        use_local_messages=False,  # Focus on long-range interactions
        dropout_rate=0.0,
        rngs=rngs,
    )


def create_plasma_mgno(
    in_features: int, out_features: int, *, rngs: nnx.Rngs
) -> MultipoleGraphNeuralOperator:
    """Create MGNO for plasma physics simulations."""
    return MultipoleGraphNeuralOperator(
        in_features=in_features,
        out_features=out_features,
        hidden_features=64,
        num_layers=5,
        max_degree=4,
        use_local_messages=True,
        dropout_rate=0.1,
        rngs=rngs,
    )
