"""Multi-physics enhanced DeepONet implementation.

This module provides the MultiPhysicsDeepONet class, which extends the basic
DeepONet architecture with physics-aware attention, multi-physics coupling,
and sensor optimization for improved operator learning in complex systems.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.physics.attention import PhysicsAwareAttention
from opifex.neural.operators.sensor_optimization import SensorOptimization

# Import base components from other modules
from .base import DeepONet


class MultiPhysicsDeepONet(nnx.Module):
    """Enhanced DeepONet with multi-physics support and attention mechanisms.

    Extends the basic DeepONet architecture with physics-aware attention,
    multi-physics coupling, and sensor optimization for improved operator learning.
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        branch_hidden_dims: list[int],
        trunk_hidden_dims: list[int],
        latent_dim: int,
        *,
        num_physics_systems: int = 1,
        use_attention: bool = True,
        attention_heads: int = 8,
        physics_constraints: list[str] | None = None,
        sensor_optimization: bool = False,
        num_sensors: int | None = None,
        activation: Callable[[jax.Array], jax.Array] = nnx.tanh,
        rngs: nnx.Rngs,
    ):
        """Initialize Multi-Physics DeepONet.

        Args:
            branch_input_dim: Branch network input dimension
            trunk_input_dim: Trunk network input dimension
            branch_hidden_dims: Branch network hidden dimensions
            trunk_hidden_dims: Trunk network hidden dimensions
            latent_dim: Latent dimension for inner product
            num_physics_systems: Number of physics systems to handle
            use_attention: Whether to use physics-aware attention
            attention_heads: Number of attention heads
            physics_constraints: List of physics constraints to enforce
            sensor_optimization: Whether to use sensor optimization
            num_sensors: Number of sensors (required if sensor_optimization=True)
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__()
        self.num_physics_systems = num_physics_systems
        self.use_attention = use_attention
        self.latent_dim = latent_dim
        self.sensor_optimization = sensor_optimization

        # Validate and configure sensor parameters
        num_sensors = self._configure_sensors(
            sensor_optimization, num_sensors, branch_input_dim
        )

        # Store physics constraints for test compatibility
        self.physics_constraints = physics_constraints or []

        # Create physics operators
        self._create_physics_operators(
            num_physics_systems,
            branch_input_dim,
            trunk_input_dim,
            branch_hidden_dims,
            trunk_hidden_dims,
            latent_dim,
            sensor_optimization,
            num_sensors,
            activation,
            rngs,
        )

        # Initialize optional components
        self._initialize_attention(
            use_attention, latent_dim, attention_heads, physics_constraints, rngs
        )
        self._initialize_sensor_optimization(
            sensor_optimization, num_sensors, trunk_input_dim, rngs
        )
        self._initialize_system_coupling(num_physics_systems, latent_dim, rngs)

    def _configure_sensors(
        self, sensor_optimization: bool, num_sensors: int | None, branch_input_dim: int
    ) -> int | None:
        """Configure sensor parameters and validate inputs."""
        if sensor_optimization and num_sensors is None:
            # Auto-configure reasonable default for num_sensors
            return max(32, branch_input_dim // 2)
        return num_sensors

    def _convert_activation_to_string(
        self, activation: Callable[[jax.Array], jax.Array] | str
    ) -> str:
        """Convert activation function to string representation."""
        if callable(activation):
            # Convert common JAX/NNX activation functions to strings
            activation_name = getattr(activation, "__name__", "gelu")
            activation_mapping = {"tanh": "tanh", "relu": "relu", "sigmoid": "sigmoid"}
            return activation_mapping.get(activation_name, "gelu")
        return activation

    def _create_physics_operators(
        self,
        num_physics_systems: int,
        branch_input_dim: int,
        trunk_input_dim: int,
        branch_hidden_dims: list[int],
        trunk_hidden_dims: list[int],
        latent_dim: int,
        sensor_optimization: bool,
        num_sensors: int | None,
        activation: Callable[[jax.Array], jax.Array],
        rngs: nnx.Rngs,
    ) -> None:
        """Create individual DeepONets for each physics system."""
        physics_operators_temp = []

        for _i in range(num_physics_systems):
            # Determine the actual branch input dimension
            if sensor_optimization:
                if num_sensors is None:
                    raise ValueError(
                        "num_sensors should be set by now when "
                        "sensor_optimization is enabled"
                    )
                actual_branch_input_dim = num_sensors
            else:
                actual_branch_input_dim = branch_input_dim

            # Convert old API to new API format
            branch_sizes = [actual_branch_input_dim, *branch_hidden_dims, latent_dim]
            trunk_sizes = [trunk_input_dim, *trunk_hidden_dims, latent_dim]
            activation_str = self._convert_activation_to_string(activation)

            operator = DeepONet(
                branch_sizes=branch_sizes,
                trunk_sizes=trunk_sizes,
                activation=activation_str,
                rngs=rngs,
            )
            physics_operators_temp.append(operator)
            self.physics_operators = nnx.List(physics_operators_temp)

    def _initialize_attention(
        self,
        use_attention: bool,
        latent_dim: int,
        attention_heads: int,
        physics_constraints: list[str] | None,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize physics-aware attention mechanism."""
        if use_attention:
            self.physics_attention = PhysicsAwareAttention(
                embed_dim=latent_dim,
                num_heads=attention_heads,
                physics_constraints=physics_constraints,
                rngs=rngs,
            )

    def _initialize_sensor_optimization(
        self,
        sensor_optimization: bool,
        num_sensors: int | None,
        trunk_input_dim: int,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize sensor optimization if enabled."""
        if sensor_optimization and num_sensors is not None:
            self.sensor_optimizer = SensorOptimization(
                num_sensors=num_sensors,
                spatial_dim=trunk_input_dim,
                rngs=rngs,
            )

    def _initialize_system_coupling(
        self, num_physics_systems: int, latent_dim: int, rngs: nnx.Rngs
    ) -> None:
        """Initialize system coupling weights for multi-physics interaction."""
        if num_physics_systems > 1:
            self.system_coupling = nnx.Linear(
                in_features=latent_dim * num_physics_systems,
                out_features=latent_dim,
                rngs=rngs,
            )

    def _prepare_branch_inputs(
        self, branch_inputs: jax.Array | list[jax.Array]
    ) -> list[jax.Array]:
        """Prepare branch inputs for multiple physics systems."""
        if isinstance(branch_inputs, jax.Array):
            # Single input - replicate for all systems
            return [branch_inputs] * self.num_physics_systems
        # Multiple inputs - assume one per system
        if len(branch_inputs) != self.num_physics_systems:
            raise ValueError(
                f"Expected {self.num_physics_systems} branch inputs, "
                f"got {len(branch_inputs)}"
            )
        return branch_inputs

    def _apply_sensor_optimization(
        self, branch_input_list: list[jax.Array], spatial_coords: jax.Array
    ) -> list[jax.Array]:
        """Apply sensor optimization to branch inputs."""
        if hasattr(self, "sensor_optimizer"):
            optimized_inputs = []
            for branch_input in branch_input_list:
                # Expand branch input for sensor sampling
                if len(branch_input.shape) == 2:  # [batch, features]
                    # Assume features correspond to spatial points
                    batch_size, num_features = branch_input.shape
                    # Reshape to [batch, num_points, 1] for sensor sampling
                    expanded_input = branch_input.reshape(batch_size, num_features, 1)
                    optimized_input = self.sensor_optimizer(
                        expanded_input, spatial_coords
                    )
                    # Flatten back to [batch, num_sensors]
                    optimized_input = optimized_input.reshape(batch_size, -1)
                    optimized_inputs.append(optimized_input)
                else:
                    optimized_inputs.append(
                        self.sensor_optimizer(branch_input, spatial_coords)
                    )
            return optimized_inputs
        return branch_input_list

    def _encode_branch_inputs(self, branch_input_list: list[jax.Array]) -> jax.Array:
        """Encode branch inputs using physics-specific operators."""
        encodings = []
        for i, branch_input in enumerate(branch_input_list):
            # Handle batch dimension properly
            if len(branch_input.shape) == 1:
                # Add batch dimension if missing
                branch_input_list[i] = jnp.expand_dims(branch_input, 0)
            elif len(branch_input.shape) > 2:
                # Flatten if multi-dimensional
                batch_size = branch_input.shape[0]
                branch_input_list[i] = branch_input.reshape(batch_size, -1)

            # Get expected dimension from the branch network
            expected_dim = self.physics_operators[i].branch_net.layers[0].in_features
            actual_dim = branch_input_list[i].shape[-1]

            # If sensor optimization is enabled, the input dimension might be reduced
            if hasattr(self, "sensor_optimizer") and self.sensor_optimization:
                # When sensor optimization is enabled, the input dimension is
                # reduced to num_sensors
                # We need to check if the actual dimension matches either the
                # original expected_dim
                # or the reduced dimension from sensor optimization
                sensor_reduced_dim = self.sensor_optimizer.num_sensors
                if actual_dim not in {expected_dim, sensor_reduced_dim}:
                    raise ValueError(
                        f"Branch input dimension {actual_dim} does not match "
                        f"expected dimension {expected_dim} (original) or "
                        f"{sensor_reduced_dim} (sensor-optimized) for system {i}"
                    )
            # No sensor optimization: check against original expected dimension
            elif actual_dim != expected_dim:
                raise ValueError(
                    f"Branch input dimension {actual_dim} "
                    f"does not match expected dimension {expected_dim} for system {i}"
                )

            encoding = self.physics_operators[i].branch_net(branch_input_list[i])
            encodings.append(encoding)

        # Stack encodings: [batch, num_systems, latent_dim]
        return jnp.stack(encodings, axis=1)

    def _encode_trunk_inputs(self, trunk_input: jax.Array) -> jax.Array:
        """Encode trunk inputs using physics-specific operators."""
        # Handle trunk input shape: expected (batch, trunk_dim) or
        # (batch, num_locations, trunk_dim)
        if len(trunk_input.shape) == 2:
            # Single location per batch: (batch, trunk_dim)
            batch_size, trunk_dim = trunk_input.shape
            num_locations = 1
            trunk_input = trunk_input.reshape(batch_size, num_locations, trunk_dim)
        elif len(trunk_input.shape) == 3:
            # Multiple locations per batch: (batch, num_locations, trunk_dim)
            batch_size, num_locations, trunk_dim = trunk_input.shape
        else:
            raise ValueError(
                f"Expected trunk_input to have 2 or 3 dimensions, "
                f"got {len(trunk_input.shape)}"
            )

        # Flatten for processing: [batch * num_locations, trunk_dim]
        trunk_input_flat = trunk_input.reshape(batch_size * num_locations, trunk_dim)

        # For now, use the first physics operator's trunk network
        # (assumption: shared trunk network across physics systems)
        # TODO: Consider physics-specific trunk networks in future
        expected_dim = self.physics_operators[0].trunk_net.layers[0].in_features

        if trunk_input_flat.shape[-1] != expected_dim:
            raise ValueError(
                f"Trunk input dimension {trunk_input_flat.shape[-1]} "
                f"does not match expected dimension {expected_dim}"
            )

        # Encode using trunk network
        trunk_encoding_flat = self.physics_operators[0].trunk_net(trunk_input_flat)

        # Reshape back: [batch, num_locations, latent_dim]
        return trunk_encoding_flat.reshape(batch_size, num_locations, self.latent_dim)

    def _apply_physics_attention(
        self,
        branch_encoding: jax.Array,
        physics_info: jax.Array | None,
        training: bool,
    ) -> jax.Array:
        """Apply physics-aware attention to branch encodings."""
        if self.use_attention and hasattr(self, "physics_attention"):
            # Apply attention across physics systems
            # branch_encoding: [batch, num_systems, latent_dim]
            batch_size, num_systems, latent_dim = branch_encoding.shape

            # Reshape for attention: [batch * num_systems, 1, latent_dim]
            reshaped_encoding = branch_encoding.reshape(
                batch_size * num_systems, 1, latent_dim
            )

            # Apply attention
            attended_encoding = self.physics_attention(
                reshaped_encoding, physics_info=physics_info, training=training
            )

            # Reshape back: [batch, num_systems, latent_dim]
            return attended_encoding.reshape(batch_size, num_systems, latent_dim)

        return branch_encoding

    def __call__(
        self,
        branch_inputs: jax.Array | list[jax.Array],
        trunk_input: jax.Array,
        *,
        spatial_coords: jax.Array | None = None,
        physics_info: jax.Array | None = None,
        training: bool = False,
    ) -> jax.Array:
        """Apply Multi-Physics DeepONet.

        Args:
            branch_inputs: Function values or list of function values for each system
            trunk_input: Query coordinates [batch, trunk_dim] or
                [batch, num_locations, trunk_dim]
            spatial_coords: Spatial coordinates for sensor optimization
            physics_info: Physics information for attention mechanism
            training: Whether in training mode

        Returns:
            Function values at query locations [batch] or [batch, num_locations]
        """
        # Determine if trunk input has single or multiple locations
        original_trunk_shape = trunk_input.shape
        single_location = len(original_trunk_shape) == 2

        # Prepare branch inputs for multiple systems
        branch_input_list = self._prepare_branch_inputs(branch_inputs)

        # Apply sensor optimization if enabled
        if spatial_coords is not None:
            branch_input_list = self._apply_sensor_optimization(
                branch_input_list, spatial_coords
            )

        # Encode branch inputs
        branch_encoding = self._encode_branch_inputs(branch_input_list)

        # Apply physics-aware attention
        branch_encoding = self._apply_physics_attention(
            branch_encoding, physics_info, training
        )

        # Encode trunk inputs
        trunk_encoding = self._encode_trunk_inputs(trunk_input)

        # Combine multi-physics outputs
        if self.num_physics_systems == 1:
            # Single system: direct inner product
            branch_vec = branch_encoding.squeeze(1)  # [batch, latent_dim]
            output = jnp.einsum("bl,bnl->bn", branch_vec, trunk_encoding)
        else:
            # Multiple systems: combine using coupling weights
            # branch_encoding: [batch, num_systems, latent_dim]
            # trunk_encoding: [batch, num_locations, latent_dim]

            # Flatten systems dimension for coupling
            batch_size = branch_encoding.shape[0]
            branch_flat = branch_encoding.reshape(
                batch_size, self.num_physics_systems * self.latent_dim
            )
            coupled_branch = self.system_coupling(branch_flat)  # [batch, latent_dim]

            # Inner product with coupled representation
            output = jnp.einsum("bl,bnl->bn", coupled_branch, trunk_encoding)

        # Return appropriate shape based on input
        if single_location:
            # Squeeze out the location dimension for single location inputs
            return output.squeeze(-1)
        return output

    def get_sensor_positions(self) -> jax.Array | None:
        """Get current sensor positions if sensor optimization is enabled."""
        if hasattr(self, "sensor_optimizer"):
            return self.sensor_optimizer.sensor_positions.value
        return None

    def set_physics_constraints(self, constraints: list[str]) -> None:
        """Update physics constraints for attention mechanism."""
        self.physics_constraints = constraints
        if self.use_attention and hasattr(self, "physics_attention"):
            self.physics_attention.physics_constraints = constraints

    @property
    def branch_nets(self) -> list[nnx.Module]:
        """Get branch networks from all physics operators."""
        return [op.branch_net for op in self.physics_operators]
