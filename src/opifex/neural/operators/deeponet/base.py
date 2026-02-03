"""Deep Operator Networks (DeepONet) implementation.

This module provides a comprehensive implementation of Deep Operator Networks
for learning nonlinear operators mapping between function spaces.
Fully compliant with modern Flax NNX patterns and optimized for scientific computing.

MODERNIZATION APPLIED:
- Full Flax NNX compliance with proper RNG handling
- Enhanced branch and trunk network architectures
- Optimized tensor operations for operator learning
- Support for multiple DeepONet variants (vanilla, adaptive, multifidelity)
- Robust handling of function and location inputs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable

# Import neural network base classes
from opifex.neural.activations import get_activation
from opifex.neural.base import StandardMLP


class DeepONet(nnx.Module):
    """Deep Operator Network for learning function-to-function mappings.

    DeepONet learns to approximate nonlinear operators G that map functions
    to functions: G: u → G(u), where u and G(u) are functions.

    The architecture consists of:
    - Branch network: Processes input function u evaluated at sensors
    - Trunk network: Processes evaluation locations y
    - Dot product combination of branch and trunk outputs

    Fully compliant with modern Flax NNX patterns.
    """

    def __init__(
        self,
        branch_sizes: list[int],
        trunk_sizes: list[int],
        *,
        activation: str = "gelu",
        output_activation: str | None = None,
        use_bias: bool = True,
        rngs: nnx.Rngs,
    ):
        """Initialize DeepONet following modern NNX patterns.

        Args:
            branch_sizes: Layer sizes for branch network
                [input_sensors, hidden1, hidden2, ..., output_dim]
            trunk_sizes: Layer sizes for trunk network
                [location_dim, hidden1, hidden2, ..., output_dim]
                Note: output_dim should match branch output_dim
            activation: Activation function name for hidden layers
            output_activation: Optional activation for final output
                (None means no activation on output)
            use_bias: Whether to use bias in linear layers
            rngs: Random number generators (keyword-only)
        """
        super().__init__()

        # Validate that branch and trunk have same output dimension
        if branch_sizes[-1] != trunk_sizes[-1]:
            raise ValueError(
                f"Branch output dim ({branch_sizes[-1]}) must match "
                f"trunk output dim ({trunk_sizes[-1]})"
            )

        self.branch_sizes = branch_sizes
        self.trunk_sizes = trunk_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.output_dim = branch_sizes[-1]

        # Create branch network for processing input functions
        self.branch_net = StandardMLP(
            layer_sizes=branch_sizes,
            activation=activation,
            dropout_rate=0.0,  # No dropout in standard DeepONet
            use_bias=use_bias,
            apply_final_dropout=False,
            rngs=rngs,
        )

        # Create trunk network for processing evaluation locations
        self.trunk_net = StandardMLP(
            layer_sizes=trunk_sizes,
            activation=activation,
            dropout_rate=0.0,  # No dropout in standard DeepONet
            use_bias=use_bias,
            apply_final_dropout=False,
            rngs=rngs,
        )

        # Optional output activation
        if output_activation is not None:
            self.output_activation_fn = get_activation(output_activation)
        else:
            self.output_activation_fn = None

    def __call__(
        self,
        branch_input: jax.Array,
        trunk_input: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Apply DeepONet to compute operator output.

        Following NNX best practices, this method does NOT include rngs parameter
        as all random state is managed during initialization.

        Args:
            branch_input: Function values at sensor locations
                Shape: (batch_size, n_sensors) where n_sensors = branch_sizes[0]
            trunk_input: Evaluation locations
                Shape: (batch_size, n_locations, location_dim) or
                       (n_locations, location_dim) for single batch
            deterministic: Whether to use deterministic mode (unused here but
                          kept for API consistency)

        Returns:
            Operator output at evaluation locations
            Shape: (batch_size, n_locations) or (n_locations,) matching trunk_input
        """
        # Handle single batch case for trunk_input
        if trunk_input.ndim == 2:
            # Single batch: (n_locations, location_dim)
            single_batch = True
            trunk_input = trunk_input[None, :, :]  # Add batch dimension
        else:
            # Batched: (batch_size, n_locations, location_dim)
            single_batch = False

        batch_size, n_locations, location_dim = trunk_input.shape

        # Ensure branch input has correct batch size
        if branch_input.shape[0] != batch_size:
            raise ValueError(
                f"Branch input batch size ({branch_input.shape[0]}) must match "
                f"trunk input batch size ({batch_size})"
            )

        # Process branch input (function values at sensors)
        # Shape: (batch_size, output_dim)
        branch_output = self.branch_net(branch_input, deterministic=deterministic)

        # Process trunk input (evaluation locations)
        # Reshape to (batch_size * n_locations, location_dim) for batch processing
        trunk_flat = trunk_input.reshape(batch_size * n_locations, location_dim)

        # Apply trunk network
        # Shape: (batch_size * n_locations, output_dim)
        trunk_output_flat = self.trunk_net(trunk_flat, deterministic=deterministic)

        # Reshape back to (batch_size, n_locations, output_dim)
        trunk_output = trunk_output_flat.reshape(
            batch_size, n_locations, self.output_dim
        )

        # Compute dot product between branch and trunk outputs
        # branch_output: (batch_size, output_dim)
        # trunk_output: (batch_size, n_locations, output_dim)
        # Result: (batch_size, n_locations)
        operator_output = jnp.sum(branch_output[:, None, :] * trunk_output, axis=-1)

        # Apply output activation if specified
        if self.output_activation_fn is not None:
            operator_output = self.output_activation_fn(operator_output)

        # Return in original format
        if single_batch:
            return operator_output.squeeze(0)  # Remove batch dimension
        return operator_output

    def get_branch_output(
        self,
        branch_input: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Get branch network output for analysis purposes.

        Args:
            branch_input: Function values at sensor locations
            deterministic: Whether to use deterministic mode

        Returns:
            Branch network output
        """
        return self.branch_net(branch_input, deterministic=deterministic)

    def get_trunk_output(
        self,
        trunk_input: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Get trunk network output for analysis purposes.

        Args:
            trunk_input: Evaluation locations
            deterministic: Whether to use deterministic mode

        Returns:
            Trunk network output
        """
        # Handle reshaping for batch processing
        original_shape = trunk_input.shape
        if trunk_input.ndim == 2:
            # Single batch
            trunk_flat = trunk_input
        else:
            # Multiple batches
            trunk_flat = trunk_input.reshape(-1, trunk_input.shape[-1])

        output_flat = self.trunk_net(trunk_flat, deterministic=deterministic)

        # Reshape back to match input structure
        if trunk_input.ndim == 2:
            return output_flat
        new_shape = (*original_shape[:-1], self.output_dim)
        return output_flat.reshape(new_shape)


class AdaptiveDeepONet(nnx.Module):
    """Adaptive DeepONet with learned sensor selection.

    Extends standard DeepONet with adaptive sensor placement,
    allowing the network to learn optimal sensor locations
    during training.
    """

    def __init__(
        self,
        branch_sizes: list[int],
        trunk_sizes: list[int],
        sensor_dim: int,
        *,
        activation: str = "gelu",
        output_activation: str | None = None,
        use_bias: bool = True,
        sensor_init: str = "uniform",
        rngs: nnx.Rngs,
    ):
        """Initialize Adaptive DeepONet following NNX patterns.

        Args:
            branch_sizes: Layer sizes for branch network
            trunk_sizes: Layer sizes for trunk network
            sensor_dim: Dimensionality of sensor locations
            activation: Activation function name
            output_activation: Optional output activation
            use_bias: Whether to use bias in linear layers
            sensor_init: Sensor initialization strategy ('uniform', 'normal')
            rngs: Random number generators (keyword-only)
        """
        super().__init__()

        self.branch_sizes = branch_sizes
        self.trunk_sizes = trunk_sizes
        self.sensor_dim = sensor_dim
        self.n_sensors = branch_sizes[0]  # Number of sensors from branch input size
        self.activation = activation
        self.output_activation = output_activation

        # Initialize learnable sensor locations
        if sensor_init == "uniform":
            # Initialize sensors uniformly in [-1, 1]^sensor_dim
            sensor_locations = jax.random.uniform(
                rngs.params(),
                (self.n_sensors, sensor_dim),
                minval=-1.0,
                maxval=1.0,
            )
        elif sensor_init == "normal":
            # Initialize sensors with normal distribution
            sensor_locations = jax.random.normal(
                rngs.params(),
                (self.n_sensors, sensor_dim),
            )
        else:
            raise ValueError(f"Unknown sensor_init: {sensor_init}")

        self.sensor_locations = nnx.Param(sensor_locations)

        # Create the underlying DeepONet
        self.deeponet = DeepONet(
            branch_sizes=branch_sizes,
            trunk_sizes=trunk_sizes,
            activation=activation,
            output_activation=output_activation,
            use_bias=use_bias,
            rngs=rngs,
        )

    def __call__(
        self,
        input_function: Callable[[jax.Array], jax.Array],
        trunk_input: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Apply Adaptive DeepONet.

        Args:
            input_function: Function to evaluate at sensor locations
            trunk_input: Evaluation locations
            deterministic: Whether to use deterministic mode

        Returns:
            Operator output at evaluation locations
        """
        # Evaluate input function at learned sensor locations
        # sensor_locations: (n_sensors, sensor_dim)
        sensor_values = jax.vmap(input_function)(self.sensor_locations.value)

        # Add batch dimension if needed
        if trunk_input.ndim == 3:
            # Multiple batches: repeat sensor values for each batch
            batch_size = trunk_input.shape[0]
            branch_input = jnp.tile(sensor_values[None, :], (batch_size, 1))
        else:
            # Single batch
            branch_input = sensor_values[None, :]

        # Apply standard DeepONet
        return self.deeponet(
            branch_input=branch_input,
            trunk_input=trunk_input,
            deterministic=deterministic,
        )

    def get_sensor_locations(self) -> jax.Array:
        """Get current sensor locations.

        Returns:
            Sensor locations array of shape (n_sensors, sensor_dim)
        """
        return self.sensor_locations.value


class MultiFidelityDeepONet(nnx.Module):
    """Multi-fidelity DeepONet for handling data of different fidelities.

    Combines multiple DeepONets to handle low-fidelity and high-fidelity data,
    with learned fusion strategies.
    """

    def __init__(
        self,
        branch_sizes: list[int],
        trunk_sizes: list[int],
        n_fidelities: int = 2,
        *,
        activation: str = "gelu",
        output_activation: str | None = None,
        use_bias: bool = True,
        fusion_strategy: str = "linear",
        rngs: nnx.Rngs,
    ):
        """Initialize Multi-fidelity DeepONet following NNX patterns.

        Args:
            branch_sizes: Layer sizes for branch networks
            trunk_sizes: Layer sizes for trunk networks
            n_fidelities: Number of fidelity levels
            activation: Activation function name
            output_activation: Optional output activation
            use_bias: Whether to use bias in linear layers
            fusion_strategy: How to combine fidelities ('linear', 'nonlinear')
            rngs: Random number generators (keyword-only)
        """
        super().__init__()

        self.branch_sizes = branch_sizes
        self.trunk_sizes = trunk_sizes
        self.n_fidelities = n_fidelities
        self.fusion_strategy = fusion_strategy

        # Create separate DeepONets for each fidelity level
        fidelity_nets_temp = []
        for _i in range(n_fidelities):
            net = DeepONet(
                branch_sizes=branch_sizes,
                trunk_sizes=trunk_sizes,
                activation=activation,
                output_activation=output_activation,
                use_bias=use_bias,
                rngs=rngs,
            )
            fidelity_nets_temp.append(net)
            self.fidelity_nets = nnx.List(fidelity_nets_temp)

        # Fusion network for combining fidelity outputs
        if fusion_strategy == "linear":
            # Simple linear combination with learned weights
            self.fusion_weights = nnx.Param(jnp.ones(n_fidelities) / n_fidelities)
        elif fusion_strategy == "nonlinear":
            # Nonlinear fusion network
            self.fusion_net = StandardMLP(
                layer_sizes=[n_fidelities, n_fidelities * 2, 1],
                activation=activation,
                dropout_rate=0.0,
                use_bias=use_bias,
                apply_final_dropout=False,
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unknown fusion_strategy: {fusion_strategy}")

    def __call__(
        self,
        branch_inputs: list[jax.Array],
        trunk_input: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Apply Multi-fidelity DeepONet.

        Args:
            branch_inputs: List of function values for each fidelity level
                Each element has shape (batch_size, n_sensors_i)
            trunk_input: Evaluation locations
            deterministic: Whether to use deterministic mode

        Returns:
            Fused operator output at evaluation locations
        """
        if len(branch_inputs) != self.n_fidelities:
            raise ValueError(
                f"Expected {self.n_fidelities} branch inputs, got {len(branch_inputs)}"
            )

        # Get outputs from each fidelity network
        fidelity_outputs = []
        for _i, (branch_input, net) in enumerate(
            zip(branch_inputs, self.fidelity_nets, strict=False)
        ):
            output = net(
                branch_input=branch_input,
                trunk_input=trunk_input,
                deterministic=deterministic,
            )
            fidelity_outputs.append(output)

        # Stack outputs for fusion
        # Shape: (batch_size, n_locations, n_fidelities)
        stacked_outputs = jnp.stack(fidelity_outputs, axis=-1)

        # Apply fusion strategy
        if self.fusion_strategy == "linear":
            # Linear combination with learned weights
            # fusion_weights: (n_fidelities,)
            # stacked_outputs: (..., n_fidelities)
            fused_output = jnp.sum(stacked_outputs * self.fusion_weights.value, axis=-1)
        elif self.fusion_strategy == "nonlinear":
            # Nonlinear fusion - apply fusion network pointwise
            original_shape = stacked_outputs.shape[:-1]
            flat_outputs = stacked_outputs.reshape(-1, self.n_fidelities)

            # Apply fusion network
            fusion_result = self.fusion_net(flat_outputs, deterministic=deterministic)
            fused_output = fusion_result.reshape(original_shape)

        return fused_output

    def get_fusion_weights(self) -> jax.Array | None:
        """Get fusion weights for linear fusion strategy.

        Returns:
            Fusion weights if using linear strategy, None otherwise
        """
        if self.fusion_strategy == "linear":
            return self.fusion_weights.value
        return None
