"""Base neural network classes following FLAX NNX patterns.

All neural networks must use FLAX NNX exclusively as per critical guidelines.
This module provides foundational neural network components optimized for
scientific machine learning with full Flax NNX compliance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from flax import nnx


# JAX configuration is now handled automatically at import


# Importing get_activation at the bottom to avoid circular import issues
# since activations.py might reference base.py for type hints


if TYPE_CHECKING:
    from collections.abc import Callable


# Import after TYPE_CHECKING to avoid circular imports
from opifex.neural.activations import get_activation


class StandardMLP(nnx.Module):
    """Modern Multi-Layer Perceptron implementation using FLAX NNX.

    Fully compliant with Flax NNX best practices including:
    - Proper RNG handling with keyword-only rngs parameter
    - Modern activation functions (GELU default, configurable)
    - Efficient dropout strategies with deterministic control
    - Custom initialization strategies following NNX patterns
    - Automatic differentiation with JAX
    - Performance-optimized state management

    Attributes:
        layer_sizes: List of layer sizes including input and output dimensions
        activation: Name of the activation function to use
        dropout_rate: Dropout probability (0.0 means no dropout)
        use_bias: Whether to include bias terms in linear layers
        apply_final_dropout: Whether to apply dropout after the final layer
        layers: Sequence of linear transformation layers
        activation_fn: The actual activation function
        dropout: Dropout layer (None if dropout_rate is 0)
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "gelu",
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        apply_final_dropout: bool = False,
        *,
        rngs: nnx.Rngs,
        kernel_init: Callable = nnx.initializers.xavier_uniform(),
        bias_init: Callable = nnx.initializers.zeros,
    ):
        """Initialize the StandardMLP following modern NNX patterns.

        Args:
            layer_sizes: List of layer sizes, e.g.,
                [input_dim, hidden1, hidden2, output_dim]
            activation: Activation function name
                ('gelu', 'tanh', 'relu', 'sigmoid', 'silu')
                Default is 'gelu' for modern neural networks
            dropout_rate: Dropout probability for regularization
                (0.0 = no dropout)
            use_bias: Whether to use bias in linear projections
            apply_final_dropout: Whether to apply dropout after final layer
                (useful for some transformer-style architectures)
            rngs: FLAX NNX random number generator state (keyword-only)
            kernel_init: Kernel initialization function (callable)
            bias_init: Bias initialization function (callable)
        """
        super().__init__()

        # Store configuration
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.apply_final_dropout = apply_final_dropout

        # Validate layer sizes
        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes must have at least 2 elements (input and output)"
            )

        # Create layers following NNX patterns (use nnx.List for Flax 0.12.0+)
        layers = []
        for i in range(len(layer_sizes) - 1):
            layer = nnx.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                use_bias=use_bias,
                kernel_init=kernel_init,
                bias_init=bias_init,
                rngs=rngs,
            )
            layers.append(layer)
        self.layers = nnx.List(layers)

        # Set activation function using the activation library
        self.activation_fn = get_activation(activation)

        # Initialize dropout if needed - pass rngs directly
        if dropout_rate > 0.0:
            self.dropout: nnx.Dropout | None = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass through the network.

        Following NNX best practices, this method does NOT include rngs parameter
        as dropout layers manage their own RNG state internally.

        Args:
            x: Input array of shape (batch_size, input_dim)
            deterministic: Whether to apply dropout
                (False for training, True for inference)

        Returns:
            Output array of shape (batch_size, output_dim)
        """

        # Forward pass through all layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply activation to all layers except the last one
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)

                # Apply dropout after activation for hidden layers
                if self.dropout is not None and not deterministic:
                    x = self.dropout(x, deterministic=deterministic)

        # Apply final dropout if requested (useful for transformer-style architectures)
        if self.apply_final_dropout and self.dropout is not None and not deterministic:
            x = self.dropout(x, deterministic=deterministic)

        return x


class QuantumMLP(nnx.Module):
    """Modern Quantum-aware Multi-Layer Perceptron for molecular and quantum systems.

    Fully compliant with Flax NNX best practices while providing
    quantum-specific features:
    - Proper RNG handling with keyword-only rngs parameter
    - Symmetry enforcement for molecular systems
    - Specialized initialization for quantum properties
    - Physics-informed constraints with numerical stability
    - Modern dropout strategies with deterministic control
    - Quantum-specific energy and force computation methods

    Attributes:
        layer_sizes: List of layer sizes including input and output dimensions
        activation: Activation function name
        enforce_symmetry: Whether to enforce permutation symmetry
        dropout_rate: Dropout probability for regularization
        use_bias: Whether to use bias in linear layers
        apply_final_dropout: Whether to apply dropout after the final layer
        layers: Sequence of linear layers
        activation_fn: Activation function
        dropout: Dropout layer (if dropout_rate > 0)
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "tanh",
        enforce_symmetry: bool = True,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        apply_final_dropout: bool = False,
        symmetry_type: str = "permutation",
        *,
        rngs: nnx.Rngs,
        kernel_init: Callable = nnx.initializers.xavier_uniform(),
        bias_init: Callable = nnx.initializers.zeros,
    ):
        """Initialize Quantum MLP following modern NNX patterns.

        Args:
            layer_sizes: List of layer sizes for the network architecture
            activation: Activation function name
                ('gelu', 'tanh', 'relu', 'sigmoid', 'silu')
                Default is 'tanh' for quantum neural networks
            enforce_symmetry: Whether to enforce molecular symmetries
            dropout_rate: Dropout probability for regularization
                (0.0 = no dropout)
            use_bias: Whether to use bias in linear projections
            apply_final_dropout: Whether to apply dropout after final layer
                (useful for quantum transformer-style architectures)
            symmetry_type: Type of symmetry to enforce
                ('permutation', 'rotation', 'both')
            rngs: FLAX NNX random number generator state (keyword-only)
            kernel_init: Kernel initialization function
                (callable, quantum-aware)
            bias_init: Bias initialization function
                (callable, quantum-aware)
        """
        super().__init__()

        # Store configuration
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.enforce_symmetry = enforce_symmetry
        self.symmetry_type = symmetry_type
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.apply_final_dropout = apply_final_dropout

        # Validate layer sizes
        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes must have at least 2 elements (input and output)"
            )

        # Apply quantum-aware initialization scaling if needed
        quantum_kernel_init = self._apply_quantum_scaling(kernel_init)
        quantum_bias_init = self._apply_quantum_scaling(bias_init)

        # Create layers with quantum-aware initialization (use nnx.List)
        layers = []
        for i in range(len(layer_sizes) - 1):
            layer = nnx.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                use_bias=use_bias,
                kernel_init=quantum_kernel_init,
                bias_init=quantum_bias_init,
                rngs=rngs,
            )
            layers.append(layer)
        self.layers = nnx.List(layers)

        # Set activation function optimized for quantum calculations
        self.activation_fn = get_activation(activation)

        # Initialize dropout if needed - pass rngs directly
        if dropout_rate > 0.0:
            self.dropout: nnx.Dropout | None = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

        # Setup symmetry constraints if needed
        self._setup_symmetry_constraints()

    def _apply_quantum_scaling(self, init_fn: Callable) -> Callable:
        """Apply quantum-specific scaling to initialization functions.

        Quantum systems often require more careful initialization to avoid
        vanishing/exploding gradients in energy calculations.

        Args:
            init_fn: Original initialization function

        Returns:
            Quantum-scaled initialization function
        """

        def quantum_scaled_init(*args, **kwargs):
            # Apply base initialization
            weights = init_fn(*args, **kwargs)
            # Apply quantum-specific scaling (conservative for stability)
            quantum_scale = 0.5
            return weights * quantum_scale

        return quantum_scaled_init

    def _setup_symmetry_constraints(self) -> None:
        """Setup symmetry enforcement mechanisms.

        # TODO: Implement sophisticated symmetry enforcement based on quantum system
        # requirements
        # Current implementation is a simplified approximation
        """
        if self.enforce_symmetry:
            # For now, just flag that we want to use symmetry
            # More sophisticated implementation would involve:
            # - Permutation matrices for atomic permutations
            # - Rotation matrices for orientational symmetry
            # - Group theory operations for molecular point groups
            self._use_permutation_symmetry = self.symmetry_type in [
                "permutation",
                "both",
            ]
            self._use_rotation_symmetry = self.symmetry_type in ["rotation", "both"]
        else:
            self._use_permutation_symmetry = False
            self._use_rotation_symmetry = False

    def _enforce_permutation_symmetry(self, x: jax.Array) -> jax.Array:
        """Enforce permutation symmetry for identical particles.

        # TODO: Implement full permutation symmetry enforcement:
        # 1. Identify groups of identical particles
        # 2. Apply permutation operations
        # 3. Average over all permutations
        # 4. Return symmetrized representation

        Args:
            x: Input array representing particle configurations

        Returns:
            Symmetrized representation
        """
        # TODO: Implement symmetry enforcement logic
        # Currently just returning input as pass-through
        # Full implementation would require knowledge of particle types
        # and molecular structure
        return x

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass through the quantum-aware network.

        Following NNX best practices, this method does NOT include rngs parameter
        as dropout layers manage their own RNG state internally.

        Args:
            x: Input array of shape (batch_size, input_dim)
            deterministic: Whether to apply dropout
                (False for training, True for inference)

        Returns:
            Output array of shape (batch_size, output_dim)
        """

        # Apply symmetry constraints if enabled
        if self.enforce_symmetry and self._use_permutation_symmetry:
            x = self._enforce_permutation_symmetry(x)

        # Forward pass through all layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply activation to all layers except the last one
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)

                # Apply dropout after activation for hidden layers
                if self.dropout is not None and not deterministic:
                    x = self.dropout(x, deterministic=deterministic)

        # Apply final dropout if requested (useful for transformer-style architectures)
        if self.apply_final_dropout and self.dropout is not None and not deterministic:
            x = self.dropout(x, deterministic=deterministic)

        return x

    def compute_energy(
        self,
        positions: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Compute energy for given atomic positions.

        Args:
            positions: Atomic positions array of shape (batch, n_atoms, 3)
                or (n_atoms, 3) or flattened (n_atoms*3,)
            deterministic: Whether to use deterministic mode
                (True for inference)

        Returns:
            Energy array with shape (batch_size, 1) for consistency
        """
        # Handle flattened 1D input (common in tests)
        if positions.ndim == 1:
            # Assume 3D coordinates, create batch with single item
            if positions.shape[0] % 3 != 0:
                raise ValueError(
                    f"Flattened positions length {positions.shape[0]} must be "
                    f"divisible by 3"
                )
            flat_positions = positions[None, :]  # Add batch dimension
        # Handle both batched and single inputs for 2D/3D
        elif positions.ndim == 2:
            # Single molecule: shape (n_atoms, 3)
            flat_positions = positions.flatten()[None, :]  # Add batch dim after flatten
        elif positions.ndim == 3:
            # Batched molecules: shape (batch, n_atoms, 3)
            # Flatten each batch item separately
            batch_size = positions.shape[0]
            flat_positions = positions.reshape(batch_size, -1)  # (batch, n_atoms*3)
        else:
            raise ValueError(
                f"Expected positions with 1, 2 or 3 dimensions, got {positions.ndim}"
            )

        # Forward pass to get energy
        energy = self(flat_positions, deterministic=deterministic)

        # For 1D input, return scalar for API consistency with test expectations
        if positions.ndim == 1:
            return energy.squeeze()  # Return scalar energy

        # Ensure energy has shape (batch_size, 1) for API consistency
        if energy.ndim == 2 and energy.shape[1] == 1:
            return energy  # Already correct shape
        # Reshape to (batch_size, 1)
        return energy.reshape(-1, 1)

    def _compute_energy_scalar(
        self,
        positions: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Compute energy as scalar for gradient computation.

        Args:
            positions: Atomic positions array of shape (n_atoms, 3)
            deterministic: Whether to use deterministic mode
                (True for inference)

        Returns:
            Energy as a scalar array for gradient computation
        """
        # This method is used for gradient computation and expects single molecule
        if positions.ndim != 2:
            raise ValueError(
                f"_compute_energy_scalar expects 2D positions, got {positions.ndim}D"
            )

        # Flatten positions for network input
        flat_positions = positions.flatten()[None, :]  # Add batch dimension

        # Forward pass to get energy
        energy = self(flat_positions, deterministic=deterministic)

        # Return scalar energy for gradient computation
        return energy.squeeze()

    def compute_forces(
        self,
        positions: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Compute forces as negative gradient of energy.

        Args:
            positions: Atomic positions array of shape (batch, n_atoms, 3)
                or (n_atoms, 3) or flattened (n_atoms*3,)
            deterministic: Whether to use deterministic mode
                (True for inference)

        Returns:
            Forces array of shape (batch, n_atoms, 3) or (n_atoms, 3) or (n_atoms*3,)
            matching the input shape
        """
        # Handle flattened 1D input (common in tests)
        if positions.ndim == 1:
            # Assume 3D coordinates, reshape to (n_atoms, 3)
            if positions.shape[0] % 3 != 0:
                raise ValueError(
                    f"Flattened positions length {positions.shape[0]} must be "
                    f"divisible by 3"
                )
            n_atoms = positions.shape[0] // 3
            positions_reshaped = positions.reshape(n_atoms, 3)

            def energy_fn_1d(pos):
                return self._compute_energy_scalar(pos, deterministic=deterministic)

            # Compute forces for reshaped positions
            forces_reshaped = -jax.grad(energy_fn_1d)(positions_reshaped)

            # Return in original flattened shape
            return forces_reshaped.flatten()

        def energy_fn_2d3d(pos):
            return self._compute_energy_scalar(pos, deterministic=deterministic)

        if positions.ndim == 2:
            # Single molecule case
            return -jax.grad(energy_fn_2d3d)(positions)
        if positions.ndim == 3:
            # Batched case - use vmap to handle batch dimension
            batched_grad = jax.vmap(jax.grad(energy_fn_2d3d))
            return -batched_grad(positions)
        raise ValueError(
            f"Expected positions with 1, 2 or 3 dimensions, got {positions.ndim}"
        )

    def compute_energy_and_forces(
        self,
        positions: jax.Array,
        *,
        deterministic: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        """Efficiently compute both energy and forces.

        Args:
            positions: Atomic positions array of shape (n_atoms, 3) or
                flattened (n_atoms*3,)
            deterministic: Whether to use deterministic mode
                (True for inference)

        Returns:
            Tuple of (energy, forces) where energy is scalar and
            forces has shape (n_atoms, 3) or (n_atoms*3,) matching input
        """
        # Handle flattened 1D input (common in tests)
        if positions.ndim == 1:
            # Assume 3D coordinates, reshape to (n_atoms, 3)
            if positions.shape[0] % 3 != 0:
                raise ValueError(
                    f"Flattened positions length {positions.shape[0]} must be "
                    f"divisible by 3"
                )
            n_atoms = positions.shape[0] // 3
            positions_reshaped = positions.reshape(n_atoms, 3)

            def energy_fn_1d(pos):
                return self._compute_energy_scalar(pos, deterministic=deterministic)

            # Use value_and_grad for efficiency
            energy, grad_energy = jax.value_and_grad(energy_fn_1d)(positions_reshaped)
            forces = -grad_energy  # Forces are negative gradient

            # Return forces in original flattened shape
            return energy, forces.flatten()

        def energy_fn_2d(pos):
            return self._compute_energy_scalar(pos, deterministic=deterministic)

        # Use value_and_grad for efficiency
        energy, grad_energy = jax.value_and_grad(energy_fn_2d)(positions)
        forces = -grad_energy  # Forces are negative gradient
        return energy, forces
