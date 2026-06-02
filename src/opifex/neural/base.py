"""Base neural network classes following FLAX NNX patterns.

All neural networks must use FLAX NNX exclusively as per critical guidelines.
This module provides foundational neural network components optimized for
scientific machine learning with full Flax NNX compliance.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
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
        dtype: Any | None = None,
        param_dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
        kernel_init: Callable = nnx.initializers.xavier_uniform(),
        bias_init: Callable = nnx.initializers.zeros,
    ) -> None:
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
            dtype: Computation dtype for NNX linear layers. ``None`` preserves
                the Flax default promotion behavior.
            param_dtype: Parameter storage dtype for NNX linear layers.
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
        self.dtype = dtype
        self.param_dtype = param_dtype

        # Validate layer sizes
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements (input and output)")

        # Create layers following NNX patterns (use nnx.List for Flax 0.12.0+)
        layers = []
        for i in range(len(layer_sizes) - 1):
            layer = nnx.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                use_bias=use_bias,
                kernel_init=kernel_init,
                bias_init=bias_init,
                dtype=dtype,
                param_dtype=param_dtype,
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
