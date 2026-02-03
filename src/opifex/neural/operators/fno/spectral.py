"""
Spectral Neural Operators for Fourier Neural Operator architectures.

This module contains complete neural operator architectures that leverage spectral
methods and normalization techniques for operator learning.
"""

from collections.abc import Sequence

import jax
from flax import nnx

# Import spectral normalization components from the correct location
from opifex.neural.operators.specialized.spectral_normalization import SpectralLinear


class SpectralNeuralOperator(nnx.Module):
    """Neural operator with spectral normalization throughout.

    This is a complete neural operator architecture that applies spectral
    normalization to all linear transformations. It's designed for operator
    learning tasks where stability and regularization are important.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (128, 128, 64),
        num_heads: int = 8,
        power_iterations: int = 1,
        use_adaptive_bounds: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spectral neural operator.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dims: Hidden layer dimensions
            num_heads: Number of attention heads (reserved for future extensions)
            power_iterations: Number of power iteration steps for spectral norm
            use_adaptive_bounds: Whether to use adaptive spectral bounds
            rngs: Random number generators
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Input projection with spectral normalization
        self.input_proj = SpectralLinear(
            input_dim, hidden_dims[0], power_iterations=power_iterations, rngs=rngs
        )

        # Hidden layers with spectral normalization
        hidden_layers_temp = []
        for i in range(len(hidden_dims) - 1):
            hidden_layers_temp.append(
                SpectralLinear(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    power_iterations=power_iterations,
                    rngs=rngs,
                )
            )
        self.hidden_layers = nnx.List(hidden_layers_temp)

        # Output projection with spectral normalization
        self.output_proj = SpectralLinear(
            hidden_dims[-1], output_dim, power_iterations=power_iterations, rngs=rngs
        )

    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Forward pass through spectral neural operator.

        Args:
            x: Input tensor of shape (batch, features) or (batch, seq_len, features)
            training: Whether in training mode

        Returns:
            Output tensor with same spatial dims as input
        """
        # Input projection
        x = self.input_proj(x, training=training)
        x = nnx.gelu(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x, training=training)
            x = nnx.gelu(x)

        # Output projection
        return self.output_proj(x, training=training)


def create_spectral_neural_operator(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int] = (128, 128, 64),
    num_heads: int = 8,
    power_iterations: int = 1,
    use_adaptive_bounds: bool = False,
    *,
    rngs: nnx.Rngs,
) -> SpectralNeuralOperator:
    """Create a neural operator with spectral normalization throughout.

    This factory function creates a complete neural operator architecture
    with spectral normalization applied to all linear transformations.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        hidden_dims: Hidden layer dimensions
        num_heads: Number of attention heads (reserved for future extensions)
        power_iterations: Number of power iteration steps
        use_adaptive_bounds: Whether to use adaptive spectral bounds
        rngs: Random number generators

    Returns:
        Complete spectral normalized neural operator
    """
    return SpectralNeuralOperator(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_heads=num_heads,
        power_iterations=power_iterations,
        use_adaptive_bounds=use_adaptive_bounds,
        rngs=rngs,
    )
