"""Fourier-Enhanced Deep Operator Networks.

Enhanced DeepONet architectures that integrate Fourier Neural Operator concepts
into DeepONet for improved performance on problems with spectral structure.
Following FLAX NNX patterns and critical technical guidelines.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.base import StandardMLP
from opifex.neural.operators.fno.base import FourierSpectralConvolution


class FourierEnhancedDeepONet(nnx.Module):
    """Fourier-Enhanced DeepONet combining spectral and operator learning.

    This variant integrates Fourier Neural Operator concepts into DeepONet
    architecture for improved performance on problems with spectral structure.
    """

    def __init__(
        self,
        branch_sizes: list[int],
        trunk_sizes: list[int],
        *,
        fourier_modes: int = 16,
        use_spectral_branch: bool = True,
        use_spectral_trunk: bool = False,
        activation: str = "tanh",
        rngs: nnx.Rngs,
    ):
        """Initialize Fourier-Enhanced DeepONet.

        Args:
            branch_sizes: Branch network layer sizes [input, hidden..., output]
            trunk_sizes: Trunk network layer sizes [input, hidden..., output]
            fourier_modes: Number of Fourier modes for spectral layers
            use_spectral_branch: Whether to use spectral convolution in branch
            use_spectral_trunk: Whether to use spectral convolution in trunk
            activation: Activation function name
            rngs: Random number generators
        """
        super().__init__()
        self.branch_sizes = branch_sizes
        self.trunk_sizes = trunk_sizes
        self.fourier_modes = fourier_modes
        self.use_spectral_branch = use_spectral_branch
        self.use_spectral_trunk = use_spectral_trunk

        # Enhanced branch network with optional spectral layers
        if use_spectral_branch:
            # Add spectral convolution layer before standard MLP
            self.branch_spectral = FourierSpectralConvolution(
                in_channels=1,  # Single channel input function
                out_channels=branch_sizes[1] // 4,  # Reduce for efficiency
                modes=fourier_modes,
                rngs=rngs,
            )

            # Adjust layer sizes for spectral preprocessing
            adjusted_branch_sizes = [
                branch_sizes[0] + branch_sizes[1] // 4,  # Input + spectral features
                *branch_sizes[1:],  # Rest of the layers
            ]

            self.branch_net = StandardMLP(
                layer_sizes=adjusted_branch_sizes,
                activation=activation,
                dropout_rate=0.0,
                use_bias=True,
                apply_final_dropout=False,
                rngs=rngs,
            )
        else:
            self.branch_net = StandardMLP(
                layer_sizes=branch_sizes,
                activation=activation,
                dropout_rate=0.0,
                use_bias=True,
                apply_final_dropout=False,
                rngs=rngs,
            )

        # Enhanced trunk network with optional spectral layers
        if use_spectral_trunk:
            self.trunk_spectral = FourierSpectralConvolution(
                in_channels=1,  # Single channel for coordinate processing
                out_channels=trunk_sizes[1] // 2,
                modes=max(
                    fourier_modes, trunk_sizes[0]
                ),  # Ensure modes >= spatial size
                rngs=rngs,
            )

            adjusted_trunk_sizes = [
                trunk_sizes[0] + trunk_sizes[1] // 2,  # Input + spectral features
                *trunk_sizes[1:],  # Rest of the layers
            ]

            self.trunk_net = StandardMLP(
                layer_sizes=adjusted_trunk_sizes,
                activation=activation,
                dropout_rate=0.0,
                use_bias=True,
                apply_final_dropout=False,
                rngs=rngs,
            )
        else:
            self.trunk_net = StandardMLP(
                layer_sizes=trunk_sizes,
                activation=activation,
                dropout_rate=0.0,
                use_bias=True,
                apply_final_dropout=False,
                rngs=rngs,
            )

        # Fourier-enhanced combination layer
        self.fourier_combiner = nnx.Linear(
            in_features=branch_sizes[-1],  # Output dimension
            out_features=branch_sizes[-1],
            rngs=rngs,
        )

    def __call__(
        self,
        branch_input: jax.Array,
        trunk_input: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Apply Fourier-Enhanced DeepONet.

        Args:
            branch_input: Function values at sensor locations (batch, branch_input_dim)
            trunk_input: Query coordinates (batch, num_locations, trunk_input_dim)
            deterministic: Whether to use deterministic mode

        Returns:
            Function values at query locations (batch, num_locations)
        """
        batch_size, num_locations, trunk_dim = trunk_input.shape

        # Process branch input with optional spectral enhancement
        if self.use_spectral_branch:
            # Reshape for spectral convolution (assume 1D spatial function)
            branch_spatial = branch_input.reshape(
                batch_size, 1, -1
            )  # (batch, 1, spatial)
            branch_spectral_features = self.branch_spectral(
                branch_spatial
            )  # (batch, channels, spatial)
            branch_spectral_flat = branch_spectral_features.mean(
                axis=-1
            )  # (batch, channels)

            # Combine original and spectral features
            branch_combined = jnp.concatenate(
                [branch_input, branch_spectral_flat], axis=-1
            )
            branch_encoding = self.branch_net(
                branch_combined, deterministic=deterministic
            )
        else:
            branch_encoding = self.branch_net(branch_input, deterministic=deterministic)

        # Process trunk input with optional spectral enhancement
        trunk_input_flat = trunk_input.reshape(batch_size * num_locations, trunk_dim)

        if self.use_spectral_trunk:
            # Apply spectral processing to coordinates
            trunk_spatial = trunk_input_flat.reshape(
                -1, 1, trunk_dim
            )  # (batch*locations, 1, trunk_dim)
            trunk_spectral_features = self.trunk_spectral(trunk_spatial)
            trunk_spectral_flat = trunk_spectral_features.mean(
                axis=-1
            )  # (batch*locations, out_channels)

            # Combine original and spectral features
            trunk_combined = jnp.concatenate(
                [trunk_input_flat, trunk_spectral_flat], axis=-1
            )
            trunk_encoding_flat = self.trunk_net(
                trunk_combined, deterministic=deterministic
            )
        else:
            trunk_encoding_flat = self.trunk_net(
                trunk_input_flat, deterministic=deterministic
            )

        trunk_encoding = trunk_encoding_flat.reshape(
            batch_size, num_locations, self.branch_sizes[-1]
        )

        # Enhanced combination with Fourier processing
        branch_encoding_enhanced = self.fourier_combiner(branch_encoding)

        # Inner product to get final output
        return jnp.einsum("bl,bnl->bn", branch_encoding_enhanced, trunk_encoding)
