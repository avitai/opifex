# FILE PLACEMENT: opifex/neural/operators/fno/local.py
#
# Local Fourier Neural Operator Implementation
# Combines global Fourier operations with local convolutions
#
# This file should be placed at: opifex/neural/operators/fno/local.py
# After placement, update opifex/neural/operators/fno/__init__.py to include:
# from .local import LocalFourierNeuralOperator, LocalFourierLayer

"""
Local Fourier Neural Operator (Local FNO) implementation.

This module provides FNO variants that combine global Fourier operations
with local convolutions to capture both global patterns and local features.
Excellent for problems requiring both long-range dependencies and local detail.
"""

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array

from opifex.neural.operators.common.tensor_ops import (
    apply_linear_with_channel_transform,
    standardized_fft,
    standardized_ifft,
    StandardSpectralConv,
)


class LocalFourierLayer(nnx.Module):
    """
    Fourier layer with local convolution for capturing short-range interactions.

    Combines global spectral convolution with local spatial convolution
    for comprehensive feature extraction.
    """

    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        kernel_size: int = 3,
        activation: Callable = nnx.gelu,
        mixing_weight: float = 0.5,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize local Fourier layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Fourier modes for spectral convolution
            kernel_size: Kernel size for local convolution
            activation: Activation function
            mixing_weight: Weight for combining spectral and local branches
            rngs: Random number generator state
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = tuple(modes)
        self.mixing_weight = mixing_weight
        self.activation = activation
        self.spatial_dims = len(modes)

        # Spectral convolution branch
        self.spectral_conv = StandardSpectralConv(
            in_channels, out_channels, modes, rngs=rngs
        )

        # Local convolution branch
        spatial_dims = len(modes)
        if spatial_dims == 1:
            # Use standard Conv for 1D (JAX treats 1D as 2D with singleton dimension)
            self.local_conv = nnx.Conv(
                in_channels,  # Input: in_channels
                out_channels,  # Output: out_channels (match spectral branch)
                kernel_size=kernel_size,
                padding="SAME",
                rngs=rngs,
            )
        elif spatial_dims == 2:
            self.local_conv = nnx.Conv(
                in_channels,  # Input: in_channels
                out_channels,  # Output: out_channels (match spectral branch)
                kernel_size=kernel_size,
                padding="SAME",
                rngs=rngs,
            )
        elif spatial_dims == 3:
            # For 3D, use standard Conv with 3D kernel
            self.local_conv = nnx.Conv(
                in_channels,  # Input: in_channels
                out_channels,  # Output: out_channels (match spectral branch)
                kernel_size=(kernel_size, kernel_size, kernel_size),
                padding="SAME",
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

        # Skip connection for residual learning
        self.skip = nnx.Linear(in_channels, out_channels, rngs=rngs)

    def _spectral_branch(self, x: Array) -> Array:
        """Apply spectral convolution branch."""
        spatial_dims = len(self.modes)
        x_ft = standardized_fft(x, spatial_dims)
        x_spectral = self.spectral_conv(x_ft)
        # Fixed target shape calculation
        target_shape = (x.shape[0], self.spectral_conv.out_channels, *x.shape[2:])
        return standardized_ifft(x_spectral, target_shape, spatial_dims)

    def _local_conv_branch(self, x: Array) -> Array:
        """
        Apply local convolution branch.

        Processes input through local convolution for high-frequency details.
        """
        # Apply local convolution: in_channels -> out_channels directly
        # Fix: Convert from NCHW to NHWC for Flax Conv, then back to NCHW
        x_nhwc = jnp.moveaxis(x, 1, -1)  # NCHW -> NHWC
        x_conv_nhwc = self.local_conv(x_nhwc)
        return jnp.moveaxis(x_conv_nhwc, -1, 1)  # NHWC -> NCHW

    def __call__(self, x: Array) -> Array:
        """
        Forward pass through local Fourier layer.

        Args:
            x: Input tensor (batch, in_channels, *spatial)

        Returns:
            Output tensor (batch, out_channels, *spatial)
        """
        # Spectral branch
        x_spectral = self._spectral_branch(x)

        # Local convolution branch
        x_local = self._local_conv_branch(x)

        # Skip connection with proper channel transformation
        x_skip = apply_linear_with_channel_transform(x, self.skip)

        # Combine all branches with learned weighting
        spectral_contribution = self.mixing_weight * x_spectral
        local_contribution = (1 - self.mixing_weight) * x_local
        combined = spectral_contribution + local_contribution + x_skip

        return self.activation(combined)


class LocalFourierNeuralOperator(nnx.Module):
    """
    Local Fourier Neural Operator combining global and local operations.

    This operator is designed for problems that require both:
    - Long-range dependencies (captured by Fourier operations)
    - Local features and fine details (captured by convolutions)

    Examples include:
    - Turbulent flows with both large-scale structures and small eddies
    - Wave propagation with local scattering and global modes
    - Multi-physics problems with different characteristic scales
    """

    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        modes: Sequence[int],
        num_layers: int = 4,
        kernel_size: int = 3,
        use_adaptive_mixing: bool = True,
        use_residual_connections: bool = True,
        activation: Callable = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize Local FNO.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Hidden layer width
            modes: Fourier modes for global operations
            num_layers: Number of Local Fourier layers
            kernel_size: Kernel size for local convolutions
            use_adaptive_mixing: Whether to use adaptive feature mixing
            use_residual_connections: Whether to use residual connections
            activation: Activation function
            rngs: Random number generator state
        """
        super().__init__()
        self.num_layers = num_layers
        self.use_residual_connections = use_residual_connections
        self.activation = activation

        # Lifting layer
        self.lifting = nnx.Linear(in_channels, hidden_channels, rngs=rngs)

        # Local Fourier layers
        for i in range(num_layers):
            layer = LocalFourierLayer(
                hidden_channels,
                hidden_channels,
                modes,
                kernel_size,
                activation=activation,
                rngs=rngs,
            )
            setattr(self, f"layer_{i}", layer)

            # Skip connection layers
            if use_residual_connections:
                skip = nnx.Linear(hidden_channels, hidden_channels, rngs=rngs)
                setattr(self, f"skip_{i}", skip)

        # Projection layer
        self.projection = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

        # Optional output refinement with local convolution
        spatial_dims = len(modes)
        self.output_refine = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=[3] * spatial_dims,
            padding=[(1, 1)] * spatial_dims,
            rngs=rngs,
        )

    def __call__(
        self, x: Array, return_intermediates: bool = False
    ) -> Array | tuple[Array, list[Array]]:
        """
        Forward pass through Local FNO.

        Args:
            x: Input tensor (batch, in_channels, *spatial_dims)
            return_intermediates: Whether to return intermediate layer outputs

        Returns:
            Output tensor or tuple of (output, intermediates)
        """
        # Lifting
        x = jnp.moveaxis(x, 1, -1)  # Move channels to last: (batch, *spatial, channels)
        x = self.lifting(x)  # Apply linear layer
        x = jnp.moveaxis(x, -1, 1)  # Move channels back: (batch, channels, *spatial)

        intermediates = []

        # Local Fourier layers
        for i in range(self.num_layers):
            layer = getattr(self, f"layer_{i}")

            # Apply Local Fourier layer
            x_out = layer(x)

            # Optional skip connection - FIXED: Handle channel dimensions properly
            if self.use_residual_connections:
                skip = getattr(self, f"skip_{i}")
                x_skip_input = jnp.moveaxis(x, 1, -1)  # Move channels to last
                x_skip = skip(x_skip_input)
                x_skip = jnp.moveaxis(x_skip, -1, 1)  # Move channels back
                x = self.activation(x_out + x_skip)
            else:
                x = self.activation(x_out)

            if return_intermediates:
                intermediates.append(x)

        # Projection to output channels - FIXED: Handle channel dimensions properly
        x_proj_input = jnp.moveaxis(x, 1, -1)  # Move channels to last
        x_proj = self.projection(x_proj_input)
        x = jnp.moveaxis(x_proj, -1, 1)  # Move channels back

        # Optional output refinement with data format conversion
        # Fix: Convert from NCHW to NHWC for Flax Conv, then back to NCHW
        x_refine_nhwc = jnp.moveaxis(x, 1, -1)  # NCHW -> NHWC
        x_refine_out_nhwc = self.output_refine(x_refine_nhwc)
        x_refine_out_nchw = jnp.moveaxis(x_refine_out_nhwc, -1, 1)  # NHWC -> NCHW
        x = x + x_refine_out_nchw

        if return_intermediates:
            return x, intermediates
        return x

    def analyze_global_local_contributions(self, x: Array) -> dict[str, list[Array]]:
        """
        Analyze global vs local contributions at each layer.

        Returns:
            Dictionary with global and local feature maps
        """
        x = jnp.moveaxis(x, 1, -1)  # Move channels to last: (batch, *spatial, channels)
        x = self.lifting(x)  # Apply linear layer
        x = jnp.moveaxis(x, -1, 1)  # Move channels back: (batch, channels, *spatial)

        analysis: dict[str, list[Array]] = {
            "global_features": [],
            "local_features": [],
            "mixing_weights": [],
            "layer_outputs": [],
        }

        for i in range(self.num_layers):
            layer = getattr(self, f"layer_{i}")

            # Get layer analysis
            global_feat, local_feat, mixing_weights = layer.get_mixing_analysis(x)

            analysis["global_features"].append(global_feat)
            analysis["local_features"].append(local_feat)
            analysis["mixing_weights"].append(mixing_weights)

            # Apply layer
            x_out = layer(x)
            if self.use_residual_connections:
                skip = getattr(self, f"skip_{i}")
                x = self.activation(x_out + skip(x))
            else:
                x = self.activation(x_out)

            analysis["layer_outputs"].append(x)

        return analysis


# Utility functions for different Local FNO configurations
def create_turbulence_local_fno(
    in_channels: int = 3,  # u, v, p
    out_channels: int = 3,
    modes: Sequence[int] = (32, 32),
    **kwargs,
) -> LocalFourierNeuralOperator:
    """Create Local FNO optimized for turbulent flow modeling."""
    return LocalFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=96,
        modes=modes,
        num_layers=6,
        kernel_size=5,  # Larger kernels for turbulent structures
        use_adaptive_mixing=True,
        use_residual_connections=True,
        **kwargs,
    )


def create_wave_local_fno(
    in_channels: int = 2,  # pressure, velocity
    out_channels: int = 2,
    modes: Sequence[int] = (64, 64),
    **kwargs,
) -> LocalFourierNeuralOperator:
    """Create Local FNO for wave propagation with scattering."""
    return LocalFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,
        modes=modes,
        num_layers=4,
        kernel_size=3,
        use_adaptive_mixing=True,
        **kwargs,
    )


def create_multiphysics_local_fno(
    in_channels: int = 5,  # Multiple physical quantities
    out_channels: int = 5,
    modes: Sequence[int] = (24, 24),
    **kwargs,
) -> LocalFourierNeuralOperator:
    """Create Local FNO for multi-physics problems."""
    return LocalFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=80,
        modes=modes,
        num_layers=5,
        kernel_size=7,  # Larger receptive field for multi-physics coupling
        use_adaptive_mixing=True,
        use_residual_connections=True,
        **kwargs,
    )


__all__ = [
    "LocalFourierLayer",
    "LocalFourierNeuralOperator",
    "create_multiphysics_local_fno",
    "create_turbulence_local_fno",
    "create_wave_local_fno",
]
