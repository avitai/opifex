# FILE PLACEMENT: opifex/neural/operators/fno/ufno.py
#
# U-Net Style Fourier Neural Operator (U-FNO) Implementation
# Multi-scale encoder-decoder architecture for hierarchical problems
#
# This file should be placed at: opifex/neural/operators/fno/ufno.py
# After placement, update opifex/neural/operators/fno/__init__.py to include:
# from .ufno import UFourierNeuralOperator, UFNOEncoderBlock, UFNODecoderBlock

"""
U-Net style Fourier Neural Operator (U-FNO) with clean architecture.

Redesigned with standardized tensor operations for consistent behavior.
"""

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jax import Array

from opifex.neural.operators.common.tensor_ops import (
    apply_linear_with_channel_transform,
    match_spatial_dimensions,
    standardized_fft,
    standardized_ifft,
    StandardSpectralConv,
)


class UFNOEncoderBlock(nnx.Module):
    """
    Clean U-FNO encoder block with standardized tensor operations.

    Performs: spectral convolution + skip connection + downsampling
    """

    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        downsample_factor: int = 2,
        activation: Callable = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = tuple(modes)
        self.activation = activation

        # Spectral convolution: in_channels -> out_channels
        self.spectral_conv = StandardSpectralConv(
            in_channels, out_channels, modes, rngs=rngs
        )

        # Skip connection: in_channels -> out_channels
        self.skip = nnx.Linear(in_channels, out_channels, rngs=rngs)

        # Downsampling: out_channels -> out_channels (no channel change)
        spatial_dims = len(modes)
        kernel_size = [downsample_factor] * spatial_dims
        strides = [downsample_factor] * spatial_dims
        self.downsample = nnx.Conv(
            out_channels,  # Input: out_channels (from combined spectral+skip)
            out_channels,  # Output: out_channels (maintain same channels)
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x: Array) -> tuple[Array, Array]:
        """
        Forward pass through encoder block with robust channel management.

        Args:
            x: Input tensor (batch, in_channels, *spatial)

        Returns:
            Tuple of (downsampled_output, skip_connection)
        """
        from opifex.neural.operators.common.tensor_ops import (
            validate_channel_compatibility,
        )

        # Validate input channels
        validate_channel_compatibility(
            x.shape[1],
            self.in_channels,
            f"UFNOEncoder(in={self.in_channels}, out={self.out_channels})",
        )

        original_shape = x.shape
        spatial_dims = len(self.modes)

        # Spectral branch: in_channels -> out_channels
        x_ft = standardized_fft(x, spatial_dims)
        x_spectral = self.spectral_conv(x_ft)
        target_shape = (original_shape[0], self.out_channels, *original_shape[2:])
        x_spectral = standardized_ifft(x_spectral, target_shape, spatial_dims)

        # Skip branch: in_channels -> out_channels
        x_skip = apply_linear_with_channel_transform(x, self.skip)

        # Combine and activate: both branches now have out_channels
        x_combined = self.activation(x_spectral + x_skip)

        # Validate combined output channels before downsampling
        validate_channel_compatibility(
            x_combined.shape[1],
            self.out_channels,
            "UFNOEncoder combined output (spectral+skip)",
        )

        # Downsampling: out_channels -> out_channels (no channel change)
        # Fix: Convert from NCHW to NHWC for Flax Conv, then back to NCHW
        x_combined_nhwc = jnp.moveaxis(x_combined, 1, -1)  # NCHW -> NHWC
        x_down_nhwc = self.downsample(x_combined_nhwc)
        x_down = jnp.moveaxis(x_down_nhwc, -1, 1)  # NHWC -> NCHW

        return x_down, x_combined  # Return combined for skip connection


class UFNODecoderBlock(nnx.Module):
    """
    Clean U-FNO decoder block with standardized tensor operations.

    Performs: upsampling + skip fusion + spectral convolution
    """

    @beartype
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        modes: Sequence[int],
        upsample_factor: int = 2,
        activation: Callable = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.modes = tuple(modes)
        self.activation = activation

        # Upsampling with consistent channel handling
        spatial_dims = len(modes)
        kernel_size = [upsample_factor] * spatial_dims
        strides = [upsample_factor] * spatial_dims
        self.upsample = nnx.ConvTranspose(
            in_channels,
            in_channels,  # Keep same channels for concatenation
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            rngs=rngs,
        )

        # Spectral convolution after concatenation
        combined_channels = in_channels + skip_channels
        self.spectral_conv = StandardSpectralConv(
            combined_channels, out_channels, modes, rngs=rngs
        )

        # Skip connection for concatenated channels
        self.skip = nnx.Linear(combined_channels, out_channels, rngs=rngs)

    def __call__(self, x: Array, skip: Array) -> Array:
        """
        Forward pass through decoder block.

        Args:
            x: Input from lower resolution (batch, in_channels, *spatial)
            skip: Skip connection from encoder (batch, skip_channels, *spatial)

        Returns:
            Upsampled and processed output (batch, out_channels, *spatial)
        """
        # Upsampling with data format conversion
        # Fix: Convert from NCHW to NHWC for Flax ConvTranspose, then back to NCHW
        x_nhwc = jnp.moveaxis(x, 1, -1)  # NCHW -> NHWC
        x_up_nhwc = self.upsample(x_nhwc)
        x_up = jnp.moveaxis(x_up_nhwc, -1, 1)  # NHWC -> NCHW

        # Handle spatial size mismatch with skip connection using standardized function
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = match_spatial_dimensions(x_up, skip.shape[2:])

        # Concatenate along channel dimension
        x_cat = jnp.concatenate([x_up, skip], axis=1)

        original_shape = x_cat.shape
        spatial_dims = len(self.modes)

        # Spectral branch with standardized operations
        x_ft = standardized_fft(x_cat, spatial_dims)
        x_spectral = self.spectral_conv(x_ft)
        x_spectral = standardized_ifft(
            x_spectral,
            (original_shape[0], self.spectral_conv.out_channels, *original_shape[2:]),
            spatial_dims,
        )

        # Skip branch with standardized channel handling
        x_skip = apply_linear_with_channel_transform(x_cat, self.skip)

        # Combine and activate
        return self.activation(x_spectral + x_skip)


class UFourierNeuralOperator(nnx.Module):
    """
    U-Net style Fourier Neural Operator with clean, standardized architecture.

    Features:
    - Consistent tensor dimension handling
    - Standardized spectral operations
    - Clean encoder-decoder structure
    - Proper channel management throughout
    """

    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        modes: Sequence[int],
        num_levels: int = 3,
        downsample_factor: int = 2,
        activation: Callable = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.modes = tuple(modes)
        self.activation = activation

        # Lifting layer
        self.lifting = nnx.Linear(in_channels, hidden_channels, rngs=rngs)

        # Calculate channel dimensions for each level
        # Use consistent channel progression
        channels = [hidden_channels * (2**i) for i in range(num_levels)]

        # Encoder path
        for i in range(num_levels - 1):
            encoder = UFNOEncoderBlock(
                channels[i],  # Input channels at this level
                channels[i + 1],  # Output channels (double for next level)
                modes,
                downsample_factor,
                activation,
                rngs=rngs,
            )
            setattr(self, f"encoder_{i}", encoder)

        # Bottleneck (deepest level) with standardized spectral conv
        bottleneck_channels = channels[-1]
        self.bottleneck_spectral = StandardSpectralConv(
            bottleneck_channels, bottleneck_channels, modes, rngs=rngs
        )
        self.bottleneck_skip = nnx.Linear(
            bottleneck_channels, bottleneck_channels, rngs=rngs
        )

        # Decoder path
        for i in range(num_levels - 1):
            # Decoder configuration: deeper -> shallower
            level_idx = num_levels - 2 - i
            decoder = UFNODecoderBlock(
                channels[level_idx + 1],  # Input from deeper level
                channels[
                    level_idx + 1
                ],  # Skip channels (same as output channels of encoder)
                channels[level_idx],  # Output channels (reduce by half)
                modes,
                downsample_factor,
                activation,
                rngs=rngs,
            )
            setattr(self, f"decoder_{i}", decoder)

        # Projection layer
        self.projection = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """
        Forward pass through U-FNO with clean architecture.

        Args:
            x: Input tensor (batch, in_channels, *spatial)

        Returns:
            Output tensor (batch, out_channels, *spatial)
        """
        # Lifting with standardized channel handling
        x = apply_linear_with_channel_transform(x, self.lifting)

        # Encoder path with skip connections
        skips = []
        for i in range(self.num_levels - 1):
            encoder = getattr(self, f"encoder_{i}")
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck processing with standardized operations
        original_shape = x.shape
        spatial_dims = len(self.modes)

        x_ft = standardized_fft(x, spatial_dims)
        x_spectral = self.bottleneck_spectral(x_ft)
        x_spectral = standardized_ifft(x_spectral, original_shape, spatial_dims)

        x_skip = apply_linear_with_channel_transform(x, self.bottleneck_skip)
        x = self.activation(x_spectral + x_skip)

        # Decoder path with skip connections (reverse order)
        for i in range(self.num_levels - 1):
            decoder = getattr(self, f"decoder_{i}")
            skip = skips[-(i + 1)]  # Use skips in reverse order
            x = decoder(x, skip)

        # Final projection with standardized channel handling
        return apply_linear_with_channel_transform(x, self.projection)


# Utility functions with clean configurations
def create_shallow_ufno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    modes: Sequence[int] = (16, 16),
    **kwargs,
) -> UFourierNeuralOperator:
    """Create shallow U-FNO (2 levels) for simple multi-scale problems."""
    return UFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        num_levels=2,
        **kwargs,
    )


def create_deep_ufno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 32,
    modes: Sequence[int] = (32, 32),
    **kwargs,
) -> UFourierNeuralOperator:
    """Create deep U-FNO (5 levels) for complex multi-scale problems."""
    return UFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        num_levels=5,
        **kwargs,
    )


def create_turbulence_ufno(
    in_channels: int = 4,  # u, v, p, density
    out_channels: int = 3,  # u_next, v_next, p_next
    **kwargs,
) -> UFourierNeuralOperator:
    """Create U-FNO optimized for turbulent flow modeling."""
    return UFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=48,
        modes=(32, 32),
        num_levels=4,
        **kwargs,
    )


__all__ = [
    "UFNODecoderBlock",
    "UFNOEncoderBlock",
    "UFourierNeuralOperator",
    "create_deep_ufno",
    "create_shallow_ufno",
    "create_turbulence_ufno",
]
