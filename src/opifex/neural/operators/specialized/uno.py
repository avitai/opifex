"""
U-Net Neural Operator (UNO) Implementation

This module implements the U-Net style Neural Operator, which combines the multi-scale
feature extraction capabilities of U-Net architectures with spectral convolutions
for operator learning.

Key Features:
- U-Net encoder-decoder architecture with skip connections
- 2D Spectral convolutions using existing Opifex FourierLayer
- Multi-scale feature processing
- Zero-shot super-resolution capabilities
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.fno.base import FourierLayer


class UNetBlock(nnx.Module):
    """U-Net style convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_norm: bool = True,
        activation: Callable = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize U-Net block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            use_norm: Whether to use normalization
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__()

        self.activation = activation
        self.use_norm = use_norm

        # First convolution
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride),
            padding="SAME",
            rngs=rngs,
        )

        # Second convolution
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        # Normalization layers
        if use_norm:
            self.norm1 = nnx.GroupNorm(
                num_groups=min(8, out_channels),
                num_features=out_channels,
                rngs=rngs,
            )
            self.norm2 = nnx.GroupNorm(
                num_groups=min(8, out_channels),
                num_features=out_channels,
                rngs=rngs,
            )

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        """Apply U-Net block.

        Args:
            x: Input tensor
            deterministic: Whether to use deterministic mode

        Returns:
            Processed tensor
        """
        # First convolution and normalization
        h = self.conv1(x)

        if self.use_norm:
            h = self.norm1(h)

        h = self.activation(h)

        # Second convolution and normalization
        h = self.conv2(h)

        if self.use_norm:
            h = self.norm2(h)

        return self.activation(h)


class UNeuralOperator(nnx.Module):
    """U-Net Neural Operator for operator learning.

    Combines U-Net architecture with Fourier layers for effective
    operator learning with multi-scale feature processing.

    This implementation follows the Opifex framework patterns and uses
    existing FourierLayer components for 2D spectral convolutions.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 64,
        modes: int = 16,
        n_layers: int = 4,
        use_spectral: bool = True,
        activation: Callable = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize UNO.

        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            hidden_channels: Base number of hidden channels
            modes: Number of Fourier modes for spectral convolutions
            n_layers: Number of U-Net layers (encoder/decoder pairs)
            use_spectral: Whether to use spectral convolutions
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.modes = modes
        self.n_layers = n_layers
        self.use_spectral = use_spectral
        self.activation = activation

        # Input projection
        self.input_proj = nnx.Conv(
            in_features=input_channels,
            out_features=hidden_channels,
            kernel_size=(1, 1),
            rngs=rngs,
        )

        # Build architecture components
        self.encoder_blocks, encoder_channels = self._build_encoder(
            hidden_channels, n_layers, activation, rngs
        )

        if use_spectral:
            self.spectral_layers = self._build_spectral_layers(
                encoder_channels[-1], modes, activation, rngs
            )

        self.decoder_blocks, self.upsample_convs, self.needs_upsample = (
            self._build_decoder(
                encoder_channels, hidden_channels, n_layers, activation, rngs
            )
        )

        # Output projection
        self.output_proj = nnx.Conv(
            in_features=hidden_channels,
            out_features=output_channels,
            kernel_size=(1, 1),
            rngs=rngs,
        )

    def _build_encoder(
        self,
        hidden_channels: int,
        n_layers: int,
        activation: Callable,
        rngs: nnx.Rngs,
    ) -> tuple[nnx.List[UNetBlock], list[int]]:
        """Build encoder blocks."""
        encoder_blocks_temp = []
        encoder_channels = [hidden_channels]
        current_channels = hidden_channels

        for i in range(n_layers):
            if i < n_layers - 1:
                stride = 2
                next_channels = current_channels * 2
            else:
                stride = 1
                next_channels = current_channels

            block = UNetBlock(
                in_channels=current_channels,
                out_channels=next_channels,
                stride=stride,
                activation=activation,
                rngs=rngs,
            )
            encoder_blocks_temp.append(block)
            encoder_channels.append(next_channels)
            current_channels = next_channels

        return nnx.List(encoder_blocks_temp), encoder_channels

    def _build_spectral_layers(
        self,
        channels: int,
        modes: int,
        activation: Callable,
        rngs: nnx.Rngs,
    ) -> nnx.List[FourierLayer]:
        """Build spectral layers."""
        spectral_layers_temp = []
        for _ in range(2):
            layer = FourierLayer(
                in_channels=channels,
                out_channels=channels,
                modes=modes,
                activation=activation,
                rngs=rngs,
            )
            spectral_layers_temp.append(layer)
        return nnx.List(spectral_layers_temp)

    def _build_decoder(
        self,
        encoder_channels: list[int],
        hidden_channels: int,
        n_layers: int,
        activation: Callable,
        rngs: nnx.Rngs,
    ) -> tuple[nnx.List[UNetBlock], nnx.List[nnx.Conv], nnx.List[bool]]:
        """Build decoder blocks."""
        decoder_blocks_temp = []
        upsample_convs_temp = []
        needs_upsample_temp = []

        skip_channels = list(reversed(encoder_channels[:-1]))
        current_channels = encoder_channels[-1]

        for i in range(n_layers):
            if i < n_layers - 1:
                upsampled_ch = current_channels // 2
                skip_ch = skip_channels[i]
                in_ch = upsampled_ch + skip_ch
                out_ch = upsampled_ch

                upsample_conv = nnx.Conv(
                    in_features=current_channels,
                    out_features=upsampled_ch,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="SAME",
                    rngs=rngs,
                )
                upsample_convs_temp.append(upsample_conv)
                needs_upsample_temp.append(True)
                current_channels = upsampled_ch
            else:
                skip_ch = skip_channels[i]
                in_ch = current_channels + skip_ch
                out_ch = hidden_channels
                needs_upsample_temp.append(False)

            block = UNetBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                activation=activation,
                rngs=rngs,
            )
            decoder_blocks_temp.append(block)

            if i == n_layers - 1:
                current_channels = out_ch

        return (
            nnx.List(decoder_blocks_temp),
            nnx.List(upsample_convs_temp),
            nnx.List(needs_upsample_temp),
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Forward pass through UNO.

        Args:
            x: Input tensor of shape (batch, height, width, channels)
            deterministic: Whether to use deterministic mode

        Returns:
            Output tensor of shape (batch, height, width, output_channels)
        """
        # Input projection
        x = self.input_proj(x)

        # Store skip connections
        skip_connections = [x]

        # Encoder path
        h = x
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, deterministic=deterministic)
            if i < len(self.encoder_blocks) - 1:
                skip_connections.append(h)

        # Apply Fourier layers at bottleneck using existing Opifex components
        if self.use_spectral:
            for layer in self.spectral_layers:
                # Convert from (batch, height, width, channels) to (batch, channels, height, width)  # noqa: E501
                h_fourier = jnp.transpose(h, (0, 3, 1, 2))

                # Apply Fourier layer
                h_fourier = layer(h_fourier)

                # Convert back to (batch, height, width, channels)
                h = jnp.transpose(h_fourier, (0, 2, 3, 1))

        # Decoder path
        upsample_idx = 0
        for i, (block, skip) in enumerate(
            zip(self.decoder_blocks, reversed(skip_connections), strict=False)
        ):
            # Upsample if not the final layer
            if self.needs_upsample[i]:
                # First upsample spatially (2x)
                h_upsampled = jax.image.resize(
                    h,
                    (h.shape[0], h.shape[1] * 2, h.shape[2] * 2, h.shape[3]),
                    method="bilinear",
                )
                # Then apply convolution to reduce channels
                h = self.upsample_convs[upsample_idx](h_upsampled)
                upsample_idx += 1

            # Ensure spatial dimensions match for concatenation
            if h.shape[1:3] != skip.shape[1:3]:
                # Resize h to match skip connection spatial dimensions
                target_shape = (h.shape[0], skip.shape[1], skip.shape[2], h.shape[3])
                h = jax.image.resize(h, target_shape, method="bilinear")

            # Concatenate skip connection
            h = jnp.concatenate([h, skip], axis=-1)

            # Apply decoder block
            h = block(h, deterministic=deterministic)

            # Output projection
        return self.output_proj(h)


def create_uno(
    input_channels: int = 1,
    output_channels: int = 1,
    hidden_channels: int = 64,
    modes: int = 16,
    n_layers: int = 4,
    *,
    rngs: nnx.Rngs,
) -> UNeuralOperator:
    """Create a UNO model with standard configuration.

    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        hidden_channels: Base number of hidden channels
        modes: Number of Fourier modes
        n_layers: Number of U-Net layers
        rngs: Random number generators

    Returns:
        Configured UNO model
    """
    return UNeuralOperator(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        n_layers=n_layers,
        use_spectral=True,
        rngs=rngs,
    )


# Export the main components
__all__ = [
    "UNetBlock",
    "UNeuralOperator",
    "create_uno",
]
