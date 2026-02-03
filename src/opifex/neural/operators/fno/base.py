"""Core Fourier Neural Operator (FNO) components.

This module contains the fundamental building blocks for Fourier Neural Operators,
including spectral convolution layers, Fourier layers, and the main FNO architecture.
Fully compliant with Flax NNX best practices.

MODERNIZATION APPLIED:
- Full Flax NNX compliance with proper RNG handling
- Optimized spectral convolution with complex parameter initialization
- Enhanced 2D/3D support with simplified channel handling
- Performance-optimized residual connection architecture
- Robust edge case handling for various input dimensions
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


class FourierSpectralConvolution(nnx.Module):
    """Spectral convolution layer for Fourier Neural Operators.

    Performs convolution in the Fourier domain using learnable spectral weights.
    This is the core building block of FNO architectures, fully compliant with
    modern Flax NNX patterns.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spectral convolution layer following NNX patterns.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes to use
            rngs: Random number generators (keyword-only)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Initialize complex spectral weights using proper complex initialization
        # Xavier initialization split between real/imaginary components
        # Shape: (in_channels, out_channels, modes)
        fan_in = in_channels
        fan_out = out_channels
        # Xavier initialization variance, split between real and imaginary parts
        variance = 2.0 / (fan_in + fan_out)
        std = jnp.sqrt(variance / 2)  # Divide by 2 for real and imaginary components

        # Generate separate real and imaginary parts for proper complex initialization
        key_real, key_imag = jax.random.split(rngs.params())
        real_part = (
            jax.random.normal(key_real, (in_channels, out_channels, modes)) * std
        )
        imag_part = (
            jax.random.normal(key_imag, (in_channels, out_channels, modes)) * std
        )

        # JAX X64 handles complex precision naturally
        self.weights = nnx.Param(real_part + 1j * imag_part)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply spectral convolution.

        Following NNX best practices, this method does NOT include rngs parameter
        as all random state is managed during initialization.

        Args:
            x: Input in spectral domain (batch, in_channels, spectral_size)

        Returns:
            Output in spectral domain (batch, out_channels, spectral_size)
            Maintains the same spectral size as input
        """
        batch_size, _, spectral_size = x.shape

        # Handle edge case: no modes to process
        effective_modes = min(self.modes, spectral_size)
        if effective_modes == 0:
            return jnp.zeros(
                (batch_size, self.out_channels, spectral_size), dtype=x.dtype
            )

        # Use only effective modes
        x_effective = x[:, :, :effective_modes]
        weights_effective = self.weights.value[:, :, :effective_modes]

        # Vectorized computation using einsum for better GPU performance
        output_effective = jnp.einsum("bik,iok->bok", x_effective, weights_effective)

        # Pad output back to original spectral size if necessary
        if effective_modes < spectral_size:
            output = jnp.zeros(
                (batch_size, self.out_channels, spectral_size),
                dtype=output_effective.dtype,
            )
            output = output.at[:, :, :effective_modes].set(output_effective)
        else:
            output = output_effective

        return output


class FourierLayer(nnx.Module):
    """Fourier layer combining spectral convolution with activation.

    This layer performs:
    1. FFT to transform input to spectral domain
    2. Spectral convolution
    3. IFFT to transform back to spatial domain
    4. Linear transformation and activation with proper residual connection

    Fully compliant with modern Flax NNX patterns.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        *,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ):
        """Initialize Fourier layer following NNX patterns.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes
            activation: Activation function
            rngs: Random number generators (keyword-only)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.activation = activation

        # Spectral convolution in Fourier domain
        self.spectral_conv = FourierSpectralConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            rngs=rngs,
        )

        # Linear transformation for residual connection
        self.linear = nnx.Linear(
            in_features=in_channels, out_features=out_channels, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply Fourier layer transformation with performance optimizations.

        Following NNX best practices. Memory donation removed to avoid conflicts
        with gradient computation.

        Args:
            x: Input tensor (batch, in_channels, *spatial_dims)

        Returns:
            Output tensor (batch, out_channels, *spatial_dims)
        """
        # Spectral pathway: FFT -> SpectralConv -> IFFT
        spectral_output = self._apply_spectral_transform(x)

        # Skip connection pathway: Linear transform
        skip_output = self._apply_skip_connection(x)

        # Proper FNO residual connection: spectral + skip + activation
        combined = spectral_output + skip_output
        # JAX X64 handles precision naturally
        return self.activation(combined)

    def _apply_spectral_transform(self, x: jax.Array) -> jax.Array:
        """Apply spectral transformation.

        Gradient checkpointing removed to avoid tracer leaks in JAX transformations.
        """
        spatial_dims = x.shape[2:]

        if len(spatial_dims) == 1:
            return self._spectral_1d(x)
        if len(spatial_dims) == 2:
            return self._spectral_2d(x)
        if len(spatial_dims) == 3:
            return self._spectral_3d(x)
        raise ValueError(
            f"Unsupported number of spatial dimensions: {len(spatial_dims)}"
        )

    def _spectral_1d(self, x: jax.Array) -> jax.Array:
        """1D spectral transform."""
        batch_size = x.shape[0]
        grid_size = x.shape[-1]

        # 1D FFT to spectral domain
        x_ft = jnp.fft.rfft(x, axis=-1)

        # Apply spectral convolution
        out_ft = self.spectral_conv(x_ft)

        # Pad back to original frequency resolution if needed
        if out_ft.shape[-1] < grid_size // 2 + 1:
            out_ft_padded = jnp.zeros(
                (batch_size, self.out_channels, grid_size // 2 + 1), dtype=out_ft.dtype
            )
            out_ft_padded = out_ft_padded.at[:, :, : out_ft.shape[-1]].set(out_ft)
            out_ft = out_ft_padded

        # IFFT back to spatial domain - JAX X64 handles precision naturally
        return jnp.fft.irfft(out_ft, n=grid_size, axis=-1)

    def _spectral_2d(self, x: jax.Array) -> jax.Array:
        """2D spectral transform - MODERNIZED VERSION."""
        batch_size = x.shape[0]
        h, w = x.shape[-2:]

        # 2D FFT to spectral domain
        x_ft = jnp.fft.rfft2(x, axes=(-2, -1))

        # Keep only low-frequency modes
        ft_h, ft_w = x_ft.shape[-2:]
        modes_h = min(self.modes, ft_h)
        modes_w = min(self.modes, ft_w)

        # Take low-frequency block
        x_ft_truncated = x_ft[:, :, :modes_h, :modes_w]

        # Reshape for spectral convolution: treat as batch of 1D spectral data
        # Shape: (batch, channels, modes_h * modes_w)
        x_ft_flat = x_ft_truncated.reshape(batch_size, self.in_channels, -1)

        # Apply spectral convolution
        out_ft_flat = self.spectral_conv(x_ft_flat)

        # Reshape back to 2D frequency domain
        out_ft_truncated = out_ft_flat.reshape(
            batch_size, self.out_channels, modes_h, modes_w
        )

        # Pad back to original frequency resolution - JAX X64 handles precision
        out_ft_padded = jnp.zeros(
            (batch_size, self.out_channels, ft_h, ft_w), dtype=out_ft_truncated.dtype
        )
        out_ft_padded = out_ft_padded.at[:, :, :modes_h, :modes_w].set(out_ft_truncated)

        # IFFT back to spatial domain - JAX X64 handles precision naturally
        return jnp.fft.irfft2(out_ft_padded, s=(h, w), axes=(-2, -1))

    def _spectral_3d(self, x: jax.Array) -> jax.Array:
        """3D spectral transform - ENHANCED VERSION."""
        batch_size = x.shape[0]
        d, h, w = x.shape[-3:]

        # 3D FFT to spectral domain
        x_ft = jnp.fft.rfftn(x, axes=(-3, -2, -1))

        # Keep only low-frequency modes in all dimensions
        ft_d, ft_h, ft_w = x_ft.shape[-3:]
        modes_d = min(self.modes, ft_d)
        modes_h = min(self.modes, ft_h)
        modes_w = min(self.modes, ft_w)

        # Take low-frequency block
        x_ft_truncated = x_ft[:, :, :modes_d, :modes_h, :modes_w]

        # Reshape for spectral convolution: treat as batch of 1D spectral data
        # Shape: (batch, channels, modes_d * modes_h * modes_w)
        x_ft_flat = x_ft_truncated.reshape(batch_size, self.in_channels, -1)

        # Apply spectral convolution
        out_ft_flat = self.spectral_conv(x_ft_flat)

        # Reshape back to 3D frequency domain
        out_ft_truncated = out_ft_flat.reshape(
            batch_size, self.out_channels, modes_d, modes_h, modes_w
        )

        # Pad back to original frequency resolution - JAX X64 handles precision
        out_ft_padded = jnp.zeros(
            (batch_size, self.out_channels, ft_d, ft_h, ft_w),
            dtype=out_ft_truncated.dtype,
        )
        out_ft_padded = out_ft_padded.at[:, :, :modes_d, :modes_h, :modes_w].set(
            out_ft_truncated
        )

        # IFFT back to spatial domain - JAX X64 handles precision naturally
        return jnp.fft.irfftn(out_ft_padded, s=(d, h, w), axes=(-3, -2, -1))

    def _apply_skip_connection(self, x: jax.Array) -> jax.Array:
        """Apply skip connection using linear transformation.

        This handles channel dimension changes in the skip connection.
        """
        # For spatial inputs, apply linear layer along channel dimension
        # Input shape: (batch, in_channels, *spatial_dims)
        # We need to apply linear transform to the channel dimension

        # Move channel dim to last for linear layer
        # (batch, in_channels, *spatial) -> (batch, *spatial, in_channels)
        spatial_dims = len(x.shape) - 2
        perm = [0, *list(range(2, 2 + spatial_dims)), 1]
        x_transposed = jnp.transpose(x, perm)

        # Apply linear transformation
        result_transposed = self.linear(x_transposed)

        # Move channel dim back to position 1
        # (batch, *spatial, out_channels) -> (batch, out_channels, *spatial)
        inv_perm = [0, spatial_dims + 1, *list(range(1, spatial_dims + 1))]

        # JAX X64 handles precision naturally
        return jnp.transpose(result_transposed, inv_perm)


class FourierNeuralOperator(nnx.Module):
    """Fourier Neural Operator for learning solution operators of PDEs.

    Implements the complete FNO architecture with optional tensor factorization
    and mixed precision training capabilities. Fully compliant with modern
    Flax NNX patterns.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        modes: int,
        num_layers: int,
        *,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        factorization_type: str | None = None,
        factorization_rank: int | None = None,
        use_mixed_precision: bool = False,
        rngs: nnx.Rngs,
    ):
        """Initialize Fourier Neural Operator following NNX patterns.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Number of hidden channels
            modes: Number of Fourier modes
            num_layers: Number of Fourier layers
            activation: Activation function
            factorization_type: Optional tensor factorization ('tucker', 'cp')
            factorization_rank: Rank for tensor factorization
            use_mixed_precision: Whether to use mixed precision
            rngs: Random number generators (keyword-only)
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.modes = modes
        self.num_layers = num_layers
        self.activation = activation
        self.factorization_type = factorization_type
        self.factorization_rank = factorization_rank
        self.use_mixed_precision = use_mixed_precision

        # Input projection (lifting) - JAX X64 handles precision naturally
        self.input_projection = nnx.Linear(
            in_features=in_channels,
            out_features=hidden_channels,
            rngs=rngs,
        )

        # Fourier layers - use nnx.List for proper Flax 0.12.0+ compatibility
        layers = []
        for _i in range(num_layers):
            layer = self._create_fourier_layer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                modes=modes,
                activation=activation,
                rngs=rngs,
            )
            layers.append(layer)
        self.fourier_layers = nnx.List(layers)

        # Output projection - JAX X64 handles precision naturally
        self.output_projection = nnx.Linear(
            in_features=hidden_channels,
            out_features=out_channels,
            rngs=rngs,
        )

    def _create_fourier_layer(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        activation: Callable[[jax.Array], jax.Array],
        rngs: nnx.Rngs,
    ) -> Any:
        """Create a Fourier layer with optional factorization."""
        # Try to use factorized layer if available and requested
        if self.factorization_type is not None:
            try:
                from .factorized import FactorizedFourierLayer

                return FactorizedFourierLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    modes=modes,
                    factorization_type=self.factorization_type,
                    factorization_rank=self.factorization_rank or modes // 2,
                    activation=activation,
                    rngs=rngs,
                )
            except ImportError:
                # Fall back to regular Fourier layer if factorized not available
                pass

        return FourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            activation=activation,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply Fourier Neural Operator.

        Following NNX best practices, this method does NOT include rngs parameter
        as all random state is managed during initialization.

        Args:
            x: Input tensor (batch, in_channels, *spatial_dims)

        Returns:
            Output tensor (batch, out_channels, *spatial_dims)
        """
        # Input projection (lifting) - simplified channel handling
        x = self._apply_pointwise_linear(x, self.input_projection)

        # Apply Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)

        # Output projection
        return self._apply_pointwise_linear(x, self.output_projection)

    def _apply_pointwise_linear(
        self, x: jax.Array, linear_layer: nnx.Linear
    ) -> jax.Array:
        """Apply linear layer pointwise - simplified channel handling."""
        # Move channels to last dimension for linear layer
        x_permuted = jnp.moveaxis(x, 1, -1)  # (batch, *spatial_dims, channels)

        # Apply linear layer (operates on last dimension)
        out_permuted = linear_layer(x_permuted)  # (batch, *spatial_dims, out_channels)

        # Move channels back to second dimension
        return jnp.moveaxis(out_permuted, -1, 1)  # (batch, out_channels, *spatial_dims)

    def count_parameters(self) -> int:
        """Count total number of trainable parameters in the model."""
        total_params = 0

        # Count input projection parameters
        total_params += (
            self.input_projection.in_features * self.input_projection.out_features
        )
        if (
            hasattr(self.input_projection, "bias")
            and self.input_projection.bias is not None
        ):
            total_params += self.input_projection.out_features

        # Count parameters in each Fourier layer
        for layer in self.fourier_layers:
            if hasattr(layer, "count_parameters"):
                total_params += layer.count_parameters()
            else:
                # Estimate parameters for basic Fourier layer
                # Spectral convolution + skip connection
                spectral_params = (
                    layer.spectral_conv.in_channels
                    * layer.spectral_conv.out_channels
                    * layer.spectral_conv.modes
                )
                skip_params = layer.linear.in_features * layer.linear.out_features
                if hasattr(layer.linear, "bias") and layer.linear.bias is not None:
                    skip_params += layer.linear.out_features
                total_params += spectral_params + skip_params

        # Count output projection parameters
        total_params += (
            self.output_projection.in_features * self.output_projection.out_features
        )
        if (
            hasattr(self.output_projection, "bias")
            and self.output_projection.bias is not None
        ):
            total_params += self.output_projection.out_features

        return total_params
