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

    Note: Weights are stored as separate real/imaginary nnx.Param arrays to
    avoid the JAX complex gradient convention issue (optax issue #196). JAX's
    ``jax.grad`` returns the conjugate gradient for complex parameters, which
    causes standard optimizers to update the imaginary part in the wrong
    direction. Storing as real pairs ensures correct optimization.
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

        # Li et al. (2021) initialization: scale = 1 / (in_channels * out_channels)
        # Keeps spectral weights small — critical for convergence in Fourier domain.
        scale = 1.0 / (in_channels * out_channels)
        key_real, key_imag = jax.random.split(rngs.params())
        shape = (in_channels, out_channels, modes)
        self.weights_real = nnx.Param(scale * jax.random.uniform(key_real, shape))
        self.weights_imag = nnx.Param(scale * jax.random.uniform(key_imag, shape))

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
            return jnp.zeros((batch_size, self.out_channels, spectral_size), dtype=x.dtype)

        # Use only effective modes
        x_effective = x[:, :, :effective_modes]
        w_complex = (
            self.weights_real[...][:, :, :effective_modes]
            + 1j * self.weights_imag[...][:, :, :effective_modes]
        )

        # Vectorized computation using einsum for better GPU performance
        output_effective = jnp.einsum("bik,iok->bok", x_effective, w_complex)

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
        spatial_dims: int = 2,
        rngs: nnx.Rngs,
    ):
        """Initialize Fourier layer following NNX patterns.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes
            activation: Activation function
            spatial_dims: Number of spatial dimensions (1, 2, or 3). Controls which
                spectral weights are allocated — avoids dead parameters.
            rngs: Random number generators (keyword-only)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.activation = activation
        self.spatial_dims = spatial_dims

        # Spectral weights — only allocate for the target dimensionality.
        # Stored as separate real/imaginary Params to avoid JAX complex gradient
        # convention issue (optax issue #196) — see FourierSpectralConvolution docstring.
        scale = 1.0 / (in_channels * out_channels)

        if spatial_dims == 1:
            self.spectral_conv = FourierSpectralConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                modes=modes,
                rngs=rngs,
            )
        elif spatial_dims == 2:
            shape_2d = (in_channels, out_channels, modes, modes)
            k1r, k1i, k2r, k2i = jax.random.split(rngs.params(), 4)
            self.weights_2d_1_real = nnx.Param(scale * jax.random.uniform(k1r, shape_2d))
            self.weights_2d_1_imag = nnx.Param(scale * jax.random.uniform(k1i, shape_2d))
            self.weights_2d_2_real = nnx.Param(scale * jax.random.uniform(k2r, shape_2d))
            self.weights_2d_2_imag = nnx.Param(scale * jax.random.uniform(k2i, shape_2d))
        else:
            # 3D: allocate 1D spectral conv (used via flatten for 3D)
            self.spectral_conv = FourierSpectralConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                modes=modes,
                rngs=rngs,
            )

        # Linear transformation for residual/skip connection (1x1 conv equivalent)
        self.linear = nnx.Linear(in_features=in_channels, out_features=out_channels, rngs=rngs)

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

        Dispatches based on ``self.spatial_dims`` set at init time, which
        determines which spectral weights were allocated.
        """
        if self.spatial_dims == 1:
            return self._spectral_1d(x)
        if self.spatial_dims == 2:
            return self._spectral_2d(x)
        if self.spatial_dims == 3:
            return self._spectral_3d(x)
        raise ValueError(f"Unsupported number of spatial dimensions: {self.spatial_dims}")

    def _spectral_1d(self, x: jax.Array) -> jax.Array:
        """1D spectral transform."""
        batch_size = x.shape[0]
        grid_size = x.shape[-1]

        # 1D FFT to spectral domain
        x_ft = jnp.fft.rfft(x, axis=-1)

        # Apply spectral convolution
        out_ft = self.spectral_conv(x_ft)

        # Pad back to original frequency resolution if needed
        from opifex.neural.operators.common.tensor_ops import pad_spectral_1d

        out_ft = pad_spectral_1d(out_ft, batch_size, self.out_channels, grid_size // 2 + 1)

        # IFFT back to spatial domain - JAX X64 handles precision naturally
        return jnp.fft.irfft(out_ft, n=grid_size, axis=-1)

    def _spectral_2d(self, x: jax.Array) -> jax.Array:
        """2D spectral transform following Li et al. (2021).

        Uses BOTH quadrants of the rfft2 spectrum (positive and negative
        y-frequencies) with separate weight tensors and proper 2D einsum.
        """
        batch_size = x.shape[0]
        h, w = x.shape[-2:]

        # 2D FFT to spectral domain (rfft2 along last two axes)
        x_ft = jnp.fft.rfft2(x, axes=(-2, -1))

        ft_h, ft_w = x_ft.shape[-2:]
        modes_h = min(self.modes, ft_h)
        modes_w = min(self.modes, ft_w)

        # Construct complex weights from real/imaginary components
        w1 = (
            self.weights_2d_1_real[...][:, :, :modes_h, :modes_w]
            + 1j * self.weights_2d_1_imag[...][:, :, :modes_h, :modes_w]
        )
        w2 = (
            self.weights_2d_2_real[...][:, :, :modes_h, :modes_w]
            + 1j * self.weights_2d_2_imag[...][:, :, :modes_h, :modes_w]
        )

        # Quadrant 1: positive y-frequencies (top-left of rfft2 spectrum)
        x_ft_1 = x_ft[:, :, :modes_h, :modes_w]
        out_1 = jnp.einsum("bixy,ioxy->boxy", x_ft_1, w1)

        # Quadrant 2: negative y-frequencies (bottom-left of rfft2 spectrum)
        x_ft_2 = x_ft[:, :, -modes_h:, :modes_w]
        out_2 = jnp.einsum("bixy,ioxy->boxy", x_ft_2, w2)

        # Assemble output in full frequency space
        out_ft = jnp.zeros((batch_size, self.out_channels, ft_h, ft_w), dtype=out_1.dtype)
        out_ft = out_ft.at[:, :, :modes_h, :modes_w].set(out_1)
        out_ft = out_ft.at[:, :, -modes_h:, :modes_w].set(out_2)

        # IFFT back to spatial domain
        return jnp.fft.irfft2(out_ft, s=(h, w), axes=(-2, -1))

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
        out_ft_padded = out_ft_padded.at[:, :, :modes_d, :modes_h, :modes_w].set(out_ft_truncated)

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
        domain_padding: int = 0,
        spatial_dims: int = 2,
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
            domain_padding: Pixels to pad spatial dims (reduces Gibbs phenomenon
                for non-periodic problems). Set to 2 for Darcy flow.
            spatial_dims: Number of spatial dimensions (1, 2, or 3). Determines
                which spectral weights are allocated per layer.
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
        self.domain_padding = domain_padding
        self.spatial_dims = spatial_dims

        # Input projection (lifting): in_channels -> hidden_channels
        self.input_projection = nnx.Linear(
            in_features=in_channels,
            out_features=hidden_channels,
            rngs=rngs,
        )

        # Fourier layers
        layers = []
        for _i in range(num_layers):
            layer = self._create_fourier_layer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                modes=modes,
                activation=activation,
                spatial_dims=spatial_dims,
                rngs=rngs,
            )
            layers.append(layer)
        self.fourier_layers = nnx.List(layers)

        # Two-layer output projection (Li et al.):
        # hidden_channels -> 128 -> GELU -> out_channels
        projection_width = max(128, hidden_channels)
        self.output_projection_1 = nnx.Linear(
            in_features=hidden_channels,
            out_features=projection_width,
            rngs=rngs,
        )
        self.output_projection_2 = nnx.Linear(
            in_features=projection_width,
            out_features=out_channels,
            rngs=rngs,
        )

    def _create_fourier_layer(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        activation: Callable[[jax.Array], jax.Array],
        spatial_dims: int,
        rngs: nnx.Rngs,
    ) -> Any:
        """Create a Fourier layer with optional factorization."""
        # Try to use factorized layer if available and requested
        if self.factorization_type is not None:
            try:
                from opifex.neural.operators.fno.factorized import (
                    FactorizedFourierLayer,
                )

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
            spatial_dims=spatial_dims,
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
        # Input projection (lifting)
        x = self._apply_pointwise_linear(x, self.input_projection)

        # Domain padding for non-periodic problems (reduces Gibbs phenomenon)
        if self.domain_padding > 0:
            pad_widths = [(0, 0), (0, 0)] + [(0, self.domain_padding)] * (x.ndim - 2)
            x = jnp.pad(x, pad_widths, mode="constant")

        # Apply Fourier layers (no activation on last layer per Li et al.)
        for i, layer in enumerate(self.fourier_layers):
            if i < len(self.fourier_layers) - 1:
                x = layer(x)
            else:
                # Last layer: spectral + skip, NO activation
                spectral = layer._apply_spectral_transform(x)
                skip = layer._apply_skip_connection(x)
                x = spectral + skip

        # Remove domain padding
        if self.domain_padding > 0:
            slices = [slice(None), slice(None)] + [
                slice(None, -self.domain_padding) for _ in range(x.ndim - 2)
            ]
            x = x[tuple(slices)]

        # Two-layer output projection: hidden -> 128 -> GELU -> out
        x = self._apply_pointwise_linear(x, self.output_projection_1)
        x = nnx.gelu(x)
        return self._apply_pointwise_linear(x, self.output_projection_2)

    def _apply_pointwise_linear(self, x: jax.Array, linear_layer: nnx.Linear) -> jax.Array:
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
        total_params += self.input_projection.in_features * self.input_projection.out_features
        if hasattr(self.input_projection, "bias") and self.input_projection.bias is not None:
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

        # Count output projection parameters (two-layer MLP)
        for proj in [self.output_projection_1, self.output_projection_2]:
            total_params += proj.in_features * proj.out_features
            if hasattr(proj, "bias") and proj.bias is not None:
                total_params += proj.out_features

        return total_params
