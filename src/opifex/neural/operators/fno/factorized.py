"""Factorized Fourier Neural Operator implementation.

This module contains the FactorizedFourierLayer class that implements tensor
factorization for parameter reduction in Fourier Neural Operators.

FIXES APPLIED:
- Added 3D support for complete dimensionality coverage
- Simplified channel handling using moveaxis
- Improved edge case handling and error management
- Enhanced parameter reduction efficiency
- Fixed residual connection architecture
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx


class FactorizedFourierLayer(nnx.Module):
    """Fourier layer with tensor factorization for parameter reduction.

    Implements Tucker or CP factorization of the spectral convolution weights
    to achieve significant parameter reduction (up to 95%) while maintaining
    performance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        factorization_type: str,
        factorization_rank: int,
        *,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ):
        """Initialize factorized Fourier layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes
            factorization_type: Type of factorization ("tucker" or "cp")
            factorization_rank: Rank for factorization
            activation: Activation function
            rngs: Random number generators
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.factorization_type = factorization_type
        self.factorization_rank = factorization_rank
        self.activation = activation

        if factorization_type == "tucker":
            self._init_tucker_factorization(rngs)
        elif factorization_type == "cp":
            self._init_cp_factorization(rngs)
        else:
            raise ValueError("factorization_type must be 'tucker' or 'cp'")

        # Linear transformation for residual connection
        self.linear = nnx.Linear(
            in_features=in_channels, out_features=out_channels, rngs=rngs
        )

    def _init_tucker_factorization(self, rngs: nnx.Rngs) -> None:
        """Initialize Tucker factorization parameters.

        Tucker factorization: A = C x1 U1 x2 U2 x3 U3
        Core tensor: (rank, rank, rank)
        Factor matrices: (in_channels, rank), (out_channels, rank), (modes, rank)
        """
        rank = self.factorization_rank

        # Improved Xavier initialization for factor matrices
        def init_factor_matrix(shape, key):
            fan_in, fan_out = shape
            std = jnp.sqrt(2.0 / (fan_in + fan_out)) / jnp.sqrt(2)  # Complex adjustment
            real = jax.random.normal(key, shape) * std
            imag = jax.random.normal(key, shape) * std
            return jnp.complex_(real + 1j * imag)

        # Specialized initialization for core tensor
        def init_core_tensor(shape, key):
            # For 3D core tensor, use total number of elements for initialization
            total_elements = jnp.prod(jnp.array(shape))
            std = jnp.sqrt(2.0 / total_elements) / jnp.sqrt(2)
            real = jax.random.normal(key, shape) * std
            imag = jax.random.normal(key, shape) * std
            return jnp.complex_(real + 1j * imag)

        # Split keys for all components
        key_core, key_u1, key_u2, key_u3 = jax.random.split(rngs.params(), 4)

        # Core tensor
        self.tucker_core = nnx.Param(init_core_tensor((rank, rank, rank), key_core))

        # Factor matrices
        self.tucker_u1 = nnx.Param(init_factor_matrix((self.in_channels, rank), key_u1))
        self.tucker_u2 = nnx.Param(
            init_factor_matrix((self.out_channels, rank), key_u2)
        )
        self.tucker_u3 = nnx.Param(init_factor_matrix((self.modes, rank), key_u3))

    def _init_cp_factorization(self, rngs: nnx.Rngs) -> None:
        """Initialize CP factorization parameters.

        CP factorization: A = Σr ar ° br ° cr
        Factor matrices: (in_channels, rank), (out_channels, rank), (modes, rank)
        """
        rank = self.factorization_rank

        # Improved initialization with proper scaling
        def init_cp_factor(shape, key):
            fan_in = shape[0]
            fan_out = rank
            std = jnp.sqrt(2.0 / (fan_in + fan_out + self.modes)) / jnp.sqrt(2)
            real = jax.random.normal(key, shape) * std
            imag = jax.random.normal(key, shape) * std
            return jnp.complex_(real + 1j * imag)

        # Split keys for all factors
        key_a, key_b, key_c = jax.random.split(rngs.params(), 3)

        # Factor matrices
        self.cp_factor_a = nnx.Param(init_cp_factor((self.in_channels, rank), key_a))
        self.cp_factor_b = nnx.Param(init_cp_factor((self.out_channels, rank), key_b))
        self.cp_factor_c = nnx.Param(init_cp_factor((self.modes, rank), key_c))

    def _reconstruct_tucker_weights(self) -> jax.Array:
        """Reconstruct full tensor from Tucker factorization."""
        return jnp.einsum(
            "rst,ir,js,kt->ijk",
            self.tucker_core.value,
            self.tucker_u1.value,
            self.tucker_u2.value,
            self.tucker_u3.value,
        )

    def _reconstruct_cp_weights(self) -> jax.Array:
        """Reconstruct full tensor from CP factorization."""
        return jnp.einsum(
            "ir,jr,kr->ijk",
            self.cp_factor_a.value,
            self.cp_factor_b.value,
            self.cp_factor_c.value,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply factorized Fourier layer.

        Args:
            x: Input tensor (batch, channels, *spatial_dims)

        Returns:
            Output tensor (batch, out_channels, *spatial_dims)
        """
        # Spectral pathway: FFT -> FactorizedSpectralConv -> IFFT
        spectral_output = self._apply_factorized_spectral_transform(x)

        # Skip connection pathway
        skip_output = self._apply_skip_connection(x)

        # Combine and activate (proper FNO residual connection)
        combined = spectral_output + skip_output
        return self.activation(combined)

    def _apply_factorized_spectral_transform(self, x: jax.Array) -> jax.Array:
        """Apply factorized spectral transformation."""
        spatial_dims = x.shape[2:]

        if len(spatial_dims) == 1:
            return self._factorized_spectral_1d(x)
        if len(spatial_dims) == 2:
            return self._factorized_spectral_2d(x)
        if len(spatial_dims) == 3:
            return self._factorized_spectral_3d(x)  # NEW: 3D support
        raise ValueError(
            f"Unsupported number of spatial dimensions: {len(spatial_dims)}"
        )

    def _factorized_spectral_1d(self, x: jax.Array) -> jax.Array:
        """1D factorized spectral transform."""
        batch_size = x.shape[0]
        grid_size = x.shape[-1]

        # Reconstruct weights from factorization
        spectral_weights = self._get_spectral_weights()

        # FFT to spectral domain
        x_ft = jnp.fft.rfft(x, axis=-1)

        # Apply factorized spectral convolution with proper mode handling
        effective_modes = min(self.modes, x_ft.shape[-1])
        if effective_modes == 0:
            return jnp.zeros((batch_size, self.out_channels, grid_size), dtype=x.dtype)

        x_ft_truncated = x_ft[:, :, :effective_modes]
        weights_truncated = spectral_weights[:, :, :effective_modes]

        out_ft = jnp.einsum("bik,iok->bok", x_ft_truncated, weights_truncated)

        # Pad back to original frequency resolution
        if out_ft.shape[-1] < grid_size // 2 + 1:
            out_ft_padded = jnp.zeros(
                (batch_size, self.out_channels, grid_size // 2 + 1), dtype=out_ft.dtype
            )
            out_ft_padded = out_ft_padded.at[:, :, : out_ft.shape[-1]].set(out_ft)
            out_ft = out_ft_padded

        # IFFT back to spatial domain
        return jnp.fft.irfft(out_ft, n=grid_size, axis=-1)

    def _factorized_spectral_2d(self, x: jax.Array) -> jax.Array:
        """2D factorized spectral transform - IMPROVED VERSION."""
        batch_size = x.shape[0]
        h, w = x.shape[-2:]

        # Reconstruct weights from factorization
        spectral_weights = self._get_spectral_weights()

        # 2D FFT to spectral domain
        x_ft = jnp.fft.rfft2(x, axes=(-2, -1))

        # Handle modes properly for 2D case
        ft_h, ft_w = x_ft.shape[-2:]
        modes_h = min(self.modes, ft_h)
        modes_w = min(self.modes, ft_w)

        if modes_h == 0 or modes_w == 0:
            return jnp.zeros((batch_size, self.out_channels, h, w), dtype=x.dtype)

        # Keep only low-frequency modes
        x_ft_truncated = x_ft[:, :, :modes_h, :modes_w]

        # Reshape for efficient computation
        x_ft_flat = x_ft_truncated.reshape(batch_size, self.in_channels, -1)

        # Apply spectral convolution (use only available modes)
        available_modes = x_ft_flat.shape[-1]
        if available_modes <= spectral_weights.shape[-1]:
            weights_used = spectral_weights[:, :, :available_modes]
        else:
            # Pad weights if needed (rare edge case)
            weights_used = jnp.pad(
                spectral_weights,
                ((0, 0), (0, 0), (0, available_modes - spectral_weights.shape[-1])),
                mode="constant",
            )

        out_ft_flat = jnp.einsum("bik,iok->bok", x_ft_flat, weights_used)

        # Reshape back to 2D
        out_ft_truncated = out_ft_flat.reshape(
            batch_size, self.out_channels, modes_h, modes_w
        )

        # Pad back to original frequency resolution
        out_ft_padded = jnp.zeros(
            (batch_size, self.out_channels, ft_h, ft_w), dtype=out_ft_truncated.dtype
        )
        out_ft_padded = out_ft_padded.at[:, :, :modes_h, :modes_w].set(out_ft_truncated)

        # IFFT back to spatial domain
        return jnp.fft.irfft2(out_ft_padded, s=(h, w), axes=(-2, -1))

    def _factorized_spectral_3d(self, x: jax.Array) -> jax.Array:
        """3D factorized spectral transform - NEW FEATURE."""
        batch_size = x.shape[0]
        d, h, w = x.shape[-3:]

        # Reconstruct weights from factorization
        spectral_weights = self._get_spectral_weights()

        # 3D FFT to spectral domain
        x_ft = jnp.fft.rfftn(x, axes=(-3, -2, -1))

        # Handle modes properly for 3D case
        ft_d, ft_h, ft_w = x_ft.shape[-3:]
        modes_d = min(self.modes, ft_d)
        modes_h = min(self.modes, ft_h)
        modes_w = min(self.modes, ft_w)

        if modes_d == 0 or modes_h == 0 or modes_w == 0:
            return jnp.zeros((batch_size, self.out_channels, d, h, w), dtype=x.dtype)

        # Keep only low-frequency modes
        x_ft_truncated = x_ft[:, :, :modes_d, :modes_h, :modes_w]

        # Reshape for efficient computation
        x_ft_flat = x_ft_truncated.reshape(batch_size, self.in_channels, -1)

        # Apply spectral convolution
        available_modes = x_ft_flat.shape[-1]
        if available_modes <= spectral_weights.shape[-1]:
            weights_used = spectral_weights[:, :, :available_modes]
        else:
            # Pad weights if needed
            weights_used = jnp.pad(
                spectral_weights,
                ((0, 0), (0, 0), (0, available_modes - spectral_weights.shape[-1])),
                mode="constant",
            )

        out_ft_flat = jnp.einsum("bik,iok->bok", x_ft_flat, weights_used)

        # Reshape back to 3D
        out_ft_truncated = out_ft_flat.reshape(
            batch_size, self.out_channels, modes_d, modes_h, modes_w
        )

        # Pad back to original frequency resolution
        out_ft_padded = jnp.zeros(
            (batch_size, self.out_channels, ft_d, ft_h, ft_w),
            dtype=out_ft_truncated.dtype,
        )
        out_ft_padded = out_ft_padded.at[:, :, :modes_d, :modes_h, :modes_w].set(
            out_ft_truncated
        )

        # IFFT back to spatial domain
        return jnp.fft.irfftn(out_ft_padded, s=(d, h, w), axes=(-3, -2, -1))

    def _apply_skip_connection(self, x: jax.Array) -> jax.Array:
        """Apply skip connection with simplified channel handling."""
        # Move channels to last dimension for linear layer
        x_permuted = jnp.moveaxis(x, 1, -1)  # (batch, *spatial, channels)

        # Apply linear transformation
        skip_out = self.linear(x_permuted)  # (batch, *spatial, out_channels)

        # Move channels back to second dimension
        return jnp.moveaxis(skip_out, -1, 1)  # (batch, out_channels, *spatial)

    def _get_spectral_weights(self) -> jax.Array:
        """Get spectral weights from factorization."""
        if self.factorization_type == "tucker":
            return self._reconstruct_tucker_weights()
        # cp
        return self._reconstruct_cp_weights()

    def get_parameter_count(self) -> dict[str, int | float]:
        """Get parameter count breakdown for analysis."""
        if self.factorization_type == "tucker":
            core_params = self.factorization_rank**3
            u1_params = self.in_channels * self.factorization_rank
            u2_params = self.out_channels * self.factorization_rank
            u3_params = self.modes * self.factorization_rank
            factorized_total = core_params + u1_params + u2_params + u3_params
        else:  # cp
            factorized_total = (
                self.in_channels * self.factorization_rank
                + self.out_channels * self.factorization_rank
                + self.modes * self.factorization_rank
            )

        # Compare to full tensor
        full_tensor_params = self.in_channels * self.out_channels * self.modes

        # Add linear layer parameters
        linear_params = self.in_channels * self.out_channels

        return {
            "factorized_spectral": factorized_total * 2,  # Complex numbers
            "full_tensor_equivalent": full_tensor_params * 2,  # Complex numbers
            "linear_layer": linear_params,
            "total": factorized_total * 2 + linear_params,
            "compression_ratio": factorized_total / full_tensor_params,
            "parameter_reduction": 1.0 - (factorized_total / full_tensor_params),
        }
