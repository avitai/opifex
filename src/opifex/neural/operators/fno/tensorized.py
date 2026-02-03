# FILE PLACEMENT: opifex/neural/operators/fno/tensorized.py
#
# SIMPLIFIED Tensorized Fourier Neural Operator Implementation
# Focusing on mathematical stability and basic functionality
#
# This implementation prioritizes working functionality over advanced tensor operations

"""
Simplified Tensorized FNO Architecture

Following critical technical guidelines for strong design and modern architecture
"""

from collections.abc import Sequence
from typing import Literal, Protocol

import jax
import jax.numpy as jnp
from flax import nnx


class TensorDecomposition(Protocol):
    """Protocol for tensor decomposition strategies."""

    def multiply_factorized(self, input_tensor: jax.Array) -> jax.Array:
        """Efficient multiplication using factorized form."""
        ...

    def reconstruct(self) -> jax.Array:
        """Reconstruct full tensor from factors."""
        ...

    def parameter_count(self) -> int:
        """Count parameters in factorized representation."""
        ...


class TuckerDecomposition(nnx.Module):
    """Tucker decomposition for spectral convolution weights.

    Stores weights in full form and applies them via einsum contraction.
    Weight tensor shape: (out_channels, in_channels, *modes)
    """

    def __init__(
        self,
        tensor_shape: Sequence[int],
        rank: float | Sequence[int],
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.tensor_shape = tuple(tensor_shape)
        self.out_channels = tensor_shape[0]
        self.in_channels = tensor_shape[1]
        self.modes = tuple(tensor_shape[2:])
        self.ndim = len(self.modes)  # Spatial dimensions

        # Convert rank ratio to actual ranks (for compression stats)
        if isinstance(rank, float):
            self.ranks = [max(1, int(rank * dim)) for dim in tensor_shape]
        elif isinstance(rank, (list, tuple)):
            self.ranks = list(rank)
        elif isinstance(rank, int):
            self.ranks = [rank] * len(tensor_shape)
        else:
            self.ranks = [1] * len(tensor_shape)

        # Initialize complex weights for spectral convolution
        self.weights = nnx.Param(self._initialize_weights(tensor_shape, rngs.params()))

    def _initialize_weights(self, shape: Sequence[int], key: jax.Array) -> jax.Array:
        """Initialize complex weights with Xavier scaling."""
        fan_in = int(jnp.prod(jnp.asarray(shape[1:])))
        fan_out = shape[0]
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        # Complex weights for spectral domain
        key1, key2 = jax.random.split(key)
        real = jax.random.normal(key1, shape) * std
        imag = jax.random.normal(key2, shape) * std
        return real + 1j * imag

    def multiply_factorized(self, x: jax.Array) -> jax.Array:
        """Contract input with spectral weights.

        Args:
            x: Input tensor of shape (batch, in_channels, *spatial_modes)

        Returns:
            Output tensor of shape (batch, out_channels, *spatial_modes)
        """
        weights = self.weights.value

        # Handle different spatial dimensions with appropriate einsum
        if self.ndim == 1:
            # 1D: (batch, in_ch, m) x (out_ch, in_ch, m) -> (batch, out_ch, m)
            return jnp.einsum("bim,oim->bom", x, weights)
        if self.ndim == 2:
            # 2D: (batch, in_ch, h, w) x (out_ch, in_ch, h, w) -> (batch, out_ch, h, w)
            return jnp.einsum("bihw,oihw->bohw", x, weights)
        if self.ndim == 3:
            # 3D: (batch, in_ch, d, h, w) x (out_ch, in_ch, d, h, w)
            return jnp.einsum("bidhw,oidhw->bodhw", x, weights)
        # Generic N-D using tensordot
        # Contract over in_channels and all spatial modes
        result = jnp.tensordot(x, weights, axes=([1], [1]))
        # Move out_channels to position 1
        return jnp.moveaxis(result, -self.ndim - 1, 1)

    def reconstruct(self) -> jax.Array:
        """Return full weight tensor."""
        return self.weights.value

    def parameter_count(self) -> int:
        """Count parameters in weight tensor."""
        return int(jnp.prod(jnp.asarray(self.tensor_shape)))


class CPDecomposition(nnx.Module):
    """CP decomposition for spectral convolution weights.

    Stores weights in full form and applies them via einsum contraction.
    Weight tensor shape: (out_channels, in_channels, *modes)
    """

    def __init__(
        self,
        tensor_shape: Sequence[int],
        rank: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.tensor_shape = tuple(tensor_shape)
        self.out_channels = tensor_shape[0]
        self.in_channels = tensor_shape[1]
        self.modes = tuple(tensor_shape[2:])
        self.ndim = len(self.modes)
        self.rank = min(rank, *tensor_shape)

        # Initialize complex weights for spectral convolution
        self.weights = nnx.Param(self._initialize_weights(tensor_shape, rngs.params()))

    def _initialize_weights(self, shape: Sequence[int], key: jax.Array) -> jax.Array:
        """Initialize complex weights with Xavier scaling."""
        fan_in = int(jnp.prod(jnp.asarray(shape[1:])))
        fan_out = shape[0]
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        key1, key2 = jax.random.split(key)
        real = jax.random.normal(key1, shape) * std
        imag = jax.random.normal(key2, shape) * std
        return real + 1j * imag

    def multiply_factorized(self, x: jax.Array) -> jax.Array:
        """Contract input with spectral weights."""
        weights = self.weights.value

        if self.ndim == 1:
            return jnp.einsum("bim,oim->bom", x, weights)
        if self.ndim == 2:
            return jnp.einsum("bihw,oihw->bohw", x, weights)
        if self.ndim == 3:
            return jnp.einsum("bidhw,oidhw->bodhw", x, weights)
        result = jnp.tensordot(x, weights, axes=([1], [1]))
        return jnp.moveaxis(result, -self.ndim - 1, 1)

    def parameter_count(self) -> int:
        """Count parameters in weight tensor."""
        return int(jnp.prod(jnp.asarray(self.tensor_shape)))


class TensorTrainDecomposition(nnx.Module):
    """Tensor Train decomposition for spectral convolution weights.

    Stores weights in full form and applies them via einsum contraction.
    Weight tensor shape: (out_channels, in_channels, *modes)
    """

    def __init__(
        self,
        tensor_shape: Sequence[int],
        max_rank: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.tensor_shape = tuple(tensor_shape)
        self.out_channels = tensor_shape[0]
        self.in_channels = tensor_shape[1]
        self.modes = tuple(tensor_shape[2:])
        self.ndim = len(self.modes)
        self.max_rank = max_rank

        # Initialize complex weights for spectral convolution
        self.weights = nnx.Param(self._initialize_weights(tensor_shape, rngs.params()))

    def _initialize_weights(self, shape: Sequence[int], key: jax.Array) -> jax.Array:
        """Initialize complex weights with Xavier scaling."""
        fan_in = int(jnp.prod(jnp.asarray(shape[1:])))
        fan_out = shape[0]
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        key1, key2 = jax.random.split(key)
        real = jax.random.normal(key1, shape) * std
        imag = jax.random.normal(key2, shape) * std
        return real + 1j * imag

    def multiply_factorized(self, x: jax.Array) -> jax.Array:
        """Contract input with spectral weights."""
        weights = self.weights.value

        if self.ndim == 1:
            return jnp.einsum("bim,oim->bom", x, weights)
        if self.ndim == 2:
            return jnp.einsum("bihw,oihw->bohw", x, weights)
        if self.ndim == 3:
            return jnp.einsum("bidhw,oidhw->bodhw", x, weights)
        result = jnp.tensordot(x, weights, axes=([1], [1]))
        return jnp.moveaxis(result, -self.ndim - 1, 1)

    def parameter_count(self) -> int:
        """Count parameters in weight tensor."""
        return int(jnp.prod(jnp.asarray(self.tensor_shape)))


class TensorizedSpectralConvolution(nnx.Module):
    """Simplified tensorized spectral convolution for stability."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        decomposition_type: Literal["tucker", "cp", "tt"] = "tucker",
        rank: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = tuple(modes)
        self.decomposition_type = decomposition_type

        # Create tensor shape for spectral weights
        tensor_shape = (out_channels, in_channels, *modes)

        # Initialize appropriate decomposition with simplified implementations
        self.decomposition: (
            TuckerDecomposition | CPDecomposition | TensorTrainDecomposition
        )
        if decomposition_type == "tucker":
            self.decomposition = TuckerDecomposition(tensor_shape, rank, rngs=rngs)
        elif decomposition_type == "cp":
            cp_rank = max(1, int(rank * min(tensor_shape)))
            self.decomposition = CPDecomposition(tensor_shape, cp_rank, rngs=rngs)
        elif decomposition_type == "tt":
            tt_rank = max(1, int(rank * min(tensor_shape)))
            self.decomposition = TensorTrainDecomposition(
                tensor_shape, tt_rank, rngs=rngs
            )
        else:
            raise ValueError(f"Unknown decomposition type: {decomposition_type}")

    def __call__(self, x_ft: jax.Array) -> jax.Array:
        """Apply spectral convolution to Fourier-transformed input.

        Args:
            x_ft: Complex input of shape (batch, in_channels, *spatial_ft)
                  where spatial_ft are the FFT dimensions.

        Returns:
            Output of shape (batch, out_channels, *spatial_ft)
        """
        spatial_dims = x_ft.shape[2:]

        # Build slices to truncate to mode dimensions
        slices = [slice(None), slice(None)]  # batch, in_channels
        for size, n_modes in zip(spatial_dims, self.modes, strict=False):
            # Take first n_modes frequencies (including DC)
            slices.append(slice(0, min(size, n_modes)))

        # Truncate input to mode dimensions
        x_truncated = x_ft[tuple(slices)]

        # Apply spectral convolution via decomposition
        out_truncated = self.decomposition.multiply_factorized(x_truncated)

        # Zero-pad back to original spatial dimensions
        out_shape = (x_ft.shape[0], self.out_channels, *spatial_dims)
        result = jnp.zeros(out_shape, dtype=out_truncated.dtype)

        # Build output slices
        out_slices = [slice(None), slice(None)]  # batch, out_channels
        for size, n_modes in zip(spatial_dims, self.modes, strict=False):
            out_slices.append(slice(0, min(size, n_modes)))

        # Place truncated output into zero-padded result
        return result.at[tuple(out_slices)].set(out_truncated)

    def get_compression_stats(self) -> dict[str, float]:
        """Get compression statistics."""
        factorized_params = self.decomposition.parameter_count()
        dense_params = (
            self.in_channels * self.out_channels * jnp.prod(jnp.asarray(self.modes))
        )
        compression_ratio = factorized_params / dense_params if dense_params > 0 else 0

        return {
            "factorized_parameters": factorized_params,
            "equivalent_dense_parameters": int(dense_params),
            "compression_ratio": float(compression_ratio),
            "parameter_reduction": float(1 - compression_ratio),
        }


class TensorizedFourierNeuralOperator(nnx.Module):
    """Simplified Tensorized FNO with stable implementations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        modes: Sequence[int] = (16, 16),
        num_layers: int = 4,
        factorization: Literal["tucker", "cp", "tt"] = "tucker",
        rank: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.modes = modes
        self.num_layers = num_layers
        self.factorization = factorization

        # Generate keys for all layers
        keys = jax.random.split(rngs.params(), num_layers + 2)

        # Input projection - takes channels as last dimension
        self.input_proj = nnx.Linear(
            in_channels, hidden_channels, rngs=nnx.Rngs(keys[0])
        )

        # Tensorized spectral convolution layers
        tfno_layers_temp = []
        for i in range(num_layers):
            layer = TensorizedSpectralConvolution(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                modes=modes,
                decomposition_type=factorization,
                rank=rank,
                rngs=nnx.Rngs(keys[i + 1]),
            )
            tfno_layers_temp.append(layer)
            self.tfno_layers = nnx.List(tfno_layers_temp)

        # Output projection - outputs channels as last dimension
        self.output_proj = nnx.Linear(
            hidden_channels, out_channels, rngs=nnx.Rngs(keys[-1])
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through simplified tensorized FNO."""
        batch_size = x.shape[0]
        spatial_dims = x.shape[2:]

        # Reshape input for Linear layer: flatten spatial dimensions
        x_flat = x.reshape(batch_size, self.in_channels, -1)
        x_flat = x_flat.transpose(0, 2, 1)  # (batch, spatial_flat, in_channels)

        # Apply input projection
        x_flat = self.input_proj(x_flat)  # (batch, spatial_flat, hidden_channels)

        # Reshape back to spatial format
        x_flat = x_flat.transpose(0, 2, 1)  # (batch, hidden_channels, spatial_flat)
        x = x_flat.reshape(batch_size, self.hidden_channels, *spatial_dims)

        # Apply simplified tensorized spectral layers
        for layer in self.tfno_layers:
            # Take FFT
            x_ft = jnp.fft.fftn(x, axes=tuple(range(2, x.ndim)))

            # Apply simplified tensorized spectral convolution
            x_ft = layer(x_ft)

            # Take inverse FFT
            x = jnp.fft.ifftn(x_ft, axes=tuple(range(2, x.ndim))).real

        # Reshape output for Linear layer: flatten spatial dimensions
        x_flat = x.reshape(batch_size, self.hidden_channels, -1)
        x_flat = x_flat.transpose(0, 2, 1)  # (batch, spatial_flat, hidden_channels)

        # Apply output projection
        x_flat = self.output_proj(x_flat)  # (batch, spatial_flat, out_channels)

        # Reshape back to spatial format
        x_flat = x_flat.transpose(0, 2, 1)  # (batch, out_channels, spatial_flat)
        return x_flat.reshape(batch_size, self.out_channels, *spatial_dims)


# Factory functions for convenience
def create_tucker_fno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    modes: Sequence[int] = (16, 16),
    rank: float = 0.1,
    num_layers: int = 4,
    *,
    rngs: nnx.Rngs,
) -> TensorizedFourierNeuralOperator:
    """Create Tucker factorized FNO."""
    return TensorizedFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        num_layers=num_layers,
        factorization="tucker",
        rank=rank,
        rngs=rngs,
    )


def create_cp_fno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    modes: Sequence[int] = (16, 16),
    rank: float = 0.1,
    num_layers: int = 4,
    *,
    rngs: nnx.Rngs,
) -> TensorizedFourierNeuralOperator:
    """Create CP factorized FNO."""
    return TensorizedFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        num_layers=num_layers,
        factorization="cp",
        rank=rank,
        rngs=rngs,
    )


def create_tt_fno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    modes: Sequence[int] = (16, 16),
    rank: float = 0.1,
    num_layers: int = 4,
    *,
    rngs: nnx.Rngs,
) -> TensorizedFourierNeuralOperator:
    """Create Tensor Train factorized FNO."""
    return TensorizedFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        num_layers=num_layers,
        factorization="tt",
        rank=rank,
        rngs=rngs,
    )
