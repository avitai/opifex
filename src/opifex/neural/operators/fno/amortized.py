# FILE PLACEMENT: opifex/neural/operators/fno/amortized.py
#
# Complete Rewrite - Amortized Fourier Neural Operator (AM-FNO)
# Fixed all tensor reshape issues and JIT compatibility
#
# This file should REPLACE: opifex/neural/operators/fno/amortized.py

"""
Amortized Fourier Neural Operator (AM-FNO) implementation.

Provides neural kernel networks for high-frequency problems with
parameter-efficient design and arbitrary frequency mode support.
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from beartype import beartype
from flax import nnx


class KernelNetwork(nnx.Module):
    """Neural network to parameterize Fourier kernels."""

    @beartype
    def __init__(
        self,
        freq_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: Callable = nnx.gelu,
        use_frequency_encoding: bool = True,
        max_frequency: float = 10.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize kernel network."""
        super().__init__()
        self.freq_dim = freq_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_frequency_encoding = use_frequency_encoding
        self.max_frequency = max_frequency

        # Calculate input dimension
        if use_frequency_encoding:
            encoding_dim = freq_dim * 20  # 10 sin + 10 cos per frequency
            input_dim = freq_dim + encoding_dim
        else:
            input_dim = freq_dim

        # Build network layers
        layers = []
        current_dim = input_dim

        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nnx.Linear(current_dim, hidden_dim, rngs=rngs),
                    self.activation,
                ]
            )
            current_dim = hidden_dim

        layers.append(nnx.Linear(current_dim, output_dim, rngs=rngs))
        self.network = nnx.Sequential(*layers)

    def _frequency_encoding(self, frequencies: jax.Array) -> jax.Array:
        """Apply positional encoding to frequency coordinates."""
        if not self.use_frequency_encoding:
            return frequencies

        scales = jnp.logspace(0, jnp.log10(self.max_frequency), 10)
        encoded = []
        for scale in scales:
            encoded.append(jnp.sin(scale * frequencies))
            encoded.append(jnp.cos(scale * frequencies))

        return jnp.concatenate([frequencies, *encoded], axis=-1)

    def __call__(self, frequencies: jax.Array) -> jax.Array:
        """Generate kernel weights for given frequencies."""
        freq_encoded = self._frequency_encoding(frequencies)
        return self.network(freq_encoded)


class AmortizedSpectralConvolution(nnx.Module):
    """Amortized spectral convolution with neural kernel parameterization."""

    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        kernel_hidden_dim: int = 128,
        kernel_layers: int = 3,
        max_frequency: float = 10.0,
        use_kernel_regularization: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize amortized spectral convolution."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.n_dim = len(modes)
        self.use_kernel_regularization = use_kernel_regularization

        # Kernel network
        freq_input_dim = self.n_dim
        kernel_output_dim = in_channels * out_channels * 2  # Real and imaginary parts

        self.kernel_network = KernelNetwork(
            freq_input_dim,
            kernel_output_dim,
            kernel_hidden_dim,
            kernel_layers,
            max_frequency=max_frequency,
            rngs=rngs,
        )

        # Learnable scaling
        self.kernel_scale = nnx.Param(
            jnp.ones((1,)) * (2 / (in_channels + out_channels)) ** 0.5
        )

    def _generate_frequency_coordinates(
        self, modes_shape: tuple[int, ...]
    ) -> jax.Array:
        """Generate frequency coordinates for given modes shape."""
        coords = []
        for mode_size in modes_shape:
            coord = jnp.linspace(0, 1, mode_size)
            coords.append(coord)

        # Create meshgrid and flatten
        mesh_coords = jnp.meshgrid(*coords, indexing="ij")
        stacked_coords = jnp.stack(mesh_coords, axis=-1)
        return stacked_coords.reshape(-1, self.n_dim)

    def _generate_kernel_weights(self, actual_modes: tuple[int, ...]) -> jax.Array:
        """Generate kernel weights for given mode configuration."""
        # Generate frequency coordinates
        freq_coords = self._generate_frequency_coordinates(actual_modes)

        # Apply kernel network
        kernel_weights_flat = self.kernel_network(freq_coords)

        # Reshape to (num_points, in_channels, out_channels, 2)
        num_points = freq_coords.shape[0]
        kernel_weights = kernel_weights_flat.reshape(
            num_points, self.in_channels, self.out_channels, 2
        )

        # Convert to complex
        kernel_complex = kernel_weights[..., 0] + 1j * kernel_weights[..., 1]

        # Reshape to spatial format: (in_channels, out_channels, *actual_modes)
        spatial_kernel = kernel_complex.reshape(
            *actual_modes, self.in_channels, self.out_channels
        )

        # Permute to get channels first
        perm = [self.n_dim, self.n_dim + 1, *range(self.n_dim)]
        return jnp.transpose(spatial_kernel, perm)

    def __call__(
        self, x_ft: jax.Array, training: bool = True
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass through amortized spectral convolution."""
        # Extract spatial dimensions from input
        input_modes = x_ft.shape[2:]  # Skip batch and channel dimensions
        actual_modes = tuple(
            min(m, im) for m, im in zip(self.modes, input_modes, strict=False)
        )

        # Generate kernel weights
        kernel_weights = self._generate_kernel_weights(actual_modes)

        # Extract relevant modes from input
        x_modes = x_ft
        for i, mode in enumerate(actual_modes):
            axis = i + 2  # Skip batch and channel dimensions
            if axis < len(x_modes.shape):
                x_modes = (
                    x_modes[..., :mode]
                    if axis == len(x_modes.shape) - 1
                    else jnp.take(x_modes, jnp.arange(mode), axis=axis)
                )

        # Apply spectral convolution
        batch_size = x_modes.shape[0]
        in_channels_x = x_modes.shape[1]
        in_channels_k = kernel_weights.shape[0]
        out_channels = kernel_weights.shape[1]

        if in_channels_x != in_channels_k:
            raise ValueError(
                f"Input channels mismatch: x_modes has {in_channels_x}, "
                f"kernel_weights expects {in_channels_k}"
            )

        # Reshape for matrix multiplication
        spatial_shape = x_modes.shape[2:]
        x_reshaped = x_modes.reshape(batch_size, in_channels_x, -1)
        kernel_reshaped = kernel_weights.reshape(in_channels_k, out_channels, -1)

        # Apply convolution
        output_reshaped = jnp.einsum("bis,ios->bos", x_reshaped, kernel_reshaped)
        output = output_reshaped.reshape(batch_size, out_channels, *spatial_shape)

        # Apply scaling
        output = output * self.kernel_scale.value

        # Compute regularization loss
        reg_loss = jnp.array(0.0)
        if self.use_kernel_regularization:
            reg_loss = jnp.sum(jnp.abs(kernel_weights) ** 2) * 1e-4

        return output, reg_loss


class AmortizedFourierNeuralOperator(nnx.Module):
    """Amortized Fourier Neural Operator with neural kernel parameterization."""

    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        modes: Sequence[int] = (16, 16),
        num_layers: int = 4,
        kernel_hidden_dim: int = 128,
        kernel_layers: int = 3,
        max_frequency: float = 10.0,
        activation: Callable = nnx.gelu,
        use_layer_norm: bool = False,
        use_kernel_regularization: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Amortized FNO."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.modes = modes
        self.num_layers = num_layers
        self.activation = activation
        self.use_layer_norm = use_layer_norm

        # Input projection
        self.lifting = nnx.Linear(in_channels, hidden_channels, rngs=rngs)

        # Amortized spectral convolution layers
        for i in range(num_layers):
            conv = AmortizedSpectralConvolution(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                modes=modes,
                kernel_hidden_dim=kernel_hidden_dim,
                kernel_layers=kernel_layers,
                max_frequency=max_frequency,
                use_kernel_regularization=use_kernel_regularization,
                rngs=rngs,
            )
            setattr(self, f"conv_{i}", conv)

            # Skip connection layers
            skip = nnx.Linear(hidden_channels, hidden_channels, rngs=rngs)
            setattr(self, f"skip_{i}", skip)

            # Optional layer normalization
            if use_layer_norm:
                norm = nnx.LayerNorm(hidden_channels, rngs=rngs)
                setattr(self, f"norm_{i}", norm)

        # Output projection
        self.projection = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def _resize_tensor(self, x: jax.Array, target_shape: tuple[int, ...]) -> jax.Array:
        """Resize tensor to target spatial shape."""
        current_shape = x.shape[2:]  # Skip batch and channel dimensions
        if current_shape == target_shape:
            return x

        # Simple resize using indexing
        indices = []
        for current_size, target_size in zip(current_shape, target_shape, strict=False):
            if target_size <= current_size:
                # Downsample
                step = current_size // target_size
                idx = jnp.arange(0, current_size, step)[:target_size]
            else:
                # Upsample
                idx = jnp.asarray(
                    jnp.round(jnp.linspace(0, current_size - 1, target_size)),
                    dtype=jnp.int_,
                )
            indices.append(idx)

        # Apply resizing
        result = x
        for i, idx in enumerate(indices):
            axis = i + 2  # Skip batch and channel dimensions
            result = jnp.take(result, idx, axis=axis)

        return result

    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Forward pass through Amortized FNO."""
        # Input projection - handle channels properly
        x_input = jnp.moveaxis(x, 1, -1)  # Move channels to last
        x = self.activation(self.lifting(x_input))
        x = jnp.moveaxis(x, -1, 1)  # Move channels back to second position

        # Apply Fourier layers
        for i in range(self.num_layers):
            conv = getattr(self, f"conv_{i}")
            skip = getattr(self, f"skip_{i}")

            # Get the target spatial size from the first convolution layer
            # This will be the size determined by the modes
            conv_modes = conv.modes
            target_size = tuple(conv_modes)

            # FFT to frequency domain
            x_ft = jnp.fft.rfftn(x, axes=tuple(range(2, x.ndim)))

            # Apply amortized spectral convolution
            x_conv, _reg_loss = conv(x_ft, training)

            # IFFT back to spatial domain with target size
            x_conv = jnp.fft.irfftn(
                x_conv, s=target_size, axes=tuple(range(2, x_conv.ndim))
            )

            # Skip connection - resize input to match target size
            x_resized = self._resize_tensor(x, target_size)
            x_skip_input = jnp.moveaxis(x_resized, 1, -1)
            x_skip_processed = skip(x_skip_input)
            x_skip_processed = jnp.moveaxis(x_skip_processed, -1, 1)

            # Combine
            x_combined = x_conv + x_skip_processed

            # Optional layer normalization
            if self.use_layer_norm:
                norm = getattr(self, f"norm_{i}")
                x_norm_input = jnp.moveaxis(x_combined, 1, -1)
                x_normalized = norm(x_norm_input)
                x_combined = jnp.moveaxis(x_normalized, -1, 1)

            # Activation and update x for next iteration
            x = self.activation(x_combined)

        # Final projection
        x_proj_input = jnp.moveaxis(x, 1, -1)
        output = self.projection(x_proj_input)

        return jnp.moveaxis(output, -1, 1)

    def get_regularization_loss(self, x: jax.Array) -> jax.Array:
        """Compute regularization loss on demand."""
        total_reg_loss = jnp.array(0.0)

        # Compute FFT
        x_ft = jnp.fft.rfftn(x, axes=tuple(range(2, x.ndim)))

        # Compute regularization for each layer
        for i in range(self.num_layers):
            conv = getattr(self, f"conv_{i}")
            input_modes = x_ft.shape[2:]
            actual_modes = tuple(
                min(m, im) for m, im in zip(conv.modes, input_modes, strict=False)
            )
            kernel_weights = conv._generate_kernel_weights(actual_modes)
            reg_loss = jnp.sum(jnp.abs(kernel_weights) ** 2) * 1e-4
            total_reg_loss += reg_loss

        return total_reg_loss

    def get_kernel_analysis(
        self, freq_range: tuple[float, float], num_points: int = 100
    ) -> dict[str, jax.Array]:
        """Analyze learned kernel functions."""
        analysis = {}

        # Create frequency grid
        freqs = jnp.linspace(freq_range[0], freq_range[1], num_points)
        if len(self.modes) == 1:
            freq_grid = freqs[:, None]
        else:
            freq_grid = jnp.stack([freqs] * len(self.modes), axis=1)

        # Analyze each layer's kernel network
        for i in range(self.num_layers):
            conv = getattr(self, f"conv_{i}")
            kernel_weights = conv.kernel_network(freq_grid)

            # Split into real and imaginary parts
            half_dim = conv.in_channels * conv.out_channels
            real_part = kernel_weights[..., :half_dim]
            imag_part = kernel_weights[..., half_dim:]

            analysis[f"layer_{i}_real"] = real_part
            analysis[f"layer_{i}_imag"] = imag_part
            analysis[f"layer_{i}_magnitude"] = jnp.sqrt(real_part**2 + imag_part**2)

        analysis["frequencies"] = freqs
        return analysis


# Utility functions for different AM-FNO configurations
def create_high_frequency_amfno(
    in_channels: int,
    out_channels: int,
    modes: Sequence[int] = (128, 128),
    **kwargs,
) -> AmortizedFourierNeuralOperator:
    """Create AM-FNO optimized for high-frequency problems."""
    return AmortizedFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=64,
        modes=modes,
        num_layers=4,
        kernel_hidden_dim=256,
        kernel_layers=4,
        max_frequency=20.0,
        **kwargs,
    )


def create_wave_amfno(
    in_channels: int = 2,
    out_channels: int = 2,
    modes: Sequence[int] = (64, 64),
    **kwargs,
) -> AmortizedFourierNeuralOperator:
    """Create AM-FNO for wave propagation problems."""
    return AmortizedFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=96,
        modes=modes,
        num_layers=5,
        kernel_hidden_dim=128,
        kernel_layers=3,
        max_frequency=15.0,
        use_layer_norm=True,
        **kwargs,
    )


def create_shock_amfno(
    in_channels: int = 3,
    out_channels: int = 3,
    modes: Sequence[int] = (96, 96),
    **kwargs,
) -> AmortizedFourierNeuralOperator:
    """Create AM-FNO for problems with shocks/discontinuities."""
    return AmortizedFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,
        modes=modes,
        num_layers=6,
        kernel_hidden_dim=256,
        kernel_layers=4,
        max_frequency=25.0,
        use_layer_norm=True,
        use_kernel_regularization=True,
        **kwargs,
    )
