"""Wavelet Neural Operator implementation.

This module contains the WaveletNeuralOperator class that uses wavelet transforms
to capture multi-scale features for operator learning.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

# Import PhysicsAwareAttention from physics module
from opifex.neural.operators.physics.attention import PhysicsAwareAttention


class WaveletNeuralOperator(nnx.Module):
    """Wavelet Neural Operator for multi-scale wavelet-based learning.

    This operator uses wavelet transforms to capture multi-scale features
    in the input functions, enabling efficient learning of operators with
    multi-scale characteristics like turbulence and material heterogeneity.

    Features:
    - Discrete Wavelet Transform (DWT) for multi-scale decomposition
    - Learnable wavelet coefficients processing
    - Multi-resolution reconstruction
    - Adaptive wavelet basis selection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_levels: int,
        *,
        wavelet_type: str = "db4",
        mode: str = "symmetric",
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        use_learnable_wavelets: bool = False,
        rngs: nnx.Rngs,
    ):
        """Initialize Wavelet Neural Operator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Hidden channel dimension
            num_levels: Number of wavelet decomposition levels
            wavelet_type: Type of wavelet (e.g., 'db4', 'haar')
            mode: Boundary condition mode
            activation: Activation function
            use_learnable_wavelets: Whether to use learnable wavelet bases
            rngs: Random number generators
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.wavelet_type = wavelet_type
        self.mode = mode
        self.activation = activation
        self.use_learnable_wavelets = use_learnable_wavelets

        # Input projection
        self.input_proj = nnx.Linear(
            in_features=in_channels,
            out_features=hidden_channels,
            rngs=rngs,
        )

        # Wavelet coefficient processors for each level
        wavelet_processors_temp = []
        for _level in range(num_levels):
            # Each level processes approximation and detail coefficients
            # Input: 2 coefficients (approx, detail), Output: hidden_channels features
            processor = nnx.Sequential(
                nnx.Linear(
                    in_features=2,  # approx and detail coefficients
                    out_features=hidden_channels * 2,
                    rngs=rngs,
                ),
                activation,
                nnx.Linear(
                    in_features=hidden_channels * 2,
                    out_features=hidden_channels,
                    rngs=rngs,
                ),
            )
            wavelet_processors_temp.append(processor)
            self.wavelet_processors = nnx.List(wavelet_processors_temp)

        # Cross-level interaction layers
        self.cross_level_attention = PhysicsAwareAttention(
            embed_dim=hidden_channels,
            num_heads=8,
            dropout_rate=0.0,
            rngs=rngs,
        )

        # Output projection
        self.output_proj = nnx.Linear(
            in_features=hidden_channels,
            out_features=out_channels,
            rngs=rngs,
        )

        # Learnable wavelet filters (if enabled)
        if use_learnable_wavelets:
            filter_length = 8  # Typical length for db4-like wavelets
            self.low_pass_filter = nnx.Param(
                jax.random.normal(rngs.params(), (filter_length,)) * 0.1
            )
            self.high_pass_filter = nnx.Param(
                jax.random.normal(rngs.params(), (filter_length,)) * 0.1
            )
        else:
            # Use fixed Daubechies-4 wavelet coefficients
            db4_low = jnp.array(
                [
                    0.23037781,
                    0.71484657,
                    0.63088076,
                    -0.02798376,
                    -0.18703481,
                    0.03084138,
                    0.03288301,
                    -0.01059740,
                ],
            )
            db4_high = jnp.array(
                [
                    -0.01059740,
                    -0.03288301,
                    0.03084138,
                    0.18703481,
                    -0.02798376,
                    -0.63088076,
                    0.71484657,
                    -0.23037781,
                ],
            )

            self.low_pass_filter = nnx.Param(db4_low)
            self.high_pass_filter = nnx.Param(db4_high)

    def _dwt_1d(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """1D Discrete Wavelet Transform."""
        # Convolve with low-pass and high-pass filters
        low_pass = jnp.convolve(x, self.low_pass_filter.value, mode="same")
        high_pass = jnp.convolve(x, self.high_pass_filter.value, mode="same")

        # Downsample by factor of 2
        approx = low_pass[::2]
        detail = high_pass[::2]

        return approx, detail

    def _idwt_1d(self, approx: jax.Array, detail: jax.Array) -> jax.Array:
        """1D Inverse Discrete Wavelet Transform."""
        # Upsample by inserting zeros
        upsampled_approx = jnp.zeros(len(approx) * 2, dtype=approx.dtype)
        upsampled_approx = upsampled_approx.at[::2].set(approx)

        upsampled_detail = jnp.zeros(len(detail) * 2, dtype=detail.dtype)
        upsampled_detail = upsampled_detail.at[::2].set(detail)

        # Convolve with reconstruction filters
        recon_low = jnp.convolve(
            upsampled_approx, self.low_pass_filter.value[::-1], mode="same"
        )
        recon_high = jnp.convolve(
            upsampled_detail, self.high_pass_filter.value[::-1], mode="same"
        )

        return recon_low + recon_high

    def _multi_level_dwt(self, x: jax.Array) -> list[tuple[jax.Array, jax.Array]]:
        """Multi-level wavelet decomposition."""
        coefficients = []
        current = x

        for _level in range(self.num_levels):
            approx, detail = self._dwt_1d(current)
            coefficients.append((approx, detail))
            current = approx  # Use approximation for next level

        return coefficients

    def _multi_level_idwt(
        self, coefficients: list[tuple[jax.Array, jax.Array]]
    ) -> jax.Array:
        """Multi-level wavelet reconstruction."""
        # Start with the coarsest approximation
        current = coefficients[-1][0]  # Coarsest approximation

        # Reconstruct from coarsest to finest
        for level in range(self.num_levels - 1, -1, -1):
            _, detail = coefficients[level]
            if level == self.num_levels - 1:
                # For the coarsest level, use the approximation directly
                current = self._idwt_1d(current, detail)
            else:
                # For other levels, use the reconstructed signal as approximation
                current = self._idwt_1d(current, detail)

        return current

    def __call__(self, x: jax.Array, *, training: bool = False) -> jax.Array:
        """Apply Wavelet Neural Operator.

        Args:
            x: Input tensor (batch, channels, spatial_dim)
            training: Whether in training mode

        Returns:
            Output tensor (batch, out_channels, spatial_dim)
        """
        batch_size = x.shape[0]

        # Input projection
        x_projected = self.input_proj(x.transpose(0, 2, 1)).transpose(0, 2, 1)

        # Process each channel separately
        processed_channels = []

        for ch in range(self.hidden_channels):
            channel_data = x_projected[:, ch, :]  # (batch, spatial)

            # Multi-level wavelet decomposition for each sample in batch
            batch_coefficients = []
            for b in range(batch_size):
                coeffs = self._multi_level_dwt(channel_data[b])
                batch_coefficients.append(coeffs)

            # Process wavelet coefficients at each level
            processed_coefficients = []
            for level in range(self.num_levels):
                # Collect coefficients for this level across batch
                level_approx = jnp.stack(
                    [batch_coefficients[b][level][0] for b in range(batch_size)]
                )
                level_detail = jnp.stack(
                    [batch_coefficients[b][level][1] for b in range(batch_size)]
                )

                # Process coefficients through neural network
                # Combine approximation and detail coefficients
                combined_coeffs = jnp.concatenate(
                    [level_approx[:, None, :], level_detail[:, None, :]], axis=1
                )  # (batch, 2, coeff_length)

                # Reshape for processing: (batch * coeff_length, 2)
                batch_size_level, _, coeff_length = combined_coeffs.shape
                reshaped_coeffs = combined_coeffs.transpose(0, 2, 1).reshape(-1, 2)

                # Apply wavelet processor
                processed_coeffs = self.wavelet_processors[level](reshaped_coeffs)

                # Reshape back: (batch, coeff_length, hidden_channels) ->
                # (batch, hidden_channels, coeff_length)
                processed_coeffs = processed_coeffs.reshape(
                    batch_size_level, coeff_length, self.hidden_channels
                ).transpose(0, 2, 1)

                # Since we now have (batch, hidden_channels, coeff_length),
                # we need to use the original coefficients structure but with
                # processed features
                # For simplicity, use the processed features as both approx and detail
                processed_approx = processed_coeffs.mean(
                    axis=1
                )  # Average across channels
                processed_detail = processed_coeffs.mean(
                    axis=1
                )  # Average across channels

                processed_coefficients.append((processed_approx, processed_detail))

            # Reconstruct signal from processed coefficients
            reconstructed_channel = []
            for b in range(batch_size):
                batch_processed_coeffs = [
                    (
                        processed_coefficients[level][0][b],
                        processed_coefficients[level][1][b],
                    )
                    for level in range(self.num_levels)
                ]
                reconstructed = self._multi_level_idwt(batch_processed_coeffs)
                reconstructed_channel.append(reconstructed)

            processed_channels.append(jnp.stack(reconstructed_channel))

        # Combine processed channels
        processed_output = jnp.stack(processed_channels, axis=1)

        # Apply cross-level attention for global feature interaction
        # Reshape for attention: (batch, spatial, channels)
        attended_output = self.cross_level_attention(
            processed_output.transpose(0, 2, 1), training=training
        ).transpose(0, 2, 1)

        # Output projection
        return self.output_proj(attended_output.transpose(0, 2, 1)).transpose(0, 2, 1)
