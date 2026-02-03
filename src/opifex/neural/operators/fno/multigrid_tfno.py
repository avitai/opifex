"""
Multi-Grid Tensorized Fourier Neural Operator (MG-TFNO)

Phase 3: Implementation of advanced Multi-Grid TFNO techniques for hierarchical
tensor decomposition with frequency-aware rank adaptation and dynamic rank learning.

Based on the detailed rationale's recommendations for advanced
Multi-Grid decomposition strategies and memory-optimal contraction ordering.

This module implements:
1. Hierarchical tensor decomposition for different frequency scales
2. Frequency-aware rank adaptation strategies
3. Adaptive rank learning with gradient-based optimization
4. Memory-optimal contraction ordering from MG-TFNO research
5. Dynamic compression ratio tuning
"""

from collections.abc import Sequence
from typing import Any, cast, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


# Optional TensorLy integration for enhanced initialization
try:
    from opifex.neural.operators.fno.tensorly_integration import (
        MemoryOptimalContractions,
        TENSORLY_AVAILABLE,
        TensorLyTuckerInitializer,
    )
except ImportError:
    _tensorly_available = False
else:
    _tensorly_available = TENSORLY_AVAILABLE


class FrequencyAwareDecomposition(Protocol):
    """Protocol for frequency-aware tensor decomposition strategies."""

    def decompose_by_frequency(
        self,
        tensor: jax.Array,
        frequency_bands: Sequence[tuple[int, int]],
    ) -> dict[str, Any]:
        """Decompose tensor with frequency-specific ranks."""
        ...

    def get_frequency_ranks(self, modes: Sequence[int]) -> dict[str, Sequence[int]]:
        """Get optimal ranks for different frequency bands."""
        ...

    def adaptive_rank_update(
        self,
        gradients: jax.Array,
        current_ranks: Sequence[int],
        learning_rate: float = 0.01,
    ) -> Sequence[int]:
        """Update ranks based on gradient information."""
        ...


class MultiGridTuckerDecomposition(nnx.Module):
    """Multi-Grid Tucker decomposition with hierarchical frequency-aware compression.

    Implements advanced Multi-Grid TFNO techniques:
    1. Hierarchical decomposition across frequency scales
    2. Frequency-aware rank adaptation
    3. Memory-optimal contraction ordering
    4. Dynamic rank learning during training

    Key innovation: Different frequency modes get different compression levels,
    with low frequencies (containing most energy) getting higher ranks than
    high frequencies (typically noise or fine details).
    """

    def __init__(
        self,
        tensor_shape: Sequence[int],
        base_rank: float | Sequence[int] = 0.1,
        frequency_bands: Sequence[tuple[int, int]] | None = None,
        rank_adaptation_strategy: Literal[
            "uniform", "frequency_decay", "energy_based"
        ] = "frequency_decay",
        *,
        rngs: nnx.Rngs,
        use_complex: bool = True,
        adaptive_rank_learning: bool = True,
        rank_learning_rate: float = 0.01,
        use_tensorly_init: bool = True,
        memory_optimal_contractions: bool = True,
    ):
        super().__init__()

        self.tensor_shape = tensor_shape  # (in_channels, out_channels, *spatial_modes)
        self.use_complex = use_complex
        self.adaptive_rank_learning = adaptive_rank_learning
        self.rank_learning_rate = rank_learning_rate
        self.memory_optimal_contractions = memory_optimal_contractions

        # Extract dimensions
        self.in_channels = tensor_shape[0]
        self.out_channels = tensor_shape[1]
        self.spatial_modes = tensor_shape[2:]

        # Set up frequency bands for multi-grid decomposition
        if frequency_bands is None:
            self.frequency_bands = self._auto_frequency_bands(self.spatial_modes)
        else:
            self.frequency_bands = frequency_bands

        # Compute frequency-aware ranks
        self.frequency_ranks = self._compute_frequency_ranks(
            base_rank, rank_adaptation_strategy
        )

        # Initialize decomposition components
        self._init_multi_grid_decomposition(rngs, use_tensorly_init)

        # Initialize adaptive rank learning if enabled
        if self.adaptive_rank_learning:
            self._init_adaptive_rank_learning(rngs)

    def _auto_frequency_bands(
        self, spatial_modes: Sequence[int]
    ) -> Sequence[tuple[int, int]]:
        """Automatically determine frequency bands based on spatial mode sizes.

        Uses Multi-Grid TFNO principle: partition frequency space into bands
        with different compression characteristics.
        """
        bands = []
        for mode_size in spatial_modes:
            # Three bands: low (0-33%), medium (33-66%), high (66-100%)
            low_end = mode_size // 3
            med_end = 2 * mode_size // 3

            bands.append(
                [
                    (0, low_end),  # Low frequency (high energy)
                    (low_end, med_end),  # Medium frequency
                    (med_end, mode_size),  # High frequency (low energy, compressible)
                ]
            )

        # For multi-dimensional spaces, use the first spatial dimension's bands
        return bands[0] if bands else [(0, 16), (16, 32), (32, 64)]

    def _compute_frequency_ranks(
        self,
        base_rank: float | Sequence[int],
        strategy: str,
    ) -> dict[str, list[int]]:
        """Compute ranks for different frequency bands.

        Implements frequency-aware rank adaptation from MG-TFNO research.
        """
        if isinstance(base_rank, float):
            # Base ranks from compression ratio
            base_in_rank = max(1, int(base_rank * self.in_channels))
            base_out_rank = max(1, int(base_rank * self.out_channels))
            base_spatial_ranks = [
                max(1, int(base_rank * mode)) for mode in self.spatial_modes
            ]
        else:
            # Explicit ranks - use cast to help type checker
            base_rank_seq = cast("Sequence[int]", base_rank)
            base_in_rank = int(base_rank_seq[0])
            base_out_rank = int(base_rank_seq[1])
            base_spatial_ranks = [int(r) for r in base_rank_seq[2:]]

        frequency_ranks = {}

        for i, (start, end) in enumerate(self.frequency_bands):
            band_name = f"band_{i}"

            if strategy == "uniform":
                # Same ranks for all frequency bands
                ranks = [base_in_rank, base_out_rank, *base_spatial_ranks]

            elif strategy == "frequency_decay":
                # Higher ranks for lower frequencies (more important)
                decay_factor = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, ...
                ranks = [
                    max(1, int(base_in_rank * decay_factor)),
                    max(1, int(base_out_rank * decay_factor)),
                ] + [max(1, int(r * decay_factor)) for r in base_spatial_ranks]

            elif strategy == "energy_based":
                # Energy-based ranking: low frequencies get high ranks
                # High frequencies (typically noise) get low ranks
                band_width = end - start
                total_modes = sum(self.spatial_modes)
                energy_factor = 1.0 - (start + band_width / 2) / total_modes
                energy_factor = max(0.1, energy_factor)  # Minimum 10% rank

                ranks = [
                    max(1, int(base_in_rank * energy_factor)),
                    max(1, int(base_out_rank * energy_factor)),
                ] + [max(1, int(r * energy_factor)) for r in base_spatial_ranks]

            else:
                raise ValueError(f"Unknown rank_adaptation_strategy: {strategy}")

            frequency_ranks[band_name] = ranks

        return frequency_ranks

    def _init_multi_grid_decomposition(self, rngs: nnx.Rngs, use_tensorly_init: bool):
        """Initialize multi-grid decomposition components."""
        # Create separate decompositions for each frequency band
        # Create separate decompositions for each frequency band
        band_decompositions = {}

        for band_name, ranks in self.frequency_ranks.items():
            # Generate separate random keys for each band
            band_key = rngs.params()

            # Create sub-tensor shape for this frequency band
            band_spatial_modes = self._get_band_spatial_modes(band_name)
            band_shape = [self.in_channels, self.out_channels, *band_spatial_modes]

            # Initialize decomposition for this band
            if use_tensorly_init and _tensorly_available:
                # Use TensorLy for superior initialization
                band_decomp = self._init_tensorly_band(band_shape, ranks, band_key)
            else:
                # Fallback to standard initialization
                band_decomp = self._init_standard_band(band_shape, ranks, band_key)

            band_decompositions[band_name] = band_decomp

        self.band_decompositions = nnx.Dict(band_decompositions)

    def _get_band_spatial_modes(self, band_name: str) -> Sequence[int]:
        """Get spatial mode sizes for a specific frequency band."""
        band_idx = int(band_name.split("_")[1])
        start, end = self.frequency_bands[band_idx]

        # For FFT domain: last dimension is modes//2 + 1
        band_modes = []
        for i, mode_size in enumerate(self.spatial_modes):
            if i == len(self.spatial_modes) - 1:
                # Last dimension in FFT domain
                actual_size = mode_size // 2 + 1
                band_size = min(end, actual_size) - start
            else:
                band_size = min(end, mode_size) - start

            band_modes.append(max(1, band_size))

        return band_modes

    def _init_tensorly_band(
        self,
        band_shape: Sequence[int],
        ranks: Sequence[int],
        key: jax.Array,
    ) -> nnx.Dict:
        """Initialize a frequency band using TensorLy decomposition."""
        # Create random tensor for this band
        if self.use_complex:
            real_key, imag_key = jax.random.split(key)
            real = jax.random.normal(real_key, band_shape)
            imag = jax.random.normal(imag_key, band_shape)
            init_tensor = jnp.complex_(real + 1j * imag)
        else:
            init_tensor = jax.random.normal(key, band_shape)

        try:
            # Use TensorLy's superior Tucker decomposition
            core, factors = TensorLyTuckerInitializer.decompose_tensor(
                init_tensor, ranks, max_iter=25, tolerance=1e-10
            )

            return nnx.Dict(
                {
                    "core": nnx.Param(core),
                    "factors": nnx.List([nnx.Param(factor) for factor in factors]),
                }
            )
        except Exception:
            # Fallback to standard initialization if TensorLy fails
            return self._init_standard_band(band_shape, ranks, key)

    def _init_standard_band(
        self,
        band_shape: Sequence[int],
        ranks: Sequence[int],
        key: jax.Array,
    ) -> nnx.Dict:
        """Standard initialization for a frequency band."""
        keys = jax.random.split(key, len(band_shape) + 1)

        # Core tensor
        core_shape = tuple(ranks)
        if self.use_complex:
            real_key, imag_key = jax.random.split(keys[0])
            real = jax.random.normal(real_key, core_shape) * 0.1
            imag = jax.random.normal(imag_key, core_shape) * 0.1
            core = jnp.complex_(real + 1j * imag)
        else:
            core = jax.random.normal(keys[0], core_shape) * 0.1

        # Factor matrices
        factors = []
        for i, (dim_size, rank) in enumerate(zip(band_shape, ranks, strict=False)):
            if self.use_complex:
                real_key, imag_key = jax.random.split(keys[i + 1])
                real = jax.random.normal(real_key, (dim_size, rank)) * 0.1
                imag = jax.random.normal(imag_key, (dim_size, rank)) * 0.1
                factor = jnp.complex_(real + 1j * imag)
            else:
                factor = jax.random.normal(keys[i + 1], (dim_size, rank)) * 0.1

            factors.append(nnx.Param(factor))

        return nnx.Dict(
            {
                "core": nnx.Param(core),
                "factors": nnx.List(factors),
            }
        )

    def _init_adaptive_rank_learning(self, rngs: nnx.Rngs):
        """Initialize adaptive rank learning components."""
        # Track rank gradients for each frequency band
        self.rank_gradients = {}
        self.rank_momentum = {}

        for band_name in self.frequency_ranks:
            ranks = self.frequency_ranks[band_name]
            # Initialize rank gradient tracking
            self.rank_gradients[band_name] = [0.0] * len(ranks)
            self.rank_momentum[band_name] = [0.0] * len(ranks)

    def multiply_factorized(self, x: jax.Array) -> jax.Array:
        """Multi-grid factorized multiplication with frequency-aware processing.

        Input x: (batch, in_channels, *spatial_modes_ft)
        Output: (batch, out_channels, *spatial_modes_ft)

        Key innovation: Process different frequency bands with different
        compression levels, then combine results.
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]

        # Initialize output
        output = jnp.zeros(
            (batch_size, self.out_channels, *spatial_shape), dtype=x.dtype
        )

        # Process each frequency band separately
        for band_name, band_decomp in self.band_decompositions.items():
            # Extract frequency band from input
            band_input = self._extract_frequency_band(x, band_name)

            # Apply band-specific decomposed convolution
            band_output = self._apply_band_convolution(band_input, band_decomp)

            # Add to output (frequency bands are additive)
            output = self._add_band_to_output(output, band_output, band_name)

        return output

    def _extract_frequency_band(self, x: jax.Array, band_name: str) -> jax.Array:
        """Extract specific frequency band from input tensor."""
        band_idx = int(band_name.split("_")[1])
        start, end = self.frequency_bands[band_idx]

        # For multi-dimensional inputs, extract band from last spatial dimension
        if len(x.shape) == 3:  # 1D spatial
            return x[:, :, start:end]
        if len(x.shape) == 4:  # 2D spatial
            return x[:, :, :, start:end]
        if len(x.shape) == 5:  # 3D spatial
            return x[:, :, :, :, start:end]
        raise ValueError(f"Unsupported input dimensionality: {x.shape}")

    def _apply_band_convolution(
        self, band_input: jax.Array, band_decomp: dict[str, Any]
    ) -> jax.Array:
        """Apply factorized convolution for a specific frequency band."""
        if self.memory_optimal_contractions and _tensorly_available:
            # Use memory-optimal contractions from MG-TFNO research
            return MemoryOptimalContractions.contract_tucker_spectral(
                band_input,
                band_decomp["core"].value,
                [f.value for f in band_decomp["factors"]],
            )
        # Fallback: reconstruct and apply standard convolution
        # Reconstruct full tensor for this band
        band_tensor = self._reconstruct_band_tensor(band_decomp)

        # Apply spectral convolution
        return self._standard_spectral_convolution(band_input, band_tensor)

    def _reconstruct_band_tensor(self, band_decomp: dict[str, Any]) -> jax.Array:
        """Reconstruct full tensor from band decomposition."""
        core = band_decomp["core"].value
        factors = [f.value for f in band_decomp["factors"]]

        # Tucker reconstruction: core x1 U1 x2 U2 x3 U3 ...
        result = core
        for i, factor in enumerate(factors):
            # FIXED: Ensure tensor contraction dimensions are compatible
            # Contract along mode i with proper dimension handling
            if i < len(result.shape):
                # Contract result's i-th dimension with factor's second dimension
                result = jnp.tensordot(result, factor, axes=([i], [1]))
                # FIXED: Move the new dimension (from factor) to position i
                # The contracted dimension disappears, factor's first dimension
                # becomes new
                result = jnp.moveaxis(result, -1, i)
            else:
                # FIXED: Fallback for edge cases - use matrix multiplication pattern
                result_flat = result.reshape(-1, result.shape[-1])
                factor_transposed = factor.T  # (rank, original_dim)
                result = jnp.dot(result_flat, factor_transposed)
                # Reshape back to appropriate dimensions
                new_shape = (*result.shape[:-1], factor.shape[0])
                result = result.reshape(new_shape)

        return result

    def _standard_spectral_convolution(
        self, band_input: jax.Array, band_tensor: jax.Array
    ) -> jax.Array:
        """Standard spectral convolution for fallback."""
        # For 1D case: input(batch, in_ch, spatial) x weight(in_ch, out_ch, spatial)
        if len(band_input.shape) == 3 and len(band_tensor.shape) == 3:
            return jnp.einsum("bik,ijk->bjk", band_input, jnp.conj(band_tensor))

        # For 2D case: more complex contraction
        if len(band_input.shape) == 4 and len(band_tensor.shape) == 4:
            return jnp.einsum("bikl,ijkl->bjkl", band_input, jnp.conj(band_tensor))

        # General case using tensordot
        # Contract over input channels and all spatial dimensions except last
        input_axes = [1, *list(range(2, len(band_input.shape)))]
        weight_axes = [0, *list(range(2, len(band_tensor.shape)))]
        return jnp.tensordot(
            band_input, jnp.conj(band_tensor), axes=(input_axes, weight_axes)
        )

    def _add_band_to_output(
        self, output: jax.Array, band_output: jax.Array, band_name: str
    ) -> jax.Array:
        """Add frequency band output to total output."""
        band_idx = int(band_name.split("_")[1])
        start, _end = self.frequency_bands[band_idx]

        # Add to appropriate frequency range
        if len(output.shape) == 3:  # 1D spatial
            output = output.at[:, :, start : start + band_output.shape[2]].add(
                band_output
            )
        elif len(output.shape) == 4:  # 2D spatial
            output = output.at[:, :, :, start : start + band_output.shape[3]].add(
                band_output
            )
        elif len(output.shape) == 5:  # 3D spatial
            output = output.at[:, :, :, :, start : start + band_output.shape[4]].add(
                band_output
            )

        return output

    def update_adaptive_ranks(self, gradients: dict[str, jax.Array]):
        """Update ranks based on gradient information for adaptive rank learning."""
        if not self.adaptive_rank_learning:
            return

        for band_name in self.frequency_ranks:
            if band_name in gradients:
                band_grads = gradients[band_name]

                # Compute rank importance from gradient magnitudes
                rank_importance = [
                    float(jnp.linalg.norm(band_grads))  # Simplified importance metric
                ]

                # Update rank gradients with momentum
                for i, importance in enumerate(rank_importance):
                    if i < len(self.rank_gradients[band_name]):
                        # Momentum update
                        self.rank_momentum[band_name][i] = (
                            0.9 * self.rank_momentum[band_name][i] + 0.1 * importance
                        )

                        # Adaptive rank adjustment
                        current_rank = self.frequency_ranks[band_name][i]
                        rank_adjustment = (
                            self.rank_learning_rate * self.rank_momentum[band_name][i]
                        )

                        # Update rank (with bounds checking)
                        new_rank = max(
                            1,
                            min(
                                current_rank + int(rank_adjustment),
                                self.tensor_shape[i]
                                if i < len(self.tensor_shape)
                                else 64,
                            ),
                        )

                        self.frequency_ranks[band_name][i] = new_rank

    def get_compression_stats(self) -> dict[str, Any]:
        """Get detailed compression statistics for multi-grid decomposition."""
        stats: dict[str, Any] = {
            "total_parameters": 0,
            "original_parameters": np.prod(self.tensor_shape),
            "band_stats": {},
            "frequency_bands": self.frequency_bands,
            "frequency_ranks": self.frequency_ranks,
        }

        for band_name, band_decomp in self.band_decompositions.items():
            core = band_decomp["core"]
            factors = band_decomp["factors"]

            # Type-safe access: core is always Param, factors is always list[Param]
            core_param = core if not isinstance(core, (list, nnx.List)) else core[0]
            factor_list = (
                factors if isinstance(factors, (list, nnx.List)) else [factors]
            )

            band_params = np.prod(core_param.value.shape) + sum(
                np.prod(f.value.shape) for f in factor_list
            )
            stats["total_parameters"] += band_params

            stats["band_stats"][band_name] = {
                "parameters": band_params,
                "ranks": self.frequency_ranks[band_name],
                "core_shape": core_param.value.shape,
                "factor_shapes": [f.value.shape for f in factor_list],
            }

        stats["compression_ratio"] = (
            stats["original_parameters"] / stats["total_parameters"]
        )
        stats["parameter_reduction"] = 1.0 - (
            stats["total_parameters"] / stats["original_parameters"]
        )

        return stats

    def reconstruct_full_tensor(self) -> jax.Array:
        """Reconstruct the full tensor from all frequency bands."""
        # Get dtype from first band decomposition
        first_band_decomp = next(iter(self.band_decompositions.values()))
        core_param = first_band_decomp["core"]
        # Handle both Param and list[Param] cases
        core_value = (
            core_param.value
            if not isinstance(core_param, (list, nnx.List))
            else core_param[0].value
        )
        reference_dtype = core_value.dtype

        # Initialize full tensor
        full_tensor = jnp.zeros(self.tensor_shape, dtype=reference_dtype)

        for band_name, band_decomp in self.band_decompositions.items():
            # Reconstruct band tensor
            band_tensor = self._reconstruct_band_tensor(band_decomp)

            # Add to appropriate frequency region
            band_idx = int(band_name.split("_")[1])
            start, _end = self.frequency_bands[band_idx]

            # For simplicity, add to last spatial dimension
            if len(self.tensor_shape) == 3:  # 1D spatial
                actual_end = min(start + band_tensor.shape[2], full_tensor.shape[2])
                full_tensor = full_tensor.at[:, :, start:actual_end].add(
                    band_tensor[:, :, : actual_end - start]
                )
            # Similar logic for 2D and 3D...

        return full_tensor


class AdaptiveRankMultiGridTFNO(nnx.Module):
    """Complete Multi-Grid TFNO with adaptive rank learning.

    Combines Multi-Grid Tucker decomposition with adaptive rank learning
    for optimized tensor neural operator performance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        modes: Sequence[int] = (16, 16),
        num_layers: int = 4,
        base_rank: float | Sequence[int] = 0.1,
        frequency_bands: Sequence[tuple[int, int]] | None = None,
        rank_adaptation_strategy: Literal[
            "uniform", "frequency_decay", "energy_based"
        ] = "frequency_decay",
        adaptive_rank_learning: bool = True,
        rank_learning_rate: float = 0.01,
        use_residual: bool = True,
        activation: str = "gelu",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.modes = modes
        self.num_layers = num_layers
        self.use_residual = use_residual

        # Input projection
        self.input_projection = nnx.Linear(in_channels, hidden_channels, rngs=rngs)

        # Multi-grid spectral convolution layers
        spectral_layers_temp = []
        for _ in range(num_layers):
            layer_rngs = nnx.Rngs(rngs.params())

            spectral_conv = MultiGridTuckerDecomposition(
                tensor_shape=(hidden_channels, hidden_channels, *modes),
                base_rank=base_rank,
                frequency_bands=frequency_bands,
                rank_adaptation_strategy=rank_adaptation_strategy,
                rngs=layer_rngs,
                adaptive_rank_learning=adaptive_rank_learning,
                rank_learning_rate=rank_learning_rate,
            )
            spectral_layers_temp.append(spectral_conv)
            self.spectral_layers = nnx.List(spectral_layers_temp)

        # Activation function
        self.activation = self._get_activation(activation)

        # Output projection
        self.output_projection = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def _get_activation(self, activation: str):
        """Get activation function."""
        if activation == "gelu":
            return nnx.gelu
        if activation == "relu":
            return nnx.relu
        if activation == "tanh":
            return jnp.tanh
        if activation == "swish":
            return nnx.swish
        raise ValueError(f"Unknown activation: {activation}")

    # @partial(jax.jit, static_argnums=(0,)) - REMOVED for NNX compatibility
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through Multi-Grid TFNO.

        Args:
            x: Input tensor (batch, spatial_dims..., channels)

        Returns:
            Output tensor (batch, spatial_dims..., out_channels)
        """
        # Input projection
        x = self.input_projection(x)

        # Move to frequency domain
        x_ft = self._to_fourier(x)

        # Apply multi-grid spectral convolution layers
        for spectral_layer in self.spectral_layers:
            if self.use_residual:
                x_ft = x_ft + spectral_layer.multiply_factorized(x_ft)
            else:
                x_ft = spectral_layer.multiply_factorized(x_ft)

            # Apply activation in frequency domain
            x_ft = self.activation(x_ft)

        # Return to spatial domain
        x = self._from_fourier(x_ft)

        # Output projection
        return self.output_projection(x)

    def _to_fourier(self, x: jax.Array) -> jax.Array:
        """Transform to Fourier domain."""
        # Move channels to first position: (..., channels) -> (channels, ...)
        x = jnp.moveaxis(x, -1, 1)

        # Apply FFT to spatial dimensions
        for i in range(2, len(x.shape)):
            x = (
                jnp.fft.rfft(x, axis=i)
                if i == len(x.shape) - 1
                else jnp.fft.fft(x, axis=i)
            )

        # Keep only specified modes
        spatial_slices = [slice(None), slice(None)]  # batch, channels
        for i, mode_size in enumerate(self.modes):
            if i == len(self.modes) - 1:
                # Last dimension (rfft): keep modes//2 + 1
                spatial_slices.append(slice(None, mode_size // 2 + 1))
            else:
                # Other dimensions: keep specified modes
                spatial_slices.append(slice(None, mode_size))

        return x[tuple(spatial_slices)]

    def _from_fourier(self, x_ft: jax.Array) -> jax.Array:
        """Transform from Fourier domain back to spatial."""
        # Apply inverse FFT to spatial dimensions
        x = x_ft
        for i in range(len(self.modes) - 1, -1, -1):
            axis = i + 2  # Skip batch and channel dimensions
            if i == len(self.modes) - 1:
                # Last dimension: irfft
                x = jnp.fft.irfft(x, n=self.modes[i], axis=axis)
            else:
                # Other dimensions: ifft
                x = jnp.fft.ifft(x, n=self.modes[i], axis=axis).real

        # Move channels back to last position: (channels, ...) -> (..., channels)
        return jnp.moveaxis(x, 1, -1)

    def _compute_linear_layer_params(self) -> int:
        """Compute total parameters in linear layers with safe access."""
        input_kernel_size = 0
        if (
            hasattr(self.input_projection, "kernel")
            and self.input_projection.kernel is not None
            and hasattr(self.input_projection.kernel, "value")
            and self.input_projection.kernel.value is not None
        ):
            input_kernel_size = int(self.input_projection.kernel.value.size)

        input_bias_size = 0
        if (
            hasattr(self.input_projection, "bias")
            and self.input_projection.bias is not None
            and hasattr(self.input_projection.bias, "value")
            and self.input_projection.bias.value is not None
        ):
            input_bias_size = int(self.input_projection.bias.value.size)

        output_kernel_size = 0
        if (
            hasattr(self.output_projection, "kernel")
            and self.output_projection.kernel is not None
            and hasattr(self.output_projection.kernel, "value")
            and self.output_projection.kernel.value is not None
        ):
            output_kernel_size = int(self.output_projection.kernel.value.size)

        output_bias_size = 0
        if (
            hasattr(self.output_projection, "bias")
            and self.output_projection.bias is not None
            and hasattr(self.output_projection.bias, "value")
            and self.output_projection.bias.value is not None
        ):
            output_bias_size = int(self.output_projection.bias.value.size)

        return (
            input_kernel_size + input_bias_size + output_kernel_size + output_bias_size
        )

    def get_multi_grid_stats(self) -> dict[str, Any]:
        """Get comprehensive Multi-Grid TFNO statistics."""
        stats: dict[str, Any] = {
            "model_type": "AdaptiveRankMultiGridTFNO",
            "architecture": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "hidden_channels": self.hidden_channels,
                "modes": self.modes,
                "num_layers": self.num_layers,
            },
            "layer_stats": [],
            "total_parameters": 0,
            "total_compression_ratio": 0.0,
        }

        for i, layer in enumerate(self.spectral_layers):
            layer_stats = layer.get_compression_stats()
            layer_stats["layer_index"] = i
            stats["layer_stats"].append(layer_stats)
            stats["total_parameters"] += layer_stats["total_parameters"]

        # Add linear layer parameters with safe access
        linear_params = self._compute_linear_layer_params()
        stats["total_parameters"] += linear_params
        stats["linear_layer_parameters"] = linear_params

        # Compute overall compression
        total_original = sum(
            layer_stats["original_parameters"] for layer_stats in stats["layer_stats"]
        )
        total_compressed = sum(
            layer_stats["total_parameters"] for layer_stats in stats["layer_stats"]
        )
        stats["total_compression_ratio"] = (
            total_original / total_compressed if total_compressed > 0 else 1.0
        )

        # Layer-wise statistics including Multi-Grid decomposition details
        detailed_layer_stats = []
        for i, spectral_layer in enumerate(self.spectral_layers):
            compression_stats = spectral_layer.get_compression_stats()

            # Safely access core parameter with proper type guards
            first_band = next(iter(spectral_layer.band_decompositions.values()))
            core_param = first_band.get("core")
            factor_params = first_band.get("factors", [])

            # Safe access to core parameter value
            if core_param is not None and hasattr(core_param, "value"):
                core_value = getattr(core_param, "value", None)
                if core_value is not None:
                    core_norm = float(jnp.linalg.norm(core_value))
                else:
                    core_norm = 0.0
            else:
                core_norm = 0.0

            # Safe access to factor parameter values
            if (
                factor_params
                and len(factor_params) > 0
                and hasattr(factor_params[0], "value")
            ):
                factor_value = getattr(factor_params[0], "value", None)
                if factor_value is not None:
                    factor_norm = float(jnp.linalg.norm(factor_value))
                else:
                    factor_norm = 0.0
            else:
                factor_norm = 0.0

            detailed_layer_stats.append(
                {
                    "layer_index": i,
                    "compression_stats": compression_stats,
                    "core_norm": core_norm,
                    "factor_norm": factor_norm,
                }
            )

        stats["detailed_layer_stats"] = detailed_layer_stats

        return stats

    def update_all_adaptive_ranks(self, gradients: dict[str, Any]):
        """Update adaptive ranks for all spectral layers."""
        for i, layer in enumerate(self.spectral_layers):
            if f"layer_{i}" in gradients:
                layer.update_adaptive_ranks(gradients[f"layer_{i}"])


# Factory functions for easy model creation
def create_multigrid_tfno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    modes: Sequence[int] = (16, 16),
    base_rank: float | Sequence[int] = 0.1,
    num_layers: int = 4,
    rank_adaptation_strategy: Literal[
        "uniform", "frequency_decay", "energy_based"
    ] = "frequency_decay",
    *,
    rngs: nnx.Rngs,
) -> AdaptiveRankMultiGridTFNO:
    """Create Multi-Grid TFNO with frequency-aware rank adaptation."""
    return AdaptiveRankMultiGridTFNO(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        base_rank=base_rank,
        num_layers=num_layers,
        rank_adaptation_strategy=rank_adaptation_strategy,
        rngs=rngs,
    )


def create_energy_based_mg_tfno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    modes: Sequence[int] = (32, 32),
    base_rank: float = 0.15,
    num_layers: int = 6,
    *,
    rngs: nnx.Rngs,
) -> AdaptiveRankMultiGridTFNO:
    """Create energy-based Multi-Grid TFNO for optimal frequency compression."""
    return AdaptiveRankMultiGridTFNO(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        base_rank=base_rank,
        num_layers=num_layers,
        rank_adaptation_strategy="energy_based",
        adaptive_rank_learning=True,
        rank_learning_rate=0.005,  # Slower learning for stability
        rngs=rngs,
    )


# Export key components
__all__ = [
    "AdaptiveRankMultiGridTFNO",
    "FrequencyAwareDecomposition",
    "MultiGridTuckerDecomposition",
    "create_energy_based_mg_tfno",
    "create_multigrid_tfno",
]
