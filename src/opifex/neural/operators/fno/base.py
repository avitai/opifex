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

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.fno._decompositions import make_decomposition
from opifex.neural.operators.fno._factorized import factorized_spectral_conv
from opifex.neural.operators.fno._positional import append_grid_coordinates


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
    ) -> None:
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
        factorization: str | None = None,
        factorization_rank: float | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize Fourier layer following NNX patterns.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes
            activation: Activation function
            spatial_dims: Number of spatial dimensions (1, 2, or 3). Controls which
                spectral weights are allocated — avoids dead parameters.
            factorization: Optional low-rank factorization of the spectral weight
                ('tucker', 'cp', or 'tt'); ``None`` uses a dense weight.
            factorization_rank: Compression ratio for the factorization (per-mode
                Tucker ratio, or ratio of ``min(shape)`` for CP/TT); defaults to 0.5.
            rngs: Random number generators (keyword-only)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.activation = activation
        self.spatial_dims = spatial_dims
        self.factorization = factorization

        # Spectral weights — only allocate for the target dimensionality.
        # Stored as separate real/imaginary Params to avoid JAX complex gradient
        # convention issue (optax issue #196) — see FourierSpectralConvolution docstring.
        scale = 1.0 / (in_channels * out_channels)

        if factorization is not None:
            # Low-rank factorized spectral weight (CP / Tucker / TT).
            tensor_shape = (out_channels, in_channels, *((modes,) * spatial_dims))
            rank = factorization_rank if factorization_rank is not None else 0.5
            self.decomposition = make_decomposition(
                factorization, tensor_shape, float(rank), rngs=rngs
            )
        elif spatial_dims == 1:
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
        if self.factorization is not None:
            return factorized_spectral_conv(
                self.decomposition, x, (self.modes,) * self.spatial_dims
            )
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

    def get_compression_stats(self) -> dict[str, float]:
        """Report factorized-vs-dense parameter compression for this layer.

        Returns:
            Mapping with the factorized parameter count, the equivalent dense
            spectral-weight count, their ratio, and the fractional reduction.

        Raises:
            ValueError: If the layer uses dense (non-factorized) spectral weights.
        """
        if self.factorization is None:
            raise ValueError("get_compression_stats is only defined for factorized layers")
        factorized = self.decomposition.parameter_count()
        dense = self.in_channels * self.out_channels * (self.modes**self.spatial_dims)
        ratio = factorized / dense if dense > 0 else 0.0
        return {
            "factorized_parameters": float(factorized),
            "equivalent_dense_parameters": float(dense),
            "compression_ratio": float(ratio),
            "parameter_reduction": float(1.0 - ratio),
        }


def spectral_resample(
    x: jax.Array,
    output_size: tuple[int, ...],
    axes: Sequence[int],
) -> jax.Array:
    """Resize a real spatial field along ``axes`` purely in the Fourier domain.

    The field is transformed with a real n-dimensional FFT, its low-frequency
    modes (around DC, plus the matching negative frequencies on every non-last
    axis) are copied into a new-sized spectral array, and an inverse real FFT of
    size ``output_size`` is taken. Because data only moves between Fourier modes
    — never through pixel pooling or spatial interpolation — this is the
    discretisation-invariant resize used by U-NO: a band-limited field sampled
    at any resolution maps to the same continuous function.

    Mirrors the spectral branch of ``neuralop.layers.resample.resample`` (mode
    indexing and ``norm="forward"``), adapted to JAX. ``output_size`` is a static
    Python tuple so the function traces cleanly under ``jax.jit``.

    Reference: Rahman et al., "U-NO: U-shaped Neural Operators", TMLR 2022,
    https://arxiv.org/abs/2204.11127, and the neuraloperator library.

    Args:
        x: Input of shape ``(batch, channels, *spatial)`` (channels-first).
        output_size: Target spatial size, one entry per axis in ``axes``.
        axes: Spatial axes to resize (e.g. ``(-2, -1)`` for 2D).

    Returns:
        Resized field with the spatial extent on ``axes`` set to ``output_size``.
    """
    axes = tuple(axes)
    old_size = tuple(x.shape[a] for a in axes)
    if old_size == tuple(output_size):
        return x

    x_ft = jnp.fft.rfftn(x, axes=axes, norm="forward")

    # rfft halves the last axis; build the target spectral shape accordingly.
    new_fft_size = list(output_size)
    new_fft_size[-1] = new_fft_size[-1] // 2 + 1
    src_fft_size = tuple(x_ft.shape[a] for a in axes)
    # Only copy modes that exist in BOTH the source and target spectra.
    keep = [min(n, s) for n, s in zip(new_fft_size, src_fft_size, strict=True)]

    out_shape = list(x_ft.shape)
    for a, size in zip(axes, new_fft_size, strict=True):
        out_shape[a] = size
    out_fft = jnp.zeros(tuple(out_shape), dtype=x_ft.dtype)

    # Low positive + low negative frequencies around DC on every non-last axis;
    # only the low (non-redundant) half on the last (rfft) axis.
    mode_indexing = [((None, m // 2), (-(m // 2), None)) for m in keep[:-1]] + [((None, keep[-1]),)]
    lead = (slice(None),) * (x.ndim - len(axes))  # leading (batch, channels, ...) axes
    for boundaries in itertools.product(*mode_indexing):
        idx = (*lead, *(slice(*b) for b in boundaries))
        out_fft = out_fft.at[idx].set(x_ft[idx])

    return jnp.fft.irfftn(out_fft, s=tuple(output_size), axes=axes, norm="forward")


def _resolve_output_size(
    spatial: tuple[int, ...],
    output_shape: tuple[int, ...] | None,
    scaling_factor: Sequence[float] | None,
) -> tuple[int, ...]:
    """Resolve the target spatial size from an explicit shape or a scale factor.

    ``output_shape`` takes precedence; otherwise ``round(N * scale)`` per axis;
    otherwise the input size is preserved.
    """
    if output_shape is not None:
        return tuple(output_shape)
    if scaling_factor is not None:
        return tuple(round(n * s) for n, s in zip(spatial, scaling_factor, strict=True))
    return spatial


class SpectralConvResize(nnx.Module):
    """2D spectral convolution that can change resolution in the Fourier domain.

    Combines the Li et al. (2021) low-mode spectral contraction with the U-NO
    Fourier-domain resize: after contracting the kept modes, the inverse real
    FFT is taken at the requested output size, so the layer maps a field at
    resolution ``N`` to one at ``round(N * scale)`` without any strided
    convolution or pixel interpolation. This is what makes U-NO discretisation
    invariant (genuine zero-shot super-resolution).

    Both quadrants of the ``rfft2`` spectrum (positive and negative
    y-frequencies) are used with separate weight tensors, matching opifex's
    :class:`FourierLayer` 2D convention. Weights are stored as separate
    real/imaginary :class:`nnx.Param` arrays to avoid the JAX complex-gradient
    convention issue (optax issue #196).

    Reference: Rahman et al., "U-NO: U-shaped Neural Operators", TMLR 2022,
    https://arxiv.org/abs/2204.11127, and
    ``neuralop.layers.spectral_convolution.SpectralConv``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the resolution-scaling spectral convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_modes: Number of retained Fourier modes ``(modes_h, modes_w)``.
            rngs: Random number generators (keyword-only).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h, self.modes_w = int(n_modes[0]), int(n_modes[1])

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, self.modes_h, self.modes_w)
        k1r, k1i, k2r, k2i = jax.random.split(rngs.params(), 4)
        self.weights_1_real = nnx.Param(scale * jax.random.uniform(k1r, shape))
        self.weights_1_imag = nnx.Param(scale * jax.random.uniform(k1i, shape))
        self.weights_2_real = nnx.Param(scale * jax.random.uniform(k2r, shape))
        self.weights_2_imag = nnx.Param(scale * jax.random.uniform(k2i, shape))

    def __call__(
        self,
        x: jax.Array,
        *,
        output_scaling_factor: Sequence[float] | None = None,
        output_shape: tuple[int, ...] | None = None,
    ) -> jax.Array:
        """Apply the spectral convolution and inverse-transform at a new size.

        Args:
            x: Input of shape ``(batch, in_channels, height, width)``.
            output_scaling_factor: Per-axis spatial scale; the output size is
                ``round(N * scale)``. Ignored if ``output_shape`` is given.
            output_shape: Explicit output spatial size ``(height, width)``.

        Returns:
            Output of shape ``(batch, out_channels, *output_size)``.
        """
        batch_size = x.shape[0]
        h, w = x.shape[-2:]
        out_h, out_w = _resolve_output_size((h, w), output_shape, output_scaling_factor)

        x_ft = jnp.fft.rfft2(x, axes=(-2, -1), norm="forward")
        ft_h, ft_w = x_ft.shape[-2:]
        modes_h = min(self.modes_h, ft_h // 2)
        modes_w = min(self.modes_w, ft_w)

        w1 = (
            self.weights_1_real[...][:, :, :modes_h, :modes_w]
            + 1j * self.weights_1_imag[...][:, :, :modes_h, :modes_w]
        )
        w2 = (
            self.weights_2_real[...][:, :, :modes_h, :modes_w]
            + 1j * self.weights_2_imag[...][:, :, :modes_h, :modes_w]
        )

        # Positive and negative y-frequency quadrants (top-left / bottom-left).
        out_1 = jnp.einsum("bixy,ioxy->boxy", x_ft[:, :, :modes_h, :modes_w], w1)
        out_2 = jnp.einsum("bixy,ioxy->boxy", x_ft[:, :, -modes_h:, :modes_w], w2)

        # Assemble in a spectrum sized for the OUTPUT resolution, so the inverse
        # real FFT directly produces the resized field (U-NO Fourier resize).
        out_ft_h = out_h
        out_ft_w = out_w // 2 + 1
        out_ft = jnp.zeros((batch_size, self.out_channels, out_ft_h, out_ft_w), dtype=out_1.dtype)
        place_h = min(modes_h, out_ft_h // 2 if out_ft_h > 1 else out_ft_h)
        place_w = min(modes_w, out_ft_w)
        out_ft = out_ft.at[:, :, :place_h, :place_w].set(out_1[:, :, :place_h, :place_w])
        out_ft = out_ft.at[:, :, -place_h:, :place_w].set(out_2[:, :, -place_h:, :place_w])

        return jnp.fft.irfft2(out_ft, s=(out_h, out_w), axes=(-2, -1), norm="forward")


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
        factorization_rank: float | None = None,
        positional_embedding: bool = False,
        use_mixed_precision: bool = False,
        domain_padding: float = 0.0,
        spatial_dims: int = 2,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize Fourier Neural Operator following NNX patterns.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Number of hidden channels
            modes: Number of Fourier modes
            num_layers: Number of Fourier layers
            activation: Activation function
            factorization_type: Optional tensor factorization ('tucker', 'cp', 'tt')
            factorization_rank: Rank for tensor factorization
            positional_embedding: If True, append normalised grid-coordinate
                channels to the input before lifting (needed for boundary-value
                problems such as Darcy flow).
            use_mixed_precision: Whether to use mixed precision
            domain_padding: Fraction of each spatial dimension to zero-pad before the
                spectral layers (reduces the Gibbs phenomenon for non-periodic problems
                such as Darcy flow). Specified as a fraction (e.g. 0.25), NOT pixels, so
                the padding scales with resolution and preserves the FNO's
                discretisation-invariance / zero-shot super-resolution property. 0 disables.
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
        self.positional_embedding = positional_embedding

        # Input projection (lifting). With positional embedding the lifted input
        # also carries one normalised coordinate channel per spatial axis.
        lifting_in_channels = in_channels + (spatial_dims if positional_embedding else 0)
        self.input_projection = nnx.Linear(
            in_features=lifting_in_channels,
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
        """Create a Fourier layer, dense or low-rank factorized (CP / Tucker / TT)."""
        return FourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            activation=activation,
            spatial_dims=spatial_dims,
            factorization=self.factorization_type,
            factorization_rank=self.factorization_rank,
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
        # Positional embedding: append normalised grid-coordinate channels so the
        # translation-equivariant operator can resolve boundary-dependent solutions.
        if self.positional_embedding:
            x = append_grid_coordinates(x)

        # Input projection (lifting)
        x = self._apply_pointwise_linear(x, self.input_projection)

        # Domain padding for non-periodic problems (reduces Gibbs phenomenon). The pad is a
        # FRACTION of each spatial size computed from the current input, so it scales with
        # resolution and keeps the operator discretisation-invariant (zero-shot super-resolution).
        pad_amounts = [round(self.domain_padding * size) for size in x.shape[2:]]
        if self.domain_padding > 0:
            pad_widths = [(0, 0), (0, 0)] + [(0, pad) for pad in pad_amounts]
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

        # Remove domain padding (per-axis amounts computed above)
        if self.domain_padding > 0:
            slices = [slice(None), slice(None)] + [
                (slice(None, -pad) if pad > 0 else slice(None)) for pad in pad_amounts
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

    def get_compression_stats(self) -> dict[str, float]:
        """Aggregate factorized-vs-dense spectral compression across all layers.

        Returns:
            Mapping with summed factorized and equivalent-dense spectral parameter
            counts, their ratio, and the fractional reduction.

        Raises:
            ValueError: If the operator uses dense (non-factorized) spectral weights.
        """
        if self.factorization_type is None:
            raise ValueError("get_compression_stats is only defined for factorized operators")
        per_layer = [layer.get_compression_stats() for layer in self.fourier_layers]
        factorized = sum(stats["factorized_parameters"] for stats in per_layer)
        dense = sum(stats["equivalent_dense_parameters"] for stats in per_layer)
        ratio = factorized / dense if dense > 0 else 0.0
        return {
            "factorized_parameters": float(factorized),
            "equivalent_dense_parameters": float(dense),
            "compression_ratio": float(ratio),
            "parameter_reduction": float(1.0 - ratio),
        }

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
