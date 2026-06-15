"""Spherical Fourier Neural Operator (SFNO) implementation.

This module provides FNO variants for data naturally defined on spherical
domains using spherical harmonic decompositions. Ideal for global climate
modeling, atmospheric science, and planetary-scale phenomena.

The spectral transform is a genuine orthonormalized real spherical harmonic
transform (SHT) -- a faithful JAX port of NVIDIA ``torch-harmonics`` provided by
:mod:`opifex.neural.operators.fno._spherical_harmonics` -- replacing the earlier
2D-FFT approximation. The spectral-conv weight multiply mirrors the SFNO spherical
convolution of Bonev et al. 2023 (arXiv:2306.03838) and ``neuralop`` ``SphericalConv``.
"""

from collections.abc import Callable, Sequence
from functools import cache

import jax
import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array

from opifex.neural.operators.fno._spherical_harmonics import SphericalHarmonicBasis


@cache
def _get_spherical_basis(
    nlat: int, nlon: int, lmax: int, mmax: int, grid: str
) -> SphericalHarmonicBasis:
    """Return a cached real SHT basis for a fixed grid and truncation.

    The basis holds only static (closed-over) JAX-array constants, so caching it
    by concrete grid/truncation keeps the forward/inverse transforms ``jit`` /
    ``grad`` / ``vmap`` compatible while avoiding repeated Legendre precompute.

    Args:
        nlat: Number of latitude grid points.
        nlon: Number of longitude grid points.
        lmax: Maximum spherical harmonic degree ``+ 1`` (non-inclusive).
        mmax: Maximum azimuthal order ``+ 1`` (non-inclusive).
        grid: Latitude quadrature grid identifier.

    Returns:
        Cached :class:`SphericalHarmonicBasis` for the requested configuration.
    """
    return SphericalHarmonicBasis(nlat=nlat, nlon=nlon, lmax=lmax, mmax=mmax, grid=grid)


class SphericalHarmonicConvolution(nnx.Module):
    """Spherical harmonic convolution for spherical domains.

    Operates in spherical harmonic space analogous to how standard FNO operates
    in Fourier space, but adapted for spherical geometry. The coefficient layout
    is ``(batch, channels, lmax, mmax)`` with non-negative orders ``m`` only,
    matching the real SHT of ``torch-harmonics`` / ``neuralop`` ``SphericalConv``.
    A learnable complex weight contracts the channel axis per spherical mode.
    """

    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lmax: int,  # Maximum spherical harmonic degree (non-inclusive)
        mmax: int | None = None,  # Maximum azimuthal order (non-inclusive)
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize spherical harmonic convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            lmax: Maximum spherical harmonic degree (controls resolution).
            mmax: Maximum azimuthal order (if ``None``, uses ``lmax``).
            rngs: Random number generator state.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lmax = lmax
        self.mmax = mmax if mmax is not None else lmax

        # Spherical harmonic weights in the real-SHT (l, m) layout.
        # Shape: (in_channels, out_channels, lmax, mmax) with m >= 0.
        weight_shape = (in_channels, out_channels, lmax, self.mmax)
        scale = (2 / (in_channels + out_channels)) ** 0.5

        # Store real/imaginary parts separately to avoid JAX complex gradient
        # convention issue (optax #196). See FourierSpectralConvolution docstring.
        self.weight_real = nnx.Param(jax.random.normal(rngs.params(), weight_shape) * scale)
        self.weight_imag = nnx.Param(jax.random.normal(rngs.params(), weight_shape) * scale)

    def __call__(self, x_sht: Array) -> Array:
        """Apply spherical harmonic convolution.

        Args:
            x_sht: Spherical harmonic coefficients ``(batch, channels, lmax, mmax)``.

        Returns:
            Transformed coefficients ``(batch, out_channels, lmax, mmax)``.
        """
        # Truncate the coefficients to the configured (lmax, mmax) band.
        l_end = min(self.lmax, x_sht.shape[-2])
        m_end = min(self.mmax, x_sht.shape[-1])
        x_modes = x_sht[:, :, :l_end, :m_end]

        # Construct the complex weight from real/imaginary parts and align to the
        # truncated band (mirrors neuralop SphericalConv weight slicing).
        weight_real = self.weight_real[:, :, :l_end, :m_end]
        weight_imag = self.weight_imag[:, :, :l_end, :m_end]
        weight = weight_real + 1j * weight_imag

        # Per-mode channel contraction (analogous to the FNO spectral convolution).
        return jnp.einsum("bilm,iolm->bolm", x_modes, weight)


class SphericalFourierNeuralOperator(nnx.Module):
    """
    Spherical Fourier Neural Operator for data on spherical domains.

    Uses spherical harmonic transforms instead of regular FFTs,
    making it ideal for:
    - Global atmospheric modeling
    - Ocean circulation
    - Planetary science
    - Any data naturally defined on spheres
    """

    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int | None = None,
        num_layers: int = 4,
        activation: Callable = nnx.gelu,
        grid: str = "legendre-gauss",  # Latitude quadrature grid for the SHT
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize Spherical FNO.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Hidden layer width.
            lmax: Maximum spherical harmonic degree (controls spectral resolution).
            mmax: Maximum azimuthal order (if ``None``, uses ``lmax``).
            num_layers: Number of SFNO layers.
            activation: Activation function.
            grid: Latitude quadrature grid for the real SHT
                (``"legendre-gauss"``).
            rngs: Random number generator state.
        """
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax if mmax is not None else lmax
        self.num_layers = num_layers
        self.activation = activation
        self.grid = grid

        # Lifting layer
        self.lifting = nnx.Linear(in_channels, hidden_channels, rngs=rngs)

        # Spherical convolution layers
        for i in range(num_layers):
            conv = SphericalHarmonicConvolution(
                hidden_channels, hidden_channels, lmax, mmax, rngs=rngs
            )
            skip = nnx.Linear(hidden_channels, hidden_channels, rngs=rngs)
            setattr(self, f"conv_{i}", conv)
            setattr(self, f"skip_{i}", skip)

        # Projection layer
        self.projection = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def _basis_for(self, nlat: int, nlon: int) -> SphericalHarmonicBasis:
        """Return the cached real SHT basis for the given spatial grid.

        Args:
            nlat: Number of latitude grid points.
            nlon: Number of longitude grid points.

        Returns:
            The :class:`SphericalHarmonicBasis` for ``(nlat, nlon)`` truncated to
            this operator's ``(lmax, mmax)``.
        """
        return _get_spherical_basis(nlat, nlon, self.lmax, self.mmax, self.grid)

    def _spherical_harmonic_transform(self, x: Array) -> Array:
        """Compute the forward real spherical harmonic transform.

        Uses the orthonormalized real SHT (forward/analysis) ported from
        ``torch-harmonics`` rather than a 2D-FFT approximation: a real FFT over
        longitude followed by a Gauss-Legendre latitude quadrature against the
        associated Legendre polynomials.

        Args:
            x: Input on sphere ``(batch, channels, nlat, nlon)``.

        Returns:
            Complex SHT coefficients ``(batch, channels, lmax, mmax)``.
        """
        nlat, nlon = x.shape[-2:]
        return self._basis_for(nlat, nlon).forward(x)

    def _inverse_spherical_harmonic_transform(
        self, x_sht: Array, target_shape: Sequence[int]
    ) -> Array:
        """Compute the inverse real spherical harmonic transform (synthesis).

        Args:
            x_sht: SHT coefficients ``(batch, channels, lmax, mmax)``.
            target_shape: Target spatial shape ``(nlat, nlon)``.

        Returns:
            Real spatial field on sphere ``(batch, channels, nlat, nlon)``.
        """
        nlat, nlon = target_shape
        return self._basis_for(nlat, nlon).inverse(x_sht)

    @staticmethod
    def _embed_coefficients(coeffs: Array, full_shape: Sequence[int]) -> Array:
        """Zero-pad convolved coefficients back to the full SHT grid.

        The spherical convolution may act on a truncated ``(l, m)`` band; the
        inverse transform requires coefficients on the basis's full grid, so the
        convolved band is re-embedded with zero padding for the dropped modes.

        Args:
            coeffs: Convolved coefficients ``(batch, channels, l_band, m_band)``.
            full_shape: Target coefficient shape from the forward transform.

        Returns:
            Coefficients zero-padded to ``full_shape``.
        """
        pad_l = full_shape[-2] - coeffs.shape[-2]
        pad_m = full_shape[-1] - coeffs.shape[-1]
        if pad_l == 0 and pad_m == 0:
            return coeffs
        return jnp.pad(coeffs, ((0, 0), (0, 0), (0, pad_l), (0, pad_m)))

    def __call__(self, x: Array) -> Array:
        """
        Forward pass through SFNO.

        Args:
            x: Input tensor on sphere (batch, in_channels, nlat, nlon)
               nlat: number of latitude points
               nlon: number of longitude points

        Returns:
            Output tensor (batch, out_channels, nlat, nlon)
        """
        spatial_shape = x.shape[-2:]

        # Lifting
        x = jnp.moveaxis(x, 1, -1)  # Move channels to last: (batch, *spatial, channels)
        x = self.lifting(x)  # Apply linear layer
        x = jnp.moveaxis(x, -1, 1)  # Move channels back: (batch, channels, *spatial)

        # SFNO layers
        for i in range(self.num_layers):
            conv = getattr(self, f"conv_{i}")
            skip = getattr(self, f"skip_{i}")

            # Spherical harmonic transform (forward / analysis)
            x_sht = self._spherical_harmonic_transform(x)

            # Spherical convolution in (l, m) coefficient space
            x_conv = conv(x_sht)

            # Re-embed the (possibly truncated) convolved band into the full SHT
            # coefficient grid before synthesis.
            x_conv = self._embed_coefficients(x_conv, x_sht.shape)

            # Inverse spherical harmonic transform (synthesis)
            x_conv = self._inverse_spherical_harmonic_transform(x_conv, spatial_shape)

            # Skip connection and activation - FIXED: Handle channel dimensions properly
            x_skip_input = jnp.moveaxis(x, 1, -1)  # Move channels to last
            x_skip = skip(x_skip_input)
            x_skip = jnp.moveaxis(x_skip, -1, 1)  # Move channels back to second position

            x = self.activation(x_conv + x_skip)

        # Projection - FIXED: Handle channel dimensions properly
        x_proj_input = jnp.moveaxis(x, 1, -1)  # Move channels to last
        x_proj = self.projection(x_proj_input)
        return jnp.moveaxis(x_proj, -1, 1)  # Move channels back to second position

    def get_spherical_modes(self, x: Array) -> Array:
        """
        Get spherical harmonic coefficients for analysis.

        Args:
            x: Input tensor on sphere

        Returns:
            Spherical harmonic coefficients
        """
        x = jnp.moveaxis(x, 1, -1)  # Move channels to last: (batch, *spatial, channels)
        x = self.lifting(x)  # Apply linear layer
        x = jnp.moveaxis(x, -1, 1)  # Move channels back: (batch, channels, *spatial)
        return self._spherical_harmonic_transform(x)

    def compute_power_spectrum(self, x: Array) -> Array:
        """Compute the spherical harmonic power spectrum per degree ``l``.

        The real SHT stores only non-negative orders ``m``; the negative orders of
        a real field are their conjugates, so the angular power at degree ``l`` is
        ``|c_l^0|^2 + 2 * sum_{m>0} |c_l^m|^2``.

        Args:
            x: Input tensor on sphere ``(batch, channels, nlat, nlon)``.

        Returns:
            Power spectrum ``(batch, channels, lmax)`` as a function of degree.
        """
        x_sht = self.get_spherical_modes(x)
        squared = jnp.abs(x_sht) ** 2  # (batch, channels, lmax, mmax)

        # Double the m > 0 contributions to account for the folded negative orders.
        order_weights = jnp.full((squared.shape[-1],), 2.0).at[0].set(1.0)
        return jnp.sum(squared * order_weights, axis=-1)


# Utility functions for climate/atmospheric applications
def create_climate_sfno(
    in_channels: int = 5,  # T, P, humidity, u_wind, v_wind
    out_channels: int = 5,  # Next state
    lmax: int = 32,
    **kwargs,
) -> SphericalFourierNeuralOperator:
    """Create SFNO optimized for global climate modeling."""
    return SphericalFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,
        lmax=lmax,
        mmax=lmax,
        num_layers=6,
        **kwargs,
    )


def create_weather_sfno(
    in_channels: int = 7,  # Extended meteorological variables
    out_channels: int = 7,
    lmax: int = 64,  # Higher resolution for weather
    **kwargs,
) -> SphericalFourierNeuralOperator:
    """Create SFNO for high-resolution weather prediction."""
    return SphericalFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=256,
        lmax=lmax,
        mmax=lmax,
        num_layers=8,
        **kwargs,
    )


def create_ocean_sfno(
    in_channels: int = 4,  # Temperature, salinity, u_current, v_current
    out_channels: int = 4,
    lmax: int = 48,
    **kwargs,
) -> SphericalFourierNeuralOperator:
    """Create SFNO for global ocean circulation modeling."""
    return SphericalFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=160,
        lmax=lmax,
        mmax=lmax,
        num_layers=6,
        **kwargs,
    )


def create_planetary_sfno(
    in_channels: int = 3,  # Generic planetary variables
    out_channels: int = 3,
    lmax: int = 16,  # Lower resolution for planetary scale
    **kwargs,
) -> SphericalFourierNeuralOperator:
    """Create SFNO for planetary-scale phenomena."""
    return SphericalFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=96,
        lmax=lmax,
        mmax=lmax,
        num_layers=4,
        **kwargs,
    )


__all__ = [
    "SphericalFourierNeuralOperator",
    "SphericalHarmonicConvolution",
    "create_climate_sfno",
    "create_ocean_sfno",
    "create_planetary_sfno",
    "create_weather_sfno",
]
