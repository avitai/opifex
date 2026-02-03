# FILE PLACEMENT: opifex/neural/operators/fno/spherical.py
#
# Spherical Fourier Neural Operator (SFNO) Implementation
# For data defined on spherical domains (climate, planetary science)
#
# This file should be placed at: opifex/neural/operators/fno/spherical.py
# After placement, update opifex/neural/operators/fno/__init__.py to include:
# from .spherical import SphericalFourierNeuralOperator, SphericalHarmonicConvolution

"""
Spherical Fourier Neural Operator (SFNO) implementation.

This module provides FNO variants for data naturally defined on spherical
domains using spherical harmonic decompositions. Ideal for global climate
modeling, atmospheric science, and planetary-scale phenomena.
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array


class SphericalHarmonicConvolution(nnx.Module):
    """
    Spherical harmonic convolution for spherical domains.

    Operates in spherical harmonic space analogous to how standard FNO
    operates in Fourier space, but adapted for spherical geometry.
    """

    @beartype
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lmax: int,  # Maximum spherical harmonic degree
        mmax: int | None = None,  # Maximum azimuthal order
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize spherical harmonic convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            lmax: Maximum spherical harmonic degree (controls resolution)
            mmax: Maximum azimuthal order (if None, uses lmax)
            rngs: Random number generator state
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lmax = lmax
        self.mmax = mmax if mmax is not None else lmax

        # Spherical harmonic weights
        # Shape: (in_channels, out_channels, lmax+1, 2*mmax+1)
        # The last dimension covers m from -mmax to +mmax
        weight_shape = (in_channels, out_channels, lmax + 1, 2 * self.mmax + 1)
        scale = (2 / (in_channels + out_channels)) ** 0.5

        self.weight = nnx.Param(
            (
                jax.random.normal(rngs.params(), weight_shape)
                + 1j * jax.random.normal(rngs.params(), weight_shape)
            )
            * scale
        )

    def _extract_spherical_modes(self, x_sht: Array) -> Array:
        """Extract relevant spherical harmonic modes."""
        # x_sht shape: (batch, channels, lmax_input+1, 2*mmax_input+1)
        # Extract up to our lmax and mmax

        l_end = min(self.lmax + 1, x_sht.shape[2])
        m_start = max(0, x_sht.shape[3] // 2 - self.mmax)
        m_end = min(x_sht.shape[3], x_sht.shape[3] // 2 + self.mmax + 1)

        return x_sht[:, :, :l_end, m_start:m_end]

    def __call__(self, x_sht: Array) -> Array:
        """
        Apply spherical harmonic convolution.

        Args:
            x_sht: Spherical harmonic coefficients (batch, channels, l_modes, m_modes)

        Returns:
            Transformed coefficients (batch, out_channels, l_modes, m_modes)
        """
        # Extract spherical harmonic modes up to configured limits
        x_modes = self._extract_spherical_modes(x_sht)

        # Get weight dimensions and adjust if necessary
        weight = (
            self.weight.value
        )  # Shape: (in_channels, out_channels, lmax+1, 2*mmax+1)

        # Ensure weight modes match input modes
        input_l, input_m = x_modes.shape[-2:]
        weight_l, weight_m = weight.shape[-2:]

        # Handle mode dimension mismatches
        if weight_l > input_l:
            weight = weight[:, :, :input_l, :]
        elif weight_l < input_l:
            # Pad with zeros
            pad_l = input_l - weight_l
            weight = jnp.pad(weight, ((0, 0), (0, 0), (0, pad_l), (0, 0)))

        if weight_m > input_m:
            weight = weight[:, :, :, :input_m]
        elif weight_m < input_m:
            # Pad with zeros
            pad_m = input_m - weight_m
            weight = jnp.pad(weight, ((0, 0), (0, 0), (0, 0), (0, pad_m)))

        # Spherical harmonic multiplication (analogous to spectral convolution)
        # Use standardized einsum pattern for channel contraction
        return jnp.einsum("bi...,ij...->bj...", x_modes, weight)


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
        use_real_sht: bool = False,  # Whether to use real-valued SHT
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize Spherical FNO.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Hidden layer width
            lmax: Maximum spherical harmonic degree
            mmax: Maximum azimuthal order (if None, uses lmax)
            num_layers: Number of SFNO layers
            activation: Activation function
            use_real_sht: Whether to use real spherical harmonics
            rngs: Random number generator state
        """
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax if mmax is not None else lmax
        self.num_layers = num_layers
        self.activation = activation
        self.use_real_sht = use_real_sht

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

    def _spherical_harmonic_transform(self, x: Array) -> Array:
        """
        Compute spherical harmonic transform.

        This is a simplified implementation using 2D FFT as approximation.
        In a full implementation, this would use proper spherical harmonic
        transforms with associated Legendre polynomials.

        Args:
            x: Input on sphere (batch, channels, nlat, nlon)

        Returns:
            SHT coefficients (batch, channels, lmax+1, 2*mmax+1)
        """
        if self.use_real_sht:
            # For real spherical harmonics (simplified)
            return jnp.fft.fft2(x, axes=(-2, -1))
        # Complex spherical harmonics (simplified with 2D FFT)
        # In practice, would use scipy.special.sph_harm or similar
        x_fft = jnp.fft.fft2(x, axes=(-2, -1))

        # Extract relevant modes up to lmax, mmax
        nlat, nlon = x.shape[-2:]
        l_modes = min(self.lmax + 1, nlat)
        _ = min(2 * self.mmax + 1, nlon)  # m_modes not used in current implementation

        # Center the modes around DC component
        m_start = nlon // 2 - self.mmax
        m_end = nlon // 2 + self.mmax + 1

        if m_start >= 0 and m_end <= nlon:
            x_modes = x_fft[:, :, :l_modes, m_start:m_end]
        else:
            # Handle wrapping for negative frequencies
            x_modes = jnp.concatenate(
                [x_fft[:, :, :l_modes, m_start:], x_fft[:, :, :l_modes, :m_end]],
                axis=-1,
            )

        return x_modes

    def _inverse_spherical_harmonic_transform(
        self, x_sht: Array, target_shape: Sequence[int]
    ) -> Array:
        """
        Compute inverse spherical harmonic transform.

        Args:
            x_sht: SHT coefficients
            target_shape: Target spatial shape (nlat, nlon)

        Returns:
            Spatial field on sphere
        """
        if self.use_real_sht:
            # Real ISHT (simplified)
            return jnp.fft.ifft2(x_sht, s=target_shape, axes=(-2, -1)).real
        # Complex ISHT (simplified with 2D IFFT)
        nlat, nlon = target_shape

        # Pad to full frequency grid
        full_spectrum = jnp.zeros((*x_sht.shape[:-2], nlat, nlon), dtype=x_sht.dtype)

        # Place modes in correct positions
        l_modes = x_sht.shape[-2]
        m_modes = x_sht.shape[-1]
        m_center = nlon // 2
        m_start = m_center - m_modes // 2
        m_end = m_start + m_modes

        if m_start >= 0 and m_end <= nlon:
            full_spectrum = full_spectrum.at[:, :, :l_modes, m_start:m_end].set(x_sht)
        else:
            # Handle negative frequency wrapping
            split_point = nlon - m_start
            full_spectrum = full_spectrum.at[:, :, :l_modes, m_start:].set(
                x_sht[..., :split_point]
            )
            full_spectrum = full_spectrum.at[:, :, :l_modes, :m_end].set(
                x_sht[..., split_point:]
            )

        return jnp.fft.ifft2(full_spectrum, axes=(-2, -1)).real

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

            # Spherical harmonic transform
            x_sht = self._spherical_harmonic_transform(x)

            # Spherical convolution
            x_conv = conv(x_sht)

            # Inverse spherical harmonic transform
            x_conv = self._inverse_spherical_harmonic_transform(x_conv, spatial_shape)

            # Skip connection and activation - FIXED: Handle channel dimensions properly
            x_skip_input = jnp.moveaxis(x, 1, -1)  # Move channels to last
            x_skip = skip(x_skip_input)
            x_skip = jnp.moveaxis(
                x_skip, -1, 1
            )  # Move channels back to second position

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
        """
        Compute spherical harmonic power spectrum.

        Args:
            x: Input tensor on sphere

        Returns:
            Power spectrum as function of spherical harmonic degree l
        """
        x_sht = self.get_spherical_modes(x)

        # Compute power for each degree l
        power_spectrum = []
        for l in range(self.lmax + 1):
            if l < x_sht.shape[-2]:
                # Sum over all m modes for this l
                l_power = jnp.sum(jnp.abs(x_sht[:, :, l, :]) ** 2, axis=-1)
                power_spectrum.append(l_power)
            else:
                power_spectrum.append(jnp.zeros_like(x_sht[:, :, 0, 0]))

        return jnp.stack(power_spectrum, axis=-1)


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
        use_real_sht=True,  # Ocean data often real-valued
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
