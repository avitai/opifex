"""Core spectral operations for the Opifex framework.

This package provides fundamental spectral analysis utilities that are used
throughout the Opifex framework. It contains the core FFT operations, validation
utilities, and spectral analysis tools that serve as the foundation for all
other spectral functionality.

Key Components:
- fft_operations: Standardized FFT/IFFT operations with consistent API
- spectral_utils: Common spectral analysis functions and utilities
- validation: Input validation and error handling for spectral operations
"""

from opifex.core.spectral.fft_operations import (
    fft_frequency_grid,
    spectral_derivative,
    spectral_filter,
    standardized_fft,
    standardized_ifft,
)
from opifex.core.spectral.spectral_utils import (
    energy_spectrum,
    power_spectral_density,
    spectral_energy,
    wavenumber_grid,
)
from opifex.core.spectral.validation import (
    validate_fft_shape,
    validate_spatial_dims,
    validate_spectral_input,
)


__all__ = [
    # Sorted alphabetically
    "energy_spectrum",
    "fft_frequency_grid",
    "power_spectral_density",
    "spectral_derivative",
    "spectral_energy",
    "spectral_filter",
    "standardized_fft",
    "standardized_ifft",
    "validate_fft_shape",
    "validate_spatial_dims",
    "validate_spectral_input",
    "wavenumber_grid",
]
