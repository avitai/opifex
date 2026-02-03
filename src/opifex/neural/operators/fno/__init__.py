# FILE PLACEMENT: opifex/neural/operators/fno/__init__.py
#
# Updated FNO Package Init File
# Exports all FNO variants including new implementations
#
# This file should REPLACE the existing: opifex/neural/operators/fno/__init__.py

"""
Fourier Neural Operator (FNO) Package

This package contains all variants of Fourier Neural Operators:
- Standard FNO with spectral convolutions
- Tensorized FNO (TFNO) with parameter factorization
- U-Net FNO (U-FNO) with encoder-decoder architecture
- Spherical FNO (SFNO) for spherical domains
- Local FNO combining global and local operations
- Amortized FNO (AM-FNO) with neural kernel networks
- Multi-scale FNO (MS-FNO) for hierarchical problems
"""

# Core FNO implementation
# NEW: Amortized FNO with neural kernels
from opifex.neural.operators.fno.amortized import (
    AmortizedFourierNeuralOperator,
    AmortizedSpectralConvolution,
    create_high_frequency_amfno,
    create_shock_amfno,
    create_wave_amfno,
    KernelNetwork,
)
from opifex.neural.operators.fno.base import (
    FourierLayer,
    FourierNeuralOperator,
    FourierSpectralConvolution,
)
from opifex.neural.operators.fno.factorized import FactorizedFourierLayer

# NEW: Local FNO with global + local operations
from opifex.neural.operators.fno.local import (
    create_multiphysics_local_fno,
    create_turbulence_local_fno,
    create_wave_local_fno,
    LocalFourierLayer,
    LocalFourierNeuralOperator,
)

# Existing variants
from opifex.neural.operators.fno.multiscale import MultiScaleFourierNeuralOperator

# Spectral neural operators with normalization
from opifex.neural.operators.fno.spectral import (
    create_spectral_neural_operator,
    SpectralNeuralOperator,
)

# NEW: Spherical FNO for global domains
from opifex.neural.operators.fno.spherical import (
    create_climate_sfno,
    create_ocean_sfno,
    create_planetary_sfno,
    create_weather_sfno,
    SphericalFourierNeuralOperator,
    SphericalHarmonicConvolution,
)

# NEW: Tensorized FNO variants
from opifex.neural.operators.fno.tensorized import (
    create_cp_fno,
    create_tt_fno,
    create_tucker_fno,
    TensorizedFourierNeuralOperator,
    TensorizedSpectralConvolution,
)

# NEW: U-Net style FNO
from opifex.neural.operators.fno.ufno import (
    create_deep_ufno,
    create_shallow_ufno,
    create_turbulence_ufno,
    UFNODecoderBlock,
    UFNOEncoderBlock,
    UFourierNeuralOperator,
)


__all__ = [
    "AmortizedFourierNeuralOperator",
    "AmortizedSpectralConvolution",
    "FactorizedFourierLayer",
    "FourierLayer",
    "FourierNeuralOperator",
    "FourierSpectralConvolution",
    "KernelNetwork",
    "LocalFourierLayer",
    "LocalFourierNeuralOperator",
    "MultiScaleFourierNeuralOperator",
    "SpectralNeuralOperator",
    "SphericalFourierNeuralOperator",
    "SphericalHarmonicConvolution",
    "TensorizedFourierNeuralOperator",
    "TensorizedSpectralConvolution",
    "UFNODecoderBlock",
    "UFNOEncoderBlock",
    "UFourierNeuralOperator",
    "create_climate_sfno",
    "create_cp_fno",
    "create_deep_ufno",
    "create_high_frequency_amfno",
    "create_multiphysics_local_fno",
    "create_ocean_sfno",
    "create_planetary_sfno",
    "create_shallow_ufno",
    "create_shock_amfno",
    "create_spectral_neural_operator",
    "create_tt_fno",
    "create_tucker_fno",
    "create_turbulence_local_fno",
    "create_turbulence_ufno",
    "create_wave_amfno",
    "create_wave_local_fno",
    "create_weather_sfno",
]
