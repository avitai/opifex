# FILE PLACEMENT: opifex/neural/operators/specialized/__init__.py
#
# Updated Specialized Operators Package Init File
# Exports all specialized neural operator variants
#
# This file should be placed at: opifex/neural/operators/specialized/__init__.py
# If this directory doesn't exist, create it first

"""
Specialized Neural Operators Package

This package contains specialized neural operator architectures for
specific problem domains and advanced use cases:

- GINO: Geometry-Informed Neural Operators for complex geometries
- MGNO: Multipole Graph Neural Operators for long-range interactions
- UQNO: Uncertainty Quantification Neural Operators with Bayesian inference
- LNO: Latent Neural Operators with attention-based compression
- WNO: Wavelet Neural Operators for multi-scale analysis
- OperatorNetwork: General operator network architectures
- Spectral Normalization: Stability enhancement through spectral norm control
"""

# Existing specialized operators
# NEW: Geometry-Informed Neural Operator
# DISCO (discrete-continuous) Convolution Layers
from opifex.neural.operators.specialized.disco import (
    build_disco_filter,
    DiscreteContinuousConv2d,
    regular_grid,
)

# Fourier Continuation Layers
from opifex.neural.operators.specialized.fourier_continuation import (
    create_continuation_pipeline,
    FourierBoundaryHandler,
    FourierContinuationExtender,
    PeriodicContinuation,
    SmoothContinuation,
    SymmetricContinuation,
)
from opifex.neural.operators.specialized.gino import (
    create_3d_gino,
    create_adaptive_mesh_gino,
    create_cad_gino,
    create_multiscale_gino,
    GeometryAttention,
    GeometryEncoder,
    GeometryInformedNeuralOperator,
    GINOBlock,
)
from opifex.neural.operators.specialized.latent import LatentNeuralOperator

# NEW: Multipole Graph Neural Operator
from opifex.neural.operators.specialized.mgno import (
    create_molecular_mgno,
    create_nbody_mgno,
    create_plasma_mgno,
    MGNOLayer,
    MultipoleExpansion,
    MultipoleGraphNeuralOperator,
)
from opifex.neural.operators.specialized.operator_network import OperatorNetwork

# NEW: Spectral Normalization Layers
from opifex.neural.operators.specialized.spectral_normalization import (
    AdaptiveSpectralNorm,
    PowerIteration,
    spectral_norm_summary,
    SpectralLinear,
    SpectralMultiHeadAttention,
    SpectralNorm,
    SpectralNormalizedConv,
)

# U-shaped Neural Operator
from opifex.neural.operators.specialized.uno import (
    create_uno,
    UNeuralOperator,
)
from opifex.neural.operators.specialized.uqno import (
    BayesianLinear,
    BayesianSpectralConvolution,
    UncertaintyQuantificationNeuralOperator,
)
from opifex.neural.operators.specialized.wavelet import WaveletNeuralOperator


__all__ = [
    "AdaptiveSpectralNorm",
    "BayesianLinear",
    "BayesianSpectralConvolution",
    "DiscreteContinuousConv2d",
    "FourierBoundaryHandler",
    "FourierContinuationExtender",
    "GINOBlock",
    "GeometryAttention",
    "GeometryEncoder",
    "GeometryInformedNeuralOperator",
    "LatentNeuralOperator",
    "MGNOLayer",
    "MultipoleExpansion",
    "MultipoleGraphNeuralOperator",
    "OperatorNetwork",
    "PeriodicContinuation",
    "PowerIteration",
    "SmoothContinuation",
    "SpectralLinear",
    "SpectralMultiHeadAttention",
    "SpectralNorm",
    "SpectralNormalizedConv",
    "SymmetricContinuation",
    "UNeuralOperator",
    "UncertaintyQuantificationNeuralOperator",
    "WaveletNeuralOperator",
    "build_disco_filter",
    "create_3d_gino",
    "create_adaptive_mesh_gino",
    "create_cad_gino",
    "create_continuation_pipeline",
    "create_molecular_mgno",
    "create_multiscale_gino",
    "create_nbody_mgno",
    "create_plasma_mgno",
    "create_uno",
    "regular_grid",
    "spectral_norm_summary",
]
