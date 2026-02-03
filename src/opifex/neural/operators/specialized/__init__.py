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
# DISCO Convolution Layers
from .disco import (
    create_disco_decoder,
    create_disco_encoder,
    DiscreteContinuousConv2d,
    DiscreteContinuousConvTranspose2d,
    EquidistantDiscreteContinuousConv2d,
)

# Fourier Continuation Layers
from .fourier_continuation import (
    create_continuation_pipeline,
    FourierBoundaryHandler,
    FourierContinuationExtender,
    PeriodicContinuation,
    SmoothContinuation,
    SymmetricContinuation,
)
from .gino import (
    create_3d_gino,
    create_adaptive_mesh_gino,
    create_cad_gino,
    create_multiscale_gino,
    GeometryAttention,
    GeometryEncoder,
    GeometryInformedNeuralOperator,
    GINOBlock,
)
from .latent import LatentNeuralOperator

# NEW: Multipole Graph Neural Operator
from .mgno import (
    create_molecular_mgno,
    create_nbody_mgno,
    create_plasma_mgno,
    MGNOLayer,
    MultipoleExpansion,
    MultipoleGraphNeuralOperator,
)
from .operator_network import OperatorNetwork

# NEW: Spectral Normalization Layers
from .spectral_normalization import (
    AdaptiveSpectralNorm,
    PowerIteration,
    spectral_norm_summary,
    SpectralLinear,
    SpectralMultiHeadAttention,
    SpectralNorm,
    SpectralNormalizedConv,
)

# NEW: Uncertainty Quantification Neural Operator
from .uno import (
    create_uno,
    UNetBlock,
    UNeuralOperator,
)
from .uqno import (
    BayesianLinear,
    BayesianSpectralConvolution,
    UncertaintyQuantificationNeuralOperator,
    UQNOLayer,
)
from .wavelet import WaveletNeuralOperator


__all__ = [
    "AdaptiveSpectralNorm",
    "BayesianLinear",
    "BayesianSpectralConvolution",
    "DiscreteContinuousConv2d",
    "DiscreteContinuousConvTranspose2d",
    "EquidistantDiscreteContinuousConv2d",
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
    "UNetBlock",
    "UNeuralOperator",
    "UQNOLayer",
    "UncertaintyQuantificationNeuralOperator",
    "WaveletNeuralOperator",
    "create_3d_gino",
    "create_adaptive_mesh_gino",
    "create_cad_gino",
    "create_continuation_pipeline",
    "create_disco_decoder",
    "create_disco_encoder",
    "create_molecular_mgno",
    "create_multiscale_gino",
    "create_nbody_mgno",
    "create_plasma_mgno",
    "create_uno",
    "spectral_norm_summary",
]
