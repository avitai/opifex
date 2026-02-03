# FILE PLACEMENT: opifex/neural/operators/__init__.py
#
# Updated Neural Operators Package Init File
# Comprehensive API for all neural operator variants
#
# This file should REPLACE the existing: opifex/neural/operators/__init__.py
# It provides a complete API for all neural operator variants including the new ones

"""
Opifex Neural Operators: Comprehensive Operator Learning Library

This module provides the most complete collection of neural operators for
scientific machine learning, including all major variants from the neuraloperator
repository and advanced architectures.

The library includes:

- Fourier Neural Operators (FNO, TFNO, U-FNO, SFNO, Local FNO, AM-FNO)
- Deep Operator Networks (DeepONet and variants)
- Specialized operators (GINO, MGNO, UQNO, LNO, WNO, GNO)
- Physics-informed operators (PINO)
- Graph-based operators
- Uncertainty quantification operators

All operators are built with JAX/FLAX NNX for high performance and support
automatic differentiation, just-in-time compilation, and multi-device parallelization.
"""

from collections.abc import Sequence
from typing import Any

from opifex.neural.operators.deeponet.adaptive import AdaptiveDeepONet

# DeepONet implementations
from opifex.neural.operators.deeponet.base import DeepONet

# DeepONet variants
from opifex.neural.operators.deeponet.enhanced import FourierEnhancedDeepONet
from opifex.neural.operators.deeponet.multiphysics import MultiPhysicsDeepONet

# Amortized FNO - Neural kernel networks
from opifex.neural.operators.fno.amortized import (
    AmortizedFourierNeuralOperator,
    AmortizedSpectralConvolution,
    create_high_frequency_amfno,
    create_shock_amfno,
    create_wave_amfno,
    KernelNetwork,
)

# =============================================================================
# EXISTING OPERATORS (Your current implementations)
# =============================================================================
# Core FNO implementations
from opifex.neural.operators.fno.base import FourierLayer, FourierNeuralOperator

# Spectral neural operators available from fno.spectral submodule
from opifex.neural.operators.fno.factorized import FactorizedFourierLayer

# Local FNO - Global + local operations
from opifex.neural.operators.fno.local import (
    create_multiphysics_local_fno,
    create_turbulence_local_fno,
    create_wave_local_fno,
    LocalFourierLayer,
    LocalFourierNeuralOperator,
)

# Existing FNO variants
from opifex.neural.operators.fno.multiscale import MultiScaleFourierNeuralOperator

# Spherical FNO - For spherical domains
from opifex.neural.operators.fno.spherical import (
    create_climate_sfno,
    create_ocean_sfno,
    create_planetary_sfno,
    create_weather_sfno,
    SphericalFourierNeuralOperator,
    SphericalHarmonicConvolution,
)

# =============================================================================
# NEW OPERATORS (From neuraloperator repository)
# =============================================================================
# Tensorized FNO - Parameter-efficient factorized FNO
from opifex.neural.operators.fno.tensorized import (
    create_cp_fno,
    create_tt_fno,
    create_tucker_fno,
    TensorizedFourierNeuralOperator,
    TensorizedSpectralConvolution,
)

# U-FNO - U-Net style encoder-decoder FNO
from opifex.neural.operators.fno.ufno import (
    create_deep_ufno,
    create_shallow_ufno,
    create_turbulence_ufno,
    UFNODecoderBlock,
    UFNOEncoderBlock,
    UFourierNeuralOperator,
)

# Graph operators
from opifex.neural.operators.graph.gno import GraphNeuralOperator, MessagePassingLayer
from opifex.neural.operators.physics.attention import (
    PhysicsAwareAttention,
    PhysicsCrossAttention,
)

# Physics-informed operators
from opifex.neural.operators.physics.informed import PhysicsInformedOperator

# GINO - Geometry-informed operators
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

# MGNO - Multipole graph operators
from opifex.neural.operators.specialized.mgno import (
    create_molecular_mgno,
    create_nbody_mgno,
    create_plasma_mgno,
    MGNOLayer,
    MultipoleExpansion,
    MultipoleGraphNeuralOperator,
)
from opifex.neural.operators.specialized.operator_network import OperatorNetwork

# UQNO - Uncertainty quantification operators
from opifex.neural.operators.specialized.uqno import (
    BayesianLinear,
    BayesianSpectralConvolution,
    create_bayesian_inverse_uqno,
    create_robust_design_uqno,
    create_safety_critical_uqno,
    UncertaintyQuantificationNeuralOperator,
    UQNOLayer,
)

# Specialized existing operators
from opifex.neural.operators.specialized.wavelet import WaveletNeuralOperator


# =============================================================================
# OPERATOR REGISTRY AND FACTORY FUNCTIONS
# =============================================================================

# Complete operator registry for easy selection
OPERATOR_REGISTRY: dict[str, type] = {
    # Traditional operators
    "FNO": FourierNeuralOperator,
    "DeepONet": DeepONet,
    "PINO": PhysicsInformedOperator,
    # FNO variants
    "TFNO": TensorizedFourierNeuralOperator,
    "UFNO": UFourierNeuralOperator,
    "SFNO": SphericalFourierNeuralOperator,
    "LocalFNO": LocalFourierNeuralOperator,
    "AM-FNO": AmortizedFourierNeuralOperator,
    "MS-FNO": MultiScaleFourierNeuralOperator,
    # DeepONet variants
    "FourierDeepONet": FourierEnhancedDeepONet,
    "AdaptiveDeepONet": AdaptiveDeepONet,
    "MultiPhysicsDeepONet": MultiPhysicsDeepONet,
    # Specialized operators
    "GINO": GeometryInformedNeuralOperator,
    "MGNO": MultipoleGraphNeuralOperator,
    "UQNO": UncertaintyQuantificationNeuralOperator,
    "LNO": LatentNeuralOperator,
    "WNO": WaveletNeuralOperator,
    "GNO": GraphNeuralOperator,
    "OperatorNet": OperatorNetwork,
}

# Application-specific operator recommendations
APPLICATION_RECOMMENDATIONS: dict[str, dict[str, str | Sequence[str]]] = {
    "turbulent_flow": {
        "primary": "UFNO",
        "alternatives": ["LocalFNO", "LNO"],
        "reason": "Multi-scale encoder-decoder for turbulent structures",
    },
    "global_climate": {
        "primary": "SFNO",
        "alternatives": ["FNO"],
        "reason": "Spherical harmonics for global atmospheric modeling",
    },
    "molecular_dynamics": {
        "primary": "MGNO",
        "alternatives": ["GNO"],
        "reason": "Multipole expansion for long-range molecular interactions",
    },
    "cad_geometry": {
        "primary": "GINO",
        "alternatives": ["GNO"],
        "reason": "Geometry-aware processing for complex CAD shapes",
    },
    "safety_critical": {
        "primary": "UQNO",
        "alternatives": ["FNO"],
        "reason": "Uncertainty quantification for safety-critical decisions",
    },
    "high_frequency": {
        "primary": "AM-FNO",
        "alternatives": ["LocalFNO"],
        "reason": "Neural kernels for high-frequency phenomena",
    },
    "parameter_efficient": {
        "primary": "TFNO",
        "alternatives": ["LNO"],
        "reason": "Tensor factorization for memory efficiency",
    },
    "irregular_mesh": {
        "primary": "GNO",
        "alternatives": ["GINO"],
        "reason": "Graph-based processing for irregular grids",
    },
    "wave_propagation": {
        "primary": "LocalFNO",
        "alternatives": ["AM-FNO", "FNO"],
        "reason": "Combined local and global operations for waves",
    },
    "inverse_problems": {
        "primary": "UQNO",
        "alternatives": ["PINO"],
        "reason": "Bayesian uncertainty for ill-posed inverse problems",
    },
}


def create_operator(operator_type: str, **kwargs: Any) -> Any:
    """
    Factory function to create any operator by name.

    Args:
        operator_type: Type of operator to create
        **kwargs: Arguments for operator initialization

    Returns:
        Initialized operator instance

    Raises:
        ValueError: If operator_type is not recognized

    Example:
        >>> # Create a Tensorized FNO
        >>> tfno = create_operator("TFNO",
        ...                       in_channels=3, out_channels=1,
        ...                       hidden_channels=64, modes=(16, 16),
        ...                       factorization="tucker", rank=0.1,
        ...                       rngs=rngs)
    """
    if operator_type not in OPERATOR_REGISTRY:
        available = ", ".join(sorted(OPERATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown operator type '{operator_type}'. Available: {available}"
        )

    return OPERATOR_REGISTRY[operator_type](**kwargs)


def recommend_operator(application: str) -> dict[str, Any]:
    """
    Recommend the best operator for a specific application.

    Args:
        application: Application domain

    Returns:
        Dictionary with recommendations

    Example:
        >>> rec = recommend_operator("turbulent_flow")
        >>> print(f"Recommended: {rec['primary']}")
        >>> print(f"Reason: {rec['reason']}")
    """
    if application in APPLICATION_RECOMMENDATIONS:
        return APPLICATION_RECOMMENDATIONS[application].copy()
    available = ", ".join(sorted(APPLICATION_RECOMMENDATIONS.keys()))
    return {
        "primary": "FNO",
        "alternatives": ["DeepONet"],
        "reason": f"Unknown application '{application}'. Available: {available}",
    }


def list_operators(category: str | None = None) -> dict[str, Sequence[str]]:
    """
    List available operators by category.

    Args:
        category: Optional category filter

    Returns:
        Dictionary of operators by category
    """
    categories: dict[str, Sequence[str]] = {
        "fourier_operators": ["FNO", "TFNO", "UFNO", "SFNO", "LocalFNO", "AM-FNO"],
        "deeponet_family": ["DeepONet", "FourierDeepONet", "AdaptiveDeepONet"],
        "graph_operators": ["GNO", "MGNO"],
        "uncertainty_aware": ["UQNO"],
        "geometry_aware": ["GINO", "GNO", "MGNO"],
        "parameter_efficient": ["TFNO", "LNO"],
    }

    if category:
        return {category: categories.get(category, [])}
    return categories


def get_operator_info(operator_type: str) -> dict[str, Any]:
    """
    Get detailed information about a specific operator.

    Args:
        operator_type: Type of operator

    Returns:
        Dictionary with operator information
    """
    info_map = {
        "FNO": {
            "description": (
                "Standard Fourier Neural Operator with spectral convolutions"
            ),
            "best_for": ["regular grids", "periodic problems", "smooth solutions"],
            "parameters": "Medium",
            "computational_cost": "Medium",
        },
        "TFNO": {
            "description": (
                "Tensorized FNO with 10-20x parameter reduction via factorization"
            ),
            "best_for": ["memory-constrained", "large-scale", "parameter efficiency"],
            "parameters": "Low",
            "computational_cost": "Medium",
        },
        "UFNO": {
            "description": (
                "U-Net style FNO with encoder-decoder for multi-scale problems"
            ),
            "best_for": ["turbulent flow", "multi-scale", "fine details"],
            "parameters": "High",
            "computational_cost": "High",
        },
        "SFNO": {
            "description": (
                "Spherical FNO using spherical harmonics for global domains"
            ),
            "best_for": ["climate modeling", "global problems", "spherical geometry"],
            "parameters": "Medium",
            "computational_cost": "Medium",
        },
        "LocalFNO": {
            "description": ("Combines global Fourier and local convolution operations"),
            "best_for": ["wave propagation", "local + global features", "turbulence"],
            "parameters": "Medium",
            "computational_cost": "Medium",
        },
        "AM-FNO": {
            "description": (
                "Amortized FNO with neural kernel networks for high frequencies"
            ),
            "best_for": ["high frequency", "shocks", "discontinuities"],
            "parameters": "Medium",
            "computational_cost": "High",
        },
        "GINO": {
            "description": ("Geometry-Informed Neural Operator for complex geometries"),
            "best_for": ["CAD geometry", "irregular domains", "complex boundaries"],
            "parameters": "Medium",
            "computational_cost": "Medium",
        },
        "MGNO": {
            "description": (
                "Multipole Graph Neural Operator for long-range interactions"
            ),
            "best_for": ["molecular dynamics", "N-body", "electrostatics"],
            "parameters": "High",
            "computational_cost": "High",
        },
        "UQNO": {
            "description": (
                "Uncertainty Quantification Neural Operator with Bayesian inference"
            ),
            "best_for": ["safety critical", "uncertainty bounds", "robust design"],
            "parameters": "High",
            "computational_cost": "High",
        },
        "DeepONet": {
            "description": ("Deep Operator Network with branch-trunk architecture"),
            "best_for": ["sensor data", "irregular observations", "general operators"],
            "parameters": "Medium",
            "computational_cost": "Low",
        },
    }

    return info_map.get(
        operator_type,
        {
            "description": "Information not available",
            "best_for": ["general use"],
            "parameters": "Unknown",
            "computational_cost": "Unknown",
        },
    )


# =============================================================================
# COMPREHENSIVE EXPORTS
# =============================================================================

__all__ = [
    "APPLICATION_RECOMMENDATIONS",
    "OPERATOR_REGISTRY",
    # Specialized components
    "AdaptiveDeepONet",
    "AmortizedFourierNeuralOperator",
    "AmortizedSpectralConvolution",
    "BayesianLinear",
    "BayesianSpectralConvolution",
    # DeepONet variants
    "DeepONet",
    "FactorizedFourierLayer",
    # Core FNO variants
    "FourierEnhancedDeepONet",
    "FourierLayer",
    "FourierNeuralOperator",
    "GINOBlock",
    "GeometryAttention",
    "GeometryEncoder",
    "GeometryInformedNeuralOperator",
    "GraphNeuralOperator",
    "KernelNetwork",
    "LatentNeuralOperator",
    "LocalFourierLayer",
    "LocalFourierNeuralOperator",
    "MGNOLayer",
    "MessagePassingLayer",
    "MultiPhysicsDeepONet",
    "MultiScaleFourierNeuralOperator",
    "MultipoleExpansion",
    "MultipoleGraphNeuralOperator",
    "OperatorNetwork",
    "PhysicsAwareAttention",
    "PhysicsCrossAttention",
    "PhysicsInformedOperator",
    # FNO components
    "SphericalFourierNeuralOperator",
    "SphericalHarmonicConvolution",
    "TensorizedFourierNeuralOperator",
    "TensorizedSpectralConvolution",
    "UFNODecoderBlock",
    "UFNOEncoderBlock",
    "UFourierNeuralOperator",
    "UQNOLayer",
    "UncertaintyQuantificationNeuralOperator",
    "WaveletNeuralOperator",
    # Utility constructors
    "create_3d_gino",
    "create_adaptive_mesh_gino",
    "create_bayesian_inverse_uqno",
    "create_cad_gino",
    "create_climate_sfno",
    "create_cp_fno",
    "create_deep_ufno",
    "create_high_frequency_amfno",
    "create_molecular_mgno",
    "create_multiphysics_local_fno",
    "create_multiscale_gino",
    "create_nbody_mgno",
    "create_ocean_sfno",
    # Factory and utility functions
    "create_operator",
    "create_planetary_sfno",
    "create_plasma_mgno",
    "create_robust_design_uqno",
    "create_safety_critical_uqno",
    "create_shallow_ufno",
    "create_shock_amfno",
    "create_tt_fno",
    "create_tucker_fno",
    "create_turbulence_local_fno",
    "create_turbulence_ufno",
    "create_wave_amfno",
    "create_wave_local_fno",
    "create_weather_sfno",
    "get_operator_info",
    "list_operators",
    "recommend_operator",
]
