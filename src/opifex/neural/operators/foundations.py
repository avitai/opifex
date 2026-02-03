"""Neural operator foundations - Modern modular architecture.

This module provides a clean interface to all neural operator components
through their specialized modular packages for optimal maintainability.

Core implementations are located in:
- `fno/`: Fourier Neural Operators and related components
- `deeponet/`: Deep Operator Networks and variants
- `physics/`: Physics-aware attention and constraints
- `graph/`: Graph neural operators for irregular domains
- `specialized/`: Unified interfaces and specialized operators

This module provides access to the complete modular neural operator ecosystem
with clean, canonical imports and modern API design.
"""

# Import all components from modular packages

# Core protocols and base components
from .common.protocols import CallableModule

# Deep Operator Networks (DeepONet) components
from .deeponet.adaptive import AdaptiveDeepONet
from .deeponet.base import (
    DeepONet,
)
from .deeponet.enhanced import FourierEnhancedDeepONet
from .deeponet.multiphysics import MultiPhysicsDeepONet

# Fourier Neural Operators (FNO) components
from .fno.base import (
    FourierLayer,
    FourierNeuralOperator,
    FourierSpectralConvolution,
)
from .fno.factorized import FactorizedFourierLayer
from .fno.multiscale import MultiScaleFourierNeuralOperator

# Graph neural operators
from .graph.gno import (
    GraphNeuralOperator,
    MessagePassingLayer,
)

# Physics-aware components
from .physics.attention import (
    PhysicsAwareAttention,
    PhysicsCrossAttention,
)
from .physics.informed import PhysicsInformedOperator

# Specialized operators and unified interfaces
from .specialized.latent import LatentNeuralOperator
from .specialized.operator_network import OperatorNetwork
from .specialized.uno import create_uno, UNeuralOperator
from .specialized.wavelet import WaveletNeuralOperator


# Export all components with clean, canonical API
__all__ = [
    # Sorted alphabetically for consistency
    "AdaptiveDeepONet",
    "CallableModule",
    "DeepONet",
    "FactorizedFourierLayer",
    "FourierEnhancedDeepONet",
    "FourierLayer",
    "FourierNeuralOperator",
    "FourierSpectralConvolution",
    "GraphNeuralOperator",
    "LatentNeuralOperator",
    "MessagePassingLayer",
    "MultiPhysicsDeepONet",
    "MultiScaleFourierNeuralOperator",
    "OperatorNetwork",
    "PhysicsAwareAttention",
    "PhysicsCrossAttention",
    "PhysicsInformedOperator",
    "UNeuralOperator",
    "WaveletNeuralOperator",
    "create_uno",
]
