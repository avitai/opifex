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
from opifex.neural.operators.common.protocols import CallableModule

# Deep Operator Networks (DeepONet) components
from opifex.neural.operators.deeponet.adaptive import AdaptiveDeepONet
from opifex.neural.operators.deeponet.base import (
    DeepONet,
)
from opifex.neural.operators.deeponet.enhanced import FourierEnhancedDeepONet
from opifex.neural.operators.deeponet.multiphysics import MultiPhysicsDeepONet

# Fourier Neural Operators (FNO) components
from opifex.neural.operators.fno.base import (
    FourierLayer,
    FourierNeuralOperator,
    FourierSpectralConvolution,
)
from opifex.neural.operators.fno.factorized import FactorizedFourierLayer
from opifex.neural.operators.fno.multiscale import MultiScaleFourierNeuralOperator

# Graph neural operators
from opifex.neural.operators.graph.gno import (
    GraphNeuralOperator,
    MessagePassingLayer,
)

# Physics-aware components
from opifex.neural.operators.physics.attention import (
    PhysicsAwareAttention,
    PhysicsCrossAttention,
)
from opifex.neural.operators.physics.informed import PhysicsInformedOperator

# Specialized operators and unified interfaces
from opifex.neural.operators.specialized.latent import LatentNeuralOperator
from opifex.neural.operators.specialized.operator_network import OperatorNetwork
from opifex.neural.operators.specialized.uno import create_uno, UNeuralOperator
from opifex.neural.operators.specialized.wavelet import WaveletNeuralOperator


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
