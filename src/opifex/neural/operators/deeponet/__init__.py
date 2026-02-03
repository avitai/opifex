"""DeepONet package for operator learning.

This package contains DeepONet architectures and variants for learning operators
from function space to function space mapping.
"""

from opifex.neural.operators.deeponet.adaptive import AdaptiveDeepONet
from opifex.neural.operators.deeponet.base import DeepONet
from opifex.neural.operators.deeponet.enhanced import FourierEnhancedDeepONet
from opifex.neural.operators.deeponet.multiphysics import MultiPhysicsDeepONet


__all__ = [
    "AdaptiveDeepONet",
    "DeepONet",
    "FourierEnhancedDeepONet",
    "MultiPhysicsDeepONet",
]
