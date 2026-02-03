"""DeepONet package for operator learning.

This package contains DeepONet architectures and variants for learning operators
from function space to function space mapping.
"""

from .adaptive import AdaptiveDeepONet
from .base import DeepONet
from .enhanced import FourierEnhancedDeepONet
from .multiphysics import MultiPhysicsDeepONet


__all__ = [
    "AdaptiveDeepONet",
    "DeepONet",
    "FourierEnhancedDeepONet",
    "MultiPhysicsDeepONet",
]
