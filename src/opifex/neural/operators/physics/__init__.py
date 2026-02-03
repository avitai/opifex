"""Physics-aware neural operator components.

This package provides physics-informed neural operators and attention mechanisms
that incorporate conservation laws and physics constraints.
"""

from .attention import PhysicsAwareAttention, PhysicsCrossAttention
from .informed import PhysicsInformedOperator


__all__ = [
    "PhysicsAwareAttention",
    "PhysicsCrossAttention",
    "PhysicsInformedOperator",
]
