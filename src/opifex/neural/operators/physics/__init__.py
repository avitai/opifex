"""Physics-aware neural operator components.

This package provides physics-informed neural operators and attention mechanisms
that incorporate conservation laws and physics constraints.
"""

from opifex.neural.operators.physics.attention import (
    PhysicsAwareAttention,
    PhysicsCrossAttention,
)
from opifex.neural.operators.physics.informed import PhysicsInformedOperator


__all__ = [
    "PhysicsAwareAttention",
    "PhysicsCrossAttention",
    "PhysicsInformedOperator",
]
