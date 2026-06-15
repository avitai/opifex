"""Neural network components for scientific machine learning.

This package provides neural network architectures and components optimized for
scientific computing: standard MLPs (:mod:`opifex.neural.base`), neural operators
(:mod:`opifex.neural.operators`), equivariant building blocks
(:mod:`opifex.neural.equivariant`) and the atomistic machine-learning interatomic
potentials (:mod:`opifex.neural.atomistic`).
"""

from opifex.neural import activations, base


__all__ = ["activations", "base"]
