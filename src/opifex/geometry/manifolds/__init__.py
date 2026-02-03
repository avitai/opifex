"""Manifold structures for geometric deep learning.

This module provides differentiable manifolds and related structures
for geometric computations.
"""

from .base import Manifold, TangentSpace
from .hyperbolic import HyperbolicManifold

# Import neural operators
from .operators import (
    HyperbolicNeuralOperator,
    ManifoldNeuralOperator,
    RiemannianNeuralOperator,
)
from .riemannian import (
    euclidean_metric,
    hyperbolic_metric,
    product_metric,
    RiemannianManifold,
    spherical_metric,
)
from .spherical import SphericalManifold


__all__ = [
    "HyperbolicManifold",
    "HyperbolicNeuralOperator",
    "Manifold",
    "ManifoldNeuralOperator",
    "RiemannianManifold",
    "RiemannianNeuralOperator",
    "SphericalManifold",
    "TangentSpace",
    "euclidean_metric",
    "hyperbolic_metric",
    "product_metric",
    "spherical_metric",
]
