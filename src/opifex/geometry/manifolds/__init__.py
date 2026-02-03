"""Manifold structures for geometric deep learning.

This module provides differentiable manifolds and related structures
for geometric computations.
"""

from opifex.geometry.manifolds.base import Manifold, TangentSpace
from opifex.geometry.manifolds.hyperbolic import HyperbolicManifold

# Import neural operators
from opifex.geometry.manifolds.operators import (
    HyperbolicNeuralOperator,
    ManifoldNeuralOperator,
    RiemannianNeuralOperator,
)
from opifex.geometry.manifolds.riemannian import (
    euclidean_metric,
    hyperbolic_metric,
    product_metric,
    RiemannianManifold,
    spherical_metric,
)
from opifex.geometry.manifolds.spherical import SphericalManifold


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
