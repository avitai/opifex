"""Geometric structures and computations for scientific ML.

This module provides comprehensive geometric tools including:
- Lie groups and algebraic structures
- Differentiable manifolds
- Topological spaces and complexes
- Graph neural networks
- Constructive solid geometry (CSG)
"""

# Import core geometry functionality
# Import algebra structures
# Import pytree utilities to register geometric objects
from opifex.geometry import pytree_utils as _pytree_utils  # Register pytrees on import
from opifex.geometry.algebra import SE3Group, SO3Group

# Import CSG structures explicitly
from opifex.geometry.csg import (
    Circle,
    compute_boundary_normals,
    create_computational_domain_with_molecular_exclusion,
    create_molecular_geometry_from_dft_problem,
    CSGDifference,
    CSGIntersection,
    CSGUnion,
    difference,
    intersection,
    Interval,
    MolecularGeometry,
    PeriodicCell,
    Polygon,
    Rectangle,
    sample_boundary_points,
    Shape2D,
    union,
)

# Import manifold structures
from opifex.geometry.manifolds import Manifold, SphericalManifold, TangentSpace

# Import topology structures
from opifex.geometry.topology import (
    GraphMessagePassing,
    GraphNeuralOperator,
    GraphTopology,
    SimplicialComplex,
    TopologicalSpace,
)


__all__ = [
    "CSGDifference",
    "CSGIntersection",
    "CSGUnion",
    "Circle",
    "GraphMessagePassing",
    "GraphNeuralOperator",
    "GraphTopology",
    "Interval",
    "Manifold",
    "MolecularGeometry",
    "PeriodicCell",
    "Polygon",
    "Rectangle",
    "SE3Group",
    "SO3Group",
    "Shape2D",
    "SimplicialComplex",
    "SphericalManifold",
    "TangentSpace",
    "TopologicalSpace",
    "_pytree_utils",
    "compute_boundary_normals",
    "create_computational_domain_with_molecular_exclusion",
    "create_molecular_geometry_from_dft_problem",
    "difference",
    "intersection",
    "sample_boundary_points",
    "union",
]
