"""Enhanced Constructive Solid Geometry (CSG) operations for Opifex.

Sub-modules:

* :mod:`opifex.geometry.csg.types` — Point / Points jaxtyping aliases,
  the :class:`Shape2D` protocol, and the :class:`_EnhancedShapeBase` mixin.
* :mod:`opifex.geometry.csg.primitives` — Base 2D / 1D primitives:
  :class:`Interval`, :class:`Rectangle`, :class:`Circle`, :class:`Polygon`.
* :mod:`opifex.geometry.csg.operations` — :class:`CSGUnion`,
  :class:`CSGIntersection`, :class:`CSGDifference`, the :func:`union`,
  :func:`intersection`, :func:`difference` constructors, boundary helpers
  (:func:`compute_boundary_normals`, :func:`sample_boundary_points`), and
  the smooth-blended :func:`smooth_union`.
* :mod:`opifex.geometry.csg.molecular` — :class:`MolecularGeometry`,
  :class:`PeriodicCell`, and DFT/exclusion-zone helpers.
* :mod:`opifex.geometry.csg.utils` — Monte-Carlo :func:`compute_shape_area`
  and :func:`validate_implementation` self-test.
"""

from opifex.geometry.csg.molecular import (
    create_computational_domain_with_molecular_exclusion,
    create_molecular_geometry_from_dft_problem,
    MolecularGeometry,
    PeriodicCell,
)
from opifex.geometry.csg.operations import (
    compute_boundary_normals,
    CSGDifference,
    CSGIntersection,
    CSGUnion,
    difference,
    intersection,
    sample_boundary_points,
    smooth_union,
    union,
)
from opifex.geometry.csg.primitives import Circle, Interval, Polygon, Rectangle
from opifex.geometry.csg.types import (
    Point1D,
    Point2D,
    Point3D,
    Points1D,
    Points2D,
    Points3D,
    Shape2D,
)
from opifex.geometry.csg.utils import compute_shape_area, validate_implementation


__all__ = [
    "CSGDifference",
    "CSGIntersection",
    "CSGUnion",
    "Circle",
    "Interval",
    "MolecularGeometry",
    "PeriodicCell",
    "Point1D",
    "Point2D",
    "Point3D",
    "Points1D",
    "Points2D",
    "Points3D",
    "Polygon",
    "Rectangle",
    "Shape2D",
    "compute_boundary_normals",
    "compute_shape_area",
    "create_computational_domain_with_molecular_exclusion",
    "create_molecular_geometry_from_dft_problem",
    "difference",
    "intersection",
    "sample_boundary_points",
    "smooth_union",
    "union",
    "validate_implementation",
]
