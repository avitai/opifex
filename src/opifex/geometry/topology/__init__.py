"""Topological structures for geometric deep learning.

This module provides topological spaces and graph structures for
geometric computations.
"""

from opifex.geometry.topology.base import SimplicialComplex, TopologicalSpace
from opifex.geometry.topology.graphs import GraphTopology


__all__ = [
    "GraphTopology",
    "SimplicialComplex",
    "TopologicalSpace",
]
