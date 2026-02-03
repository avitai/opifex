"""Graph neural operators for irregular domains.

This package provides graph neural operators for learning on irregular
geometries and unstructured meshes.
"""

from .gno import GraphNeuralOperator, MessagePassingLayer
from .utils import graph_to_grid, grid_to_graph_data


__all__ = [
    "GraphNeuralOperator",
    "MessagePassingLayer",
    "graph_to_grid",
    "grid_to_graph_data",
]
