"""Graph neural operators for irregular domains.

This package provides graph neural operators for learning on irregular
geometries and unstructured meshes.
"""

from opifex.neural.operators.graph.gno import GraphNeuralOperator, MessagePassingLayer
from opifex.neural.operators.graph.utils import graph_to_grid, grid_to_graph_data


__all__ = [
    "GraphNeuralOperator",
    "MessagePassingLayer",
    "graph_to_grid",
    "grid_to_graph_data",
]
