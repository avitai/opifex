"""Graph neural operators for irregular domains.

This package provides graph neural operators for learning on irregular
geometries and unstructured meshes.
"""

from .gno import GraphNeuralOperator, MessagePassingLayer


__all__ = [
    "GraphNeuralOperator",
    "MessagePassingLayer",
]
