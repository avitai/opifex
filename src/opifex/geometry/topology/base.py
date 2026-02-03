"""Base protocols and interfaces for topological structures.

This module defines fundamental abstractions for discrete topological spaces
used in graph neural networks, simplicial complexes, and topological data analysis.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class TopologicalSpace(Protocol):
    """Protocol for discrete topological spaces.

    Base interface for all topological structures used in geometric deep learning.
    All operations must be JAX-compatible for efficient computation.
    """

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Number of nodes/vertices in the topological space."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Topological dimension of the space."""
        ...

    @abstractmethod
    def adjacency_matrix(self) -> jax.Array:
        """Adjacency matrix representation.

        Returns:
            Sparse adjacency matrix for efficient computation
        """
        ...

    @abstractmethod
    def boundary_operator(self, k: int) -> jax.Array:
        """Boundary operator from k-cells to (k-1)-cells.

        Args:
            k: Dimension of cells

        Returns:
            Boundary matrix ∂_k: C_k → C_{k-1}
        """
        ...


@runtime_checkable
class GraphTopology(TopologicalSpace, Protocol):
    """Protocol for graph structures with nodes and edges.

    Fundamental structure for graph neural networks and message passing.
    """

    @property
    @abstractmethod
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        ...

    @property
    @abstractmethod
    def edge_index(self) -> jax.Array:
        """Edge connectivity as [source, target] pairs.

        Returns:
            Array of shape [2, num_edges] with source and target indices
        """
        ...

    @abstractmethod
    def degree(self, node_idx: int | None = None) -> jax.Array | int:
        """Node degree(s).

        Args:
            node_idx: Specific node index, or None for all nodes

        Returns:
            Degree of specified node or array of all node degrees
        """
        ...

    @abstractmethod
    def neighbors(self, node_idx: int) -> jax.Array:
        """Neighbor nodes of given node.

        Args:
            node_idx: Node index

        Returns:
            Array of neighbor node indices
        """
        ...


@runtime_checkable
class SimplicialComplex(TopologicalSpace, Protocol):
    """Protocol for simplicial complexes.

    Generalization of graphs allowing higher-dimensional simplices
    for topological deep learning and persistent homology.
    """

    @property
    @abstractmethod
    def max_dimension(self) -> int:
        """Maximum dimension of simplices in the complex."""
        ...

    @abstractmethod
    def num_simplices(self, k: int) -> int:
        """Number of k-dimensional simplices.

        Args:
            k: Simplex dimension

        Returns:
            Count of k-simplices
        """
        ...

    @abstractmethod
    def simplices(self, k: int) -> jax.Array:
        """K-dimensional simplices as vertex index arrays.

        Args:
            k: Simplex dimension

        Returns:
            Array where each row contains vertex indices of a k-simplex
        """
        ...

    @abstractmethod
    def betti_numbers(self) -> jax.Array:
        """Betti numbers measuring topological holes.

        Returns:
            Array of Betti numbers β_0, β_1, β_2, ...
        """
        ...


@runtime_checkable
class Hypergraph(TopologicalSpace, Protocol):
    """Protocol for hypergraphs with higher-order relationships.

    Generalizes graphs by allowing edges (hyperedges) to connect
    arbitrary numbers of vertices simultaneously.
    """

    @property
    @abstractmethod
    def num_hyperedges(self) -> int:
        """Number of hyperedges in the hypergraph."""
        ...

    @abstractmethod
    def incidence_matrix(self) -> jax.Array:
        """Vertex-hyperedge incidence matrix.

        Returns:
            Binary matrix where H[i,j] = 1 if vertex i is in hyperedge j
        """
        ...

    @abstractmethod
    def hyperedge_cardinality(self, edge_idx: int | None = None) -> jax.Array:
        """Number of vertices in hyperedge(s).

        Args:
            edge_idx: Specific hyperedge index, or None for all edges

        Returns:
            Cardinality of specified hyperedge or array of all cardinalities
        """
        ...

    @abstractmethod
    def vertex_hyperedges(self, vertex_idx: int) -> jax.Array:
        """Hyperedges containing given vertex.

        Args:
            vertex_idx: Vertex index

        Returns:
            Array of hyperedge indices containing the vertex
        """
        ...


# JAX pytree registration for topological types
import jax


def _topology_tree_flatten(topo):
    """Flatten topology for JAX transforms."""
    return (topo,), None


def _topology_tree_unflatten(aux_data, children):
    """Unflatten topology from JAX transforms."""
    return children[0]


# Register all topological types as JAX pytrees
for topo_type in [TopologicalSpace, GraphTopology, SimplicialComplex, Hypergraph]:
    jax.tree_util.register_pytree_node(
        topo_type, _topology_tree_flatten, _topology_tree_unflatten
    )
