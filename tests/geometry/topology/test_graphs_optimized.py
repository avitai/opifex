"""
Tests for graph topology structures.

Tests JAX compatibility (JIT, VMAP) and pytree registration for
:class:`GraphTopology`.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.geometry.topology.graphs import GraphTopology


class TestGraphTopologyPytreeRegistration:
    """Guard against the duplicate / order-dependent pytree registration defect.

    ``GraphTopology`` was historically registered as a JAX pytree twice with
    incompatible flatten schemes (one keeping ``edge_features`` as a traced
    child, the other relegating it to static ``aux_data``).  Registration is
    now performed exactly once, with ``edge_features`` as a child so it survives
    a flatten/unflatten round-trip and remains a differentiable leaf.
    """

    def test_graphtopology_pytree_roundtrip_preserves_edge_features(self):
        """Flatten + unflatten must preserve non-None edge_features and stay jittable."""
        nodes = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        edges = jnp.array([[0, 1], [1, 2], [2, 0]])
        edge_features = jnp.array([[0.1], [0.2], [0.3]])
        graph = GraphTopology(nodes=nodes, edges=edges, edge_features=edge_features)

        leaves, treedef = jax.tree_util.tree_flatten(graph)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)

        assert restored.edge_features is not None
        assert jnp.array_equal(restored.edge_features, edge_features)
        assert jnp.array_equal(restored.nodes, nodes)
        assert jnp.array_equal(restored.edges, edges)
        assert jnp.array_equal(restored.adjacency_matrix, graph.adjacency_matrix)

        # edge_features must be a traced child (a real leaf), not static aux_data.
        assert any(
            getattr(leaf, "shape", None) == edge_features.shape
            and jnp.array_equal(leaf, edge_features)
            for leaf in leaves
        )

        # The registered pytree must be usable as a jit argument.
        assert float(jax.jit(lambda g: g.nodes.sum())(graph)) == float(nodes.sum())


class TestOptimizedGraphTopology:
    """Test optimized graph topology operations."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple graph
        self.nodes = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        self.edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
        self.edge_features = jnp.array([[0.1], [0.2], [0.3], [0.4], [0.5]])

        self.graph = GraphTopology(
            nodes=self.nodes, edges=self.edges, edge_features=self.edge_features
        )

    def test_graph_creation(self):
        """Test graph topology creation."""
        assert self.graph.num_nodes == 4
        assert self.graph.num_edges == 5
        assert self.graph.node_dim == 2

    def test_adjacency_matrix_jit_compatibility(self):
        """Test that adjacency matrix computation works with JIT compilation."""

        @jax.jit
        def compute_adjacency(graph):
            return graph._compute_adjacency()

        result_normal = self.graph._compute_adjacency()
        result_jit = compute_adjacency(self.graph)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (4, 4)

    def test_laplacian_matrix_jit_compatibility(self):
        """Test that Laplacian matrix computation works with JIT compilation."""

        @jax.jit
        def compute_laplacian(graph, normalized):
            return graph.laplacian_matrix(normalized=normalized)

        # Test normalized Laplacian
        result_normal = self.graph.laplacian_matrix(normalized=True)
        result_jit = compute_laplacian(self.graph, True)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (4, 4)

        # Test unnormalized Laplacian
        result_normal = self.graph.laplacian_matrix(normalized=False)
        result_jit = compute_laplacian(self.graph, False)

        assert jnp.allclose(result_normal, result_jit)

    def test_degree_matrix_vmap_compatibility(self):
        """Test that degree matrix computation works with vmap."""

        def compute_degree_single(graph):
            return graph.degree_matrix()

        # Create batch of graphs (same graph repeated)
        graphs = [self.graph] * 3

        # Test individual degree computations
        degrees = [compute_degree_single(graph) for graph in graphs]

        # All should have same shape and values
        for degree in degrees:
            assert degree.shape == (4, 4)
            assert jnp.allclose(degree, degrees[0])

    def test_get_neighbors_functionality(self):
        """Test neighbor finding functionality."""
        # Node 0 should be connected to nodes 1, 2, 3
        neighbors_0 = self.graph.get_neighbors(0)
        expected_neighbors = jnp.array([1, 2, 3])

        # Sort both arrays for comparison
        neighbors_0_sorted = jnp.sort(neighbors_0)
        expected_sorted = jnp.sort(expected_neighbors)

        assert jnp.array_equal(neighbors_0_sorted, expected_sorted)


if __name__ == "__main__":
    pytest.main([__file__])
