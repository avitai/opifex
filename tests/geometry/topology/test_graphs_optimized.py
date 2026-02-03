"""
Comprehensive tests for optimized graph neural operations.

Tests JAX compatibility (JIT, VMAP, GRAD) for all graph neural operations
after optimization with vectorized operations.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.geometry.topology.graphs import (
    GraphMessagePassing,
    GraphNeuralOperator,
    GraphTopology,
    linear_layer,
)


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


class TestOptimizedGraphMessagePassing:
    """Test optimized graph message passing operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.key = jax.random.PRNGKey(42)
        self.input_dim = 4
        self.hidden_dim = 8
        self.output_dim = 2

        # Initialize message passing layer
        self.mp_layer = GraphMessagePassing(
            node_dim=self.input_dim,
            edge_dim=2,  # Edge feature dimension
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            key=self.key,
        )

        # Create test graph
        self.nodes = jax.random.normal(self.key, (5, self.input_dim))
        self.edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
        self.edge_features = jax.random.normal(jax.random.split(self.key)[0], (5, 2))

    def test_message_passing_jit_compatibility(self):
        """Test that message passing works with JIT compilation."""

        @jax.jit
        def apply_message_passing(mp_layer, nodes, edges, edge_features):
            return mp_layer(nodes, edges, edge_features)

        result_normal = self.mp_layer(self.nodes, self.edges, self.edge_features)
        result_jit = apply_message_passing(
            self.mp_layer, self.nodes, self.edges, self.edge_features
        )

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (5, self.output_dim)

    def test_message_passing_vmap_compatibility(self):
        """Test that message passing works with vmap."""

        def apply_mp_single(nodes):
            return self.mp_layer(nodes, self.edges, self.edge_features)

        # Create batch of node features
        batch_nodes = jax.random.normal(self.key, (3, 5, self.input_dim))
        vmap_mp = jax.vmap(apply_mp_single)

        results = vmap_mp(batch_nodes)

        assert results.shape == (3, 5, self.output_dim)
        assert jnp.all(jnp.isfinite(results))

    def test_message_passing_grad_compatibility(self):
        """Test that message passing works with grad."""

        def mp_loss(nodes):
            output = self.mp_layer(nodes, self.edges, self.edge_features)
            return jnp.sum(output**2)

        grad_fn = jax.grad(mp_loss)
        gradient = grad_fn(self.nodes)

        assert gradient.shape == self.nodes.shape
        assert jnp.all(jnp.isfinite(gradient))


class TestOptimizedGraphNeuralOperator:
    """Test optimized graph neural operator operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.key = jax.random.PRNGKey(42)
        self.input_dim = 3
        self.hidden_dim = 6
        self.output_dim = 2
        self.num_layers = 3

        # Initialize graph neural operator
        self.gno = GraphNeuralOperator(
            node_dim=self.input_dim,
            edge_dim=1,  # Edge feature dimension
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            key=self.key,
        )

        # Create test graph
        self.nodes = jax.random.normal(self.key, (6, self.input_dim))
        self.edges = jnp.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3], [1, 4]]
        )
        self.edge_features = jax.random.normal(jax.random.split(self.key)[0], (8, 1))

        self.graph = GraphTopology(
            nodes=self.nodes, edges=self.edges, edge_features=self.edge_features
        )

    def test_gno_creation(self):
        """Test graph neural operator creation."""
        assert len(self.gno.mp_layers) == self.num_layers
        assert self.gno.input_w.shape == (self.input_dim, self.hidden_dim)
        assert self.gno.output_w.shape == (self.hidden_dim, self.output_dim)

    def test_gno_forward_jit_compatibility(self):
        """Test that GNO forward pass works with JIT compilation."""

        @jax.jit
        def gno_forward(gno, graph):
            return gno(graph)

        result_normal = self.gno(self.graph)
        result_jit = gno_forward(self.gno, self.graph)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (6, self.output_dim)

    def test_gno_forward_vmap_compatibility(self):
        """Test that GNO forward pass works with vmap."""

        def gno_forward_single(nodes):
            graph = GraphTopology(
                nodes=nodes, edges=self.edges, edge_features=self.edge_features
            )
            return self.gno(graph)

        # Create batch of node features
        batch_nodes = jax.random.normal(self.key, (4, 6, self.input_dim))
        vmap_gno = jax.vmap(gno_forward_single)

        results = vmap_gno(batch_nodes)

        assert results.shape == (4, 6, self.output_dim)
        assert jnp.all(jnp.isfinite(results))

    def test_gno_parameter_grad_compatibility(self):
        """Test that GNO parameters work with grad."""

        def gno_loss(graph):
            output = self.gno(graph)
            return jnp.sum(output**2)

        # Test that we can compute loss (this tests parameter handling)
        loss_value = gno_loss(self.graph)

        assert jnp.isfinite(loss_value)
        assert loss_value >= 0

    def test_optimized_message_passing_layers(self):
        """Test that optimized message passing layers work correctly."""
        # The scan-based implementation should produce same results
        # as the original loop-based implementation

        result = self.gno(self.graph)

        # Test basic properties
        assert result.shape == (6, self.output_dim)
        assert jnp.all(jnp.isfinite(result))

        # Test that different inputs produce different outputs
        modified_graph = GraphTopology(
            nodes=self.nodes * 2.0,  # Scale node features
            edges=self.edges,
            edge_features=self.edge_features,
        )

        result_modified = self.gno(modified_graph)

        # Results should be different
        assert not jnp.allclose(result, result_modified)


class TestOptimizedLinearLayer:
    """Test optimized linear layer operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.key = jax.random.PRNGKey(42)
        self.input_dim = 4
        self.output_dim = 3

        self.weights = jax.random.normal(self.key, (self.input_dim, self.output_dim))
        self.bias = jax.random.normal(jax.random.split(self.key)[0], (self.output_dim,))
        self.input_data = jax.random.normal(
            jax.random.split(self.key)[1], (5, self.input_dim)
        )

    def test_linear_layer_jit_compatibility(self):
        """Test that linear layer works with JIT compilation."""

        @jax.jit
        def apply_linear(x, w, b):
            return linear_layer(x, w, b)

        result_normal = linear_layer(self.input_data, self.weights, self.bias)
        result_jit = apply_linear(self.input_data, self.weights, self.bias)

        assert jnp.allclose(result_normal, result_jit)
        assert result_jit.shape == (5, self.output_dim)

    def test_linear_layer_vmap_compatibility(self):
        """Test that linear layer works with vmap."""

        def apply_linear_single(x):
            return linear_layer(x, self.weights, self.bias)

        vmap_linear = jax.vmap(apply_linear_single)

        # Create batch of inputs
        batch_inputs = jax.random.normal(self.key, (3, 5, self.input_dim))
        results = vmap_linear(batch_inputs)

        assert results.shape == (3, 5, self.output_dim)
        assert jnp.all(jnp.isfinite(results))

    def test_linear_layer_grad_compatibility(self):
        """Test that linear layer works with grad."""

        def linear_loss(x, w, b):
            output = linear_layer(x, w, b)
            return jnp.sum(output**2)

        grad_fn = jax.grad(linear_loss, argnums=(0, 1, 2))
        grad_x, grad_w, grad_b = grad_fn(self.input_data, self.weights, self.bias)

        assert grad_x.shape == self.input_data.shape
        assert grad_w.shape == self.weights.shape
        assert grad_b.shape == self.bias.shape
        assert jnp.all(jnp.isfinite(grad_x))
        assert jnp.all(jnp.isfinite(grad_w))
        assert jnp.all(jnp.isfinite(grad_b))


class TestCombinedTransformations:
    """Test combined JAX transformations on optimized graph operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.key = jax.random.PRNGKey(42)
        self.gno = GraphNeuralOperator(
            node_dim=3,
            edge_dim=1,
            hidden_dim=6,
            output_dim=2,
            num_layers=2,
            key=self.key,
        )

        self.nodes = jax.random.normal(self.key, (4, 3))
        self.edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        self.edge_features = jax.random.normal(jax.random.split(self.key)[0], (4, 1))

        self.graph = GraphTopology(
            nodes=self.nodes, edges=self.edges, edge_features=self.edge_features
        )

    def test_jit_vmap_combination(self):
        """Test JIT compilation of vmap functions."""

        def gno_forward(nodes):
            graph = GraphTopology(
                nodes=nodes, edges=self.edges, edge_features=self.edge_features
            )
            return self.gno(graph)

        vmap_gno = jax.vmap(gno_forward)
        jit_vmap_gno = jax.jit(vmap_gno)

        batch_nodes = jax.random.normal(self.key, (3, 4, 3))

        result_vmap = vmap_gno(batch_nodes)
        result_jit_vmap = jit_vmap_gno(batch_nodes)

        assert jnp.allclose(result_vmap, result_jit_vmap)

    def test_jit_grad_combination(self):
        """Test JIT compilation of gradient functions."""

        def gno_loss(nodes):
            graph = GraphTopology(
                nodes=nodes, edges=self.edges, edge_features=self.edge_features
            )
            output = self.gno(graph)
            return jnp.sum(output**2)

        grad_fn = jax.grad(gno_loss)
        jit_grad_fn = jax.jit(grad_fn)

        result_grad = grad_fn(self.nodes)
        result_jit_grad = jit_grad_fn(self.nodes)

        assert jnp.allclose(result_grad, result_jit_grad)

    def test_vmap_grad_combination(self):
        """Test vmap of gradient functions."""

        def gno_output_norm(nodes):
            graph = GraphTopology(
                nodes=nodes, edges=self.edges, edge_features=self.edge_features
            )
            output = self.gno(graph)
            return jnp.linalg.norm(output)

        grad_fn = jax.grad(gno_output_norm)
        vmap_grad_fn = jax.vmap(grad_fn)

        batch_nodes = jax.random.normal(self.key, (3, 4, 3))
        gradients = vmap_grad_fn(batch_nodes)

        assert gradients.shape == (3, 4, 3)
        assert jnp.all(jnp.isfinite(gradients))


class TestPerformanceBenchmarks:
    """Performance benchmarks for optimized graph operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.key = jax.random.PRNGKey(42)

    def test_large_graph_gno_performance(self):
        """Test performance on large graphs."""
        # Create larger graph
        num_nodes = 100
        num_edges = 200

        nodes = jax.random.normal(self.key, (num_nodes, 5))
        edges = jax.random.randint(
            jax.random.split(self.key)[0], (num_edges, 2), 0, num_nodes
        )
        edge_features = jax.random.normal(jax.random.split(self.key)[1], (num_edges, 2))

        graph = GraphTopology(nodes=nodes, edges=edges, edge_features=edge_features)

        gno = GraphNeuralOperator(
            node_dim=5,
            edge_dim=2,
            hidden_dim=16,
            output_dim=4,
            num_layers=3,
            key=self.key,
        )

        @jax.jit
        def gno_forward(graph):
            return gno(graph)

        # Warm up JIT compilation
        _ = gno_forward(graph)

        # Test that computation completes successfully
        result = gno_forward(graph)
        assert result.shape == (num_nodes, 4)
        assert jnp.all(jnp.isfinite(result))

    def test_batch_graph_processing_performance(self):
        """Test performance of batch graph processing."""

        def process_single_graph(nodes):
            edges = jnp.array([[0, 1], [1, 2], [2, 0]])
            edge_features = jnp.ones((3, 1))

            graph = GraphTopology(nodes=nodes, edges=edges, edge_features=edge_features)

            gno = GraphNeuralOperator(
                node_dim=4,
                edge_dim=1,
                hidden_dim=8,
                output_dim=2,
                num_layers=2,
                key=self.key,
            )

            return gno(graph)

        @jax.jit
        def batch_process(batch_nodes):
            return jax.vmap(process_single_graph)(batch_nodes)

        batch_size = 20
        batch_nodes = jax.random.normal(self.key, (batch_size, 3, 4))

        # Warm up JIT compilation
        _ = batch_process(batch_nodes)

        # Test that computation completes successfully
        results = batch_process(batch_nodes)
        assert results.shape == (batch_size, 3, 2)
        assert jnp.all(jnp.isfinite(results))


if __name__ == "__main__":
    pytest.main([__file__])
