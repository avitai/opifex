"""Test Graph Neural Operator implementation.

Modern tests for GraphNeuralOperator aligned with current API.
Focuses on proper graph operator testing without legacy compatibility.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.graph.gno import GraphNeuralOperator, MollifiedGNO


class TestGraphNeuralOperator:
    """Test suite for GraphNeuralOperator with modern API."""

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data for testing."""
        batch_size = 4
        num_nodes = 16
        node_dim = 8
        num_edges = 32

        # Generate graph data
        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_dim)
        )

        # Random edge connectivity (ensure valid indices)
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )

        return {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
        }

    def test_gno_initialization(self):
        """Test GNO initialization with modern parameters."""
        model = GraphNeuralOperator(
            node_dim=8,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        assert hasattr(model, "input_projection")
        assert hasattr(model, "output_projection")
        assert hasattr(model, "message_passing_layers")
        assert len(model.message_passing_layers) == 3

    def test_gno_forward_pass(self, sample_graph_data):
        """Test GNO forward pass."""
        model = GraphNeuralOperator(
            node_dim=8,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        output = model(
            sample_graph_data["node_features"],
            sample_graph_data["edge_indices"],
        )

        expected_shape = sample_graph_data["node_features"].shape
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_gno_with_edge_features(self, sample_graph_data):
        """Test GNO with edge features."""
        edge_dim = 4
        edge_features = jax.random.normal(
            jax.random.PRNGKey(2),
            (
                sample_graph_data["node_features"].shape[0],
                sample_graph_data["num_edges"],
                edge_dim,
            ),
        )

        model = GraphNeuralOperator(
            node_dim=8,
            hidden_dim=32,
            num_layers=3,
            edge_dim=edge_dim,
            rngs=nnx.Rngs(0),
        )

        output = model(
            sample_graph_data["node_features"],
            sample_graph_data["edge_indices"],
            edge_features,
        )

        expected_shape = sample_graph_data["node_features"].shape
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_gno_different_hidden_dims(self, sample_graph_data):
        """Test GNO with different hidden dimensions."""
        for hidden_dim in [16, 32, 64]:
            model = GraphNeuralOperator(
                node_dim=8,
                hidden_dim=hidden_dim,
                num_layers=3,
                rngs=nnx.Rngs(0),
            )

            output = model(
                sample_graph_data["node_features"],
                sample_graph_data["edge_indices"],
            )

            expected_shape = sample_graph_data["node_features"].shape
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_gno_different_num_layers(self, sample_graph_data):
        """Test GNO with different numbers of layers."""
        for num_layers in [1, 3, 5]:
            model = GraphNeuralOperator(
                node_dim=8,
                hidden_dim=32,
                num_layers=num_layers,
                rngs=nnx.Rngs(0),
            )

            output = model(
                sample_graph_data["node_features"],
                sample_graph_data["edge_indices"],
            )

            expected_shape = sample_graph_data["node_features"].shape
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_gno_activation_functions(self, sample_graph_data):
        """Test GNO with different activation functions."""
        for activation in [nnx.gelu, nnx.relu, nnx.tanh]:
            model = GraphNeuralOperator(
                node_dim=8,
                hidden_dim=32,
                num_layers=3,
                activation=activation,
                rngs=nnx.Rngs(0),
            )

            output = model(
                sample_graph_data["node_features"],
                sample_graph_data["edge_indices"],
            )

            expected_shape = sample_graph_data["node_features"].shape
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_gno_different_graph_sizes(self):
        """Test GNO with different graph sizes."""
        graph_sizes = [(8, 16), (16, 32), (32, 64)]  # (num_nodes, num_edges)

        for num_nodes, num_edges in graph_sizes:
            batch_size = 2
            node_dim = 8

            node_features = jax.random.normal(
                jax.random.PRNGKey(0), (batch_size, num_nodes, node_dim)
            )
            edge_indices = jax.random.randint(
                jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
            )

            model = GraphNeuralOperator(
                node_dim=8,
                hidden_dim=32,
                num_layers=3,
                rngs=nnx.Rngs(0),
            )

            output = model(node_features, edge_indices)

            expected_shape = (batch_size, num_nodes, node_dim)
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_gno_without_edge_features(self, sample_graph_data):
        """Test GNO without edge features (edge_dim=0)."""
        model = GraphNeuralOperator(
            node_dim=8,
            hidden_dim=32,
            num_layers=3,
            edge_dim=0,  # No edge features
            rngs=nnx.Rngs(0),
        )

        output = model(
            sample_graph_data["node_features"],
            sample_graph_data["edge_indices"],
            None,  # No edge features
        )

        expected_shape = sample_graph_data["node_features"].shape
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_gno_single_layer(self, sample_graph_data):
        """Test GNO with single layer."""
        model = GraphNeuralOperator(
            node_dim=8,
            hidden_dim=32,
            num_layers=1,
            rngs=nnx.Rngs(0),
        )

        output = model(
            sample_graph_data["node_features"],
            sample_graph_data["edge_indices"],
        )

        expected_shape = sample_graph_data["node_features"].shape
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_gno_gradient_computation(self, sample_graph_data):
        """Test gradient computation through GNO."""
        model = GraphNeuralOperator(
            node_dim=8,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        def loss_fn(model, node_features, edge_indices):
            output = model(node_features, edge_indices)
            return jnp.mean(output**2)

        grads = nnx.grad(loss_fn)(
            model,
            sample_graph_data["node_features"],
            sample_graph_data["edge_indices"],
        )

        # Check gradient properties
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)

    def test_gno_jax_transformations(self, sample_graph_data):
        """Test GNO compatibility with JAX transformations."""
        model = GraphNeuralOperator(
            node_dim=8,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        @jax.jit
        def jitted_forward(node_features, edge_indices):
            return model(node_features, edge_indices)

        output = jitted_forward(
            sample_graph_data["node_features"],
            sample_graph_data["edge_indices"],
        )

        expected_shape = sample_graph_data["node_features"].shape
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_gno_residual_connections(self, sample_graph_data):
        """Test that GNO properly handles residual connections."""
        model = GraphNeuralOperator(
            node_dim=8,
            hidden_dim=32,
            num_layers=5,  # More layers to test residual effects
            rngs=nnx.Rngs(0),
        )

        # Compare output with multi-layer processing
        output = model(
            sample_graph_data["node_features"],
            sample_graph_data["edge_indices"],
        )

        # Should preserve input shape
        expected_shape = sample_graph_data["node_features"].shape
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

        # Output should be different from input (model actually processes)
        assert not jnp.allclose(output, sample_graph_data["node_features"])

    def test_gno_memory_efficiency(self):
        """Test GNO with larger graphs for memory efficiency."""
        batch_size = 2
        num_nodes = 64
        node_dim = 16
        num_edges = 128

        # Create larger graph data
        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_dim)
        )
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )

        model = GraphNeuralOperator(
            node_dim=node_dim,
            hidden_dim=64,
            num_layers=4,
            rngs=nnx.Rngs(0),
        )

        output = model(node_features, edge_indices)

        expected_shape = (batch_size, num_nodes, node_dim)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_gno_parameter_efficiency(self):
        """Test parameter efficiency for different configurations."""
        # Create GNO with moderate complexity
        model = GraphNeuralOperator(
            node_dim=32,
            hidden_dim=64,
            num_layers=6,
            edge_dim=8,
            rngs=nnx.Rngs(0),
        )

        # Count parameters
        def count_parameters(model):
            return sum(
                jnp.prod(jnp.array(param.shape))
                for param in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
            )

        param_count = count_parameters(model)

        # Should have reasonable number of parameters
        assert param_count > 1000  # Not too small
        assert param_count < 1_000_000  # Not too large

        # Test functionality
        batch_size = 2
        num_nodes = 16
        num_edges = 32

        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, 32)
        )
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )
        edge_features = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, num_edges, 8)
        )

        output = model(node_features, edge_indices, edge_features)

        assert output.shape == (batch_size, num_nodes, 32)
        assert jnp.all(jnp.isfinite(output))


class TestMollifiedGNO:
    """Test the mGNO mollified smoothing extension."""

    @pytest.fixture
    def sample_graph_positions(self):
        """Create sample graph data with positions for mGNO testing."""
        batch_size = 2
        num_nodes = 8
        node_dim = 4
        num_edges = 12

        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_dim)
        )
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )
        positions = jax.random.normal(
            jax.random.PRNGKey(99), (batch_size, num_nodes, 2)
        )
        return {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "positions": positions,
        }

    def test_mgno_initialization(self):
        """Test mGNO initialization."""
        model = MollifiedGNO(
            node_dim=4,
            hidden_dim=16,
            num_layers=2,
            smoothing_radius=0.1,
            rngs=nnx.Rngs(0),
        )
        assert model.smoothing_radius == 0.1

    def test_mgno_forward_shape(self, sample_graph_positions):
        """Output shape should match input node shape."""
        model = MollifiedGNO(
            node_dim=4,
            hidden_dim=16,
            num_layers=2,
            smoothing_radius=0.1,
            rngs=nnx.Rngs(0),
        )
        data = sample_graph_positions
        out = model(
            data["node_features"],
            data["edge_indices"],
            positions=data["positions"],
        )
        assert out.shape == data["node_features"].shape

    def test_mgno_finite_output(self, sample_graph_positions):
        """Smoothed output should be finite."""
        model = MollifiedGNO(
            node_dim=4,
            hidden_dim=16,
            num_layers=2,
            smoothing_radius=0.1,
            rngs=nnx.Rngs(0),
        )
        data = sample_graph_positions
        out = model(
            data["node_features"],
            data["edge_indices"],
            positions=data["positions"],
        )
        assert jnp.all(jnp.isfinite(out))

    def test_mgno_jit_compatible(self, sample_graph_positions):
        """MollifiedGNO should work under JIT."""
        model = MollifiedGNO(
            node_dim=4,
            hidden_dim=16,
            num_layers=2,
            smoothing_radius=0.1,
            rngs=nnx.Rngs(0),
        )
        data = sample_graph_positions

        @nnx.jit
        def jitted_forward(m, n, e, p):
            return m(n, e, positions=p)

        out = jitted_forward(
            model,
            data["node_features"],
            data["edge_indices"],
            data["positions"],
        )
        assert out.shape == data["node_features"].shape

    def test_mgno_gradient_flow(self, sample_graph_positions):
        """Gradients should flow through MollifiedGNO."""
        model = MollifiedGNO(
            node_dim=4,
            hidden_dim=16,
            num_layers=2,
            smoothing_radius=0.1,
            rngs=nnx.Rngs(0),
        )
        data = sample_graph_positions

        def loss_fn(model, node_features, edge_indices, positions):
            return jnp.mean(
                model(node_features, edge_indices, positions=positions) ** 2
            )

        grads = nnx.grad(loss_fn)(
            model,
            data["node_features"],
            data["edge_indices"],
            data["positions"],
        )
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)

    def test_mgno_zero_radius(self, sample_graph_positions):
        """With radius=0, mGNO should act like base GNO (no smoothing)."""
        model = MollifiedGNO(
            node_dim=4,
            hidden_dim=16,
            num_layers=2,
            smoothing_radius=0.0,
            rngs=nnx.Rngs(0),
        )
        data = sample_graph_positions
        out = model(
            data["node_features"],
            data["edge_indices"],
            positions=data["positions"],
        )
        assert out.shape == data["node_features"].shape
        assert jnp.all(jnp.isfinite(out))

    def test_mgno_without_positions(self, sample_graph_positions):
        """Without positions, mGNO should skip smoothing."""
        model = MollifiedGNO(
            node_dim=4,
            hidden_dim=16,
            num_layers=2,
            smoothing_radius=0.1,
            rngs=nnx.Rngs(0),
        )
        data = sample_graph_positions
        # No positions → no smoothing, just base GNO
        out = model(data["node_features"], data["edge_indices"])
        assert out.shape == data["node_features"].shape
        assert jnp.all(jnp.isfinite(out))
