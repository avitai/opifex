"""Tests for MeshGraphNet implementation.

Tests encoder-processor-decoder architecture for mesh-based simulation,
following the DeepMind MeshGraphNet paper. Tests are written before
implementation (TDD).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.graph.mesh_graph_net import MeshGraphNet
from opifex.neural.operators.graph.utils import grid_to_graph_data


class TestMeshGraphNetForwardShapes:
    """Test output shapes for variable mesh sizes."""

    def test_basic_forward_shape(self) -> None:
        """Output shape is [batch, num_nodes, output_dim]."""
        batch_size = 2
        num_nodes = 16
        num_edges = 32
        node_input_dim = 5
        edge_input_dim = 3
        output_dim = 4

        model = MeshGraphNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_input_dim)
        )
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )
        edge_features = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, num_edges, edge_input_dim)
        )

        output = model(node_features, edge_indices, edge_features)

        assert output.shape == (batch_size, num_nodes, output_dim)

    @pytest.mark.parametrize(
        ("num_nodes", "num_edges"),
        [(8, 16), (32, 64), (64, 128)],
    )
    def test_variable_mesh_sizes(self, num_nodes: int, num_edges: int) -> None:
        """Output shape adapts to different mesh sizes."""
        batch_size = 2
        node_input_dim = 5
        edge_input_dim = 3
        output_dim = 4

        model = MeshGraphNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_input_dim)
        )
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )
        edge_features = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, num_edges, edge_input_dim)
        )

        output = model(node_features, edge_indices, edge_features)

        assert output.shape == (batch_size, num_nodes, output_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_without_edge_features(self) -> None:
        """Forward pass works when edge_features is None."""
        batch_size = 2
        num_nodes = 16
        num_edges = 32
        node_input_dim = 5
        output_dim = 4

        model = MeshGraphNet(
            node_input_dim=node_input_dim,
            edge_input_dim=0,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_input_dim)
        )
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )

        output = model(node_features, edge_indices)

        assert output.shape == (batch_size, num_nodes, output_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_output_dim_different_from_input_dim(self) -> None:
        """Output dimension can differ from input node dimension."""
        batch_size = 2
        num_nodes = 16
        num_edges = 32
        node_input_dim = 5
        edge_input_dim = 3
        output_dim = 10  # Different from node_input_dim

        model = MeshGraphNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            hidden_dim=64,
            num_layers=4,
            rngs=nnx.Rngs(0),
        )

        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_input_dim)
        )
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )
        edge_features = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, num_edges, edge_input_dim)
        )

        output = model(node_features, edge_indices, edge_features)

        assert output.shape == (batch_size, num_nodes, output_dim)


class TestMeshGraphNetGradientFlow:
    """Test gradient flow through MeshGraphNet."""

    def test_no_nan_gradients(self) -> None:
        """All parameter gradients are finite (no NaN or Inf)."""
        batch_size = 2
        num_nodes = 16
        num_edges = 32
        node_input_dim = 5
        edge_input_dim = 3
        output_dim = 4

        model = MeshGraphNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_input_dim)
        )
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )
        edge_features = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, num_edges, edge_input_dim)
        )

        def loss_fn(
            model: MeshGraphNet,
            nodes: jax.Array,
            edges: jax.Array,
            edge_feats: jax.Array,
        ) -> jax.Array:
            output = model(nodes, edges, edge_feats)
            return jnp.mean(output**2)

        grads = nnx.grad(loss_fn)(model, node_features, edge_indices, edge_features)

        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0
        for leaf in grad_leaves:
            assert jnp.all(jnp.isfinite(leaf)), "Gradient contains NaN or Inf"

    def test_gradients_nonzero(self) -> None:
        """Gradients are nonzero, confirming information flows through all layers."""
        batch_size = 2
        num_nodes = 16
        num_edges = 32
        node_input_dim = 5
        edge_input_dim = 3
        output_dim = 4

        model = MeshGraphNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_input_dim)
        )
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )
        edge_features = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, num_edges, edge_input_dim)
        )

        def loss_fn(
            model: MeshGraphNet,
            nodes: jax.Array,
            edges: jax.Array,
            edge_feats: jax.Array,
        ) -> jax.Array:
            output = model(nodes, edges, edge_feats)
            return jnp.mean(output**2)

        grads = nnx.grad(loss_fn)(model, node_features, edge_indices, edge_features)

        grad_leaves = jax.tree_util.tree_leaves(grads)
        has_nonzero = any(jnp.any(leaf != 0) for leaf in grad_leaves)
        assert has_nonzero, "All gradients are zero -- no information flow"


class TestMeshGraphNetWithGridToGraph:
    """Test MeshGraphNet with existing grid_to_graph_data() utility."""

    def test_works_with_grid_to_graph_data(self) -> None:
        """MeshGraphNet accepts output from grid_to_graph_data() directly."""
        batch_size = 2
        channels = 3
        height = 4
        width = 4

        grid = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, channels, height, width)
        )

        node_features, edge_indices, edge_features = grid_to_graph_data(
            grid, connectivity=4
        )

        # node_features: [B, H*W, channels+2], edge_features: [B, num_edges, 2]
        node_input_dim = channels + 2  # grid values + (x, y) positions
        edge_input_dim = 2  # relative position vectors
        output_dim = channels

        model = MeshGraphNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        output = model(node_features, edge_indices, edge_features)

        assert output.shape == (batch_size, height * width, output_dim)
        assert jnp.all(jnp.isfinite(output))

    def test_works_with_8_connectivity(self) -> None:
        """MeshGraphNet works with 8-connected grid graphs."""
        batch_size = 2
        channels = 1
        height = 4
        width = 4

        grid = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, channels, height, width)
        )

        node_features, edge_indices, edge_features = grid_to_graph_data(
            grid, connectivity=8
        )

        node_input_dim = channels + 2
        edge_input_dim = 2
        output_dim = channels

        model = MeshGraphNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )

        output = model(node_features, edge_indices, edge_features)

        assert output.shape == (batch_size, height * width, output_dim)
        assert jnp.all(jnp.isfinite(output))


class TestMeshGraphNetMultiScale:
    """Test multi-scale behavior via different processor depths."""

    def test_different_depths_produce_different_results(self) -> None:
        """Models with different num_layers produce different outputs.

        More message-passing layers increase the receptive field, so the model
        captures information at a larger spatial scale. Different depths should
        yield different predictions.
        """
        batch_size = 2
        num_nodes = 16
        num_edges = 32
        node_input_dim = 5
        edge_input_dim = 3
        output_dim = 4

        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_input_dim)
        )
        edge_indices = jax.random.randint(
            jax.random.PRNGKey(1), (batch_size, num_edges, 2), 0, num_nodes
        )
        edge_features = jax.random.normal(
            jax.random.PRNGKey(2), (batch_size, num_edges, edge_input_dim)
        )

        model_shallow = MeshGraphNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=1,
            rngs=nnx.Rngs(0),
        )

        model_deep = MeshGraphNet(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=output_dim,
            hidden_dim=32,
            num_layers=6,
            rngs=nnx.Rngs(0),
        )

        output_shallow = model_shallow(node_features, edge_indices, edge_features)
        output_deep = model_deep(node_features, edge_indices, edge_features)

        assert output_shallow.shape == output_deep.shape
        assert not jnp.allclose(output_shallow, output_deep, atol=1e-5), (
            "Shallow and deep models should produce different outputs"
        )

    def test_more_layers_increases_receptive_field(self) -> None:
        """Deeper model's output depends on more distant nodes.

        Build a chain graph (0->1->2->...->N). Perturb node 0 and measure
        the total change at distant nodes (indices >= 3). A deeper model
        should propagate the perturbation further along the chain.
        """
        num_nodes = 8
        node_input_dim = 4
        output_dim = 4
        batch_size = 1

        # Chain graph: 0-1, 1-2, ..., 6-7 (bidirectional), no edge features
        src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
        dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
        edge_indices = jnp.array([[src, dst]]).transpose(0, 2, 1)  # [1, 2*(N-1), 2]

        node_features = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, num_nodes, node_input_dim)
        )

        # Perturbed version: add large noise at node 0
        perturbed = node_features.at[0, 0, :].add(10.0)

        # With 1 layer, perturbation at node 0 reaches only node 1
        # With 6 layers, perturbation reaches nodes 0-5 at least
        # Compare total change at distant nodes (index 3..7)
        distant_slice = slice(3, None)

        diff_shallow = 0.0
        diff_deep = 0.0
        for num_layers in [1, 6]:
            model = MeshGraphNet(
                node_input_dim=node_input_dim,
                edge_input_dim=0,
                output_dim=output_dim,
                hidden_dim=32,
                num_layers=num_layers,
                rngs=nnx.Rngs(42),
            )

            out_original = model(node_features, edge_indices)
            out_perturbed = model(perturbed, edge_indices)

            # Sum absolute differences at distant nodes
            diff = jnp.abs(
                out_perturbed[0, distant_slice] - out_original[0, distant_slice]
            ).sum()
            if num_layers == 1:
                diff_shallow = float(diff)
            else:
                diff_deep = float(diff)

        assert diff_deep > diff_shallow, (
            f"Deep model (diff={diff_deep:.6f}) should propagate perturbation "
            f"further than shallow (diff={diff_shallow:.6f})"
        )


class TestMeshGraphNetConfigValidation:
    """Test that invalid configurations are rejected."""

    def test_num_layers_zero_raises(self) -> None:
        """num_layers=0 is invalid and should raise ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            MeshGraphNet(
                node_input_dim=5,
                edge_input_dim=3,
                output_dim=4,
                hidden_dim=32,
                num_layers=0,
                rngs=nnx.Rngs(0),
            )

    def test_num_layers_negative_raises(self) -> None:
        """Negative num_layers should raise ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            MeshGraphNet(
                node_input_dim=5,
                edge_input_dim=3,
                output_dim=4,
                hidden_dim=32,
                num_layers=-1,
                rngs=nnx.Rngs(0),
            )

    def test_hidden_dim_zero_raises(self) -> None:
        """hidden_dim=0 is invalid and should raise ValueError."""
        with pytest.raises(ValueError, match="hidden_dim"):
            MeshGraphNet(
                node_input_dim=5,
                edge_input_dim=3,
                output_dim=4,
                hidden_dim=0,
                num_layers=3,
                rngs=nnx.Rngs(0),
            )

    def test_output_dim_zero_raises(self) -> None:
        """output_dim=0 is invalid and should raise ValueError."""
        with pytest.raises(ValueError, match="output_dim"):
            MeshGraphNet(
                node_input_dim=5,
                edge_input_dim=3,
                output_dim=0,
                hidden_dim=32,
                num_layers=3,
                rngs=nnx.Rngs(0),
            )

    def test_node_input_dim_zero_raises(self) -> None:
        """node_input_dim=0 is invalid and should raise ValueError."""
        with pytest.raises(ValueError, match="node_input_dim"):
            MeshGraphNet(
                node_input_dim=0,
                edge_input_dim=3,
                output_dim=4,
                hidden_dim=32,
                num_layers=3,
                rngs=nnx.Rngs(0),
            )

    def test_valid_config_succeeds(self) -> None:
        """Valid configuration should not raise."""
        model = MeshGraphNet(
            node_input_dim=5,
            edge_input_dim=3,
            output_dim=4,
            hidden_dim=32,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )
        assert model.num_layers == 3
        assert model.hidden_dim == 32
