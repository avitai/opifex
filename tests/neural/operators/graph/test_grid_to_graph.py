"""Tests for grid to graph conversion utilities.

Following TDD principles, these tests define expected behavior for converting
2D grid data to graph format compatible with GraphNeuralOperator.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.neural.operators.graph.utils import grid_to_graph_data


class TestGridToGraphData:
    """Tests for grid_to_graph_data function."""

    def test_output_shapes_single_sample(self):
        """Test output shapes for a single sample."""
        # 2D grid: 4x4 with 1 channel
        grid = jnp.ones((1, 1, 4, 4))  # [batch, channels, H, W]

        node_features, edge_indices, edge_features = grid_to_graph_data(grid)

        # 4x4 grid = 16 nodes
        assert node_features.shape == (
            1,
            16,
            3,
        )  # [batch, num_nodes, node_dim=value+x+y]
        # 4-connectivity: each interior node has 4 edges, boundary nodes have fewer
        # Expected edges: 2 * (H * (W-1) + (H-1) * W) = 2 * (4*3 + 3*4) = 2 * 24 = 48
        assert edge_indices.shape[0] == 1  # batch
        assert edge_indices.shape[2] == 2  # [src, dst]
        assert edge_features.shape[0] == 1  # batch
        assert edge_features.shape[2] == 2  # [dx, dy] relative positions

    def test_output_shapes_batch(self):
        """Test output shapes for batched input."""
        grid = jnp.ones((8, 1, 4, 4))  # [batch, channels, H, W]

        node_features, edge_indices, edge_features = grid_to_graph_data(grid)

        assert node_features.shape[0] == 8  # batch dimension preserved
        assert edge_indices.shape[0] == 8
        assert edge_features.shape[0] == 8

    def test_node_features_contain_values(self):
        """Test that node features contain grid values."""
        # Create grid with known values
        grid = jnp.arange(4).reshape(1, 1, 2, 2).astype(jnp.float32)
        # Grid: [[0, 1], [2, 3]] -> nodes: [0, 1, 2, 3]

        node_features, _, _ = grid_to_graph_data(grid)

        # First column of node features should be the flattened grid values
        values = node_features[0, :, 0]  # [num_nodes]
        expected = jnp.array([0.0, 1.0, 2.0, 3.0])  # row-major flatten
        assert jnp.allclose(values, expected)

    def test_node_features_contain_positions(self):
        """Test that node features include normalized positions."""
        grid = jnp.ones((1, 1, 3, 3))

        node_features, _, _ = grid_to_graph_data(grid)

        # Node (0, 0) should have position (0, 0) normalized to [0, 1]
        # Node (2, 2) should have position (1, 1)
        pos_x = node_features[0, :, 1]  # x positions
        pos_y = node_features[0, :, 2]  # y positions

        # First node (top-left)
        assert jnp.isclose(pos_x[0], 0.0)
        assert jnp.isclose(pos_y[0], 0.0)

        # Last node (bottom-right) in 3x3 grid: index 8
        assert jnp.isclose(pos_x[8], 1.0)
        assert jnp.isclose(pos_y[8], 1.0)

    def test_edge_connectivity_4neighbor(self):
        """Test 4-neighbor connectivity pattern."""
        grid = jnp.ones((1, 1, 2, 2))  # 2x2 = 4 nodes

        _, edge_indices, _ = grid_to_graph_data(grid, connectivity=4)

        # 2x2 grid with 4-connectivity:
        # Node 0 (0,0) connects to 1 (0,1) and 2 (1,0)
        # Node 1 (0,1) connects to 0 and 3
        # Node 2 (1,0) connects to 0 and 3
        # Node 3 (1,1) connects to 1 and 2
        # Total: 8 directed edges (4 undirected * 2)
        num_edges = edge_indices.shape[1]
        assert num_edges == 8

    def test_edge_connectivity_8neighbor(self):
        """Test 8-neighbor connectivity pattern."""
        grid = jnp.ones((1, 1, 2, 2))  # 2x2 = 4 nodes

        _, edge_indices, _ = grid_to_graph_data(grid, connectivity=8)

        # 2x2 grid with 8-connectivity adds diagonal edges
        # All 4 nodes can connect to each other
        # Total: 12 directed edges (6 undirected * 2)
        num_edges = edge_indices.shape[1]
        assert num_edges == 12

    def test_edge_features_are_relative_positions(self):
        """Test that edge features encode relative positions."""
        grid = jnp.ones((1, 1, 2, 2))

        node_features, edge_indices, edge_features = grid_to_graph_data(grid)

        # For an edge (src, dst), edge_feature should be (dst_pos - src_pos)
        batch_idx = 0
        for edge_idx in range(edge_indices.shape[1]):
            src = int(edge_indices[batch_idx, edge_idx, 0])
            dst = int(edge_indices[batch_idx, edge_idx, 1])

            src_pos = node_features[batch_idx, src, 1:3]  # [x, y]
            dst_pos = node_features[batch_idx, dst, 1:3]

            expected_edge_feat = dst_pos - src_pos
            actual_edge_feat = edge_features[batch_idx, edge_idx]

            assert jnp.allclose(actual_edge_feat, expected_edge_feat, atol=1e-5)

    def test_multi_channel_input(self):
        """Test handling of multi-channel input grids."""
        grid = jnp.ones((1, 3, 4, 4))  # 3 channels

        node_features, _, _ = grid_to_graph_data(grid)

        # node_dim = channels + 2 (for positions)
        assert node_features.shape[2] == 5  # 3 channels + x + y

    def test_jit_compatibility(self):
        """Test that function works with JIT compilation."""
        grid = jnp.ones((1, 1, 4, 4))

        @jax.jit
        def convert(g):
            return grid_to_graph_data(g)

        node_features, _, _ = convert(grid)

        assert node_features.shape == (1, 16, 3)
        assert jnp.all(jnp.isfinite(node_features))

    def test_vmap_compatibility(self):
        """Test that function is vmap-able over batch dimension."""
        grids = jnp.ones((4, 1, 4, 4))

        # Direct call should handle batch
        node_features, _, _ = grid_to_graph_data(grids)

        assert node_features.shape[0] == 4

    def test_radius_based_connectivity(self):
        """Test radius-based connectivity pattern."""
        grid = jnp.ones((1, 1, 4, 4))

        # Radius of 1.5 should connect diagonal neighbors (distance sqrt(2) < 1.5)
        _, edge_indices, _ = grid_to_graph_data(grid, connectivity="radius", radius=1.5)

        # Should have more edges than 4-connectivity
        _, edge_indices_4, _ = grid_to_graph_data(grid, connectivity=4)

        assert edge_indices.shape[1] > edge_indices_4.shape[1]


class TestGridToGraphDataIntegration:
    """Integration tests with GraphNeuralOperator."""

    def test_compatible_with_gno_forward(self):
        """Test that output can be passed to GraphNeuralOperator."""
        from flax import nnx

        from opifex.neural.operators.graph.gno import GraphNeuralOperator

        # Create grid data
        grid = jnp.ones((2, 1, 4, 4))

        # Convert to graph
        node_features, edge_indices, edge_features = grid_to_graph_data(grid)

        # Create GNO
        gno = GraphNeuralOperator(
            node_dim=node_features.shape[-1],
            hidden_dim=16,
            num_layers=2,
            edge_dim=edge_features.shape[-1],
            rngs=nnx.Rngs(42),
        )

        # Forward pass should work
        output = gno(node_features, edge_indices, edge_features)

        assert output.shape == node_features.shape
        assert jnp.all(jnp.isfinite(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
