r"""Tests for the neighbour graph and segment-scatter utilities.

``radius_graph`` follows the dense pairwise-distance + fixed-size mask concept of
``../e3nn-jax/e3nn_jax/_src/radius_graph.py`` and ``jax_md.partition`` (the
neighbour-list idea); the scatter helpers wrap ``jax.ops.segment_*`` as in
``../e3nn-jax/e3nn_jax/_src/scatter.py``.

The load-bearing checks are: the correct edge set for a known geometry, scatter
correctness against a manual reduction, and ``jit``/``vmap`` compatibility under
the documented fixed-output-size (static-shape) contract.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.neural.equivariant import (
    radius_graph,
    scatter_max,
    scatter_mean,
    scatter_sum,
)


class TestRadiusGraph:
    def test_known_geometry_edges(self) -> None:
        """A line of 3 points spaced 1.0 apart, cutoff 1.5: only adjacent pairs."""
        positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        senders, receivers = radius_graph(positions, cutoff=1.5, max_edges=8)
        pairs = {
            (int(s), int(r))
            for s, r in zip(senders.tolist(), receivers.tolist(), strict=True)
            if s >= 0
        }
        assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}

    def test_no_self_loops_by_default(self) -> None:
        positions = jnp.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        senders, receivers = radius_graph(positions, cutoff=1.0, max_edges=8)
        valid = [
            (int(s), int(r))
            for s, r in zip(senders.tolist(), receivers.tolist(), strict=True)
            if s >= 0
        ]
        assert all(s != r for s, r in valid)

    def test_self_loops_when_requested(self) -> None:
        positions = jnp.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        senders, receivers = radius_graph(positions, cutoff=1.0, max_edges=8, self_loops=True)
        valid = {
            (int(s), int(r))
            for s, r in zip(senders.tolist(), receivers.tolist(), strict=True)
            if s >= 0
        }
        assert (0, 0) in valid and (1, 1) in valid

    def test_fixed_output_shape(self) -> None:
        """Output arrays have static length ``max_edges`` (padded with ``-1``)."""
        positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        senders, receivers = radius_graph(positions, cutoff=1.5, max_edges=10)
        assert senders.shape == (10,)
        assert receivers.shape == (10,)
        assert int(jnp.sum(senders == -1)) == 8

    def test_far_points_have_no_edges(self) -> None:
        positions = jnp.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        senders, _ = radius_graph(positions, cutoff=1.5, max_edges=8)
        assert int(jnp.sum(senders >= 0)) == 0

    def test_jit_compatibility(self) -> None:
        positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        jitted = jax.jit(lambda p: radius_graph(p, cutoff=1.5, max_edges=8))
        senders, receivers = jitted(positions)
        ref_s, ref_r = radius_graph(positions, cutoff=1.5, max_edges=8)
        assert jnp.array_equal(senders, ref_s)
        assert jnp.array_equal(receivers, ref_r)


class TestScatter:
    def test_scatter_sum(self) -> None:
        data = jnp.asarray([1.0, 2.0, 3.0, 4.0])
        index = jnp.asarray([0, 1, 0, 1])
        result = scatter_sum(data, index, num_segments=2)
        assert jnp.allclose(result, jnp.asarray([4.0, 6.0]))

    def test_scatter_sum_vector_features(self) -> None:
        data = jnp.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        index = jnp.asarray([0, 0, 1])
        result = scatter_sum(data, index, num_segments=2)
        assert jnp.allclose(result, jnp.asarray([[3.0, 3.0], [3.0, 3.0]]))

    def test_scatter_mean(self) -> None:
        data = jnp.asarray([2.0, 4.0, 9.0])
        index = jnp.asarray([0, 0, 1])
        result = scatter_mean(data, index, num_segments=2)
        assert jnp.allclose(result, jnp.asarray([3.0, 9.0]))

    def test_scatter_mean_empty_segment_is_zero(self) -> None:
        data = jnp.asarray([1.0, 2.0])
        index = jnp.asarray([0, 0])
        result = scatter_mean(data, index, num_segments=3)
        assert jnp.allclose(result, jnp.asarray([1.5, 0.0, 0.0]))

    def test_scatter_max(self) -> None:
        data = jnp.asarray([1.0, 5.0, 3.0, 2.0])
        index = jnp.asarray([0, 0, 1, 1])
        result = scatter_max(data, index, num_segments=2)
        assert jnp.allclose(result, jnp.asarray([5.0, 3.0]))

    def test_jit_compatibility(self) -> None:
        data = jnp.asarray([1.0, 2.0, 3.0])
        index = jnp.asarray([0, 1, 1])
        jitted = jax.jit(lambda d: scatter_sum(d, index, num_segments=2))
        assert jnp.allclose(jitted(data), scatter_sum(data, index, num_segments=2))

    def test_vmap_compatibility(self) -> None:
        data = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        index = jnp.asarray([0, 1, 1])
        batched = jax.vmap(lambda d: scatter_sum(d, index, num_segments=2))(data)
        assert batched.shape == (2, 2)
        assert jnp.allclose(batched[0], jnp.asarray([1.0, 5.0]))
        assert jnp.allclose(batched[1], jnp.asarray([4.0, 11.0]))
