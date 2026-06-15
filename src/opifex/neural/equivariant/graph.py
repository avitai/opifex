r"""Neighbour graph construction and segment-scatter for message passing.

These utilities turn a point cloud into the ``(senders, receivers)`` edge index
used by equivariant message-passing networks, and aggregate per-edge messages
back onto nodes.

* :func:`radius_graph` follows the dense pairwise-distance + fixed-size mask
  approach of ``../e3nn-jax/e3nn_jax/_src/radius_graph.py`` (it uses
  ``jnp.where(mask, size=...)`` to return a statically shaped edge list).  The
  neighbour-list concept is that of ``../jax-md/jax_md/partition.py``; unlike
  ``jax-md``'s cell-list partitioning, this implementation is a simple **dense
  ``O(N^2)``** pairwise computation -- correct and ``jit``-friendly for the small
  to medium molecules typical of interatomic-potential workloads, but not
  intended for very large ``N``.

* :func:`scatter_sum`, :func:`scatter_mean` and :func:`scatter_max` wrap
  ``jax.ops.segment_*`` (cf. ``../e3nn-jax/e3nn_jax/_src/scatter.py``).

Static-shape contract: ``radius_graph`` returns edge arrays of fixed length
``max_edges`` (padded with ``-1``), so the output shape does not depend on the
data -- a requirement for ``jit``.  Callers must size ``max_edges`` to an upper
bound on the true edge count; excess edges beyond ``max_edges`` are dropped.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def radius_graph(
    positions: Float[Array, "n 3"],
    cutoff: float,
    *,
    max_edges: int,
    self_loops: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Build the radius-graph edge index for a point cloud.

    Two nodes ``i`` and ``j`` are connected (with ``i`` the sender and ``j`` the
    receiver) when ``|positions[i] - positions[j]| < cutoff``.

    Args:
        positions: Node coordinates of shape ``(n, 3)``.
        cutoff: Connection radius ``r_c`` (positive).
        max_edges: Static upper bound on the number of returned edges; the output
            arrays have this length, padded with ``-1`` for unused slots.
        self_loops: If ``True``, include ``(i, i)`` edges. Default ``False``.

    Returns:
        A pair ``(senders, receivers)`` of integer arrays of shape
        ``(max_edges,)``; padding entries hold ``-1``.
    """
    deltas = positions[:, None, :] - positions[None, :, :]
    squared_distances = jnp.sum(deltas**2, axis=-1)
    within = squared_distances < cutoff**2
    mask = within if self_loops else within & ~jnp.eye(positions.shape[0], dtype=bool)
    senders, receivers = jnp.where(mask, size=max_edges, fill_value=-1)
    return senders, receivers


def scatter_sum(
    data: Float[Array, "edges ..."],
    index: jax.Array,
    *,
    num_segments: int,
) -> Float[Array, "num_segments ..."]:
    """Sum ``data`` into segments addressed by ``index``.

    ``output[index[e]] += data[e]`` for every edge ``e`` (negative indices, e.g.
    the ``-1`` padding from :func:`radius_graph`, are dropped by
    ``segment_sum``).

    Args:
        data: Per-edge values of shape ``(edges, ...)``.
        index: Destination node index per edge, shape ``(edges,)``.
        num_segments: Number of output segments (nodes).

    Returns:
        Aggregated array of shape ``(num_segments, ...)``.
    """
    return jax.ops.segment_sum(data, index, num_segments=num_segments)


def scatter_mean(
    data: Float[Array, "edges ..."],
    index: jax.Array,
    *,
    num_segments: int,
) -> Float[Array, "num_segments ..."]:
    """Average ``data`` within segments addressed by ``index``.

    Empty segments map to zero.

    Args:
        data: Per-edge values of shape ``(edges, ...)``.
        index: Destination node index per edge, shape ``(edges,)``.
        num_segments: Number of output segments (nodes).

    Returns:
        Per-segment mean of shape ``(num_segments, ...)``.
    """
    totals = jax.ops.segment_sum(data, index, num_segments=num_segments)
    counts = jax.ops.segment_sum(
        jnp.ones_like(index, dtype=data.dtype), index, num_segments=num_segments
    )
    safe_counts = jnp.where(counts == 0, 1.0, counts)
    extra_axes = (1,) * (data.ndim - 1)
    return totals / safe_counts.reshape(num_segments, *extra_axes)


def scatter_max(
    data: Float[Array, "edges ..."],
    index: jax.Array,
    *,
    num_segments: int,
) -> Float[Array, "num_segments ..."]:
    """Segment-wise maximum of ``data`` addressed by ``index``.

    Args:
        data: Per-edge values of shape ``(edges, ...)``.
        index: Destination node index per edge, shape ``(edges,)``.
        num_segments: Number of output segments (nodes).

    Returns:
        Per-segment maximum of shape ``(num_segments, ...)``.
    """
    return jax.ops.segment_max(data, index, num_segments=num_segments)
