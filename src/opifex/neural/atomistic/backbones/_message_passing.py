r"""Shared message-passing primitives for atomistic backbones (DRY core).

Every interatomic-potential backbone in opifex -- :class:`SchNet`
(:mod:`opifex.neural.atomistic.backbones.schnet`), :class:`PaiNN`
(:mod:`opifex.neural.atomistic.backbones.painn`) and :class:`NequIP`
(:mod:`opifex.neural.atomistic.backbones.nequip`) -- shares the same
edge-construction and aggregation skeleton: given the padded
``(senders, receivers)`` edge index produced by
:func:`opifex.neural.equivariant.radius_graph`, compute per-edge geometry
(displacement vectors, lengths, unit directions), embed the lengths in an
invariant radial basis with a smooth cutoff envelope, and scatter per-edge
messages back onto receiver atoms with
:func:`opifex.neural.equivariant.scatter_sum`.

This module factors that skeleton once so the three backbones differ only in
their *message function* (SchNet: continuous-filter convolution; PaiNN:
scalar/vector gated update; NequIP: Clebsch-Gordan tensor product), reusing the
Q0 equivariant kit (:mod:`opifex.neural.equivariant`) for every primitive
(radius graph, scatter, radial basis, cutoff) rather than reimplementing it.

Padding contract: ``radius_graph`` pads unused edge slots with ``-1``. Gathering
``positions[-1]`` would silently wrap to the last atom, so :func:`edge_geometry`
clamps padded indices to ``0`` and zeroes the cutoff envelope on those slots; the
zero envelope multiplies every downstream message, so padded edges contribute
nothing to the scatter (``segment_sum`` itself also drops the original ``-1``
receiver indices). This mirrors the masking used by the trivial backbone in
``tests/neural/atomistic/test_base.py``.

References:
    * Schuett et al. 2018, "SchNet -- A deep learning architecture for molecules
      and materials", J. Chem. Phys. 148, 241722 (arXiv:1706.08566) -- the
      continuous-filter message/scatter pattern.
    * ``../e3nn-jax/e3nn_jax/_src/radius_graph.py`` and ``.../scatter.py`` -- the
      static-shape edge index and segment-scatter aggregation reused here.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int  # noqa: TC002


_DISTANCE_EPSILON = 1e-12
"""Additive guard so unit-vector division and ``grad(norm)`` stay finite at r=0."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeGeometry:
    """Per-edge geometric quantities derived from the padded edge index.

    Attributes:
        senders: Clamped sender indices of shape ``(max_edges,)`` (padded slots
            point at atom ``0``; use :attr:`mask` to identify them).
        receivers: Clamped receiver indices of shape ``(max_edges,)``.
        vectors: Displacement vectors ``r_i - r_j`` of shape ``(max_edges, 3)``.
        lengths: Edge lengths ``|r_i - r_j|`` of shape ``(max_edges, 1)``.
        unit_vectors: Unit displacement directions of shape ``(max_edges, 3)``.
        mask: Boolean validity mask of shape ``(max_edges, 1)`` -- ``True`` for
            real edges, ``False`` for ``-1`` padding.
    """

    senders: Int[Array, " max_edges"]
    receivers: Int[Array, " max_edges"]
    vectors: Float[Array, "max_edges 3"]
    lengths: Float[Array, "max_edges 1"]
    unit_vectors: Float[Array, "max_edges 3"]
    mask: Bool[Array, "max_edges 1"]


def edge_geometry(positions: Float[Array, "n_atoms 3"], graph: tuple[Array, Array]) -> EdgeGeometry:
    """Compute per-edge displacement vectors, lengths and directions.

    Args:
        positions: Atomic coordinates of shape ``(n_atoms, 3)``.
        graph: The ``(senders, receivers)`` padded edge index from
            :func:`opifex.neural.equivariant.radius_graph` (``-1`` padding).

    Returns:
        An :class:`EdgeGeometry` with clamped indices, displacement vectors,
        lengths, unit directions and the validity mask. Padded slots carry zero
        vectors/lengths and a ``False`` mask.
    """
    senders, receivers = graph
    valid = senders >= 0
    safe_senders = jnp.where(valid, senders, 0)
    safe_receivers = jnp.where(valid, receivers, 0)
    vectors = positions[safe_senders] - positions[safe_receivers]
    mask = valid[:, None]
    vectors = jnp.where(mask, vectors, 0.0)
    lengths = jnp.sqrt(jnp.sum(vectors**2, axis=-1, keepdims=True) + _DISTANCE_EPSILON)
    unit_vectors = jnp.where(mask, vectors / lengths, 0.0)
    return EdgeGeometry(
        senders=safe_senders,
        receivers=safe_receivers,
        vectors=vectors,
        lengths=lengths,
        unit_vectors=unit_vectors,
        mask=mask,
    )


__all__ = ["EdgeGeometry", "edge_geometry"]
