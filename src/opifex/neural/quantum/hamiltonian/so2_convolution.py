r"""eSCN SO(2)-frame edge convolution for the equivariant Hamiltonian predictor.

This module implements the *eSCN* reduction of an :math:`SO(3)` edge tensor
product to a set of per-order :math:`SO(2)` operations (Passaro & Zitnick 2023,
"Reducing SO(3) Convolutions to SO(2)", arXiv:2302.03655), as adopted by QHNetV2
for scalable equivariant Hamiltonian prediction (Yu et al. 2023,
arXiv:2306.04922). It is a drop-in replacement for the dense
:class:`opifex.neural.equivariant.tensor_product.FullyConnectedTensorProduct`
used as the ``edge_tensor_product`` of
:class:`opifex.neural.quantum.hamiltonian.predictor.HamiltonianPredictor`:
both consume a node feature and an edge geometry and emit an
:class:`~opifex.neural.equivariant.IrrepsArray` in ``irreps_out``.

The idea
--------
A full Clebsch-Gordan tensor product of a feature with the spherical harmonics
of an edge costs :math:`O(L^3)` per edge. eSCN observes that if the edge vector
is first rotated onto the reference (quantisation) axis, the harmonics of the
edge collapse to their :math:`m = 0` components, and the Clebsch-Gordan coupling
:math:`C[l_1, m_1; l_2, 0; l_3, m_3]` is non-zero only when :math:`m_1 = m_3`.
The :math:`SO(3)` tensor product therefore becomes block-diagonal in the order
:math:`m`, and reduces to an :math:`SO(2)` convolution -- an independent linear
mixing within each :math:`\pm m` pair -- costing :math:`O(L^2)` per edge.

Concretely, for each edge:

#. Build the rotation :math:`R` that aligns the edge direction with the
   reference axis (in opifex's real basis -- shared with
   :func:`opifex.geometry.algebra.wigner.wigner_d` and
   :func:`opifex.neural.equivariant.spherical_harmonics.spherical_harmonics` --
   the quantisation / :math:`m = 0` axis is :math:`+y`; cf. fairchem's
   ``init_edge_rot_euler_angles`` which uses ``beta = acos(y)``,
   ``../fairchem/src/fairchem/core/models/uma/common/rotation.py``).
#. Rotate the node feature into that edge frame with the Wigner-D matrices
   :math:`D^l(R)`.
#. Apply the per-order :math:`SO(2)` mixing. Under a rotation about :math:`y`,
   each :math:`\pm m` pair transforms as a 2D rotation by :math:`m\theta`; a
   channel/degree mixing commutes with *every* such rotation iff it acts as a
   complex-linear map :math:`(W_1 + i W_2)` on that pair (the real :math:`m = 0`
   subspace mixes with an ordinary real linear map). This is exactly fairchem's
   ``SO2_m_Conv`` (``../fairchem/src/fairchem/core/models/uma/nn/so2_layers.py``)
   and the QHNet edge update (``../AIRS/OpenDFT/QHBench/QH9/models/QHNet.py``),
   expressed here over opifex irreps.
#. Rotate the result back out of the edge frame with :math:`D^l(R^{\top})`.

Because the frame co-rotates with the geometry, the whole map is exactly
:math:`SO(3)`-equivariant: :math:`f(D(R)x, Rv) = D(R) f(x, v)`.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.atomistic.backbones._message_passing import EdgeGeometry  # noqa: TC001
from opifex.neural.atomistic.backbones.nequip import (
    _gate_input_irreps,
    _RadialNetwork,
    NequIPConfig,
)
from opifex.neural.equivariant import (
    apply_scalar_weights,
    EquivariantLinear,
    gate,
    scatter_sum,
)
from opifex.neural.equivariant._assembly import from_chunks
from opifex.neural.equivariant.irreps import Irreps, IrrepsArray


logger = logging.getLogger(__name__)

_DISTANCE_EPSILON = 1e-12
"""Additive guard so the edge-frame construction stays finite at zero length."""


def _edge_frame_rotation(
    unit_vectors: Float[Array, "edges 3"],
) -> Float[Array, "edges 3 3"]:
    r"""Rotation matrices aligning each unit edge vector with the ``+y`` axis.

    Returns per-edge ``R`` with ``R @ u`` close to ``(0, 1, 0)``. The frame is an
    orthonormal basis ``(e1, u, e2)`` built by Gram-Schmidt against a reference
    helper axis, chosen per edge to avoid the degeneracy at ``u || helper``
    (cf. the local-frame construction in eSCN, arXiv:2302.03655 §3). ``R`` is the
    transpose of ``[e1 | u | e2]`` so that ``R u = e_y``.

    Args:
        unit_vectors: Per-edge unit direction vectors of shape ``(edges, 3)``.

    Returns:
        Per-edge rotation matrices of shape ``(edges, 3, 3)``.
    """
    # Pick, per edge, the global axis least parallel to u as the Gram-Schmidt seed.
    abs_components = jnp.abs(unit_vectors)
    least_aligned = jnp.argmin(abs_components, axis=-1)
    helper = jax.nn.one_hot(least_aligned, 3, dtype=unit_vectors.dtype)

    # e1 = normalize(helper - (helper . u) u);  e2 = e1 x u.  (e1, u, e2) is a
    # right-handed orthonormal frame (det = +1), so R is a proper rotation.
    projection = jnp.sum(helper * unit_vectors, axis=-1, keepdims=True)
    e1 = helper - projection * unit_vectors
    # Double-where so the sqrt never sees 0 (degenerate helper||u): both the
    # forward value and the backward gradient stay finite (the argmin seed makes
    # ||e1|| >= sqrt(2/3) in practice, but stay defensive for grad safety).
    e1_sq = jnp.sum(e1**2, axis=-1, keepdims=True)
    e1_safe = jnp.where(e1_sq > _DISTANCE_EPSILON, e1_sq, jnp.ones_like(e1_sq))
    e1 = e1 / jnp.sqrt(e1_safe)
    e2 = jnp.cross(e1, unit_vectors)

    # Rows of R are (e1, u, e2) -> R u = (0, 1, 0).
    return jnp.stack([e1, unit_vectors, e2], axis=-2)


class _PerOrderPlan:
    """Static bookkeeping: which (block, degree) feature columns carry each order ``m``.

    For each order ``m`` it records, for the negative (``l - m``) and positive
    (``l + m``) components, the flat feature-array column indices and the channel
    grouping needed to apply the per-``m`` mixing. Stored as Python tuples so the
    plan lives in ``nnx`` static aux-data and stays ``jit``-stable.
    """

    def __init__(self, irreps: Irreps, max_order: int) -> None:
        """Index the feature layout by order ``m`` for ``m = 0 .. max_order``."""
        self.max_order = max_order
        # columns_pos[m], columns_neg[m]: flat indices of the +m and -m components,
        # one per (block, multiplicity) channel carrying order m.
        columns_pos: list[list[int]] = [[] for _ in range(max_order + 1)]
        columns_neg: list[list[int]] = [[] for _ in range(max_order + 1)]
        start = 0
        for mul, irrep in irreps.blocks:
            degree = irrep.l
            dim = irrep.dim
            for channel in range(mul):
                base = start + channel * dim
                center = base + degree  # the m = 0 component sits at the centre.
                columns_pos[0].append(center)
                columns_neg[0].append(center)  # mirror; only +0 used for m = 0.
                for m in range(1, min(degree, max_order) + 1):
                    columns_neg[m].append(base + degree - m)
                    columns_pos[m].append(base + degree + m)
            start += mul * dim
        self.columns_pos = tuple(tuple(cols) for cols in columns_pos)
        self.columns_neg = tuple(tuple(cols) for cols in columns_neg)
        self.channels_per_order = tuple(len(cols) for cols in columns_pos)


class SO2EdgeConvolution(nnx.Module):
    r"""eSCN SO(2)-frame edge convolution (drop-in for the SO(3) edge tensor product).

    Computes an equivariant edge message from a node feature and the edge vector
    by rotating into the edge-aligned frame, mixing per order :math:`m` with the
    cheap :math:`SO(2)` operations of eSCN (arXiv:2302.03655) / QHNetV2
    (arXiv:2306.04922), and rotating back. The call signature mirrors the
    :class:`opifex.neural.equivariant.tensor_product.TensorProduct` protocol so it
    substitutes directly for
    :class:`~opifex.neural.equivariant.tensor_product.FullyConnectedTensorProduct`
    as the predictor's ``edge_tensor_product``.

    Args:
        irreps_in: Layout of the input node feature (and, by default, the output).
        sh_lmax: Maximum spherical-harmonic degree of the edge embedding the dense
            tensor product would have consumed; it caps the order ``m`` of the
            :math:`SO(2)` mixing at ``min(sh_lmax, lmax(irreps))``.
        irreps_out: Desired output layout. Defaults to ``irreps_in``. Every output
            degree must also appear in ``irreps_in`` (the :math:`SO(2)` mixing maps
            order to order, degree to itself).
        rngs: Random number generators (keyword-only); ``rngs.params()`` seeds the
            per-order mixing weights.
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        *,
        sh_lmax: int,
        irreps_out: Irreps | str | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the per-order SO(2) mixing weights and the static column plan."""
        super().__init__()
        self.irreps_in1 = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out) if irreps_out is not None else self.irreps_in1

        in_lmax = max((irrep.l for _, irrep in self.irreps_in1.blocks), default=0)
        out_lmax = max((irrep.l for _, irrep in self.irreps_out.blocks), default=0)
        self.max_order = min(sh_lmax, in_lmax, out_lmax)
        if sh_lmax < 0:
            raise ValueError(f"sh_lmax must be non-negative, got {sh_lmax}")

        # The dense tensor product the predictor uses also names a second input
        # (the spherical harmonics); for the SO(2) frame that input is replaced by
        # the edge direction, so we expose ``irreps_in2`` for protocol parity.
        from opifex.neural.equivariant import spherical_harmonics  # local: read-only sibling

        self.irreps_in2 = spherical_harmonics(sh_lmax, jnp.zeros((1, 3))).irreps

        self._in_plan = _PerOrderPlan(self.irreps_in1, self.max_order)
        self._out_plan = _PerOrderPlan(self.irreps_out, self.max_order)

        # Per-order learnable mixings: real (m = 0) and complex (m > 0, two real
        # parts) linear maps from the input channels of order m to the output
        # channels of order m. Initialised at unit-ish scale (fan-in normalised).
        key = rngs.params()
        real_weights: list[nnx.Param] = []
        complex_weights: list[nnx.Param] = []
        for m in range(self.max_order + 1):
            fan_in = max(self._in_plan.channels_per_order[m], 1)
            scale = 1.0 / (fan_in**0.5)
            shape = (self._in_plan.channels_per_order[m], self._out_plan.channels_per_order[m])
            if m == 0:
                key, subkey = jax.random.split(key)
                real_weights.append(nnx.Param(scale * jax.random.normal(subkey, shape)))
            else:
                key, sub1 = jax.random.split(key)
                key, sub2 = jax.random.split(key)
                # Two real matrices form one complex weight W1 + i W2; the 1/sqrt(2)
                # keeps the complex multiplication variance-preserving (cf. fairchem
                # SO2_m_Conv ``fc.weight.data.mul_(1/sqrt(2))``).
                complex_weights.append(
                    nnx.Param(
                        scale
                        / (2.0**0.5)
                        * jnp.stack(
                            [jax.random.normal(sub1, shape), jax.random.normal(sub2, shape)]
                        )
                    )
                )
        self._real_weights = nnx.List(real_weights)
        self._complex_weights = nnx.List(complex_weights)

    def _gather_orders(
        self, array: Float[Array, "edges dim"]
    ) -> tuple[list[Float[Array, "edges channels"]], list[Float[Array, "edges channels"]]]:
        """Split a feature array into per-order ``(negative, positive)`` component stacks."""
        negatives: list[Float[Array, "edges channels"]] = []
        positives: list[Float[Array, "edges channels"]] = []
        for m in range(self.max_order + 1):
            cols_pos = jnp.asarray(self._in_plan.columns_pos[m], dtype=jnp.int32)
            cols_neg = jnp.asarray(self._in_plan.columns_neg[m], dtype=jnp.int32)
            positives.append(array[..., cols_pos])
            negatives.append(array[..., cols_neg])
        return negatives, positives

    def _mix_orders(
        self,
        negatives: list[Float[Array, "edges channels"]],
        positives: list[Float[Array, "edges channels"]],
        dtype: jnp.dtype,
    ) -> Float[Array, "edges dim"]:
        r"""Apply the per-order SO(2) mixing and scatter back into the output layout.

        For ``m = 0`` the central components mix with a real matrix. For ``m > 0``
        the ``(+m, -m)`` pair is treated as a complex number ``z = pos + i neg``
        and multiplied by the complex weight ``W1 + i W2``; the real/imaginary
        parts are written to the ``+m`` / ``-m`` output columns. Complex
        multiplication commutes with the 2D rotation each pair undergoes, so the
        map is :math:`SO(2)`-equivariant.
        """
        leading = positives[0].shape[:-1]
        output = jnp.zeros((*leading, self.irreps_out.dim), dtype=dtype)

        # m = 0: real linear mixing of the central components.
        mixed_zero = positives[0].astype(dtype) @ self._real_weights[0][...].astype(dtype)
        cols_zero = jnp.asarray(self._out_plan.columns_pos[0], dtype=jnp.int32)
        output = output.at[..., cols_zero].set(mixed_zero)

        # m > 0: complex linear mixing of each +-m pair.
        for m in range(1, self.max_order + 1):
            weight = self._complex_weights[m - 1][...].astype(dtype)
            w_real, w_imag = weight[0], weight[1]
            pos = positives[m].astype(dtype)
            neg = negatives[m].astype(dtype)
            out_pos = pos @ w_real - neg @ w_imag
            out_neg = pos @ w_imag + neg @ w_real
            cols_pos = jnp.asarray(self._out_plan.columns_pos[m], dtype=jnp.int32)
            cols_neg = jnp.asarray(self._out_plan.columns_neg[m], dtype=jnp.int32)
            output = output.at[..., cols_pos].set(out_pos)
            output = output.at[..., cols_neg].set(out_neg)
        return output

    def _rotate(
        self, array: Float[Array, "edges dim"], irreps: Irreps, rotations: Float[Array, "edges 3 3"]
    ) -> Float[Array, "edges dim"]:
        """Apply the block-diagonal Wigner-D of per-edge ``rotations`` to a feature array."""
        leading = array.shape[:-1]
        chunks: list[Float[Array, "edges channels dim_l"] | None] = []
        start = 0
        dtype = array.dtype
        for mul, irrep in irreps.blocks:
            width = mul * irrep.dim
            block = array[..., start : start + width].reshape(*leading, mul, irrep.dim)
            # ``wigner_d`` takes a single 3x3 rotation; vmap it over the edge axis.
            wig = jax.vmap(lambda r, degree=irrep.l: wigner_d(degree, r))(rotations).astype(dtype)
            rotated = jnp.einsum("...ij,...uj->...ui", wig, block)
            chunks.append(rotated)
            start += width
        return from_chunks(irreps, chunks, leading, dtype).array

    def __call__(self, x: IrrepsArray, edge_vectors: Float[Array, "edges 3"]) -> IrrepsArray:
        """Compute the eSCN SO(2)-frame edge message.

        Args:
            x: Node feature with ``x.irreps == self.irreps_in1`` and a leading edge
                axis (shape ``(edges, irreps_in.dim)``).
            edge_vectors: Per-edge displacement vectors of shape ``(edges, 3)``;
                their direction defines the local frame.

        Returns:
            An :class:`~opifex.neural.equivariant.IrrepsArray` with
            ``self.irreps_out``.

        Raises:
            ValueError: If ``x.irreps`` does not match the configured input layout.
        """
        if x.irreps != self.irreps_in1:
            raise ValueError(
                f"SO2EdgeConvolution expected input irreps {self.irreps_in1!r}, got {x.irreps!r}"
            )
        dtype = x.array.dtype
        # Double-where normalisation: sqrt never sees 0, so the gradient through
        # padded / zero-length edges is finite (a plain ``where`` would still
        # propagate NaN from the unselected ``edge/0`` branch in the backward pass).
        norm_sq = jnp.sum(edge_vectors**2, axis=-1, keepdims=True)
        is_zero = norm_sq <= _DISTANCE_EPSILON
        safe_norm_sq = jnp.where(is_zero, jnp.ones_like(norm_sq), norm_sq)
        reference_axis = jnp.array([0.0, 1.0, 0.0], dtype=edge_vectors.dtype)
        unit_vectors = jnp.where(is_zero, reference_axis, edge_vectors / jnp.sqrt(safe_norm_sq))

        rotations = _edge_frame_rotation(unit_vectors).astype(dtype)
        inverse_rotations = jnp.swapaxes(rotations, -1, -2)

        rotated_in = self._rotate(x.array, self.irreps_in1, rotations)
        negatives, positives = self._gather_orders(rotated_in)
        mixed = self._mix_orders(negatives, positives, dtype)
        rotated_out = self._rotate(mixed, self.irreps_out, inverse_rotations)
        return IrrepsArray(self.irreps_out, rotated_out)


class SO2ConvolutionLayer(nnx.Module):
    r"""NequIP-style message-passing layer whose edge message is the eSCN SO(2) conv.

    Identical in structure to
    :class:`~opifex.neural.atomistic.backbones.nequip._ConvolutionLayer` -- radial
    modulation, neighbour-sum aggregation, equivariant gate and a residual
    self-interaction -- but the ``O(L^3)`` dense ``node (x) Y(edge)`` tensor product
    is replaced by the ``O(L^2)`` :class:`SO2EdgeConvolution` (QHNetV2, arXiv
    2506.09398). Drop-in for the Hamiltonian predictor's convolution trunk.

    Args:
        node_irreps: Per-atom feature layout (input and output of the layer).
        sh_lmax: Maximum spherical-harmonic degree the dense product would have used
            (caps the SO(2) order).
        config: The shared :class:`NequIPConfig` (radial width, neighbour norm).
        rngs: Random number generators (keyword-only).
    """

    def __init__(
        self, node_irreps: Irreps, sh_lmax: int, config: NequIPConfig, *, rngs: nnx.Rngs
    ) -> None:
        """Build the SO(2) edge convolution, radial network and self-interaction."""
        super().__init__()
        self.node_irreps = node_irreps
        self._gate_irreps = _gate_input_irreps(node_irreps)
        self.so2_conv = SO2EdgeConvolution(
            node_irreps, sh_lmax=sh_lmax, irreps_out=self._gate_irreps, rngs=rngs
        )
        self.radial_network = _RadialNetwork(config, self._gate_irreps.num_irreps, rngs=rngs)
        self.self_interaction = EquivariantLinear(node_irreps, node_irreps, rngs=rngs)
        self.average_num_neighbors = config.average_num_neighbors

    def __call__(
        self,
        node_features: IrrepsArray,
        geometry: EdgeGeometry,
        radial: Float[Array, "max_edges num_radial_basis"],
        envelope: Float[Array, "max_edges 1"],
        num_atoms: int,
    ) -> IrrepsArray:
        """Return the post-gate node features after one SO(2) convolution layer."""
        sender_features = IrrepsArray(node_features.irreps, node_features.array[geometry.senders])
        message = self.so2_conv(sender_features, geometry.vectors)
        weights = self.radial_network(radial) * envelope
        message = apply_scalar_weights(message, weights)
        aggregated = scatter_sum(message.array, geometry.receivers, num_segments=num_atoms)
        aggregated = aggregated / jnp.sqrt(self.average_num_neighbors)
        gated = gate(IrrepsArray(self._gate_irreps, aggregated))
        self_connection = self.self_interaction(node_features)
        return IrrepsArray(self.node_irreps, gated.array + self_connection.array)
