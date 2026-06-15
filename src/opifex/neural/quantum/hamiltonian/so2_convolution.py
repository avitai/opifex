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

from opifex.geometry.algebra.wigner import _matrix_to_euler, _wigner_d_from_euler
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
    NormGate,
    rms_normalize,
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


def _edge_unit_vectors(edge_vectors: Float[Array, "edges 3"]) -> Float[Array, "edges 3"]:
    r"""Normalise edge vectors with a grad-safe fallback at zero length.

    Padded / zero-length edges (sentinel direction ``+y``, the quantisation axis)
    keep both the forward value and the backward gradient finite via the
    double-``where`` trick -- a plain ``where`` would still propagate ``NaN`` from
    the unselected ``edge / 0`` branch in the backward pass.

    Args:
        edge_vectors: Per-edge displacement vectors of shape ``(edges, 3)``.

    Returns:
        Per-edge unit direction vectors of shape ``(edges, 3)``.
    """
    norm_sq = jnp.sum(edge_vectors**2, axis=-1, keepdims=True)
    is_zero = norm_sq <= _DISTANCE_EPSILON
    safe_norm_sq = jnp.where(is_zero, jnp.ones_like(norm_sq), norm_sq)
    reference_axis = jnp.array([0.0, 1.0, 0.0], dtype=edge_vectors.dtype)
    return jnp.where(is_zero, reference_axis, edge_vectors / jnp.sqrt(safe_norm_sq))


def _rotate_irreps(
    array: Float[Array, "edges dim"],
    irreps: Irreps,
    alpha: Float[Array, " edges"],
    beta: Float[Array, " edges"],
    gamma: Float[Array, " edges"],
    dtype: jnp.dtype,
) -> Float[Array, "edges dim"]:
    r"""Apply the block-diagonal Wigner-D of per-edge Euler angles to a feature array.

    The eSCN frame rotation is the same per edge for every degree, so the Euler
    angles (extracted once by the caller) are shared across all blocks; each
    block's Wigner-D is the cheap ``Z_l J_l Z_l J_l Z_l`` product
    (:func:`opifex.geometry.algebra.wigner._wigner_d_from_euler`), with no per-edge
    ``expm`` -- the throughput fix over the matrix-exponential path.

    Args:
        array: Flat irreps feature of shape ``(edges, irreps.dim)``.
        irreps: Layout of ``array``.
        alpha: Per-edge first Euler angle (rotation about ``+y``).
        beta: Per-edge second Euler angle (rotation about ``+x``).
        gamma: Per-edge third Euler angle (rotation about ``+y``).
        dtype: Output dtype.

    Returns:
        The rotated feature, same shape and layout as ``array``.
    """
    leading = array.shape[:-1]
    chunks: list[Float[Array, "edges channels dim_l"] | None] = []
    start = 0
    for mul, irrep in irreps.blocks:
        width = mul * irrep.dim
        block = array[..., start : start + width].reshape(*leading, mul, irrep.dim)
        wig = jax.vmap(lambda a, b, g, degree=irrep.l: _wigner_d_from_euler(degree, a, b, g))(
            alpha, beta, gamma
        ).astype(dtype)
        rotated = jnp.einsum("...ij,...uj->...ui", wig, block)
        chunks.append(rotated)
        start += width
    return from_chunks(irreps, chunks, leading, dtype).array


class SO2Linear(nnx.Module):
    r"""Per-order :math:`SO(2)` mixing of in-frame features (eSCN, no rotation).

    The order-diagonal linear map at the heart of the eSCN reduction: given
    features already rotated into the edge frame, it mixes channels within each
    order :math:`m` -- a real map for :math:`m = 0` and a complex map
    :math:`W_1 + i W_2` for the :math:`(+m, -m)` pair, which is the most general
    channel mixing that commutes with every rotation about the quantisation axis
    (Passaro & Zitnick 2023, arXiv:2302.03655; fairchem ``SO2_m_Conv``). Both
    :class:`SO2EdgeConvolution` (a single feature) and
    :class:`SO2PairInteractionLayer` (a concatenated endpoint pair) sandwich this
    map between a rotate-in and a rotate-out.

    Args:
        irreps_in: Layout of the in-frame input feature.
        irreps_out: Desired output layout; every output degree must appear in
            ``irreps_in`` (the mixing maps order to order, degree to itself).
        max_order: Highest order ``m`` to mix.
        rngs: Random number generators (keyword-only); ``rngs.params()`` seeds the
            per-order mixing weights.
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str,
        *,
        max_order: int,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the per-order SO(2) mixing weights and the static column plans."""
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.max_order = max_order
        self._in_plan = _PerOrderPlan(self.irreps_in, max_order)
        self._out_plan = _PerOrderPlan(self.irreps_out, max_order)

        # Per-order learnable mixings: real (m = 0) and complex (m > 0, two real
        # parts) linear maps from the input channels of order m to the output
        # channels of order m. Initialised at unit-ish scale (fan-in normalised).
        key = rngs.params()
        real_weights: list[nnx.Param] = []
        complex_weights: list[nnx.Param] = []
        for m in range(max_order + 1):
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

    def __call__(self, array: Float[Array, "edges dim"]) -> Float[Array, "edges dim_out"]:
        """Mix an in-frame feature array per order ``m`` into ``irreps_out``."""
        negatives, positives = self._gather_orders(array)
        return self._mix_orders(negatives, positives, array.dtype)


class SO2EdgeConvolution(nnx.Module):
    r"""eSCN SO(2)-frame edge convolution (drop-in for the SO(3) edge tensor product).

    Computes an equivariant edge message from a node feature and the edge vector
    by rotating into the edge-aligned frame, mixing per order :math:`m` with the
    cheap :class:`SO2Linear` operation of eSCN (arXiv:2302.03655) / QHNetV2
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
        """Build the per-order SO(2) mixing and record the static order cap."""
        super().__init__()
        if sh_lmax < 0:
            raise ValueError(f"sh_lmax must be non-negative, got {sh_lmax}")
        self.irreps_in1 = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out) if irreps_out is not None else self.irreps_in1

        in_lmax = max((irrep.l for _, irrep in self.irreps_in1.blocks), default=0)
        out_lmax = max((irrep.l for _, irrep in self.irreps_out.blocks), default=0)
        self.max_order = min(sh_lmax, in_lmax, out_lmax)

        # The dense tensor product the predictor uses also names a second input
        # (the spherical harmonics); for the SO(2) frame that input is replaced by
        # the edge direction, so we expose ``irreps_in2`` for protocol parity.
        from opifex.neural.equivariant import spherical_harmonics  # local: read-only sibling

        self.irreps_in2 = spherical_harmonics(sh_lmax, jnp.zeros((1, 3))).irreps
        self.so2_linear = SO2Linear(
            self.irreps_in1, self.irreps_out, max_order=self.max_order, rngs=rngs
        )

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
        unit_vectors = _edge_unit_vectors(edge_vectors)
        rotations = _edge_frame_rotation(unit_vectors).astype(dtype)
        alpha, beta, gamma = jax.vmap(_matrix_to_euler)(rotations)
        inverse_alpha, inverse_beta, inverse_gamma = jax.vmap(_matrix_to_euler)(
            jnp.swapaxes(rotations, -1, -2)
        )

        rotated_in = _rotate_irreps(x.array, self.irreps_in1, alpha, beta, gamma, dtype)
        mixed = self.so2_linear(rotated_in)
        rotated_out = _rotate_irreps(
            mixed, self.irreps_out, inverse_alpha, inverse_beta, inverse_gamma, dtype
        )
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


class SO2PairInteractionLayer(nnx.Module):
    r"""eSCN SO(2)-frame off-diagonal pair refinement (replaces the O(L^3) node(x)node CG).

    QHNetV2's reduction of QHNet's ``PairNetLayer`` to SO(2) local frames (Yu et
    al. 2025, "Efficient Prediction of SO(3)-Equivariant Hamiltonian Matrices via
    SO(2) Local Frames", arXiv:2506.09398; reference OrbEvo
    ``../AIRS/OpenDFT/OrbEvo/orbevo/models/orbevo/{so2_ops.py,
    transformer_block_dm.py}``). For a directed edge ``i -> j`` the two endpoint
    node features are rotated into the ``i -> j`` edge frame, concatenated
    channel-wise, and coupled by an in-frame ``SO2Linear -> gate -> SO2Linear``
    (``O(L^2)`` per edge), then rotated back to the global frame and residually
    accumulated onto the running off-diagonal-block feature.

    This replaces QHNet's dense channel-wise Clebsch-Gordan ``tp(x[src], x[dst])``
    -- the dominant cost of the block predictor (the complete edge graph times an
    ``O(L^3)`` product) -- with the cheap order-diagonal SO(2) operations, while
    staying SO(3)-equivariant: the frame co-rotates with the geometry, so rotating
    every node feature **and** the edge vectors by ``R`` rotates the per-edge
    output by ``R``. The directed frame (``i -> j`` differs from ``j -> i``) gives
    the genuine off-diagonal asymmetry the Fock block needs.

    Args:
        irreps: The (all-even) node / per-edge output feature layout.
        sh_lmax: Maximum spherical-harmonic degree (caps the SO(2) order).
        edge_radial_dim: Width of the per-edge radial embedding.
        weight_hidden_dim: Hidden width of the per-edge radial weight MLP.
        rngs: Random number generators (keyword-only) seeding the weights.

    Raises:
        ValueError: If ``sh_lmax`` is negative.
    """

    def __init__(
        self,
        irreps: Irreps | str,
        *,
        sh_lmax: int,
        edge_radial_dim: int,
        weight_hidden_dim: int = 64,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the concat SO(2) coupling, radial modulation and output gate."""
        super().__init__()
        if sh_lmax < 0:
            raise ValueError(f"sh_lmax must be non-negative, got {sh_lmax}")
        self.irreps = Irreps(irreps)
        self._gate_irreps = _gate_input_irreps(self.irreps)
        # Channel-wise concatenation of the two rotated endpoints: the in-frame
        # message carries 2x the multiplicity of each degree (OrbEvo
        # ``cat((x_source, x_target), dim=channel)``).
        self._concat_irreps = Irreps(tuple((2 * mul, irrep) for mul, irrep in self.irreps.blocks))
        lmax = max((irrep.l for _, irrep in self.irreps.blocks), default=0)
        self.max_order = min(sh_lmax, lmax)
        self.so2_in = SO2Linear(
            self._concat_irreps, self._gate_irreps, max_order=self.max_order, rngs=rngs
        )
        self.so2_out = SO2Linear(self.irreps, self.irreps, max_order=self.max_order, rngs=rngs)
        # Per-edge radial modulation of the in-frame message (eSCN m-share radial).
        self.radial_hidden = nnx.Linear(edge_radial_dim, weight_hidden_dim, rngs=rngs)
        self.radial_out = nnx.Linear(weight_hidden_dim, self._gate_irreps.num_irreps, rngs=rngs)
        self.gate_out = NormGate(self.irreps, rngs=rngs)
        self.linear_out = EquivariantLinear(self.irreps, self.irreps, rngs=rngs)

    def _concatenate_endpoints(
        self,
        source: Float[Array, "edges dim"],
        target: Float[Array, "edges dim"],
        dtype: jnp.dtype,
    ) -> Float[Array, "edges concat_dim"]:
        """Concatenate two in-frame endpoint features channel-wise per degree."""
        leading = source.shape[:-1]
        source_chunks = IrrepsArray(self.irreps, source).chunks
        target_chunks = IrrepsArray(self.irreps, target).chunks
        concat_chunks: list[Float[Array, "edges channels dim_l"] | None] = [
            jnp.concatenate([s, t], axis=-2)
            for s, t in zip(source_chunks, target_chunks, strict=True)
        ]
        return from_chunks(self._concat_irreps, concat_chunks, leading, dtype).array

    def __call__(
        self,
        node_features: IrrepsArray,
        geometry: EdgeGeometry,
        edge_radial: Float[Array, "max_edges num_radial_basis"],
        accumulated: IrrepsArray | None = None,
    ) -> IrrepsArray:
        """Return the refined per-edge off-diagonal feature, optionally accumulated.

        Args:
            node_features: Per-atom feature ``(n_atoms, irreps.dim)``.
            geometry: Edge geometry carrying the ``(senders, receivers)`` indices
                and per-edge displacement ``vectors`` defining each frame.
            edge_radial: ``(max_edges, num_radial_basis)`` per-edge radial embedding
                (already envelope-modulated by the caller).
            accumulated: Running off-diagonal-block feature from earlier layers
                (added residually), or ``None`` for the first refinement layer.

        Returns:
            The refined per-edge feature ``(max_edges, irreps.dim)``.
        """
        # Bound the (unnormalised trunk) magnitudes before the coupling; the
        # norm-based scaling is rotation-invariant, so equivariance is preserved.
        node_features = rms_normalize(node_features)
        dtype = node_features.array.dtype
        unit_vectors = _edge_unit_vectors(geometry.vectors)
        rotations = _edge_frame_rotation(unit_vectors).astype(dtype)
        alpha, beta, gamma = jax.vmap(_matrix_to_euler)(rotations)
        inverse_alpha, inverse_beta, inverse_gamma = jax.vmap(_matrix_to_euler)(
            jnp.swapaxes(rotations, -1, -2)
        )

        # Rotate both endpoints into the i->j edge frame, then concatenate.
        source = _rotate_irreps(
            node_features.array[geometry.senders], self.irreps, alpha, beta, gamma, dtype
        )
        target = _rotate_irreps(
            node_features.array[geometry.receivers], self.irreps, alpha, beta, gamma, dtype
        )
        message = self._concatenate_endpoints(source, target, dtype)

        # In-frame coupling: SO2Linear -> radial modulate -> gate -> SO2Linear.
        message = self.so2_in(message)
        radial_weights = self.radial_out(jax.nn.silu(self.radial_hidden(edge_radial)))
        message = apply_scalar_weights(IrrepsArray(self._gate_irreps, message), radial_weights)
        coupled = self.so2_out(gate(message).array)

        # Rotate back to the global frame, then the QHNet output gate + linear.
        coupled = _rotate_irreps(
            coupled, self.irreps, inverse_alpha, inverse_beta, inverse_gamma, dtype
        )
        refined = self.linear_out(self.gate_out(IrrepsArray(self.irreps, coupled)))
        if accumulated is not None:
            refined = IrrepsArray(self.irreps, refined.array + accumulated.array)
        return refined
