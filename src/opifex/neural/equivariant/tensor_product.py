r"""Equivariant tensor product of two steerable features via Clebsch-Gordan.

The tensor product couples two :class:`IrrepsArray`\ s into a third by contracting
each pair of input blocks with the real Clebsch-Gordan tensor.  For input irreps
``ir_1`` and ``ir_2``, the selection rule
``|l_1 - l_2| <= l_3 <= l_1 + l_2`` (with parity ``p_1 p_2``) lists the allowed
output irreps; the contraction
``einsum("...ui,...vj,ijk->...uvk", x_1, x_2, C)`` (with ``C`` the
Clebsch-Gordan tensor) is equivariant by construction.

Ported from ``e3nn-jax`` (``../e3nn-jax/e3nn_jax/_src/tensor_products.py:122``
and the ``uvw`` connection of
``../e3nn-jax/e3nn_jax/_src/legacy/core_tensor_product.py``).  The
*fully-connected* variant gives every path ``(i_1, i_2) -> i_3`` a learnable
``(mul_1, mul_2, mul_3)`` weight:
``einsum("uvw,ijk,...ui,...vj->...wk", w, C, x_1, x_2)``.  Normalisation follows
the e3nn default config (``irrep_normalization="component"``,
``path_normalization="element"``): the per-path scale
``alpha = dim(ir_3) / sum_paths(mul_1 mul_2)`` (cf.
``_normalize_instruction_path_weights``) is folded into the weight
initialisation standard deviation, so the forward pass uses the raw
Clebsch-Gordan tensor.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.geometry.algebra.wigner import clebsch_gordan
from opifex.neural.equivariant._assembly import from_chunks
from opifex.neural.equivariant.irreps import Irrep, Irreps, IrrepsArray


def _to_nested_tuple(value: object) -> object:
    """Convert an array to nested (hashable) Python tuples of floats.

    The result is array-free, so it lives in ``nnx`` static aux-data rather than
    being treated as a parameter leaf.
    """
    listed = np.asarray(value).tolist()

    def _freeze(item: object) -> object:
        return tuple(_freeze(sub) for sub in item) if isinstance(item, list) else item

    return _freeze(listed)


@runtime_checkable
class TensorProduct(Protocol):
    r"""Protocol for an equivariant bilinear map of two :class:`IrrepsArray`\ s.

    Concrete implementations (the dense Clebsch-Gordan
    :class:`FullyConnectedTensorProduct`, or a future Cartesian /
    cuEquivariance-backed variant) expose the same call signature so they are
    substitutable.
    """

    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps

    def __call__(self, x: IrrepsArray, y: IrrepsArray) -> IrrepsArray:
        """Combine two equivariant features into the output layout."""
        ...


class FullyConnectedTensorProduct(nnx.Module):
    """Fully-connected Clebsch-Gordan tensor product with learnable path weights.

    Every selection-rule-allowed path ``(i_1, i_2) -> i_3`` whose output irrep is
    present in ``irreps_out`` carries an independent ``(mul_1, mul_2, mul_3)``
    weight; the contributions are summed per output block.
    """

    def __init__(
        self,
        irreps_in1: Irreps | str,
        irreps_in2: Irreps | str,
        irreps_out: Irreps | str,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the tensor product and its per-path weights.

        Args:
            irreps_in1: Layout of the first input.
            irreps_in2: Layout of the second input.
            irreps_out: Desired output layout; only paths producing an irrep that
                appears here are instantiated.
            rngs: Random number generators (keyword-only); ``rngs.params()``
                seeds the path weights.
        """
        super().__init__()
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)

        in1_muls = [mul for mul, _ in self.irreps_in1.blocks]
        in2_muls = [mul for mul, _ in self.irreps_in2.blocks]
        out_lookup: dict[Irrep, list[int]] = {}
        for out_index, (_, irrep_out) in enumerate(self.irreps_out.blocks):
            out_lookup.setdefault(irrep_out, []).append(out_index)

        # Enumerate paths and the per-output path-normalisation sums.
        raw_paths: list[tuple[int, int, int, tuple[int, int, int]]] = []
        path_norm_sum: dict[int, float] = {}
        for i1, (_, ir1) in enumerate(self.irreps_in1.blocks):
            for i2, (_, ir2) in enumerate(self.irreps_in2.blocks):
                for ir_out in Irrep(ir1) * Irrep(ir2):
                    for out_index in out_lookup.get(ir_out, []):
                        ls = (ir1.l, ir2.l, ir_out.l)
                        raw_paths.append((i1, i2, out_index, ls))
                        path_norm_sum[out_index] = (
                            path_norm_sum.get(out_index, 0.0) + in1_muls[i1] * in2_muls[i2]
                        )

        weights: list[nnx.Param] = []
        in1_idx: list[int] = []
        in2_idx: list[int] = []
        out_idx: list[int] = []
        # Clebsch-Gordan tensors are static constants; stored as nested Python
        # tuples (not arrays) so they live in nnx static aux-data, then rebuilt
        # as compile-time-constant arrays in ``__call__``.
        couplings: list[object] = []
        key = rngs.params()
        for i1, i2, out_index, (l1, l2, l3) in raw_paths:
            mul_out = self.irreps_out.blocks[out_index][0]
            out_dim = 2 * l3 + 1
            # e3nn "component" irrep norm + "element" path norm folded into init std.
            alpha = out_dim / path_norm_sum[out_index]
            weight_std = math.sqrt(alpha)
            key, subkey = jax.random.split(key)
            shape = (in1_muls[i1], in2_muls[i2], mul_out)
            weights.append(nnx.Param(weight_std * jax.random.normal(subkey, shape)))
            couplings.append(_to_nested_tuple(clebsch_gordan(l1, l2, l3)))
            in1_idx.append(i1)
            in2_idx.append(i2)
            out_idx.append(out_index)

        self.weights = nnx.List(weights)
        self._couplings = tuple(couplings)
        self._path_in1 = tuple(in1_idx)
        self._path_in2 = tuple(in2_idx)
        self._path_out = tuple(out_idx)

    def __call__(self, x: IrrepsArray, y: IrrepsArray) -> IrrepsArray:
        """Apply the tensor product.

        Args:
            x: First input with ``x.irreps == self.irreps_in1``.
            y: Second input with ``y.irreps == self.irreps_in2``; ``x`` and ``y``
                share their leading (batch) dimensions.

        Returns:
            An :class:`IrrepsArray` with ``self.irreps_out``.

        Raises:
            ValueError: If either input's irreps do not match the configured ones.
        """
        if x.irreps != self.irreps_in1:
            raise ValueError(
                f"FullyConnectedTensorProduct expected first input irreps "
                f"{self.irreps_in1!r}, got {x.irreps!r}"
            )
        if y.irreps != self.irreps_in2:
            raise ValueError(
                f"FullyConnectedTensorProduct expected second input irreps "
                f"{self.irreps_in2!r}, got {y.irreps!r}"
            )
        chunks_x = x.chunks
        chunks_y = y.chunks
        leading_shape = jnp.broadcast_shapes(x.array.shape[:-1], y.array.shape[:-1])
        dtype = jnp.result_type(x.array.dtype, y.array.dtype)
        accumulators: list[jax.Array | None] = [None] * len(self.irreps_out.blocks)
        for path in range(len(self._path_out)):
            i1 = self._path_in1[path]
            i2 = self._path_in2[path]
            out_index = self._path_out[path]
            weight = self.weights[path][...]
            coupling = jnp.asarray(self._couplings[path], dtype=dtype)
            contribution = jnp.einsum(
                "uvw,ijk,...ui,...vj->...wk",
                weight.astype(dtype),
                coupling,
                chunks_x[i1].astype(dtype),
                chunks_y[i2].astype(dtype),
            )
            current = accumulators[out_index]
            accumulators[out_index] = contribution if current is None else current + contribution
        return from_chunks(self.irreps_out, accumulators, leading_shape, dtype)
