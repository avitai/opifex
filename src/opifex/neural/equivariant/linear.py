r"""Equivariance-preserving linear layer ``Irreps_in -> Irreps_out``.

A linear map between steerable feature spaces is equivariant *iff* it only mixes
the multiplicities of input and output irreps that share the same ``(l, p)`` (no
cross-``l`` mixing, no bias on non-scalars).  Within each such irrep, the map is
an arbitrary learnable mixing matrix over multiplicities.

Ported from ``e3nn-jax``'s ``FunctionalLinear``
(``../e3nn-jax/e3nn_jax/_src/linear.py:22``).  Each path ``(i_in -> i_out)`` with
``ir_in == ir_out`` carries a weight block of shape ``(mul_in, mul_out)`` applied
as ``einsum("uw,...ui->...wi", w, x_chunk)`` (reference line 187).  Following the
e3nn default config (``path_normalization`` / ``gradient_normalization`` =
``"element"`` = ``0``), the per-path scale ``alpha = 1 / sum(mul_in)`` over all
input blocks feeding a given output block (reference lines 72-89) is folded into
the weight initialisation standard deviation, so the forward pass is an unscaled
``einsum`` -- numerically identical to the reference and ``jit`` friendly.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.equivariant._assembly import from_chunks, group_by_irrep
from opifex.neural.equivariant.irreps import Irrep, Irreps, IrrepsArray


class EquivariantLinear(nnx.Module):
    """An equivariant linear map between two :class:`Irreps` layouts.

    For every output block, the layer sums learnable ``(mul_in, mul_out)``
    mixings over all input blocks carrying the *same* irrep ``(l, p)``.  Output
    blocks whose irrep is absent from the input are left at zero (no weights).
    """

    def __init__(
        self, irreps_in: Irreps | str, irreps_out: Irreps | str, *, rngs: nnx.Rngs
    ) -> None:
        """Build the layer and its per-path weight blocks.

        Args:
            irreps_in: Input layout.
            irreps_out: Output layout.
            rngs: Random number generators (keyword-only); ``rngs.params()``
                seeds the weight blocks.
        """
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)

        input_groups = group_by_irrep(self.irreps_in)
        in_muls = [mul for mul, _ in self.irreps_in.blocks]

        # Each path connects an input block to an output block sharing the irrep.
        # Weights are kept in a flat ``nnx.List`` (params); routing is static.
        weights: list[nnx.Param] = []
        self._path_in_index: tuple[int, ...] = ()
        self._path_out_index: tuple[int, ...] = ()
        in_indices: list[int] = []
        out_indices: list[int] = []
        key = rngs.params()
        for out_index, (mul_out, irrep_out) in enumerate(self.irreps_out.blocks):
            matching = input_groups.get(Irrep(irrep_out), [])
            # Path normalization (e3nn "element"): alpha = 1 / sum(matching mul_in).
            total_in = sum(in_muls[in_index] for in_index in matching)
            weight_std = 1.0 / math.sqrt(total_in) if total_in > 0 else 0.0
            for in_index in matching:
                key, subkey = jax.random.split(key)
                weight = weight_std * jax.random.normal(subkey, (in_muls[in_index], mul_out))
                weights.append(nnx.Param(weight))
                in_indices.append(in_index)
                out_indices.append(out_index)
        self.weights = nnx.List(weights)
        self._path_in_index = tuple(in_indices)
        self._path_out_index = tuple(out_indices)

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        """Apply the equivariant linear map.

        Args:
            x: Input feature with ``x.irreps == self.irreps_in``; arbitrary
                leading (batch) dimensions are supported.

        Returns:
            An :class:`IrrepsArray` with ``self.irreps_out``.

        Raises:
            ValueError: If ``x.irreps`` does not match ``self.irreps_in``.
        """
        if x.irreps != self.irreps_in:
            raise ValueError(
                f"EquivariantLinear expected input irreps {self.irreps_in!r}, got {x.irreps!r}"
            )
        input_chunks = x.chunks
        leading_shape = x.array.shape[:-1]
        accumulators: list[jax.Array | None] = [None] * len(self.irreps_out.blocks)
        for path, (in_index, out_index) in enumerate(
            zip(self._path_in_index, self._path_out_index, strict=True)
        ):
            weight = self.weights[path][...]
            contribution = jnp.einsum("uw,...ui->...wi", weight, input_chunks[in_index])
            current = accumulators[out_index]
            accumulators[out_index] = contribution if current is None else current + contribution
        return from_chunks(self.irreps_out, accumulators, leading_shape, x.array.dtype)
