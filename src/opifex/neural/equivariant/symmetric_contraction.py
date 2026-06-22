"""MACE-style symmetric contraction (higher-body-order product basis).

Raises a per-node single-particle basis ``A`` (shape ``(..., channels, irreps)``)
to body order ``correlation + 1`` by contracting the symmetric tensor powers of
``A`` against the precomputed generalized-Clebsch-Gordan ``U`` tensors
(:func:`reduced_symmetric_tensor_product_basis`) with learnable, per-element
weights -- the Atomic Cluster Expansion contraction of MACE (Batatia et al. 2022;
Drautz 2019).

The runtime forward is a Horner recursion ``((w_n A + w_{n-1}) A + ... ) A`` of
``jnp`` einsums (the e3nn-jax ``SymmetricTensorProduct`` reference), fully
``jit`` / ``grad`` / ``vmap`` compatible (so conservative forces, an energy
gradient, differentiate through it). The ``U`` tensors are host-computed constants
stored as static data and rebuilt as compile-time ``jnp`` constants in
``__call__`` -- the same split :class:`FullyConnectedTensorProduct` uses for its
Clebsch-Gordan coupling, keeping them out of the optimised parameter state.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.neural.dtypes import default_float_dtype
from opifex.neural.equivariant._reduced_tensor_product import (
    reduced_symmetric_tensor_product_basis,
)
from opifex.neural.equivariant.irreps import Irrep, Irreps, IrrepsArray


class SymmetricContraction(nnx.Module):
    r"""Per-element symmetric contraction up to a given correlation (body order).

    Maps a per-channel single-particle basis ``A`` of layout ``irreps_in`` to
    ``irreps_out`` features, channel-wise, summing the contributions of every
    tensor-power order ``1 .. correlation``. Weights are indexed by atomic element,
    so each species gets its own contraction (the MACE convention).

    Args:
        irreps_in: Per-channel input irreps (multiplicity one each; the channel
            axis carries the multiplicity).
        irreps_out: Per-channel output irreps to keep.
        correlation: Maximum tensor-power order ``nu`` (body order ``nu + 1``).
        num_species: Number of distinct atomic elements (the per-element weight
            axis).
        num_channels: Number of feature channels.
        rngs: Random number generators (keyword-only) seeding the weights.

    Raises:
        ValueError: If ``correlation < 1``.
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        irreps_out: Irreps | str,
        *,
        correlation: int,
        num_species: int,
        num_channels: int,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the per-order ``U`` constants and per-element weight parameters."""
        super().__init__()
        if correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {correlation}.")
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.num_channels = num_channels
        self._correlation = correlation
        keep = [ir for _, ir in self.irreps_out]

        # Per order (high -> low): the kept output irreps, their static U tensors
        # (normalised by num_paths, e3nn layout (d,)*order + (num_paths, ir.dim)),
        # and a (num_species, num_paths, num_channels) weight parameter each.
        self._orders = tuple(range(correlation, 0, -1))
        u_static: dict[tuple[int, int], tuple[tuple[int, ...], tuple[float, ...]]] = {}
        weights: list[nnx.Param] = []
        self._term_meta: list[tuple[int, int, int]] = []  # (order, l, p) per weight
        dtype = default_float_dtype()
        for order in self._orders:
            basis = reduced_symmetric_tensor_product_basis(
                self.irreps_in, order, keep_ir=Irreps(tuple((1, ir) for ir in keep))
            )
            for ir in sorted(basis):
                u = np.moveaxis(basis[ir], 0, -2)  # (d,)*order + (num_paths, ir.dim)
                num_paths = u.shape[-2]
                u = u / num_paths  # normalise (e3nn: fold the 1/num_paths into U)
                u_static[(order, _ir_key(ir))] = (u.shape, tuple(u.reshape(-1).tolist()))
                std = 1.0 / np.sqrt(float(num_paths))
                weights.append(
                    nnx.Param(
                        std
                        * nnx.initializers.normal(stddev=1.0)(
                            rngs.params(), (num_species, num_paths, num_channels), dtype
                        )
                    )
                )
                self._term_meta.append((order, ir.l, ir.p))
        self._u_static = u_static
        self.weights = nnx.List(weights)

    def __call__(
        self,
        node_features: IrrepsArray,
        node_attrs: Float[Array, "n_nodes num_species"],
    ) -> IrrepsArray:
        """Contract the symmetric powers of ``node_features`` with per-element weights.

        Args:
            node_features: Per-node, per-channel single-particle basis of layout
                ``irreps_in`` and array shape ``(n_nodes, num_channels, irreps_in.dim)``.
            node_attrs: One-hot atom-type attributes, shape ``(n_nodes, num_species)``.

        Returns:
            An :class:`IrrepsArray` of layout ``irreps_out`` and array shape
            ``(n_nodes, num_channels, irreps_out.dim)``.
        """
        features = node_features.array  # (n_nodes, channels, d_in)
        dtype = features.dtype
        accumulated: dict[int, Array] = {}
        just_initialised: set[int] = set()
        for weight_index, (order, l_out, p_out) in enumerate(self._term_meta):
            ir_key = (l_out << 1) | (0 if p_out == 1 else 1)
            shape, flat = self._u_static[(order, ir_key)]
            u = jnp.asarray(flat, dtype=dtype).reshape(shape)
            weight = self.weights[weight_index].value.astype(dtype)
            # Select the per-node weight slab from the one-hot atom type.
            node_weight = jnp.einsum("epc,ne->npc", weight, node_attrs.astype(dtype))
            if ir_key not in accumulated:
                # Special init: also contracts one copy of A (one input axis).
                accumulated[ir_key] = jnp.einsum("...jki,nkc,ncj->nc...i", u, node_weight, features)
                just_initialised.add(ir_key)
            else:
                accumulated[ir_key] = accumulated[ir_key] + jnp.einsum(
                    "...ki,nkc->nc...i", u, node_weight
                )
            # When the next order changes, fold one A into every running term that
            # was not initialised on this order.
            next_order = (
                self._term_meta[weight_index + 1][0]
                if weight_index + 1 < len(self._term_meta)
                else 0
            )
            if next_order != order:
                accumulated = {
                    key: value
                    if key in just_initialised
                    else jnp.einsum("nc...ji,ncj->nc...i", value, features)
                    for key, value in accumulated.items()
                }
                just_initialised.clear()

        n_nodes = features.shape[0]
        blocks = [
            accumulated[_ir_key(ir)].reshape(n_nodes, self.num_channels, ir.dim)
            for _, ir in self.irreps_out
            if _ir_key(ir) in accumulated
        ]
        return IrrepsArray(self.irreps_out, jnp.concatenate(blocks, axis=-1))


def _ir_key(ir: Irrep) -> int:
    """Pack an irrep into a hashable int key (``2*l + parity_bit``)."""
    return (ir.l << 1) | (0 if ir.p == 1 else 1)


__all__ = ["SymmetricContraction"]
