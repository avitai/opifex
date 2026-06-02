r"""Internal helpers for assembling :class:`IrrepsArray` outputs from chunks.

The equivariant layers (:mod:`linear`, :mod:`tensor_product`, :mod:`gate`)
compute their output one ``(mul, Irrep)`` block at a time and then concatenate
the blocks along the feature axis.  This mirrors ``e3nn_jax.from_chunks``
(``../e3nn-jax/e3nn_jax/_src/irreps_array.py``) but is specialised to opifex's
:class:`~opifex.neural.equivariant.IrrepsArray` (which always stores a dense
array, with ``None`` chunks materialised as zeros).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float  # noqa: TC002

from opifex.neural.equivariant.irreps import Irrep, Irreps, IrrepsArray


def from_chunks(
    irreps: Irreps,
    chunks: list[Float[Array, "... mul dim_l"] | None],
    leading_shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> IrrepsArray:
    """Assemble an :class:`IrrepsArray` from per-block chunks.

    Args:
        irreps: The output layout; one entry per chunk.
        chunks: Per-block arrays of shape ``leading_shape + (mul, 2l+1)``. A
            ``None`` chunk is materialised as zeros (an unconnected output path).
        leading_shape: The broadcast leading shape shared by all chunks.
        dtype: Output dtype (used for ``None`` chunks).

    Returns:
        An :class:`IrrepsArray` with the requested ``irreps`` and an array of
        shape ``leading_shape + (irreps.dim,)``.
    """
    flat_blocks: list[Float[Array, "... width"]] = []
    for (mul, irrep), chunk in zip(irreps.blocks, chunks, strict=True):
        width = mul * irrep.dim
        if chunk is None:
            flat_blocks.append(jnp.zeros((*leading_shape, width), dtype=dtype))
        else:
            flat_blocks.append(chunk.reshape(*leading_shape, width))
    if not flat_blocks:
        return IrrepsArray(irreps, jnp.zeros((*leading_shape, 0), dtype=dtype))
    return IrrepsArray(irreps, jnp.concatenate(flat_blocks, axis=-1))


def group_by_irrep(irreps: Irreps) -> dict[Irrep, list[int]]:
    """Map each distinct :class:`Irrep` to the indices of the blocks carrying it.

    Used by the equivariant linear layer to connect input and output blocks that
    share the same ``(l, p)``.

    Args:
        irreps: A layout.

    Returns:
        A dict from :class:`Irrep` to the list of block indices with that irrep.
    """
    groups: dict[Irrep, list[int]] = {}
    for index, (_, irrep) in enumerate(irreps.blocks):
        groups.setdefault(irrep, []).append(index)
    return groups
