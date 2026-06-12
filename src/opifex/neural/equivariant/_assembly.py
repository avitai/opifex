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


def apply_scalar_weights(
    features: IrrepsArray, weights: Float[Array, "... num_irreps"]
) -> IrrepsArray:
    r"""Scale each multiplicity of ``features`` by an invariant scalar weight.

    Multiplying a steerable feature's multiplicity by a rotation-invariant scalar
    is equivariant, so this realises the radial / scalar-gated modulation used by
    the NequIP convolution and the equivariant Hamiltonian pair head (one weight
    per irrep, ``weights.shape[-1] == features.irreps.num_irreps``).

    Args:
        features: The steerable feature to scale.
        weights: Per-multiplicity scalars of width ``features.irreps.num_irreps``
            (multiplicity-major, matching the block order), broadcasting over the
            leading axes of ``features``.

    Returns:
        An :class:`IrrepsArray` with the same irreps as ``features``, each
        multiplicity scaled by its weight.
    """
    scaled: list[Float[Array, "... mul dim_l"] | None] = []
    cursor = 0
    for (mul, _), chunk in zip(features.irreps.blocks, features.chunks, strict=True):
        block_weights = weights[..., cursor : cursor + mul]
        scaled.append(chunk * block_weights[..., None])
        cursor += mul
    return from_chunks(features.irreps, scaled, features.array.shape[:-1], features.array.dtype)


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
