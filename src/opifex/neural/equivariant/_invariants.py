r"""Rotation-invariant reductions of steerable features: ``norm`` and ``inner_product``.

Both map an :class:`~opifex.neural.equivariant.IrrepsArray` to ``0e`` scalars,
one per input multiplicity, and are invariant under rotation because the
Euclidean norm and the dot product of two vectors transforming under the same
Wigner-``D`` matrix are unchanged by an orthogonal change of basis.

* :func:`norm` -- the per-irrep Euclidean norm (ported from ``e3nn-jax``
  ``e3nn.norm``, ``../e3nn-jax/e3nn_jax/_src/basic.py``), using the NaN-safe
  ``sqrt`` (mask the zero entries before the square root and after) so the
  gradient is finite at a zero vector.
* :func:`inner_product` -- the per-multiplicity dot product, component-normalised
  by ``1 / dim`` (the QHNet ``InnerProduct``,
  ``../AIRS/OpenDFT/QHBench/QH9/models/QHNet.py``; e3nn ``"uuu"`` tensor product
  to ``0e`` with ``1 / ir.dim`` path normalisation). The scalar gating signal of
  the :class:`~opifex.neural.equivariant.NormGate` and of the pair-interaction
  refinement layers is built from these invariants.

Both functions are pure (no parameters) and ``jit``/``grad``/``vmap`` clean.
"""

from __future__ import annotations

import jax.numpy as jnp

from opifex.neural.equivariant._assembly import from_chunks
from opifex.neural.equivariant.irreps import Irrep, Irreps, IrrepsArray


def norm(x: IrrepsArray, *, squared: bool = False) -> IrrepsArray:
    r"""Return the per-irrep Euclidean norm of ``x`` as ``0e`` scalars.

    Each ``mul x (l, p)`` block contributes ``mul`` scalar norms, one per
    multiplicity, so the output layout replaces every block's irrep by ``0e``
    while keeping its multiplicity (e.g. ``2x0e + 3x1o -> 2x0e + 3x0e``).

    Args:
        x: Input steerable feature.
        squared: If ``True`` return the squared norm (no square root), which
            avoids the gradient singularity entirely.

    Returns:
        An :class:`IrrepsArray` of all-``0e`` scalars with the same leading shape
        as ``x`` and one scalar per input multiplicity.
    """
    out_blocks: list[tuple[int, Irrep]] = []
    out_chunks: list[jnp.ndarray | None] = []
    for (mul, _), chunk in zip(x.irreps.blocks, x.chunks, strict=True):
        squared_norm = jnp.sum(chunk * chunk, axis=-1, keepdims=True)
        if squared:
            value = squared_norm
        else:
            # NaN-safe sqrt: mask zeros before the root and restore them after,
            # so d/dx sqrt(0) does not propagate a NaN (e3nn-jax ``norm``).
            safe = jnp.where(squared_norm == 0.0, 1.0, squared_norm)
            value = jnp.where(squared_norm == 0.0, 0.0, jnp.sqrt(safe))
        out_blocks.append((mul, Irrep(0, 1)))
        out_chunks.append(value)
    return from_chunks(Irreps(tuple(out_blocks)), out_chunks, x.array.shape[:-1], x.array.dtype)


def inner_product(x: IrrepsArray, y: IrrepsArray, *, normalize: bool = True) -> IrrepsArray:
    r"""Return the per-multiplicity dot product of ``x`` and ``y`` as ``0e`` scalars.

    ``x`` and ``y`` must share their layout. Each ``mul x (l, p)`` block yields
    ``mul`` invariant scalars ``<x_u, y_u>`` (optionally divided by ``dim = 2l+1``,
    the e3nn ``"component"`` path normalisation used by QHNet's ``InnerProduct``).

    Args:
        x: First steerable feature.
        y: Second steerable feature with ``y.irreps == x.irreps``.
        normalize: If ``True`` (default) divide each block's dot product by its
            irrep dimension (component normalisation).

    Returns:
        An :class:`IrrepsArray` of all-``0e`` scalars, one per input multiplicity.

    Raises:
        ValueError: If ``x`` and ``y`` do not share the same irreps layout.
    """
    if x.irreps != y.irreps:
        raise ValueError(
            f"inner_product expects matching irreps, got {x.irreps!r} and {y.irreps!r}"
        )
    out_blocks: list[tuple[int, Irrep]] = []
    out_chunks: list[jnp.ndarray | None] = []
    for (mul, irrep), chunk_x, chunk_y in zip(x.irreps.blocks, x.chunks, y.chunks, strict=True):
        dot = jnp.sum(chunk_x * chunk_y, axis=-1, keepdims=True)
        if normalize:
            dot = dot / irrep.dim
        out_blocks.append((mul, Irrep(0, 1)))
        out_chunks.append(dot)
    return from_chunks(Irreps(tuple(out_blocks)), out_chunks, x.array.shape[:-1], x.array.dtype)


def rms_normalize(x: IrrepsArray, *, eps: float = 1e-6) -> IrrepsArray:
    r"""Scale each sample of ``x`` to unit root-mean-square (equivariant RMSNorm).

    Divides the whole feature vector by the per-sample scalar
    ``sqrt(mean(x**2) + eps)``. That scalar is **rotation-invariant** (each irrep
    block's squared norm is preserved by rotation, so their mean is too), so
    dividing by it preserves the transformation law -- this is an equivariant
    analogue of RMSNorm. It bounds the feature magnitude before operations that
    *square* the features (the self / pair interaction tensor products), without
    which an unnormalised trunk's growing activations blow up the squared output.

    Args:
        x: Input steerable feature; the RMS is taken over its last (feature) axis.
        eps: Numerical floor added to the mean square before the square root.

    Returns:
        An :class:`IrrepsArray` with the same layout as ``x``, each sample scaled
        to unit RMS.
    """
    scale = jnp.sqrt(jnp.mean(x.array**2, axis=-1, keepdims=True) + eps)
    return IrrepsArray(x.irreps, x.array / scale)


__all__ = ["inner_product", "norm", "rms_normalize"]
