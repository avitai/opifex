"""Halko-Martinsson-Tropp randomised SVD with subspace iteration.

Computes a rank-``rank`` approximate singular value decomposition of a
matrix-free operator ``A`` (accessed through ``matvec`` and
``matvec_transpose``) following Algorithm 5.1 of Halko, Martinsson, Tropp
arXiv:0909.4061.

Algorithm
---------
1. Sample a Gaussian sketch matrix ``Ω`` of shape ``(dim_cols, rank +
   oversampling)``.
2. ``Y = A @ Ω`` (``rank + oversampling`` matvecs).
3. ``num_iterations`` subspace iterations: ``Y = A @ (A.T @ Y)`` —
   amplifies leading directions by powers of the spectrum.
4. ``Q = qr(Y).Q`` — orthonormal basis of the captured range.
5. ``B = Q.T @ A`` (``rank + oversampling`` transposed matvecs).
6. Dense SVD of ``B``: ``U_b, S, V_t = svd(B)``.
7. Lift left singular vectors: ``U = Q @ U_b``.
8. Truncate to top ``rank`` triplets.

cola's ``cola/linalg/tbd/randomized_svd.py`` lacks the subspace iteration
and does not call ``qr`` on the sketch image; this implementation follows
the canonical HMT recipe directly from the paper.

References
----------
* Halko, Martinsson, Tropp arXiv:0909.4061 — *Finding structure with
  randomness: probabilistic algorithms for constructing approximate
  matrix decompositions*, SIAM Review 2011.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable


def randomized_svd(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    matvec_transpose: Callable[[jax.Array], jax.Array],
    dim_rows: int,
    dim_cols: int,
    rank: int,
    oversampling: int,
    num_iterations: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Approximate the top-``rank`` SVD of ``A`` via HMT randomised SVD.

    Args:
        matvec: callable computing ``A @ v`` for ``v`` of shape ``(dim_cols,)``.
        matvec_transpose: callable computing ``A.T @ u`` for ``u`` of
            shape ``(dim_rows,)``.
        dim_rows: number of rows of ``A`` (static).
        dim_cols: number of columns of ``A`` (static).
        rank: target rank (static); ``rank <= min(dim_rows, dim_cols)``.
        oversampling: extra sketch columns. Typical values 5–10 trade
            additional matvec cost for robust spectral capture
            (Halko+ §4.4).
        num_iterations: number of subspace iterations (static). ``0``
            recovers the basic projection-based estimate; ``>= 1``
            amplifies leading directions when the spectrum decays slowly.
        key: PRNG key, split into one substream for the initial sketch
            and one per subspace-iteration step.

    Returns:
        ``(left_vectors, singvals, right_vectors)`` where ``left_vectors``
        has shape ``(dim_rows, rank)``, ``singvals`` has shape
        ``(rank,)``, and ``right_vectors`` has shape ``(dim_cols, rank)``.
    """
    sketch_width = rank + oversampling
    sketch_key, _ = jax.random.split(key)
    sketch = jax.random.normal(sketch_key, (dim_cols, sketch_width))

    sketched_range = jax.vmap(matvec, in_axes=1, out_axes=1)(sketch)
    for _ in range(num_iterations):
        adjoint_image = jax.vmap(matvec_transpose, in_axes=1, out_axes=1)(sketched_range)
        sketched_range = jax.vmap(matvec, in_axes=1, out_axes=1)(adjoint_image)

    range_basis, _ = jnp.linalg.qr(sketched_range)
    projected = jax.vmap(matvec_transpose, in_axes=1, out_axes=1)(range_basis)
    small_b = projected.T
    small_u, singvals, small_vt = jnp.linalg.svd(small_b, full_matrices=False)
    left_full = range_basis @ small_u
    right_full = small_vt.T
    return left_full[:, :rank], singvals[:rank], right_full[:, :rank]


__all__ = ["randomized_svd"]
