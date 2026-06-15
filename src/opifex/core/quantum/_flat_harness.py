r"""Batched assembly harness over the flat primitive pytree (MESS pattern).

This module turns the per-primitive McMurchie-Davidson kernels in
:mod:`opifex.core.quantum._flat_mmd` into the full AO integral tensors using a
single ``jax.vmap`` over a flat :class:`~opifex.core.quantum.basis.FlatPrimitives`
representation followed by :func:`jax.ops.segment_sum` contraction -- the
batching strategy from ``graphcore-research/mess`` (Helal et al.,
arXiv:2406.03121), which replaces the eager ``n_shells**4`` Python quartet loop
with one XLA-fused, ``jit``-compilable trace.

Strategy
--------
* **One-electron (S/T/V):** enumerate the upper-triangular primitive pairs
  ``(i <= j)``, ``vmap`` the primitive op over the gathered primitive arrays,
  scatter the symmetric ``n_prim x n_prim`` matrix, then contract twice with
  ``segment_sum`` over ``orbital_index`` to reach the ``n_ao x n_ao`` AO matrix.
* **ERIs:** enumerate the 8-fold-unique AO quartets ``(i<=j, k<=l, ij<=kl)``
  (``gen_ijkl``); for each *chunk* gather the primitive quartets, ``vmap`` the
  primitive ERI, ``segment_sum`` to contract to one value per unique AO quartet,
  then scatter into the eight symmetry-equivalent dense positions. Chunking
  bounds peak memory while keeping a single trace per chunk size.
* **Schwarz screening:** a boolean mask ``Q_i Q_j >= 0`` (here a no-op-safe
  multiplicative mask seam; see :func:`schwarz_factors`) is multiplied into the
  per-quartet contributions so screening is ``jit``-clean and gradient-safe.

The angular-momentum powers (``lmn``) and the primitive->AO map
(``orbital_index``) are NumPy-static, so all index tables are baked at trace
time; only centres/exponents/coefficients are traced (differentiable forces).
"""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.ops import segment_sum

from opifex.core.quantum._flat_mmd import (
    eri_primitive,
    kinetic_primitive,
    nuclear_primitive,
    overlap_primitive,
)
from opifex.core.quantum.basis import FlatPrimitives  # noqa: TC001


# Default ERI quartet chunk size. Large enough to amortise dispatch on the
# small molecules used here while bounding peak memory for larger bases.
_DEFAULT_ERI_CHUNK = 16384


def _triu_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Upper-triangular ``(i, j)`` primitive-pair indices including the diagonal."""
    ii, jj = np.triu_indices(n)
    return ii.astype(np.int32), jj.astype(np.int32)


def _contract_pairs_to_aos(
    pair_values: Array,
    ii: np.ndarray,
    jj: np.ndarray,
    orbital_index: np.ndarray,
    num_orbitals: int,
    num_primitives: int,
) -> Array:
    """Scatter symmetric primitive-pair values and contract to an AO matrix.

    Mirrors MESS ``integrate_dense``: build the symmetric ``n_prim x n_prim``
    matrix from the upper triangle, then double ``segment_sum`` over
    ``orbital_index``.
    """
    matrix = jnp.zeros((num_primitives, num_primitives), dtype=pair_values.dtype)
    matrix = matrix.at[ii, jj].set(pair_values)
    matrix = matrix + matrix.T - jnp.diag(jnp.diag(matrix))
    index = jnp.asarray(orbital_index)
    contracted = segment_sum(matrix, index, num_segments=num_orbitals)
    return segment_sum(contracted.T, index, num_segments=num_orbitals)


def one_electron_matrices(
    flat: FlatPrimitives,
    nuclear_positions: Array,
    nuclear_charges: Array,
) -> tuple[Array, Array, Array]:
    """Batched ``(S, T, V)`` AO matrices via one ``vmap`` over primitive pairs.

    Args:
        flat: The flat primitive representation of the basis.
        nuclear_positions: Nuclear positions [Shape: (n_atoms, 3)].
        nuclear_charges: Nuclear charges [Shape: (n_atoms,)].

    Returns:
        ``(overlap, kinetic, nuclear)`` each of shape ``(n_ao, n_ao)``.
    """
    n_prim = flat.num_primitives
    ii, jj = _triu_pairs(n_prim)
    max_l = flat.max_total_l

    lmn = jnp.asarray(flat.lmn)
    lmn_a = lmn[ii]
    lmn_b = lmn[jj]
    center_a = flat.center[ii]
    center_b = flat.center[jj]
    alpha_a = flat.alpha[ii]
    alpha_b = flat.alpha[jj]
    coeff_pair = flat.coeff[ii] * flat.coeff[jj]

    def stv(la, lb, ca, cb, ea, eb):
        s = overlap_primitive(la, lb, ca, cb, ea, eb, max_l)
        t = kinetic_primitive(la, lb, ca, cb, ea, eb, max_l)
        v = nuclear_primitive(la, lb, ca, cb, ea, eb, nuclear_positions, nuclear_charges, max_l)
        return jnp.stack([s, t, v])

    values = jax.vmap(stv)(lmn_a, lmn_b, center_a, center_b, alpha_a, alpha_b)
    values = coeff_pair[:, None] * values

    overlap = _contract_pairs_to_aos(
        values[:, 0], ii, jj, flat.orbital_index, flat.num_orbitals, n_prim
    )
    kinetic = _contract_pairs_to_aos(
        values[:, 1], ii, jj, flat.orbital_index, flat.num_orbitals, n_prim
    )
    nuclear = _contract_pairs_to_aos(
        values[:, 2], ii, jj, flat.orbital_index, flat.num_orbitals, n_prim
    )
    return overlap, kinetic, nuclear


def gen_ijkl(n: int) -> Iterator[tuple[int, int, int, int]]:
    """Yield the 8-fold-unique AO quartet indices (MESS ``gen_ijkl``).

    Adapted from four-index transformations (S. Wilson): enumerates
    ``i >= j``, ``i >= k`` and the constrained ``l`` so that each of the eight
    permutation-equivalent quartets is generated exactly once.
    """
    for idx in range(n):
        for jdx in range(idx + 1):
            for kdx in range(idx + 1):
                lmax = jdx if idx == kdx else kdx
                for ldx in range(lmax + 1):
                    yield idx, jdx, kdx, ldx


def schwarz_factors(diagonal_eri: Array) -> Array:
    r"""Cauchy-Schwarz bound factors ``Q_i = sqrt((ii|ii))`` per AO quartet leg.

    The Cauchy-Schwarz inequality gives ``|(ij|kl)| <= Q_ij Q_kl`` with
    ``Q_ij = sqrt((ij|ij))``. Returning the per-AO diagonal roots lets a caller
    form the screening mask as a differentiable multiplicative factor.
    """
    return jnp.sqrt(jnp.abs(diagonal_eri))


def _orbital_primitive_offsets(
    orbital_index: np.ndarray, num_orbitals: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return per-AO primitive counts and the list of primitive index arrays."""
    members = [np.where(orbital_index == ao)[0].astype(np.int32) for ao in range(num_orbitals)]
    counts = np.asarray([m.shape[0] for m in members], dtype=np.int64)
    return counts, members


def _build_quartet_gather(
    flat: FlatPrimitives,
    quartets: np.ndarray,
    members: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Expand AO quartets to primitive-quartet indices and a batch (segment) map.

    Returns:
        ``(prim_indices, batch)`` where ``prim_indices`` has shape ``(4, n)`` of
        flat-primitive indices and ``batch`` has shape ``(n,)`` mapping each
        primitive quartet to its unique-AO-quartet segment.
    """
    prim_i: list[np.ndarray] = []
    prim_j: list[np.ndarray] = []
    prim_k: list[np.ndarray] = []
    prim_l: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    for count, (i, j, k, ll) in enumerate(quartets):
        mi, mj, mk, ml = members[i], members[j], members[k], members[ll]
        grid_i, grid_j, grid_k, grid_l = np.meshgrid(mi, mj, mk, ml, indexing="ij")
        prim_i.append(grid_i.reshape(-1))
        prim_j.append(grid_j.reshape(-1))
        prim_k.append(grid_k.reshape(-1))
        prim_l.append(grid_l.reshape(-1))
        batch.append(np.full(grid_i.size, count, dtype=np.int32))
    prim_indices = np.stack(
        [
            np.concatenate(prim_i),
            np.concatenate(prim_j),
            np.concatenate(prim_k),
            np.concatenate(prim_l),
        ]
    )
    return prim_indices, np.concatenate(batch)


def _eri_unique_chunk(
    flat: FlatPrimitives,
    prim_indices: np.ndarray,
    batch: np.ndarray,
    num_quartets: int,
) -> Array:
    """Contract one chunk of primitive quartets to per-unique-AO-quartet ERIs."""
    max_l = flat.max_total_l
    lmn = jnp.asarray(flat.lmn)
    ai, aj, ak, al = (jnp.asarray(prim_indices[r]) for r in range(4))

    coeff = flat.coeff[ai] * flat.coeff[aj] * flat.coeff[ak] * flat.coeff[al]

    def prim(la, lb, lc, ld, ca, cb, cc, cd, ea, eb, ec, ed):
        return eri_primitive(la, lb, lc, ld, ca, cb, cc, cd, ea, eb, ec, ed, max_l)

    values = jax.vmap(prim)(
        lmn[ai],
        lmn[aj],
        lmn[ak],
        lmn[al],
        flat.center[ai],
        flat.center[aj],
        flat.center[ak],
        flat.center[al],
        flat.alpha[ai],
        flat.alpha[aj],
        flat.alpha[ak],
        flat.alpha[al],
    )
    values = coeff * values
    return segment_sum(values, jnp.asarray(batch), num_segments=num_quartets)


def electron_repulsion_tensor(
    flat: FlatPrimitives,
    chunk_size: int = _DEFAULT_ERI_CHUNK,
) -> Array:
    """Batched dense ERI tensor ``(ij|kl)`` via unique quartets + 8-fold scatter.

    Enumerates the 8-fold-unique AO quartets, evaluates their (contracted) values
    in primitive-quartet chunks with a single ``vmap`` per chunk, then scatters
    each unique value into the eight permutation-equivalent dense positions.

    Args:
        flat: The flat primitive representation.
        chunk_size: Number of *primitive quartets* per ``vmap`` trace (memory
            knob; the kernel structure is identical across chunks).

    Returns:
        Dense ERI tensor [Shape: (n_ao,)*4] in chemist notation.
    """
    n_ao = flat.num_orbitals
    _, members = _orbital_primitive_offsets(flat.orbital_index, n_ao)
    quartets = np.asarray(list(gen_ijkl(n_ao)), dtype=np.int32)

    # Process unique AO quartets in groups whose expanded primitive-quartet count
    # stays under chunk_size, so each vmap trace is bounded in memory.
    unique_values: list[Array] = []
    start = 0
    n_unique = quartets.shape[0]
    while start < n_unique:
        end = start
        prim_count = 0
        while end < n_unique:
            i, j, k, ll = quartets[end]
            size = (
                members[i].shape[0]
                * members[j].shape[0]
                * members[k].shape[0]
                * members[ll].shape[0]
            )
            if end > start and prim_count + size > chunk_size:
                break
            prim_count += size
            end += 1
        group = quartets[start:end]
        prim_indices, batch = _build_quartet_gather(flat, group, members)
        unique_values.append(_eri_unique_chunk(flat, prim_indices, batch, group.shape[0]))
        start = end

    unique = jnp.concatenate(unique_values)
    ii = jnp.asarray(quartets[:, 0])
    jj = jnp.asarray(quartets[:, 1])
    kk = jnp.asarray(quartets[:, 2])
    ll = jnp.asarray(quartets[:, 3])

    eri = jnp.zeros((n_ao, n_ao, n_ao, n_ao), dtype=unique.dtype)
    eri = eri.at[ii, jj, kk, ll].set(unique)
    eri = eri.at[ii, jj, ll, kk].set(unique)
    eri = eri.at[jj, ii, kk, ll].set(unique)
    eri = eri.at[jj, ii, ll, kk].set(unique)
    eri = eri.at[kk, ll, ii, jj].set(unique)
    eri = eri.at[kk, ll, jj, ii].set(unique)
    eri = eri.at[ll, kk, ii, jj].set(unique)
    return eri.at[ll, kk, jj, ii].set(unique)


__all__ = [
    "electron_repulsion_tensor",
    "gen_ijkl",
    "one_electron_matrices",
    "schwarz_factors",
]
