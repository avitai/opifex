"""Reduced symmetric tensor-product basis (host-side precompute).

Builds the generalized Clebsch-Gordan ``U`` tensors that project the symmetric
``degree``-th tensor power of an irreps set onto each output irrep -- the change
of basis the MACE / Atomic Cluster Expansion symmetric contraction contracts
learnable weights against (Batatia et al. 2022, MACE; Drautz 2019, ACE). The
``U`` tensors are **constants** computed once on the host with NumPy; the runtime
contraction (a separate jit/grad/vmap-clean module) folds them with per-element
weights, the same split the :class:`FullyConnectedTensorProduct` uses for its
Clebsch-Gordan coupling.

The equivariant coupling reuses opifex's real Clebsch-Gordan
(:func:`opifex.geometry.algebra.wigner.clebsch_gordan_numpy`) via
:func:`_reduce_basis_product` -- the ``...ui,...vj,ijk->...uvk`` coupling of the
e3nn-jax ``reduce_basis_product`` reference. The symmetric subspace is then
obtained by averaging the coupling basis over the symmetric group ``S_degree``
and orthonormalising (Gram-Schmidt) -- the definition of the symmetric coupling
basis, equivalent to the permutation-basis intersection of the e3nn-jax
``reduced_symmetric_tensor_product_basis`` reference.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from opifex.geometry.algebra.wigner import clebsch_gordan_numpy
from opifex.neural.equivariant.irreps import Irrep, Irreps


# Internal basis representation: an ``Irreps`` layout plus one NumPy chunk per
# irrep block, each of shape ``(*free_axes, mul, ir.dim)`` where ``free_axes`` are
# the per-factor input indices (one axis of size ``d`` per coupled factor).
_Basis = tuple[Irreps, list[np.ndarray]]


def gram_schmidt(rows: np.ndarray, *, epsilon: float = 1e-5) -> np.ndarray:
    """Orthonormalise the rows of ``rows`` (dropping near-zero residuals).

    Faithful port of the e3nn-jax ``gram_schmidt`` reference: returns an
    ``(n_independent, dim)`` array whose rows are an orthonormal basis of the row
    space of ``rows``, discarding rows whose residual norm falls below ``epsilon``.
    """
    if rows.ndim != 2:
        raise ValueError(f"gram_schmidt expects a matrix, got shape {rows.shape}.")
    basis: list[np.ndarray] = []
    for row in rows:
        residual = row.astype(np.float64).copy()
        for previous in basis:
            residual = residual - np.dot(previous, residual) * previous
        norm = float(np.linalg.norm(residual))
        if norm > epsilon:
            basis.append(residual / norm)
    if not basis:
        return np.empty((0, rows.shape[1]), dtype=np.float64)
    return np.stack(basis)


def _factor_basis(irreps: Irreps, factor: int, degree: int) -> _Basis:
    """Identity change-of-basis for one tensor factor, placed in its free axis.

    The ``factor``-th of ``degree`` factors gets a length-``irreps.dim`` free axis
    in slot ``factor`` (size 1 in the others), so coupling the factors with
    :func:`_reduce_basis_product` broadcasts the free axes into the full
    ``(d,) * degree`` index grid (the e3nn-jax factor-placement convention).
    """
    dim = irreps.dim
    leading = (1,) * factor + (dim,) + (1,) * (degree - 1 - factor)
    identity = np.eye(dim, dtype=np.float64)
    chunks: list[np.ndarray] = []
    cursor = 0
    for mul, irrep in irreps:
        width = mul * irrep.dim
        block = identity[:, cursor : cursor + width].reshape(dim, mul, irrep.dim)
        chunks.append(block.reshape((*leading, mul, irrep.dim)))
        cursor += width
    return irreps, chunks


def _reduce_basis_product(
    basis1: _Basis, basis2: _Basis, keep_ir: frozenset[Irrep] | None = None
) -> _Basis:
    """Couple two bases with Clebsch-Gordan (e3nn-jax ``reduce_basis_product``).

    Each output irrep ``ir`` accumulates ``sqrt(ir.dim) * CG`` contractions of the
    inputs (``...ui,...vj,ijk->...uvk``), with the broadcast free axes preserved;
    same-irrep blocks are concatenated along the multiplicity axis.
    """
    irreps1, chunks1 = basis1
    irreps2, chunks2 = basis2
    collected: dict[Irrep, list[np.ndarray]] = {}
    for (mul1, ir1), x1 in zip(irreps1, chunks1, strict=True):
        for (mul2, ir2), x2 in zip(irreps2, chunks2, strict=True):
            for ir in ir1 * ir2:
                if keep_ir is not None and ir not in keep_ir:
                    continue
                coupling = math.sqrt(ir.dim) * clebsch_gordan_numpy(ir1.l, ir2.l, ir.l)
                coupled = np.einsum("...ui,...vj,ijk->...uvk", x1, x2, coupling)
                coupled = coupled.reshape((*coupled.shape[:-3], mul1 * mul2, ir.dim))
                collected.setdefault(ir, []).append(coupled)
    irreps_out: list[tuple[int, Irrep]] = []
    chunks_out: list[np.ndarray] = []
    for ir in sorted(collected):
        merged = np.concatenate(collected[ir], axis=-2)
        irreps_out.append((merged.shape[-2], ir))
        chunks_out.append(merged)
    return Irreps(tuple(irreps_out)), chunks_out


def _symmetrize(chunk: np.ndarray, degree: int) -> np.ndarray:
    """Average a coupled chunk over all permutations of its ``degree`` free axes."""
    accumulated = np.zeros_like(chunk)
    tail = (degree, degree + 1)  # the (mul, ir.dim) axes stay fixed
    for permutation in itertools.permutations(range(degree)):
        accumulated = accumulated + np.transpose(chunk, (*permutation, *tail))
    return accumulated / math.factorial(degree)


def reduced_symmetric_tensor_product_basis(
    irreps: Irreps | str,
    degree: int,
    *,
    keep_ir: Irreps | str | None = None,
    epsilon: float = 1e-5,
) -> dict[Irrep, np.ndarray]:
    r"""Return the symmetric ``degree``-fold coupling ``U`` tensors per output irrep.

    For each reachable output irrep ``ir_out`` the returned array ``U`` has shape
    ``(num_paths,) + (d,) * degree + (ir_out.dim,)`` where ``d = irreps.dim`` and
    ``num_paths`` is the number of independent **permutation-symmetric** coupling
    paths. ``U`` is invariant under permuting its ``degree`` input axes and its
    flattened paths are orthonormal.

    Args:
        irreps: Irreps of each (identical) tensor factor.
        degree: Number of factors (the tensor-power degree ``nu``); ``>= 1``.
        keep_ir: Optional restriction of the output irreps.
        epsilon: Gram-Schmidt tolerance for dropping dependent paths.

    Returns:
        Mapping from output :class:`Irrep` to its ``U`` tensor (empty paths
        omitted).

    Raises:
        ValueError: If ``degree < 1``.
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}.")
    irreps = Irreps(irreps)
    keep: frozenset[Irrep] | None = None
    if keep_ir is not None:
        keep = frozenset(ir for _, ir in Irreps(keep_ir))

    # Couple the factors one at a time, keeping all intermediate irreps so no path
    # is dropped early; restrict to keep_ir only on the final coupling.
    basis = _factor_basis(irreps, 0, degree)
    for factor in range(1, degree):
        keep_step = keep if factor == degree - 1 else None
        basis = _reduce_basis_product(basis, _factor_basis(irreps, factor, degree), keep_step)
    irreps_out, chunks = basis
    if degree == 1 and keep is not None:
        kept = [(block, x) for block, x in zip(irreps_out, chunks, strict=True) if block[1] in keep]
        irreps_out = [block for block, _ in kept]
        chunks = [x for _, x in kept]

    dim = irreps.dim
    result: dict[Irrep, np.ndarray] = {}
    for (mul, ir), chunk in zip(irreps_out, chunks, strict=True):
        symmetric = _symmetrize(chunk, degree)  # (d,)*degree + (mul, ir.dim)
        # Orthonormalise across the (now partly dependent) path/multiplicity axis.
        paths = np.moveaxis(symmetric, -2, 0).reshape(mul, -1)  # (mul, d^degree * ir.dim)
        orthonormal = gram_schmidt(paths, epsilon=epsilon)  # (num_paths, d^degree * ir.dim)
        if orthonormal.shape[0] == 0:
            continue
        result[ir] = orthonormal.reshape((orthonormal.shape[0], *(dim,) * degree, ir.dim))
    return result


__all__ = ["gram_schmidt", "reduced_symmetric_tensor_product_basis"]
