r"""Structured linear operators for scalable curvature representations.

Curvature matrices in second-order uncertainty quantification (GGN,
Fisher, Laplace posterior precision) are too large to store densely for
realistic networks, yet they carry exploitable structure: per-layer
Kronecker factorisation, block-diagonality across layers, and low-rank
plus diagonal corrections. This module provides a small family of
structured operators whose ``matvec`` / ``trace`` / ``logdet`` / ``solve``
use the *structured* identity rather than densifying first.

Every operator is a frozen pytree (the factor arrays are the leaves), so
each method composes with :func:`jax.jit`, :func:`jax.grad`, and
:func:`jax.vmap`.

Structured identities implemented here
--------------------------------------
* Kronecker ``A ⊗ B``: matvec via the vec trick ``vec(B X A^T)`` where
  ``X = vec^{-1}(v)``; ``trace(A ⊗ B) = tr(A) tr(B)``;
  ``logdet(A ⊗ B) = n_B logdet(A) + n_A logdet(B)``;
  ``(A ⊗ B)^{-1} = A^{-1} ⊗ B^{-1}``.
* Block-diagonal ``diag(M_1, …, M_k)``: every reduction is the per-block
  reduction — ``trace`` sums block traces, ``logdet`` sums block logdets,
  ``solve`` solves each block independently.
* Low-rank update ``D + U V^T`` (diagonal ``D`` plus a rank-``r``
  correction): ``solve`` via the Woodbury identity, ``logdet`` via the
  matrix-determinant lemma.
* Diagonal and identity operators as the scalable terminal cases.

References
----------
* Potapczynski, A. et al. 2023 — *CoLA: Exploiting Compositional
  Structure for Automatic and Efficient Numerical Linear Algebra*,
  arXiv:2309.03060. The matvec / trace / logdet / solve identities mirror
  ``cola.ops.operators`` (Kronecker, BlockDiag, Diagonal, LowRank) and
  ``cola.linalg`` (logdet, trace, inverse), reimplemented natively in
  JAX.
* Henderson, H. V. & Searle, S. R. 1981 — *On deriving the inverse of a
  sum of matrices*, SIAM Review 23(1) (Woodbury / matrix-determinant
  lemma).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp


@runtime_checkable
class StructuredOperator(Protocol):
    """Common interface for the structured curvature operators.

    Concrete operators are square (``shape == (n, n)``) and expose a
    matrix-vector product plus the three scalar reductions used by the
    Laplace posterior: ``trace``, ``logdet``, and ``solve``. The
    :meth:`to_dense` materialisation is provided for testing and small
    cases only.
    """

    @property
    def shape(self) -> tuple[int, int]:
        """Square shape ``(n, n)`` of the represented matrix."""
        ...

    def matvec(self, vector: jax.Array) -> jax.Array:
        """Return ``self @ vector`` using the structured identity."""
        ...

    def to_dense(self) -> jax.Array:
        """Materialise the dense ``(n, n)`` matrix (testing / small cases)."""
        ...

    def trace(self) -> jax.Array:
        """Return ``tr(self)`` as a scalar array."""
        ...

    def logdet(self) -> jax.Array:
        """Return ``log|det(self)|`` as a scalar array."""
        ...

    def solve(self, vector: jax.Array) -> jax.Array:
        """Return ``self^{-1} @ vector`` using the structured identity."""
        ...


@jax.tree_util.register_pytree_node_class
class IdentityOperator:
    """Identity operator ``I_n`` with ``O(1)`` storage.

    Args:
        dimension: Side length ``n`` of the identity.
        dtype: Element dtype used by :meth:`to_dense` and :meth:`logdet`.
    """

    def __init__(self, dimension: int, *, dtype: jnp.dtype = jnp.float32) -> None:
        """Store the side length and element dtype."""
        self._dimension = int(dimension)
        self._dtype = jnp.dtype(dtype)

    @property
    def shape(self) -> tuple[int, int]:
        """Square shape ``(n, n)``."""
        return (self._dimension, self._dimension)

    def matvec(self, vector: jax.Array) -> jax.Array:
        """Return ``vector`` unchanged."""
        return vector

    def to_dense(self) -> jax.Array:
        """Materialise the dense identity matrix."""
        return jnp.eye(self._dimension, dtype=self._dtype)

    def trace(self) -> jax.Array:
        """Return ``n`` (the trace of the identity)."""
        return jnp.asarray(self._dimension, dtype=self._dtype)

    def logdet(self) -> jax.Array:
        """Return ``0`` (``det(I) = 1``)."""
        return jnp.asarray(0.0, dtype=self._dtype)

    def solve(self, vector: jax.Array) -> jax.Array:
        """Return ``vector`` unchanged (``I^{-1} = I``)."""
        return vector

    def tree_flatten(self) -> tuple[tuple[()], tuple[int, jnp.dtype]]:
        """Pytree flatten: no array leaves; static side length + dtype."""
        return (), (self._dimension, self._dtype)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[int, jnp.dtype], _children: tuple[()]
    ) -> IdentityOperator:
        """Rebuild from the static ``(dimension, dtype)`` aux data."""
        dimension, dtype = aux_data
        return cls(dimension, dtype=dtype)


@jax.tree_util.register_pytree_node_class
class DiagonalOperator:
    """Diagonal operator ``diag(d)``.

    Args:
        diagonal: 1-D array ``d`` of diagonal entries.
    """

    def __init__(self, diagonal: jax.Array) -> None:
        """Store the diagonal vector ``d``."""
        self._diagonal = diagonal

    @property
    def diagonal(self) -> jax.Array:
        """The diagonal vector ``d``."""
        return self._diagonal

    @property
    def shape(self) -> tuple[int, int]:
        """Square shape ``(n, n)`` where ``n = d.shape[0]``."""
        size = self._diagonal.shape[0]
        return (size, size)

    def matvec(self, vector: jax.Array) -> jax.Array:
        """Return the elementwise product ``d * vector``."""
        return self._diagonal * vector

    def to_dense(self) -> jax.Array:
        """Materialise ``diag(d)``."""
        return jnp.diag(self._diagonal)

    def trace(self) -> jax.Array:
        """Return ``sum(d)``."""
        return jnp.sum(self._diagonal)

    def logdet(self) -> jax.Array:
        """Return ``sum(log|d|)``."""
        return jnp.sum(jnp.log(jnp.abs(self._diagonal)))

    def solve(self, vector: jax.Array) -> jax.Array:
        """Return the elementwise quotient ``vector / d``."""
        return vector / self._diagonal

    def tree_flatten(self) -> tuple[tuple[jax.Array], None]:
        """Pytree flatten: the diagonal array is the only leaf."""
        return (self._diagonal,), None

    @classmethod
    def tree_unflatten(cls, _aux_data: None, children: tuple[jax.Array]) -> DiagonalOperator:
        """Rebuild from the single diagonal-array leaf."""
        (diagonal,) = children
        return cls(diagonal)


@jax.tree_util.register_pytree_node_class
class KroneckerProduct:
    """Kronecker product ``A ⊗ B`` of two square factors.

    The dense matrix is never formed: ``matvec`` uses the vec trick, and
    the reductions use ``trace(A ⊗ B) = tr(A) tr(B)`` and
    ``logdet(A ⊗ B) = n_B logdet(A) + n_A logdet(B)``.

    Args:
        a: Left square factor ``A`` of shape ``(m, m)``.
        b: Right square factor ``B`` of shape ``(p, p)``.
    """

    def __init__(self, a: jax.Array, b: jax.Array) -> None:
        """Store the two square Kronecker factors ``A`` and ``B``."""
        self._a = a
        self._b = b

    @property
    def factor_a(self) -> jax.Array:
        """Left Kronecker factor ``A``."""
        return self._a

    @property
    def factor_b(self) -> jax.Array:
        """Right Kronecker factor ``B``."""
        return self._b

    @property
    def shape(self) -> tuple[int, int]:
        """Square shape ``(m p, m p)``."""
        size = self._a.shape[0] * self._b.shape[0]
        return (size, size)

    def matvec(self, vector: jax.Array) -> jax.Array:
        r"""Return ``(A ⊗ B) v`` via the vec trick ``vec(B X A^T)``.

        With ``v = vec(X)`` and ``X`` of shape ``(p, m)`` in row-major
        (C) order, ``(A ⊗ B) vec(X) = vec(B X A^T)``.
        """
        rows_a = self._a.shape[0]
        rows_b = self._b.shape[0]
        reshaped = vector.reshape(rows_a, rows_b)
        # vec(B X A^T): X is (rows_a, rows_b) so transpose to (rows_b, rows_a).
        product = self._b @ reshaped.T @ self._a.T
        return product.T.reshape(-1)

    def to_dense(self) -> jax.Array:
        """Materialise ``jnp.kron(A, B)``."""
        return jnp.kron(self._a, self._b)

    def trace(self) -> jax.Array:
        """Return ``tr(A) tr(B)``."""
        return jnp.trace(self._a) * jnp.trace(self._b)

    def logdet(self) -> jax.Array:
        """Return ``n_B logdet(A) + n_A logdet(B)``."""
        rows_a = self._a.shape[0]
        rows_b = self._b.shape[0]
        _, logdet_a = jnp.linalg.slogdet(self._a)
        _, logdet_b = jnp.linalg.slogdet(self._b)
        return rows_b * logdet_a + rows_a * logdet_b

    def solve(self, vector: jax.Array) -> jax.Array:
        r"""Return ``(A ⊗ B)^{-1} v = (A^{-1} ⊗ B^{-1}) v``.

        Uses the same vec trick with ``A^{-1}`` and ``B^{-1}`` applied via
        triangular-free linear solves ``vec(B^{-1} X A^{-T})``.
        """
        rows_a = self._a.shape[0]
        rows_b = self._b.shape[0]
        reshaped = vector.reshape(rows_a, rows_b).T  # (rows_b, rows_a)
        # B^{-1} X: solve B Y = X.
        solved_b = jnp.linalg.solve(self._b, reshaped)
        # (B^{-1} X) A^{-T}: solve A^T Z^T = (B^{-1} X)^T  ->  Z = (B^{-1}X) A^{-T}.
        solved = jnp.linalg.solve(self._a, solved_b.T).T
        return solved.T.reshape(-1)

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], None]:
        """Pytree flatten: the two factor arrays are the leaves."""
        return (self._a, self._b), None

    @classmethod
    def tree_unflatten(
        cls, _aux_data: None, children: tuple[jax.Array, jax.Array]
    ) -> KroneckerProduct:
        """Rebuild from the two factor-array leaves."""
        a, b = children
        return cls(a, b)


@jax.tree_util.register_pytree_node_class
class BlockDiagonal:
    """Block-diagonal operator ``diag(M_1, …, M_k)``.

    Each block is itself a :class:`StructuredOperator`, so every reduction
    is the per-block reduction. Blocks need not share a side length.

    Args:
        blocks: Ordered tuple of square structured operators.
    """

    def __init__(self, blocks: tuple[StructuredOperator, ...]) -> None:
        """Store the ordered tuple of diagonal blocks."""
        if not blocks:
            raise ValueError("BlockDiagonal requires at least one block.")
        self._blocks = tuple(blocks)
        self._sizes = tuple(block.shape[0] for block in self._blocks)

    @property
    def blocks(self) -> tuple[StructuredOperator, ...]:
        """The ordered tuple of diagonal blocks."""
        return self._blocks

    @property
    def shape(self) -> tuple[int, int]:
        """Square shape ``(sum n_i, sum n_i)``."""
        total = sum(self._sizes)
        return (total, total)

    def _split(self, vector: jax.Array) -> list[jax.Array]:
        """Split ``vector`` into the per-block segments."""
        offsets = jnp.cumsum(jnp.asarray(self._sizes))[:-1]
        return list(jnp.split(vector, offsets))

    def matvec(self, vector: jax.Array) -> jax.Array:
        """Apply each block to its segment and concatenate the results."""
        segments = self._split(vector)
        applied = [
            block.matvec(segment) for block, segment in zip(self._blocks, segments, strict=True)
        ]
        return jnp.concatenate(applied)

    def to_dense(self) -> jax.Array:
        """Materialise the dense block-diagonal matrix."""
        dense_blocks = [block.to_dense() for block in self._blocks]
        return jax.scipy.linalg.block_diag(*dense_blocks)

    def trace(self) -> jax.Array:
        """Return ``sum_i tr(M_i)``."""
        return sum((block.trace() for block in self._blocks), start=jnp.asarray(0.0))

    def logdet(self) -> jax.Array:
        """Return ``sum_i logdet(M_i)``."""
        return sum((block.logdet() for block in self._blocks), start=jnp.asarray(0.0))

    def solve(self, vector: jax.Array) -> jax.Array:
        """Solve each block independently and concatenate the results."""
        segments = self._split(vector)
        solved = [
            block.solve(segment) for block, segment in zip(self._blocks, segments, strict=True)
        ]
        return jnp.concatenate(solved)

    def tree_flatten(
        self,
    ) -> tuple[tuple[StructuredOperator, ...], None]:
        """Pytree flatten: the blocks are the (sub-pytree) children."""
        return self._blocks, None

    @classmethod
    def tree_unflatten(
        cls, _aux_data: None, children: tuple[StructuredOperator, ...]
    ) -> BlockDiagonal:
        """Rebuild from the block children."""
        return cls(tuple(children))


@jax.tree_util.register_pytree_node_class
class LowRankUpdate:
    r"""Diagonal-plus-low-rank operator ``D + U V^T``.

    Stores a diagonal ``D = diag(d)`` plus a rank-``r`` correction
    ``U V^T`` with ``U, V`` of shape ``(n, r)``. When ``V`` is omitted the
    symmetric form ``D + U U^T`` is used. ``solve`` uses the Woodbury
    identity and ``logdet`` the matrix-determinant lemma, both at
    ``O(n r^2 + r^3)`` cost instead of ``O(n^3)``.

    Args:
        diagonal: 1-D diagonal ``d`` of length ``n``.
        u: Low-rank left factor ``U`` of shape ``(n, r)``.
        v: Low-rank right factor ``V`` of shape ``(n, r)``; defaults to
            ``U`` for the symmetric ``D + U U^T`` form.
    """

    def __init__(self, diagonal: jax.Array, u: jax.Array, v: jax.Array | None = None) -> None:
        """Store the diagonal ``d`` and low-rank factors ``U``, ``V``."""
        self._diagonal = diagonal
        self._u = u
        self._v = u if v is None else v

    @property
    def diagonal(self) -> jax.Array:
        """The diagonal vector ``d``."""
        return self._diagonal

    @property
    def u(self) -> jax.Array:
        """Low-rank left factor ``U``."""
        return self._u

    @property
    def v(self) -> jax.Array:
        """Low-rank right factor ``V``."""
        return self._v

    @property
    def shape(self) -> tuple[int, int]:
        """Square shape ``(n, n)``."""
        size = self._diagonal.shape[0]
        return (size, size)

    def matvec(self, vector: jax.Array) -> jax.Array:
        """Return ``(D + U V^T) v = d * v + U (V^T v)``."""
        return self._diagonal * vector + self._u @ (self._v.T @ vector)

    def to_dense(self) -> jax.Array:
        """Materialise ``diag(d) + U V^T``."""
        return jnp.diag(self._diagonal) + self._u @ self._v.T

    def trace(self) -> jax.Array:
        """Return ``sum(d) + tr(U V^T) = sum(d) + sum(U * V)``."""
        return jnp.sum(self._diagonal) + jnp.sum(self._u * self._v)

    def logdet(self) -> jax.Array:
        r"""Return ``logdet(D + U V^T)`` via the matrix-determinant lemma.

        ``det(D + U V^T) = det(D) det(I_r + V^T D^{-1} U)``.
        """
        rank = self._u.shape[1]
        d_inverse_u = self._u / self._diagonal[:, None]
        capacitance = jnp.eye(rank, dtype=self._u.dtype) + self._v.T @ d_inverse_u
        logdet_diagonal = jnp.sum(jnp.log(jnp.abs(self._diagonal)))
        _, logdet_capacitance = jnp.linalg.slogdet(capacitance)
        return logdet_diagonal + logdet_capacitance

    def solve(self, vector: jax.Array) -> jax.Array:
        r"""Return ``(D + U V^T)^{-1} v`` via the Woodbury identity.

        ``(D + U V^T)^{-1} = D^{-1} - D^{-1} U
        (I_r + V^T D^{-1} U)^{-1} V^T D^{-1}``.
        """
        rank = self._u.shape[1]
        d_inverse_vector = vector / self._diagonal
        d_inverse_u = self._u / self._diagonal[:, None]
        capacitance = jnp.eye(rank, dtype=self._u.dtype) + self._v.T @ d_inverse_u
        correction = d_inverse_u @ jnp.linalg.solve(capacitance, self._v.T @ d_inverse_vector)
        return d_inverse_vector - correction

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        """Pytree flatten: the diagonal and both factors are the leaves."""
        return (self._diagonal, self._u, self._v), None

    @classmethod
    def tree_unflatten(
        cls, _aux_data: None, children: tuple[jax.Array, jax.Array, jax.Array]
    ) -> LowRankUpdate:
        """Rebuild from the diagonal and factor leaves."""
        diagonal, u, v = children
        return cls(diagonal, u, v)


__all__ = [
    "BlockDiagonal",
    "DiagonalOperator",
    "IdentityOperator",
    "KroneckerProduct",
    "LowRankUpdate",
    "StructuredOperator",
]
