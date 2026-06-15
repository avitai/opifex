"""Tests for the structured curvature linear operators.

Each operator's ``matvec`` / ``trace`` / ``logdet`` / ``solve`` must equal
the dense reference computed from :meth:`to_dense`, and every operator must
round-trip through :func:`jax.tree_util` and compose with ``jit`` / ``grad``
/ ``vmap``.

The structured identities are those of CoLA (Potapczynski et al. 2023,
arXiv:2309.03060): the Kronecker vec trick and
``logdet(A ⊗ B) = n_B logdet(A) + n_A logdet(B)``; the per-block
block-diagonal reductions; and the Woodbury / matrix-determinant lemma for
the diagonal-plus-low-rank update.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.curvature.structured import (
    BlockDiagonal,
    DiagonalOperator,
    IdentityOperator,
    KroneckerProduct,
    LowRankUpdate,
    StructuredOperator,
)


# All residual checks run in float64 so the structured-vs-dense agreement can
# be asserted at ~1e-10; the jit/grad/vmap smokes are dtype-agnostic.
jax.config.update("jax_enable_x64", True)

_RESIDUAL_TOLERANCE = 1e-9


def _symmetric_positive_definite(key: jax.Array, size: int) -> jax.Array:
    """Return a random SPD matrix ``M M^T + n I`` of side ``size``."""
    matrix = jax.random.normal(key, (size, size), dtype=jnp.float64)
    return matrix @ matrix.T + size * jnp.eye(size, dtype=jnp.float64)


# --------------------------------------------------------------------------- #
# KroneckerProduct
# --------------------------------------------------------------------------- #
def test_kronecker_to_dense_equals_jnp_kron() -> None:
    """``KroneckerProduct.to_dense`` equals ``jnp.kron(A, B)``."""
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    a = jax.random.normal(key_a, (3, 3), dtype=jnp.float64)
    b = jax.random.normal(key_b, (4, 4), dtype=jnp.float64)
    operator = KroneckerProduct(a, b)
    assert jnp.allclose(operator.to_dense(), jnp.kron(a, b), atol=_RESIDUAL_TOLERANCE)


def test_kronecker_matvec_equals_dense_matvec() -> None:
    """The Kronecker vec-trick matvec equals the dense matvec."""
    key_a, key_b, key_v = jax.random.split(jax.random.PRNGKey(1), 3)
    a = jax.random.normal(key_a, (3, 3), dtype=jnp.float64)
    b = jax.random.normal(key_b, (4, 4), dtype=jnp.float64)
    operator = KroneckerProduct(a, b)
    vector = jax.random.normal(key_v, (12,), dtype=jnp.float64)
    assert jnp.allclose(
        operator.matvec(vector), operator.to_dense() @ vector, atol=_RESIDUAL_TOLERANCE
    )


def test_kronecker_trace_equals_dense_trace() -> None:
    """``trace(A ⊗ B) = tr(A) tr(B)`` equals the dense trace."""
    key_a, key_b = jax.random.split(jax.random.PRNGKey(2))
    a = jax.random.normal(key_a, (3, 3), dtype=jnp.float64)
    b = jax.random.normal(key_b, (5, 5), dtype=jnp.float64)
    operator = KroneckerProduct(a, b)
    assert jnp.allclose(operator.trace(), jnp.trace(operator.to_dense()), atol=_RESIDUAL_TOLERANCE)


def test_kronecker_logdet_equals_dense_slogdet() -> None:
    """``logdet(A ⊗ B)`` equals ``slogdet`` of the dense Kronecker product."""
    key_a, key_b = jax.random.split(jax.random.PRNGKey(3))
    a = _symmetric_positive_definite(key_a, 3)
    b = _symmetric_positive_definite(key_b, 4)
    operator = KroneckerProduct(a, b)
    _, dense_logdet = jnp.linalg.slogdet(operator.to_dense())
    assert jnp.allclose(operator.logdet(), dense_logdet, atol=1e-7)


def test_kronecker_solve_equals_dense_solve() -> None:
    """``(A ⊗ B)^{-1} v`` equals the dense linear solve."""
    key_a, key_b, key_v = jax.random.split(jax.random.PRNGKey(4), 3)
    a = _symmetric_positive_definite(key_a, 3)
    b = _symmetric_positive_definite(key_b, 4)
    operator = KroneckerProduct(a, b)
    vector = jax.random.normal(key_v, (12,), dtype=jnp.float64)
    expected = jnp.linalg.solve(operator.to_dense(), vector)
    assert jnp.allclose(operator.solve(vector), expected, atol=1e-7)


# --------------------------------------------------------------------------- #
# DiagonalOperator and IdentityOperator
# --------------------------------------------------------------------------- #
def test_diagonal_operator_methods_equal_dense() -> None:
    """Every diagonal-operator reduction equals its dense reference."""
    key_d, key_v = jax.random.split(jax.random.PRNGKey(5))
    diagonal = jnp.abs(jax.random.normal(key_d, (6,), dtype=jnp.float64)) + 0.5
    vector = jax.random.normal(key_v, (6,), dtype=jnp.float64)
    operator = DiagonalOperator(diagonal)
    dense = operator.to_dense()
    assert jnp.allclose(operator.matvec(vector), dense @ vector, atol=_RESIDUAL_TOLERANCE)
    assert jnp.allclose(operator.trace(), jnp.trace(dense), atol=_RESIDUAL_TOLERANCE)
    _, dense_logdet = jnp.linalg.slogdet(dense)
    assert jnp.allclose(operator.logdet(), dense_logdet, atol=_RESIDUAL_TOLERANCE)
    assert jnp.allclose(
        operator.solve(vector), jnp.linalg.solve(dense, vector), atol=_RESIDUAL_TOLERANCE
    )


def test_identity_operator_methods_match_definition() -> None:
    """The identity operator is the multiplicative identity with zero logdet."""
    operator = IdentityOperator(5, dtype=jnp.float64)
    vector = jnp.arange(5, dtype=jnp.float64)
    assert jnp.allclose(operator.matvec(vector), vector)
    assert jnp.allclose(operator.solve(vector), vector)
    assert jnp.allclose(operator.trace(), 5.0)
    assert jnp.allclose(operator.logdet(), 0.0)
    assert jnp.allclose(operator.to_dense(), jnp.eye(5, dtype=jnp.float64))


# --------------------------------------------------------------------------- #
# BlockDiagonal
# --------------------------------------------------------------------------- #
def test_block_diagonal_methods_equal_dense() -> None:
    """Every block-diagonal reduction equals its dense reference."""
    key_a, key_b, key_d, key_v = jax.random.split(jax.random.PRNGKey(6), 4)
    block_a = _symmetric_positive_definite(key_a, 3)
    block_b = _symmetric_positive_definite(key_b, 2)
    diagonal = jnp.abs(jax.random.normal(key_d, (4,), dtype=jnp.float64)) + 0.5
    operator = BlockDiagonal(
        (
            KroneckerProduct(block_a[:1, :1], block_a),  # 3x3 effective via kron 1x1
            DiagonalOperator(diagonal),
            KroneckerProduct(block_b[:1, :1], block_b),
        )
    )
    dense = operator.to_dense()
    vector = jax.random.normal(key_v, (dense.shape[0],), dtype=jnp.float64)
    assert jnp.allclose(operator.matvec(vector), dense @ vector, atol=1e-7)
    assert jnp.allclose(operator.trace(), jnp.trace(dense), atol=1e-7)
    _, dense_logdet = jnp.linalg.slogdet(dense)
    assert jnp.allclose(operator.logdet(), dense_logdet, atol=1e-6)
    assert jnp.allclose(operator.solve(vector), jnp.linalg.solve(dense, vector), atol=1e-6)


def test_block_diagonal_requires_at_least_one_block() -> None:
    """An empty block tuple is rejected at construction."""
    with pytest.raises(ValueError, match="at least one block"):
        BlockDiagonal(())


# --------------------------------------------------------------------------- #
# LowRankUpdate
# --------------------------------------------------------------------------- #
def test_low_rank_update_methods_equal_dense_symmetric() -> None:
    """``D + U U^T`` matvec / trace / logdet / solve equal the dense form."""
    key_d, key_u, key_v = jax.random.split(jax.random.PRNGKey(7), 3)
    diagonal = jnp.abs(jax.random.normal(key_d, (6,), dtype=jnp.float64)) + 1.0
    u = jax.random.normal(key_u, (6, 2), dtype=jnp.float64)
    operator = LowRankUpdate(diagonal, u)
    dense = operator.to_dense()
    vector = jax.random.normal(key_v, (6,), dtype=jnp.float64)
    assert jnp.allclose(operator.matvec(vector), dense @ vector, atol=_RESIDUAL_TOLERANCE)
    assert jnp.allclose(operator.trace(), jnp.trace(dense), atol=_RESIDUAL_TOLERANCE)
    _, dense_logdet = jnp.linalg.slogdet(dense)
    assert jnp.allclose(operator.logdet(), dense_logdet, atol=1e-8)
    assert jnp.allclose(operator.solve(vector), jnp.linalg.solve(dense, vector), atol=1e-8)


def test_low_rank_update_solve_equals_dense_asymmetric() -> None:
    """The Woodbury solve matches the dense solve for ``D + U V^T``."""
    key_d, key_u, key_v, key_vec = jax.random.split(jax.random.PRNGKey(8), 4)
    diagonal = jnp.abs(jax.random.normal(key_d, (7,), dtype=jnp.float64)) + 1.0
    u = jax.random.normal(key_u, (7, 3), dtype=jnp.float64)
    v = jax.random.normal(key_v, (7, 3), dtype=jnp.float64)
    operator = LowRankUpdate(diagonal, u, v)
    dense = operator.to_dense()
    vector = jax.random.normal(key_vec, (7,), dtype=jnp.float64)
    assert jnp.allclose(operator.solve(vector), jnp.linalg.solve(dense, vector), atol=1e-8)
    _, dense_logdet = jnp.linalg.slogdet(dense)
    assert jnp.allclose(operator.logdet(), dense_logdet, atol=1e-8)


# --------------------------------------------------------------------------- #
# Pytree round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "build_operator",
    [
        lambda: KroneckerProduct(jnp.eye(2), 2.0 * jnp.eye(3)),
        lambda: DiagonalOperator(jnp.arange(1, 5, dtype=jnp.float64)),
        lambda: IdentityOperator(4, dtype=jnp.float64),
        lambda: LowRankUpdate(jnp.ones(5), jnp.ones((5, 2))),
        lambda: BlockDiagonal((DiagonalOperator(jnp.ones(2)), DiagonalOperator(jnp.ones(3)))),
    ],
)
def test_pytree_round_trip_preserves_dense(
    build_operator: Callable[[], StructuredOperator],
) -> None:
    """``tree_unflatten(tree_flatten(op))`` reproduces the dense matrix."""
    operator = build_operator()
    leaves, treedef = jax.tree_util.tree_flatten(operator)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.allclose(rebuilt.to_dense(), operator.to_dense())


# --------------------------------------------------------------------------- #
# Transform compatibility: jit / grad / vmap (REQUIRED)
# --------------------------------------------------------------------------- #
def test_kronecker_matvec_is_jit_compatible() -> None:
    """``KroneckerProduct.matvec`` runs under ``jax.jit``."""
    operator = KroneckerProduct(jnp.eye(3) * 2.0, jnp.eye(4) * 3.0)

    @jax.jit
    def apply(vector: jax.Array) -> jax.Array:
        return operator.matvec(vector)

    result = apply(jnp.ones(12))
    assert jnp.allclose(result, 6.0 * jnp.ones(12))


def test_kronecker_logdet_is_grad_compatible() -> None:
    """``KroneckerProduct.logdet`` is differentiable w.r.t. its factors."""

    def logdet_of_scale(scale: jax.Array) -> jax.Array:
        operator = KroneckerProduct(scale * jnp.eye(2), jnp.eye(3))
        return operator.logdet()

    gradient = jax.grad(logdet_of_scale)(jnp.asarray(2.0))
    # logdet((s I_2) ⊗ I_3) = 3 * logdet(s I_2) = 3 * 2 * log s = 6 log s.
    assert jnp.allclose(gradient, 6.0 / 2.0, atol=1e-5)


def test_low_rank_solve_is_grad_compatible() -> None:
    """``LowRankUpdate.solve`` is differentiable through the Woodbury path."""

    def quadratic(diagonal: jax.Array) -> jax.Array:
        operator = LowRankUpdate(diagonal, jnp.ones((4, 1)))
        return jnp.sum(operator.solve(jnp.ones(4)))

    gradient = jax.grad(quadratic)(jnp.full((4,), 2.0))
    assert jnp.all(jnp.isfinite(gradient))


def test_kronecker_matvec_is_vmap_compatible() -> None:
    """A batch of vectors maps through ``KroneckerProduct.matvec``."""
    operator = KroneckerProduct(jnp.eye(2) * 2.0, jnp.eye(2) * 0.5)
    batch = jnp.stack([jnp.ones(4), jnp.arange(4, dtype=jnp.float64)])
    mapped = jax.vmap(operator.matvec)(batch)
    expected = jnp.stack([operator.matvec(batch[0]), operator.matvec(batch[1])])
    assert jnp.allclose(mapped, expected)


def test_low_rank_logdet_is_vmap_compatible() -> None:
    """A batch of diagonals maps through ``LowRankUpdate.logdet``."""
    u = jnp.ones((4, 1))
    diagonals = jnp.stack([jnp.full(4, 2.0), jnp.full(4, 3.0)])

    def logdet_of(diagonal: jax.Array) -> jax.Array:
        return LowRankUpdate(diagonal, u).logdet()

    mapped = jax.vmap(logdet_of)(diagonals)
    expected = jnp.stack([logdet_of(diagonals[0]), logdet_of(diagonals[1])])
    assert jnp.allclose(mapped, expected)
