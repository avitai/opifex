"""Tests for matrix-free least-squares (LSMR).

References
----------
* Fong, Saunders 2011 — *LSMR: An iterative algorithm for sparse least-squares
  problems*, SIAM J. Sci. Comput.
* Roy, Hauberg, Krämer arXiv:2510.19634 — custom gradients for matrix-free
  least-squares solvers.

opifex's implementation provides the forward pass via Golub-Kahan
bidiagonalisation + small least-squares solve; differentiable VJP
support is a separate task (custom_vjp is non-trivial here).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — imported at runtime by design
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from opifex.uncertainty.linalg import lsmr


def test_lsmr_solves_overdetermined_least_squares() -> None:
    """LSMR finds the minimum-residual solution of ``Ax = b`` for tall ``A``.

    Cite: Fong, Saunders 2011 §3.
    """
    rng = jax.random.PRNGKey(0)
    matrix = jax.random.normal(rng, (8, 4))
    true_solution = jax.random.normal(jax.random.PRNGKey(1), (4,))
    rhs = matrix @ true_solution

    def matvec(vec: jax.Array) -> jax.Array:
        return matrix @ vec

    def matvec_t(vec: jax.Array) -> jax.Array:
        return matrix.T @ vec

    solution = lsmr(
        matvec=matvec,
        matvec_transpose=matvec_t,
        rhs=rhs,
        dim_cols=4,
        num_matvecs=4,
    )
    assert jnp.allclose(solution, true_solution, atol=1e-3)


def test_lsmr_matches_normal_equations_solution() -> None:
    """LSMR solution matches ``(A.T A)^{-1} A.T b`` on a small problem."""
    rng = jax.random.PRNGKey(2)
    matrix = jax.random.normal(rng, (6, 4))
    rhs = jax.random.normal(jax.random.PRNGKey(3), (6,))

    def matvec(vec: jax.Array) -> jax.Array:
        return matrix @ vec

    def matvec_t(vec: jax.Array) -> jax.Array:
        return matrix.T @ vec

    reference = jnp.linalg.solve(matrix.T @ matrix, matrix.T @ rhs)
    solution = lsmr(
        matvec=matvec,
        matvec_transpose=matvec_t,
        rhs=rhs,
        dim_cols=4,
        num_matvecs=4,
    )
    assert jnp.allclose(solution, reference, atol=1e-3)


def test_lsmr_damping_shrinks_solution_toward_zero() -> None:
    """The damped LSMR solution has smaller norm than the undamped one.

    Cite: Fong, Saunders 2011. Damping ``λ > 0`` regularises the system
    ``min ||Ax - b||² + λ² ||x||²``; the solution norm decreases as ``λ``
    grows.
    """
    rng = jax.random.PRNGKey(11)
    matrix = jax.random.normal(rng, (6, 4))
    rhs = jax.random.normal(jax.random.PRNGKey(12), (6,))

    def matvec(vec: jax.Array) -> jax.Array:
        return matrix @ vec

    def matvec_t(vec: jax.Array) -> jax.Array:
        return matrix.T @ vec

    undamped = lsmr(
        matvec=matvec,
        matvec_transpose=matvec_t,
        rhs=rhs,
        dim_cols=4,
        num_matvecs=4,
        damping=0.0,
    )
    damped = lsmr(
        matvec=matvec,
        matvec_transpose=matvec_t,
        rhs=rhs,
        dim_cols=4,
        num_matvecs=4,
        damping=2.0,
    )
    assert jnp.linalg.norm(damped) < jnp.linalg.norm(undamped)


def test_lsmr_is_jit_compatible() -> None:
    """LSMR compiles under ``jax.jit``."""
    rng = jax.random.PRNGKey(21)
    matrix = jax.random.normal(rng, (5, 3))

    def matvec(vec: jax.Array) -> jax.Array:
        return matrix @ vec

    def matvec_t(vec: jax.Array) -> jax.Array:
        return matrix.T @ vec

    def call(rhs: jax.Array) -> jax.Array:
        return lsmr(
            matvec=matvec,
            matvec_transpose=matvec_t,
            rhs=rhs,
            dim_cols=3,
            num_matvecs=3,
        )

    jitted = jax.jit(call)
    solution = jitted(jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert solution.shape == (3,)
    assert jnp.all(jnp.isfinite(solution))


class _MatrixFree(TypedDict):
    """The ``lsmr`` arguments that reach a matrix only through products."""

    matvec: Callable[[jax.Array], jax.Array]
    matvec_transpose: Callable[[jax.Array], jax.Array]
    dim_cols: int


def _matrix_free(matrix: jax.Array) -> _MatrixFree:
    """``lsmr`` arguments that reach ``matrix`` only through products."""
    return _MatrixFree(
        matvec=lambda v: matrix @ v,
        matvec_transpose=lambda u: matrix.T @ u,
        dim_cols=matrix.shape[1],
    )


class TestAgainstTheReferenceImplementation:
    """The solution SciPy's ``lsmr`` reports, which is Fong and Saunders' own code."""

    @staticmethod
    def _system(
        rows: int, columns: int, *, condition: float = 1.0, dtype: jnp.dtype = jnp.float32
    ) -> tuple[jax.Array, jax.Array]:
        left = jax.random.normal(jax.random.key(0), (rows, columns), dtype)
        scale = jnp.logspace(0.0, -jnp.log10(condition), columns, dtype=dtype)
        return left * scale, jax.random.normal(jax.random.key(1), (rows,), dtype)

    def test_it_matches_scipy_on_a_well_conditioned_system(self) -> None:
        from scipy.sparse.linalg import lsmr as scipy_lsmr

        matrix, rhs = self._system(30, 6)
        reference = scipy_lsmr(np.asarray(matrix), np.asarray(rhs), atol=1e-12, btol=1e-12)[0]

        solution = lsmr(rhs=rhs, num_matvecs=6, **_matrix_free(matrix))

        np.testing.assert_allclose(solution, reference, rtol=1e-4, atol=1e-5)

    def test_it_matches_scipy_when_damped(self) -> None:
        from scipy.sparse.linalg import lsmr as scipy_lsmr

        matrix, rhs = self._system(40, 8)
        damping = 0.5
        reference = scipy_lsmr(
            np.asarray(matrix), np.asarray(rhs), damp=damping, atol=1e-12, btol=1e-12
        )[0]

        solution = lsmr(rhs=rhs, num_matvecs=8, damping=damping, **_matrix_free(matrix))

        np.testing.assert_allclose(solution, reference, rtol=1e-4, atol=1e-5)

    def test_an_ill_conditioned_system_converges_in_double_precision(self) -> None:
        """Condition 1e4 needs more steps than columns, and needs float64 to keep them."""
        from scipy.sparse.linalg import lsmr as scipy_lsmr

        with jax.enable_x64(True):
            matrix, rhs = self._system(60, 10, condition=1e4, dtype=jnp.float64)
            # SciPy stops at min(rows, columns) steps by default, which this system
            # outruns: it needs 22.
            reference = scipy_lsmr(
                np.asarray(matrix), np.asarray(rhs), atol=1e-14, btol=1e-14, maxiter=200
            )[0]

            solution = lsmr(rhs=rhs, num_matvecs=30, **_matrix_free(matrix))

            np.testing.assert_allclose(solution, reference, rtol=1e-8, atol=1e-10)

    def test_more_steps_than_columns_are_needed_when_the_system_is_ill_conditioned(
        self,
    ) -> None:
        with jax.enable_x64(True):
            matrix, rhs = self._system(60, 10, condition=1e4, dtype=jnp.float64)
            arguments = _matrix_free(matrix)
            residual = lambda x: float(jnp.linalg.norm(matrix @ x - rhs))

            at_rank = residual(lsmr(rhs=rhs, num_matvecs=10, **arguments))
            past_rank = residual(lsmr(rhs=rhs, num_matvecs=30, **arguments))

        assert past_rank < at_rank

    def test_it_traces_under_jit(self) -> None:
        matrix, rhs = self._system(30, 6)
        arguments = _matrix_free(matrix)

        compiled = jax.jit(
            lambda b: lsmr(rhs=b, num_matvecs=6, **arguments),
        )(rhs)

        np.testing.assert_allclose(compiled, lsmr(rhs=rhs, num_matvecs=6, **arguments), rtol=1e-6)
