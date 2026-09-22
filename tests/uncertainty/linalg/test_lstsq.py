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

    def test_a_rank_deficient_system_gets_the_minimum_norm_solution(self) -> None:
        # The property the pressure projection depends on: where the operator is singular,
        # LSMR picks the solution of least norm rather than any least-squares solution, so
        # the iterate never wanders into the null space.
        from scipy.sparse.linalg import lsmr as scipy_lsmr

        matrix, _ = self._system(12, 4)
        repeated = jnp.hstack([matrix, matrix[:, :1]])
        rhs = repeated @ jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

        reference = scipy_lsmr(np.asarray(repeated, dtype=np.float64), np.asarray(rhs))[0]
        solution = lsmr(rhs=rhs, num_matvecs=200, **_matrix_free(repeated))

        np.testing.assert_allclose(solution, reference, rtol=1e-4, atol=1e-5)

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


class TestAnExhaustedKrylovSpace:
    """Steps taken after the bidiagonalisation has nothing left to build on.

    The recurrence divides by the lengths the rotations leave on the diagonal, and those
    reach zero once the Krylov space is exhausted -- immediately for a zero right-hand side,
    and after ``rank`` steps for any system solved exactly. The iterate is converged there,
    so the remaining steps must add nothing rather than divide zero by zero.
    """

    @staticmethod
    def _system(rows: int, columns: int) -> jax.Array:
        return jax.random.normal(jax.random.key(0), (rows, columns))

    def test_a_zero_right_hand_side_gives_a_zero_solution(self) -> None:
        matrix = self._system(12, 4)

        solution = lsmr(rhs=jnp.zeros(12), num_matvecs=10, **_matrix_free(matrix))

        assert jnp.all(jnp.isfinite(solution))
        np.testing.assert_array_equal(solution, jnp.zeros(4))

    def test_steps_past_an_exact_solution_leave_it_where_it_is(self) -> None:
        # Rounding leaves the last rotations a residue to work off, so the space exhausts
        # well after `rank` and at no fixed step: across six seeds and three sizes the
        # iterate settled anywhere between 10 and 35 steps on systems of 4 to 8 columns.
        # Both counts below are past that range, which is what makes the comparison a
        # statement about exhaustion rather than about where this system happened to reach
        # it.
        matrix = self._system(6, 6)
        truth = jnp.arange(6, dtype=matrix.dtype)
        arguments = _matrix_free(matrix)

        exhausted = lsmr(rhs=matrix @ truth, num_matvecs=50, **arguments)
        long_after = lsmr(rhs=matrix @ truth, num_matvecs=100, **arguments)

        assert jnp.all(jnp.isfinite(long_after))
        np.testing.assert_array_equal(long_after, exhausted)
        np.testing.assert_allclose(long_after, truth, atol=1e-5)

    def test_the_gradient_stays_finite_where_the_space_is_exhausted(self) -> None:
        matrix = self._system(12, 4)
        arguments = _matrix_free(matrix)

        gradient = jax.grad(lambda b: jnp.sum(lsmr(rhs=b, num_matvecs=10, **arguments) ** 2))(
            jnp.zeros(12)
        )

        assert jnp.all(jnp.isfinite(gradient))

    def test_it_maps_over_right_hand_sides_including_a_degenerate_one(self) -> None:
        matrix = self._system(12, 4)
        arguments = _matrix_free(matrix)
        batch = jnp.stack([jnp.zeros(12), matrix @ jnp.ones(4)])

        solutions = jax.vmap(lambda b: lsmr(rhs=b, num_matvecs=10, **arguments))(batch)

        assert jnp.all(jnp.isfinite(solutions))
        np.testing.assert_array_equal(solutions[0], jnp.zeros(4))
        np.testing.assert_allclose(solutions[1], jnp.ones(4), rtol=1e-4, atol=1e-5)
