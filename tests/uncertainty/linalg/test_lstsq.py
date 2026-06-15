"""Tests for matrix-free least-squares (LSMR).

References
----------
* Fong, Saunders 2011 — *LSMR: An iterative algorithm for sparse least-squares
  problems*, SIAM J. Sci. Comput.
* Roy et al. arXiv:2510.19634 — differentiable LSMR via custom VJP.

Sibling reference: ``matfree/matfree/lstsq.py`` (full LSMR with
``custom_vjp`` for low-memory exact reverse-mode gradients). opifex's
implementation provides the forward pass via Golub-Kahan
bidiagonalisation + small least-squares solve; differentiable VJP
support is a separate task (custom_vjp is non-trivial here).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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
