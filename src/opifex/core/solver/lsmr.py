"""Matrix-free least squares by the LSMR recurrence.

Solves the damped least-squares problem

    min_x || A x - b ||^2  +  damping^2 || x ||^2

where ``A`` is reached only through ``matvec`` and ``matvec_transpose``, by the LSMR
recurrence of Fong and Saunders: a Golub-Kahan bidiagonalisation whose growing system is kept
factorised by two Givens rotations per step, so each iterate minimises the residual over the
Krylov subspace built so far and no normal-equation matrix is ever formed. Forming one, as a
projected solve does, squares the condition number and costs half the digits.

References:
----------
* Fong, Saunders 2011 — *LSMR: An iterative algorithm for sparse
  least-squares problems*, SIAM J. Sci. Comput. 33(5), 2950.
* Roy, Hauberg, Krämer arXiv:2510.19634 — *Matrix-free least squares
  solvers: values, gradients, and what to do with them*.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — imported at runtime by design
from typing import NamedTuple

import jax
import jax.numpy as jnp


class _Rotation(NamedTuple):
    """A Givens rotation, and the length it leaves on the diagonal."""

    cosine: jax.Array
    sine: jax.Array
    radius: jax.Array


def _rotate(first: jax.Array, second: jax.Array) -> _Rotation:
    """The rotation annihilating ``second`` against ``first``; the zero vector stays put."""
    radius = jnp.hypot(first, second)
    safe = jnp.where(radius == 0, 1.0, radius)
    return _Rotation(
        cosine=jnp.where(radius == 0, 1.0, first / safe),
        sine=jnp.where(radius == 0, 0.0, second / safe),
        radius=radius,
    )


def _normalise(vector: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The vector's norm and its direction; an exhausted direction stays zero.

    ``jnp.linalg.norm`` is differentiated as ``v / ||v||`` and so reports NaN for the
    gradient at the origin, which is exactly where an exhausted bidiagonalisation lands. It
    is given a nonzero vector there instead, and the result discarded, so the branch that is
    not taken carries no NaN back through reverse mode.
    """
    exhausted = jnp.all(vector == 0)
    norm = jnp.where(exhausted, 0.0, jnp.linalg.norm(jnp.where(exhausted, 1.0, vector)))
    return norm, jnp.where(exhausted, vector, vector / jnp.where(exhausted, 1.0, norm))


def _divide(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """``numerator / denominator``, reading zero where the denominator has run out.

    The rotations leave zero on the diagonal once the Krylov space is exhausted, which a
    zero right-hand side reaches immediately and an exactly solvable one after ``rank``
    steps. The iterate has converged there, so the step is zero rather than ``0 / 0``. The
    denominator is substituted inside the division as well as outside it, so the branch not
    taken holds no division by zero for reverse-mode to differentiate.
    """
    exhausted = denominator == 0
    return jnp.where(exhausted, 0.0, numerator / jnp.where(exhausted, 1.0, denominator))


class _State(NamedTuple):
    """One LSMR iterate, in the notation of Fong and Saunders section 2."""

    left: jax.Array
    right: jax.Array
    alpha: jax.Array
    alpha_bar: jax.Array
    zeta_bar: jax.Array
    rho: jax.Array
    rho_bar: jax.Array
    cosine_bar: jax.Array
    sine_bar: jax.Array
    direction: jax.Array
    direction_bar: jax.Array
    solution: jax.Array


def lsmr(
    *,
    matvec: Callable[[jax.Array], jax.Array],
    matvec_transpose: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    dim_cols: int,
    num_matvecs: int,
    damping: float = 0.0,
) -> jax.Array:
    """Solve ``min ||A x - rhs||^2 + damping^2 ||x||^2`` by the LSMR recurrence.

    Args:
        matvec: callable computing ``A @ v`` for ``v`` of length ``dim_cols``.
        matvec_transpose: callable computing ``A.T @ u`` for ``u`` of
            length ``len(rhs)``.
        rhs: right-hand-side vector ``b``.
        dim_cols: number of columns of ``A`` (static); ``x`` has this length.
        num_matvecs: number of LSMR iterations (static). Each iterate minimises the
            residual over the Krylov subspace built so far; in exact arithmetic ``rank``
            steps reach the least-squares solution. Rounding costs the bidiagonalisation
            its orthogonality, so an ill-conditioned system needs more: a 60x10 system of
            condition 1e4 converges in 22 float64 steps, and in float32 the iterates drift
            away instead (relative error 0.8 at 10 steps, 36 at 40).
        damping: Tikhonov damping coefficient, ``0`` for plain least squares.

    Returns:
        Solution ``x`` of shape ``(dim_cols,)``.
    """
    damping_value = jnp.asarray(damping, dtype=rhs.dtype)
    beta, left = _normalise(rhs)
    alpha, right = _normalise(matvec_transpose(left))
    zero = jnp.zeros((), dtype=rhs.dtype)
    one = jnp.ones((), dtype=rhs.dtype)
    start = _State(
        left=left,
        right=right,
        alpha=alpha,
        alpha_bar=alpha,
        zeta_bar=alpha * beta,
        rho=one,
        rho_bar=one,
        cosine_bar=one,
        sine_bar=zero,
        direction=right,
        direction_bar=jnp.zeros((dim_cols,), dtype=rhs.dtype),
        solution=jnp.zeros((dim_cols,), dtype=rhs.dtype),
    )

    def step(_index: jax.Array, current: _State) -> _State:
        beta_next, left = _normalise(matvec(current.right) - current.alpha * current.left)
        alpha_next, right = _normalise(matvec_transpose(left) - beta_next * current.right)

        # Damping enters as an extra row, annihilated before the bidiagonal one.
        damped = _rotate(current.alpha_bar, damping_value)
        rotation = _rotate(damped.radius, beta_next)
        theta_next = rotation.sine * alpha_next
        alpha_bar = rotation.cosine * alpha_next

        theta_bar = current.sine_bar * rotation.radius
        rotation_bar = _rotate(current.cosine_bar * rotation.radius, theta_next)
        zeta = rotation_bar.cosine * current.zeta_bar
        zeta_bar = -rotation_bar.sine * current.zeta_bar

        scale = _divide(theta_bar * rotation.radius, current.rho * current.rho_bar)
        direction_bar = current.direction - scale * current.direction_bar
        step_length = _divide(zeta, rotation.radius * rotation_bar.radius)
        solution = current.solution + step_length * direction_bar
        direction = right - _divide(theta_next, rotation.radius) * current.direction

        return _State(
            left=left,
            right=right,
            alpha=alpha_next,
            alpha_bar=alpha_bar,
            zeta_bar=zeta_bar,
            rho=rotation.radius,
            rho_bar=rotation_bar.radius,
            cosine_bar=rotation_bar.cosine,
            sine_bar=rotation_bar.sine,
            direction=direction,
            direction_bar=direction_bar,
            solution=solution,
        )

    return jax.lax.fori_loop(0, num_matvecs, step, start).solution


__all__ = ["lsmr"]
