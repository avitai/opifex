"""Matrix-free least-squares and linear solves by the LSMR recurrence.

Solves the damped least-squares problem

    min_x || A x - b ||^2  +  damping^2 || x ||^2

where ``A`` is reached only through ``matvec`` and ``matvec_transpose``, by the LSMR
recurrence of Fong and Saunders: a Golub-Kahan bidiagonalisation whose growing system is kept
factorised by two Givens rotations per step, so each iterate minimises the residual over the
Krylov subspace built so far and no normal-equation matrix is ever formed. Forming one, as a
projected solve does, squares the condition number and costs half the digits.

Both the fixed-budget differentiable solver :func:`lsmr` (:func:`jax.lax.fori_loop`) and the
stopping-criterion driver ``_solve_unit_lsmr_stopping`` (:func:`jax.lax.while_loop` inside
:func:`jax.lax.custom_linear_solve`) advance the same single-step recurrence.

References:
----------
* Fong, D. C.-L., & Saunders, M. (2011). *LSMR: An iterative algorithm for sparse
  least-squares problems*, SIAM J. Sci. Comput. 33(5), 2950-2971.
* Choi, S.-C. T. (2006). *Iterative Methods for Singular Linear Equations and
  Least-Squares Problems*, PhD thesis, Stanford University.
* Roy, F., Hauberg, S., & Krämer, N. (2025). *Matrix-free least squares
  solvers: values, gradients, and what to do with them*, arXiv:2510.19634.
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
    is_zero = radius == 0.0
    safe = jnp.where(is_zero, jnp.ones_like(radius), radius)
    return _Rotation(
        cosine=jnp.where(is_zero, jnp.ones_like(first), first / safe),
        sine=jnp.where(is_zero, jnp.zeros_like(second), second / safe),
        radius=radius,
    )


def _normalise(vector: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The vector's real Euclidean norm and unit direction; an exhausted direction stays zero.

    ``jnp.linalg.norm`` is differentiated as ``v / ||v||`` and so reports NaN for the
    gradient at the origin, which is exactly where an exhausted bidiagonalisation lands. It
    is given a nonzero vector there instead, and the result discarded, so the branch that is
    not taken carries no NaN back through reverse mode.
    """
    real_dtype = jnp.finfo(vector.dtype).dtype
    exhausted = jnp.all(vector == 0)
    safe_vec = jnp.where(exhausted, jnp.ones_like(vector), vector)
    raw_norm = jnp.linalg.norm(safe_vec).astype(real_dtype)
    norm = jnp.where(exhausted, jnp.zeros((), dtype=real_dtype), raw_norm)
    safe_norm = jnp.where(exhausted, jnp.ones((), dtype=real_dtype), norm).astype(vector.dtype)
    direction = jnp.where(exhausted, jnp.zeros_like(vector), vector / safe_norm)
    return norm, direction


def _divide(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    """``numerator / denominator``, reading zero where the denominator has run out.

    The rotations leave zero on the diagonal once the Krylov space is exhausted, which a
    zero right-hand side reaches immediately and an exactly solvable one after ``rank``
    steps. The iterate has converged there, so the step is zero rather than ``0 / 0``. The
    denominator is substituted inside the division as well as outside it, so the branch not
    taken holds no division by zero for reverse-mode to differentiate.
    """
    exhausted = denominator == 0.0
    safe = jnp.where(exhausted, jnp.ones_like(denominator), denominator)
    return jnp.where(exhausted, jnp.zeros_like(numerator), numerator / safe)


class _CoreState(NamedTuple):
    """One LSMR iterate, in the notation of Fong and Saunders (2011, section 2)."""

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


class _StepTransition(NamedTuple):
    """Updated core iterate plus real bidiagonal scalars needed by stopping rules."""

    state: _CoreState
    beta_next: jax.Array
    alpha_next: jax.Array
    rotation: _Rotation
    theta_bar: jax.Array
    rho_temp: jax.Array
    zeta: jax.Array


def _init_core_state(
    matvec_transpose: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    dim_cols: int,
) -> tuple[_CoreState, jax.Array]:
    """Initialise Golub-Kahan bidiagonalisation and return ``(state0, beta0)``."""
    real_dtype = jnp.finfo(rhs.dtype).dtype
    beta, left = _normalise(rhs)
    v_raw = jnp.where(
        beta == 0.0,
        jnp.zeros((dim_cols,), dtype=rhs.dtype),
        matvec_transpose(left),
    )
    alpha, right = _normalise(v_raw)
    zero = jnp.zeros((), dtype=real_dtype)
    one = jnp.ones((), dtype=real_dtype)
    state = _CoreState(
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
    return state, beta


def _advance_core_step(
    current: _CoreState,
    matvec: Callable[[jax.Array], jax.Array],
    matvec_transpose: Callable[[jax.Array], jax.Array],
    damping_value: jax.Array,
) -> _StepTransition:
    """Advance one Golub-Kahan bidiagonalisation and double-QR step."""
    vec_dtype = current.solution.dtype
    u_cand = matvec(current.right) - current.alpha.astype(vec_dtype) * current.left
    beta_next, left_raw = _normalise(u_cand)
    has_b = beta_next > 0.0
    left = jnp.where(has_b, left_raw, u_cand)
    v_cand = jnp.where(
        has_b,
        matvec_transpose(left) - beta_next.astype(vec_dtype) * current.right,
        current.right,
    )
    alpha_raw, right_raw = _normalise(v_cand)
    alpha_next = jnp.where(has_b, alpha_raw, current.alpha)
    right = jnp.where(has_b & (alpha_raw > 0.0), right_raw, v_cand)

    # Damping enters as an extra row, annihilated before the bidiagonal one.
    damped = _rotate(current.alpha_bar, damping_value)
    rotation = _rotate(damped.radius, beta_next)
    theta_next = rotation.sine * alpha_next
    alpha_bar = rotation.cosine * alpha_next

    theta_bar = current.sine_bar * rotation.radius
    rho_temp = current.cosine_bar * rotation.radius
    rotation_bar = _rotate(rho_temp, theta_next)
    zeta = rotation_bar.cosine * current.zeta_bar
    zeta_bar = -rotation_bar.sine * current.zeta_bar

    scale = _divide(theta_bar * rotation.radius, current.rho * current.rho_bar)
    direction_bar = current.direction - scale.astype(vec_dtype) * current.direction_bar
    step_length = _divide(zeta, rotation.radius * rotation_bar.radius)
    solution = current.solution + step_length.astype(vec_dtype) * direction_bar
    dir_scale = _divide(theta_next, rotation.radius)
    direction = right - dir_scale.astype(vec_dtype) * current.direction

    next_state = _CoreState(
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
    return _StepTransition(
        state=next_state,
        beta_next=beta_next,
        alpha_next=alpha_next,
        rotation=rotation,
        theta_bar=theta_bar,
        rho_temp=rho_temp,
        zeta=zeta,
    )


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
        matvec: Callable computing ``A @ v`` for ``v`` of length ``dim_cols``.
        matvec_transpose: Callable computing ``A^H @ u`` for ``u`` of
            length ``len(rhs)``.
        rhs: Right-hand-side vector ``b``.
        dim_cols: Number of columns of ``A`` (static); ``x`` has this length.
        num_matvecs: Number of LSMR iterations (static). Each iterate minimises the
            residual over the Krylov subspace built so far; in exact arithmetic ``rank``
            steps reach the least-squares solution. Rounding costs the bidiagonalisation
            its orthogonality, so an ill-conditioned system needs more: a 60x10 system of
            condition 1e4 converges in 22 float64 steps, and in float32 the iterates drift
            away instead (relative error 0.8 at 10 steps, 36 at 40).
        damping: Tikhonov damping coefficient, ``0`` for plain least squares.

    Returns:
        Solution ``x`` of shape ``(dim_cols,)``.
    """
    real_dtype = jnp.finfo(rhs.dtype).dtype
    damping_value = jnp.asarray(damping, dtype=real_dtype)
    start, _ = _init_core_state(matvec_transpose, rhs, dim_cols)

    def step(_index: jax.Array, current: _CoreState) -> _CoreState:
        return _advance_core_step(current, matvec, matvec_transpose, damping_value).state

    return jax.lax.fori_loop(0, num_matvecs, step, start).solution


class _StoppingState(NamedTuple):
    """Adaptive Fong & Saunders (2011, §2.5-2.8) residual and condition monitors."""

    core: _CoreState
    itn: jax.Array
    betadd: jax.Array
    betad: jax.Array
    rhodold: jax.Array
    tautildeold: jax.Array
    thetatilde: jax.Array
    zeta: jax.Array
    norm_a2: jax.Array
    maxrbar: jax.Array
    minrbar: jax.Array
    cond_a: jax.Array
    istop: jax.Array
    normr: jax.Array


def solve_unit_lsmr_stopping(
    matvec: Callable[[jax.Array], jax.Array],
    matvec_transpose: Callable[[jax.Array], jax.Array],
    unit_b: jax.Array,
    *,
    rtol: float,
    max_steps: int,
    conlim: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run undamped LSMR with Fong & Saunders (2011) stopping criteria S1-S3.

    Args:
        matvec: Forward linear operator ``v -> A v``.
        matvec_transpose: Hermitian-adjoint operator ``u -> A^H u``.
        unit_b: Normalized 1-D right-hand side vector.
        rtol: Relative residual tolerance for stopping rules S1 and S2.
        max_steps: Maximum Golub-Kahan bidiagonalization iterations.
        conlim: Upper bound on the bidiagonal condition estimate ``acond``.

    Returns:
        Tuple ``(solution, frobenius_norm_est, cond_est)``.
    """
    real_dtype = jnp.finfo(unit_b.dtype).dtype
    core0, beta0 = _init_core_state(matvec_transpose, unit_b, unit_b.shape[0])
    alpha0 = core0.alpha
    one = jnp.ones((), dtype=real_dtype)
    zero = jnp.zeros((), dtype=real_dtype)
    istop0 = jnp.where(
        beta0 == 0.0,
        jnp.int32(1),
        jnp.where(alpha0 == 0.0, jnp.int32(2), jnp.int32(0)),
    )
    init = _StoppingState(
        core=core0,
        itn=jnp.int32(0),
        betadd=beta0,
        betad=zero,
        rhodold=one,
        tautildeold=zero,
        thetatilde=zero,
        zeta=zero,
        norm_a2=alpha0 * alpha0,
        maxrbar=zero,
        minrbar=jnp.asarray(jnp.finfo(real_dtype).max, dtype=real_dtype),
        cond_a=one,
        istop=istop0,
        normr=beta0,
    )

    def body_fn(st: _StoppingState) -> _StoppingState:
        itn = st.itn + jnp.int32(1)
        trans = _advance_core_step(st.core, matvec, matvec_transpose, zero)
        nxt = trans.state

        betahat = trans.rotation.cosine * st.betadd
        betadd = -trans.rotation.sine * st.betadd
        rot_tilde = _rotate(st.rhodold, trans.theta_bar)
        thetatilde = rot_tilde.sine * nxt.rho_bar
        rhodold = rot_tilde.cosine * nxt.rho_bar
        betad = -rot_tilde.sine * st.betad + rot_tilde.cosine * betahat
        tautilde = _divide(st.zeta - st.thetatilde * st.tautildeold, rot_tilde.radius)
        taud = _divide(trans.zeta - thetatilde * tautilde, rhodold)
        normr = jnp.hypot(betad - taud, betadd)

        norm_a2_mid = st.norm_a2 + trans.beta_next * trans.beta_next
        norm_a = jnp.sqrt(norm_a2_mid)
        norm_a2_next = norm_a2_mid + trans.alpha_next * trans.alpha_next

        maxrbar = jnp.maximum(st.maxrbar, st.core.rho_bar)
        minrbar = jnp.where(itn > 1, jnp.minimum(st.minrbar, st.core.rho_bar), st.minrbar)
        denom_c = jnp.minimum(minrbar, trans.rho_temp)
        cond_a = jnp.maximum(maxrbar, trans.rho_temp) / jnp.where(denom_c > 0.0, denom_c, one)

        sol_norm = jnp.linalg.norm(nxt.solution).astype(real_dtype)
        well_posed_tol = rtol * (norm_a * sol_norm + one)
        istop = jnp.where(itn >= max_steps, jnp.int32(4), st.istop)
        istop = jnp.where(cond_a > conlim, jnp.int32(3), istop)
        istop = jnp.where(jnp.abs(nxt.zeta_bar) < rtol * norm_a * normr, jnp.int32(2), istop)
        istop = jnp.where(normr < well_posed_tol, jnp.int32(1), istop)

        return _StoppingState(
            core=nxt,
            itn=itn,
            betadd=betadd,
            betad=betad,
            rhodold=rhodold,
            tautildeold=tautilde,
            thetatilde=thetatilde,
            zeta=trans.zeta,
            norm_a2=norm_a2_next,
            maxrbar=maxrbar,
            minrbar=minrbar,
            cond_a=cond_a,
            istop=istop,
            normr=normr,
        )

    final = jax.lax.while_loop(lambda s: s.istop == 0, body_fn, init)
    return final.core.solution, jnp.sqrt(final.norm_a2), final.cond_a


__all__ = ["lsmr", "solve_unit_lsmr_stopping"]
