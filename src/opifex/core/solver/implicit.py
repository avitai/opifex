r"""Shared implicit-differentiation, tangent linear solvers, and dtype utilities.

Provides three user-selectable tangent linear solvers for :func:`jax.lax.custom_root`,
all wrapped in :func:`jax.lax.custom_linear_solve` and returning ``NaN`` on singular,
ill-conditioned, or stagnated linear systems rather than silent inaccurate tangents:

1. :func:`dense_tangent_solve` (default): materializes and partial-pivoting LU-factors
   the dense Jacobian once per linear operator (even when the right-hand side is batched
   under :func:`jax.vmap` or :func:`jax.jacrev`), enforcing an :math:`O(n^2)` alternating
   triangular-solve conditioning policy and a Rigal-Gaches (1967; Higham 2002, Thm. 7.1)
   normwise backward-error acceptance contract.
2. :func:`lsmr_tangent_solve`: matrix-free Golub-Kahan bidiagonalization (Fong & Saunders
   2011) via :mod:`opifex.core.solver.lsmr` and :func:`jax.linear_transpose` without
   forming :math:`A^\top A` (so :math:`\kappa(A)` is never squared), converging on
   indefinite non-symmetric spectra where restarted GMRES stagnates.
3. :func:`gmres_tangent_solve`: matrix-free restarted Arnoldi GMRES (Saad & Schultz 1986)
   with explicit stagnation-counter early termination and normwise backward-error
   verification.

References:
    * Rigal, J. L., & Gaches, J. (1967). "On the compatibility of a given solution with
      the data of a linear system." *J. ACM* 14(3), 543-548.
    * Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.).
      SIAM, §7.1 (Thm. 7.1) & §9.3 (Thm. 9.4).
    * Fong, D. C.-L., & Saunders, M. (2011). "LSMR: An iterative algorithm for sparse
      least-squares problems." *SIAM J. Sci. Comput.* 33(5), 2950-2971.
    * Saad, Y., & Schultz, M. H. (1986). "GMRES: A generalized minimal residual algorithm
      for solving nonsymmetric linear systems." *SIAM J. Sci. Stat. Comput.* 7(3), 856-869.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from jax.flatten_util import ravel_pytree

from opifex.core.solver.lsmr import solve_unit_lsmr_stopping
from opifex.core.solver.status import Status


TangentSolverFn = Callable[[Callable[[Any], Any], Any], Any]


def _is_unary_when_args_omitted(fn: Callable[..., Any]) -> bool:
    """Return ``True`` when ``fn`` should be called as ``fn(params)`` if ``args is None``."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params_list = list(sig.parameters.values())
    pos_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    positional = [param for param in params_list if param.kind in pos_kinds]
    if len(positional) == 1:
        return True
    if not positional:
        return any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params_list)
    required_count = sum(param.default is inspect.Parameter.empty for param in positional)
    return required_count == 1 and positional[1].default is not None


def call_unary_or_binary(
    fn: Callable[..., Any],
    params: Any,
    args: Any,
) -> Any:
    """Invoke ``fn(params)`` when ``args is None`` and ``fn`` is unary, else ``fn(params, args)``.

    Args:
        fn: Callable accepting ``(params,)`` or ``(params, args)``.
        params: Primary parameter pytree.
        args: Optional auxiliary argument pytree.

    Returns:
        Raw output of ``fn``.
    """
    if args is None and _is_unary_when_args_omitted(fn):
        return fn(params)
    return fn(params, args)


@jax.custom_jvp
def guard_unconverged_tangents(value: jax.Array, is_converged: jax.Array) -> jax.Array:
    """Return ``value`` unchanged while poisoning forward/reverse tangents if unconverged.

    Args:
        value: Flattened solver output vector ``x*``.
        is_converged: Scalar boolean array indicating root/stationarity convergence.

    Returns:
        Primal ``value`` unchanged; ``NaN`` tangents/cotangents when ``~is_converged``.
    """
    del is_converged
    return value


def _guard_unconverged_tangents_jvp(
    primals: tuple[jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Multiply tangent vectors by ``1`` when converged and ``NaN`` otherwise."""
    value, is_converged = primals
    value_dot, _ = tangents
    scale = jnp.where(
        is_converged,
        jnp.ones((), dtype=value_dot.dtype),
        jnp.asarray(jnp.nan, dtype=value_dot.dtype),
    )
    return value, scale * value_dot


guard_unconverged_tangents.defjvp(_guard_unconverged_tangents_jvp)


def loop_status(*, has_nonfinite: jax.Array, converged: jax.Array) -> jax.Array:
    """Map loop termination predicates to a scalar ``int32`` :class:`Status` array.

    Args:
        has_nonfinite: Boolean array indicating whether any non-finite value appeared.
        converged: Boolean array indicating whether the convergence criterion was met.

    Returns:
        Scalar ``int32`` status code (``NONFINITE``, ``SUCCESS``, or ``MAX_ITERS``).
    """
    return jnp.where(
        has_nonfinite,
        jnp.asarray(int(Status.NONFINITE), dtype=jnp.int32),
        jnp.where(
            converged,
            jnp.asarray(int(Status.SUCCESS), dtype=jnp.int32),
            jnp.asarray(int(Status.MAX_ITERS), dtype=jnp.int32),
        ),
    )


def _is_inexact_leaf(leaf: Any) -> bool:
    """Return ``True`` when ``leaf`` is a floating or complex scalar or array."""
    if isinstance(leaf, bool):
        return False
    if isinstance(leaf, float | complex):
        return True
    if isinstance(leaf, jax.Array | np.ndarray | np.number):
        return bool(jax.dtypes.issubdtype(leaf.dtype, jnp.inexact))
    return False


def infer_inexact_dtype(*pytrees: Any) -> jnp.dtype:
    """Resolve the common floating/complex dtype across ``pytrees``, floored at ``float32``.

    Preserves JAX weak-type promotion for Python ``float``/``complex`` scalars and
    weakly typed arrays so ``float32`` inputs stay ``float32`` when ``jax_enable_x64``
    is active.

    Args:
        *pytrees: Arbitrary pytrees whose inexact leaves determine the working dtype.

    Returns:
        Promoted floating/complex ``jnp.dtype`` (at least ``float32`` / ``complex64``).
    """
    inexact_leaves = [leaf for leaf in jax.tree.leaves(pytrees) if _is_inexact_leaf(leaf)]
    if not inexact_leaves:
        return jnp.dtype(jnp.float32)
    return jnp.promote_types(jnp.result_type(*inexact_leaves), jnp.float32)


def cast_inexact_pytree(pytree: Any, dtype: jnp.dtype) -> Any:
    """Cast all inexact leaves of ``pytree`` to ``dtype``, leaving other leaves unchanged.

    Args:
        pytree: Arbitrary pytree of parameters or state arrays.
        dtype: Target floating or complex ``jnp.dtype``.

    Returns:
        Pytree with identical structure whose inexact leaves are ``jax.Array`` of ``dtype``.
    """
    return jax.tree.map(
        lambda leaf: jnp.asarray(leaf, dtype=dtype) if _is_inexact_leaf(leaf) else leaf,
        pytree,
    )


def _real_unit_roundoff(dtype: jnp.dtype) -> float:
    """Return unit roundoff ``u = 0.5 * eps`` of the real part of ``dtype``."""
    real_dtype = jnp.finfo(dtype).dtype
    return 0.5 * float(jnp.finfo(real_dtype).eps)


def _lu_backward_error_ceiling(size: int, unit_roundoff: float) -> float:
    r"""Solver acceptance contract on the Rigal-Gaches (1967) normwise backward error.

    Defines the maximum normwise backward error

    .. math::

        \eta(\hat{x}) = \frac{\|A\hat{x} - b\|_2}
        {(\|A\|_F / \sqrt{n})\,\|\hat{x}\|_2 + \|b\|_2}

    accepted by :func:`dense_tangent_solve`, :func:`lsmr_tangent_solve`, and
    :func:`gmres_tangent_solve` before declaring the linear solve inaccurate and
    returning ``NaN``. Note that Higham's worst-case partial-pivoting LU theorem
    (2002, Thm. 9.4) bounds :math:`|\Delta A| \le \gamma_{3n}|L||U|`, which scales as
    :math:`\rho_n n^{5/2} \gamma_{3n}` in this denominator and exceeds ``1`` at
    moderate :math:`n`; the tighter :math:`\min(\max(64\,u, 8\sqrt{n}\,\gamma_{3n}), 0.1)`
    threshold here is an explicit solver acceptance contract rather than a worst-case
    rounding bound.
    """
    denom = max(1.0 - 3.0 * size * unit_roundoff, 0.5)
    gamma_3n = (3.0 * size * unit_roundoff) / denom
    return min(max(64.0 * unit_roundoff, 8.0 * math.sqrt(size) * gamma_3n), 0.1)


def _normalize_rhs_lane(
    b_vec: jax.Array,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Normalize active RHS vectors to unit 2-norm and isolate zero/non-finite lanes."""
    has_nonfinite = ~jnp.all(jnp.isfinite(b_vec))
    clean_b = jnp.where(has_nonfinite, jnp.zeros_like(b_vec), b_vec)
    b_norm = jnp.linalg.norm(clean_b)
    active = (~has_nonfinite) & (b_norm > 0.0)
    safe_b_norm = jnp.where(active, b_norm, jnp.ones_like(b_norm))
    unit_e0 = jnp.zeros_like(b_vec).at[0].set(jnp.asarray(1.0, dtype=dtype))
    unit_b = jnp.where(active, clean_b / safe_b_norm.astype(dtype), unit_e0)
    return unit_b, b_norm, safe_b_norm, active, has_nonfinite


def _finalize_rhs_lane(
    b_vec: jax.Array,
    unit_sol: jax.Array,
    is_valid: jax.Array,
    b_norm: jax.Array,
    safe_b_norm: jax.Array,
    active: jax.Array,
    has_nonfinite: jax.Array,
) -> jax.Array:
    """Rescale ``unit_sol`` on active lanes, zero zero-RHS lanes, and ``NaN``-mask failures."""
    dtype = b_vec.dtype
    finite_sol = jnp.where(jnp.isfinite(unit_sol), unit_sol, jnp.zeros_like(unit_sol))
    scaled_sol = jnp.where(
        b_norm > 0.0,
        finite_sol * safe_b_norm.astype(dtype),
        jnp.zeros_like(b_vec),
    )
    invalid = has_nonfinite | (active & (~is_valid))
    return jnp.where(invalid, jnp.full_like(b_vec, jnp.nan), scaled_sol)


def _wrap_iterative_tangent_solve(
    lin_op: Callable[[Any], Any],
    rhs: Any,
    solve_unit_fn: Callable[
        [Callable[[jax.Array], jax.Array], jax.Array],
        tuple[jax.Array, jax.Array],
    ],
) -> Any:
    """Wrap a matrix-free iterative unit-RHS solver inside ``jax.lax.custom_linear_solve``."""
    rhs_flat, unravel = ravel_pytree(rhs)
    if rhs_flat.size == 0:
        return jax.tree.map(jnp.zeros_like, rhs)
    dtype = rhs_flat.dtype

    def lin_op_flat(v_vec: jax.Array) -> jax.Array:
        """Apply ``lin_op`` to an unravelled vector and flatten the result."""
        out_vec, _ = ravel_pytree(lin_op(unravel(v_vec)))
        return out_vec.astype(dtype)

    def _solve_one(
        mv: Callable[[jax.Array], jax.Array],
        b_vec: jax.Array,
    ) -> jax.Array:
        """Normalize active RHS lanes, dispatch ``solve_unit_fn``, and poison failed lanes."""
        unit_b, b_norm, safe_b_norm, active, has_nonfinite = _normalize_rhs_lane(b_vec, dtype)

        def safe_mv(vec: jax.Array) -> jax.Array:
            """Substitute the identity operator on inactive zero/non-finite RHS lanes."""
            return jnp.where(active, mv(vec), vec)

        unit_sol, is_valid = solve_unit_fn(safe_mv, unit_b)
        return _finalize_rhs_lane(
            b_vec, unit_sol, is_valid, b_norm, safe_b_norm, active, has_nonfinite
        )

    sol_flat = jax.lax.custom_linear_solve(
        lin_op_flat,
        rhs_flat,
        solve=_solve_one,
        transpose_solve=_solve_one,
    )
    return unravel(sol_flat)


def dense_tangent_solve(
    lin_op: Callable[[Any], Any],
    rhs: Any,
) -> Any:
    r"""Solve ``lin_op(v) = rhs`` by dense Jacobian materialization and partial-pivoting LU.

    Materializes ``dense_op`` and computes its LU factorization and :math:`O(n^2)`
    alternating triangular-solve condition estimate from ``lin_op`` alone before
    inspecting ``rhs``, so batching over multiple right-hand sides (:func:`jax.vmap` or
    :func:`jax.jacrev`) factors an unbatched :math:`n \times n` operator once and reuses
    ``(lu, piv)`` across all lanes. Enforces the conditioning policy
    :math:`\hat{\kappa}(A) < (\sqrt{n}\,u)^{-1}` and the normwise backward-error contract
    :func:`_lu_backward_error_ceiling`, returning ``NaN`` on singular or ill-conditioned
    systems.

    Args:
        lin_op: Linear operator ``v -> A(v)`` on pytrees matching ``rhs``.
        rhs: Right-hand-side pytree ``b``.

    Returns:
        Solution pytree ``v`` matching ``rhs`` (``NaN`` if ``lin_op`` fails the
        conditioning or backward-error contract).
    """
    rhs_flat, unravel = ravel_pytree(rhs)
    if rhs_flat.size == 0:
        return jax.tree.map(jnp.zeros_like, rhs)
    size = rhs_flat.shape[0]
    dtype = rhs_flat.dtype
    u_round = _real_unit_roundoff(dtype)
    bwd_ceil = _lu_backward_error_ceiling(size, u_round)
    # Explicit conditioning policy: require estimated kappa_F < 1 / (sqrt(n) * u).
    cond_ceil = 1.0 / (math.sqrt(size) * u_round)
    adj_trans = 2 if jnp.issubdtype(dtype, jnp.complexfloating) else 1

    def lin_op_flat(v_vec: jax.Array) -> jax.Array:
        """Apply ``lin_op`` to an unravelled vector and flatten the result."""
        out_vec, _ = ravel_pytree(lin_op(unravel(v_vec)))
        return out_vec.astype(dtype)

    def _solve_lu_one(
        mv: Callable[[jax.Array], jax.Array],
        b_vec: jax.Array,
    ) -> jax.Array:
        """Factor ``mv`` independent of ``b_vec``, then solve and validate active lanes."""
        eye = jnp.eye(size, dtype=dtype)
        dense_op = jax.vmap(mv, out_axes=-1)(eye)
        lu, piv = jsp_linalg.lu_factor(dense_op)

        probe = jnp.linspace(1.0, 2.0, size, dtype=dtype)
        probe = probe / jnp.linalg.norm(probe)
        for _ in range(2):
            w_vec = jsp_linalg.lu_solve((lu, piv), probe, trans=adj_trans)
            w_norm = jnp.linalg.norm(w_vec)
            probe_mid = jnp.where(w_norm > 0.0, w_vec / w_norm, probe)
            q_vec = jsp_linalg.lu_solve((lu, piv), probe_mid, trans=0)
            q_norm = jnp.linalg.norm(q_vec)
            probe = jnp.where(q_norm > 0.0, q_vec / q_norm, probe_mid)
        inv_sigma_min = jnp.linalg.norm(jsp_linalg.lu_solve((lu, piv), probe, trans=adj_trans))

        frob_norm = jnp.linalg.norm(dense_op)
        op_norm_lower = frob_norm / math.sqrt(size)
        cond_est = frob_norm * inv_sigma_min
        diag_u = jnp.abs(jnp.diag(lu))
        op_valid = (
            jnp.all(jnp.isfinite(dense_op))
            & jnp.all(jnp.isfinite(lu))
            & (jnp.min(diag_u) > 0.0)
            & jnp.isfinite(cond_est)
            & (cond_est < cond_ceil)
        )

        unit_b, b_norm, safe_b_norm, active, has_nonfinite = _normalize_rhs_lane(b_vec, dtype)
        unit_sol = jsp_linalg.lu_solve((lu, piv), unit_b)
        finite_sol = jnp.where(jnp.isfinite(unit_sol), unit_sol, jnp.zeros_like(unit_sol))
        res_norm = jnp.linalg.norm(dense_op @ finite_sol - unit_b)
        bwd_err = res_norm / (op_norm_lower * jnp.linalg.norm(finite_sol) + 1.0)
        lane_valid = (
            op_valid
            & jnp.all(jnp.isfinite(unit_sol))
            & jnp.isfinite(bwd_err)
            & (bwd_err <= bwd_ceil)
        )
        return _finalize_rhs_lane(
            b_vec, unit_sol, lane_valid, b_norm, safe_b_norm, active, has_nonfinite
        )

    sol_flat = jax.lax.custom_linear_solve(
        lin_op_flat,
        rhs_flat,
        solve=_solve_lu_one,
        transpose_solve=_solve_lu_one,
    )
    return unravel(sol_flat)


def lsmr_tangent_solve(
    lin_op: Callable[[Any], Any],
    rhs: Any,
    *,
    rtol: float | None = None,
    max_steps: int | None = None,
    conlim: float | None = None,
) -> Any:
    r"""Matrix-free tangent solve via Fong & Saunders (2011) LSMR.

    Applies Golub-Kahan bidiagonalization through ``lin_op`` and
    :func:`jax.linear_transpose` without forming :math:`A^\top A` (so :math:`\kappa(A)`
    is never squared), avoiding restarted-GMRES stagnation on indefinite non-symmetric
    spectra while returning ``NaN`` on singular or unconverged systems. Default policy
    parameters use ``rtol = max(16 * u, 1e-12)``, ``max_steps = 10 * dim``,
    ``conlim = 1 / (sqrt(dim) * u)``, and post-solve residual acceptance ceiling
    ``max(100 * rtol, bwd_ceil)``.

    Args:
        lin_op: Matrix-free linear operator ``v -> A(v)`` on pytrees matching ``rhs``.
        rhs: Right-hand-side pytree ``b``.
        rtol: Optional relative stopping tolerance (default policy ``max(16 * u, 1e-12)``).
        max_steps: Optional Krylov step limit (default policy ``10 * dim``).
        conlim: Optional condition-number ceiling (default policy ``1 / (sqrt(dim) * u)``).

    Returns:
        Solution pytree ``v`` matching ``rhs`` (``NaN`` if singular or unconverged).
    """

    def _solve_unit_lsmr(
        safe_mv: Callable[[jax.Array], jax.Array],
        unit_b: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        size = unit_b.shape[0]
        u_round = _real_unit_roundoff(unit_b.dtype)
        eff_rtol = max(16.0 * u_round, 1e-12) if rtol is None else rtol
        eff_steps = 10 * size if max_steps is None else max_steps
        eff_conlim = (1.0 / (math.sqrt(size) * u_round)) if conlim is None else conlim
        check_ceil = max(100.0 * eff_rtol, _lu_backward_error_ceiling(size, u_round))

        vjp_fn = jax.linear_transpose(safe_mv, unit_b)
        unit_sol, est_norm_a, est_cond = solve_unit_lsmr_stopping(
            safe_mv,
            lambda u_vec: jnp.conj(vjp_fn(jnp.conj(u_vec))[0]),
            unit_b,
            rtol=eff_rtol,
            max_steps=eff_steps,
            conlim=eff_conlim,
        )
        finite_sol = jnp.where(jnp.isfinite(unit_sol), unit_sol, jnp.zeros_like(unit_sol))
        res_norm = jnp.linalg.norm(safe_mv(finite_sol) - unit_b)
        op_lower = jnp.maximum(est_norm_a / math.sqrt(size), jnp.linalg.norm(safe_mv(unit_b)))
        bwd_err = res_norm / (op_lower * jnp.linalg.norm(finite_sol) + 1.0)
        valid = (
            jnp.all(jnp.isfinite(unit_sol))
            & jnp.isfinite(est_cond)
            & (est_cond < eff_conlim)
            & jnp.isfinite(bwd_err)
            & (bwd_err <= check_ceil)
        )
        return unit_sol, valid

    return _wrap_iterative_tangent_solve(lin_op, rhs, _solve_unit_lsmr)


def gmres_tangent_solve(
    lin_op: Callable[[Any], Any],
    rhs: Any,
    *,
    restart: int = 32,
    max_cycles: int | None = None,
    stagnation_iters: int = 20,
    rtol: float | None = None,
) -> Any:
    r"""Matrix-free restarted GMRES tangent solve with stagnation detection.

    Runs restarted Arnoldi cycles (Saad & Schultz 1986) with an explicit stagnation
    counter that halts early when the minimum residual norm fails to decrease for
    ``stagnation_iters`` cycles, enforcing the normwise backward-error acceptance
    policy ``max(100 * rtol, bwd_ceil)`` (with default policy ``rtol = max(20 * u,
    1e-12)``) and returning ``NaN`` on singular or stagnated systems.

    Args:
        lin_op: Matrix-free linear operator ``v -> A(v)`` on pytrees matching ``rhs``.
        rhs: Right-hand-side pytree ``b``.
        restart: Krylov subspace dimension per Arnoldi cycle (clamped to ``dim``).
        max_cycles: Maximum restart cycles (default policy ``max(20, ceil(10 * dim / restart))``).
        stagnation_iters: Maximum consecutive non-decreasing cycles before declaring stagnation.
        rtol: Optional relative residual tolerance (default policy ``max(20 * u, 1e-12)``).

    Returns:
        Solution pytree ``v`` matching ``rhs`` (``NaN`` if singular or stagnated).

    Raises:
        ValueError: When ``restart < 1`` or ``stagnation_iters < 1``.
    """
    if restart < 1 or stagnation_iters < 1:
        raise ValueError("restart and stagnation_iters must be at least 1.")

    def _solve_unit_gmres(
        safe_mv: Callable[[jax.Array], jax.Array],
        unit_b: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        size = unit_b.shape[0]
        dtype = unit_b.dtype
        u_round = _real_unit_roundoff(dtype)
        eff_restart = min(restart, size)
        eff_cycles = (
            max(20, math.ceil(10 * size / eff_restart)) if max_cycles is None else max_cycles
        )
        eff_rtol = max(20.0 * u_round, 1e-12) if rtol is None else rtol
        check_ceil = max(100.0 * eff_rtol, _lu_backward_error_ceiling(size, u_round))
        cond_ceil = 1.0 / (math.sqrt(size) * u_round)
        real_dtype = jnp.finfo(dtype).dtype

        def cond_fn(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> jax.Array:
            _y, r_norm, _r_min, step, stag = carry
            return (step < eff_cycles) & (stag < stagnation_iters) & (r_norm > eff_rtol)

        def body_fn(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            y_curr, _r_curr, r_min, step, stag = carry
            y_next, _ = jax.scipy.sparse.linalg.gmres(
                safe_mv,
                unit_b,
                x0=y_curr,
                tol=eff_rtol,
                atol=0.0,
                restart=eff_restart,
                maxiter=1,
                solve_method="incremental",
            )
            r_next = jnp.linalg.norm(safe_mv(y_next) - unit_b).astype(real_dtype)
            improved = r_next < r_min * (1.0 - 10.0 * u_round)
            next_stag = jnp.where(improved, jnp.int32(0), stag + jnp.int32(1))
            next_min = jnp.minimum(r_next, r_min)
            return y_next, r_next, next_min, step + jnp.int32(1), next_stag

        init_carry = (
            jnp.zeros_like(unit_b),
            jnp.ones((), dtype=real_dtype),
            jnp.asarray(jnp.inf, dtype=real_dtype),
            jnp.int32(0),
            jnp.int32(0),
        )
        unit_sol, _, _, _, final_stag = jax.lax.while_loop(cond_fn, body_fn, init_carry)

        finite_sol = jnp.where(jnp.isfinite(unit_sol), unit_sol, jnp.zeros_like(unit_sol))
        sol_norm = jnp.linalg.norm(finite_sol)
        mv_sol = safe_mv(finite_sol)
        res_norm = jnp.linalg.norm(mv_sol - unit_b)
        op_est = jnp.maximum(
            jnp.linalg.norm(safe_mv(unit_b)),
            jnp.where(sol_norm > 0.0, jnp.linalg.norm(mv_sol) / jnp.maximum(sol_norm, 1e-30), 0.0),
        )
        bwd_err = res_norm / (op_est * sol_norm + 1.0)
        stagnated = (final_stag >= stagnation_iters) & (bwd_err > check_ceil)
        valid = (
            jnp.all(jnp.isfinite(unit_sol))
            & (~stagnated)
            & ((op_est * sol_norm) < cond_ceil)
            & jnp.isfinite(bwd_err)
            & (bwd_err <= check_ceil)
        )
        return unit_sol, valid

    return _wrap_iterative_tangent_solve(lin_op, rhs, _solve_unit_gmres)


__all__ = [
    "TangentSolverFn",
    "call_unary_or_binary",
    "cast_inexact_pytree",
    "dense_tangent_solve",
    "gmres_tangent_solve",
    "guard_unconverged_tangents",
    "infer_inexact_dtype",
    "loop_status",
    "lsmr_tangent_solve",
]
