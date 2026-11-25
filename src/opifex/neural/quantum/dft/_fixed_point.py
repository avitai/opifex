r"""Pure-JAX Anderson-accelerated fixed-point solver for differentiable SCF.

Plain Roothaan iteration :math:`D \to \mathrm{SCF\_step}(D)` charge-sloshes and
fails to converge for many molecules -- water/LDA, for instance, stalls at a
density residual of :math:`\sim 0.9`. **Anderson acceleration** [Anderson1965]_
[WalkerNi2011]_ -- equivalent to Pulay's DIIS [Pulay1980]_ applied to the density
residual -- mixes a short history of iterates to damp the oscillation and
converge in a handful of steps.

:func:`anderson_fixed_point` runs the Anderson iteration inside a pure-JAX
:func:`jax.lax.while_loop` and wraps the converged root in
:func:`jax.lax.custom_root` using :func:`~opifex.core.solver.dense_tangent_solve`
by default (dense partial-pivoting LU) or a caller-supplied matrix-free tangent
solver such as :func:`~opifex.core.solver.lsmr_tangent_solve` or
:func:`~opifex.core.solver.gmres_tangent_solve` for the implicit-function-theorem
tangent/adjoint linear system :math:`(\partial f/\partial D - I) v = b`. The
gradient therefore depends only on the converged density and the bare Roothaan map
:math:`f`, never on the forward history or step count (Zhang & Chan 2022).

The Anderson mixing coefficients solve the constrained least-squares problem

.. math::

    \min_{c}\ \Big\| \sum_i c_i\, r_i \Big\|^2 \quad\text{s.t.}\quad \sum_i c_i = 1,

over the stored residuals :math:`r_i = f(y_i) - y_i`, whose Lagrange solution is
:math:`c \propto M^{-1}\mathbf{1}` with the Gram matrix :math:`M_{ij}=r_i^\top r_j`
-- exactly Pulay's DIIS system.

References:
----------
.. [Anderson1965] D. G. Anderson, "Iterative procedures for nonlinear integral
   equations," *J. ACM* **12**, 547 (1965).
.. [WalkerNi2011] H. F. Walker and P. Ni, "Anderson acceleration for fixed-point
   iterations," *SIAM J. Numer. Anal.* **49**, 1715 (2011).
.. [Pulay1980] P. Pulay, "Convergence acceleration of iterative sequences. The
   case of SCF iteration," *Chem. Phys. Lett.* **73**, 393 (1980).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import struct
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from opifex.core.solver.implicit import (
    cast_inexact_pytree,
    dense_tangent_solve,
    guard_unconverged_tangents,
    infer_inexact_dtype,
    loop_status,
    TangentSolverFn,
)
from opifex.core.solver.status import is_successful, Status


def _max_norm(tree: PyTree) -> Array:
    """Supremum norm ``max_i |x_i|`` over all leaves of ``tree``."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    maxima = [jnp.max(jnp.abs(jnp.asarray(leaf))) for leaf in leaves]
    return jnp.max(jnp.stack(maxima))


class _AndersonCarry(struct.PyTreeNode):
    """Internal carry for the pure-JAX Anderson fixed-point loop."""

    y_vec: Array
    function_history: Array
    residual_history: Array
    slot: Array
    filled: Array
    relative_error: Array
    step_count: Array
    has_nonfinite: Array


class FixedPointSolution(struct.PyTreeNode):
    """Converged iterate, status code, and iteration count of a fixed-point solve.

    Attributes:
        value: Final iterate pytree matching ``y0``.
        status: Scalar ``int32`` status code (:class:`~opifex.core.solver.status.Status`).
        n_iterations: Scalar ``int32`` step count taken by the forward iteration.
    """

    value: Any
    status: Array
    n_iterations: Array

    @property
    def is_converged(self) -> Array:
        """Boolean array indicating whether ``status`` is a successful outcome."""
        return is_successful(self.status)


@dataclass(frozen=True, slots=True)
class AndersonAcceleration:
    r"""Pure-JAX Anderson-accelerated fixed-point solver (Pulay-DIIS on the residual).

    Args:
        rtol: Relative tolerance on the residual.
        atol: Absolute tolerance on the residual.
        history_size: Number of past iterates retained for mixing (depth ``m``).
        mixing: Relaxation factor ``beta`` in ``y_i + beta * r_i``; ``1.0`` is
            pure Anderson extrapolation, smaller values add damping.
        regularization: Relative Tikhonov factor scaling the mean squared residual
            norm on the valid history block. Defaults to ``sqrt(eps)`` of the
            working floating-point dtype.
        norm: Norm callable applied to the scaled residual ``r / (atol + rtol * max(|f(y)|, |y|))``.
    """

    rtol: float
    atol: float
    history_size: int = 6
    mixing: float = 1.0
    regularization: float | None = None
    norm: Callable[[PyTree], Array] = _max_norm

    def _mixing_coefficients(self, residuals: Array, filled: Array) -> Array:
        r"""Pulay-DIIS coefficients :math:`c \propto M^{-1}\mathbf 1`, masked.

        Invalid (unfilled) history slots are masked out of the Gram matrix so they
        receive zero weight, while a dtype-aware Tikhonov stabilizer scaled by the
        mean squared norm of the valid slots keeps rank-deficient near-converged
        histories nonsingular in both ``float32`` and ``float64``.

        Args:
            residuals: Stored residual history of shape ``(m, n)``.
            filled: Scalar integer count of valid history slots.

        Returns:
            Mixing coefficients of shape ``(m,)`` summing to ``1``.
        """
        valid = jnp.arange(self.history_size) < filled
        gram = residuals @ residuals.T
        pair_mask = valid[:, None] & valid[None, :]
        identity = jnp.eye(self.history_size, dtype=gram.dtype)
        occupancy = jnp.maximum(filled, 1).astype(gram.dtype)
        scale = jnp.sum(jnp.where(valid, jnp.diagonal(gram), 0.0)) / occupancy
        gram = jnp.where(pair_mask, gram, identity)
        reg_floor = jnp.asarray(float(jnp.finfo(gram.dtype).eps) ** 4, dtype=gram.dtype)
        reg = jnp.maximum(self._tikhonov_factor(gram.dtype) * scale, reg_floor)
        gram = gram + reg * identity

        rhs = valid.astype(gram.dtype)
        weights = cast("Array", jnp.linalg.solve(gram, rhs))
        weights = jnp.where(valid, weights, 0.0)
        weight_sum = jnp.sum(weights)
        safe_sum = jnp.where(weight_sum > 0.0, weight_sum, jnp.ones_like(weight_sum))
        return weights / safe_sum

    def _tikhonov_factor(self, dtype: jnp.dtype) -> float:
        """Return the relative Tikhonov factor for ``dtype``.

        Args:
            dtype: Working floating-point dtype of the Gram matrix.

        Returns:
            Configured regularization or ``sqrt(finfo(dtype).eps)``.
        """
        if self.regularization is not None:
            return self.regularization
        return float(math.sqrt(jnp.finfo(dtype).eps))


def _run_anderson_while(
    step_flat: Callable[[Array], Array],
    solver: AndersonAcceleration,
    y0_vec: Array,
    *,
    max_steps: int,
) -> tuple[Array, Array, Array]:
    """Execute the forward Anderson fixed-point loop via ``jax.lax.while_loop``."""
    dtype = y0_vec.dtype
    size = y0_vec.shape[0]
    zeros_hist = jnp.zeros((solver.history_size, size), dtype=dtype)

    def cond_fn(carry: _AndersonCarry) -> Array:
        """Check whether the Anderson loop should continue."""
        return (
            (carry.step_count < max_steps) & (carry.relative_error >= 1.0) & (~carry.has_nonfinite)
        )

    def body_fn(carry: _AndersonCarry) -> _AndersonCarry:
        """Execute one damped Anderson extrapolation iteration."""
        g_vec = step_flat(carry.y_vec)
        residual = g_vec - carry.y_vec
        scale = solver.atol + solver.rtol * jnp.maximum(jnp.abs(g_vec), jnp.abs(carry.y_vec))
        safe_scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
        scaled_res = jnp.where(residual == 0.0, jnp.zeros_like(residual), residual / safe_scale)
        rel_err = jnp.asarray(solver.norm(scaled_res), dtype=dtype)
        nonfinite = ~jnp.all(jnp.isfinite(g_vec)) | ~jnp.isfinite(rel_err)

        functions = carry.function_history.at[carry.slot].set(g_vec)
        residuals = carry.residual_history.at[carry.slot].set(residual)
        filled = jnp.minimum(carry.filled + 1, solver.history_size)

        coeffs = solver._mixing_coefficients(residuals, filled)
        relaxed = functions - (1.0 - solver.mixing) * residuals
        mixed_vec = coeffs @ relaxed
        step_converged = (~nonfinite) & (rel_err < 1.0)
        next_y = jnp.where(nonfinite | step_converged, carry.y_vec, mixed_vec)

        return _AndersonCarry(
            y_vec=next_y,
            function_history=functions,
            residual_history=residuals,
            slot=(carry.slot + 1) % solver.history_size,
            filled=filled,
            relative_error=rel_err,
            step_count=carry.step_count + jnp.asarray(1, dtype=jnp.int32),
            has_nonfinite=carry.has_nonfinite | nonfinite,
        )

    init_carry = _AndersonCarry(
        y_vec=y0_vec,
        function_history=zeros_hist,
        residual_history=zeros_hist,
        slot=jnp.asarray(0, dtype=jnp.int32),
        filled=jnp.asarray(0, dtype=jnp.int32),
        relative_error=jnp.asarray(jnp.inf, dtype=dtype),
        step_count=jnp.asarray(0, dtype=jnp.int32),
        has_nonfinite=~jnp.all(jnp.isfinite(y0_vec)),
    )
    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    status = loop_status(
        has_nonfinite=final_carry.has_nonfinite,
        converged=final_carry.relative_error < 1.0,
    )
    return final_carry.y_vec, status, final_carry.step_count


def anderson_fixed_point(
    fn: Callable[[Any, Any], Any],
    solver: AndersonAcceleration,
    y0: PyTree,
    *,
    args: Any = None,
    max_steps: int = 64,
    tangent_solver: TangentSolverFn = dense_tangent_solve,
) -> FixedPointSolution:
    """Find a fixed point ``fn(y*, args) = y*`` with implicit differentiation.

    Args:
        fn: Fixed-point map ``(y, args) -> y_next``.
        solver: :class:`AndersonAcceleration` configuration.
        y0: Initial guess pytree.
        args: Optional differentiable/static parameters passed to ``fn``.
        max_steps: Maximum number of Anderson iterations.
        tangent_solver: Linear solver ``(lin_op, rhs) -> v`` passed to
            :func:`jax.lax.custom_root`. Defaults to
            :func:`~opifex.core.solver.dense_tangent_solve`; pass
            :func:`~opifex.core.solver.lsmr_tangent_solve` or
            :func:`~opifex.core.solver.gmres_tangent_solve` for matrix-free solves.

    Returns:
        :class:`FixedPointSolution` carrying ``value``, ``status``, and ``n_iterations``.

    Raises:
        ValueError: When ``max_steps < 1``, ``solver.history_size < 1``,
            ``solver.rtol <= 0``, or ``solver.atol < 0``.
    """
    if max_steps < 1 or solver.history_size < 1:
        raise ValueError(
            f"max_steps ({max_steps}) and history_size ({solver.history_size}) must be at least 1."
        )
    if solver.rtol <= 0.0 or solver.atol < 0.0:
        raise ValueError(
            f"rtol must be strictly positive and atol non-negative; got rtol={solver.rtol}, "
            f"atol={solver.atol}."
        )
    dtype = infer_inexact_dtype(y0, args)
    y0_typed = cast_inexact_pytree(y0, dtype)
    y0_flat, unravel = ravel_pytree(y0_typed)

    def step_flat(y_vec: Array) -> Array:
        """Evaluate the fixed-point map on a flattened iterate vector."""
        out_vec, _ = ravel_pytree(fn(unravel(y_vec), args))
        return out_vec.astype(dtype)

    def residual_flat(y_vec: Array) -> Array:
        """Evaluate the flattened fixed-point residual ``f(y) - y``."""
        return step_flat(y_vec) - y_vec

    def root_solver(
        _: Callable[[Array], Array],
        init_vec: Array,
    ) -> tuple[Array, tuple[Array, Array]]:
        """Run forward Anderson iteration with floating auxiliary leaves."""
        sol_vec, status_i32, steps_i32 = _run_anderson_while(
            step_flat, solver, init_vec, max_steps=max_steps
        )
        # Cast int32 aux leaves to floating dtype inside jax.lax.custom_root (JAX 0.11.1)
        # and restore int32 immediately after custom_root returns.
        return sol_vec, (status_i32.astype(dtype), steps_i32.astype(dtype))

    root_out: Any = jax.lax.custom_root(
        residual_flat,
        y0_flat,
        root_solver,
        tangent_solver,
        has_aux=True,
    )
    y_star_flat, (status_f, steps_f) = root_out
    status_i32 = status_f.astype(jnp.int32)
    guarded_star = guard_unconverged_tangents(y_star_flat, status_i32 == int(Status.SUCCESS))
    return FixedPointSolution(
        value=unravel(guarded_star),
        status=status_i32,
        n_iterations=steps_f.astype(jnp.int32),
    )


__all__ = [
    "AndersonAcceleration",
    "FixedPointSolution",
    "anderson_fixed_point",
]
