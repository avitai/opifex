"""Pure-JAX second-order optimization wrappers and full-memory BFGS solver.

Provides unified factory functions and :mod:`opifex.core.solver` methods for:
* :func:`create_lbfgs_optimizer` — :func:`optax.lbfgs` with zoom or backtracking line search.
* :func:`create_gauss_newton_solver` — pure-JAX Gauss-Newton via
  :class:`~opifex.solvers.nonlinear_least_squares.LevenbergMarquardt` with minimal damping
  and identity scaling.
* :func:`create_levenberg_marquardt_solver` — pure-JAX damped Levenberg-Marquardt via
  :class:`~opifex.solvers.nonlinear_least_squares.LevenbergMarquardt`.
* :class:`BFGSSolver` / :func:`create_bfgs_solver` — pure-JAX Nocedal & Wright (2006,
  Algorithm 6.1) full-memory BFGS minimizer using :func:`optax.scale_by_zoom_linesearch`
  and :func:`jax.lax.custom_root` implicit differentiation.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax.flatten_util import ravel_pytree

from opifex.core.solver.implicit import (
    call_unary_or_binary,
    cast_inexact_pytree,
    dense_tangent_solve,
    guard_unconverged_tangents,
    infer_inexact_dtype,
    TangentSolverFn,
)
from opifex.core.solver.problem import Method, Problem, solve
from opifex.core.solver.solution import Solution
from opifex.core.solver.status import Status
from opifex.optimization.second_order.config import (
    GaussNewtonConfig,
    LBFGSConfig,
    LinesearchType,
)
from opifex.solvers.nonlinear_least_squares import LevenbergMarquardt


class OptimizationProblem(Problem):
    """Smooth unconstrained minimization problem ``min_x objective_fn(x, args)``.

    Attributes:
        objective_fn: Scalar objective ``(params, args) -> loss`` or ``(params,) -> loss``.
        initial_params: Initial parameter pytree ``x_0``.
        args: Optional problem arguments pytree forwarded to ``objective_fn``.
    """

    objective_fn: Callable[..., jax.Array] = struct.field(pytree_node=False)
    initial_params: Any
    args: Any = None


class _BFGSCarry(struct.PyTreeNode):
    """Internal while-loop carry for pure-JAX full-memory BFGS."""

    theta: jax.Array
    value: jax.Array
    grad: jax.Array
    inv_hess: jax.Array
    ls_state: Any
    step_count: jax.Array
    converged: jax.Array
    stalled: jax.Array
    reset_available: jax.Array
    has_nonfinite: jax.Array


def _call_objective(
    objective_fn: Callable[..., jax.Array],
    params: Any,
    args: Any,
) -> jax.Array:
    """Invoke ``objective_fn(params)`` or ``objective_fn(params, args)``."""
    return jnp.asarray(call_unary_or_binary(objective_fn, params, args)).squeeze()


def _bfgs_rank2_update(
    inv_hess: jax.Array,
    s_vec: jax.Array,
    y_vec: jax.Array,
    step_count: jax.Array,
) -> jax.Array:
    """Nocedal & Wright (2006) Algorithm 6.1 inverse-Hessian update with Powell guard."""
    dtype = inv_hess.dtype
    dim = inv_hess.shape[0]
    eye = jnp.eye(dim, dtype=dtype)
    ys = jnp.dot(y_vec, s_vec)
    yy = jnp.dot(y_vec, y_vec)
    curv_floor = float(jnp.finfo(dtype).eps) * jnp.linalg.norm(s_vec) * jnp.linalg.norm(y_vec)
    valid = jnp.isfinite(ys) & jnp.isfinite(yy) & (ys > curv_floor) & (yy > 0.0)

    safe_yy = jnp.where(yy > 0.0, yy, jnp.asarray(1.0, dtype=dtype))
    init_scaled = (ys / safe_yy) * eye
    base_h = jnp.where((step_count == 0) & valid, init_scaled, inv_hess)

    rho = 1.0 / jnp.where(valid, ys, jnp.asarray(1.0, dtype=dtype))
    proj = eye - rho * jnp.outer(s_vec, y_vec)
    cand_h = proj @ base_h @ proj.T + rho * jnp.outer(s_vec, s_vec)
    sym_h = 0.5 * (cand_h + cand_h.T)
    return jnp.where(valid, sym_h, inv_hess)


def _cauchy_termination(
    rtol: float,
    atol: float,
    y: jax.Array,
    y_diff: jax.Array,
    f_val: jax.Array,
    f_diff: jax.Array,
) -> jax.Array:
    """Scale-aware supremum-norm iterate and objective Cauchy stopping criterion."""
    y_scale = atol + rtol * jnp.abs(y)
    f_scale = atol + rtol * jnp.abs(f_val)
    y_converged = jnp.max(jnp.abs(y_diff) / y_scale) < 1.0 if y.size > 0 else jnp.asarray(True)
    f_converged = (jnp.abs(f_diff) / f_scale) < 1.0
    return y_converged & f_converged


def _run_bfgs_while(
    loss_vec_fn: Callable[[jax.Array], jax.Array],
    theta0: jax.Array,
    *,
    max_iterations: int,
    rtol: float,
    atol: float,
    max_linesearch_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Execute the forward full-memory BFGS loop with zoom line search."""
    dtype = theta0.dtype
    dim = theta0.shape[0]
    eye = jnp.eye(dim, dtype=dtype)
    ls = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy="one",
    )
    val_and_grad = optax.value_and_grad_from_state(loss_vec_fn)

    init_ls = ls.init(theta0)
    val0, grad0 = jax.value_and_grad(loss_vec_fn)(theta0)
    nonfinite0 = ~jnp.isfinite(val0) | ~jnp.all(jnp.isfinite(grad0))
    conv0 = (~nonfinite0) & (dim == 0)

    def cond_fn(carry: _BFGSCarry) -> jax.Array:
        """Check whether the outer BFGS loop should continue."""
        return (
            (carry.step_count < max_iterations)
            & (~carry.converged)
            & (~carry.stalled)
            & (~carry.has_nonfinite)
        )

    def body_fn(carry: _BFGSCarry) -> _BFGSCarry:
        """Execute one BFGS search-direction, zoom line-search, and rank-2 update step."""
        grad_zero = jnp.all(carry.grad == 0.0)
        raw_dir = -(carry.inv_hess @ carry.grad)
        not_descent = (~grad_zero) & (jnp.dot(raw_dir, carry.grad) >= 0.0)
        search_dir = jnp.where(not_descent, -carry.grad, raw_dir)
        active_h = jnp.where(not_descent, eye, carry.inv_hess)

        raw_step_vec, next_ls = ls.update(
            search_dir,
            carry.ls_state,
            carry.theta,
            value=carry.value,  # pyright: ignore[reportCallIssue]
            grad=carry.grad,  # pyright: ignore[reportCallIssue]
            value_fn=loss_vec_fn,  # pyright: ignore[reportCallIssue]
        )
        step_vec = cast("jax.Array", raw_step_vec)
        ls_info: Any = getattr(next_ls, "info", None)
        ls_accepted = grad_zero | (
            (ls_info.decrease_error <= 0.0) if ls_info is not None else jnp.any(step_vec != 0.0)
        )

        cand_theta = jnp.where(ls_accepted, carry.theta + step_vec, carry.theta)
        raw_cand_val, raw_cand_grad = val_and_grad(cand_theta, state=next_ls)
        eval_val = jnp.asarray(raw_cand_val, dtype=dtype)
        eval_grad = cast("jax.Array", raw_cand_grad)
        cand_val = jnp.where(ls_accepted, eval_val, carry.value)
        cand_grad = jnp.where(ls_accepted, eval_grad, carry.grad)
        step_nonfinite = ~jnp.isfinite(eval_val) | ~jnp.all(jnp.isfinite(eval_grad))

        eff_step = cand_theta - carry.theta
        eff_f_diff = cand_val - carry.value
        cauchy_ok = _cauchy_termination(rtol, atol, cand_theta, eff_step, cand_val, eff_f_diff)
        # Require at least one inverse-Hessian update (step_count > 0) and an accepted
        # line-search step before declaring Cauchy convergence.
        terminate = (
            (~step_nonfinite)
            & ls_accepted
            & jnp.where(carry.step_count == 0, jnp.asarray(False), cauchy_ok)
        )

        ls_rejected = (~step_nonfinite) & (~ls_accepted)
        ranked_h = _bfgs_rank2_update(active_h, eff_step, cand_grad - carry.grad, carry.step_count)
        next_h = jnp.where(ls_rejected, eye, ranked_h)
        reset_ls = ls.init(cand_theta)
        chosen_ls = jax.tree.map(
            lambda r_leaf, n_leaf: jnp.where(ls_rejected, r_leaf, n_leaf),
            reset_ls,
            next_ls,
        )
        next_reset = jnp.where(ls_rejected, jnp.asarray(False), jnp.asarray(True))
        next_stalled = ls_rejected & (~carry.reset_available)

        next_theta = jnp.where(step_nonfinite, carry.theta, cand_theta)
        return _BFGSCarry(
            theta=next_theta,
            value=cand_val,
            grad=cand_grad,
            inv_hess=next_h,
            ls_state=chosen_ls,
            step_count=carry.step_count + jnp.asarray(1, dtype=jnp.int32),
            converged=terminate,
            stalled=next_stalled,
            reset_available=next_reset,
            has_nonfinite=carry.has_nonfinite | step_nonfinite,
        )

    init_carry = _BFGSCarry(
        theta=theta0,
        value=val0,
        grad=grad0,
        inv_hess=eye,
        ls_state=init_ls,
        step_count=jnp.asarray(0, dtype=jnp.int32),
        converged=conv0,
        stalled=jnp.asarray(False),
        reset_available=jnp.asarray(True),
        has_nonfinite=nonfinite0,
    )
    final = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    status = jnp.where(
        final.has_nonfinite,
        jnp.asarray(int(Status.NONFINITE), dtype=jnp.int32),
        jnp.where(
            final.converged,
            jnp.asarray(int(Status.SUCCESS), dtype=jnp.int32),
            jnp.where(
                final.stalled,
                jnp.asarray(int(Status.STALLED), dtype=jnp.int32),
                jnp.asarray(int(Status.MAX_ITERS), dtype=jnp.int32),
            ),
        ),
    )
    return final.theta, status, final.step_count


class BFGSSolver(Method[OptimizationProblem]):
    """Pure-JAX full-memory BFGS minimizer with zoom line search and implicit diff.

    Attributes:
        max_iterations: Maximum number of BFGS outer iterations.
        rtol: Relative Cauchy iterate/objective convergence tolerance.
        atol: Absolute Cauchy iterate/objective convergence tolerance.
        max_linesearch_steps: Maximum zoom line-search steps per outer iteration.
        implicit_diff: Differentiate through ``theta*`` via ``jax.lax.custom_root``
            on ``grad f(theta*) = 0``.
        tangent_solver: Linear solver ``(lin_op, rhs) -> v`` for implicit
            differentiation; defaults to :func:`~opifex.core.solver.dense_tangent_solve`.
    """

    max_iterations: int = struct.field(pytree_node=False, default=100)
    rtol: float = struct.field(pytree_node=False, default=1e-6)
    atol: float = struct.field(pytree_node=False, default=1e-6)
    max_linesearch_steps: int = struct.field(pytree_node=False, default=20)
    implicit_diff: bool = struct.field(pytree_node=False, default=True)
    tangent_solver: TangentSolverFn = struct.field(pytree_node=False, default=dense_tangent_solve)

    def _validate(self) -> None:
        """Validate static BFGS hyperparameters before tracing.

        Raises:
            ValueError: When iteration budgets or tolerances are non-positive.
        """
        if self.max_iterations < 1 or self.max_linesearch_steps < 1:
            raise ValueError("max_iterations and max_linesearch_steps must be at least 1.")
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("rtol and atol must be strictly positive.")

    def solve(self, problem: OptimizationProblem) -> Solution:
        """Minimize ``problem`` and return a traced :class:`Solution`.

        Args:
            problem: Unconstrained minimization problem instance.

        Returns:
            :class:`Solution` with ``fields={"params": ...}``, ``metrics={"loss": ...,
            "grad_inf_norm": ...}``, ``status``, and ``stats={"num_iterations": ...}``.
        """
        self._validate()
        dtype = infer_inexact_dtype(problem.initial_params, problem.args)
        init_typed = cast_inexact_pytree(problem.initial_params, dtype)
        theta0, unravel = ravel_pytree(init_typed)

        def loss_vec_fn(theta_vec: jax.Array) -> jax.Array:
            """Evaluate the scalar objective on a flattened parameter vector."""
            return _call_objective(problem.objective_fn, unravel(theta_vec), problem.args).astype(
                dtype
            )

        if self.implicit_diff:
            stationarity = jax.grad(loss_vec_fn)

            def root_solver(
                _: Callable[[jax.Array], jax.Array],
                init_vec: jax.Array,
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
                """Run forward BFGS minimization with floating auxiliary leaves."""
                sol_vec, st_i32, n_i32 = _run_bfgs_while(
                    loss_vec_fn,
                    init_vec,
                    max_iterations=self.max_iterations,
                    rtol=self.rtol,
                    atol=self.atol,
                    max_linesearch_steps=self.max_linesearch_steps,
                )
                return sol_vec, (st_i32.astype(dtype), n_i32.astype(dtype))

            root_out: Any = jax.lax.custom_root(
                stationarity,
                theta0,
                root_solver,
                self.tangent_solver,
                has_aux=True,
            )
            theta_star, (status_f, steps_f) = root_out
            status = status_f.astype(jnp.int32)
            steps = steps_f.astype(jnp.int32)
            theta_star = guard_unconverged_tangents(theta_star, status == int(Status.SUCCESS))
        else:
            theta_star, status, steps = _run_bfgs_while(
                loss_vec_fn,
                theta0,
                max_iterations=self.max_iterations,
                rtol=self.rtol,
                atol=self.atol,
                max_linesearch_steps=self.max_linesearch_steps,
            )

        final_loss, final_grad = jax.value_and_grad(loss_vec_fn)(theta_star)
        grad_inf = (
            jnp.max(jnp.abs(final_grad)) if final_grad.size > 0 else jnp.asarray(0.0, dtype=dtype)
        )
        return Solution(
            fields={"params": unravel(theta_star)},
            metrics={"loss": final_loss, "grad_inf_norm": grad_inf},
            status=status,
            stats={"num_iterations": steps},
        )

    def minimise(
        self,
        objective_fn: Callable[..., jax.Array],
        initial_params: Any,
        *,
        args: Any = None,
    ) -> Solution:
        """Convenience functional entry point solving ``OptimizationProblem``."""
        return solve(
            OptimizationProblem(
                objective_fn=objective_fn,
                initial_params=initial_params,
                args=args,
            ),
            self,
        )


def create_lbfgs_optimizer(
    config: LBFGSConfig | None = None,
) -> optax.GradientTransformation:
    """Create L-BFGS optimizer using :func:`optax.lbfgs`.

    Args:
        config: L-BFGS configuration. Uses defaults if ``None``.

    Returns:
        Optax L-BFGS gradient transformation.
    """
    if config is None:
        config = LBFGSConfig()

    if config.linesearch == LinesearchType.ZOOM:
        linesearch = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=config.max_linesearch_steps,
        )
    else:
        linesearch = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=config.max_linesearch_steps,
        )

    return optax.lbfgs(
        memory_size=config.memory_size,
        scale_init_precond=config.scale_init_precond,
        linesearch=linesearch,
    )


def create_gauss_newton_solver(
    config: GaussNewtonConfig | None = None,
    *,
    tangent_solver: TangentSolverFn | None = None,
) -> LevenbergMarquardt:
    """Create a pure-JAX Gauss-Newton nonlinear least-squares solver.

    Configures :class:`~opifex.solvers.nonlinear_least_squares.LevenbergMarquardt`
    with minimal initial damping (``max(config.min_damping, 1e-12)``) and identity
    scaling for Gauss-Newton steps under :mod:`opifex.core.solver`.

    Args:
        config: Gauss-Newton configuration. Uses defaults if ``None``.
        tangent_solver: Optional linear solver ``(lin_op, rhs) -> v`` for implicit
            differentiation; defaults to :func:`~opifex.core.solver.dense_tangent_solve`.

    Returns:
        Configured pure-JAX :class:`LevenbergMarquardt` solver.
    """
    cfg = config or GaussNewtonConfig()
    return LevenbergMarquardt(
        max_iterations=cfg.max_iterations,
        gtol=cfg.atol,
        xtol=cfg.rtol,
        ftol=cfg.rtol,
        initial_damping=max(cfg.min_damping, 1e-12),
        damping_up=max(cfg.damping_increase_factor, 1.0 + 1e-6),
        damping_down=min(max(cfg.damping_decrease_factor, 1e-6), 1.0 - 1e-6),
        scaling="identity",
        tangent_solver=tangent_solver,
    )


def create_levenberg_marquardt_solver(
    config: GaussNewtonConfig | None = None,
    *,
    tangent_solver: TangentSolverFn | None = None,
) -> LevenbergMarquardt:
    """Create a pure-JAX damped Levenberg-Marquardt nonlinear least-squares solver.

    Args:
        config: Gauss-Newton/LM configuration. Uses defaults if ``None``.
        tangent_solver: Optional linear solver ``(lin_op, rhs) -> v`` for implicit
            differentiation; defaults to :func:`~opifex.core.solver.dense_tangent_solve`.

    Returns:
        Configured pure-JAX :class:`LevenbergMarquardt` solver.
    """
    cfg = config or GaussNewtonConfig()
    return LevenbergMarquardt(
        max_iterations=cfg.max_iterations,
        gtol=cfg.atol,
        xtol=cfg.rtol,
        ftol=cfg.rtol,
        initial_damping=max(cfg.damping_factor, 1e-12),
        damping_up=max(cfg.damping_increase_factor, 1.0 + 1e-6),
        damping_down=min(max(cfg.damping_decrease_factor, 1e-6), 1.0 - 1e-6),
        scaling="marquardt",
        tangent_solver=tangent_solver,
    )


def create_bfgs_solver(
    config: GaussNewtonConfig | None = None,
    *,
    tangent_solver: TangentSolverFn = dense_tangent_solve,
) -> BFGSSolver:
    """Create a pure-JAX full-memory :class:`BFGSSolver` minimizer.

    Args:
        config: Configuration with iteration and tolerance settings. Uses defaults if ``None``.
        tangent_solver: Linear solver ``(lin_op, rhs) -> v`` for implicit
            differentiation; defaults to :func:`~opifex.core.solver.dense_tangent_solve`.

    Returns:
        Configured pure-JAX :class:`BFGSSolver`.
    """
    cfg = config or GaussNewtonConfig()
    return BFGSSolver(
        max_iterations=cfg.max_iterations,
        rtol=cfg.rtol,
        atol=cfg.atol,
        tangent_solver=tangent_solver,
    )


__all__ = [
    "BFGSSolver",
    "OptimizationProblem",
    "create_bfgs_solver",
    "create_gauss_newton_solver",
    "create_lbfgs_optimizer",
    "create_levenberg_marquardt_solver",
]
