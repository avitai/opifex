r"""Pure-JAX batched Levenberg-Marquardt nonlinear least-squares solver.

Solves weighted nonlinear least-squares problems

.. math::

    \theta^\star \in \arg\min_{\theta \in \mathbb{R}^p}
    \frac{1}{2} \|W^{1/2} r(\theta, d)\|_2^2, \qquad W = \operatorname{diag}(w),

inside :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad` with zero ``equinox``-stack
dependencies, conforming to :mod:`opifex.core.solver` and reusing
:func:`opifex.uncertainty.linalg.lstsq.lsmr`.

Algorithm & Reference Implementations
-------------------------------------
* **Forward trust-ratio iteration** follows Madsen & Nielsen (2010, Algorithm 6.18) and
  Moré (1978, MINPACK), adapted from the permissive-licensed reference implementations in
  ``google/jaxopt`` (``ca884af``, Apache-2.0, ``jaxopt/_src/levenberg_marquardt.py``),
  ``patrick-kidger/optimistix`` (``5bad66f``, Apache-2.0, ``_least_squares.py`` &
  ``_solver/trust_region.py``), and ``scipy.optimize._lsq`` (BSD-3-Clause, ``trf.py`` &
  ``common.py``): each iteration runs inside a fixed-length :func:`jax.lax.scan` carrying
  a per-instance :class:`~opifex.core.solver.status.Status` code and short-circuiting
  converged batch elements via :func:`jax.lax.cond`.
* **Three linear subproblem solvers** are supported:
  1. ``"qr"`` (default): thin Householder QR on the augmented damped system
     :math:`\begin{bmatrix} W^{1/2} J \\ \sqrt{\mu}\,\operatorname{diag}(s) \end{bmatrix}
     \delta = \begin{bmatrix} -W^{1/2} r \\ 0 \end{bmatrix}` (``jaxopt`` LM QR branch),
     avoiding forming :math:`J^\top W J` so :math:`\kappa(J)` is never squared in ``float32``.
  2. ``"cholesky"``: Cholesky factorization of the damped normal equations
     :math:`(J^\top W J + \mu\,\operatorname{diag}(s^2))\,\delta = -J^\top W r`.
  3. ``"lsmr"``: matrix-free Golub-Kahan bidiagonalization via
     :func:`opifex.uncertainty.linalg.lstsq.lsmr` (Fong & Saunders 2011) using only
     :func:`jax.linearize` and :func:`jax.linear_transpose`, never materializing
     :math:`J \in \mathbb{R}^{m \times p}`.
* **Implicit differentiation** (``implicit_diff=True``) wraps the converged optimum in
  :func:`jax.lax.custom_root` on the exact stationarity system
  :math:`\nabla_\theta \Phi(\theta^\star) = J(\theta^\star)^\top W r(\theta^\star) = 0`,
  evaluating exact second-order curvature (including the non-zero-residual term
  :math:`\sum_i w_i r_i \nabla_{\theta\theta}^2 r_i`) via Pearlmutter forward-over-reverse
  Hessian-vector products (dense solve or matrix-free :func:`jax.lax.custom_linear_solve`
  backed by :func:`opifex.uncertainty.linalg.lstsq.lsmr`).

References:
    * Marquardt, D. W. (1963). "An Algorithm for Least-Squares Estimation of Nonlinear
      Parameters." *J. Soc. Indust. Appl. Math.* 11(2), 431-441.
    * Moré, J. J. (1978). "The Levenberg-Marquardt algorithm: Implementation and theory."
      *Numerical Analysis*, Lecture Notes in Mathematics 630, 105-116.
    * Nielsen, H. B. (1999). *Damping Parameter in Marquardt's Method*. Technical Report
      IMM-REP-1999-05, Technical University of Denmark.
    * Conn, A. R., Gould, N. I. M., & Toint, P. L. (2000). *Trust-Region Methods*. SIAM,
      §17.4.2, Eq. 17.4.4.
    * Madsen, K., & Nielsen, H. B. (2010). *Introduction to Optimization and Data
      Fitting*. Technical University of Denmark, §6.3, Algorithm 6.18.
    * Blondel, M., et al. (2022). *JAXopt: Hardware-Accelerated Implicit Differentiation*
      (``google/jaxopt`` ``ca884af``, Apache-2.0).
"""

import inspect
from collections.abc import Callable
from typing import Any, cast, Literal

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import optax
from flax import struct
from jax.flatten_util import ravel_pytree

from opifex.core.solver.lsmr import lsmr
from opifex.core.solver.problem import Method, Problem, solve
from opifex.core.solver.solution import Solution
from opifex.core.solver.status import Status


LinearSolverKind = Literal["qr", "cholesky", "lsmr"]
ScalingKind = Literal["marquardt", "identity"]


class NonlinearLeastSquaresProblem(Problem):
    r"""Weighted nonlinear least-squares problem :math:`\frac{1}{2} \|W^{1/2} r(\theta, d)\|_2^2`.

    Instance arrays (``initial_params``, ``data``, ``weights``) are pytree leaves so
    batches of curve fits compile once under :func:`jax.vmap` and differentiate with
    respect to observations or weights; ``residual_fn`` is static metadata.

    Attributes:
        residual_fn: Callable ``(params, data) -> residuals`` (or ``(params,) -> residuals``
            when ``data`` is ``None``).
        initial_params: Initial parameter pytree :math:`\theta_0`.
        data: Optional problem data pytree passed to ``residual_fn``.
        weights: Optional non-negative observation weights broadcastable to the
            flattened residual vector. ``None`` corresponds to unit weights.
    """

    residual_fn: Callable[..., jax.Array] = struct.field(pytree_node=False)
    initial_params: Any
    data: Any = None
    weights: jax.Array | None = None


class LevenbergMarquardt(Method[NonlinearLeastSquaresProblem]):
    r"""Madsen-Nielsen / Moré damped Gauss-Newton solver with implicit differentiation.

    All configuration fields are static metadata (``pytree_node=False``), so one compiled
    XLA program serves arbitrary batches of problem instances.

    Attributes:
        max_iterations: Static scan iteration budget.
        gtol: Infinity-norm tolerance on the normal-equation gradient :math:`J^\top W r`.
        xtol: Relative parameter-step norm tolerance.
        ftol: Relative cost-reduction tolerance.
        initial_damping: Initial damping factor :math:`\tau > 0`.
        damping_up: Initial multiplicative growth factor :math:`\nu_0 > 1` on rejected steps.
        damping_down: Minimum shrink factor :math:`\in (0, 1)` in Nielsen's cubic rule.
        linear_solver: Subproblem solver: ``"qr"`` (augmented Householder QR, default),
            ``"cholesky"`` (normal-equations Cholesky), or ``"lsmr"`` (matrix-free
            :func:`opifex.uncertainty.linalg.lstsq.lsmr`).
        scaling: Trust-region diagonal scaling: ``"marquardt"`` (Moré 1978 monotone column
            norms, default) or ``"identity"`` (Madsen & Nielsen 2010).
        lsmr_iterations: Static Krylov iteration budget when ``linear_solver="lsmr"``.
        implicit_diff: Differentiate through :math:`\theta^\star` via
            :func:`jax.lax.custom_root` on :math:`\nabla_\theta \Phi(\theta) = 0`.
    """

    max_iterations: int = struct.field(pytree_node=False, default=50)
    gtol: float = struct.field(pytree_node=False, default=1e-8)
    xtol: float = struct.field(pytree_node=False, default=1e-8)
    ftol: float = struct.field(pytree_node=False, default=1e-8)
    initial_damping: float = struct.field(pytree_node=False, default=1e-3)
    damping_up: float = struct.field(pytree_node=False, default=2.0)
    damping_down: float = struct.field(pytree_node=False, default=1.0 / 3.0)
    linear_solver: LinearSolverKind = struct.field(pytree_node=False, default="qr")
    scaling: ScalingKind = struct.field(pytree_node=False, default="marquardt")
    lsmr_iterations: int = struct.field(pytree_node=False, default=32)
    implicit_diff: bool = struct.field(pytree_node=False, default=True)

    def _validate_configuration(self) -> None:
        """Validate static solver hyperparameters on the host at trace time."""
        if min(self.max_iterations, self.lsmr_iterations) < 1:
            msg = "max_iterations and lsmr_iterations must be at least 1"
            raise ValueError(msg)
        _validate_positive_scalars(
            {
                "gtol": self.gtol,
                "xtol": self.xtol,
                "ftol": self.ftol,
                "initial_damping": self.initial_damping,
            }
        )
        if self.damping_up <= 1.0 or not 0.0 < self.damping_down < 1.0:
            msg = "damping_up must exceed 1.0 and damping_down must lie in (0, 1)"
            raise ValueError(msg)
        if self.linear_solver not in ("qr", "cholesky", "lsmr"):
            msg = f"linear_solver must be 'qr', 'cholesky', or 'lsmr', got {self.linear_solver!r}"
            raise ValueError(msg)
        if self.scaling not in ("marquardt", "identity"):
            msg = f"scaling must be 'marquardt' or 'identity', got {self.scaling!r}"
            raise ValueError(msg)

    def solve(self, problem: NonlinearLeastSquaresProblem) -> Solution:
        """Solve ``problem`` and return a traced :class:`~opifex.core.solver.solution.Solution`.

        Args:
            problem: Nonlinear least-squares problem instance.

        Returns:
            Solution carrying ``fields={"params": ..., "residuals": ...}``,
            ``metrics={"cost": ..., "grad_inf_norm": ..., "final_damping": ...}``,
            traced ``status``, and ``stats={"num_iterations": ...}``.
        """
        self._validate_configuration()
        dtype = _common_floating_dtype(problem.initial_params, problem.data, problem.weights)
        init_typed = jax.tree.map(
            lambda x: (
                jnp.asarray(x, dtype=dtype)
                if jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact)
                else x
            ),
            problem.initial_params,
        )
        theta0, unravel_typed = ravel_pytree(init_typed)

        def raw_eval(theta_vec: jax.Array) -> jax.Array:
            """Evaluate unweighted flattened residual vector at ``theta_vec``."""
            return _call_residual(
                problem.residual_fn, unravel_typed(theta_vec), problem.data
            ).astype(dtype)

        num_obs = jax.eval_shape(raw_eval, theta0).shape[0]
        sqrt_w = _prepare_sqrt_weights(problem.weights, num_obs, dtype)

        def weighted_eval(theta_vec: jax.Array) -> jax.Array:
            r"""Evaluate :math:`W^{1/2} r(\theta)` with zero-safe weight square root."""
            return sqrt_w * raw_eval(theta_vec)

        def loss_fn(theta_vec: jax.Array) -> jax.Array:
            r"""Scalar weighted least-squares objective :math:`\frac{1}{2}\|W^{1/2}r\|_2^2`."""
            rw = weighted_eval(theta_vec)
            return 0.5 * jnp.sum(rw * rw)

        if self.implicit_diff:
            optimality = jax.grad(loss_fn)

            def root_solver(
                _: Callable[[jax.Array], jax.Array],
                init_vec: jax.Array,
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
                """Run forward Levenberg-Marquardt scan with floating aux leaves."""
                sol_vec, (st_i32, n_i32, damp_f) = _run_lm_loop(init_vec, weighted_eval, self)
                # jax 0.11.1 custom_root creates aux tangents via zeros_like_jaxval rather
                # than float0 on int32 leaves; carry integer codes as floating aux leaves and
                # restore int32 immediately after custom_root returns.
                return sol_vec, (st_i32.astype(dtype), n_i32.astype(dtype), damp_f)

            def tangent_solve(g: Callable[[jax.Array], jax.Array], b: jax.Array) -> jax.Array:
                """Dispatch dense HVP Hessian solve vs matrix-free LSMR solve."""
                if self.linear_solver == "lsmr":
                    return _matrix_free_tangent_solve(g, b, self.lsmr_iterations)
                return _dense_tangent_solve(g, b)

            root_out: Any = jax.lax.custom_root(
                optimality,
                theta0,
                root_solver,
                tangent_solve,
                has_aux=True,
            )
            theta_star, (status_f, steps_f, final_damping) = root_out
            status = status_f.astype(jnp.int32)
            steps = steps_f.astype(jnp.int32)
        else:
            theta_star, (status, steps, final_damping) = _run_lm_loop(theta0, weighted_eval, self)

        final_raw_res = raw_eval(theta_star)
        final_res_w = sqrt_w * final_raw_res
        final_cost = 0.5 * jnp.sum(final_res_w * final_res_w)
        final_grad = jax.grad(loss_fn)(theta_star)

        return Solution(
            fields={
                "params": unravel_typed(theta_star),
                "residuals": final_raw_res,
            },
            metrics={
                "cost": final_cost,
                "grad_inf_norm": jnp.max(jnp.abs(final_grad)),
                "final_damping": final_damping,
            },
            status=status,
            stats={"num_iterations": steps},
        )


class _LMCarry(struct.PyTreeNode):
    """Internal scan state for one Levenberg-Marquardt solve."""

    theta: jax.Array
    res_w: jax.Array
    cost: jax.Array
    jac_w: jax.Array
    grad: jax.Array
    col_scale: jax.Array
    damping: jax.Array
    nu: jax.Array
    status: jax.Array
    steps: jax.Array


def _call_residual(
    residual_fn: Callable[..., jax.Array],
    params: Any,
    data: Any,
) -> jax.Array:
    """Invoke ``residual_fn`` with ``(params, data)`` or ``(params,)`` when ``data`` is ``None``."""
    if data is None:
        try:
            sig = inspect.signature(residual_fn)
            positional = [
                p
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            has_varargs = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
            )
            if len(positional) == 1 and not has_varargs:
                return jnp.asarray(residual_fn(params)).reshape(-1)
        except (TypeError, ValueError):
            pass
    return jnp.asarray(residual_fn(params, data)).reshape(-1)


def _linearize_dense(
    weighted_eval: Callable[[jax.Array], jax.Array],
    theta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate weighted residual, Jacobian, and gradient in a single primal pass."""
    res_w, lin = jax.linearize(weighted_eval, theta)
    eye = jnp.eye(theta.shape[0], dtype=theta.dtype)
    jac_w = jax.vmap(lin, out_axes=-1)(eye)
    return res_w, jac_w, jac_w.T @ res_w


def _linearize_matrix_free(
    weighted_eval: Callable[[jax.Array], jax.Array],
    theta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate weighted residual and VJP gradient without materializing the Jacobian."""
    res_w, lin = jax.linearize(weighted_eval, theta)
    vjp_fn = jax.linear_transpose(lin, theta)
    (grad,) = vjp_fn(res_w)
    empty_jac = jnp.zeros((0, theta.shape[0]), dtype=theta.dtype)
    return res_w, empty_jac, grad


def _eval_derivatives(
    weighted_eval: Callable[[jax.Array], jax.Array],
    theta: jax.Array,
    linear_solver: LinearSolverKind,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Dispatch derivative evaluation according to ``linear_solver``."""
    if linear_solver == "lsmr":
        return _linearize_matrix_free(weighted_eval, theta)
    return _linearize_dense(weighted_eval, theta)


def _solve_qr_step(
    jac_w: jax.Array,
    res_w: jax.Array,
    col_scale: jax.Array,
    damping: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solve augmented damped least-squares step via thin Householder QR (jaxopt LM)."""
    num_obs = res_w.shape[0]
    sqrt_mu_diag = jnp.diag(jnp.sqrt(damping) * col_scale)
    aug_jac = jnp.concatenate([jac_w, sqrt_mu_diag], axis=0)
    q_mat, r_tri = jsp_linalg.qr(aug_jac, mode="economic")
    rhs_proj = -(q_mat[:num_obs, :].T @ res_w)
    delta = jsp_linalg.solve_triangular(r_tri, rhs_proj, lower=False)
    return delta, jac_w @ delta


def _solve_cholesky_step(
    jac_w: jax.Array,
    grad: jax.Array,
    col_scale: jax.Array,
    damping: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solve damped normal equations via Cholesky with safe ridge fallback."""
    dtype = grad.dtype
    jtj = jac_w.T @ jac_w
    lhs = jtj + damping * jnp.diag(col_scale * col_scale)
    neg_grad = -grad
    cho_fac = jsp_linalg.cho_factor(lhs, lower=True)
    delta_cho = jsp_linalg.cho_solve(cho_fac, neg_grad)
    ridge = jnp.finfo(dtype).eps * jnp.maximum(jnp.max(jnp.diag(jtj)), jnp.asarray(1.0, dtype))
    delta_lu = jnp.linalg.solve(lhs + ridge * jnp.eye(grad.shape[0], dtype=dtype), neg_grad)
    delta = jnp.where(jnp.all(jnp.isfinite(delta_cho)), delta_cho, delta_lu)
    return delta, jac_w @ delta


def _solve_lsmr_step(
    weighted_eval: Callable[[jax.Array], jax.Array],
    theta: jax.Array,
    res_w: jax.Array,
    damping: jax.Array,
    lsmr_iterations: int,
) -> tuple[jax.Array, jax.Array]:
    """Solve matrix-free damped least squares via :func:`opifex.uncertainty.linalg.lstsq.lsmr`."""
    _, jvp_fn = jax.linearize(weighted_eval, theta)
    vjp_fn = jax.linear_transpose(jvp_fn, theta)
    delta = lsmr(
        matvec=jvp_fn,
        matvec_transpose=lambda u: vjp_fn(u)[0],
        rhs=-res_w,
        dim_cols=theta.shape[0],
        num_matvecs=lsmr_iterations,
        damping=cast("Any", jnp.sqrt(damping)),
    )
    return delta, jvp_fn(delta)


def _compute_trial_step(
    carry: _LMCarry,
    weighted_eval: Callable[[jax.Array], jax.Array],
    method: LevenbergMarquardt,
) -> tuple[jax.Array, jax.Array]:
    """Compute trial parameter increment ``delta`` and linearized residual step ``J_w @ delta``."""
    if method.linear_solver == "qr":
        return _solve_qr_step(carry.jac_w, carry.res_w, carry.col_scale, carry.damping)
    if method.linear_solver == "cholesky":
        return _solve_cholesky_step(carry.jac_w, carry.grad, carry.col_scale, carry.damping)
    return _solve_lsmr_step(
        weighted_eval,
        carry.theta,
        carry.res_w,
        carry.damping,
        method.lsmr_iterations,
    )


def _update_column_scale(
    prev_scale: jax.Array,
    jac_w: jax.Array,
    scaling: ScalingKind,
    linear_solver: LinearSolverKind,
) -> jax.Array:
    """Update diagonal scaling (Moré 1978 / ``scipy.optimize._lsq.common.compute_jac_scale``)."""
    if scaling == "identity" or linear_solver == "lsmr":
        return prev_scale
    col_norms = jnp.sqrt(jnp.sum(jac_w * jac_w, axis=0))
    return jnp.maximum(prev_scale, col_norms)


def _classify_step_status(
    *,
    finite_state: jax.Array,
    pre_grad_ok: jax.Array,
    grad_upd_inf: jax.Array,
    cauchy_converged: jax.Array,
    overdamped_away_from_min: jax.Array,
    damping_overflow: jax.Array,
    gtol: float,
) -> jax.Array:
    """Map step diagnostics to a traced :class:`~opifex.core.solver.status.Status` code."""
    stalled_code = jnp.where(
        overdamped_away_from_min,
        jnp.asarray(Status.STALLED, dtype=jnp.int32),
        jnp.asarray(Status.STALLED_SUCCESS, dtype=jnp.int32),
    )
    cand_status = jnp.asarray(Status.DEFAULT, dtype=jnp.int32)
    cand_status = jnp.where(damping_overflow | cauchy_converged, stalled_code, cand_status)
    cand_status = jnp.where(
        grad_upd_inf <= gtol,
        jnp.asarray(Status.SUCCESS, dtype=jnp.int32),
        cand_status,
    )
    cand_status = jnp.where(
        pre_grad_ok,
        jnp.asarray(Status.SUCCESS, dtype=jnp.int32),
        cand_status,
    )
    return jnp.where(
        ~finite_state,
        jnp.asarray(Status.NONFINITE, dtype=jnp.int32),
        cand_status,
    )


def _lm_active_step(
    carry: _LMCarry,
    weighted_eval: Callable[[jax.Array], jax.Array],
    method: LevenbergMarquardt,
) -> _LMCarry:
    """Execute one Madsen-Nielsen (2010, Algorithm 6.18) trust-ratio step on an active problem."""
    dtype = carry.theta.dtype
    eps = jnp.finfo(dtype).eps

    finite_state = (
        jnp.all(jnp.isfinite(carry.theta))
        & jnp.isfinite(carry.cost)
        & jnp.all(jnp.isfinite(carry.grad))
    )
    grad_inf = jnp.max(jnp.abs(carry.grad))
    pre_grad_ok = grad_inf <= method.gtol

    delta, j_delta = _compute_trial_step(carry, weighted_eval, method)
    step_norm = optax.safe_norm(delta, 0.0)
    step_tol = method.xtol * (method.xtol + optax.safe_norm(carry.theta, 0.0))
    # Madsen & Nielsen (2010, Algorithm 6.18) & scipy.optimize._lsq.common.check_termination:
    # check parameter-step convergence ||delta||_2 <= xtol * (xtol + ||theta||_2) directly.
    xtol_converged = step_norm <= step_tol

    theta_cand = carry.theta + delta
    res_cand = weighted_eval(theta_cand)
    cost_cand = 0.5 * jnp.sum(res_cand * res_cand)

    # Factored quadratic reduction (optimistix ClassicalTrustRegion / scipy evaluate_quadratic)
    # and difference-of-squares objective reduction with Conn, Gould & Toint (2000, §17.4.2,
    # Eq. 17.4.4) rounding offset 10 * eps * max(cost, 1).
    pred_reduction = -jnp.dot(delta, carry.grad) - 0.5 * jnp.sum(j_delta * j_delta)
    actual_reduction = 0.5 * jnp.sum((carry.res_w - res_cand) * (carry.res_w + res_cand))
    delta_f = 10.0 * eps * jnp.maximum(carry.cost, jnp.asarray(1.0, dtype=dtype))
    has_pred = pred_reduction > jnp.asarray(0.0, dtype=dtype)
    safe_pred = jnp.where(has_pred, pred_reduction + delta_f, jnp.asarray(1.0, dtype=dtype))
    rho = jnp.where(
        has_pred,
        jnp.clip((actual_reduction + delta_f) / safe_pred, -10.0, 10.0),
        jnp.asarray(0.0, dtype=dtype),
    )
    accept = (
        (~xtol_converged)
        & has_pred
        & (rho > 0.0)
        & jnp.isfinite(cost_cand)
        & jnp.all(jnp.isfinite(res_cand))
    )

    def accepted_derivs(th_c: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Re-linearize only when the trial step is accepted."""
        _, j_c, g_c = _eval_derivatives(weighted_eval, th_c, method.linear_solver)
        s_c = _update_column_scale(carry.col_scale, j_c, method.scaling, method.linear_solver)
        return j_c, g_c, s_c

    jac_upd, grad_upd, scale_upd = jax.lax.cond(
        accept,
        accepted_derivs,
        lambda _: (carry.jac_w, carry.grad, carry.col_scale),
        theta_cand,
    )
    theta_upd = jnp.where(accept, theta_cand, carry.theta)
    res_upd = jnp.where(accept, res_cand, carry.res_w)
    cost_upd = jnp.where(accept, cost_cand, carry.cost)

    # Nielsen (1999) smooth cubic shrink on acceptance; geometric growth on rejection.
    cubic_shrink = jnp.maximum(
        jnp.asarray(method.damping_down, dtype=dtype),
        1.0 - (2.0 * rho - 1.0) ** 3,
    )
    damping_acc = jnp.maximum(carry.damping * cubic_shrink, eps)
    damping_rej = jnp.minimum(carry.damping * carry.nu, jnp.asarray(1e16, dtype=dtype))
    damping_upd = jnp.where(accept, damping_acc, damping_rej)
    nu_upd = jnp.where(accept, jnp.asarray(method.damping_up, dtype=dtype), 2.0 * carry.nu)

    # Residual-space Cauchy check (optimistix cauchy_termination + scipy ratio > 0.25 gate).
    res_tol = method.ftol * (method.ftol + optax.safe_norm(res_cand, 0.0))
    ftol_converged = (
        accept & (rho > 0.25) & (optax.safe_norm(res_cand - carry.res_w, 0.0) <= res_tol)
    )
    damping_overflow = (
        (~accept) & (~xtol_converged) & (damping_rej >= jnp.asarray(1e15, dtype=dtype))
    )
    grad_floor = jnp.sqrt(eps) * jnp.maximum(carry.cost, jnp.asarray(1.0, dtype=dtype))
    overdamped_away_from_min = (carry.damping >= jnp.asarray(1e12, dtype=dtype)) & (
        grad_inf > grad_floor
    )

    cand_status = _classify_step_status(
        finite_state=finite_state,
        pre_grad_ok=pre_grad_ok,
        grad_upd_inf=jnp.max(jnp.abs(grad_upd)),
        cauchy_converged=xtol_converged | ftol_converged,
        overdamped_away_from_min=overdamped_away_from_min,
        damping_overflow=damping_overflow,
        gtol=method.gtol,
    )

    take = finite_state & (~pre_grad_ok)
    return _LMCarry(
        theta=jnp.where(take, theta_upd, carry.theta),
        res_w=jnp.where(take, res_upd, carry.res_w),
        cost=jnp.where(take, cost_upd, carry.cost),
        jac_w=jnp.where(take, jac_upd, carry.jac_w),
        grad=jnp.where(take, grad_upd, carry.grad),
        col_scale=jnp.where(take, scale_upd, carry.col_scale),
        damping=jnp.where(take, damping_upd, carry.damping),
        nu=jnp.where(take, nu_upd, carry.nu),
        status=cand_status,
        steps=carry.steps + take.astype(jnp.int32),
    )


def _initial_scale_and_damping(
    jac_w: jax.Array,
    theta0: jax.Array,
    method: LevenbergMarquardt,
) -> tuple[jax.Array, jax.Array]:
    """Initialize column scaling and damping (``scipy.optimize._lsq.common`` / Madsen-Nielsen)."""
    dtype = theta0.dtype
    tau = jnp.asarray(method.initial_damping, dtype=dtype)
    ones = jnp.ones((theta0.shape[0],), dtype=dtype)
    if method.linear_solver == "lsmr":
        return ones, tau
    col_sq = jnp.sum(jac_w * jac_w, axis=0)
    if method.scaling == "identity":
        max_diag = jnp.maximum(jnp.max(col_sq), jnp.asarray(1.0, dtype=dtype))
        return ones, tau * max_diag
    col_norms = jnp.sqrt(col_sq)
    return jnp.where(col_norms == 0.0, ones, col_norms), tau


def _run_lm_loop(
    theta0: jax.Array,
    weighted_eval: Callable[[jax.Array], jax.Array],
    method: LevenbergMarquardt,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Run ``method.max_iterations`` scan steps and finalize ``Status``."""
    dtype = theta0.dtype
    res0, jac0, grad0 = _eval_derivatives(weighted_eval, theta0, method.linear_solver)
    cost0 = 0.5 * jnp.sum(res0 * res0)
    col_scale0, damping0 = _initial_scale_and_damping(jac0, theta0, method)

    init_carry = _LMCarry(
        theta=theta0,
        res_w=res0,
        cost=cost0,
        jac_w=jac0,
        grad=grad0,
        col_scale=col_scale0,
        damping=damping0,
        nu=jnp.asarray(method.damping_up, dtype=dtype),
        status=jnp.asarray(Status.DEFAULT, dtype=jnp.int32),
        steps=jnp.asarray(0, dtype=jnp.int32),
    )

    def body(carry: _LMCarry, _: None) -> tuple[_LMCarry, None]:
        """Short-circuit already-terminated elements via :func:`jax.lax.cond`."""
        active = carry.status == jnp.asarray(Status.DEFAULT, dtype=jnp.int32)
        next_carry = jax.lax.cond(
            active,
            lambda c: _lm_active_step(c, weighted_eval, method),
            lambda c: c,
            carry,
        )
        return next_carry, None

    final_carry, _ = jax.lax.scan(body, init_carry, None, length=method.max_iterations)

    unset = final_carry.status == jnp.asarray(Status.DEFAULT, dtype=jnp.int32)
    final_grad_ok = jnp.max(jnp.abs(final_carry.grad)) <= method.gtol
    fallback_status = jnp.where(
        final_grad_ok,
        jnp.asarray(Status.SUCCESS, dtype=jnp.int32),
        jnp.asarray(Status.MAX_ITERS, dtype=jnp.int32),
    )
    status = jnp.where(unset, fallback_status, final_carry.status)
    status = jnp.where(
        jnp.all(jnp.isfinite(final_carry.theta)) & jnp.isfinite(final_carry.cost),
        status,
        jnp.asarray(Status.NONFINITE, dtype=jnp.int32),
    )
    return final_carry.theta, (status, final_carry.steps, final_carry.damping)


def _dense_tangent_solve(
    g: Callable[[jax.Array], jax.Array],
    b: jax.Array,
) -> jax.Array:
    """Solve linearized stationarity system via Pearlmutter HVP columns ``g(e_j)``."""
    eye = jnp.eye(b.shape[0], dtype=b.dtype)
    hess = jax.vmap(g, out_axes=-1)(eye)
    sym_h = 0.5 * (hess + hess.T)
    safe_h = jnp.where(jnp.all(jnp.isfinite(sym_h)), sym_h, eye)
    min_eig = jnp.linalg.eigvalsh(safe_h)[0]
    scale = jnp.maximum(jnp.max(jnp.abs(jnp.diag(safe_h))), jnp.asarray(1.0, dtype=b.dtype))
    eig_floor = jnp.finfo(b.dtype).eps * scale * 16.0
    shift = jnp.maximum(jnp.asarray(0.0, dtype=b.dtype), eig_floor - min_eig)
    return jnp.linalg.solve(safe_h + shift * eye, b)


def _matrix_free_tangent_solve(
    g: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    lsmr_iterations: int,
) -> jax.Array:
    """Matrix-free implicit tangent solve via :func:`jax.lax.custom_linear_solve` + LSMR."""
    ridge = jnp.finfo(b.dtype).eps * 16.0

    def reg_matvec(v: jax.Array) -> jax.Array:
        """Apply regularized symmetric Hessian-vector product."""
        return g(v) + ridge * v

    return jax.lax.custom_linear_solve(
        reg_matvec,
        b,
        solve=lambda mv, rhs: lsmr(
            matvec=mv,
            matvec_transpose=mv,
            rhs=rhs,
            dim_cols=rhs.shape[0],
            num_matvecs=lsmr_iterations,
        ),
        symmetric=True,
    )


def _validate_positive_scalars(values: dict[str, float]) -> None:
    """Ensure every named scalar hyperparameter is strictly positive."""
    for name, val in values.items():
        if val <= 0.0:
            msg = f"{name} must be strictly positive, got {val}"
            raise ValueError(msg)


def _common_floating_dtype(*trees: Any) -> jnp.dtype:
    """Promote all inexact array leaves across ``trees`` with a ``float32`` floor."""
    dtypes = [jnp.float32]
    for tree in trees:
        for leaf in jax.tree_util.tree_leaves(tree):
            arr_dtype = getattr(leaf, "dtype", None)
            if arr_dtype is not None and jnp.issubdtype(arr_dtype, jnp.inexact):
                dtypes.append(arr_dtype)
    return jnp.result_type(*dtypes)


def _prepare_sqrt_weights(
    weights: jax.Array | None,
    num_obs: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Broadcast observation weights and take a zero-safe elementwise square root."""
    if weights is None:
        return jnp.ones((num_obs,), dtype=dtype)
    raw_w = jnp.asarray(weights, dtype=dtype).reshape(-1)
    w_vec = jnp.broadcast_to(raw_w, (num_obs,))
    has_pos_w = w_vec > 0.0
    return jnp.where(has_pos_w, jnp.sqrt(jnp.where(has_pos_w, w_vec, 1.0)), 0.0)


def levenberg_marquardt(
    residual_fn: Callable[..., jax.Array],
    initial_params: Any,
    *,
    data: Any = None,
    weights: jax.Array | None = None,
    method: LevenbergMarquardt = LevenbergMarquardt(),
) -> Solution:
    r"""Functional convenience wrapper solving a nonlinear least-squares problem.

    Args:
        residual_fn: Residual function ``(params, data) -> residuals`` (or
            ``(params,) -> residuals`` when ``data`` is ``None``).
        initial_params: Initial parameter guess :math:`\theta_0`.
        data: Optional observation data pytree passed to ``residual_fn``.
        weights: Optional non-negative observation weights.
        method: Configured :class:`LevenbergMarquardt` method instance.

    Returns:
        Traced :class:`~opifex.core.solver.solution.Solution`.
    """
    problem = NonlinearLeastSquaresProblem(
        residual_fn=residual_fn,
        initial_params=initial_params,
        data=data,
        weights=weights,
    )
    return solve(problem, method)


__all__ = [
    "LevenbergMarquardt",
    "LinearSolverKind",
    "NonlinearLeastSquaresProblem",
    "ScalingKind",
    "levenberg_marquardt",
]
