"""Verification suite for pure-JAX Levenberg-Marquardt nonlinear least squares.

Covers three verification pillars:
1. Numerical and theoretical accuracy in ``float32`` and ``float64`` for both primal
   solutions and reverse-mode gradients (closed-form weighted linear least squares,
   exact ``1 / 1.1`` nonzero-residual strong-convexity minimiser and full-Hessian IFT
   sensitivity bounds, pyadjoint Taylor remainder convergence rate >= 1.9 on
   non-zero-residual nonlinear curves, ill-conditioned ``float32`` QR vs Cholesky
   comparison, and ``float32``-vs-``float64`` accuracy).
2. JAX and ``flax.nnx`` transformation compatibility (``jit``/``vmap`` ULP gradient
   agreement, ``expect_compiles`` single-trace caching, ``check_input_gradients``, and
   end-to-end ``@nnx.jit`` + ``nnx.Optimizer`` bi-level module training).
3. Speed, FLOP accounting, and architectural contracts (10,000-curve batched ``vmap``
   fit timed via ``calibrax.profiling.time_calls`` + ``FlopsCounter``, declared import
   isolation, and ``opifex.core.solver`` / ``SolutionDistribution`` integration).
"""

import ast
import math
import typing
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from calibrax.profiling import FlopsCounter, time_calls
from flax import nnx
from jax.test_util import check_grads
from substrax.testing.compiles import expect_compiles
from substrax.testing.gradients import check_input_gradients, check_parameter_gradients
from tests.fields.staggered_support import (
    gradient_transform_agreement,
    gradient_versus_central_difference,
    JIT_GRADIENT_ULP,
    TAYLOR_RATE_FLOOR,
    taylor_remainder_rates,
)

import opifex.solvers.nonlinear_least_squares as nls_mod
from opifex.capabilities import UQ_CAPABILITIES
from opifex.core.solver import (
    IncompatibleProblemError,
    is_successful,
    Problem,
    solve,
    Status,
)
from opifex.core.solver.report import timed_solve
from opifex.solvers import (
    levenberg_marquardt,
    LevenbergMarquardt,
    NonlinearLeastSquaresProblem,
    SOLVER_CAPABILITIES,
)
from opifex.solvers.nonlinear_least_squares import LinearSolverKind
from opifex.uncertainty.scientific.solutions import aggregate_solver_solutions


class _ForeignProblem(Problem):
    """Unrelated problem type used to verify runtime problem-family rejection."""

    x: jax.Array


def _hill_4pl(params: jax.Array, log_conc: jax.Array) -> jax.Array:
    """Evaluate 4-parameter Hill curve ``bottom + (top - bottom) / (1 + 10^((ec50 - x) * h))``."""
    bottom, top, log_ec50, hill = params[0], params[1], params[2], params[3]
    exponent = jnp.clip((log_ec50 - log_conc) * hill, -30.0, 30.0)
    return bottom + (top - bottom) / (1.0 + jnp.power(10.0, exponent))


def _hill_residual(params: jax.Array, data: tuple[jax.Array, jax.Array]) -> jax.Array:
    """Residual vector ``model(params, log_conc) - y_obs``."""
    log_conc, y_obs = data
    return _hill_4pl(params, log_conc) - y_obs


class _ObservationWeightNet(nnx.Module):
    """Small NNX module predicting positive observation weights for bi-level fitting."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.hidden = nnx.Linear(1, 8, rngs=rngs)
        self.head = nnx.Linear(8, 1, rngs=rngs)

    def __call__(self, log_conc: jax.Array) -> jax.Array:
        features = nnx.tanh(self.hidden(log_conc[:, None]))
        return jax.nn.softplus(self.head(features).squeeze(-1)) + 0.1


def _tangent_bwd_contract_ceil(size: int, dtype: jnp.dtype, linear_solver: str) -> float:
    """Normwise Frobenius backward-error ceiling guaranteed by the active tangent solver."""
    u_round = 0.5 * float(jnp.finfo(dtype).eps)
    gamma_3n = (3.0 * size * u_round) / (1.0 - 3.0 * size * u_round)
    lu_ceil = min(max(64.0 * u_round, 8.0 * math.sqrt(size) * gamma_3n), 0.1)
    if linear_solver == "lsmr":
        eff_rtol = max(16.0 * u_round, 1e-12)
        return max(100.0 * eff_rtol, lu_ceil)
    return lu_ceil


def test_nls_import_isolation_and_capability_contracts() -> None:
    """Module imports only declared JAX/Flax/Optax/Opifex foundations and registers capabilities."""
    source = Path(nls_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed_roots = {
        "__future__",
        "collections",
        "flax",
        "jax",
        "opifex",
        "optax",
        "typing",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots <= allowed_roots
    assert "TYPE_CHECKING" not in source
    for obj in (NonlinearLeastSquaresProblem, LevenbergMarquardt, levenberg_marquardt):
        assert typing.get_type_hints(obj)
    assert "solver:LevenbergMarquardt" in SOLVER_CAPABILITIES
    assert "solver:LevenbergMarquardt" in UQ_CAPABILITIES

    with pytest.raises(IncompatibleProblemError, match="NonlinearLeastSquaresProblem"):
        solve(_ForeignProblem(x=jnp.ones((2,))), LevenbergMarquardt())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"max_iterations": 0}, "at least 1"),
        ({"lsmr_iterations": 0}, "at least 1"),
        ({"gtol": 0.0}, "strictly positive"),
        ({"xtol": -1e-6}, "strictly positive"),
        ({"ftol": 0.0}, "strictly positive"),
        ({"initial_damping": 0.0}, "strictly positive"),
        ({"damping_up": 1.0}, "damping_up"),
        ({"damping_down": 1.0}, "damping_down"),
        ({"linear_solver": "svd"}, "linear_solver"),
        ({"scaling": "none"}, "scaling"),
    ],
)
def test_invalid_configuration_rejected_eagerly(kwargs: dict[str, object], pattern: str) -> None:
    """Invalid static hyperparameters fail fast before tracing."""
    prob = NonlinearLeastSquaresProblem(
        residual_fn=lambda th: th - 1.0,
        initial_params=jnp.zeros((2,)),
    )
    method = LevenbergMarquardt(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=pattern):
        solve(prob, method)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("linear_solver", ["qr", "cholesky", "lsmr"])
def test_exact_weighted_linear_least_squares_values_and_jacobians(
    dtype: jnp.dtype,
    linear_solver: LinearSolverKind,
) -> None:
    """Linear WLS matches closed-form normal equations for both values and d(theta*)/dy."""
    enable_x64 = dtype == jnp.float64
    with jax.enable_x64(enable_x64):
        design = jnp.asarray(
            [
                [1.0, -0.8, 0.64],
                [1.0, -0.4, 0.16],
                [1.0, 0.0, 0.0],
                [1.0, 0.3, 0.09],
                [1.0, 0.7, 0.49],
                [1.0, 1.1, 1.21],
            ],
            dtype=dtype,
        )
        weights = jnp.asarray([1.0, 2.0, 0.5, 1.5, 3.0, 1.2], dtype=dtype)
        y_obs = jnp.asarray([2.1, 1.3, 0.9, 1.2, 2.4, 4.1], dtype=dtype)
        theta0 = jnp.zeros((3,), dtype=dtype)

        xtw = design.T * weights[None, :]
        xtwx = xtw @ design

        tol_val = 1e-12 if enable_x64 else 2e-5
        method = LevenbergMarquardt(
            max_iterations=35,
            gtol=tol_val,
            xtol=tol_val,
            ftol=tol_val,
            initial_damping=1e-6,
            linear_solver=linear_solver,
            lsmr_iterations=24,
        )

        def fit_params(y_in: jax.Array) -> jax.Array:
            sol = levenberg_marquardt(
                lambda th, y_t: design @ th - y_t,
                theta0,
                data=y_in,
                weights=weights,
                method=method,
            )
            return sol.fields["params"]

        theta_hat = fit_params(y_obs)
        jac_hat = jax.jacrev(fit_params)(y_obs)

        with jax.enable_x64(True):
            xtw_64 = np.asarray(xtw, dtype=np.float64)
            xtwx_64 = np.asarray(xtwx, dtype=np.float64)
            y_64 = np.asarray(y_obs, dtype=np.float64)
            th_64 = np.asarray(theta_hat, dtype=np.float64)
            exact_th_64 = np.linalg.solve(xtwx_64, xtw_64 @ y_64)
            exact_jac_64 = np.linalg.solve(xtwx_64, xtw_64)
            svals = np.linalg.svd(xtwx_64, compute_uv=False)
            sigma_min = float(svals[-1])
            cond = float(svals[0] / svals[-1])
            grad_64 = xtwx_64 @ th_64 - xtw_64 @ y_64
            eps64 = float(np.finfo(np.float64).eps)

        # Exact normal-equations perturbation identity: theta_hat - theta* = (X^T W X)^{-1} grad(theta_hat).
        val_err = float(np.linalg.norm(th_64 - exact_th_64))
        assert val_err <= float(np.linalg.norm(grad_64)) / sigma_min + 8.0 * eps64 * float(
            np.linalg.norm(exact_th_64)
        )

        # Higham (2002, Thm. 7.2) sensitivity of H V = X^T W under the tangent solver's
        # normwise backward-error contract (accounting for Frobenius-vs-2-norm sqrt(p)):
        bwd_ceil = _tangent_bwd_contract_ceil(3, dtype, linear_solver)
        jac_rel_bound = (
            cond * math.sqrt(3.0) * bwd_ceil / max(1.0 - cond * math.sqrt(3.0) * bwd_ceil, 0.5)
        )
        rel_jac = float(
            np.linalg.norm(np.asarray(jac_hat, dtype=np.float64) - exact_jac_64)
            / np.linalg.norm(exact_jac_64)
        )
        assert rel_jac <= jac_rel_bound + 4.0 * float(jnp.finfo(dtype).eps)


def test_qr_outperforms_cholesky_on_ill_conditioned_float32_jacobian() -> None:
    """Householder QR solves the Lauchli (1961) matrix in 1 step where J^T J rounds singular."""
    with jax.enable_x64(False):
        # Lauchli (1961) / Golub & Van Loan Example 5.3.2: delta = 2e-4 has delta^2 = 4e-8
        # < 0.5 * eps32, so A^T A rounds to [[1, 1], [1, 1]] (rank 1) in float32 while QR
        # preserves delta in A directly on a single Gauss-Newton step.
        delta = jnp.float32(2e-4)
        a_32 = jnp.array([[1.0, 1.0], [delta, 0.0], [0.0, delta]], dtype=jnp.float32)
        theta_true_32 = jnp.array([2.0, -1.0], dtype=jnp.float32)
        b_32 = jnp.array([1.0, 2.0 * delta, -delta], dtype=jnp.float32)
        init_32 = jnp.zeros((2,), dtype=jnp.float32)

        sol_qr = levenberg_marquardt(
            lambda th, rhs: a_32 @ th - rhs,
            init_32,
            data=b_32,
            method=LevenbergMarquardt(
                max_iterations=1,
                gtol=1e-12,
                xtol=1e-12,
                ftol=1e-12,
                initial_damping=1e-12,
                linear_solver="qr",
                scaling="identity",
            ),
        )
        sol_cho = levenberg_marquardt(
            lambda th, rhs: a_32 @ th - rhs,
            init_32,
            data=b_32,
            method=LevenbergMarquardt(
                max_iterations=1,
                gtol=1e-12,
                xtol=1e-12,
                ftol=1e-12,
                initial_damping=1e-12,
                linear_solver="cholesky",
                scaling="identity",
            ),
        )
        err_qr = float(jnp.linalg.norm(sol_qr.fields["params"] - theta_true_32))
        err_cho = float(jnp.linalg.norm(sol_cho.fields["params"] - theta_true_32))
        assert err_qr < 1e-4
        assert err_cho > 1.0


def _verify_1_over_1_1_method(
    method: LevenbergMarquardt,
    residual_fn: typing.Callable[[jax.Array, tuple[jax.Array, jax.Array]], jax.Array],
    theta0: jax.Array,
    a_vec: jax.Array,
    b_vec: jax.Array,
    c_vec: jax.Array,
    s_zero: jax.Array,
    dtype: jnp.dtype,
    linear_solver: str,
) -> None:
    """Assert strong-convexity optimality and full-Hessian IFT bounds for one method."""
    u_round = 0.5 * float(jnp.finfo(dtype).eps)
    mu_strong = 2.0 * (1.1**2) * float(jnp.min(a_vec + b_vec / 5.0))
    sol = levenberg_marquardt(residual_fn, theta0, data=(c_vec, s_zero), method=method)
    assert bool(sol.is_converged)

    with jax.enable_x64(True):
        th_64 = sol.fields["params"].astype(jnp.float64)
        c_64 = c_vec.astype(jnp.float64)
        a_64 = a_vec.astype(jnp.float64)
        b_64 = b_vec.astype(jnp.float64)
        th_star_64 = c_64 / 1.1
        u_64 = 1.1 * th_64 - c_64
        grad_64 = 1.1 * (2.0 * (a_64 + b_64 / 5.0) * u_64 + 0.04 * (u_64**3))
        dist = float(jnp.linalg.norm(th_64 - th_star_64))
        grad_norm = float(jnp.linalg.norm(grad_64))
        exact_cost = float(jnp.sum(b_64**2))
        th_star_norm = float(jnp.linalg.norm(th_star_64))
        eps_64 = float(jnp.finfo(jnp.float64).eps)
        hess_diag_64 = (1.1**2) * (2.0 * (a_64 + b_64 / 5.0) + 0.12 * (u_64**2))
        cond_h = float(jnp.max(hess_diag_64) / jnp.min(hess_diag_64))
        inv_h_norm = float(1.0 / jnp.min(hess_diag_64))
        lip_h = float(0.24 * (1.1**3) * jnp.max(jnp.abs(u_64)))

    gamma_12 = (12.0 * u_round) / (1.0 - 12.0 * u_round)
    assert abs(float(sol.metrics["cost"]) - exact_cost) <= (
        grad_norm * dist + gamma_12 * exact_cost
    )
    assert dist <= grad_norm / mu_strong + 4.0 * eps_64 * th_star_norm
    rep_grad_l2 = math.sqrt(3.0) * float(sol.metrics["grad_inf_norm"])
    assert dist <= (rep_grad_l2 + 12.0 * u_round * th_star_norm) / mu_strong

    jac_c, jac_s = jax.jacrev(
        lambda c_in, s_in, m=method: levenberg_marquardt(
            residual_fn, theta0, data=(c_in, s_in), method=m
        ).fields["params"],
        argnums=(0, 1),
    )(c_vec, s_zero)

    exact_jac_c = jnp.eye(3, dtype=dtype) / jnp.asarray(1.1, dtype=dtype)
    exact_jac_s = jnp.diag(1.0 / (2.2 * (a_vec + b_vec / 5.0)))
    bwd_ceil = _tangent_bwd_contract_ceil(3, dtype, linear_solver)
    rel_ift = inv_h_norm * lip_h * dist + cond_h * math.sqrt(3.0) * bwd_ceil
    for jac_num, jac_ref in ((jac_c, exact_jac_c), (jac_s, exact_jac_s)):
        jac_ref_np = np.asarray(jac_ref, dtype=np.float64)
        jac_num_np = np.asarray(jac_num, dtype=np.float64)
        ref_norm = float(np.linalg.norm(jac_ref_np))
        err_norm = float(np.linalg.norm(jac_num_np - jac_ref_np))
        assert err_norm <= rel_ift * ref_norm + 4.0 * u_round * ref_norm


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("linear_solver", ["qr", "cholesky", "lsmr"])
def test_nonzero_residual_exact_1_over_1_1_minimiser_and_full_hessian_ift(
    dtype: jnp.dtype,
    linear_solver: LinearSolverKind,
) -> None:
    r"""Verify nonzero-residual nonlinear least squares against exact closed-form solution.

    Consider residual ``r(theta, c, s)`` in ``R^{2d}`` with ``u = 1.1 * theta - c``:
      ``r_top = sqrt(2 a) * u - s / sqrt(2 a)``,
      ``r_bot = sqrt(2) * (0.1 * u^2 + b)``  (with ``a_i, b_i > 0``).
    At ``s = 0``, ``0.5 * ||r||_2^2 = sum_i a_i u_i^2 + sum_i (0.1 u_i^2 + b_i)^2`` is
    strictly strongly convex with global modulus ``mu = 2 * 1.1^2 * min_i(a_i + b_i / 5)``,
    unique minimiser ``theta*(c, 0) = c / 1.1`` with strictly nonzero residual cost
    ``sum_i b_i^2``, exact ``d(theta*)/dc = (1 / 1.1) I``, and exact full-Hessian IFT
    sensitivity ``d(theta*)/ds|_{s=0} = diag(1 / (2.2 * (a_i + b_i / 5)))`` (which differs
    from the Gauss-Newton ``J^T J`` inverse ``diag(1 / (2.2 * a_i))`` whenever ``b_i > 0``).
    """
    from opifex.optimization.second_order.config import GaussNewtonConfig
    from opifex.optimization.second_order.wrappers import (
        create_gauss_newton_solver,
        create_levenberg_marquardt_solver,
    )

    enable_x64 = dtype == jnp.float64
    with jax.enable_x64(enable_x64):
        a_vec = jnp.array([1.0, 2.0, 1.5], dtype=dtype)
        b_vec = jnp.array([2.5, 1.5, 2.0], dtype=dtype)
        c_vec = jnp.array([1.3, -2.1, 0.8], dtype=dtype)
        s_zero = jnp.zeros(3, dtype=dtype)
        theta0 = jnp.zeros(3, dtype=dtype)
        tol = 1e-12 if enable_x64 else 1e-5

        def residual_fn(theta: jax.Array, data: tuple[jax.Array, jax.Array]) -> jax.Array:
            c_in, s_in = data
            u = jnp.asarray(1.1, dtype=dtype) * theta - c_in
            sqrt_2a = jnp.sqrt(2.0 * a_vec)
            r_top = sqrt_2a * u - s_in / sqrt_2a
            r_bot = jnp.sqrt(jnp.asarray(2.0, dtype=dtype)) * (
                jnp.asarray(0.1, dtype=dtype) * (u * u) + b_vec
            )
            return jnp.concatenate([r_top, r_bot])

        methods = [
            LevenbergMarquardt(
                max_iterations=80,
                gtol=tol,
                xtol=tol,
                ftol=tol,
                linear_solver=linear_solver,
                lsmr_iterations=48,
            )
        ]
        if linear_solver == "qr":
            gn_cfg = GaussNewtonConfig(max_iterations=80, rtol=tol, atol=tol)
            methods.append(create_gauss_newton_solver(gn_cfg))
            methods.append(create_levenberg_marquardt_solver(gn_cfg))

        for method in methods:
            _verify_1_over_1_1_method(
                method, residual_fn, theta0, a_vec, b_vec, c_vec, s_zero, dtype, linear_solver
            )


@pytest.mark.parametrize("linear_solver", ["qr", "cholesky", "lsmr"])
def test_nonzero_residual_implicit_diff_taylor_test_and_central_diff(
    linear_solver: LinearSolverKind,
) -> None:
    """Taylor remainder must converge at O(eps^2) (rate >= 1.9) on noisy nonlinear fits."""
    with jax.enable_x64(True):
        log_conc = jnp.linspace(-8.5, -4.5, 10, dtype=jnp.float64)
        true_params = jnp.array([8.0, 92.0, -6.4, 1.25], dtype=jnp.float64)
        perturbation = jnp.array([1.8, -2.1, 1.4, -1.9, 2.2, -1.5, 1.1, -2.0, 1.7, -1.3])
        y_obs = _hill_4pl(true_params, log_conc) + perturbation
        weights = jnp.linspace(0.7, 1.4, 10, dtype=jnp.float64)
        init_params = jnp.array([5.0, 95.0, -6.2, 1.0], dtype=jnp.float64)
        probe = jnp.array([0.3, -0.2, 1.4, 0.7], dtype=jnp.float64)

        method = LevenbergMarquardt(
            max_iterations=100,
            gtol=1e-13,
            xtol=1e-14,
            ftol=1e-14,
            linear_solver=linear_solver,
            lsmr_iterations=64,
            implicit_diff=True,
        )

        def scalar_loss(inputs: tuple[jax.Array, ...]) -> jax.Array:
            (y_in, w_in) = inputs
            sol = levenberg_marquardt(
                _hill_residual,
                init_params,
                data=(log_conc, y_in),
                weights=w_in,
                method=method,
            )
            return jnp.dot(sol.fields["params"], probe)

        _, rates = taylor_remainder_rates(scalar_loss, (y_obs, weights), seed=13)
        assert min(rates) >= TAYLOR_RATE_FLOOR

        rel_gap = gradient_versus_central_difference(
            scalar_loss,
            (y_obs, weights),
            step=1e-4,
            seed=19,
        )
        assert rel_gap < 1e-8
        check_grads(
            lambda y_in: scalar_loss((y_in, weights)), (y_obs,), order=1, modes=("fwd", "rev")
        )


def test_float32_value_and_gradient_accuracy_against_float64_oracle() -> None:
    """float32 primal fit and implicit gradient agree with float64 oracle within cond*eps32."""
    log_conc_64 = np.linspace(-8.5, -4.5, 12, dtype=np.float64)
    y_64 = (
        10.0
        + 80.0 / (1.0 + 10.0 ** ((-6.5 - log_conc_64) * 1.2))
        + np.array([0.8, -0.9, 0.5, -0.7, 1.1, -0.6, 0.4, -0.8, 0.9, -0.5, 0.6, -0.7])
    )
    init_64 = np.array([5.0, 90.0, -6.2, 1.0], dtype=np.float64)
    probe_64 = np.array([0.2, -0.1, 1.5, 0.8], dtype=np.float64)

    with jax.enable_x64(True):
        method_64 = LevenbergMarquardt(max_iterations=60, gtol=1e-12, xtol=1e-12, ftol=1e-12)
        sol_64 = levenberg_marquardt(
            _hill_residual,
            jnp.asarray(init_64),
            data=(jnp.asarray(log_conc_64), jnp.asarray(y_64)),
            method=method_64,
        )
        grad_64 = jax.grad(
            lambda y_in: jnp.dot(
                levenberg_marquardt(
                    _hill_residual,
                    jnp.asarray(init_64),
                    data=(jnp.asarray(log_conc_64), y_in),
                    method=method_64,
                ).fields["params"],
                jnp.asarray(probe_64),
            )
        )(jnp.asarray(y_64))
        hess_64 = jax.hessian(
            lambda th: (
                0.5
                * jnp.sum(_hill_residual(th, (jnp.asarray(log_conc_64), jnp.asarray(y_64))) ** 2)
            )
        )(sol_64.fields["params"])
        svals_h = np.linalg.svd(np.asarray(hess_64), compute_uv=False)
        sigma_min_h = float(svals_h[-1])
        cond_h = float(svals_h[0] / svals_h[-1])

    with jax.enable_x64(False):
        log_conc_32 = jnp.asarray(log_conc_64, dtype=jnp.float32)
        y_32 = jnp.asarray(y_64, dtype=jnp.float32)
        init_32 = jnp.asarray(init_64, dtype=jnp.float32)
        probe_32 = jnp.asarray(probe_64, dtype=jnp.float32)
        method_32 = LevenbergMarquardt(max_iterations=50, gtol=1e-5, xtol=1e-6, ftol=1e-6)

        sol_32 = levenberg_marquardt(
            _hill_residual,
            init_32,
            data=(log_conc_32, y_32),
            method=method_32,
        )
        grad_32 = jax.grad(
            lambda y_in: jnp.dot(
                levenberg_marquardt(
                    _hill_residual,
                    init_32,
                    data=(log_conc_32, y_in),
                    method=method_32,
                ).fields["params"],
                probe_32,
            )
        )(y_32)

    with jax.enable_x64(True):
        grad_at_32_64 = jax.grad(
            lambda th: (
                0.5
                * jnp.sum(_hill_residual(th, (jnp.asarray(log_conc_64), jnp.asarray(y_64))) ** 2)
            )
        )(sol_32.fields["params"].astype(jnp.float64))
    val_err = float(
        np.linalg.norm(np.asarray(sol_32.fields["params"]) - np.asarray(sol_64.fields["params"]))
    )
    grad_rel = float(
        np.linalg.norm(np.asarray(grad_32) - np.asarray(grad_64))
        / np.linalg.norm(np.asarray(grad_64))
    )
    assert bool(sol_32.is_converged)
    # Local strong-convexity bound (factor 2 accounts for cubic remainder near sol_64):
    assert val_err <= 2.0 * float(np.linalg.norm(np.asarray(grad_at_32_64))) / sigma_min_h
    bwd_32 = _tangent_bwd_contract_ceil(4, jnp.dtype(jnp.float32), "qr")
    assert grad_rel <= cond_h * (
        math.sqrt(4.0) * bwd_32
        + val_err / float(np.linalg.norm(np.asarray(sol_64.fields["params"])))
    )


def test_singular_tangent_consistency_and_both_loop_modes_ad() -> None:
    """Singular Hessian r = th0 + th1 - y returns [nan, nan] tangents across LM, Gauss-Newton,
    and BFGSSolver, while well-posed fits differentiate in jvp/jacfwd/grad for both implicit_diff modes."""
    from opifex.optimization.second_order.wrappers import (
        BFGSSolver,
        create_gauss_newton_solver,
        create_levenberg_marquardt_solver,
    )

    y_val = jnp.float32(2.0)
    th_init = jnp.array([0.5, 0.5], dtype=jnp.float32)

    for method in (
        LevenbergMarquardt(linear_solver="qr"),
        LevenbergMarquardt(linear_solver="cholesky"),
        LevenbergMarquardt(linear_solver="lsmr"),
        create_gauss_newton_solver(),
        create_levenberg_marquardt_solver(),
    ):
        jac_y = jax.jacrev(
            lambda y, m=method: levenberg_marquardt(
                lambda th, rhs: jnp.atleast_1d(th[0] + th[1] - rhs),
                th_init,
                data=y,
                method=m,
            ).fields["params"]
        )(y_val)
        assert jac_y.shape == (2,)
        assert bool(jnp.all(jnp.isnan(jac_y)))

    bfgs_jac_y = jax.jacrev(
        lambda y: (
            BFGSSolver()
            .minimise(lambda th, rhs: 0.5 * (th[0] + th[1] - rhs) ** 2, th_init, args=y)
            .fields["params"]
        )
    )(y_val)
    assert bool(jnp.all(jnp.isnan(bfgs_jac_y)))

    # Early exit + forward/reverse AD for both implicit_diff=True and implicit_diff=False:
    with jax.enable_x64(True):
        target = jnp.array([1.5, -0.8], dtype=jnp.float64)
        init_p = jnp.zeros(2, dtype=jnp.float64)
        for imp in (True, False):
            m = LevenbergMarquardt(
                max_iterations=40, gtol=1e-12, xtol=1e-12, ftol=1e-12, implicit_diff=imp
            )
            sol = levenberg_marquardt(lambda th, t: 2.0 * th - t, init_p, data=target, method=m)
            assert int(sol.status) == int(Status.SUCCESS)
            assert int(sol.stats["num_iterations"]) <= 5

            def fit_fn(t_in: jax.Array, method_i=m) -> jax.Array:
                return levenberg_marquardt(
                    lambda th, t: 2.0 * th - t, init_p, data=t_in, method=method_i
                ).fields["params"]

            assert jnp.allclose(jax.jacfwd(fit_fn)(target), 0.5 * jnp.eye(2), atol=1e-10)
            assert jnp.allclose(jax.jacrev(fit_fn)(target), 0.5 * jnp.eye(2), atol=1e-10)


@pytest.mark.parametrize("implicit_diff", [True, False])
def test_stationary_initial_guess_and_dict_pytree_params(implicit_diff: bool) -> None:
    """Zero-residual start converges in 0 steps with finite gradients and dict pytree support."""
    target = {"offset": jnp.array(1.5), "slope": jnp.array([-0.5, 2.0])}
    method = LevenbergMarquardt(max_iterations=15, gtol=1e-6, implicit_diff=implicit_diff)

    def solve_loss(tgt_slope: jax.Array) -> jax.Array:
        sol = levenberg_marquardt(
            lambda p, s_t: jnp.concatenate([jnp.atleast_1d(p["offset"] - 1.5), p["slope"] - s_t]),
            {"offset": jnp.array(1.5), "slope": tgt_slope},
            data=tgt_slope,
            method=method,
        )
        return jnp.sum(sol.fields["params"]["slope"] ** 2)

    sol0 = levenberg_marquardt(
        lambda p: jnp.concatenate(
            [jnp.atleast_1d(p["offset"] - 1.5), p["slope"] - target["slope"]]
        ),
        target,
        method=method,
    )
    assert int(sol0.status) == int(Status.SUCCESS)
    assert int(sol0.stats["num_iterations"]) == 0

    grad_at_zero = jax.grad(solve_loss)(jnp.zeros((2,)))
    assert bool(jnp.all(jnp.isfinite(grad_at_zero)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_jit_and_vmap_gradient_ulp_agreement(dtype: jnp.dtype) -> None:
    """jit(grad) vs eager grad and vmap(grad) vs loop grad agree within ULP budgets."""
    enable_x64 = dtype == jnp.float64
    with jax.enable_x64(enable_x64):
        log_conc = jnp.linspace(-8.5, -4.5, 10, dtype=dtype)
        y_obs = _hill_4pl(jnp.array([8.0, 88.0, -6.4, 1.2], dtype=dtype), log_conc)
        init_params = jnp.array([4.0, 92.0, -6.1, 1.0], dtype=dtype)
        tol = 1e-11 if enable_x64 else 1e-5
        method = LevenbergMarquardt(max_iterations=45, gtol=tol, xtol=tol, ftol=tol)

        def loss(components: tuple[jax.Array, ...]) -> jax.Array:
            (y_in,) = components
            sol = levenberg_marquardt(
                _hill_residual,
                init_params,
                data=(log_conc, y_in),
                method=method,
            )
            return jnp.sum(sol.fields["params"] ** 2)

        jit_ulp, vmap_ulp = gradient_transform_agreement(loss, (y_obs,))
        assert jit_ulp <= JIT_GRADIENT_ULP
        # On CUDA, batched cuSOLVER geqrf/syevd under vmap uses distinct kernels from
        # single-matrix cuSOLVER (~14.5 ULP Householder reflection reassociation in float64).
        assert vmap_ulp <= JIT_GRADIENT_ULP


def test_nnx_module_bilevel_training_through_levenberg_marquardt() -> None:
    """An nnx.Module trains cleanly through LevenbergMarquardt under @nnx.jit with 1 compile."""
    log_conc = jnp.linspace(-8.5, -4.5, 10, dtype=jnp.float32)
    true_params = jnp.array([10.0, 90.0, -6.5, 1.3], dtype=jnp.float32)
    clean_y = _hill_4pl(true_params, log_conc)
    # Corrupt endpoint observations so learned weights can downweight outliers and reduce error.
    corrupted_y = clean_y.at[0].add(8.0).at[-1].add(-8.0)
    init_params = jnp.array([8.0, 88.0, -6.3, 1.1], dtype=jnp.float32)
    lm = LevenbergMarquardt(max_iterations=40, gtol=1e-5, xtol=1e-5, ftol=1e-5)

    model = _ObservationWeightNet(rngs=nnx.Rngs(0))
    lm_64 = LevenbergMarquardt(max_iterations=80, gtol=1e-13, xtol=1e-13, ftol=1e-13)
    check_parameter_gradients(
        model,
        lambda m: jnp.mean(
            (
                levenberg_marquardt(
                    _hill_residual,
                    init_params,
                    data=(log_conc, corrupted_y),
                    weights=m(log_conc),
                    method=lm_64,
                ).fields["params"]
                - true_params
            )
            ** 2
        ),
    )
    check_input_gradients(
        model,
        lambda m, y_in: jnp.mean(
            (
                levenberg_marquardt(
                    _hill_residual,
                    init_params,
                    data=(log_conc, y_in),
                    weights=m(log_conc),
                    method=lm_64,
                ).fields["params"]
                - true_params
            )
            ** 2
        ),
        corrupted_y,
    )
    optimizer = nnx.Optimizer(model, optax.adam(3e-2), wrt=nnx.Param)

    @nnx.jit
    def train_step(net: _ObservationWeightNet, opt: nnx.Optimizer) -> jax.Array:
        def outer_loss(m: _ObservationWeightNet) -> jax.Array:
            w = m(log_conc)
            sol = levenberg_marquardt(
                _hill_residual,
                init_params,
                data=(log_conc, corrupted_y),
                weights=w,
                method=lm,
            )
            return jnp.mean((sol.fields["params"] - true_params) ** 2)

        loss_val, grads = nnx.value_and_grad(outer_loss)(net)
        opt.update(net, grads)
        return loss_val

    with expect_compiles(1):
        first_loss = train_step(model, optimizer)
    with expect_compiles(0):
        losses = [train_step(model, optimizer) for _ in range(12)]

    assert bool(jnp.isfinite(first_loss))
    assert float(losses[-1]) < float(first_loss)


def test_batched_statuses_report_and_solution_distribution() -> None:
    """Non-finite inputs isolate per-element Status codes; Solutions aggregate cleanly."""
    log_conc = jnp.linspace(-8.0, -5.0, 8, dtype=jnp.float32)
    good_y = _hill_4pl(jnp.array([10.0, 90.0, -6.5, 1.2], dtype=jnp.float32), log_conc)
    nan_y = good_y.at[3].set(jnp.nan)
    y_batch = jnp.stack([good_y, nan_y], axis=0)
    init_params = jnp.array([5.0, 95.0, -6.2, 1.0], dtype=jnp.float32)
    method = LevenbergMarquardt(max_iterations=35, gtol=1e-5, xtol=1e-5, ftol=1e-5)

    batched = jax.jit(
        jax.vmap(
            lambda y_i: levenberg_marquardt(
                _hill_residual, init_params, data=(log_conc, y_i), method=method
            )
        )
    )(y_batch)
    assert bool(is_successful(batched.status[0]))
    assert int(batched.status[1]) == int(Status.NONFINITE)

    sol_a = levenberg_marquardt(_hill_residual, init_params, data=(log_conc, good_y), method=method)
    sol_b = levenberg_marquardt(
        _hill_residual,
        init_params,
        data=(log_conc, good_y + 0.2),
        method=method,
    )
    rep = timed_solve(
        lambda: levenberg_marquardt(
            _hill_residual, init_params, data=(log_conc, good_y), method=method
        ),
        warmup=1,
        iterations=2,
    )
    assert rep.converged
    assert rep.execution_time >= 0.0

    agg = aggregate_solver_solutions([sol_a, sol_b])
    assert agg.fields["params"].shape == (4,)
    assert agg.auxiliary_data["uq"]["variance"]["params"].shape == (4,)


def test_vmap_10000_hill_curves_single_compile_and_profile() -> None:
    """10,000 synthetic 4PL Hill curves compile once under jit(vmap) and profile cleanly."""
    num_curves = 10_000
    key = jax.random.key(42)
    k_b, k_t, k_e, k_h, k_n = jax.random.split(key, 5)

    log_conc = jnp.linspace(-9.0, -4.0, 8, dtype=jnp.float32)
    bottoms = jax.random.uniform(k_b, (num_curves,), minval=2.0, maxval=15.0, dtype=jnp.float32)
    tops = jax.random.uniform(k_t, (num_curves,), minval=80.0, maxval=105.0, dtype=jnp.float32)
    ec50s = jax.random.uniform(k_e, (num_curves,), minval=-7.5, maxval=-5.5, dtype=jnp.float32)
    hills = jax.random.uniform(k_h, (num_curves,), minval=0.8, maxval=1.8, dtype=jnp.float32)
    true_batch = jnp.stack([bottoms, tops, ec50s, hills], axis=-1)

    clean = jax.vmap(_hill_4pl, in_axes=(0, None))(true_batch, log_conc)
    y_batch_a = clean + 0.25 * jax.random.normal(k_n, clean.shape, dtype=jnp.float32)
    y_batch_b = y_batch_a + 0.05
    init_guess = jnp.array([8.0, 92.0, -6.5, 1.2], dtype=jnp.float32)

    method = LevenbergMarquardt(
        max_iterations=25,
        gtol=1e-4,
        xtol=1e-5,
        ftol=1e-5,
        linear_solver="cholesky",
    )

    @jax.jit
    def fit_batch(y_all: jax.Array):
        return jax.vmap(
            lambda y_one: levenberg_marquardt(
                _hill_residual,
                init_guess,
                data=(log_conc, y_one),
                method=method,
            )
        )(y_all)

    with expect_compiles(1):
        sols_a = fit_batch(y_batch_a)
    with expect_compiles(0):
        sols_b = fit_batch(y_batch_b)

    conv_fraction = float(jnp.mean(sols_a.is_converged.astype(jnp.float32)))
    assert conv_fraction >= 0.995
    assert sols_b.fields["params"].shape == (num_curves, 4)

    flops = FlopsCounter().count(fit_batch, y_batch_a)
    timing = time_calls(lambda: fit_batch(y_batch_a), warmup=1, iterations=2)
    assert flops.total_flops > 0
    assert timing.median_sec > 0.0
