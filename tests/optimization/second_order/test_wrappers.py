"""Tests for pure-JAX second-order optimization wrappers."""

import ast
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest
from substrax.testing import TraceCounter
from tests.fields.staggered_support import gradient_versus_central_difference

import opifex.optimization.second_order.wrappers as wrappers_mod
from opifex.core.solver import lsmr_tangent_solve
from opifex.core.solver.problem import solve
from opifex.core.solver.status import Status
from opifex.optimization.second_order.config import (
    GaussNewtonConfig,
    LBFGSConfig,
    LinesearchType,
)
from opifex.optimization.second_order.wrappers import (
    BFGSSolver,
    create_bfgs_solver,
    create_gauss_newton_solver,
    create_lbfgs_optimizer,
    create_levenberg_marquardt_solver,
    OptimizationProblem,
)
from opifex.solvers.nonlinear_least_squares import (
    LevenbergMarquardt,
    NonlinearLeastSquaresProblem,
)


class TestCreateLBFGSOptimizer:
    """Test L-BFGS optimizer creation."""

    def test_creates_valid_optimizer(self):
        """Should create a valid optax optimizer."""
        optimizer = create_lbfgs_optimizer()
        assert optimizer is not None
        # Should be a GradientTransformation
        assert hasattr(optimizer, "init")
        assert hasattr(optimizer, "update")

    def test_default_config_creates_optimizer(self):
        """Default config should work."""
        optimizer = create_lbfgs_optimizer(None)
        assert optimizer is not None

    def test_custom_config_applied(self):
        """Custom config should be applied."""
        config = LBFGSConfig(memory_size=5, linesearch=LinesearchType.BACKTRACKING)
        optimizer = create_lbfgs_optimizer(config)
        assert optimizer is not None

    def test_optimizer_can_be_initialized(self):
        """Optimizer should initialize with parameters."""
        optimizer = create_lbfgs_optimizer()
        params = {"w": jnp.ones(10), "b": jnp.zeros(5)}
        state = optimizer.init(params)
        assert state is not None

    def test_optimizer_can_update(self):
        """Optimizer should compute updates with value_and_grad."""
        optimizer = create_lbfgs_optimizer(LBFGSConfig(memory_size=5))
        params = jnp.array([1.0, 2.0, 3.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        # L-BFGS requires value and grad from state
        value, grad = optax.value_and_grad_from_state(loss_fn)(params, state=state)
        updates, _new_state = optimizer.update(
            grad,
            state,
            params,
            value=value,  # pyright: ignore[reportCallIssue]
            grad=grad,  # pyright: ignore[reportCallIssue]
            value_fn=loss_fn,  # pyright: ignore[reportCallIssue]
        )
        new_params = optax.apply_updates(params, updates)

        # Should make progress toward minimum
        assert loss_fn(new_params) <= loss_fn(params)

    def test_convergence_on_quadratic(self):
        """L-BFGS should converge on a quadratic function."""
        optimizer = create_lbfgs_optimizer(LBFGSConfig(memory_size=10, max_iterations=50))
        params = jnp.array([5.0, -3.0, 2.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return 0.5 * jnp.sum(p**2)

        # Run several steps
        for _ in range(20):
            value, grad = optax.value_and_grad_from_state(loss_fn)(params, state=state)
            updates, state = optimizer.update(
                grad,
                state,
                params,
                value=value,  # pyright: ignore[reportCallIssue]
                grad=grad,  # pyright: ignore[reportCallIssue]
                value_fn=loss_fn,  # pyright: ignore[reportCallIssue]
            )
            params = optax.apply_updates(params, updates)

        # Should be very close to zero
        assert jnp.allclose(params, 0.0, atol=1e-3)  # type: ignore # noqa: PGH003


class TestCreateGaussNewtonSolver:
    """Test pure-JAX Gauss-Newton solver creation."""

    def test_creates_valid_solver(self):
        """Should create a pure-JAX LevenbergMarquardt solver configured as Gauss-Newton."""
        solver = create_gauss_newton_solver()
        assert solver is not None
        assert isinstance(solver, LevenbergMarquardt)
        assert solver.scaling == "identity"

    def test_default_config_creates_solver(self):
        """Default config should work."""
        solver = create_gauss_newton_solver(None)
        assert solver is not None

    def test_custom_config_applied(self):
        """Custom config should be applied."""
        config = GaussNewtonConfig(rtol=1e-8, atol=1e-8)
        solver = create_gauss_newton_solver(config)
        assert solver.gtol == 1e-8
        assert solver.xtol == 1e-8


class TestCreateLevenbergMarquardtSolver:
    """Test pure-JAX Levenberg-Marquardt solver creation."""

    def test_creates_valid_solver(self):
        """Should create a pure-JAX LevenbergMarquardt solver with Marquardt scaling."""
        solver = create_levenberg_marquardt_solver()
        assert solver is not None
        assert isinstance(solver, LevenbergMarquardt)
        assert solver.scaling == "marquardt"

    def test_can_solve_least_squares(self):
        """Should solve a simple least-squares problem."""
        solver = create_levenberg_marquardt_solver()

        # Simple linear regression problem: find x such that Ax = b
        a_mat = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b_vec = jnp.array([3.0, 7.0, 11.0])

        def residual_fn(x, _args):
            return a_mat @ x - b_vec

        x0 = jnp.zeros(2)
        problem = NonlinearLeastSquaresProblem(residual_fn=residual_fn, initial_params=x0)
        sol = solve(problem, solver)

        # Should find approximately x = [1, 1]
        assert int(sol.status) == int(Status.SUCCESS)
        assert jnp.allclose(sol.fields["params"], jnp.array([1.0, 1.0]), atol=1e-4)


class TestCreateBFGSSolver:
    """Test pure-JAX BFGS solver creation and implicit differentiation."""

    def test_creates_valid_solver(self):
        """Should create a pure-JAX BFGSSolver."""
        solver = create_bfgs_solver()
        assert solver is not None
        assert isinstance(solver, BFGSSolver)

    def test_validation_rejects_invalid_hyperparameters(self):
        """Invalid BFGS iteration budgets or tolerances raise ValueError."""
        prob = OptimizationProblem(
            objective_fn=lambda x: jnp.sum(x**2),
            initial_params=jnp.ones(2),
        )
        with pytest.raises(ValueError, match="max_iterations"):
            solve(prob, BFGSSolver(max_iterations=0))
        with pytest.raises(ValueError, match="strictly positive"):
            solve(prob, BFGSSolver(rtol=0.0))

    def test_can_minimize_quadratic_and_rosenbrock(self):
        """Should minimize quadratic and non-convex Rosenbrock functions."""
        solver = create_bfgs_solver(GaussNewtonConfig(max_iterations=150, rtol=1e-7, atol=1e-7))

        def quad_fn(x, _args):
            return 0.5 * jnp.sum(x**2)

        x0 = jnp.array([5.0, -3.0, 2.0])
        sol = solver.minimise(quad_fn, x0, args=None)
        assert int(sol.status) == int(Status.SUCCESS)
        assert jnp.allclose(sol.fields["params"], jnp.zeros(3), atol=1e-4)

        with jax.enable_x64(True):
            rosen = solver.minimise(
                lambda u: (1.0 - u[0]) ** 2 + 100.0 * (u[1] - u[0] ** 2) ** 2,
                jnp.array([-1.2, 1.0], dtype=jnp.float64),
            )
            assert int(rosen.status) == int(Status.SUCCESS)
            assert jnp.allclose(
                rosen.fields["params"], jnp.array([1.0, 1.0], dtype=jnp.float64), atol=1e-6
            )

    def test_implicit_diff_matches_central_finite_difference_and_import_isolation(self):
        """BFGSSolver imports only declared JAX/Flax/Optax/Opifex foundations and matches FD."""
        source = Path(wrappers_mod.__file__).read_text(encoding="utf-8")
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

        with jax.enable_x64(True):
            solver = BFGSSolver(max_iterations=120, rtol=1e-11, atol=1e-11)
            q_mat = jnp.array([[3.0, 0.8], [0.8, 2.0]], dtype=jnp.float64)
            weights = jnp.array([1.3, -0.7], dtype=jnp.float64)

            def bilevel_loss(inputs: tuple[jax.Array, ...]) -> jax.Array:
                (target,) = inputs
                sol = solver.minimise(
                    lambda x, tgt: 0.5 * jnp.dot(x - tgt, q_mat @ (x - tgt)) + 0.05 * jnp.sum(x**4),
                    jnp.zeros(2, dtype=jnp.float64),
                    args=target,
                )
                return jnp.dot(sol.fields["params"], weights)

            rel_gap = gradient_versus_central_difference(
                bilevel_loss,
                (jnp.array([0.9, -0.4], dtype=jnp.float64),),
                step=1e-5,
                seed=19,
            )
            assert rel_gap < 1e-8

    def test_varargs_objective_stall_status_and_unconverged_ift_guard(self):
        """*args/defaulted objectives work with args=None, stalled runs never report SUCCESS,
        and unconverged implicit_diff runs poison forward/reverse tangents to NaN."""
        solver = BFGSSolver(max_iterations=80, rtol=1e-6, atol=1e-6)
        sol_varargs = solver.minimise(
            lambda *xs: 0.5 * jnp.sum((xs[0] - 2.0) ** 2),
            jnp.zeros(2, dtype=jnp.float32),
        )
        assert int(sol_varargs.status) == int(Status.SUCCESS)
        assert jnp.allclose(sol_varargs.fields["params"], jnp.full(2, 2.0), atol=1e-4)

        sol_default = solver.minimise(
            lambda x, scale=2.0: scale * jnp.sum((x - 1.5) ** 2),
            jnp.zeros(2, dtype=jnp.float32),
        )
        assert int(sol_default.status) == int(Status.SUCCESS)
        assert jnp.allclose(sol_default.fields["params"], jnp.full(2, 1.5), atol=1e-4)

        # Starved 1-step solve on Rosenbrock: primal finite, status MAX_ITERS, tangents NaN
        starved_solver = BFGSSolver(max_iterations=1, rtol=1e-9, atol=1e-9, implicit_diff=True)
        x_init = jnp.array([-1.2, 1.0], dtype=jnp.float32)
        t_val = jnp.float32(1.0)

        primal_sol = starved_solver.minimise(
            lambda u, th: (th - u[0]) ** 2 + 50.0 * (u[1] - u[0] ** 2) ** 2,
            x_init,
            args=t_val,
        )
        assert int(primal_sol.status) != int(Status.SUCCESS)
        assert bool(jnp.isfinite(primal_sol.fields["params"]).all())

        def starved_sum(theta: jax.Array) -> jax.Array:
            return jnp.sum(
                starved_solver.minimise(
                    lambda u, th: (th - u[0]) ** 2 + 50.0 * (u[1] - u[0] ** 2) ** 2,
                    x_init,
                    args=theta,
                ).fields["params"]
            )

        assert bool(jnp.isnan(jax.grad(starved_sum)(t_val)))
        assert bool(jnp.isnan(jax.jacfwd(starved_sum)(t_val)))

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("scale", [1.0, 10.0, 100.0])
    def test_cauchy_stopping_1_over_1_1_quartic_strong_convexity_and_exact_ift(
        self,
        dtype: jnp.dtype,
        scale: float,
    ) -> None:
        """BFGS Cauchy stopping converges across scales 1..100 on the 1/1.1 quartic, obeys
        the strong-convexity a-posteriori bound ||x - x*||_2 <= ||grad f(x)||_2 / mu
        (mu = 2 * 1.1^2 * min(A) = 2.42), and recovers exact dx*/dc = (1 / 1.1) I."""
        enable_x64 = dtype == jnp.float64
        with jax.enable_x64(enable_x64):
            a_weights = jnp.array([1.0, 2.5, 4.0, 1.5], dtype=dtype)
            c_vec = jnp.array([1.3, -2.1, 0.7, -3.4], dtype=dtype) * jnp.asarray(scale, dtype=dtype)
            x0 = jnp.zeros_like(c_vec)
            tol = 1e-10 if enable_x64 else 1e-5
            u_round = 0.5 * float(jnp.finfo(dtype).eps)

            # Strong convexity modulus: f''_ii(x) = 2 * 1.1^2 * A_i + 0.12 * 1.1^2 * u_i^2
            # >= 2 * 1.1^2 * min(A) = 2.42 everywhere on R^4.
            mu_strong = 2.0 * (1.1**2) * float(jnp.min(a_weights))
            assert mu_strong == pytest.approx(2.42)

            def quartic(x: jax.Array, c: jax.Array) -> jax.Array:
                u = jnp.asarray(1.1, dtype=dtype) * x - c
                return jnp.sum(a_weights * (u * u)) + jnp.asarray(0.01, dtype=dtype) * jnp.sum(
                    (u * u) ** 2
                )

            for solver in (
                BFGSSolver(max_iterations=200, rtol=tol, atol=tol),
                BFGSSolver(
                    max_iterations=200,
                    rtol=tol,
                    atol=tol,
                    tangent_solver=lsmr_tangent_solve,
                ),
            ):
                sol = solver.minimise(quartic, x0, args=c_vec)
                assert int(sol.status) == int(Status.SUCCESS)
                assert int(sol.stats["num_iterations"]) >= 1

                x_hat = sol.fields["params"]
                # Evaluate exact x* and exact gradient at x_hat in float64 to separate solver
                # truncation from evaluation roundoff when checking strong convexity.
                with jax.enable_x64(True):
                    c_64 = c_vec.astype(jnp.float64)
                    a_64 = a_weights.astype(jnp.float64)
                    x_hat_64 = x_hat.astype(jnp.float64)
                    x_star_64 = c_64 / 1.1
                    u_64 = 1.1 * x_hat_64 - c_64
                    grad_64 = 1.1 * (2.0 * a_64 * u_64 + 0.04 * (u_64**3))
                    dist = float(jnp.linalg.norm(x_hat_64 - x_star_64))
                    grad_norm = float(jnp.linalg.norm(grad_64))
                    hess_diag_64 = (1.1**2) * (2.0 * a_64 + 0.12 * (u_64**2))
                    cond_h = float(jnp.max(hess_diag_64) / jnp.min(hess_diag_64))
                    x_star_norm = float(jnp.linalg.norm(x_star_64))

                assert (
                    dist
                    <= grad_norm / mu_strong + 4.0 * float(jnp.finfo(jnp.float64).eps) * x_star_norm
                )
                # Reported metrics["grad_inf_norm"] bounds dist directly by strong convexity:
                rep_grad_l2 = math.sqrt(4.0) * float(sol.metrics["grad_inf_norm"])
                assert dist <= (rep_grad_l2 + 12.0 * u_round * x_star_norm) / mu_strong

                jac_rev = jax.jacrev(
                    lambda c, s=solver: s.minimise(quartic, x0, args=c).fields["params"]
                )(c_vec)
                jac_fwd = jax.jacfwd(
                    lambda c, s=solver: s.minimise(quartic, x0, args=c).fields["params"]
                )(c_vec)
                exact_jac = jnp.eye(4, dtype=dtype) / jnp.asarray(1.1, dtype=dtype)
                # At every point x, (d^2f/dx^2)^{-1} (-d^2f/dxdc) = (1 / 1.1) I identically,
                # so the only error is the tangent linear solver's backward-error contract
                # multiplied by cond_2(H(x_hat)) and sqrt(dim) = 2:
                gamma_12 = (12.0 * u_round) / (1.0 - 12.0 * u_round)
                lu_ceil = max(64.0 * u_round, 8.0 * math.sqrt(4.0) * gamma_12)
                bwd_ceil = (
                    max(100.0 * max(16.0 * u_round, 1e-12), lu_ceil)
                    if solver.tangent_solver is lsmr_tangent_solve
                    else lu_ceil
                )
                jac_bound = (cond_h * 2.0 * bwd_ceil + 2.0 * u_round) * (2.0 / 1.1)
                assert float(jnp.max(jnp.abs(jac_rev - exact_jac))) <= jac_bound
                assert float(jnp.max(jnp.abs(jac_fwd - exact_jac))) <= jac_bound


class TestJITCompatibility:
    """Test single-trace JIT and vmap compatibility of all wrappers."""

    def test_lbfgs_jit_compatible(self):
        """L-BFGS operations should be JIT-compatible."""
        optimizer = create_lbfgs_optimizer()
        params = jnp.array([1.0, 2.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        @jax.jit
        def step(params, state):
            value, grad = optax.value_and_grad_from_state(loss_fn)(params, state=state)
            updates, new_state = optimizer.update(
                grad,
                state,
                params,
                value=value,  # pyright: ignore[reportCallIssue]
                grad=grad,  # pyright: ignore[reportCallIssue]
                value_fn=loss_fn,  # pyright: ignore[reportCallIssue]
            )
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state

        new_params, _new_state = step(params, state)
        assert jnp.isfinite(new_params).all()

    def test_levenberg_marquardt_and_bfgs_single_trace_jit_vmap(self):
        """LM and BFGS solvers compile once under jit(vmap)."""
        lm_solver = create_levenberg_marquardt_solver()
        bfgs_solver = create_bfgs_solver()

        def lm_one(x0: jax.Array) -> jax.Array:
            prob = NonlinearLeastSquaresProblem(
                residual_fn=lambda x, _a: x**2 - 1.0,
                initial_params=x0,
            )
            return solve(prob, lm_solver).fields["params"]

        def bfgs_one(x0: jax.Array) -> jax.Array:
            return bfgs_solver.minimise(lambda x: 0.5 * jnp.sum((x - 1.0) ** 2), x0).fields[
                "params"
            ]

        counter_lm = TraceCounter()
        jitted_lm = jax.jit(counter_lm.wrap(jax.vmap(lm_one)))
        batch = jnp.array([[0.5], [1.5], [2.0]], dtype=jnp.float32)
        with counter_lm.expect(new_traces=1):
            res_lm = jitted_lm(batch)
        with counter_lm.expect(new_traces=0):
            _ = jitted_lm(batch + 0.2)
        assert jnp.allclose(res_lm, jnp.ones_like(res_lm), atol=1e-4)

        counter_bfgs = TraceCounter()
        jitted_bfgs = jax.jit(counter_bfgs.wrap(jax.vmap(bfgs_one)))
        with counter_bfgs.expect(new_traces=1):
            res_bfgs = jitted_bfgs(batch)
        with counter_bfgs.expect(new_traces=0):
            _ = jitted_bfgs(batch - 0.1)
        assert jnp.allclose(res_bfgs, jnp.ones_like(res_bfgs), atol=1e-4)

    def test_bfgs_nested_vmap_jvp_of_grad_and_nnx_module_bilevel_training(self) -> None:
        """BFGSSolver composes under nested vmap(vmap), jvp(grad), and @nnx.jit nnx.Module."""
        from flax import nnx

        solver = BFGSSolver(max_iterations=80, rtol=1e-7, atol=1e-7)

        def scalar_bilevel(c: jax.Array) -> jax.Array:
            x_star = solver.minimise(
                lambda x, tgt: jnp.sum((1.1 * x - tgt) ** 2) + 0.01 * jnp.sum((1.1 * x - tgt) ** 4),
                jnp.zeros_like(c),
                args=c,
            ).fields["params"]
            return 0.5 * jnp.sum(x_star * x_star)

        grid_c = jnp.array(
            [[[1.1, -2.2], [0.55, 1.65]], [[-1.1, 0.0], [2.2, -0.55]]],
            dtype=jnp.float32,
        )
        nested = jax.jit(jax.vmap(jax.vmap(scalar_bilevel)))(grid_c)
        expected_nested = 0.5 * jnp.sum((grid_c / 1.1) ** 2, axis=-1)
        assert nested.shape == (2, 2)
        assert jnp.allclose(nested, expected_nested, rtol=1e-4, atol=1e-5)

        with jax.enable_x64(True):
            solver_64 = BFGSSolver(max_iterations=100, rtol=1e-11, atol=1e-11)

            def bilevel_64(c: jax.Array) -> jax.Array:
                xs = solver_64.minimise(
                    lambda x, tgt: (
                        jnp.sum((1.1 * x - tgt) ** 2) + 0.01 * jnp.sum((1.1 * x - tgt) ** 4)
                    ),
                    jnp.zeros_like(c),
                    args=c,
                ).fields["params"]
                return 0.5 * jnp.sum(xs * xs)

            c0 = jnp.array([1.3, -2.1], dtype=jnp.float64)
            v_dir = jnp.array([0.4, -0.7], dtype=jnp.float64)
            grad_val, hvp_val = jax.jvp(jax.grad(bilevel_64), (c0,), (v_dir,))
            assert jnp.allclose(grad_val, c0 / (1.1**2), rtol=1e-9, atol=1e-9)
            assert jnp.allclose(hvp_val, v_dir / (1.1**2), rtol=1e-8, atol=1e-8)

        class _TargetShiftNet(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs) -> None:
                self.linear = nnx.Linear(2, 2, rngs=rngs)

            def __call__(self, z: jax.Array) -> jax.Array:
                return self.linear(z)

        net = _TargetShiftNet(rngs=nnx.Rngs(0))
        opt = nnx.Optimizer(net, optax.adam(5e-2), wrt=nnx.Param)
        z_in = jnp.array([1.0, -0.5], dtype=jnp.float32)
        desired_x = jnp.array([0.8, -0.4], dtype=jnp.float32)

        @nnx.jit
        def step_fn(model: _TargetShiftNet, optimizer: nnx.Optimizer) -> jax.Array:
            def outer(m: _TargetShiftNet) -> jax.Array:
                c_pred = m(z_in)
                xs = solver.minimise(
                    lambda x, tgt: jnp.sum((1.1 * x - tgt) ** 2),
                    jnp.zeros(2, dtype=jnp.float32),
                    args=c_pred,
                ).fields["params"]
                return jnp.mean((xs - desired_x) ** 2)

            loss, grads = nnx.value_and_grad(outer)(model)
            optimizer.update(model, grads)
            return loss

        first_loss = step_fn(net, opt)
        final_loss = first_loss
        for _ in range(10):
            final_loss = step_fn(net, opt)
        assert float(final_loss) < float(first_loss)
