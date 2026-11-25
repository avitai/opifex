"""Tests for the pure-JAX Anderson-accelerated fixed-point solver.

Validates convergence on a stiff scalar map, the exact implicit-function-theorem
gradient through :func:`anderson_fixed_point`, and robust convergence of the
water/LDA Kohn-Sham density where plain Roothaan iteration diverges.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from substrax.testing import TraceCounter

import opifex.neural.quantum.dft._fixed_point as fp_mod
from opifex.core.solver.status import Status
from opifex.neural.quantum.dft._fixed_point import (
    anderson_fixed_point,
    AndersonAcceleration,
)


def test_converges_scalar_fixed_point() -> None:
    """Recovers the cos fixed point ``x = cos(x)`` to high precision."""
    with jax.enable_x64(True):
        solver = AndersonAcceleration(rtol=1e-12, atol=1e-12, history_size=5)
        solution = anderson_fixed_point(
            lambda x, _: jnp.cos(x), solver, jnp.array(0.0), max_steps=64
        )
    assert float(solution.value) == pytest.approx(0.7390851332151607, abs=1e-9)


def test_converges_oscillatory_map() -> None:
    """Anderson stabilises an oscillatory map ``f(x) = sqrt(2) cos(x)``.

    The derivative at the fixed point has magnitude ``> 0`` and the iteration
    spirals; Anderson mixing damps the oscillation to convergence.
    """
    with jax.enable_x64(True):

        def stiff(x: jax.Array, _: None) -> jax.Array:
            return jnp.sqrt(2.0) * jnp.cos(x)

        solver = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=6)
        solution = anderson_fixed_point(stiff, solver, jnp.array(0.5), max_steps=64)
        residual = float(jnp.abs(stiff(solution.value, None) - solution.value))
    assert residual < 1e-9


def test_implicit_gradient_is_exact() -> None:
    r"""IFT gradient through Anderson matches the analytic ``dy*/dtheta``.

    For ``f(y, theta) = 0.5 y + theta`` the fixed point is ``y* = 2 theta`` so
    ``dy*/dtheta = 2``, independent of the forward solver. Confirms that using
    Anderson as the forward iterator leaves the ``jax.lax.custom_root`` +
    ``dense_tangent_solve`` implicit gradient exact.
    """
    with jax.enable_x64(True):

        def solve(theta: jax.Array) -> jax.Array:
            solver = AndersonAcceleration(rtol=1e-12, atol=1e-12, history_size=4)
            return anderson_fixed_point(
                lambda y, t: 0.5 * y + t, solver, jnp.array(0.0), args=theta, max_steps=64
            ).value

        value = solve(jnp.array(1.5))
        gradient = jax.grad(solve)(jnp.array(1.5))
    assert float(value) == pytest.approx(3.0, abs=1e-9)
    assert float(gradient) == pytest.approx(2.0, abs=1e-9)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_nonlinear_contraction_banach_step_bound_and_analytic_ift(dtype: jnp.dtype) -> None:
    r"""Verify nonlinear contraction fixed point against the stopping-rule Banach bound and IFT.

    For ``g(x, s) = M x + 0.1 * sin(x - c) + (I - M) c + s`` with ``c = theta / 1.1`` and
    non-symmetric ``||M||_2 <= 0.35``, ``g(., 0)`` is a global Banach contraction with
    Lipschitz constant ``L = ||M||_2 + 0.1 < 0.45 < 1`` and unique fixed point ``x*(0) = c``.

    Under the default supremum norm ``AndersonAcceleration.norm = _max_norm``, a solve
    returning ``Status.SUCCESS`` keeps the iterate ``y = sol.value`` satisfying
    ``max_i |g_i(y) - y_i| / (atol + rtol * max(|g_i(y)|, |y_i|)) < 1``. Since
    ``|g_i(y)| <= |y_i| + |g_i(y) - y_i|``, componentwise algebra gives
    ``||g(y) - y||_2 <= (sqrt(d) * atol + rtol * ||y||_2) / (1 - rtol)``, and Banach's
    contraction inequality ``||y - x*||_2 <= ||g(y) - y||_2 / (1 - L)`` with
    ``||y||_2 <= ||x*||_2 + ||y - x*||_2`` yields the non-vacuous a-priori bound
    ``||y - x*||_2 <= (sqrt(d) * atol + rtol * ||x*||_2) / ((1 - L)(1 - rtol) - rtol)``
    (plus working-precision evaluation rounding of ``g``). At ``s = 0`` the exact IFT
    Jacobian is ``dx*/ds = (I - J)^{-1} = (0.9 * I - M)^{-1}``.
    """
    from opifex.core.solver import (
        dense_tangent_solve,
        gmres_tangent_solve,
        lsmr_tangent_solve,
    )

    enable_x64 = dtype == jnp.float64
    with jax.enable_x64(enable_x64):
        mat_m = jnp.array(
            [
                [0.20, -0.08, 0.04, 0.02],
                [0.10, -0.15, 0.05, -0.03],
                [-0.04, 0.09, 0.18, 0.06],
                [0.03, -0.02, -0.07, -0.12],
            ],
            dtype=dtype,
        )
        theta = jnp.array([1.3, -2.1, 0.7, -3.4], dtype=dtype)
        c_star = theta / jnp.asarray(1.1, dtype=dtype)
        eye = jnp.eye(4, dtype=dtype)
        s_zero = jnp.zeros(4, dtype=dtype)

        with jax.enable_x64(True):
            m_64 = mat_m.astype(jnp.float64)
            lipschitz = float(jnp.linalg.svd(m_64, compute_uv=False)[0]) + 0.1
            assert lipschitz < 0.45
            a_ift_64 = 0.9 * jnp.eye(4, dtype=jnp.float64) - m_64
            svals_ift = jnp.linalg.svd(a_ift_64, compute_uv=False)
            inv_ift_norm = float(1.0 / svals_ift[-1])
            cond_ift = float(svals_ift[0] / svals_ift[-1])
            exact_ift_64 = jnp.linalg.inv(a_ift_64)
            c_star_64 = c_star.astype(jnp.float64)
            c_star_norm = float(jnp.linalg.norm(c_star_64))

        tol = 1e-11 if enable_x64 else 1e-6
        u_round = 0.5 * float(jnp.finfo(dtype).eps)
        anderson = AndersonAcceleration(rtol=tol, atol=tol, history_size=5)
        assert anderson.norm is fp_mod._max_norm

        # A-priori Banach + stopping-rule bound (independent of sol.value):
        denom_banach = (1.0 - lipschitz) * (1.0 - tol) - tol
        apriori_bound = (math.sqrt(4.0) * tol + tol * c_star_norm) / denom_banach + (
            8.0 * u_round * c_star_norm
        ) / (1.0 - lipschitz)
        assert apriori_bound < 1e-4 * c_star_norm

        def g_map(x: jax.Array, s_in: jax.Array) -> jax.Array:
            return (
                mat_m @ x
                + jnp.asarray(0.1, dtype=dtype) * jnp.sin(x - c_star)
                + (eye - mat_m) @ c_star
                + s_in
            )

        gamma_12 = (12.0 * u_round) / (1.0 - 12.0 * u_round)
        lu_ceil = max(64.0 * u_round, 8.0 * math.sqrt(4.0) * gamma_12)

        for t_solver in (dense_tangent_solve, lsmr_tangent_solve, gmres_tangent_solve):
            sol = anderson_fixed_point(
                g_map,
                anderson,
                jnp.zeros(4, dtype=dtype),
                args=s_zero,
                max_steps=64,
                tangent_solver=t_solver,
            )
            assert int(sol.status) == int(Status.SUCCESS)

            with jax.enable_x64(True):
                y_64 = sol.value.astype(jnp.float64)
                err_y = float(jnp.linalg.norm(y_64 - c_star_64))
            assert err_y <= apriori_bound

            jac_rev = jax.jacrev(
                lambda s_arg, ts=t_solver: (
                    anderson_fixed_point(
                        g_map,
                        anderson,
                        jnp.zeros(4, dtype=dtype),
                        args=s_arg,
                        max_steps=64,
                        tangent_solver=ts,
                    ).value
                )
            )(s_zero)
            # Since J_g(y) - J_g(c*) = 0.1 diag(cos(y - c*) - 1) has ||Delta J||_2 <= 0.05 ||y - c*||_2^2,
            # ||(I - J_g(y))^-1 - (I - J_g(c*))^-1||_2 <= 2 * ||(I - J_g(c*))^-1||_2^2 * 0.05 * apriori_bound^2,
            # plus linear tangent solve backward error cond(I - J_g) * sqrt(4) * bwd_ceil * ||(I - J_g)^-1||_2:
            if t_solver is dense_tangent_solve:
                bwd_ceil = lu_ceil
            elif t_solver is lsmr_tangent_solve:
                bwd_ceil = max(100.0 * max(16.0 * u_round, 1e-12), lu_ceil)
            else:
                bwd_ceil = max(100.0 * max(20.0 * u_round, 1e-12), lu_ceil)
            ift_trunc = 0.1 * (inv_ift_norm**2) * (apriori_bound**2)
            ift_bwd = cond_ift * 2.0 * bwd_ceil * inv_ift_norm + 2.0 * u_round * inv_ift_norm
            with jax.enable_x64(True):
                ift_err = float(
                    jnp.linalg.norm(jnp.asarray(jac_rev, dtype=jnp.float64) - exact_ift_64)
                )
            assert ift_err <= ift_trunc + ift_bwd


def test_mixing_coefficients_sum_to_one() -> None:
    """The Pulay/DIIS coefficients are an affine combination (sum to one)."""
    with jax.enable_x64(True):
        solver = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=4)
        residuals = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]], dtype=jnp.float64)
        coefficients = solver._mixing_coefficients(residuals, jnp.array(3))
        assert float(jnp.sum(coefficients)) == pytest.approx(1.0, abs=1e-10)
        # The unfilled fourth slot receives zero weight.
        assert float(coefficients[3]) == pytest.approx(0.0, abs=1e-12)


def test_is_jit_and_vmap_compatible() -> None:
    """The full solve jits and vmaps over a batch of fixed-point problems in one trace."""
    with jax.enable_x64(True):

        def solve(theta: jax.Array) -> jax.Array:
            solver = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=4)
            return anderson_fixed_point(
                lambda y, t: 0.5 * y + t, solver, jnp.array(0.0), args=theta, max_steps=64
            ).value

        thetas = jnp.array([1.0, 2.0, 3.0])
        counter = TraceCounter()
        jitted_vmap = jax.jit(counter.wrap(jax.vmap(solve)))
        with counter.expect(new_traces=1):
            batched = jitted_vmap(thetas)
        with counter.expect(new_traces=0):
            batched_shifted = jitted_vmap(thetas + 0.5)
        assert jnp.allclose(batched, 2.0 * thetas, atol=1e-9)
        assert jnp.allclose(batched_shifted, 2.0 * (thetas + 0.5), atol=1e-9)

    # Nested vmap(vmap) and second-order jvp(grad) across all 3 tangent solvers:
    from flax import nnx

    from opifex.core.solver import (
        dense_tangent_solve,
        gmres_tangent_solve,
        lsmr_tangent_solve,
    )

    with jax.enable_x64(True):
        solver_64 = AndersonAcceleration(rtol=1e-12, atol=1e-12, history_size=4)
        grid_th = jnp.array([[0.5, 1.0, 1.5], [-0.5, -1.0, 2.0]], dtype=jnp.float64)
        for t_solver in (dense_tangent_solve, lsmr_tangent_solve, gmres_tangent_solve):
            nested_out = jax.jit(
                jax.vmap(
                    jax.vmap(
                        lambda th, ts=t_solver: (
                            anderson_fixed_point(
                                lambda y, t: 0.5 * y + t,
                                solver_64,
                                jnp.zeros_like(th),
                                args=th,
                                max_steps=32,
                                tangent_solver=ts,
                            ).value
                        )
                    )
                )
            )(grid_th)
            assert jnp.allclose(nested_out, 2.0 * grid_th, atol=1e-10)

            def quad_root_loss(th: jax.Array, ts=t_solver) -> jax.Array:
                ystar = anderson_fixed_point(
                    lambda y, t: 0.5 * y + t,
                    solver_64,
                    jnp.zeros_like(th),
                    args=th,
                    max_steps=32,
                    tangent_solver=ts,
                ).value
                return 0.5 * jnp.sum(ystar * ystar)

            th0 = jnp.array([1.2, -0.8], dtype=jnp.float64)
            v0 = jnp.array([0.3, -0.5], dtype=jnp.float64)
            # y* = 2 * th => 0.5 * ||y*||^2 = 2 * ||th||^2 => grad = 4 * th, HVP = 4 * v
            g_val, hvp_val = jax.jvp(jax.grad(quad_root_loss), (th0,), (v0,))
            assert jnp.allclose(g_val, 4.0 * th0, atol=1e-9)
            assert jnp.allclose(hvp_val, 4.0 * v0, atol=1e-8)

    # @nnx.jit + nnx.value_and_grad bi-level training through anderson_fixed_point
    net = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    solver_32 = AndersonAcceleration(rtol=1e-6, atol=1e-6, history_size=4)
    x_feat = jnp.array([1.0, -0.5], dtype=jnp.float32)
    target_fp = jnp.array([2.0, -1.0], dtype=jnp.float32)

    @nnx.jit
    def fp_grad_norm(model: nnx.Linear) -> jax.Array:
        def loss_fn(m: nnx.Linear) -> jax.Array:
            shift = m(x_feat)
            fp = anderson_fixed_point(
                lambda y, s: 0.4 * y + s,
                solver_32,
                jnp.zeros(2, dtype=jnp.float32),
                args=shift,
                max_steps=32,
            ).value
            return jnp.sum((fp - target_fp) ** 2)

        _, grads = nnx.value_and_grad(loss_fn)(model)
        return jnp.linalg.norm(grads["kernel"][...])

    gnorm = fp_grad_norm(net)
    assert bool(jnp.isfinite(gnorm)) and float(gnorm) > 0.0


def test_fixed_point_import_isolation_and_status_codes() -> None:
    """_fixed_point imports only declared JAX/Flax/Opifex foundations and emits Status codes."""
    source = Path(fp_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed_roots = {
        "__future__",
        "collections",
        "dataclasses",
        "flax",
        "jax",
        "jaxtyping",
        "math",
        "opifex",
        "typing",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= allowed_roots

    solver = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=4)
    starved = anderson_fixed_point(
        lambda x, _: jnp.cos(x), solver, jnp.array(0.0, dtype=jnp.float32), max_steps=1
    )
    assert int(starved.status) == int(Status.MAX_ITERS)
    assert not bool(starved.is_converged)

    nonfinite = anderson_fixed_point(
        lambda x, _: x * jnp.nan, solver, jnp.array(1.0, dtype=jnp.float32), max_steps=4
    )
    assert int(nonfinite.status) == int(Status.NONFINITE)

    with pytest.raises(ValueError, match="max_steps"):
        anderson_fixed_point(lambda x, _: x, solver, jnp.array(0.0), max_steps=0)


def test_exact_fixed_point_atol_zero_unconverged_ift_and_gmres_pytree() -> None:
    """Exact fixed points avoid 0/0 Gram NaNs, atol=0 converges at y*=0, unconverged solves
    poison forward/reverse tangents, and gmres_tangent_solve handles PyTrees, dim>128, and vmap."""
    from opifex.core.solver.implicit import gmres_tangent_solve

    # 1) Starting directly at the exact fixed point y0 = 2 * theta (residual == 0, scale == 0)
    def solve_at_root(theta: jax.Array) -> jax.Array:
        solver = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=4)
        return anderson_fixed_point(
            lambda y, t: 0.5 * y + t,
            solver,
            2.0 * theta,
            args=theta,
            max_steps=16,
        ).value

    theta0 = jnp.array(1.5, dtype=jnp.float32)
    assert float(solve_at_root(theta0)) == pytest.approx(3.0, abs=1e-6)
    assert float(jax.grad(solve_at_root)(theta0)) == pytest.approx(2.0, abs=1e-5)
    assert float(jax.jacfwd(solve_at_root)(theta0)) == pytest.approx(2.0, abs=1e-5)

    # 2) atol == 0.0 converging to y* = 0.0
    zero_solver = AndersonAcceleration(rtol=1e-6, atol=0.0, history_size=4)
    zero_sol = anderson_fixed_point(
        lambda y, _: jnp.zeros_like(y),
        zero_solver,
        jnp.array([1.0, -2.0], dtype=jnp.float32),
        max_steps=8,
    )
    assert int(zero_sol.status) == int(Status.SUCCESS)
    assert jnp.allclose(zero_sol.value, jnp.zeros(2, dtype=jnp.float32))

    # 3) Unconverged solve (MAX_ITERS): finite primal, NaN forward and reverse tangents
    def starved_solve(theta: jax.Array) -> jax.Array:
        solver = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=4)
        return anderson_fixed_point(
            lambda y, t: jnp.cos(y) + 0.2 * t,
            solver,
            jnp.array(0.0, dtype=jnp.float32),
            args=theta,
            max_steps=1,
        ).value

    assert bool(jnp.isfinite(starved_solve(jnp.float32(0.5))))
    assert bool(jnp.isnan(jax.grad(starved_solve)(jnp.float32(0.5))))
    assert bool(jnp.isnan(jax.jacfwd(starved_solve)(jnp.float32(0.5))))

    # 4) gmres_tangent_solve on PyTree rhs, size=160 (> 128), and vmap mixed zero/non-zero/NaN
    rhs_tree = {
        "a": jnp.array([2.0, -4.0], dtype=jnp.float32),
        "b": jnp.array([6.0], dtype=jnp.float32),
    }
    sol_tree = gmres_tangent_solve(lambda t: jax.tree.map(lambda x: 2.0 * x, t), rhs_tree)
    assert jnp.allclose(sol_tree["a"], jnp.array([1.0, -2.0], dtype=jnp.float32), atol=1e-5)
    assert jnp.allclose(sol_tree["b"], jnp.array([3.0], dtype=jnp.float32), atol=1e-5)

    diag_160 = jnp.linspace(1.0, 5.0, 160, dtype=jnp.float32)
    b_160 = jnp.ones(160, dtype=jnp.float32)
    x_160 = gmres_tangent_solve(lambda v: diag_160 * v, b_160)
    assert jnp.allclose(x_160, b_160 / diag_160, atol=1e-4)

    batch_rhs = jnp.array(
        [[0.0, 0.0], [3.0, -6.0], [jnp.nan, 1.0]],
        dtype=jnp.float32,
    )
    vmapped_sol = jax.vmap(lambda b: gmres_tangent_solve(lambda v: 3.0 * v, b))(batch_rhs)
    assert jnp.allclose(vmapped_sol[0], jnp.zeros(2, dtype=jnp.float32))
    assert jnp.allclose(vmapped_sol[1], jnp.array([1.0, -2.0], dtype=jnp.float32), atol=1e-5)
    assert bool(jnp.all(jnp.isnan(vmapped_sol[2])))


@pytest.mark.slow
def test_converges_water_lda_density() -> None:
    """Anderson converges the water/LDA SCF density that plain Roothaan cannot.

    Plain Roothaan iteration stalls at a density residual of ~0.9 for water/LDA;
    Anderson drives it below 1e-9 in a few iterations.
    """
    from opifex.core.quantum.molecular_system import MolecularSystem
    from opifex.neural.quantum.dft._energy import _density_from_fock, _scf_step
    from opifex.neural.quantum.dft.grid import build_molecular_grid_traceable
    from opifex.neural.quantum.dft.scf import SCFSolver

    with jax.enable_x64(True):
        positions = jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])
        system = MolecularSystem(
            atomic_numbers=jnp.array([8, 1, 1]), positions=positions, basis_set="sto-3g"
        )
        grid = build_molecular_grid_traceable(system, n_radial=8, n_theta=6, n_phi=8)
        solver_obj = SCFSolver(system, grid_template=grid)
        integrals = solver_obj._integrals(positions)
        functional = solver_obj._functional
        n_occupied = solver_obj._n_occupied

        initial = _density_from_fock(
            integrals.core_hamiltonian, integrals.orthogonaliser, n_occupied
        )[0]
        anderson = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=6)
        solution = anderson_fixed_point(
            lambda density, _: _scf_step(density, integrals, functional, n_occupied, None),
            anderson,
            initial,
            max_steps=100,
        )
        converged = solution.value
        residual = jnp.max(
            jnp.abs(_scf_step(converged, integrals, functional, n_occupied, None) - converged)
        )
    assert float(residual) < 1e-9


class TestTheMixingSolveAtDefaultPrecision:
    """The Tikhonov scale has to be measured in the arithmetic that runs.

    Every test above enables x64, so the default-precision path was never exercised.
    In float32 the residuals of a converging SCF reach the arithmetic's own noise floor
    (~1e-7) while the tolerance is still unmet, and the Gram matrix of near-identical
    noise vectors is numerically singular: a stabiliser too small to perturb it leaves
    the mixing solve to return non-finite coefficients.
    """

    @staticmethod
    def _rank_deficient_history(dtype: jnp.dtype) -> jax.Array:
        """Six residuals differing only at the last representable digit."""
        base = jnp.linspace(0.1, 0.9, 16).astype(dtype)
        noise = jnp.finfo(dtype).eps * jnp.arange(1, 7, dtype=dtype)[:, None]
        return (base[None, :] * (1.0 + noise)).astype(dtype)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_coefficients_stay_finite_on_a_rank_deficient_history(self, dtype: jnp.dtype) -> None:
        with jax.enable_x64(True):
            solver = AndersonAcceleration(rtol=1e-8, atol=1e-8, history_size=6)
            residuals = self._rank_deficient_history(dtype)

            coefficients = solver._mixing_coefficients(residuals, jnp.array(6))
            assert bool(jnp.all(jnp.isfinite(coefficients)))
            assert float(jnp.sum(coefficients)) == pytest.approx(1.0, abs=1e-5)

    def test_the_scale_comes_from_the_residuals_not_the_empty_slots(self) -> None:
        # Unfilled slots are masked into the Gram as the identity, whose entries are 1
        # while a converging residual's are ~1e-12. A scale taken from the padded trace
        # therefore regularises by many times the data while the history fills.
        #
        # The two residuals are orthogonal with very different norms, so the exact
        # coefficients are c ~ (1/a, 1/b) normalised -- far from equal. A stabiliser
        # much larger than both drives them to (0.5, 0.5) instead, which is what a
        # padded scale produces and what equal-magnitude residuals would hide.
        with jax.enable_x64(True):
            solver = AndersonAcceleration(rtol=1e-8, atol=1e-8, history_size=6)
            residuals = jnp.zeros((6, 4), jnp.float64)
            residuals = residuals.at[0, 0].set(1e-6).at[1, 1].set(1e-7)

            coefficients = solver._mixing_coefficients(residuals, jnp.array(2))
            # a = 1e-12, b = 1e-14: c = (1/a, 1/b) / (1/a + 1/b) = (1/101, 100/101).
            assert float(coefficients[0]) == pytest.approx(1.0 / 101.0, rel=1e-3)
            assert float(coefficients[1]) == pytest.approx(100.0 / 101.0, rel=1e-3)
            assert float(jnp.sum(coefficients[2:])) == pytest.approx(0.0, abs=1e-12)

    def test_an_oscillatory_map_converges_without_x64(self) -> None:
        # x = sqrt(2) cos(x), whose root is 0.8900586117066737 to float64 precision
        # (Brent, residual 4e-16). At default precision the old fixed stabiliser left
        # this solve returning NaN after nine steps.
        def stiff(x: jax.Array, _: None) -> jax.Array:
            return jnp.sqrt(2.0) * jnp.cos(x)

        solver = AndersonAcceleration(rtol=1e-5, atol=1e-6, history_size=5)

        solution = anderson_fixed_point(stiff, solver, jnp.array(0.0, jnp.float32), max_steps=128)
        residual = float(jnp.abs(stiff(solution.value, None) - solution.value))

        assert float(solution.value) == pytest.approx(0.8900586117066737, abs=1e-5)
        assert residual < 1e-5

    def test_the_scf_density_converges_without_x64(self) -> None:
        # The default-precision H2 solve: the Fock build itself is accurate to seven
        # digits in float32, so a non-finite energy can only come from the mixing.
        from opifex.core.quantum.molecular_system import MolecularSystem
        from opifex.neural.quantum.dft.scf import SCFSolver

        bohr_per_angstrom = 1.0 / 0.52917721067
        system = MolecularSystem(
            atomic_numbers=jnp.array([1, 1]),
            positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 * bohr_per_angstrom]]),
            basis_set="sto-3g",
        )

        result = SCFSolver(system, convergence_tolerance=1e-6).solve()

        assert bool(jnp.isfinite(result.total_energy))
        assert float(result.total_energy) == pytest.approx(-1.1212060, abs=1e-4)
