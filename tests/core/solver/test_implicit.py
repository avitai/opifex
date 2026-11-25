"""Tests for shared pure-JAX solver dtype inference, pytree casting, and tangent solves."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from substrax.testing import TraceCounter

from opifex.core.solver import (
    cast_inexact_pytree,
    dense_tangent_solve,
    gmres_tangent_solve,
    infer_inexact_dtype,
    lsmr_tangent_solve,
)


class TestInferInexactDtypeAndCastPytree:
    """Verify float32 floor, weak-type scalar preservation under x64, and pytree casting."""

    def test_infer_inexact_dtype_floors_at_float32_and_preserves_weak_scalars_under_x64(
        self,
    ) -> None:
        with jax.enable_x64(True):
            assert infer_inexact_dtype() == jnp.float32
            assert infer_inexact_dtype(None, 3, True, "label", jnp.arange(4, dtype=jnp.int32)) == (
                jnp.float32
            )
            assert infer_inexact_dtype(jnp.ones(2, dtype=jnp.float16)) == jnp.float32
            assert infer_inexact_dtype(jnp.ones(2, dtype=jnp.bfloat16)) == jnp.float32

            # Python float scalars and weakly typed jax scalars must not promote float32 to float64
            vec_f32 = jnp.ones(3, dtype=jnp.float32)
            assert infer_inexact_dtype(vec_f32, 0.0, 1.0) == jnp.float32
            assert infer_inexact_dtype(vec_f32, jnp.asarray(0.0), jnp.asarray(1.0)) == jnp.float32
            # Standalone 0-D jax.Array(0.0) under x64 keeps its float64 dtype
            assert infer_inexact_dtype(jnp.array(0.0)) == jnp.float64

            # Explicit float64 arrays or numpy float64 scalars promote to float64 when x64 is active
            vec_f64 = jnp.ones(3, dtype=jnp.float64)
            assert infer_inexact_dtype(vec_f64, 0.0) == jnp.float64
            assert infer_inexact_dtype(vec_f32, vec_f64) == jnp.float64
            assert infer_inexact_dtype(vec_f32, np.float64(1.0)) == jnp.float64

            # Complex leaves preserve complex64 vs complex128
            assert infer_inexact_dtype(jnp.ones(2, dtype=jnp.complex64), 1.0j) == jnp.complex64
            assert infer_inexact_dtype(jnp.ones(2, dtype=jnp.complex128), 1.0) == jnp.complex128

    def test_cast_inexact_pytree_converts_only_inexact_leaves(self) -> None:
        with jax.enable_x64(True):
            raw_tree = {
                "weights": jnp.array([1.0, -2.0], dtype=jnp.float64),
                "half": jnp.array([0.5], dtype=jnp.float16),
                "scalar": 2.5,
                "indices": jnp.array([0, 2], dtype=jnp.int32),
                "flag": True,
            }
            cast_f32 = cast_inexact_pytree(raw_tree, jnp.dtype(jnp.float32))
            assert cast_f32["weights"].dtype == jnp.float32
            assert cast_f32["half"].dtype == jnp.float32
            assert isinstance(cast_f32["scalar"], jax.Array)
            assert cast_f32["scalar"].dtype == jnp.float32
            assert cast_f32["indices"].dtype == jnp.int32
            assert cast_f32["flag"] is True

            cast_f64 = cast_inexact_pytree(raw_tree, jnp.dtype(jnp.float64))
            assert cast_f64["weights"].dtype == jnp.float64
            assert cast_f64["half"].dtype == jnp.float64
            assert cast_f64["scalar"].dtype == jnp.float64
            assert cast_f64["indices"].dtype == jnp.int32


def _lu_bwd_contract_bound(size: int, dtype: jnp.dtype) -> float:
    """Solver normwise backward-error acceptance contract ``_lu_backward_error_ceiling``.

    Because ``||A||_F >= ||A||_F / sqrt(n)``, evaluating
    ``||Ax - b||_2 / (||A||_F ||x||_2 + ||b||_2)`` against this ceiling is conservative.
    """
    u_round = 0.5 * float(jnp.finfo(dtype).eps)
    gamma_3n = (3.0 * size * u_round) / (1.0 - 3.0 * size * u_round)
    return min(max(64.0 * u_round, 8.0 * math.sqrt(size) * gamma_3n), 0.1)


def _collect_lu_input_shapes(closed_jaxpr: object) -> list[tuple[int, ...]]:
    """Recursively collect every ``lu`` primitive's operand shape across nested jaxprs."""
    shapes: list[tuple[int, ...]] = []
    core = getattr(closed_jaxpr, "jaxpr", closed_jaxpr)
    for eqn in getattr(core, "eqns", ()):
        if eqn.primitive.name == "lu":
            shapes.extend(v.aval.shape for v in eqn.invars if hasattr(v, "aval"))
        for val in eqn.params.values():
            if hasattr(val, "eqns") or hasattr(val, "jaxpr"):
                shapes.extend(_collect_lu_input_shapes(val))
            elif isinstance(val, tuple | list):
                for item in val:
                    if hasattr(item, "eqns") or hasattr(item, "jaxpr"):
                        shapes.extend(_collect_lu_input_shapes(item))
    return shapes


class TestTangentSolversDenseLsmrAndGmres:
    """Verify dense LU default, matrix-free LSMR, stagnation-aware GMRES, and Higham bounds."""

    def test_nonsymmetric_indefinite_m0_50x50_and_150x150_dense_lsmr_and_gmres_stagnation(
        self,
    ) -> None:
        with jax.enable_x64(True):
            for size, seed in ((50, 0), (150, 3)):
                rng = np.random.default_rng(seed)
                u_mat, _ = np.linalg.qr(rng.standard_normal((size, size)))
                v_mat, _ = np.linalg.qr(rng.standard_normal((size, size)))
                mat = jnp.asarray(
                    (u_mat * np.logspace(0.0, -2.0, size)) @ v_mat.T, dtype=jnp.float64
                )
                rhs = jnp.asarray(rng.standard_normal(size), dtype=jnp.float64)
                bwd_bound = _lu_bwd_contract_bound(size, jnp.dtype(jnp.float64))
                frob = float(jnp.linalg.norm(mat))

                sol_dense = jax.jit(lambda b, m=mat: dense_tangent_solve(lambda x: m @ x, b))(rhs)
                bwd_dense = float(
                    jnp.linalg.norm(mat @ sol_dense - rhs)
                    / (frob * jnp.linalg.norm(sol_dense) + jnp.linalg.norm(rhs))
                )
                assert bwd_dense <= bwd_bound

                sol_lsmr = jax.jit(lambda b, m=mat: lsmr_tangent_solve(lambda x: m @ x, b))(rhs)
                # Fong & Saunders (2011) stopping rule S1 checks normr < rtol * (normA_k ||x|| + ||b||)
                # over k <= max_steps = 10 * n steps (so normA_k <= sqrt(20) ||A||_F without
                # full reorthogonalization), bounding Frobenius backward error by sqrt(20) * rtol.
                bwd_lsmr = float(
                    jnp.linalg.norm(mat @ sol_lsmr - rhs)
                    / (frob * jnp.linalg.norm(sol_lsmr) + jnp.linalg.norm(rhs))
                )
                assert bwd_lsmr <= math.sqrt(20.0) * 1e-12
                rel_lsmr = float(jnp.linalg.norm(mat @ sol_lsmr - rhs) / jnp.linalg.norm(rhs))
                assert rel_lsmr <= math.sqrt(20.0) * 1e-12 * (frob * 100.0 + 1.0)

                # Default restarted GMRES (restart=32 < size) stagnates on indefinite Haar M0
                # and must halt early and emit NaN rather than returning an inaccurate vector.
                sol_gmres_default = jax.jit(
                    lambda b, m=mat: gmres_tangent_solve(lambda x: m @ x, b)
                )(rhs)
                assert bool(jnp.all(jnp.isnan(sol_gmres_default)))

            # Full-Krylov GMRES (restart=50 on 50x50) converges cleanly.
            rng50 = np.random.default_rng(0)
            u_50, _ = np.linalg.qr(rng50.standard_normal((50, 50)))
            v_50, _ = np.linalg.qr(rng50.standard_normal((50, 50)))
            mat_50 = jnp.asarray((u_50 * np.logspace(0.0, -2.0, 50)) @ v_50.T, dtype=jnp.float64)
            rhs_50 = jnp.asarray(rng50.standard_normal(50), dtype=jnp.float64)
            sol_gmres_50 = gmres_tangent_solve(lambda x: mat_50 @ x, rhs_50, restart=50)
            assert (
                float(jnp.linalg.norm(mat_50 @ sol_gmres_50 - rhs_50) / jnp.linalg.norm(rhs_50))
                < 1e-12
            )

            # Verify forward (JVP) and reverse (grad) through custom_root for dense and LSMR
            weights = jnp.asarray(rng50.standard_normal(50), dtype=jnp.float64)
            exact_sol = jnp.linalg.solve(mat_50, rhs_50)
            expected_grad = jnp.linalg.solve(mat_50.T, weights)
            tangent_dir = jnp.asarray(rng50.standard_normal(50), dtype=jnp.float64)
            expected_jvp = float(jnp.dot(jnp.linalg.solve(mat_50, tangent_dir), weights))

            frob_50 = float(jnp.linalg.norm(mat_50))
            lsmr_50_rel = math.sqrt(20.0) * 1e-12 * (frob_50 * 100.0 + 1.0)
            for t_solver in (dense_tangent_solve, lsmr_tangent_solve):

                def root_loss(theta: jax.Array, ts=t_solver) -> jax.Array:
                    y_star = jax.lax.custom_root(
                        lambda y: mat_50 @ y - theta,
                        jnp.zeros_like(theta),
                        lambda _f, _x0: jnp.linalg.solve(mat_50, theta),
                        ts,
                    )
                    return jnp.dot(y_star, weights)

                grad_theta = jax.grad(root_loss)(rhs_50)
                rel_g50 = float(
                    jnp.linalg.norm(grad_theta - expected_grad) / jnp.linalg.norm(expected_grad)
                )
                assert rel_g50 <= 100.0 * lsmr_50_rel
                primal_val, jvp_val = jax.jvp(root_loss, (rhs_50,), (tangent_dir,))
                assert abs(float(primal_val - jnp.dot(exact_sol, weights))) < 1e-12
                jvp_scale = float(
                    jnp.linalg.norm(weights)
                    * jnp.linalg.norm(jnp.linalg.solve(mat_50, tangent_dir))
                )
                assert abs(float(jvp_val) - expected_jvp) <= 100.0 * lsmr_50_rel * jvp_scale

    def test_float32_cond_1e6_succeeds_under_dense_and_lsmr_with_derived_higham_bound(
        self,
    ) -> None:
        rng = np.random.default_rng(11)
        size = 16
        u_16, _ = np.linalg.qr(rng.standard_normal((size, size)))
        v_16, _ = np.linalg.qr(rng.standard_normal((size, size)))
        mat_cond = jnp.asarray((u_16 * np.logspace(0.0, -6.0, size)) @ v_16.T, dtype=jnp.float32)
        x_ref = jnp.linspace(-1.0, 1.0, size, dtype=jnp.float32)
        rhs = mat_cond @ x_ref
        bwd_bound = _lu_bwd_contract_bound(size, jnp.dtype(jnp.float32))
        frob = float(jnp.linalg.norm(mat_cond))

        sol_lu = dense_tangent_solve(lambda v: mat_cond @ v, rhs)
        assert bool(jnp.all(jnp.isfinite(sol_lu)))
        bwd_lu = float(
            jnp.linalg.norm(mat_cond @ sol_lu - rhs)
            / (frob * jnp.linalg.norm(sol_lu) + jnp.linalg.norm(rhs))
        )
        assert bwd_lu <= bwd_bound

        # LSMR avoids forming A^T A (never squaring kappa=1e6 to 1e12 > 1/eps32).
        sol_lsmr = lsmr_tangent_solve(lambda v: mat_cond @ v, rhs, rtol=1e-6, max_steps=512)
        assert bool(jnp.all(jnp.isfinite(sol_lsmr)))
        bwd_lsmr = float(
            jnp.linalg.norm(mat_cond @ sol_lsmr - rhs)
            / (frob * jnp.linalg.norm(sol_lsmr) + jnp.linalg.norm(rhs))
        )
        assert bwd_lsmr <= 100.0 * 1e-6

    def test_nonsymmetric_200x200_cond_1e4_gmres_and_tiny_rhs_scale_invariance(self) -> None:
        with jax.enable_x64(True):
            rng = np.random.default_rng(7)
            q_200, _ = np.linalg.qr(rng.standard_normal((200, 200)))
            diag_200 = np.logspace(0.0, -4.0, 200)
            raw_skew = rng.standard_normal((200, 200)) / np.sqrt(200.0)
            skew = 0.1 * (raw_skew - raw_skew.T)
            mat_200 = jnp.asarray(
                q_200 @ (np.diag(diag_200) + diag_200[:, None] * skew) @ q_200.T,
                dtype=jnp.float64,
            )
            rhs_200 = jnp.asarray(rng.standard_normal(200), dtype=jnp.float64)

            frob_200 = float(jnp.linalg.norm(mat_200))
            gmres_bwd_ceil = max(100.0 * 1e-12, _lu_bwd_contract_bound(200, jnp.dtype(jnp.float64)))
            for t_solver in (dense_tangent_solve, gmres_tangent_solve):
                sol_200 = jax.jit(lambda b, ts=t_solver: ts(lambda x: mat_200 @ x, b))(rhs_200)
                bwd_200 = float(
                    jnp.linalg.norm(mat_200 @ sol_200 - rhs_200)
                    / (frob_200 * jnp.linalg.norm(sol_200) + jnp.linalg.norm(rhs_200))
                )
                assert bwd_200 <= gmres_bwd_ceil

                # Tiny cotangent scale (1e-20) must not exit early at x=0 due to unscaled atol
                tiny_rhs = rhs_200 * 1e-20
                sol_tiny = t_solver(lambda x: mat_200 @ x, tiny_rhs)
                bwd_tiny = float(
                    jnp.linalg.norm(mat_200 @ sol_tiny - tiny_rhs)
                    / (frob_200 * jnp.linalg.norm(sol_tiny) + jnp.linalg.norm(tiny_rhs))
                )
                assert bwd_tiny <= gmres_bwd_ceil

            # Unpreconditioned Golub-Kahan on a dense geometric spectrum with kappa=1e4 > 2000
            # does not converge in default max_steps=10*n=2000 and must emit NaN rather than an
            # inaccurate iterate, whereas on 200x200 kappa=1e2 default LSMR converges cleanly.
            assert bool(jnp.all(jnp.isnan(lsmr_tangent_solve(lambda x: mat_200 @ x, rhs_200))))
            diag_c2 = np.logspace(0.0, -2.0, 200)
            mat_200_c2 = jnp.asarray(
                q_200 @ (np.diag(diag_c2) + diag_c2[:, None] * skew) @ q_200.T,
                dtype=jnp.float64,
            )
            sol_lsmr_200 = jax.jit(lambda b: lsmr_tangent_solve(lambda x: mat_200_c2 @ x, b))(
                rhs_200
            )
            lsmr_c2_ceil = (
                math.sqrt(20.0) * 1e-12 * (float(jnp.linalg.norm(mat_200_c2)) * 100.0 + 1.0)
            )
            assert (
                float(
                    jnp.linalg.norm(mat_200_c2 @ sol_lsmr_200 - rhs_200) / jnp.linalg.norm(rhs_200)
                )
                <= lsmr_c2_ceil
            )
            sol_lsmr_tiny = lsmr_tangent_solve(lambda x: mat_200_c2 @ x, rhs_200 * 1e-20)
            assert (
                float(
                    jnp.linalg.norm(mat_200_c2 @ sol_lsmr_tiny - rhs_200 * 1e-20)
                    / jnp.linalg.norm(rhs_200 * 1e-20)
                )
                <= lsmr_c2_ceil
            )

            # Reverse-mode transpose_solve on 200x200 non-symmetric operator (Higham Thm. 7.2:
            # forward error <= cond_2(mat_200) * rel_res_transpose, with cond_2 ~ 1e4).
            probe = jnp.asarray(rng.standard_normal(200), dtype=jnp.float64)
            expected_grad_200 = jnp.linalg.solve(mat_200.T, probe)
            cond_200 = float(np.linalg.cond(np.asarray(mat_200)))
            for t_solver in (dense_tangent_solve, gmres_tangent_solve):
                grad_200 = jax.grad(
                    lambda b, ts=t_solver: jnp.dot(ts(lambda x: mat_200 @ x, b), probe)
                )(rhs_200)
                res_t = float(jnp.linalg.norm(mat_200.T @ grad_200 - probe))
                bwd_t = res_t / (
                    frob_200 * float(jnp.linalg.norm(grad_200)) + float(jnp.linalg.norm(probe))
                )
                assert bwd_t <= gmres_bwd_ceil
                rel_res_t = res_t / float(jnp.linalg.norm(probe))
                rel_grad_err = float(
                    jnp.linalg.norm(grad_200 - expected_grad_200)
                    / jnp.linalg.norm(expected_grad_200)
                )
                assert rel_grad_err <= cond_200 * rel_res_t + 1e-13
            grad_lsmr_200 = jax.grad(
                lambda b: jnp.dot(lsmr_tangent_solve(lambda x: mat_200_c2 @ x, b), probe)
            )(rhs_200)
            expected_grad_c2 = jnp.linalg.solve(mat_200_c2.T, probe)
            rel_res_c2 = float(
                jnp.linalg.norm(mat_200_c2.T @ grad_lsmr_200 - probe) / jnp.linalg.norm(probe)
            )
            assert rel_res_c2 <= lsmr_c2_ceil
            assert (
                float(
                    jnp.linalg.norm(grad_lsmr_200 - expected_grad_c2)
                    / jnp.linalg.norm(expected_grad_c2)
                )
                <= float(np.linalg.cond(np.asarray(mat_200_c2))) * rel_res_c2 + 1e-13
            )

    def test_singular_and_stagnated_systems_return_nan_in_primal_jvp_grad_and_vmap(self) -> None:
        with jax.enable_x64(True):
            rng = np.random.default_rng(13)
            # 1) Rounded-singular non-symmetric 16x16 (both out-of-range and in-range RHS)
            u_16, _ = np.linalg.qr(rng.standard_normal((16, 16)))
            v_16, _ = np.linalg.qr(rng.standard_normal((16, 16)))
            sing_vals_16 = np.ones(16)
            sing_vals_16[-1] = 0.0
            mat_sing_16 = jnp.asarray((u_16 * sing_vals_16) @ v_16.T, dtype=jnp.float64)
            rhs_16 = jnp.ones(16, dtype=jnp.float64)
            rhs_16_in_range = mat_sing_16 @ rhs_16
            for t_solver in (dense_tangent_solve, lsmr_tangent_solve, gmres_tangent_solve):
                assert bool(jnp.all(jnp.isnan(t_solver(lambda x: mat_sing_16 @ x, rhs_16))))
            assert bool(
                jnp.all(jnp.isnan(dense_tangent_solve(lambda x: mat_sing_16 @ x, rhs_16_in_range)))
            )

            # 2) Exact zero-row singular 8x8
            mat_zero_8 = jnp.eye(8, dtype=jnp.float64).at[-1, -1].set(0.0)
            for t_solver in (dense_tangent_solve, lsmr_tangent_solve, gmres_tangent_solve):
                sol_zero_8 = t_solver(lambda x: mat_zero_8 @ x, jnp.ones(8, jnp.float64))
                assert bool(jnp.all(jnp.isnan(sol_zero_8)))

            # 3) Large 160x160 singular system (out-of-range and in-range RHS)
            u_160, _ = np.linalg.qr(rng.standard_normal((160, 160)))
            v_160, _ = np.linalg.qr(rng.standard_normal((160, 160)))
            sing_vals_160 = np.ones(160)
            sing_vals_160[-1] = 0.0
            mat_sing_160 = jnp.asarray((u_160 * sing_vals_160) @ v_160.T, dtype=jnp.float64)
            rhs_160 = jnp.ones(160, dtype=jnp.float64)
            rhs_160_in_range = mat_sing_160 @ rhs_160
            for t_solver in (dense_tangent_solve, lsmr_tangent_solve, gmres_tangent_solve):
                assert bool(jnp.all(jnp.isnan(t_solver(lambda x: mat_sing_160 @ x, rhs_160))))
            assert bool(
                jnp.all(
                    jnp.isnan(dense_tangent_solve(lambda x: mat_sing_160 @ x, rhs_160_in_range))
                )
            )

            # 4) Forced GMRES stagnation on 200x200 indefinite Haar non-symmetric matrix
            u_200, _ = np.linalg.qr(rng.standard_normal((200, 200)))
            v_200, _ = np.linalg.qr(rng.standard_normal((200, 200)))
            mat_stag_200 = jnp.asarray(
                (u_200 * np.logspace(0.0, -6.0, 200)) @ v_200.T, dtype=jnp.float64
            )
            rhs_200 = jnp.ones(200, dtype=jnp.float64)
            sol_stag_200 = gmres_tangent_solve(lambda x: mat_stag_200 @ x, rhs_200)
            assert bool(jnp.all(jnp.isnan(sol_stag_200)))

            # Forward JVP and reverse transpose_solve emit NaN on singular Jacobian
            for t_solver in (dense_tangent_solve, lsmr_tangent_solve, gmres_tangent_solve):
                _, jvp_sing = jax.jvp(
                    lambda b, ts=t_solver: ts(lambda x: mat_sing_16 @ x, b),
                    (rhs_16,),
                    (jnp.ones_like(rhs_16),),
                )
                assert bool(jnp.all(jnp.isnan(jvp_sing)))
                grad_sing = jax.grad(
                    lambda b, ts=t_solver: jnp.sum(ts(lambda x: mat_sing_16 @ x, b))
                )(rhs_16)
                assert bool(jnp.all(jnp.isnan(grad_sing)))

            # vmap isolation: valid lane succeeds, zero lane -> 0, singular lane -> NaN
            mats_batch = jnp.stack([jnp.eye(8, dtype=jnp.float64) * 2.0, mat_zero_8, jnp.eye(8)])
            rhs_batch = jnp.stack(
                [
                    jnp.ones(8, dtype=jnp.float64) * 4.0,
                    jnp.ones(8, dtype=jnp.float64),
                    jnp.zeros(8, dtype=jnp.float64),
                ]
            )
            for t_solver in (dense_tangent_solve, lsmr_tangent_solve, gmres_tangent_solve):
                vmapped = jax.vmap(lambda m, b, ts=t_solver: ts(lambda v: m @ v, b))(
                    mats_batch, rhs_batch
                )
                assert jnp.allclose(vmapped[0], jnp.full(8, 2.0, dtype=jnp.float64), atol=1e-12)
                assert bool(jnp.all(jnp.isnan(vmapped[1])))
                assert jnp.allclose(vmapped[2], jnp.zeros(8, dtype=jnp.float64), atol=1e-12)

    def test_float32_accuracy_and_single_trace_compilation_caching(self) -> None:
        rng = np.random.default_rng(29)
        size = 24
        u_24, _ = np.linalg.qr(rng.standard_normal((size, size)))
        v_24, _ = np.linalg.qr(rng.standard_normal((size, size)))
        mat_24 = jnp.asarray((u_24 * np.linspace(1.0, 0.2, size)) @ v_24.T, dtype=jnp.float32)
        rhs_a = jnp.asarray(rng.standard_normal((4, size)), dtype=jnp.float32)
        rhs_b = rhs_a * jnp.float32(0.5)

        # Derived Higham (2nd ed., Thm. 7.1 & Thm. 9.4) normwise backward-error ceiling;
        # because rhs_b = 0.5 * rhs_a scales only the binary exponent, unit-normalized
        # solves reproduce 0.5 * out_a to within 2 * cond_2(mat_24) * bwd_bound.
        bwd_bound = _lu_bwd_contract_bound(size, jnp.dtype(jnp.float32))
        frob = float(jnp.linalg.norm(mat_24))
        cond_2 = 5.0

        for t_solver, mult in (
            (dense_tangent_solve, 1.0),
            (lsmr_tangent_solve, 4.0),
            (gmres_tangent_solve, 4.0),
        ):
            counter = TraceCounter()
            batched_solve = jax.jit(
                counter.wrap(jax.vmap(lambda b, ts=t_solver: ts(lambda x: mat_24 @ x, b)))
            )
            with counter.expect(new_traces=1):
                out_a = batched_solve(rhs_a)
            with counter.expect(new_traces=0):
                out_b = batched_solve(rhs_b)

            assert out_a.dtype == jnp.float32
            res_norms = jnp.linalg.norm(jax.vmap(lambda x: mat_24 @ x)(out_a) - rhs_a, axis=-1)
            denoms = frob * jnp.linalg.norm(out_a, axis=-1) + jnp.linalg.norm(rhs_a, axis=-1)
            assert float(jnp.max(res_norms / denoms)) <= mult * bwd_bound
            scale_diff = jnp.linalg.norm(out_b - 0.5 * out_a, axis=-1) / jnp.linalg.norm(
                0.5 * out_a, axis=-1
            )
            assert float(jnp.max(scale_diff)) <= 2.0 * cond_2 * mult * bwd_bound

        # Complex64 linear systems across dense LU, Hermitian-adjoint LSMR, and GMRES:
        mat_c = jnp.array(
            [[2.0 + 1.0j, 0.5 - 0.3j], [-0.2 + 0.4j, 3.0 - 0.5j]], dtype=jnp.complex64
        )
        rhs_c = jnp.array([1.0 + 0.5j, -0.5 + 1.2j], dtype=jnp.complex64)
        frob_c = float(jnp.linalg.norm(mat_c))
        bwd_c_bound = _lu_bwd_contract_bound(2, jnp.dtype(jnp.float32))
        for t_solver, mult in (
            (dense_tangent_solve, 1.0),
            (lsmr_tangent_solve, 4.0),
            (gmres_tangent_solve, 4.0),
        ):
            sol_c = t_solver(lambda v: mat_c @ v, rhs_c)
            assert sol_c.dtype == jnp.complex64
            bwd_c = float(
                jnp.linalg.norm(mat_c @ sol_c - rhs_c)
                / (frob_c * jnp.linalg.norm(sol_c) + jnp.linalg.norm(rhs_c))
            )
            assert bwd_c <= mult * bwd_c_bound

    def test_dense_tangent_solve_factors_operator_once_under_batched_rhs_and_jacrev(
        self,
    ) -> None:
        """When only the RHS is batched (vmap over RHS or jacrev of custom_root),
        dense_tangent_solve must materialize and LU-factor an unbatched (n, n) Jacobian."""
        mat_16 = jnp.eye(16, dtype=jnp.float32) * 2.0 + 0.1 * jnp.ones((16, 16), dtype=jnp.float32)
        rhs_batch = jnp.ones((8, 16), dtype=jnp.float32)

        jaxpr_vmap = jax.make_jaxpr(
            jax.vmap(lambda b: dense_tangent_solve(lambda x: mat_16 @ x, b))
        )(rhs_batch)
        vmap_lu_shapes = _collect_lu_input_shapes(jaxpr_vmap)
        assert vmap_lu_shapes
        assert all(shape == (16, 16) for shape in vmap_lu_shapes)

        def root_fn(theta: jax.Array) -> jax.Array:
            return jax.lax.custom_root(
                lambda y: mat_16 @ y - theta,
                jnp.zeros_like(theta),
                lambda _f, _x0: jnp.linalg.solve(mat_16, theta),
                dense_tangent_solve,
            )

        jaxpr_jacrev = jax.make_jaxpr(jax.jacrev(root_fn))(jnp.ones(16, dtype=jnp.float32))
        jacrev_lu_shapes = _collect_lu_input_shapes(jaxpr_jacrev)
        assert jacrev_lu_shapes
        assert all(shape == (16, 16) for shape in jacrev_lu_shapes)
