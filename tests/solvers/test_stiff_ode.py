"""Verification suite for pure-JAX stiff Rosenbrock-Wanner and ESDIRK ODE solvers.

Covers three verification pillars across ``Rodas4``, ``Ros34PW2``, and ``Kvaerno3``:
1. Numerical and theoretical accuracy in ``float32`` and ``float64`` for both primal
   trajectories and reverse-mode gradients (bit-exact steady-state invariance,
   Robertson chemical kinetics mass conservation over ``[0, 1e5]`` in ``float32`` &
   ``float64``, exact matrix-exponential ``expm(A) y0`` trajectories and analytical
   rate/initial-state gradients, empirical convergence orders via
   ``calibrax.analysis.scaling_fit`` on smooth non-autonomous, Kaps ``eps=1e-3``, and
   Prothero-Robinson ``lam=-1e4`` benchmarks, pyadjoint Taylor remainder rate >= 1.9,
   ``checkpoint=True`` vs ``checkpoint=False`` adjoint parity, and ``float32``-vs-``float64``
   value + gradient bounds).
2. JAX and ``flax.nnx`` transformation compatibility (``jit``/``vmap`` ULP gradient
   agreement, ``expect_compiles`` single-trace caching, ``check_input_gradients``, and
   end-to-end ``@nnx.jit`` + ``nnx.Optimizer`` Neural Stiff UDE training).
3. Speed, FLOP accounting, and architectural contracts (batched ``vmap`` ensemble
   simulation timed via ``calibrax.profiling.time_calls`` + ``FlopsCounter``, declared
   import isolation, and ``opifex.core.solver`` integration).
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
from calibrax.analysis import scaling_fit
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
    VMAP_GRADIENT_ULP,
)

import opifex.solvers.stiff_ode as stiff_mod
from opifex.capabilities import UQ_CAPABILITIES
from opifex.core.solver import (
    IncompatibleProblemError,
    Method,
    Problem,
    Snapshots,
    solve,
    Status,
)
from opifex.core.solver.report import timed_solve
from opifex.solvers import (
    InitialValueProblem,
    Kvaerno3,
    Rodas4,
    Ros34PW2,
    solve_stiff_ode,
    SOLVER_CAPABILITIES,
)
from opifex.uncertainty.scientific.solutions import aggregate_solver_solutions


class _ForeignProblem(Problem):
    """Unrelated problem type used to verify runtime problem-family rejection."""

    x: jax.Array


def _robertson_rhs(_t: jax.Array, y: jax.Array, rates: jax.Array) -> jax.Array:
    """Classic Robertson (1966) autocatalytic reaction kinetics (stiffness ~ 1e11)."""
    k1, k2, k3 = rates[0], rates[1], rates[2]
    y1, y2, y3 = y[0], y[1], y[2]
    dy1 = -k1 * y1 + k3 * y2 * y3
    dy2 = k1 * y1 - k2 * (y2 * y2) - k3 * y2 * y3
    dy3 = k2 * (y2 * y2)
    return jnp.stack([dy1, dy2, dy3])


class _StiffNeuralResidual(nnx.Module):
    """Neural residual coupled to a stiff linear sink for Neural Stiff UDE tests."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.fc1 = nnx.Linear(2, 8, rngs=rngs)
        self.fc2 = nnx.Linear(8, 2, rngs=rngs)

    def __call__(self, y: jax.Array) -> jax.Array:
        return self.fc2(nnx.tanh(self.fc1(y)))


ALL_STIFF_FACTORIES = [
    pytest.param(Rodas4, id="Rodas4"),
    pytest.param(Ros34PW2, id="Ros34PW2"),
    pytest.param(Kvaerno3, id="Kvaerno3"),
]


def test_stiff_ode_import_isolation_and_capability_contracts() -> None:
    """Module imports only declared JAX/Flax/Optax/Opifex foundations and registers capabilities."""
    source = Path(stiff_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    allowed_roots = {
        "__future__",
        "collections",
        "dataclasses",
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
    for obj in (InitialValueProblem, Rodas4, Ros34PW2, Kvaerno3, solve_stiff_ode):
        assert typing.get_type_hints(obj)
    for key in ("solver:Rodas4", "solver:Ros34PW2", "solver:Kvaerno3"):
        assert key in SOLVER_CAPABILITIES
        assert key in UQ_CAPABILITIES

    with pytest.raises(IncompatibleProblemError, match="InitialValueProblem"):
        solve(_ForeignProblem(x=jnp.ones((2,))), Rodas4())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method", "pattern"),
    [
        (Rodas4(num_steps=0), "at least 1"),
        (Ros34PW2(grid="chebyshev"), "grid"),  # type: ignore[arg-type]
        (Rodas4(grid="geometric", geometric_ratio=0.0), "geometric_ratio"),
        (Kvaerno3(newton_iterations=0), "newton_iterations"),
    ],
)
def test_invalid_schedule_rejected_eagerly(
    method: Method[InitialValueProblem],
    pattern: str,
) -> None:
    """Invalid static schedule parameters fail fast before tracing."""
    prob = InitialValueProblem(
        vector_field=lambda _t, y, _a: -y,
        y0=jnp.ones((2,)),
        t0=jnp.array(0.0),
        t1=jnp.array(1.0),
    )
    with pytest.raises(ValueError, match=pattern):
        solve(prob, method)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("factory", ALL_STIFF_FACTORIES)
def test_steady_state_preserved_bit_exactly(dtype: jnp.dtype, factory: type) -> None:
    """Zero vector field preserves any initial pytree bit-for-bit across steps."""
    with jax.enable_x64(dtype == jnp.float64):
        y0 = {"fast": jnp.array([1.25, -3.5], dtype=dtype), "slow": jnp.array([0.75], dtype=dtype)}
        sol = solve_stiff_ode(
            lambda _t, y, _a: jax.tree.map(jnp.zeros_like, y),
            y0,
            t0=jnp.asarray(0.0, dtype=dtype),
            t1=jnp.asarray(2.0, dtype=dtype),
            method=factory(num_steps=8, save=Snapshots(count=4, include_initial=True)),
        )
        assert int(sol.status) == int(Status.SUCCESS)
        assert sol.fields["y"]["fast"].shape == (5, 2)
        for step_idx in range(5):
            assert bool(jnp.array_equal(sol.fields["y"]["fast"][step_idx], y0["fast"]))
            assert bool(jnp.array_equal(sol.fields["y"]["slow"][step_idx], y0["slow"]))


def test_empirical_convergence_orders_via_calibrax_scaling_fit() -> None:
    """Verify Rodas4 classical order >= 3.9 and Ros34PW2/Kvaerno3 stiff orders."""
    with jax.enable_x64(True):
        # 1. Rodas4 classical 4th-order on non-autonomous coupled nonlinear system
        def smooth_rhs(t: jax.Array, y: jax.Array, _a: None) -> jax.Array:
            return jnp.stack(
                [
                    -y[0] * (y[1] - jnp.sin(t)),
                    jnp.cos(t) - 5.0 * (y[0] - jnp.exp(-2.0 * t)),
                ]
            )

        y0_smooth = jnp.array([1.0, 2.0], dtype=jnp.float64)
        y_exact_smooth = jnp.array([jnp.exp(-2.0), jnp.sin(1.0) + 2.0], dtype=jnp.float64)
        ns_rodas = [8, 16, 32, 64]
        errs_rodas = [
            float(
                jnp.linalg.norm(
                    solve_stiff_ode(
                        smooth_rhs,
                        y0_smooth,
                        t0=0.0,
                        t1=1.0,
                        method=Rodas4(num_steps=n),
                    ).fields["y"][-1]
                    - y_exact_smooth
                )
            )
            for n in ns_rodas
        ]
        fit_rodas = scaling_fit([float(n) for n in ns_rodas], errs_rodas)
        assert -fit_rodas.exponent >= 3.9

        # 2. Ros34PW2 & Kvaerno3 3rd-order on Kaps stiff singular perturbation (eps = 1e-3)
        eps_kaps = 1e-3

        def kaps_rhs(_t: jax.Array, y: jax.Array, _a: None) -> jax.Array:
            inv_e = 1.0 / eps_kaps
            return jnp.stack(
                [
                    -(inv_e + 2.0) * y[0] + inv_e * (y[1] * y[1]),
                    y[0] - y[1] - y[1] * y[1],
                ]
            )

        y0_kaps = jnp.array([1.0, 1.0], dtype=jnp.float64)
        y_exact_kaps = jnp.array([jnp.exp(-2.0), jnp.exp(-1.0)], dtype=jnp.float64)
        ns_stiff = [16, 32, 64, 128]
        for factory in (Ros34PW2, Kvaerno3):
            errs = [
                float(
                    jnp.linalg.norm(
                        solve_stiff_ode(
                            kaps_rhs,
                            y0_kaps,
                            t0=0.0,
                            t1=1.0,
                            method=factory(num_steps=n),
                        ).fields["y"][-1]
                        - y_exact_kaps
                    )
                )
                for n in ns_stiff
            ]
            fit_stiff = scaling_fit([float(n) for n in ns_stiff], errs)
            assert -fit_stiff.exponent >= 2.85

        # 3. Prothero-Robinson stiff non-autonomous order >= 1.95 (lam = -1e4)
        def prothero_rhs(t: jax.Array, y: jax.Array, _a: None) -> jax.Array:
            return -1e4 * (y - jnp.sin(t)) + jnp.cos(t)

        for factory in (Ros34PW2, Kvaerno3):
            errs_pr = [
                float(
                    jnp.linalg.norm(
                        solve_stiff_ode(
                            prothero_rhs,
                            jnp.array([0.0], dtype=jnp.float64),
                            t0=0.0,
                            t1=1.0,
                            method=factory(num_steps=n),
                        ).fields["y"][-1]
                        - jnp.array([jnp.sin(1.0)], dtype=jnp.float64)
                    )
                )
                for n in ns_stiff
            ]
            fit_pr = scaling_fit([float(n) for n in ns_stiff], errs_pr)
            assert -fit_pr.exponent >= 1.95


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("factory", ALL_STIFF_FACTORIES)
def test_robertson_mass_conservation_and_nonnegativity(
    dtype: jnp.dtype,
    factory: type,
) -> None:
    """Robertson stiff benchmark conserves exact invariant sum(y_i) == 1 in float32 & float64."""
    enable_x64 = dtype == jnp.float64
    num_steps = 256
    with jax.enable_x64(enable_x64):
        y0 = jnp.array([1.0, 0.0, 0.0], dtype=dtype)
        rates = jnp.array([0.04, 3e7, 1e4], dtype=dtype)
        method = factory(
            num_steps=num_steps,
            grid="geometric",
            geometric_ratio=1e9,
            save=Snapshots(count=16, include_initial=True),
        )
        sol = solve_stiff_ode(
            _robertson_rhs,
            y0,
            t0=jnp.asarray(0.0, dtype=dtype),
            t1=jnp.asarray(1e5, dtype=dtype),
            args=rates,
            method=method,
        )
        traj = sol.fields["y"]
        assert int(sol.status) == int(Status.SUCCESS)
        assert bool(jnp.all(traj >= -1e-7))

        eps = float(jnp.finfo(dtype).eps)
        mass_drift = float(jnp.max(jnp.abs(jnp.sum(traj, axis=-1) - 1.0)))
        assert mass_drift <= 16.0 * num_steps * eps


def _eval_stiff_stability_scalar(factory: type, z_pos: jax.Array) -> jax.Array:
    """Evaluate rational stability function ``R(-z_pos)`` via the normalized stage recurrence.

    Evaluates the lower-triangular stage recurrence directly rather than the closed form
    ``1 - z b^T (I + z A)^{-1} 1``, which loses ``O(eps * z)`` precision to cancellation
    at stiff ``z >> 1`` (Hairer & Wanner 1996, §IV.3 & §IV.7).
    """
    if factory in (Rodas4, Ros34PW2):
        tab = stiff_mod._RODAS4_TABLEAU if factory is Rodas4 else stiff_mod._ROS34PW2_TABLEAU
        inv_gamma = 1.0 / tab.gamma
        denom = inv_gamma + z_pos
        ks: list[jax.Array] = []
        for j in range(len(tab.b_weights)):
            a_sum = sum(
                (tab.a_rows[j][ell] * ks[ell] for ell in range(j)),
                start=jnp.zeros_like(z_pos),
            )
            c_sum = sum(
                (tab.c_rows[j][ell] * ks[ell] for ell in range(j)),
                start=jnp.zeros_like(z_pos),
            )
            ks.append((-z_pos * (1.0 + a_sum) + c_sum) / denom)
        return 1.0 + sum(
            (tab.b_weights[j] * ks[j] for j in range(len(tab.b_weights))),
            start=jnp.zeros_like(z_pos),
        )

    gamma = stiff_mod._KV_GAMMA
    denom = 1.0 + gamma * z_pos
    fs: list[jax.Array] = [-z_pos]
    for imp_idx in range(3):
        partial = 1.0 + sum(
            (stiff_mod._KV_LOWER[imp_idx][ell] * fs[ell] for ell in range(imp_idx + 1)),
            start=jnp.zeros_like(z_pos),
        )
        fs.append(-z_pos * partial / denom)
    return 1.0 + sum(
        (stiff_mod._KV_WEIGHTS[j] * fs[j] for j in range(4)),
        start=jnp.zeros_like(z_pos),
    )


@pytest.mark.parametrize("factory", ALL_STIFF_FACTORIES)
def test_robertson_mazzia_magherini_published_endpoint_convergence(factory: type) -> None:
    """Verify ROBER on [0, 1e11] against Mazzia & Magherini (2008, Table II.10.3).

    Checks status SUCCESS, mass conservation within accumulated LU rounding, componentwise
    relative error < 1 in all three species (ruling out early-stopped trajectories by ~5e7),
    and strictly decreasing maximum relative error across N in (256, 512, 1024).
    """
    with jax.enable_x64(True):
        y0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
        rates = jnp.array([0.04, 3e7, 1e4], dtype=jnp.float64)
        # Mazzia & Magherini, "Test Set for Initial Value Problem Solvers", Release 2.4
        # (Feb. 2008), Problem ROBER, Table II.10.3 at t = 1e11:
        y_ref = jnp.array(
            [0.2083340149701255e-7, 0.8333360770334713e-13, 0.9999999791665050],
            dtype=jnp.float64,
        )
        eps64 = float(jnp.finfo(jnp.float64).eps)
        max_rel_errs: list[float] = []

        for num_steps in (256, 512, 1024):
            sol = solve_stiff_ode(
                _robertson_rhs,
                y0,
                t0=0.0,
                t1=1e11,
                args=rates,
                method=factory(
                    num_steps=num_steps,
                    grid="geometric",
                    geometric_ratio=1e16,
                ),
            )
            assert int(sol.status) == int(Status.SUCCESS)
            yf = sol.fields["y"][-1]
            assert abs(float(jnp.sum(yf) - 1.0)) <= 16.0 * num_steps * eps64
            rel_vec = jnp.abs(yf - y_ref) / y_ref
            assert bool(jnp.all(rel_vec < 1.0))
            max_rel_errs.append(float(jnp.max(rel_vec)))

        assert max_rel_errs[2] < max_rel_errs[1] < max_rel_errs[0]


def _stiff_linear_propagator_oracles(
    factory: type,
    hs_actual: jax.Array,
    grid_ts: jax.Array,
    dtype: jnp.dtype,
    num_steps: int,
) -> dict[str, jax.Array | float]:
    """Evaluate exact discrete stability propagator and continuous telescoping bounds in float64."""
    with jax.enable_x64(True):
        rng = np.random.default_rng(5)
        q_np, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        q_64 = jnp.asarray(q_np, dtype=jnp.float64)
        q_work_64 = q_64.astype(dtype).astype(jnp.float64)
        rates_64 = jnp.array([1e4, 1e2, 1.0], dtype=jnp.float64)
        y0_64 = jnp.array([1.2, -0.7, 0.9], dtype=jnp.float64)
        y0_work_64 = y0_64.astype(dtype).astype(jnp.float64)
        w_64 = jnp.array([0.6, -1.1, 0.8], dtype=jnp.float64)
        w_work_64 = w_64.astype(dtype).astype(jnp.float64)

        a_64 = (q_64 * (-rates_64)) @ q_64.T
        expm_ref = jax.scipy.linalg.expm(a_64) @ y0_64
        y_exact_64 = q_64 @ (jnp.exp(-rates_64) * (q_64.T @ y0_64))
        assert float(jnp.linalg.norm(expm_ref - y_exact_64)) < 1e-13

        assert abs(float(_eval_stiff_stability_scalar(factory, jnp.float64(1e12)))) < 1e-10
        hs_64 = hs_actual.astype(jnp.float64)
        ts_start_64 = grid_ts[:-1].astype(jnp.float64)
        r_and_dr = jax.vmap(jax.value_and_grad(lambda z: _eval_stiff_stability_scalar(factory, z)))

        disc_modes: list[float] = []
        disc_drates: list[float] = []
        mode_deltas: list[float] = []
        rate_deltas: list[float] = []
        eps64 = float(jnp.finfo(jnp.float64).eps)

        for i in range(3):
            r_val = float(rates_64[i])
            z_steps = hs_64 * r_val
            r_vals, dr_dz = r_and_dr(z_steps)
            assert float(jnp.max(jnp.abs(r_vals))) <= 1.0 + 16.0 * eps64
            r_np = np.asarray(r_vals, dtype=np.float64)
            dr_np = np.asarray(dr_dz, dtype=np.float64)
            h_np = np.asarray(hs_64, dtype=np.float64)
            prefix = np.concatenate([[1.0], np.cumprod(r_np[:-1])])
            suffix = np.concatenate([np.cumprod(r_np[:0:-1])[::-1], [1.0]])
            p_mode = float(np.prod(r_np))
            dp_dr = float(np.sum(prefix * (dr_np * h_np) * suffix))
            disc_modes.append(p_mode)
            disc_drates.append(dp_dr)
            local_diff = np.abs(r_np - np.exp(-np.asarray(z_steps, dtype=np.float64)))
            damping = np.exp(-r_val * np.asarray(ts_start_64, dtype=np.float64))
            delta_i = float(np.sum(damping * local_diff))
            assert abs(p_mode - math.exp(-r_val)) <= delta_i + num_steps * 4.0 * eps64
            mode_deltas.append(delta_i)
            rate_deltas.append(abs(dp_dr - (-math.exp(-r_val))))

        p_vec = jnp.asarray(disc_modes, dtype=jnp.float64)
        dp_vec = jnp.asarray(disc_drates, dtype=jnp.float64)
        return {
            "q_64": q_64,
            "rates_64": rates_64,
            "y0_64": y0_64,
            "w_64": w_64,
            "y_exact_64": y_exact_64,
            "grad_y0_exact_64": q_64 @ (jnp.exp(-rates_64) * (q_64.T @ w_64)),
            "grad_rates_exact_64": -(q_64.T @ w_64) * (q_64.T @ y0_64) * jnp.exp(-rates_64),
            "y_disc_64": q_work_64 @ (p_vec * (q_work_64.T @ y0_work_64)),
            "grad_y0_disc_64": q_work_64 @ (p_vec * (q_work_64.T @ w_work_64)),
            "grad_r_disc_64": (q_work_64.T @ w_work_64) * (q_work_64.T @ y0_work_64) * dp_vec,
            "y0_norm": float(jnp.linalg.norm(y0_64)),
            "w_norm": float(jnp.linalg.norm(w_64)),
            "y_exact_norm": float(jnp.linalg.norm(y_exact_64)),
            "max_delta": max(mode_deltas),
            "max_rate_delta": max(rate_deltas),
        }


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("factory", ALL_STIFF_FACTORIES)
def test_stiff_linear_system_exact_expm_values_and_analytical_gradients(
    dtype: jnp.dtype,
    factory: type,
) -> None:
    r"""Verify coupled stiff linear IVP ``y' = Q diag(-rates) Q^T y`` against both its
    exact discrete rational stability propagator and continuous ``expm(A)``.
    """
    enable_x64 = dtype == jnp.float64
    num_steps = 192
    ratio = 1e5

    with jax.enable_x64(enable_x64):
        idx_arr = jnp.arange(num_steps + 1, dtype=jnp.int32)
        grid_ts = jax.vmap(
            lambda idx: stiff_mod._grid_time_at_index(
                idx,
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(1.0, dtype=dtype),
                num_steps=num_steps,
                grid="geometric",
                geometric_ratio=ratio,
            )
        )(idx_arr)
        hs_actual = jnp.diff(grid_ts)

    oracles = _stiff_linear_propagator_oracles(factory, hs_actual, grid_ts, dtype, num_steps)
    u_round = 0.5 * float(jnp.finfo(dtype).eps)
    disc_round = num_steps * 64.0 * u_round
    y0_norm = float(oracles["y0_norm"])
    w_norm = float(oracles["w_norm"])
    max_delta = float(oracles["max_delta"])
    max_rate_delta = float(oracles["max_rate_delta"])
    assert (max_delta + disc_round) * y0_norm < float(oracles["y_exact_norm"])

    with jax.enable_x64(enable_x64):
        q_mat = jnp.asarray(oracles["q_64"]).astype(dtype)
        rates = jnp.asarray(oracles["rates_64"]).astype(dtype)
        y0 = jnp.asarray(oracles["y0_64"]).astype(dtype)
        w_vec = jnp.asarray(oracles["w_64"]).astype(dtype)
        method = factory(num_steps=num_steps, grid="geometric", geometric_ratio=ratio)

        def linear_rhs(_t: jax.Array, y: jax.Array, r_in: jax.Array) -> jax.Array:
            return (q_mat * (-r_in)) @ (q_mat.T @ y)

        def scalar_obj(y_in: jax.Array, r_in: jax.Array) -> tuple[jax.Array, jax.Array]:
            sol = solve_stiff_ode(
                linear_rhs,
                y_in,
                t0=jnp.asarray(0.0, dtype=dtype),
                t1=jnp.asarray(1.0, dtype=dtype),
                args=r_in,
                method=method,
            )
            yf = sol.fields["y"][-1]
            return jnp.dot(yf, w_vec), yf

        (_, y_num), (g_y0, g_rates) = jax.value_and_grad(scalar_obj, argnums=(0, 1), has_aux=True)(
            y0, rates
        )

    with jax.enable_x64(True):
        y_num_64 = y_num.astype(jnp.float64)
        gy0_64 = g_y0.astype(jnp.float64)
        gr_64 = g_rates.astype(jnp.float64)

        assert float(jnp.linalg.norm(y_num_64 - oracles["y_disc_64"])) <= disc_round * y0_norm
        assert float(jnp.linalg.norm(gy0_64 - oracles["grad_y0_disc_64"])) <= disc_round * w_norm
        assert float(jnp.linalg.norm(gr_64 - oracles["grad_r_disc_64"])) <= (
            disc_round * w_norm * y0_norm
        )
        assert float(jnp.linalg.norm(y_num_64 - oracles["y_exact_64"])) <= (
            (max_delta + disc_round) * y0_norm
        )
        assert (
            float(jnp.linalg.norm(gy0_64 - oracles["grad_y0_exact_64"]))
            <= (max_delta + disc_round) * w_norm
        )
        assert float(jnp.linalg.norm(gr_64 - oracles["grad_rates_exact_64"])) <= (
            (max_rate_delta + disc_round) * w_norm * y0_norm
        )


@pytest.mark.parametrize("factory", ALL_STIFF_FACTORIES)
def test_robertson_taylor_remainder_central_diff_and_checkpoint_parity(factory: type) -> None:
    """Adjoint passes Taylor O(eps^2) rate >= 1.9, central diff, and checkpoint parity."""
    with jax.enable_x64(True):
        # Parameterize perturbations multiplicative-positively so y_i(0) > 0 strictly holds
        # (negative y_2(0) triggers finite-time Riccati blow-up in -k_2 y_2^2).
        base_y0 = jnp.array([0.9, 1e-4, 0.1], dtype=jnp.float64)
        rates = jnp.array([0.04, 3e7, 1e4], dtype=jnp.float64)
        weights = jnp.array([0.7, 1e4, 1.3], dtype=jnp.float64)

        def loss_ckpt(inputs: tuple[jax.Array, ...]) -> jax.Array:
            log_y, log_scale = inputs
            sol = solve_stiff_ode(
                _robertson_rhs,
                base_y0 * jnp.exp(log_y),
                t0=0.0,
                t1=10.0,
                args=rates * jnp.exp(log_scale),
                method=factory(
                    num_steps=96, grid="geometric", geometric_ratio=1e5, checkpoint=True
                ),
            )
            return jnp.dot(sol.fields["y"][-1], weights)

        def loss_no_ckpt(inputs: tuple[jax.Array, ...]) -> jax.Array:
            log_y, log_scale = inputs
            sol = solve_stiff_ode(
                _robertson_rhs,
                base_y0 * jnp.exp(log_y),
                t0=0.0,
                t1=10.0,
                args=rates * jnp.exp(log_scale),
                method=factory(
                    num_steps=96, grid="geometric", geometric_ratio=1e5, checkpoint=False
                ),
            )
            return jnp.dot(sol.fields["y"][-1], weights)

        zeros = jnp.zeros((3,), dtype=jnp.float64)
        _, taylor_rates = taylor_remainder_rates(loss_ckpt, (zeros, zeros), seed=23)
        assert min(taylor_rates) >= TAYLOR_RATE_FLOOR

        rel_gap = gradient_versus_central_difference(
            loss_ckpt,
            (zeros, zeros),
            step=1e-5,
            seed=29,
        )
        assert rel_gap < 1e-8

        g_ckpt = jax.grad(loss_ckpt)((zeros, zeros))
        g_no_ckpt = jax.grad(loss_no_ckpt)((zeros, zeros))
        # Rematerializing 96 stiff LU-solve steps allows XLA to reassociate per-step cotangents
        # within O(num_steps * eps64).
        for left, right in zip(g_ckpt, g_no_ckpt, strict=True):
            scale = max(float(jnp.max(jnp.abs(left))), 1.0)
            assert float(jnp.max(jnp.abs(left - right))) <= 96.0 * scale * float(
                jnp.finfo(jnp.float64).eps
            )

        check_grads(lambda s: loss_ckpt((zeros, s)), (zeros,), order=1, modes=("rev",))


@pytest.mark.parametrize("factory", ALL_STIFF_FACTORIES)
def test_float32_value_and_gradient_accuracy_against_float64_oracle(factory: type) -> None:
    """float32 Robertson state and rate gradients match float64 AD reference cleanly."""
    weights_np = np.array([0.8, 1e3, 1.2], dtype=np.float64)

    def make_loss(dtype: jnp.dtype):
        y0 = jnp.array([1.0, 0.0, 0.0], dtype=dtype)
        base_rates = jnp.array([0.04, 3e7, 1e4], dtype=dtype)
        w = jnp.asarray(weights_np, dtype=dtype)
        method = factory(num_steps=96, grid="geometric", geometric_ratio=1e5)

        def loss(log_scale: jax.Array) -> tuple[jax.Array, jax.Array]:
            sol = solve_stiff_ode(
                _robertson_rhs,
                y0,
                t0=jnp.asarray(0.0, dtype=dtype),
                t1=jnp.asarray(10.0, dtype=dtype),
                args=base_rates * jnp.exp(log_scale),
                method=method,
            )
            yf = sol.fields["y"][-1]
            return jnp.dot(yf, w), yf

        return loss

    with jax.enable_x64(True):
        (_, yf_64), grad_64 = jax.value_and_grad(make_loss(jnp.float64), has_aux=True)(
            jnp.zeros((3,), dtype=jnp.float64)
        )

    with jax.enable_x64(False):
        (_, yf_32), grad_32 = jax.value_and_grad(make_loss(jnp.float32), has_aux=True)(
            jnp.zeros((3,), dtype=jnp.float32)
        )

    val_rel = float(np.linalg.norm(np.asarray(yf_32) - np.asarray(yf_64)) / np.linalg.norm(yf_64))
    grad_rel = float(
        np.linalg.norm(np.asarray(grad_32) - np.asarray(grad_64)) / np.linalg.norm(grad_64)
    )
    assert val_rel < 5e-5
    assert grad_rel < 2e-4


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("factory", ALL_STIFF_FACTORIES)
def test_jit_and_vmap_gradient_ulp_agreement(dtype: jnp.dtype, factory: type) -> None:
    """jit(grad) vs eager grad and vmap(grad) vs loop grad agree within ULP budgets."""
    with jax.enable_x64(dtype == jnp.float64):
        y0 = jnp.array([1.0, 1e-4, 0.05], dtype=dtype)
        rates = jnp.array([0.04, 1e5, 1e3], dtype=dtype)
        method = factory(num_steps=64, grid="geometric", geometric_ratio=1e4)

        def loss(components: tuple[jax.Array, ...]) -> jax.Array:
            (y_in,) = components
            sol = solve_stiff_ode(
                _robertson_rhs,
                y_in,
                t0=jnp.asarray(0.0, dtype=dtype),
                t1=jnp.asarray(1.0, dtype=dtype),
                args=rates,
                method=method,
            )
            return jnp.sum(sol.fields["y"][-1] ** 2)

        jit_ulp, vmap_ulp = gradient_transform_agreement(loss, (y0,))
        assert jit_ulp <= JIT_GRADIENT_ULP
        assert vmap_ulp <= VMAP_GRADIENT_ULP


@pytest.mark.parametrize("factory", [Rodas4, Kvaerno3])
def test_nnx_neural_stiff_ude_training_single_compile(factory: type) -> None:
    """An nnx.Module inside a stiff UDE vector field trains under @nnx.jit with 1 compile."""
    y0 = jnp.array([1.0, 0.5], dtype=jnp.float32)
    method = factory(num_steps=24, save=Snapshots(count=6, include_initial=False))

    def true_stiff_ude(_t: jax.Array, y: jax.Array, _a: None) -> jax.Array:
        stiff_linear = jnp.stack([-250.0 * (y[0] - y[1]), -0.8 * y[1]])
        nonlinear_source = jnp.stack([0.0, 0.6 * jnp.tanh(y[0])])
        return stiff_linear + nonlinear_source

    target_traj = solve_stiff_ode(true_stiff_ude, y0, t0=0.0, t1=1.0, method=method).fields["y"]

    residual_net = _StiffNeuralResidual(rngs=nnx.Rngs(3))

    def ude_loss_for_module(mod: _StiffNeuralResidual, y_init: jax.Array) -> jax.Array:
        graphdef, state = nnx.split(mod)

        def rhs(_t: jax.Array, y: jax.Array, s: nnx.State) -> jax.Array:
            merged = nnx.merge(graphdef, s)
            return jnp.stack([-250.0 * (y[0] - y[1]), -0.8 * y[1]]) + merged(y)

        sol = solve_stiff_ode(rhs, y_init, t0=0.0, t1=1.0, args=state, method=method)
        return jnp.mean((sol.fields["y"] - target_traj) ** 2)

    check_parameter_gradients(residual_net, lambda m: ude_loss_for_module(m, y0), modes=("rev",))
    check_input_gradients(residual_net, ude_loss_for_module, y0, modes=("rev",))
    optimizer = nnx.Optimizer(residual_net, optax.adam(2e-2), wrt=nnx.Param)

    @nnx.jit
    def train_step(net: _StiffNeuralResidual, opt: nnx.Optimizer) -> jax.Array:
        graphdef, state = nnx.split(net)

        def loss_from_state(st: nnx.State) -> jax.Array:
            def learned_rhs(_t: jax.Array, y: jax.Array, s: nnx.State) -> jax.Array:
                mod = nnx.merge(graphdef, s)
                stiff_linear = jnp.stack([-250.0 * (y[0] - y[1]), -0.8 * y[1]])
                return stiff_linear + mod(y)

            sol = solve_stiff_ode(
                learned_rhs,
                y0,
                t0=0.0,
                t1=1.0,
                args=st,
                method=method,
            )
            return jnp.mean((sol.fields["y"] - target_traj) ** 2)

        loss_val, grads = jax.value_and_grad(loss_from_state)(state)
        opt.update(net, grads)
        return loss_val

    with expect_compiles(1):
        first_loss = train_step(residual_net, optimizer)
    with expect_compiles(0):
        later_losses = [train_step(residual_net, optimizer) for _ in range(10)]

    assert bool(jnp.isfinite(first_loss))
    assert float(later_losses[-1]) < float(first_loss)


def test_batched_ensemble_profile_report_and_solution_distribution() -> None:
    """256 Robertson trajectories compile once under jit(vmap), report, and aggregate."""
    batch_size = 256
    scales = jnp.linspace(0.8, 1.2, batch_size, dtype=jnp.float32)[:, None]
    rates_a = jnp.array([0.04, 3e7, 1e4], dtype=jnp.float32)[None, :] * scales
    rates_b = rates_a * 1.03
    y0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    method = Ros34PW2(num_steps=64, grid="geometric", geometric_ratio=1e6)

    @jax.jit
    def simulate_batch(rate_batch: jax.Array):
        return jax.vmap(
            lambda r: solve_stiff_ode(_robertson_rhs, y0, t0=0.0, t1=100.0, args=r, method=method)
        )(rate_batch)

    with expect_compiles(1):
        sols_a = simulate_batch(rates_a)
    with expect_compiles(0):
        sols_b = simulate_batch(rates_b)

    assert bool(jnp.all(sols_a.is_converged))
    assert sols_b.fields["y"].shape == (batch_size, 1, 3)

    flops = FlopsCounter().count(simulate_batch, rates_a)
    timing = time_calls(lambda: simulate_batch(rates_a), warmup=1, iterations=2)
    assert flops.total_flops > 0
    assert timing.median_sec > 0.0

    rep = timed_solve(
        lambda: solve_stiff_ode(
            _robertson_rhs, y0, t0=0.0, t1=100.0, args=rates_a[0], method=method
        ),
        warmup=1,
        iterations=2,
    )
    assert rep.converged
    sol_1 = solve_stiff_ode(_robertson_rhs, y0, t0=0.0, t1=100.0, args=rates_a[0], method=method)
    sol_2 = solve_stiff_ode(_robertson_rhs, y0, t0=0.0, t1=100.0, args=rates_a[-1], method=method)
    agg = aggregate_solver_solutions([sol_1, sol_2])
    assert agg.fields["y"].shape == (1, 3)
    assert agg.auxiliary_data["uq"]["variance"]["y"].shape == (1, 3)


@pytest.mark.parametrize("factory", ALL_STIFF_FACTORIES)
def test_three_group_y0_params_teval_central_finite_difference_parity(factory: type) -> None:
    """3-group (y0, params, t_eval) float64 AD matches central finite differences."""
    with jax.enable_x64(True):

        def nonauto_stiff_rhs(t: jax.Array, y: jax.Array, params: jax.Array) -> jax.Array:
            k_fast, k_slow, forcing = params[0], params[1], params[2]
            return jnp.stack(
                [
                    -k_fast * (y[0] - jnp.cos(forcing * t)) + 0.25 * y[1],
                    -k_slow * y[1] + 0.4 * jnp.sin(forcing * t) * y[0],
                ]
            )

        y0 = jnp.array([1.15, -0.65], dtype=jnp.float64)
        params = jnp.array([40.0, 1.5, 0.9], dtype=jnp.float64)
        t_bounds = jnp.array([0.05, 1.25], dtype=jnp.float64)
        weights = jnp.array(
            [[0.3, -0.2], [0.5, 0.4], [-0.6, 0.8], [0.9, -0.5], [1.1, 0.7]],
            dtype=jnp.float64,
        )
        method = factory(num_steps=64, save=Snapshots(count=4, include_initial=True))

        def loss_three_groups(inputs: tuple[jax.Array, ...]) -> jax.Array:
            y_in, p_in, tb_in = inputs
            sol = solve_stiff_ode(
                nonauto_stiff_rhs,
                y_in,
                t0=tb_in[0],
                t1=tb_in[1],
                args=p_in,
                method=method,
            )
            return jnp.sum(sol.fields["y"] * weights) + 0.1 * jnp.sum(sol.fields["t"] ** 2)

        rel_gap = gradient_versus_central_difference(
            loss_three_groups,
            (y0, params, t_bounds),
            step=1e-5,
            seed=41,
        )
        assert rel_gap < 1e-8

        ad_y0, ad_p, ad_tb = jax.grad(loss_three_groups)((y0, params, t_bounds))
        eps = 1e-5
        fd_tb = jnp.stack(
            [
                (
                    loss_three_groups((y0, params, t_bounds.at[idx].add(eps)))
                    - loss_three_groups((y0, params, t_bounds.at[idx].add(-eps)))
                )
                / (2.0 * eps)
                for idx in range(t_bounds.shape[0])
            ]
        )
        assert bool(jnp.all(jnp.isfinite(ad_y0)))
        assert bool(jnp.all(jnp.isfinite(ad_p)))
        assert float(jnp.max(jnp.abs(ad_tb - fd_tb))) < 1e-7
