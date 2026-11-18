"""Verification suite for pure-JAX stiff Rosenbrock-Wanner and ESDIRK ODE solvers.

Covers three verification pillars across ``Rodas4``, ``Ros34PW2``, and ``Kvaerno3``:
1. Numerical and theoretical accuracy in ``float32`` and ``float64`` for both primal
   trajectories and reverse-mode gradients (bit-exact steady-state invariance,
   Robertson chemical kinetics mass conservation over ``[0, 1e5]`` in ``float32`` &
   ``float64``, SciPy ``Radau`` parity, empirical convergence orders via
   ``calibrax.analysis.scaling_fit`` on smooth non-autonomous, Kaps ``eps=1e-3``, and
   Prothero-Robinson ``lam=-1e4`` benchmarks, pyadjoint Taylor remainder rate >= 1.9,
   ``checkpoint=True`` vs ``checkpoint=False`` adjoint parity, and ``float32``-vs-``float64``
   value + gradient bounds).
2. JAX and ``flax.nnx`` transformation compatibility (``jit``/``vmap`` ULP gradient
   agreement, ``expect_compiles`` single-trace caching, ``check_input_gradients``, and
   end-to-end ``@nnx.jit`` + ``nnx.Optimizer`` Neural Stiff UDE training).
3. Speed, FLOP accounting, and architectural contracts (batched ``vmap`` ensemble
   simulation timed via ``calibrax.profiling.time_calls`` + ``FlopsCounter``, zero
   ``equinox``/``diffrax`` imports, and ``opifex.core.solver`` integration).
"""

import ast
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
from scipy.integrate import solve_ivp
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


def test_zero_equinox_imports_and_capability_contracts() -> None:
    """Module must have zero equinox/diffrax imports and register all solver capabilities."""
    source = Path(stiff_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"equinox", "diffrax", "optimistix", "lineax"}
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots.isdisjoint(banned)
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
def test_robertson_mass_conservation_and_scipy_radau_parity(
    dtype: jnp.dtype,
    factory: type,
) -> None:
    """Robertson stiff benchmark conserves mass in float32 & float64 and matches Radau."""
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

        if enable_x64:
            radau = solve_ivp(
                lambda _t, u: np.asarray(_robertson_rhs(jnp.array(0.0), jnp.asarray(u), rates)),
                (0.0, 1e5),
                [1.0, 0.0, 0.0],
                method="Radau",
                t_eval=np.asarray(sol.fields["t"]),
                rtol=1e-12,
                atol=1e-14,
            )
            max_err = float(np.max(np.abs(np.asarray(traj) - radau.y.T)))
            assert max_err < 1.5e-3


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
