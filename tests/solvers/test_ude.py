"""Tests for Universal Differential Equation (UDE) and pure-JAX Tsit5 ODE solver.

TDD tests for the UDE framework that combines known physics with neural network
residual terms, integrated via pure-JAX adaptive 7-stage FSAL Tsitouras (2011)
Runge-Kutta with 5th-order polynomial dense output and ``jax.checkpoint``.

Reference: Rackauckas et al. 2020 — "Universal Differential Equations
for Scientific Machine Learning"
"""

import ast
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from substrax.testing import TraceCounter
from substrax.testing.gradients import check_input_gradients, check_parameter_gradients
from substrax.testing.nnx_conformance import Case, CHECKS, module_classes
from tests.fields.staggered_support import gradient_versus_central_difference

import opifex.solvers as solvers_pkg
import opifex.solvers.ude as ude_mod
from opifex.capabilities import UQ_CAPABILITIES
from opifex.core.solver.solution import Solution
from opifex.core.solver.status import is_successful, Status
from opifex.solvers import SOLVER_CAPABILITIES
from opifex.solvers.ude import (
    create_ude,
    NeuralODE,
    tsit5_solve,
    UDEConfig,
    UDESolver,
)


@pytest.fixture
def rngs():
    return nnx.Rngs(0)


# =========================================================================
# UDEConfig Tests
# =========================================================================


class TestUDEConfig:
    """Test UDE configuration."""

    def test_default_config(self):
        """Default config exposes standard step-control tolerances and step limit."""
        config = UDEConfig()
        assert config.dt0 == 0.01
        assert config.rtol == 1e-5
        assert config.atol == 1e-5
        assert config.max_steps == 4096
        assert config.dtmin is None
        assert config.force_dtmin is True
        assert config.adaptive is True

    def test_config_frozen(self):
        """Config should be immutable."""
        config = UDEConfig()
        with pytest.raises(AttributeError):
            config.dt0 = 0.1  # type: ignore[misc]

    def test_custom_config(self):
        """Custom values should be accepted."""
        config = UDEConfig(dt0=0.001, rtol=1e-8, atol=1e-8, max_steps=16384)
        assert config.dt0 == 0.001
        assert config.rtol == 1e-8


# =========================================================================
# NeuralODE Tests
# =========================================================================


class TestNeuralODE:
    """Test NeuralODE wrapper integrated with pure-JAX adaptive Tsit5."""

    def test_init(self, rngs):
        """NeuralODE should initialize with a neural network."""
        net = nnx.Linear(2, 2, rngs=rngs)
        node = NeuralODE(net=net, state_dim=2, rngs=rngs)
        assert node.state_dim == 2

    def test_vector_field_signature(self, rngs):
        """vector_field(t, y, args) should return same shape as y."""
        net = nnx.Linear(2, 2, rngs=rngs)
        node = NeuralODE(net=net, state_dim=2, rngs=rngs)
        y = jnp.ones(2)
        dy = node.vector_field(0.0, y, None)
        assert dy.shape == y.shape

    def test_solve_shape(self, rngs):
        """solve should return a Solution carrying trajectory of correct shape."""
        net = nnx.Linear(2, 2, rngs=rngs)
        node = NeuralODE(net=net, state_dim=2, rngs=rngs)
        assert not hasattr(node, "solve_with_status")
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0.0, 1.0, 10)
        sol = node.solve(y0, ts)
        assert isinstance(sol, Solution)
        assert bool(is_successful(sol.status))
        assert sol.fields["y"].shape == (10, 2)

    def test_solve_finite(self, rngs):
        """Solve output should be finite."""
        net = nnx.Linear(2, 2, rngs=rngs)
        node = NeuralODE(net=net, state_dim=2, rngs=rngs)
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0.0, 1.0, 5)
        sol = node.solve(y0, ts)
        assert jnp.all(jnp.isfinite(sol.fields["y"]))

    def test_jit_compatible(self, rngs):
        """NeuralODE solve should be JIT-compatible and return Solution PyTree."""
        net = nnx.Linear(2, 2, rngs=rngs)
        node = NeuralODE(net=net, state_dim=2, rngs=rngs)
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0.0, 1.0, 5)

        @nnx.jit
        def solve_fn(model, y0, ts):
            return model.solve(y0, ts)

        result = solve_fn(node, y0, ts)
        assert isinstance(result, Solution)
        assert result.fields["y"].shape == (5, 2)
        assert jnp.all(jnp.isfinite(result.fields["y"]))

    def test_gradient_flow(self, rngs):
        """Gradients should flow through pure-JAX Tsit5 solve."""
        net = nnx.Linear(2, 2, rngs=rngs)
        node = NeuralODE(net=net, state_dim=2, rngs=rngs)
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0.0, 1.0, 5)

        @nnx.jit
        def loss_fn(model):
            traj = model.solve(y0, ts).fields["y"]
            return jnp.mean(traj**2)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(node)
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, "__len__"))
        assert has_nonzero


# =========================================================================
# UDESolver Tests
# =========================================================================


class TestUDESolver:
    """Test the UDE solver combining known dynamics + neural residual."""

    def test_init(self, rngs):
        """UDESolver should initialize with known dynamics and neural net."""

        def known_dynamics(t, y):
            return -y  # Simple decay

        net = nnx.Linear(2, 2, rngs=rngs)
        solver = UDESolver(
            known_dynamics_fn=known_dynamics,
            neural_residual=net,
            state_dim=2,
            rngs=rngs,
        )
        assert solver.state_dim == 2

    def test_vector_field_combines(self, rngs):
        """vector_field should return known(t,y) + neural(y)."""

        def known_dynamics(t, y):
            return jnp.ones_like(y)  # Constant drift

        net = nnx.Linear(2, 2, rngs=rngs)
        solver = UDESolver(
            known_dynamics_fn=known_dynamics,
            neural_residual=net,
            state_dim=2,
            rngs=rngs,
        )
        y = jnp.ones(2)
        dy = solver.vector_field(0.0, y, None)
        assert dy.shape == (2,)

        # It should be known + neural, not just one of them
        known_part = known_dynamics(0.0, y)
        neural_part = net(y[None, :])[0]
        expected = known_part + neural_part
        assert jnp.allclose(dy, expected, atol=1e-6)

    def test_solve_shape(self, rngs):
        """UDE solve should return Solution carrying trajectory."""

        def known_dynamics(t, y):
            return -0.1 * y

        net = nnx.Linear(2, 2, rngs=rngs)
        solver = UDESolver(
            known_dynamics_fn=known_dynamics,
            neural_residual=net,
            state_dim=2,
            rngs=rngs,
        )
        assert not hasattr(solver, "solve_with_status")
        y0 = jnp.array([1.0, 0.5])
        ts = jnp.linspace(0.0, 1.0, 10)
        sol = solver.solve(y0, ts)
        assert isinstance(sol, Solution)
        assert bool(is_successful(sol.status))
        assert sol.fields["y"].shape == (10, 2)

    def test_zero_neural_matches_known(self, rngs):
        """With zero neural weights, UDE should approximate known ODE.

        Reference: This is the key UDE correctness check — when NN = 0,
        the solution should match the pure known dynamics.
        """

        def exponential_decay(t, y):
            return -y  # Exact solution: y(t) = y0 * exp(-t)

        # Create a network with zero output
        net = nnx.Linear(2, 2, rngs=rngs)
        # Zero out the weights and bias
        net.kernel = nnx.Param(jnp.zeros_like(net.kernel[...]))
        net.bias = nnx.Param(jnp.zeros_like(net.bias[...]))  # type: ignore[reportOptionalSubscript]

        solver = UDESolver(
            known_dynamics_fn=exponential_decay,
            neural_residual=net,
            state_dim=2,
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 2.0])
        ts = jnp.linspace(0.0, 1.0, 20)
        sol = solver.solve(y0, ts)
        traj = sol.fields["y"]

        # Hairer, Nørsett & Wanner (2nd ed., §II.3) contractive global error bound:
        # y' = -y is contractive (mu_2(-I) = -1 <= 0); RMS step control over d=2 components
        # bounds each local step error by sqrt(2) * (atol + rtol * max|y|).
        exact = y0[None, :] * jnp.exp(-ts[:, None])
        n_steps = max(int(sol.stats["num_steps"]), 1)
        cfg = solver.config
        err_bound = math.sqrt(2.0) * n_steps * (cfg.atol + cfg.rtol * float(jnp.max(jnp.abs(y0))))
        max_err = float(jnp.max(jnp.abs(traj - exact)))
        assert max_err <= err_bound, f"Max error {max_err} exceeded derived bound {err_bound}"

    def test_trajectory_loss_scalar(self, rngs):
        """trajectory_loss should return scalar MSE."""

        def known_dynamics(t, y):
            return -y

        net = nnx.Linear(2, 2, rngs=rngs)
        solver = UDESolver(
            known_dynamics_fn=known_dynamics,
            neural_residual=net,
            state_dim=2,
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 0.5])
        t_obs = jnp.linspace(0.0, 1.0, 5)
        y_obs = jnp.ones((5, 2))  # Target observations
        loss = solver.trajectory_loss(y0, t_obs, y_obs)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_trajectory_loss_gradient(self, rngs):
        """Gradients should flow through trajectory_loss."""

        def known_dynamics(t, y):
            return -y

        net = nnx.Linear(2, 2, rngs=rngs)
        solver = UDESolver(
            known_dynamics_fn=known_dynamics,
            neural_residual=net,
            state_dim=2,
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 0.5])
        t_obs = jnp.linspace(0.0, 1.0, 5)
        y_obs = jnp.ones((5, 2))

        @nnx.jit
        def loss_fn(model):
            return model.trajectory_loss(y0, t_obs, y_obs)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(solver)
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, "__len__"))
        assert has_nonzero

    def test_jit_compatible(self, rngs):
        """UDESolver.solve should be JIT compatible."""

        def known_dynamics(t, y):
            return -y

        net = nnx.Linear(2, 2, rngs=rngs)
        solver = UDESolver(
            known_dynamics_fn=known_dynamics,
            neural_residual=net,
            state_dim=2,
            rngs=rngs,
        )

        @nnx.jit
        def solve_jit(model, y0, ts):
            return model.solve(y0, ts)

        y0 = jnp.array([1.0, 0.5])
        ts = jnp.linspace(0.0, 1.0, 5)
        result = solve_jit(solver, y0, ts)
        assert isinstance(result, Solution)
        assert result.fields["y"].shape == (5, 2)

    def test_deterministic(self, rngs):
        """Same inputs should give same outputs."""

        def known_dynamics(t, y):
            return -y

        net = nnx.Linear(2, 2, rngs=rngs)
        solver = UDESolver(
            known_dynamics_fn=known_dynamics,
            neural_residual=net,
            state_dim=2,
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 0.5])
        ts = jnp.linspace(0.0, 1.0, 5)

        traj1 = solver.solve(y0, ts).fields["y"]
        traj2 = solver.solve(y0, ts).fields["y"]
        assert jnp.allclose(traj1, traj2, atol=1e-6)

    def test_custom_config(self, rngs):
        """UDESolver should accept custom config."""

        def known_dynamics(t, y):
            return -y

        config = UDEConfig(dt0=0.005, rtol=1e-8, atol=1e-8)
        net = nnx.Linear(2, 2, rngs=rngs)
        solver = UDESolver(
            known_dynamics_fn=known_dynamics,
            neural_residual=net,
            state_dim=2,
            config=config,
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 0.5])
        ts = jnp.linspace(0.0, 1.0, 5)
        traj = solver.solve(y0, ts).fields["y"]
        assert jnp.all(jnp.isfinite(traj))

    def test_higher_dim(self, rngs):
        """UDE should work with higher-dimensional state."""

        def lorenz(t, y):
            sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
            return jnp.array(
                [
                    sigma * (y[1] - y[0]),
                    y[0] * (rho - y[2]) - y[1],
                    y[0] * y[1] - beta * y[2],
                ]
            )

        net = nnx.Linear(3, 3, rngs=rngs)
        net.kernel = nnx.Param(jnp.zeros_like(net.kernel[...]))
        net.bias = nnx.Param(jnp.zeros_like(net.bias[...]))  # type: ignore[reportOptionalSubscript]

        solver = UDESolver(
            known_dynamics_fn=lorenz,
            neural_residual=net,
            state_dim=3,
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 1.0, 1.0])
        ts = jnp.linspace(0.0, 0.1, 5)
        traj = solver.solve(y0, ts).fields["y"]
        assert traj.shape == (5, 3)
        assert jnp.all(jnp.isfinite(traj))


# =========================================================================
# Factory Function Tests
# =========================================================================


class TestCreateUDE:
    """Test the create_ude factory function."""

    def test_factory_creates_solver(self, rngs):
        """Factory should create a working UDESolver."""

        def known_dynamics(t, y):
            return -y

        solver = create_ude(
            known_dynamics_fn=known_dynamics,
            state_dim=2,
            hidden_dims=[16, 16],
            rngs=rngs,
        )
        assert isinstance(solver, UDESolver)

    def test_factory_solve(self, rngs):
        """Factory-created UDE should solve."""

        def known_dynamics(t, y):
            return -y

        solver = create_ude(
            known_dynamics_fn=known_dynamics,
            state_dim=2,
            hidden_dims=[16, 16],
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 0.5])
        ts = jnp.linspace(0.0, 1.0, 5)
        traj = solver.solve(y0, ts).fields["y"]
        assert traj.shape == (5, 2)

    def test_factory_custom_config(self, rngs):
        """Factory should accept custom config."""

        def known_dynamics(t, y):
            return -y

        config = UDEConfig(dt0=0.005)
        solver = create_ude(
            known_dynamics_fn=known_dynamics,
            state_dim=2,
            hidden_dims=[16],
            config=config,
            rngs=rngs,
        )
        assert solver.config.dt0 == 0.005


class TestPureJaxTsit5Contracts:
    """Pure-JAX Tsit5 accuracy, transform, capability, and 3-group FD parity tests."""

    def test_solver_module_import_isolation_and_exports(self) -> None:
        """Module imports only declared JAX/Flax/Opifex foundations and exports unified symbols."""
        source = Path(ude_mod.__file__).read_text(encoding="utf-8")
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
        for key in ("solver:NeuralODE", "solver:UDESolver"):
            assert key in SOLVER_CAPABILITIES
            assert key in UQ_CAPABILITIES
        for symbol in (
            "InitialValueProblem",
            "Kvaerno3",
            "LevenbergMarquardt",
            "NeuralODE",
            "NonlinearLeastSquaresProblem",
            "Rodas4",
            "Ros34PW2",
            "UDEConfig",
            "UDESolver",
            "create_ude",
            "levenberg_marquardt",
            "solve_stiff_ode",
            "tsit5_solve",
        ):
            assert hasattr(solvers_pkg, symbol)

    def test_validation_status_mode_guard_and_vector_field_dispatch(self, rngs: nnx.Rngs) -> None:
        """Invalid configs/modes fail fast, budget exhaustion surfaces MAX_ITERS/NaN,
        and subclass vector_field overrides are dispatched."""
        with pytest.raises(ValueError, match="max_steps"):
            tsit5_solve(
                lambda _t, y, _a: -y,
                jnp.ones(2),
                jnp.linspace(0.0, 1.0, 3),
                config=UDEConfig(max_steps=0),
            )
        with pytest.raises(ValueError, match="strictly positive"):
            tsit5_solve(
                lambda _t, y, _a: -y,
                jnp.ones(2),
                jnp.linspace(0.0, 1.0, 3),
                config=UDEConfig(dt0=0.0),
            )
        with pytest.raises(ValueError, match="non-empty 1-D"):
            tsit5_solve(lambda _t, y, _a: -y, jnp.ones(2), jnp.zeros((2, 2)))

        solver = create_ude(
            known_dynamics_fn=lambda _t, y: -y,
            state_dim=2,
            hidden_dims=[8],
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 0.5], dtype=jnp.float32)
        ts = jnp.linspace(0.0, 1.0, 6, dtype=jnp.float32)
        sol_ok = solver.solve(y0, ts)
        assert int(sol_ok.status) == int(Status.SUCCESS)
        assert bool(sol_ok.is_converged)
        assert int(sol_ok.stats["num_steps"]) > 0

        # max_steps=8 on y' = -y over [0, 5] with dt0=0.01 exhausts budget -> MAX_ITERS & NaN loss
        decay_net = nnx.Linear(2, 2, use_bias=False, rngs=rngs)
        decay_net.kernel = nnx.Param(-jnp.eye(2, dtype=jnp.float32))
        zero_net = nnx.Linear(2, 2, rngs=rngs)
        zero_net.kernel = nnx.Param(jnp.zeros_like(zero_net.kernel[...]))
        zero_net.bias = nnx.Param(jnp.zeros_like(zero_net.bias[...]))  # type: ignore[reportOptionalSubscript]
        starved_node = NeuralODE(
            net=decay_net, state_dim=2, config=UDEConfig(dt0=0.01, max_steps=8), rngs=rngs
        )
        starved_ude = UDESolver(
            known_dynamics_fn=lambda _t, y: -y,
            neural_residual=zero_net,
            state_dim=2,
            config=UDEConfig(dt0=0.01, max_steps=8),
            rngs=rngs,
        )
        ts_long = jnp.linspace(0.0, 5.0, 6, dtype=jnp.float32)
        sol_starved = starved_ude.solve(y0, ts_long)
        assert not bool(is_successful(sol_starved.status))
        assert int(sol_starved.status) == int(Status.MAX_ITERS)
        assert not bool(is_successful(starved_node.solve(y0, ts_long).status))
        loss_starved = starved_ude.trajectory_loss(
            y0, ts_long, jnp.zeros((6, 2), dtype=jnp.float32)
        )
        assert bool(jnp.isnan(loss_starved))

        sol_nan = tsit5_solve(
            lambda _t, y, _a: -y,
            jnp.array([jnp.nan, 1.0]),
            ts,
        )
        assert int(sol_nan.status) == int(Status.NONFINITE)

        with pytest.raises(ValueError, match="1-D state vector"):
            tsit5_solve(lambda _t, y, _a: -y, jnp.ones((3, 2)), ts)
        with pytest.raises(ValueError, match="monotonically"):
            tsit5_solve(lambda _t, y, _a: -y, y0, jnp.array([0.0, 1.0, 0.5]))

    def test_traced_nonmonotone_ts_initial_failure_and_zero_span_grad(self) -> None:
        """Traced non-monotone ts returns Status.INITIAL_FAILURE and NaN under jit/jit(vmap)."""
        y0 = jnp.array([1.0, 0.5], dtype=jnp.float32)
        jitted_ts = jax.jit(lambda t_grid: tsit5_solve(lambda _t, y, _a: -y, y0, t_grid))
        bad_traced = jitted_ts(jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32))
        assert int(bad_traced.status) == int(Status.INITIAL_FAILURE)
        assert not bool(is_successful(bad_traced.status))
        assert bool(jnp.all(jnp.isnan(bad_traced.fields["y"])))
        assert int(bad_traced.stats["num_steps"]) == 0

        batch_ts = jnp.array([[0.0, 0.5, 1.0], [0.0, 1.0, 0.5], [1.0, 0.5, 0.0]], dtype=jnp.float32)
        batched_ts_sol = jax.jit(
            jax.vmap(lambda t_grid: tsit5_solve(lambda _t, y, _a: -y, y0, t_grid))
        )(batch_ts)
        assert int(batched_ts_sol.status[0]) == int(Status.SUCCESS)
        assert int(batched_ts_sol.status[1]) == int(Status.INITIAL_FAILURE)
        assert int(batched_ts_sol.status[2]) == int(Status.SUCCESS)
        assert bool(jnp.all(jnp.isfinite(batched_ts_sol.fields["y"][0])))
        assert bool(jnp.all(jnp.isnan(batched_ts_sol.fields["y"][1])))
        assert bool(jnp.all(jnp.isfinite(batched_ts_sol.fields["y"][2])))

        # Zero-span t1 == t0 preserves dy(t1)/dt1 |_{t1=t0} = f(t0, y0)
        grad_t1 = jax.grad(
            lambda t_end: jnp.sum(
                tsit5_solve(
                    lambda _t, y, _a: -2.0 * y,
                    jnp.array([1.5, -0.5], dtype=jnp.float32),
                    jnp.stack([jnp.float32(0.5), t_end]),
                ).fields["y"][-1]
            )
        )(jnp.float32(0.5))
        assert float(grad_t1) == pytest.approx(-2.0 * (1.5 - 0.5), abs=1e-5)

        sol_piecewise = tsit5_solve(
            lambda t, y, _a: jnp.where(t < 0.2, jnp.ones_like(y), -y),
            jnp.array([0.5, -0.25], dtype=jnp.float32),
            jnp.linspace(0.0, 1.0, 6, dtype=jnp.float32),
        )
        assert int(sol_piecewise.status) == int(Status.SUCCESS)

    def test_dropout_and_batchnorm_mode_validation_and_subclass_dispatch(
        self, rngs: nnx.Rngs
    ) -> None:
        """Dropout(0.2) and BatchNorm in .train() raise ValueError; .eval() and
        subclass vector_field overrides succeed and differentiate cleanly."""
        y0 = jnp.array([1.0, 0.5], dtype=jnp.float32)
        t_short = jnp.linspace(0.0, 0.2, 4, dtype=jnp.float32)

        class _DropoutVecField(nnx.Module):
            def __init__(self, r: nnx.Rngs) -> None:
                self.lin = nnx.Linear(2, 2, rngs=r)
                self.drop = nnx.Dropout(0.2, rngs=r)

            def __call__(self, x: jax.Array) -> jax.Array:
                return self.lin(self.drop(x))

        drop_node = NeuralODE(
            net=_DropoutVecField(nnx.Rngs(params=3, dropout=4)), state_dim=2, rngs=rngs
        )
        drop_node.train()
        with pytest.raises(ValueError, match="deterministic"):
            drop_node.solve(y0, t_short)
        drop_node.eval()
        sol_drop = drop_node.solve(y0, t_short)
        assert bool(is_successful(sol_drop.status))
        drop_grads = nnx.grad(lambda m: jnp.sum(m.solve(y0, t_short).fields["y"] ** 2))(drop_node)
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(drop_grads))

        class _BatchNormVecField(nnx.Module):
            def __init__(self, r: nnx.Rngs) -> None:
                self.lin = nnx.Linear(2, 2, rngs=r)
                self.bn = nnx.BatchNorm(2, rngs=r)

            def __call__(self, x: jax.Array) -> jax.Array:
                return self.lin(self.bn(x))

        bn_ude = UDESolver(
            known_dynamics_fn=lambda _t, y: -0.2 * y,
            neural_residual=_BatchNormVecField(nnx.Rngs(params=5)),
            state_dim=2,
            rngs=rngs,
        )
        bn_ude.train()
        with pytest.raises(ValueError, match="use_running_average"):
            bn_ude.solve(y0, t_short)
        bn_ude.eval()
        assert bool(is_successful(bn_ude.solve(y0, t_short).status))

        # Subclass overriding vector_field is dispatched by solve()
        class _ScaledNeuralODE(NeuralODE):
            def vector_field(
                self, t: float | jax.Array, y: jax.Array, args: object = None
            ) -> jax.Array:
                return 3.0 * super().vector_field(t, y, args)

        base_lin = nnx.Linear(2, 2, rngs=nnx.Rngs(9))
        sub_node = _ScaledNeuralODE(
            net=base_lin, state_dim=2, config=UDEConfig(dt0=0.01), rngs=rngs
        )
        plain_node = NeuralODE(net=base_lin, state_dim=2, config=UDEConfig(dt0=0.03), rngs=rngs)
        sol_sub = sub_node.solve(y0, jnp.array([0.0, 0.2], dtype=jnp.float32))
        # By exact autonomous RK time-rescaling invariance (tau = 3 t, dt0_tau = 3 dt0_t),
        # integrating 3 * f(y) on [0, 0.2] with dt0=0.01 executes the identical sequence of
        # stage states as integrating f(y) on [0, 0.6] with dt0=0.03 up to rounding.
        sol_plain_3x = plain_node.solve(y0, jnp.array([0.0, 0.6], dtype=jnp.float32))
        n_sub = max(int(sol_sub.stats["num_steps"]), 1)
        sub_round = 16.0 * n_sub * float(jnp.finfo(jnp.float32).eps) * float(jnp.linalg.norm(y0))
        assert (
            float(jnp.max(jnp.abs(sol_sub.fields["y"][-1] - sol_plain_3x.fields["y"][-1])))
            <= sub_round
        )

    def test_single_trace_jit_and_vmap_over_initial_states(self) -> None:
        """tsit5_solve compiles in a single trace under jit(vmap)."""
        ts = jnp.linspace(0.0, 1.0, 8, dtype=jnp.float32)
        cfg = UDEConfig(dt0=0.05, rtol=1e-5, atol=1e-6, max_steps=256)

        def run_one(y0: jax.Array) -> jax.Array:
            return tsit5_solve(
                lambda t, y, _a: -y * (1.0 + 0.1 * jnp.sin(t)), y0, ts, config=cfg
            ).fields["y"]

        counter = TraceCounter()
        batched_fn = jax.jit(counter.wrap(jax.vmap(run_one)))
        batch_a = jnp.stack([jnp.array([1.0, 0.5]), jnp.array([0.2, -0.8]), jnp.array([2.0, 1.5])])
        batch_b = batch_a * 0.75

        with counter.expect(new_traces=1):
            out_a = batched_fn(batch_a)
        with counter.expect(new_traces=0):
            out_b = batched_fn(batch_b)
        assert out_a.shape == (3, 8, 2)
        assert out_b.shape == (3, 8, 2)
        assert bool(jnp.all(jnp.isfinite(out_b)))

    def test_float64_fifth_order_accuracy_and_backward_integration(self) -> None:
        """Float64 integration preserves dtype and obeys Hairer-Nørsett-Wanner §II.3 bounds."""
        with jax.enable_x64(True):
            y0 = jnp.array([1.25, -0.75], dtype=jnp.float64)
            ts_fwd = jnp.linspace(0.0, 1.5, 16, dtype=jnp.float64)
            cfg = UDEConfig(dt0=0.05, rtol=1e-10, atol=1e-12, max_steps=1024)
            sol_fwd = tsit5_solve(lambda _t, y, _a: -y, y0, ts_fwd, config=cfg)
            exact_fwd = y0[None, :] * jnp.exp(-ts_fwd[:, None])
            assert sol_fwd.fields["y"].dtype == jnp.float64
            fwd_steps = max(int(sol_fwd.stats["num_steps"]), 1)
            fwd_bound = (
                math.sqrt(2.0) * fwd_steps * (cfg.atol + cfg.rtol * float(jnp.max(jnp.abs(y0))))
            )
            assert float(jnp.max(jnp.abs(sol_fwd.fields["y"] - exact_fwd))) <= fwd_bound

            # Backward integration of y' = -y over [1, 0] amplifies local errors by at most e^1.
            ts_bwd = jnp.linspace(1.0, 0.0, 11, dtype=jnp.float64)
            sol_bwd = tsit5_solve(lambda _t, y, _a: -y, y0, ts_bwd, config=cfg)
            exact_bwd = y0[None, :] * jnp.exp(-(ts_bwd[:, None] - 1.0))
            bwd_steps = max(int(sol_bwd.stats["num_steps"]), 1)
            bwd_bound = (
                math.e
                * math.sqrt(2.0)
                * bwd_steps
                * (cfg.atol + cfg.rtol * float(jnp.max(jnp.abs(exact_bwd))))
            )
            assert float(jnp.max(jnp.abs(sol_bwd.fields["y"] - exact_bwd))) <= bwd_bound

            # Non-uniform ts & multiple save points inside one large RK step on pure quadrature
            # y' = [cos(t), -sin(t)] (mu_2(0) = 0 -> unit state-transition norm).
            cfg_cos = UDEConfig(dt0=0.25, rtol=1e-11, atol=1e-13, max_steps=256)
            ts_multi_in_step = jnp.array(
                [0.0, 0.01, 0.025, 0.07, 0.12, 0.19, 0.20, 0.55, 1.10], dtype=jnp.float64
            )
            sol_cos = tsit5_solve(
                lambda t, _y, _a: jnp.array([jnp.cos(t), -jnp.sin(t)], dtype=jnp.float64),
                jnp.array([0.0, 1.0], dtype=jnp.float64),
                ts_multi_in_step,
                config=cfg_cos,
            )
            exact_cos = jnp.stack([jnp.sin(ts_multi_in_step), jnp.cos(ts_multi_in_step)], axis=-1)
            cos_steps = max(int(sol_cos.stats["num_steps"]), 1)
            cos_bound = math.sqrt(2.0) * cos_steps * (cfg_cos.atol + cfg_cos.rtol * 1.0)
            assert float(jnp.max(jnp.abs(sol_cos.fields["y"] - exact_cos))) <= cos_bound

    def test_three_group_y0_params_teval_central_finite_difference_parity(self) -> None:
        """3-group (y0, params, t_eval) float64 AD matches central finite differences."""
        with jax.enable_x64(True):
            y0 = jnp.array([1.1, -0.6], dtype=jnp.float64)
            w1 = jnp.array([[0.25, -0.15, 0.1], [-0.2, 0.3, -0.1]], dtype=jnp.float64)
            b1 = jnp.array([0.05, -0.04, 0.03], dtype=jnp.float64)
            w2 = jnp.array([[0.2, -0.1], [-0.15, 0.25], [0.1, 0.12]], dtype=jnp.float64)
            b2 = jnp.array([0.02, -0.03], dtype=jnp.float64)
            t_eval = jnp.array([0.05, 0.35, 0.80, 1.25], dtype=jnp.float64)
            obs_weights = jnp.array(
                [[0.4, -0.3], [-0.5, 0.7], [0.8, -0.6], [1.1, 0.5]],
                dtype=jnp.float64,
            )
            cfg = UDEConfig(dt0=0.02, rtol=1e-11, atol=1e-13, max_steps=1024)

            def ude_rhs(
                t: jax.Array,
                y: jax.Array,
                net_params: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
            ) -> jax.Array:
                p_w1, p_b1, p_w2, p_b2 = net_params
                known = jnp.stack(
                    [
                        -0.6 * y[0] * (1.0 + 0.2 * jnp.sin(t)) + 0.15 * y[1],
                        -0.4 * y[1] + 0.1 * jnp.cos(t) * y[0],
                    ]
                )
                residual = jnp.tanh(y @ p_w1 + p_b1) @ p_w2 + p_b2
                return known + 0.2 * residual

            def loss_three_groups(inputs: tuple[jax.Array, ...]) -> jax.Array:
                y_in, p_w1, p_b1, p_w2, p_b2, ts_in = inputs
                sol = tsit5_solve(
                    ude_rhs,
                    y_in,
                    ts_in,
                    config=cfg,
                    args=(p_w1, p_b1, p_w2, p_b2),
                )
                return jnp.sum(sol.fields["y"] * obs_weights)

            all_inputs = (y0, w1, b1, w2, b2, t_eval)
            rel_gap = gradient_versus_central_difference(
                loss_three_groups,
                all_inputs,
                step=1e-5,
                seed=37,
            )
            assert rel_gap < 1e-7

            grads = jax.grad(loss_three_groups)(all_inputs)
            eps = 1e-5
            fd_teval = jnp.stack(
                [
                    (
                        loss_three_groups((y0, w1, b1, w2, b2, t_eval.at[idx].add(eps)))
                        - loss_three_groups((y0, w1, b1, w2, b2, t_eval.at[idx].add(-eps)))
                    )
                    / (2.0 * eps)
                    for idx in range(t_eval.shape[0])
                ]
            )
            assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
            assert float(jnp.max(jnp.abs(grads[-1] - fd_teval))) < 1e-7


class TestSubstraxNnxConformanceAndGradients:
    """Verify every concrete nnx.Module in opifex.solvers.ude against substrax contracts."""

    @staticmethod
    def _build_cases() -> dict[str, Case]:
        cfg = UDEConfig(dt0=0.025, rtol=1e-8, atol=1e-10, max_steps=128)

        def _solve_batched(module: NeuralODE | UDESolver, y0_batch: jax.Array) -> jax.Array:
            ts = jnp.linspace(0.0, 0.25, 4, dtype=y0_batch.dtype)
            return jax.vmap(lambda y0: module.solve(y0, ts).fields["y"])(y0_batch)

        return {
            "opifex.solvers.ude._ResidualMLP": Case(
                build=lambda r: ude_mod._ResidualMLP(
                    [nnx.Linear(2, 6, rngs=r), nnx.Linear(6, 2, rngs=r)],
                    jnp.tanh,
                ),
                inputs=lambda k: (jax.random.normal(k, (4, 2)),),
            ),
            "opifex.solvers.ude.NeuralODE": Case(
                build=lambda r: NeuralODE(
                    net=nnx.Linear(2, 2, rngs=r),
                    state_dim=2,
                    config=cfg,
                    rngs=r,
                ),
                inputs=lambda k: (jax.random.normal(k, (3, 2)),),
                call=_solve_batched,
            ),
            "opifex.solvers.ude.UDESolver": Case(
                build=lambda r: create_ude(
                    known_dynamics_fn=lambda t, y: -0.5 * y * (1.0 + 0.1 * jnp.sin(t)),
                    state_dim=2,
                    hidden_dims=[6],
                    config=cfg,
                    activation=jnp.tanh,
                    rngs=r,
                ),
                inputs=lambda k: (jax.random.normal(k, (3, 2)),),
                call=_solve_batched,
            ),
        }

    def test_every_concrete_ude_module_satisfies_all_nnx_conformance_checks(self) -> None:
        """All concrete nnx.Module classes in opifex.solvers.ude satisfy substrax CHECKS."""
        cases = self._build_cases()
        assert set(cases) == set(module_classes(ude_mod))
        for qualname, case in cases.items():
            for check_name, (check_fn, applies) in CHECKS.items():
                if applies(case):
                    check_fn(case)
                else:
                    assert check_name in {"train_dropout_varies", "batch_stats"}, qualname

    def test_fwd_and_rev_directional_fd_gradients_including_float32_closures(
        self, rngs: nnx.Rngs
    ) -> None:
        """Parameter and input gradients match 1e-6 float64 FD in both fwd and rev modes."""
        cfg = UDEConfig(dt0=0.025, rtol=1e-9, atol=1e-11, max_steps=128)
        y0_f32 = jnp.array([1.0, -0.5], dtype=jnp.float32)
        t_obs_f32 = jnp.linspace(0.0, 0.25, 4, dtype=jnp.float32)
        y_obs_f32 = jnp.array(
            [[1.0, -0.5], [0.9, -0.4], [0.8, -0.35], [0.7, -0.3]],
            dtype=jnp.float32,
        )

        node = NeuralODE(net=nnx.Linear(2, 2, rngs=rngs), state_dim=2, config=cfg, rngs=rngs)
        ude = create_ude(
            known_dynamics_fn=lambda t, y: -0.4 * y + 0.1 * jnp.cos(t),
            state_dim=2,
            hidden_dims=[6],
            config=cfg,
            activation=jnp.tanh,
            rngs=rngs,
        )

        check_parameter_gradients(
            node,
            lambda m: jnp.mean((m.solve(y0_f32, t_obs_f32).fields["y"] - y_obs_f32) ** 2),
            modes=("fwd", "rev"),
        )
        check_input_gradients(
            node,
            lambda m, y0: jnp.sum(jnp.sin(m.solve(y0, t_obs_f32).fields["y"])),
            y0_f32,
            modes=("fwd", "rev"),
        )

        check_parameter_gradients(
            ude,
            lambda m: m.trajectory_loss(y0_f32, t_obs_f32, y_obs_f32),
            modes=("fwd", "rev"),
        )
        check_input_gradients(
            ude,
            lambda m, y0: m.trajectory_loss(y0, t_obs_f32, y_obs_f32),
            y0_f32,
            modes=("fwd", "rev"),
        )


class TestBoundedLoopStepControlsAndAnalyticalBounds:
    """Verify vmapped early exit, dense-save jaxpr structure, step controls, and convergence."""

    def test_vmap_early_exit_and_custom_vmap_transform_regression(self) -> None:
        """vmap halts at slowest lane (<600 RHS calls vs 24577 for max_steps=4096) and
        preserves unbatched values, gradients, and jvp(grad) within ULP rounding."""
        with jax.enable_x64(True):
            ts = jnp.array([0.0, 1.0], dtype=jnp.float64)
            y0 = jnp.array([1.0], dtype=jnp.float64)
            cfg = UDEConfig(dt0=0.01, rtol=1e-8, atol=1e-8, max_steps=4096)
            rates = jnp.array([1.0, 4.0, 16.0, 64.0], dtype=jnp.float64)

            call_box = [0]

            def counted_rhs(_t: jax.Array, y: jax.Array, k: jax.Array) -> jax.Array:
                jax.debug.callback(lambda: call_box.__setitem__(0, call_box[0] + 1))
                return -k * y

            sol_batch = jax.jit(
                jax.vmap(lambda k: tsit5_solve(counted_rhs, y0, ts, config=cfg, args=k))
            )(rates)
            jax.block_until_ready(sol_batch.fields["y"])
            assert bool(jnp.all(is_successful(sol_batch.status)))
            assert 40 <= call_box[0] < 600

            def final_y(k: jax.Array) -> jax.Array:
                return tsit5_solve(
                    lambda _t, y, rate: -rate * y,
                    y0,
                    ts,
                    config=cfg,
                    args=k,
                ).fields["y"][-1, 0]

            jaxpr_vmap = jax.make_jaxpr(jax.vmap(final_y))(rates)
            assert "cond[" in str(jaxpr_vmap)

            eps64 = float(jnp.finfo(jnp.float64).eps)
            unbatched_vals = jnp.stack([jax.jit(final_y)(r) for r in rates])
            unbatched_grads = jnp.stack([jax.jit(jax.grad(final_y))(r) for r in rates])

            assert float(jnp.max(jnp.abs(jax.jit(jax.vmap(final_y))(rates) - unbatched_vals))) <= (
                16.0 * eps64
            )
            rates_2d = rates.reshape(2, 2)
            assert float(
                jnp.max(
                    jnp.abs(
                        jax.jit(jax.vmap(jax.vmap(final_y)))(rates_2d)
                        - unbatched_vals.reshape(2, 2)
                    )
                )
            ) <= (16.0 * eps64)

            g_vmap = jax.jit(jax.vmap(jax.grad(final_y)))(rates)
            g_sum = jax.jit(jax.grad(lambda ks: jnp.sum(jax.vmap(final_y)(ks))))(rates)
            jvp_vmap = jax.jit(jax.vmap(lambda k: jax.jvp(final_y, (k,), (jnp.ones_like(k),))[1]))(
                rates
            )
            assert float(jnp.max(jnp.abs(g_vmap - unbatched_grads))) <= 16.0 * eps64
            assert float(jnp.max(jnp.abs(g_sum - unbatched_grads))) <= 16.0 * eps64
            assert float(jnp.max(jnp.abs(jvp_vmap - unbatched_grads))) <= 16.0 * eps64

            # Second-order AD: vmap(jvp(grad)) matches unbatched jvp(grad) to float64 rounding
            pair_rates = jnp.array([0.5, 1.5], dtype=jnp.float64)
            d2_unbatched = jnp.stack(
                [jax.jvp(jax.grad(final_y), (r,), (jnp.float64(1.0),))[1] for r in pair_rates]
            )
            d2_batch = jax.jit(
                jax.vmap(lambda k: jax.jvp(jax.grad(final_y), (k,), (jnp.ones_like(k),))[1])
            )(pair_rates)
            assert float(jnp.max(jnp.abs(d2_batch - d2_unbatched))) <= 32.0 * eps64

    def test_in_step_save_cursor_avoids_per_step_ntimes_interpolation_tensors(self) -> None:
        """At n_times=1024, dim=64, scan loop carry holds no 1024-leading tensor except
        ys (1024, 64), and scan equations materialize no (1024, 7) or (1024, 1) tensors."""
        n_times = 1024
        dim = 64
        y0 = jnp.ones(dim, dtype=jnp.float32)
        ts = jnp.linspace(0.0, 1.0, n_times, dtype=jnp.float32)
        cfg = UDEConfig(dt0=0.05, rtol=1e-4, atol=1e-5, max_steps=256)

        def _ntimes_shapes(vars_seq: object) -> set[tuple[int, ...]]:
            return {
                shape
                for var in vars_seq  # type: ignore[attr-defined]
                if (shape := getattr(getattr(var, "aval", None), "shape", ()))
                and shape[0] == n_times
            }

        def collect_scan_ntimes_shapes(
            closed_jaxpr: object, in_scan: bool = False
        ) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]]]:
            carry_shapes: set[tuple[int, ...]] = set()
            eqn_shapes: set[tuple[int, ...]] = set()
            core = getattr(closed_jaxpr, "jaxpr", closed_jaxpr)
            for eqn in getattr(core, "eqns", ()):
                if eqn.primitive.name == "scan":
                    carry_shapes |= _ntimes_shapes(eqn.outvars)
                if in_scan:
                    eqn_shapes |= _ntimes_shapes([*eqn.invars, *eqn.outvars])
                child_in_scan = in_scan or (eqn.primitive.name == "scan")
                for val in eqn.params.values():
                    items = val if isinstance(val, tuple | list) else (val,)
                    for item in items:
                        if hasattr(item, "eqns") or hasattr(item, "jaxpr"):
                            c_sub, e_sub = collect_scan_ntimes_shapes(item, child_in_scan)
                            carry_shapes |= c_sub
                            eqn_shapes |= e_sub
            return carry_shapes, eqn_shapes

        # Positive control: a scan carrying an (n_times, 7) tensor is flagged immediately.
        bad_jaxpr = jax.make_jaxpr(
            lambda buf: jax.lax.scan(lambda c, _: (c + 1.0, None), buf, xs=None, length=4)[0]
        )(jnp.zeros((n_times, 7), dtype=jnp.float32))
        bad_carry, bad_eqns = collect_scan_ntimes_shapes(bad_jaxpr)
        assert (n_times, 7) in bad_carry
        assert (n_times, 7) in bad_eqns

        fwd_fn = lambda y_in, t_in: tsit5_solve(lambda _t, y, _a: -0.5 * y, y_in, t_in, config=cfg)
        grad_fn = jax.grad(lambda y_in, t_in: jnp.sum(fwd_fn(y_in, t_in).fields["y"]))
        fwd_jaxpr = jax.make_jaxpr(fwd_fn)(y0, ts)
        grad_jaxpr = jax.make_jaxpr(grad_fn)(y0, ts)

        fwd_carry, fwd_eqns = collect_scan_ntimes_shapes(fwd_jaxpr)
        grad_carry, grad_eqns = collect_scan_ntimes_shapes(grad_jaxpr)
        assert fwd_carry == {(n_times, dim)}
        assert grad_carry == {(n_times, dim)}
        assert fwd_eqns <= {(n_times,), (n_times, dim)}
        assert grad_eqns <= {(n_times,), (n_times, dim)}

        y_shape = jax.ShapeDtypeStruct((dim,), jnp.float32)
        t_shape = jax.ShapeDtypeStruct((n_times,), jnp.float32)
        mem_fwd = jax.jit(fwd_fn).lower(y_shape, t_shape).compile().memory_analysis()
        mem_grad = jax.jit(grad_fn).lower(y_shape, t_shape).compile().memory_analysis()
        assert mem_fwd is not None and mem_fwd.temp_size_in_bytes > 0
        assert mem_grad is not None and mem_grad.temp_size_in_bytes > 0

    def test_step_size_controls_dtmin_and_ulp_spacing_guard(self) -> None:
        """Both dtmin (force_dtmin=False) and 10 * spacing(t) return Status.STEP_TOO_SMALL;
        force_dtmin=True clamps to dtmin and succeeds."""
        with pytest.raises(ValueError, match="dtmin"):
            tsit5_solve(
                lambda _t, y, _a: -y,
                jnp.ones(1),
                jnp.array([0.0, 1.0]),
                config=UDEConfig(dtmin=0.0),
            )
        with pytest.raises(TypeError, match="force_dtmin"):
            tsit5_solve(
                lambda _t, y, _a: -y,
                jnp.ones(1),
                jnp.array([0.0, 1.0]),
                config=UDEConfig(force_dtmin=1),  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="adaptive"):
            tsit5_solve(
                lambda _t, y, _a: -y,
                jnp.ones(1),
                jnp.array([0.0, 1.0]),
                config=UDEConfig(adaptive=1),  # type: ignore[arg-type]
            )

        # 1. dtmin hit with force_dtmin=False -> Status.STEP_TOO_SMALL
        sol_dtmin_halt = tsit5_solve(
            lambda _t, y, _a: -80.0 * y,
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([0.0, 1.0], dtype=jnp.float32),
            config=UDEConfig(
                dt0=0.1,
                rtol=1e-7,
                atol=1e-9,
                dtmin=0.05,
                force_dtmin=False,
                max_steps=4096,
            ),
        )
        assert int(sol_dtmin_halt.status) == int(Status.STEP_TOO_SMALL)

        # 2. dtmin hit with force_dtmin=True -> clamps step to dtmin and finishes with SUCCESS
        sol_dtmin_force = tsit5_solve(
            lambda _t, y, _a: -10.0 * y,
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([0.0, 0.5], dtype=jnp.float32),
            config=UDEConfig(
                dt0=0.1,
                rtol=1e-7,
                atol=1e-7,
                dtmin=0.05,
                force_dtmin=True,
                max_steps=4096,
            ),
        )
        assert int(sol_dtmin_force.status) == int(Status.SUCCESS)
        assert 1 <= int(sol_dtmin_force.stats["num_steps"]) <= 12

        # Crossing t = 0.125 in float32 raises min_step(t) from 7.45e-8 (< dtmin=1e-7) to
        # 1.49e-7 (> dtmin=1e-7); force_dtmin=True must clear at_dtmin and report STEP_TOO_SMALL.
        cfg_cross = UDEConfig(
            dt0=1e-7, rtol=1e-6, atol=1e-6, dtmin=1e-7, force_dtmin=True, max_steps=500
        )
        sol_cross = tsit5_solve(
            lambda t, _y, _a: jnp.array([1e9 * jnp.cos(1e9 * t)], dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([0.124999, 0.125020], dtype=jnp.float32),
            config=cfg_cross,
        )
        assert int(sol_cross.status) == int(Status.STEP_TOO_SMALL)

        # 3. Finite-time blow-up y' = y^2 over [0, 2] (pole at t = 1) shrinks below
        # floating-point min_step = 10 * spacing(t) -> Status.STEP_TOO_SMALL well before max_steps.
        for dtype in (jnp.float32, jnp.float64):
            with jax.enable_x64(dtype == jnp.float64):
                sol_blowup = tsit5_solve(
                    lambda _t, y, _a: y * y,
                    jnp.array([1.0], dtype=dtype),
                    jnp.array([0.0, 2.0], dtype=dtype),
                    config=UDEConfig(dt0=0.1, rtol=1e-6, atol=1e-6, max_steps=4096, dtmin=None),
                )
                assert int(sol_blowup.status) == int(Status.STEP_TOO_SMALL)
                assert int(sol_blowup.stats["num_steps"]) < 500

    def test_tsit5_fixed_step_fifth_order_convergence(self) -> None:
        """Verify 5th-order convergence (|slope - 5.0| < 0.2) across fixed-step halvings
        for endpoint values, Tsitouras (2011) 4th-degree continuous-extension dense output
        (local O(h^5) interpolation error -> global O(h^5) dense error; Hairer, Nørsett &
        Wanner 2008, §II.6, Theorem 6.2), and parameter gradients.
        """
        from calibrax.analysis import scaling_fit

        with jax.enable_x64(True):
            a_mat = jnp.array([[-0.3, -1.2], [1.2, -0.3]], dtype=jnp.float64)
            y0_rot = jnp.array([1.2, -0.5], dtype=jnp.float64)
            w_rot = jnp.array([0.6, -0.9], dtype=jnp.float64)
            # Uniform 101-point query grid samples fractional step phases theta in (0, 1)
            # uniformly across every dyadic step count N in (8, 16, 32, 64), avoiding
            # phase-aliasing bias on C(theta) h^5 when max is taken over only 4 points.
            ts_query = jnp.linspace(0.0, 1.0, 101, dtype=jnp.float64)
            exact_traj = jax.vmap(lambda t: jax.scipy.linalg.expm(t * a_mat) @ y0_rot)(ts_query)
            exact_grad_a = jax.grad(
                lambda mat: jnp.dot(jax.scipy.linalg.expm(1.0 * mat) @ y0_rot, w_rot)
            )(a_mat)

            ns_fixed = [8, 16, 32, 64]
            endpoint_errs: list[float] = []
            dense_l2_errs: list[float] = []
            dense_linf_errs: list[float] = []
            grad_errs: list[float] = []

            for n_steps in ns_fixed:
                dt = 1.0 / n_steps
                cfg_fixed = UDEConfig(dt0=dt, adaptive=False, max_steps=n_steps + 4)
                sol = tsit5_solve(
                    lambda _t, y, m: m @ y, y0_rot, ts_query, config=cfg_fixed, args=a_mat
                )
                assert int(sol.status) == int(Status.SUCCESS)
                assert int(sol.stats["num_steps"]) == n_steps
                endpoint_errs.append(float(jnp.linalg.norm(sol.fields["y"][-1] - exact_traj[-1])))
                pt_norms = jnp.linalg.norm(sol.fields["y"] - exact_traj, axis=-1)
                dense_l2_errs.append(float(jnp.sqrt(jnp.mean(pt_norms * pt_norms))))
                dense_linf_errs.append(float(jnp.max(pt_norms)))
                grad_a = jax.grad(
                    lambda mat, c=cfg_fixed: jnp.dot(
                        tsit5_solve(
                            lambda _t, y, m: m @ y, y0_rot, ts_query, config=c, args=mat
                        ).fields["y"][-1],
                        w_rot,
                    )
                )(a_mat)
                grad_errs.append(float(jnp.linalg.norm(grad_a - exact_grad_a)))

            ns_float = [float(n) for n in ns_fixed]
            assert abs((-scaling_fit(ns_float, endpoint_errs).exponent) - 5.0) < 0.2
            assert abs((-scaling_fit(ns_float, dense_l2_errs).exponent) - 5.0) < 0.2
            assert abs((-scaling_fit(ns_float, dense_linf_errs).exponent) - 5.0) < 0.2
            assert abs((-scaling_fit(ns_float, grad_errs).exponent) - 5.0) < 0.2

    def test_tsit5_adaptive_tolerance_proportionality_and_float32_contractive_bound(
        self,
    ) -> None:
        """Verify adaptive tolerance proportionality (|slope - 1.0| < 0.2) and float32 bound.

        Derivation of the adaptive tolerance-proportionality exponent:
        For an embedded 5(4) Runge-Kutta pair advancing the 5th-order solution under
        error-per-step control, the embedded 4th-order error estimate satisfies
        |e_4| = C_4 h^5 + O(h^6) ~ tol, so accepted step sizes scale as h ~ tol^(1/5).
        Accumulating N = T / h ~ tol^(-1/5) steps whose local 5th-order truncation error
        is O(h^6) yields global error O(N h^6) = O(h^5) ~ tol^1.0.
        """
        from calibrax.analysis import scaling_fit

        with jax.enable_x64(True):
            a_mat = jnp.array([[-0.3, -1.2], [1.2, -0.3]], dtype=jnp.float64)
            y0_rot = jnp.array([1.2, -0.5], dtype=jnp.float64)
            w_rot = jnp.array([0.6, -0.9], dtype=jnp.float64)
            tols = [1e-4, 1e-6, 1e-8, 1e-10]
            ts_adapt = jnp.linspace(0.0, 5.0, 21, dtype=jnp.float64)
            exact_adapt = jax.vmap(lambda t: jax.scipy.linalg.expm(t * a_mat) @ y0_rot)(ts_adapt)
            exact_grad_adapt = jax.grad(
                lambda mat: jnp.dot(jax.scipy.linalg.expm(5.0 * mat) @ y0_rot, w_rot)
            )(a_mat)
            adapt_val_errs: list[float] = []
            adapt_grad_errs: list[float] = []

            for tol in tols:
                cfg_tol = UDEConfig(dt0=0.05, rtol=tol, atol=tol, max_steps=4096)
                sol_tol = tsit5_solve(
                    lambda _t, y, m: m @ y, y0_rot, ts_adapt, config=cfg_tol, args=a_mat
                )
                assert int(sol_tol.status) == int(Status.SUCCESS)
                adapt_val_errs.append(
                    float(jnp.max(jnp.linalg.norm(sol_tol.fields["y"] - exact_adapt, axis=-1)))
                )
                g_tol = jax.grad(
                    lambda mat, c=cfg_tol: jnp.dot(
                        tsit5_solve(
                            lambda _t, y, m: m @ y, y0_rot, ts_adapt, config=c, args=mat
                        ).fields["y"][-1],
                        w_rot,
                    )
                )(a_mat)
                adapt_grad_errs.append(float(jnp.linalg.norm(g_tol - exact_grad_adapt)))

            assert abs(scaling_fit(tols, adapt_val_errs).exponent - 1.0) < 0.2
            assert abs(scaling_fit(tols, adapt_grad_errs).exponent - 1.0) < 0.2

        # Float32 endpoint contractive bound on y' = -0.7 y (2-point ts = [0, 5] has no
        # interior interpolation, so Hairer-Nørsett-Wanner §II.3 contractive sum holds).
        y0_32 = jnp.array([1.4, -0.9], dtype=jnp.float32)
        ts_32 = jnp.array([0.0, 5.0], dtype=jnp.float32)
        cfg_32 = UDEConfig(dt0=0.05, rtol=1e-5, atol=1e-5, max_steps=4096)
        sol_32 = tsit5_solve(
            lambda _t, y, k: -k * y, y0_32, ts_32, config=cfg_32, args=jnp.float32(0.7)
        )
        assert int(sol_32.status) == int(Status.SUCCESS)
        exact_end_32 = y0_32 * jnp.exp(jnp.float32(-3.5))
        n_32 = max(int(sol_32.stats["num_steps"]), 1)
        bound_32 = math.sqrt(2.0) * n_32 * (1e-5 + 1e-5 * float(jnp.max(jnp.abs(y0_32))))
        assert float(jnp.max(jnp.abs(sol_32.fields["y"][-1] - exact_end_32))) <= bound_32
