"""Tests for Universal Differential Equation (UDE) Solver.

TDD tests for the UDE framework that combines known physics with
neural network residual terms, integrated via diffrax.

Reference: Rackauckas et al. 2020 — "Universal Differential Equations
for Scientific Machine Learning"
Diffrax API: ODETerm, diffeqsolve, PIDController, Tsit5/Dopri5
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.solvers.ude import (
    create_ude,
    NeuralODE,
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
        """Default config should have sensible values."""
        config = UDEConfig()
        assert config.dt0 == 0.01
        assert config.rtol == 1e-5
        assert config.atol == 1e-5
        assert config.max_steps == 4096

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
    """Test NeuralODE wrapper that uses diffrax.diffeqsolve."""

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
        """solve should return trajectory of correct shape."""
        net = nnx.Linear(2, 2, rngs=rngs)
        node = NeuralODE(net=net, state_dim=2, rngs=rngs)
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0.0, 1.0, 10)
        trajectory = node.solve(y0, ts)
        assert trajectory.shape == (10, 2)

    def test_solve_finite(self, rngs):
        """Solve output should be finite."""
        net = nnx.Linear(2, 2, rngs=rngs)
        node = NeuralODE(net=net, state_dim=2, rngs=rngs)
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0.0, 1.0, 5)
        trajectory = node.solve(y0, ts)
        assert jnp.all(jnp.isfinite(trajectory))

    def test_jit_compatible(self, rngs):
        """NeuralODE solve should be JIT-compatible."""
        net = nnx.Linear(2, 2, rngs=rngs)
        node = NeuralODE(net=net, state_dim=2, rngs=rngs)
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0.0, 1.0, 5)

        @nnx.jit
        def solve_fn(model, y0, ts):
            return model.solve(y0, ts)

        result = solve_fn(node, y0, ts)
        assert result.shape == (5, 2)
        assert jnp.all(jnp.isfinite(result))

    def test_gradient_flow(self, rngs):
        """Gradients should flow through diffrax solve."""
        net = nnx.Linear(2, 2, rngs=rngs)
        node = NeuralODE(net=net, state_dim=2, rngs=rngs)
        y0 = jnp.array([1.0, 0.0])
        ts = jnp.linspace(0.0, 1.0, 5)

        @nnx.jit
        def loss_fn(model):
            traj = model.solve(y0, ts)
            return jnp.mean(traj**2)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(node)
        # Check that at least some gradients are nonzero
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
        """UDE solve should return trajectory."""

        def known_dynamics(t, y):
            return -0.1 * y

        net = nnx.Linear(2, 2, rngs=rngs)
        solver = UDESolver(
            known_dynamics_fn=known_dynamics,
            neural_residual=net,
            state_dim=2,
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 0.5])
        ts = jnp.linspace(0.0, 1.0, 10)
        traj = solver.solve(y0, ts)
        assert traj.shape == (10, 2)

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
        net.kernel = nnx.Param(jnp.zeros_like(net.kernel.value))
        net.bias = nnx.Param(jnp.zeros_like(net.bias.value))  # type: ignore[reportOptionalMemberAccess]

        solver = UDESolver(
            known_dynamics_fn=exponential_decay,
            neural_residual=net,
            state_dim=2,
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 2.0])
        ts = jnp.linspace(0.0, 1.0, 20)
        traj = solver.solve(y0, ts)

        # Compare with exact solution
        exact = y0[None, :] * jnp.exp(-ts[:, None])
        assert jnp.allclose(traj, exact, atol=1e-3), (
            f"Max error: {jnp.max(jnp.abs(traj - exact))}"
        )

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
        assert result.shape == (5, 2)

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

        traj1 = solver.solve(y0, ts)
        traj2 = solver.solve(y0, ts)
        assert jnp.allclose(traj1, traj2)

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
        traj = solver.solve(y0, ts)
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
        net.kernel = nnx.Param(jnp.zeros_like(net.kernel.value))
        net.bias = nnx.Param(jnp.zeros_like(net.bias.value))  # type: ignore[reportOptionalMemberAccess]

        solver = UDESolver(
            known_dynamics_fn=lorenz,
            neural_residual=net,
            state_dim=3,
            rngs=rngs,
        )
        y0 = jnp.array([1.0, 1.0, 1.0])
        ts = jnp.linspace(0.0, 0.1, 5)
        traj = solver.solve(y0, ts)
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
        traj = solver.solve(y0, ts)
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


"""Tests for Universal Differential Equation (UDE) Solver."""
