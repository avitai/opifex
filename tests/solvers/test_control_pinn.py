"""Tests for ControlPINN solver.

TDD tests for the ControlPINN framework that simultaneously optimizes
state variables u(x,t) and control inputs c(x,t) under PDE constraints.

Reference: ComputationalScienceLaboratory/control-pinns (2023)
    - Single network outputs both state and control
    - Combined loss: objective + pde_constraint + control_regularization
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.solvers.control_pinn import (
    ControlPINN,
    ControlPINNConfig,
    create_control_pinn,
)


@pytest.fixture
def rngs():
    return nnx.Rngs(0)


# =========================================================================
# ControlPINNConfig Tests
# =========================================================================


class TestControlPINNConfig:
    """Test ControlPINN configuration."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ControlPINNConfig()
        assert config.n_state_outputs == 1
        assert config.n_control_outputs == 1
        assert config.control_weight > 0
        assert config.pde_weight > 0

    def test_config_frozen(self):
        """Config should be immutable."""
        config = ControlPINNConfig()
        with pytest.raises(AttributeError):
            config.n_state_outputs = 3  # type: ignore[misc]

    def test_custom_config(self):
        """Custom values should be accepted."""
        config = ControlPINNConfig(
            n_state_outputs=3,
            n_control_outputs=2,
            control_weight=0.01,
            pde_weight=100.0,
        )
        assert config.n_state_outputs == 3
        assert config.n_control_outputs == 2


# =========================================================================
# ControlPINN Module Tests
# =========================================================================


class TestControlPINN:
    """Test the ControlPINN module."""

    def test_init(self, rngs):
        """ControlPINN should initialize with state+control output split."""
        config = ControlPINNConfig(n_state_outputs=2, n_control_outputs=1)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )
        assert model.config.n_state_outputs == 2
        assert model.config.n_control_outputs == 1

    def test_forward_shape(self, rngs):
        """Forward pass should return (state, control) tuple."""
        config = ControlPINNConfig(n_state_outputs=2, n_control_outputs=1)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )
        x = jnp.ones((10, 2))
        state, control = model(x)
        assert state.shape == (10, 2)
        assert control.shape == (10, 1)

    def test_state_output(self, rngs):
        """state_output should return only state variables."""
        config = ControlPINNConfig(n_state_outputs=2, n_control_outputs=1)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )
        x = jnp.ones((5, 2))
        state = model.state_output(x)
        assert state.shape == (5, 2)

    def test_control_output(self, rngs):
        """control_output should return only control variables."""
        config = ControlPINNConfig(n_state_outputs=2, n_control_outputs=1)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )
        x = jnp.ones((5, 2))
        ctrl = model.control_output(x)
        assert ctrl.shape == (5, 1)

    def test_forward_consistent(self, rngs):
        """Forward should be consistent with state_output + control_output."""
        config = ControlPINNConfig(n_state_outputs=2, n_control_outputs=1)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )
        x = jnp.ones((5, 2))
        state_full, ctrl_full = model(x)
        state_only = model.state_output(x)
        ctrl_only = model.control_output(x)
        assert jnp.allclose(state_full, state_only)
        assert jnp.allclose(ctrl_full, ctrl_only)

    def test_pde_residual(self, rngs):
        """pde_residual should compute PDE constraint violation."""
        config = ControlPINNConfig(n_state_outputs=1, n_control_outputs=1)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )

        # Simple PDE: du/dt = c (control drives state derivative)
        def pde_fn(model, x):
            """Return du/dt - c, should be zero."""
            state, control = model(x)
            return state - control  # Simplified residual

        x = jnp.ones((10, 2))
        residual = model.pde_residual(x, pde_fn)
        assert residual.shape == ()
        assert jnp.isfinite(residual)

    def test_compute_objective(self, rngs):
        """compute_objective should evaluate cost functional."""
        config = ControlPINNConfig(n_state_outputs=1, n_control_outputs=1)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )

        # Objective: minimize integral of u^2
        def objective_fn(state, control):
            return jnp.mean(state**2)

        x = jnp.ones((10, 2))
        obj = model.compute_objective(x, objective_fn)
        assert obj.shape == ()
        assert jnp.isfinite(obj)

    def test_control_loss(self, rngs):
        """control_loss should combine objective + PDE + control reg."""
        config = ControlPINNConfig(
            n_state_outputs=1,
            n_control_outputs=1,
            pde_weight=10.0,
            control_weight=0.01,
        )
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )

        def pde_fn(model, x):
            state, control = model(x)
            return state - control

        def objective_fn(state, control):
            return jnp.mean(state**2)

        x = jnp.ones((10, 2))
        loss = model.control_loss(x, pde_fn, objective_fn)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss > 0

    def test_jit_compatible(self, rngs):
        """ControlPINN should be JIT compatible."""
        config = ControlPINNConfig(n_state_outputs=1, n_control_outputs=1)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )

        @nnx.jit
        def forward_jit(model, x):
            return model(x)

        x = jnp.ones((5, 2))
        state, ctrl = forward_jit(model, x)
        assert state.shape == (5, 1)
        assert ctrl.shape == (5, 1)

    def test_gradient_flow(self, rngs):
        """Gradients should flow through control_loss."""
        config = ControlPINNConfig(n_state_outputs=1, n_control_outputs=1)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )
        x = jnp.ones((10, 2))

        def pde_fn(model, x):
            state, control = model(x)
            return state - control

        def objective_fn(state, control):
            return jnp.mean(state**2)

        @nnx.jit
        def loss_fn(model):
            return model.control_loss(x, pde_fn, objective_fn)

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(model)
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, "__len__"))
        assert has_nonzero

    def test_batch_dimension(self, rngs):
        """Should handle different batch sizes."""
        config = ControlPINNConfig(n_state_outputs=1, n_control_outputs=1)
        model = ControlPINN(
            input_dim=3,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )
        for batch in [1, 5, 20]:
            x = jnp.ones((batch, 3))
            state, ctrl = model(x)
            assert state.shape == (batch, 1)
            assert ctrl.shape == (batch, 1)

    def test_multi_state_multi_control(self, rngs):
        """Should handle multiple state and control variables."""
        config = ControlPINNConfig(n_state_outputs=3, n_control_outputs=2)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[64, 64],
            config=config,
            rngs=rngs,
        )
        x = jnp.ones((10, 2))
        state, ctrl = model(x)
        assert state.shape == (10, 3)
        assert ctrl.shape == (10, 2)

    def test_different_activations(self, rngs):
        """Should work with different activation functions."""
        config = ControlPINNConfig(n_state_outputs=1, n_control_outputs=1)
        for act in [jnp.tanh, nnx.gelu, nnx.relu]:
            model = ControlPINN(
                input_dim=2,
                hidden_dims=[32],
                config=config,
                activation=act,
                rngs=rngs,
            )
            x = jnp.ones((5, 2))
            state, ctrl = model(x)
            assert jnp.all(jnp.isfinite(state))
            assert jnp.all(jnp.isfinite(ctrl))

    def test_deterministic(self, rngs):
        """Same inputs should give same outputs."""
        config = ControlPINNConfig(n_state_outputs=1, n_control_outputs=1)
        model = ControlPINN(
            input_dim=2,
            hidden_dims=[32, 32],
            config=config,
            rngs=rngs,
        )
        x = jnp.ones((5, 2))
        s1, c1 = model(x)
        s2, c2 = model(x)
        assert jnp.allclose(s1, s2)
        assert jnp.allclose(c1, c2)


# =========================================================================
# Factory Function Tests
# =========================================================================


class TestCreateControlPINN:
    """Test the create_control_pinn factory function."""

    def test_factory_creates_model(self, rngs):
        """Factory should create a working ControlPINN."""
        model = create_control_pinn(
            input_dim=2,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        assert isinstance(model, ControlPINN)

    def test_factory_forward(self, rngs):
        """Factory-created model should forward correctly."""
        model = create_control_pinn(
            input_dim=2,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jnp.ones((5, 2))
        state, ctrl = model(x)
        assert state.shape == (5, 1)
        assert ctrl.shape == (5, 1)

    def test_factory_custom_config(self, rngs):
        """Factory should accept custom config."""
        config = ControlPINNConfig(
            n_state_outputs=2,
            n_control_outputs=3,
            control_weight=0.1,
        )
        model = create_control_pinn(
            input_dim=4,
            hidden_dims=[64],
            config=config,
            rngs=rngs,
        )
        x = jnp.ones((5, 4))
        state, ctrl = model(x)
        assert state.shape == (5, 2)
        assert ctrl.shape == (5, 3)
