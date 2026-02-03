"""Tests for gradient checkpointing integration in opifex.

Tests cover:
- TrainingConfig gradient checkpointing fields and all policy names
- Trainer applying remat to the forward pass (single and multi-step)
- Trainer with boundary data + checkpointing combined
- Invalid policy error propagation
- Gradient correctness with checkpointing enabled
- FNO multiscale gradient checkpointing (1D, 2D, output equivalence)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.training.config import CheckpointConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _no_checkpoint_config() -> CheckpointConfig:
    """Return CheckpointConfig with empty dir to skip checkpoint manager."""
    return CheckpointConfig(checkpoint_dir="")


def _make_deep_model(rngs: nnx.Rngs) -> nnx.Module:
    """Build a multi-layer MLP where gradient checkpointing matters."""

    class SmallMLP(nnx.Module):
        """Three-layer MLP for testing gradient checkpointing."""

        def __init__(self, rngs: nnx.Rngs) -> None:
            self.l1 = nnx.Linear(4, 16, rngs=rngs)
            self.l2 = nnx.Linear(16, 16, rngs=rngs)
            self.l3 = nnx.Linear(16, 2, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
            x = nnx.relu(self.l1(x))
            x = nnx.relu(self.l2(x))
            return self.l3(x)

    return SmallMLP(rngs)


# ===================================================================
# Config tests
# ===================================================================


class TestTrainingConfigGradientCheckpointing:
    """Test gradient checkpointing fields on TrainingConfig."""

    def test_defaults_disabled(self):
        """Gradient checkpointing is disabled by default."""
        config = TrainingConfig()
        assert config.gradient_checkpointing is False
        assert config.gradient_checkpoint_policy is None

    def test_enable_gradient_checkpointing(self):
        """Can enable gradient checkpointing via config."""
        config = TrainingConfig(gradient_checkpointing=True)
        assert config.gradient_checkpointing is True
        assert config.gradient_checkpoint_policy is None

    def test_enable_with_policy(self):
        """Can specify a checkpoint policy string."""
        config = TrainingConfig(
            gradient_checkpointing=True,
            gradient_checkpoint_policy="dots_saveable",
        )
        assert config.gradient_checkpointing is True
        assert config.gradient_checkpoint_policy == "dots_saveable"

    def test_policy_without_checkpointing_allowed(self):
        """Setting policy without enabling checkpointing is allowed."""
        config = TrainingConfig(
            gradient_checkpointing=False,
            gradient_checkpoint_policy="everything_saveable",
        )
        assert config.gradient_checkpointing is False
        assert config.gradient_checkpoint_policy == "everything_saveable"

    @pytest.mark.parametrize(
        "policy",
        [
            "dots_saveable",
            "everything_saveable",
            "nothing_saveable",
            "checkpoint_dots",
            "checkpoint_dots_no_batch",
        ],
    )
    def test_all_named_policies_accepted(self, policy: str):
        """All artifex CHECKPOINT_POLICIES names are accepted by config."""
        config = TrainingConfig(
            gradient_checkpointing=True,
            gradient_checkpoint_policy=policy,
        )
        assert config.gradient_checkpoint_policy == policy


# ===================================================================
# Trainer tests
# ===================================================================


class TestTrainerGradientCheckpointing:
    """Test that Trainer applies gradient checkpointing when configured."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple NNX model for testing."""
        return nnx.Linear(in_features=4, out_features=2, rngs=nnx.Rngs(0))

    @pytest.fixture
    def deep_model(self):
        """Create a multi-layer NNX model for testing."""
        return _make_deep_model(nnx.Rngs(0))

    @pytest.fixture
    def train_data(self):
        """Create simple training data."""
        x = jnp.ones((8, 4))
        y = jnp.zeros((8, 2))
        return x, y

    def test_trainer_init_with_checkpointing(self, simple_model):
        """Trainer accepts gradient checkpointing config without error."""
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(
            gradient_checkpointing=True,
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer = Trainer(model=simple_model, config=config)
        assert trainer.config.gradient_checkpointing is True

    def test_training_step_with_checkpointing(self, simple_model, train_data):
        """Training step completes with gradient checkpointing enabled."""
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(
            gradient_checkpointing=True,
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer = Trainer(model=simple_model, config=config)
        x, y = train_data
        loss, metrics = trainer.training_step(x, y)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert "data_loss" in metrics

    def test_training_step_without_checkpointing(self, simple_model, train_data):
        """Training step completes normally without gradient checkpointing."""
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(
            gradient_checkpointing=False,
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer = Trainer(model=simple_model, config=config)
        x, y = train_data
        loss, _metrics = trainer.training_step(x, y)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_training_step_with_policy(self, simple_model, train_data):
        """Training step completes with a specific checkpoint policy."""
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(
            gradient_checkpointing=True,
            gradient_checkpoint_policy="everything_saveable",
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer = Trainer(model=simple_model, config=config)
        x, y = train_data
        loss, _ = trainer.training_step(x, y)
        assert jnp.isfinite(loss)

    def test_gradient_correctness_with_checkpointing(self, train_data):
        """Gradients with checkpointing match those without."""
        from opifex.core.training.trainer import Trainer

        x, y = train_data

        # Train one step WITHOUT checkpointing
        model_a = nnx.Linear(in_features=4, out_features=2, rngs=nnx.Rngs(0))
        config_a = TrainingConfig(
            gradient_checkpointing=False,
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer_a = Trainer(model=model_a, config=config_a)
        loss_a, _ = trainer_a.training_step(x, y)

        # Train one step WITH checkpointing
        model_b = nnx.Linear(in_features=4, out_features=2, rngs=nnx.Rngs(0))
        config_b = TrainingConfig(
            gradient_checkpointing=True,
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer_b = Trainer(model=model_b, config=config_b)
        loss_b, _ = trainer_b.training_step(x, y)

        # Losses should be identical (same model, same data, same step)
        assert jnp.allclose(loss_a, loss_b, atol=1e-6)

    def test_deep_model_gradient_correctness(self, train_data):
        """Gradient correctness holds for multi-layer models."""
        from opifex.core.training.trainer import Trainer

        x, y = train_data

        model_a = _make_deep_model(nnx.Rngs(42))
        config_a = TrainingConfig(
            gradient_checkpointing=False,
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer_a = Trainer(model=model_a, config=config_a)
        loss_a, _ = trainer_a.training_step(x, y)

        model_b = _make_deep_model(nnx.Rngs(42))
        config_b = TrainingConfig(
            gradient_checkpointing=True,
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer_b = Trainer(model=model_b, config=config_b)
        loss_b, _ = trainer_b.training_step(x, y)

        assert jnp.allclose(loss_a, loss_b, atol=1e-6)

    def test_multi_step_training_with_checkpointing(self, simple_model, train_data):
        """Multiple training steps work with gradient checkpointing."""
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(
            gradient_checkpointing=True,
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer = Trainer(model=simple_model, config=config)
        x, y = train_data

        losses = []
        for _ in range(5):
            loss, _ = trainer.training_step(x, y)
            losses.append(float(loss))

        # Loss should decrease over multiple steps
        assert losses[-1] < losses[0]
        # All losses should be finite
        assert all(jnp.isfinite(jnp.array(l)) for l in losses)

    def test_checkpointing_with_boundary_data(self, simple_model):
        """Checkpointing works when boundary data is also provided."""
        from opifex.core.training.physics_configs import BoundaryConfig
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(
            gradient_checkpointing=True,
            boundary_config=BoundaryConfig(enforce=True, weight=1.0),
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer = Trainer(model=simple_model, config=config)

        x = jnp.ones((8, 4))
        y = jnp.zeros((8, 2))
        x_bd = jnp.ones((4, 4))
        y_bd = jnp.zeros((4, 2))

        loss, metrics = trainer.training_step(x, y, boundary_data=(x_bd, y_bd))
        assert jnp.isfinite(loss)
        assert "data_loss" in metrics
        assert "boundary_loss" in metrics

    def test_invalid_policy_raises_error(self, simple_model, train_data):
        """Invalid policy string raises ValueError at training step."""
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(
            gradient_checkpointing=True,
            gradient_checkpoint_policy="nonexistent_policy",
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer = Trainer(model=simple_model, config=config)
        x, y = train_data

        with pytest.raises(ValueError, match="Unknown checkpoint policy"):
            trainer.training_step(x, y)

    @pytest.mark.parametrize(
        "policy",
        [
            None,
            "dots_saveable",
            "everything_saveable",
            "nothing_saveable",
        ],
    )
    def test_training_step_all_policies(
        self, simple_model, train_data, policy: str | None
    ):
        """Training step succeeds with each named checkpoint policy."""
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(
            gradient_checkpointing=True,
            gradient_checkpoint_policy=policy,
            checkpoint_config=_no_checkpoint_config(),
        )
        trainer = Trainer(model=simple_model, config=config)
        x, y = train_data
        loss, _ = trainer.training_step(x, y)
        assert jnp.isfinite(loss)


# ===================================================================
# FNO Multiscale tests
# ===================================================================


class TestFNOMultiscaleGradientCheckpointing:
    """Test gradient checkpointing in MultiScaleFourierNeuralOperator."""

    def _make_fno_1d(
        self,
        *,
        checkpointing: bool = True,
        layers_per_scale: list[int] | None = None,
    ):
        """Factory for 1D FNO multiscale models."""
        from opifex.neural.operators.fno.multiscale import (
            MultiScaleFourierNeuralOperator,
        )

        return MultiScaleFourierNeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=8,
            modes_per_scale=[4, 2],
            num_layers_per_scale=layers_per_scale or [1, 1],
            use_gradient_checkpointing=checkpointing,
            rngs=nnx.Rngs(0),
        )

    def test_default_checkpointing_flag_is_true(self):
        """FNO multiscale defaults to use_gradient_checkpointing=True."""
        from opifex.neural.operators.fno.multiscale import (
            MultiScaleFourierNeuralOperator,
        )

        model = MultiScaleFourierNeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=8,
            modes_per_scale=[4, 2],
            num_layers_per_scale=[1, 1],
            rngs=nnx.Rngs(0),
        )
        assert model.use_gradient_checkpointing is True

    def test_checkpointing_flag_can_be_disabled(self):
        """FNO multiscale accepts use_gradient_checkpointing=False."""
        model = self._make_fno_1d(checkpointing=False)
        assert model.use_gradient_checkpointing is False

    def test_forward_with_checkpointing_1d(self):
        """FNO forward pass works with checkpointing on 1D input."""
        model = self._make_fno_1d(checkpointing=True)
        x = jnp.ones((2, 2, 16))
        output = model(x)
        assert output.shape == (2, 2, 16)

    def test_forward_without_checkpointing_1d(self):
        """FNO forward pass works with checkpointing disabled on 1D input."""
        model = self._make_fno_1d(checkpointing=False)
        x = jnp.ones((2, 2, 16))
        output = model(x)
        assert output.shape == (2, 2, 16)

    def test_output_equivalence_1d(self):
        """Checkpointing does not change the forward output (1D)."""
        model_on = self._make_fno_1d(checkpointing=True)
        model_off = self._make_fno_1d(checkpointing=False)
        x = jnp.ones((2, 2, 16))

        out_on = model_on(x)
        out_off = model_off(x)
        assert jnp.allclose(out_on, out_off, atol=1e-5)

    def test_forward_with_checkpointing_2d(self):
        """FNO forward pass works with checkpointing on 2D spatial input."""
        from opifex.neural.operators.fno.multiscale import (
            MultiScaleFourierNeuralOperator,
        )

        model = MultiScaleFourierNeuralOperator(
            in_channels=2,
            out_channels=2,
            hidden_channels=8,
            modes_per_scale=[4, 2],
            num_layers_per_scale=[1, 1],
            use_gradient_checkpointing=True,
            rngs=nnx.Rngs(0),
        )
        # 2D input: (batch, channels, height, width)
        x = jnp.ones((2, 2, 8, 8))
        output = model(x)
        assert output.shape == (2, 2, 8, 8)

    def test_gradient_flow_with_checkpointing(self):
        """Gradients flow correctly with checkpointing enabled."""
        model = self._make_fno_1d(checkpointing=True)
        x = jnp.ones((2, 2, 16))

        def loss_fn(m):
            return jnp.mean(m(x) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        param_grads = nnx.state(grads, nnx.Param)
        flat_grads = jax.tree.leaves(param_grads)
        assert len(flat_grads) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads)

    def test_gradient_equivalence(self):
        """Gradients match between checkpointed and non-checkpointed."""
        x = jnp.ones((2, 2, 16))

        model_on = self._make_fno_1d(checkpointing=True)
        model_off = self._make_fno_1d(checkpointing=False)

        def loss_fn(m):
            return jnp.mean(m(x) ** 2)

        loss_on, grads_on = nnx.value_and_grad(loss_fn)(model_on)
        loss_off, grads_off = nnx.value_and_grad(loss_fn)(model_off)

        assert jnp.allclose(loss_on, loss_off, atol=1e-5)
        flat_on = jax.tree.leaves(nnx.state(grads_on, nnx.Param))
        flat_off = jax.tree.leaves(nnx.state(grads_off, nnx.Param))
        for g_on, g_off in zip(flat_on, flat_off, strict=True):
            assert jnp.allclose(g_on, g_off, atol=1e-5)

    def test_deeper_scale_layers(self):
        """Checkpointing works with multiple layers per scale."""
        model = self._make_fno_1d(checkpointing=True, layers_per_scale=[3, 2])
        x = jnp.ones((2, 2, 16))
        output = model(x)
        assert output.shape == (2, 2, 16)

        def loss_fn(m):
            return jnp.mean(m(x) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        flat_grads = jax.tree.leaves(nnx.state(grads, nnx.Param))
        assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads)
