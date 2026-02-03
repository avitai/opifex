"""Tests for distributed training integration with Trainer.

TDD — these tests are written FIRST to define the expected behavior
of the Trainer when used with DistributedConfig.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


pytestmark = pytest.mark.distributed

from opifex.core.training.config import TrainingConfig
from opifex.distributed.config import DistributedConfig


class _SimpleModel(nnx.Module):
    """Minimal model for distributed Trainer tests."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(4, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


@pytest.fixture
def distributed_config() -> DistributedConfig:
    """Single-device distributed config for testing."""
    return DistributedConfig(
        mesh_shape=(1,),
        mesh_axis_names=("data",),
    )


@pytest.fixture
def sample_data() -> tuple[jax.Array, jax.Array]:
    """Generate sample training data (32 samples, 4 features -> 1 output)."""
    key = jax.random.PRNGKey(42)
    k1, _ = jax.random.split(key)
    x = jax.random.normal(k1, (32, 4))
    y = jnp.sum(x, axis=1, keepdims=True)
    return x, y


class TestTrainingConfigWithDistributed:
    """Test that TrainingConfig accepts DistributedConfig."""

    def test_config_accepts_distributed_config(
        self, distributed_config: DistributedConfig
    ) -> None:
        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            distributed_config=distributed_config,
        )
        assert config.distributed_config is distributed_config

    def test_config_defaults_to_no_distributed(self) -> None:
        config = TrainingConfig(num_epochs=2)
        assert config.distributed_config is None


class TestTrainerDistributedInit:
    """Test that Trainer initializes distributed manager when config is present."""

    def test_trainer_creates_distributed_manager(
        self, distributed_config: DistributedConfig
    ) -> None:
        from opifex.core.training.trainer import Trainer
        from opifex.distributed.manager import DistributedManager

        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            distributed_config=distributed_config,
        )
        model = _SimpleModel(rngs=nnx.Rngs(0))
        trainer = Trainer(model, config)

        assert hasattr(trainer, "_distributed_manager")
        assert isinstance(trainer._distributed_manager, DistributedManager)

    def test_trainer_no_manager_without_config(self) -> None:
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(num_epochs=2, learning_rate=1e-3)
        model = _SimpleModel(rngs=nnx.Rngs(0))
        trainer = Trainer(model, config)

        assert trainer._distributed_manager is None


class TestTrainerDistributedFit:
    """Test that Trainer.fit works end-to-end with distributed config."""

    def test_fit_with_distributed_config_completes(
        self,
        distributed_config: DistributedConfig,
        sample_data: tuple[jax.Array, jax.Array],
    ) -> None:
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-2,
            batch_size=16,
            distributed_config=distributed_config,
        )
        model = _SimpleModel(rngs=nnx.Rngs(0))
        trainer = Trainer(model, config)

        x, y = sample_data
        trained_model, metrics = trainer.fit(train_data=(x, y))

        assert trained_model is not None
        assert "final_train_loss" in metrics

    def test_fit_with_distributed_reduces_loss(
        self,
        distributed_config: DistributedConfig,
        sample_data: tuple[jax.Array, jax.Array],
    ) -> None:
        from opifex.core.training.trainer import Trainer

        config = TrainingConfig(
            num_epochs=10,
            learning_rate=1e-2,
            batch_size=16,
            distributed_config=distributed_config,
        )
        model = _SimpleModel(rngs=nnx.Rngs(0))
        trainer = Trainer(model, config)

        x, y = sample_data
        _, metrics = trainer.fit(train_data=(x, y))

        assert metrics["final_train_loss"] < metrics["initial_train_loss"]
