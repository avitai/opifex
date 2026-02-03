"""Tests for gradient checkpointing (rematerialization) integration (Sprint 1 C.1).

Tests are written first per TDD. BasicTrainer should use artifex's apply_remat()
when gradient_checkpointing=True in TrainingConfig.
"""

import jax
import pytest
from flax import nnx

from opifex.core.training.config import TrainingConfig


class SimpleMLP(nnx.Module):
    """Minimal model for checkpointing tests."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(4, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, 32, rngs=rngs)
        self.linear3 = nnx.Linear(32, 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        return self.linear3(x)


@pytest.fixture
def training_data() -> tuple[jax.Array, jax.Array]:
    """Create simple training data."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (32, 4))
    y = jax.random.normal(jax.random.PRNGKey(43), (32, 2))
    return x, y


class TestGradientCheckpointing:
    """Test gradient checkpointing integration with BasicTrainer."""

    def test_remat_same_loss_trajectory(
        self, training_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Training produces same loss with and without remat (numerical tolerance)."""
        from opifex.training.basic_trainer import BasicTrainer

        x, y = training_data

        # Train without remat
        config_no_remat = TrainingConfig(
            num_epochs=5,
            gradient_checkpointing=False,
            verbose=False,
        )
        model_no_remat = SimpleMLP(rngs=nnx.Rngs(0))
        trainer_no_remat = BasicTrainer(model_no_remat, config_no_remat)
        _, metrics_no_remat = trainer_no_remat.train((x, y))

        # Train with remat
        config_remat = TrainingConfig(
            num_epochs=5,
            gradient_checkpointing=True,
            verbose=False,
        )
        model_remat = SimpleMLP(rngs=nnx.Rngs(0))
        trainer_remat = BasicTrainer(model_remat, config_remat)
        _, metrics_remat = trainer_remat.train((x, y))

        # Loss trajectories should match closely
        losses_no = metrics_no_remat.train_losses
        losses_yes = metrics_remat.train_losses
        assert len(losses_no) == len(losses_yes)
        for l_no, l_yes in zip(losses_no, losses_yes, strict=False):
            assert abs(l_no - l_yes) < 1e-4, f"Loss mismatch: {l_no:.6f} vs {l_yes:.6f}"

    def test_remat_all_policies(self) -> None:
        """All named policies can be used without error."""
        from opifex.training.basic_trainer import BasicTrainer

        policies = [
            "dots_saveable",
            "everything_saveable",
            "nothing_saveable",
            "checkpoint_dots",
            "checkpoint_dots_no_batch",
        ]
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 4))
        y = jax.random.normal(jax.random.PRNGKey(1), (8, 2))

        for policy in policies:
            config = TrainingConfig(
                num_epochs=1,
                gradient_checkpointing=True,
                gradient_checkpoint_policy=policy,
                verbose=False,
            )
            model = SimpleMLP(rngs=nnx.Rngs(0))
            trainer = BasicTrainer(model, config)
            _, metrics = trainer.train((x, y))
            assert len(metrics.train_losses) == 1, f"Policy {policy} failed"

    def test_remat_config_validation(self) -> None:
        """Invalid policy name is rejected."""
        from opifex.training.basic_trainer import BasicTrainer

        config = TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=True,
            gradient_checkpoint_policy="nonexistent_policy",
            verbose=False,
        )
        model = SimpleMLP(rngs=nnx.Rngs(0))
        with pytest.raises((ValueError, KeyError)):
            BasicTrainer(model, config)

    def test_remat_disabled_by_default(self) -> None:
        """Default config has gradient checkpointing disabled."""
        config = TrainingConfig()
        assert config.gradient_checkpointing is False
        assert config.gradient_checkpoint_policy is None

    def test_remat_no_policy_uses_default(
        self, training_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """When remat enabled without explicit policy, it still works."""
        from opifex.training.basic_trainer import BasicTrainer

        x, y = training_data
        config = TrainingConfig(
            num_epochs=2,
            gradient_checkpointing=True,
            gradient_checkpoint_policy=None,
            verbose=False,
        )
        model = SimpleMLP(rngs=nnx.Rngs(0))
        trainer = BasicTrainer(model, config)
        _, metrics = trainer.train((x, y))
        assert len(metrics.train_losses) == 2
