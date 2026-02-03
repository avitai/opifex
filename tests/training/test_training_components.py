"""Tests for training components module."""

from __future__ import annotations

import optax
import pytest
from flax import nnx

from opifex.core.training.components import (
    FlexibleOptimizerFactory,
    TrainingComponent,
)
from opifex.neural.base import StandardMLP
from opifex.training.metrics import TrainingState


class TestTrainingComponent:
    """Test base class for training components."""

    def test_initialization(self):
        """Test base component initialization."""
        component = TrainingComponent()
        assert component.config == {}

    def test_initialization_with_config(self):
        """Test base component initialization with config."""
        config = {"param1": "value1", "param2": 42}
        component = TrainingComponent(config)
        assert component.config == config

    def test_setup_method(self):
        """Test setup method (should do nothing in base class)."""
        component = TrainingComponent()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Should not raise any errors
        component.setup(model, training_state)

    def test_cleanup_method(self):
        """Test cleanup method (should do nothing in base class)."""
        component = TrainingComponent()
        # Should not raise any errors
        component.cleanup()


class TestFlexibleOptimizerFactory:
    """Test flexible optimizer factory."""

    def test_initialization_defaults(self):
        """Test factory initialization with defaults."""
        factory = FlexibleOptimizerFactory()

        # Test that OptimizerConfig is created with correct defaults
        assert factory.optimizer_config.optimizer_type == "adam"
        assert factory.optimizer_config.learning_rate == 1e-3
        assert factory.optimizer_config.weight_decay == 0.0

    def test_initialization_custom_config(self):
        """Test factory initialization with custom config."""
        config = {
            "optimizer_type": "sgd",
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "use_schedule": False,
        }
        factory = FlexibleOptimizerFactory(config)

        # Test that OptimizerConfig is created with correct custom values
        assert factory.optimizer_config.optimizer_type == "sgd"
        assert factory.optimizer_config.learning_rate == 0.01
        assert factory.optimizer_config.weight_decay == 1e-4
        assert factory.optimizer_config.schedule_type is None  # use_schedule=False

    def test_create_adam_optimizer(self):
        """Test creating Adam optimizer."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "adam",
                "learning_rate": 1e-3,
                "use_schedule": False,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = factory.create_optimizer(model)

        assert optimizer is not None
        # Test that it can initialize opt_state
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)
        assert opt_state is not None

    def test_create_adamw_optimizer(self):
        """Test creating AdamW optimizer."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "use_schedule": False,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = factory.create_optimizer(model)

        assert optimizer is not None
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)
        assert opt_state is not None

    def test_create_sgd_optimizer(self):
        """Test creating SGD optimizer."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "sgd",
                "learning_rate": 0.01,
                "momentum": 0.95,
                "use_schedule": False,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = factory.create_optimizer(model)

        assert optimizer is not None
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)
        assert opt_state is not None

    def test_unknown_optimizer_type(self):
        """Test that unknown optimizer type raises error."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "unknown_optimizer",
                "learning_rate": 1e-3,
                "use_schedule": False,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        with pytest.raises(ValueError, match="Unknown optimizer type"):
            factory.create_optimizer(model)

    def test_create_optimizer_with_cosine_schedule(self):
        """Test creating optimizer with cosine learning rate schedule."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "adam",
                "learning_rate": 1e-3,
                "use_schedule": True,
                "schedule_type": "cosine",
                "total_steps": 1000,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = factory.create_optimizer(model)

        assert optimizer is not None
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)
        assert opt_state is not None

    def test_create_optimizer_with_exponential_schedule(self):
        """Test creating optimizer with exponential learning rate schedule."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "adam",
                "learning_rate": 1e-3,
                "use_schedule": True,
                "schedule_type": "exponential",
                "decay_steps": 100,
                "decay_rate": 0.95,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = factory.create_optimizer(model)

        assert optimizer is not None

    def test_create_optimizer_with_linear_schedule(self):
        """Test creating optimizer with linear learning rate schedule."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "adam",
                "learning_rate": 1e-3,
                "use_schedule": True,
                "schedule_type": "linear",
                "total_steps": 1000,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = factory.create_optimizer(model)

        assert optimizer is not None

    def test_create_optimizer_with_constant_schedule(self):
        """Test creating optimizer with constant learning rate (fallback)."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "adam",
                "learning_rate": 1e-3,
                "use_schedule": True,
                "schedule_type": "constant",  # Unknown schedule type
                "total_steps": 1000,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = factory.create_optimizer(model)

        assert optimizer is not None

    def test_optimizer_with_gradient_clipping(self):
        """Test optimizer with gradient clipping enabled."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "adam",
                "learning_rate": 1e-3,
                "use_schedule": True,
                "grad_clip": 1.0,
                "total_steps": 1000,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = factory.create_optimizer(model)

        assert optimizer is not None

    def test_adam_with_custom_betas(self):
        """Test Adam optimizer with custom beta parameters."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "adam",
                "learning_rate": 1e-3,
                "beta1": 0.95,
                "beta2": 0.9999,
                "eps": 1e-9,
                "use_schedule": False,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = factory.create_optimizer(model)

        assert optimizer is not None

    def test_sgd_with_custom_momentum(self):
        """Test SGD optimizer with custom momentum."""
        factory = FlexibleOptimizerFactory(
            config={
                "optimizer_type": "sgd",
                "learning_rate": 0.01,
                "momentum": 0.99,
                "use_schedule": False,
            }
        )

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = factory.create_optimizer(model)

        assert optimizer is not None
