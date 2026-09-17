"""``OptimizationConfig`` is the one owner of the optimizer, built through ``substrax.optim``.

The schedule is the optimizer's learning rate, so a decaying schedule reaches the update
(a ``scale_by_schedule`` chained before Adam is cancelled by Adam's normalisation); a rate
set in ``optimization_config`` is the rate the optimizer applies; and what optax would
silently misread is refused.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from substrax.optim import create_optimizer, current_learning_rate

from opifex.core.training.config import OptimizationConfig, TrainingConfig
from opifex.core.training.optimizers import create_schedule, optimizer_spec
from opifex.core.training.trainer import Trainer
from opifex.neural.base import StandardMLP


def _model() -> StandardMLP:
    return StandardMLP([2, 4, 1], rngs=nnx.Rngs(0))


def _loss(model: StandardMLP) -> jax.Array:
    return jnp.sum(model(jnp.ones((3, 2))) ** 2)


class TestOneOwner:
    """The learning rate lives in ``optimization_config`` and nowhere else."""

    def test_training_config_has_no_learning_rate_of_its_own(self) -> None:
        with pytest.raises(TypeError, match="learning_rate"):
            TrainingConfig(learning_rate=1e-3)  # type: ignore[call-arg]

    def test_a_rate_set_in_optimization_config_reaches_the_optimizer(self) -> None:
        config = TrainingConfig(optimization_config=OptimizationConfig(learning_rate=5e-3))

        trainer = Trainer(_model(), config)

        assert float(current_learning_rate(trainer.optimizer)) == pytest.approx(5e-3)

    def test_the_step_metrics_report_the_applied_rate(self) -> None:
        config = TrainingConfig(
            optimization_config=OptimizationConfig(
                learning_rate=0.1, schedule_type="linear", end_value=0.0, transition_steps=4
            )
        )
        trainer = Trainer(_model(), config)
        grads = nnx.grad(_loss)(trainer.model)
        # optax evaluates the schedule at the count before each update, so the fifth
        # update is the first one applied at the schedule's end value.
        for _ in range(5):
            trainer.optimizer.update(trainer.model, grads)

        metrics = trainer._build_step_metrics(jnp.float32(0.0), grads, {})

        assert float(metrics["learning_rate"]) == pytest.approx(0.0)


class TestScheduleReachesTheUpdate:
    """A schedule is the base optimizer's learning rate, not a scale chained before it."""

    def test_a_schedule_decayed_to_zero_stops_the_parameters(self) -> None:
        config = OptimizationConfig(
            optimizer="adam",
            learning_rate=0.1,
            schedule_type="linear",
            end_value=0.0,
            transition_steps=4,
        )
        model = _model()
        optimizer = create_optimizer(model, optimizer_spec(config))
        for _ in range(4):
            optimizer.update(model, nnx.grad(_loss)(model))
        before = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

        optimizer.update(model, nnx.grad(_loss)(model))

        after = nnx.state(model, nnx.Param)
        assert float(current_learning_rate(optimizer)) == pytest.approx(0.0)
        assert all(
            jnp.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
        )

    def test_a_constant_rate_moves_the_parameters(self) -> None:
        model = _model()
        optimizer = create_optimizer(model, optimizer_spec(OptimizationConfig(learning_rate=0.1)))
        before = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))

        optimizer.update(model, nnx.grad(_loss)(model))

        after = nnx.state(model, nnx.Param)
        assert not all(
            jnp.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
        )


class TestOptimizerSpec:
    """The mapping from ``OptimizationConfig`` to substrax's specification."""

    def test_momentum_reaches_only_the_optimizers_that_take_it(self) -> None:
        assert optimizer_spec(OptimizationConfig(optimizer="adam")).momentum is None
        assert optimizer_spec(OptimizationConfig(optimizer="sgd", momentum=0.95)).momentum == 0.95

    def test_betas_eps_and_decay_map(self) -> None:
        spec = optimizer_spec(
            OptimizationConfig(
                optimizer="adamw", beta1=0.8, beta2=0.99, eps=1e-6, weight_decay=1e-2
            )
        )
        assert (spec.optimizer_type, spec.b1, spec.b2, spec.eps, spec.weight_decay) == (
            "adamw",
            0.8,
            0.99,
            1e-6,
            1e-2,
        )

    def test_weight_decay_on_adam_is_refused(self) -> None:
        with pytest.raises(ValueError, match="weight_decay"):
            optimizer_spec(OptimizationConfig(optimizer="adam", weight_decay=1e-2))

    def test_an_unknown_optimizer_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            optimizer_spec(OptimizationConfig(optimizer="unknown_optimizer"))

    def test_clipping_fields_map(self) -> None:
        by_norm = optimizer_spec(OptimizationConfig(gradient_clip_norm=1.0))
        by_value = optimizer_spec(OptimizationConfig(gradient_clip_value=0.5))
        assert (by_norm.gradient_clip_norm, by_norm.gradient_clip_value) == (1.0, None)
        assert (by_value.gradient_clip_norm, by_value.gradient_clip_value) == (None, 0.5)

    def test_a_schedule_becomes_the_learning_rate(self) -> None:
        spec = optimizer_spec(
            OptimizationConfig(learning_rate=0.1, schedule_type="cosine", decay_steps=10)
        )
        assert callable(spec.learning_rate)
        assert float(jnp.asarray(spec.learning_rate(0))) == pytest.approx(0.1)

    def test_without_a_schedule_the_rate_is_the_constant(self) -> None:
        assert optimizer_spec(OptimizationConfig(learning_rate=2e-3)).learning_rate == 2e-3


class TestScheduleCreation:
    """Learning-rate schedules built from ``OptimizationConfig``."""

    def test_constant(self) -> None:
        schedule = create_schedule(
            OptimizationConfig(schedule_type="constant", learning_rate=0.001)
        )
        assert schedule(0) == 0.001
        assert schedule(1000) == 0.001

    def test_cosine(self) -> None:
        schedule = create_schedule(
            OptimizationConfig(
                schedule_type="cosine", learning_rate=0.01, decay_steps=1000, alpha=0.1
            )
        )
        assert schedule(0) == pytest.approx(0.01)
        assert 0.001 <= float(jnp.asarray(schedule(1000))) <= 0.01

    def test_exponential(self) -> None:
        schedule = create_schedule(
            OptimizationConfig(
                schedule_type="exponential",
                learning_rate=0.1,
                transition_steps=100,
                decay_rate=0.96,
            )
        )
        assert schedule(0) == 0.1
        assert float(jnp.asarray(schedule(200))) < float(jnp.asarray(schedule(100))) < 0.1

    def test_linear(self) -> None:
        schedule = create_schedule(
            OptimizationConfig(
                schedule_type="linear", learning_rate=0.01, end_value=0.001, transition_steps=1000
            )
        )
        assert schedule(0) == 0.01
        assert schedule(500) == pytest.approx(0.0055)
        assert schedule(1000) == pytest.approx(0.001)

    def test_linear_is_float32_under_x64(self) -> None:
        with jax.enable_x64(True):
            schedule = create_schedule(
                OptimizationConfig(
                    schedule_type="linear",
                    learning_rate=0.01,
                    end_value=0.001,
                    transition_steps=1000,
                )
            )
            assert jnp.asarray(schedule(0)).dtype == jnp.float32
            assert schedule(500) == pytest.approx(0.0055)

    def test_step(self) -> None:
        schedule = create_schedule(
            OptimizationConfig(
                schedule_type="step",
                boundaries_and_values=([100, 200, 300], [0.1, 0.01, 0.001, 0.0001]),
            )
        )
        assert schedule(99) == pytest.approx(0.1)
        assert schedule(100) == pytest.approx(0.01)
        assert schedule(199) == pytest.approx(0.01)
        assert schedule(300) == pytest.approx(0.0001)

    def test_warmup_cosine(self) -> None:
        schedule = create_schedule(
            OptimizationConfig(
                schedule_type="warmup_cosine",
                learning_rate=0.0,
                peak_value=0.01,
                warmup_steps=100,
                decay_steps=1000,
            )
        )
        assert schedule(0) == 0.0
        assert schedule(100) == pytest.approx(0.01)
        assert float(jnp.asarray(schedule(500))) < 0.01

    def test_an_unknown_schedule_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown schedule type"):
            create_schedule(OptimizationConfig(schedule_type="invalid_schedule"))
