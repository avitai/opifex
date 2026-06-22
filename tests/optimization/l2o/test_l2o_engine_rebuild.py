"""Contracts for the rebuilt L2OEngine (P5): meta-train, apply, benchmark, persist."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.optimization.l2o.engine import L2OEngine
from opifex.optimization.l2o.learned import LearnableSGD, MLPLearnedOptimizer
from opifex.optimization.l2o.tasks import QuadraticTaskFamily


if TYPE_CHECKING:
    from pathlib import Path


def test_engine_meta_trains_and_optimizes() -> None:
    """The engine meta-trains, then applies the trained optimiser to a held-out task."""
    engine = L2OEngine(LearnableSGD(0.01), QuadraticTaskFamily(dim=5, max_log_condition=1.0))
    losses = engine.meta_train(
        jax.random.key(0),
        num_outer_steps=120,
        num_tasks=8,
        trunc_length=10,
        total_horizon=30,
        perturbation_std=0.02,
        meta_learning_rate=0.05,
    )
    assert jnp.mean(losses[-20:]) < jnp.mean(losses[:20])

    task = engine.task_family.sample(jax.random.key(1))
    curve = engine.optimize(task, task.init(jax.random.key(2)), num_steps=40, key=jax.random.key(3))
    assert curve.shape == (41,)
    assert curve[-1] < curve[0]


def test_engine_requires_training_before_use() -> None:
    """Using the engine before meta-training (or loading) raises a clear error."""
    import pytest

    engine = L2OEngine(LearnableSGD(), QuadraticTaskFamily(dim=3))
    task = engine.task_family.sample(jax.random.key(0))
    with pytest.raises(RuntimeError, match="meta_train"):
        engine.optimize(task, task.init(jax.random.key(1)), num_steps=5, key=jax.random.key(2))


def test_engine_benchmark_runs() -> None:
    """The engine benchmarks the trained optimiser against a tuned baseline."""
    engine = L2OEngine(
        MLPLearnedOptimizer(hidden_size=8, hidden_layers=1),
        QuadraticTaskFamily(dim=4, max_log_condition=1.0),
    )
    engine.meta_train(
        jax.random.key(0),
        num_outer_steps=20,
        num_tasks=4,
        trunc_length=10,
        total_horizon=20,
        perturbation_std=0.01,
        meta_learning_rate=1e-3,
    )
    result = engine.benchmark(jax.random.key(1), num_tasks=4, num_steps=30)
    assert result["learned_curve_mean"].shape == (31,)
    assert jnp.all(jnp.isfinite(result["baseline_curve_mean"]))


def test_theta_save_load_round_trip(tmp_path: Path) -> None:
    """Trained theta serialises and restores to the same values (Orbax round-trip)."""
    learned_optimizer = LearnableSGD(0.05)
    family = QuadraticTaskFamily(dim=4, max_log_condition=1.0)
    engine = L2OEngine(learned_optimizer, family)
    engine.meta_train(
        jax.random.key(0),
        num_outer_steps=10,
        num_tasks=4,
        trunc_length=10,
        total_horizon=20,
        perturbation_std=0.02,
        meta_learning_rate=0.05,
    )
    trained_leaves = jax.tree.leaves(engine.theta)

    directory = tmp_path / "theta"
    engine.save_theta(directory)

    template = learned_optimizer.init(jax.random.key(99))
    fresh = L2OEngine(learned_optimizer, family)
    restored = fresh.load_theta(directory, template)
    restored_leaves = jax.tree.leaves(restored)

    assert len(restored_leaves) == len(trained_leaves)
    assert all(jnp.allclose(a, b) for a, b in zip(restored_leaves, trained_leaves, strict=True))
