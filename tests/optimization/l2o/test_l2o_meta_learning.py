"""TDD contracts for gradient-based meta-learning (MAML, Reptile) on the L2O Task abstraction.

These meta-learners learn an *initialisation* that adapts to any task in a family within a few
inner gradient steps — distinct from the learned-optimiser L2O (which learns an update rule).
References: Finn et al. 2017 (MAML, ``arXiv:1703.03400``); Nichol et al. 2018 (Reptile,
``arXiv:1803.02999``). In SciML this meta-learns PINN/operator initialisations across PDE-parameter
families (e.g. Penwarden et al. 2023).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.optimization.l2o.meta_learning import adapt, maml_meta_train, reptile_meta_train
from opifex.optimization.l2o.tasks import QuadraticTaskFamily


def _mean_query_loss_after_adapt(family, init_params, keys, *, inner_steps, inner_lr):
    """Mean post-adaptation loss over held-out tasks for a given initialisation."""

    def per_task(key: jax.Array) -> jax.Array:
        task = family.sample(key)
        adapted = adapt(
            task,
            init_params,
            jax.random.fold_in(key, 1),
            inner_steps=inner_steps,
            inner_lr=inner_lr,
        )
        return task.loss(adapted, jax.random.fold_in(key, 2))

    return jnp.mean(jax.vmap(per_task)(keys))


def test_adapt_reduces_task_loss() -> None:
    """A few inner gradient steps reduce a single task's loss from a fixed start."""
    family = QuadraticTaskFamily(dim=5, max_log_condition=1.0)
    task = family.sample(jax.random.key(0))
    start = jnp.zeros(5)
    before = task.loss(start, jax.random.key(1))
    adapted = adapt(task, start, jax.random.key(1), inner_steps=10, inner_lr=0.1)
    after = task.loss(adapted, jax.random.key(1))
    assert after < before


def test_maml_meta_training_improves_few_shot_adaptation() -> None:
    """A MAML-meta-trained init adapts to held-out tasks better than the un-trained init."""
    family = QuadraticTaskFamily(dim=5, max_log_condition=1.0)
    init_params = jnp.ones(5) * 3.0  # deliberately far from the optimum distribution (mean 0)

    meta_params, losses = maml_meta_train(
        family,
        init_params,
        jax.random.key(0),
        num_outer_steps=200,
        num_tasks=8,
        inner_steps=3,
        inner_lr=0.05,
        meta_lr=0.05,
    )
    assert jnp.all(jnp.isfinite(losses))

    held_out = jax.random.split(jax.random.key(999), 32)
    base = _mean_query_loss_after_adapt(family, init_params, held_out, inner_steps=3, inner_lr=0.05)
    meta = _mean_query_loss_after_adapt(family, meta_params, held_out, inner_steps=3, inner_lr=0.05)
    assert meta < base


def test_first_order_maml_runs_and_improves() -> None:
    """First-order MAML (no second-order meta-grad) also lowers the meta-loss."""
    family = QuadraticTaskFamily(dim=4, max_log_condition=1.0)
    init_params = jnp.ones(4) * 2.0
    _, losses = maml_meta_train(
        family,
        init_params,
        jax.random.key(0),
        num_outer_steps=150,
        num_tasks=8,
        inner_steps=3,
        inner_lr=0.05,
        meta_lr=0.05,
        first_order=True,
    )
    assert jnp.mean(losses[-20:]) < jnp.mean(losses[:20])


def test_reptile_meta_training_improves_few_shot_adaptation() -> None:
    """A Reptile-meta-trained init adapts to held-out tasks better than the un-trained init."""
    family = QuadraticTaskFamily(dim=5, max_log_condition=1.0)
    init_params = jnp.ones(5) * 3.0

    meta_params, losses = reptile_meta_train(
        family,
        init_params,
        jax.random.key(0),
        num_outer_steps=200,
        num_tasks=8,
        inner_steps=5,
        inner_lr=0.05,
        meta_lr=0.1,
    )
    assert jnp.all(jnp.isfinite(losses))

    held_out = jax.random.split(jax.random.key(999), 32)
    base = _mean_query_loss_after_adapt(family, init_params, held_out, inner_steps=5, inner_lr=0.05)
    meta = _mean_query_loss_after_adapt(family, meta_params, held_out, inner_steps=5, inner_lr=0.05)
    assert meta < base


def test_maml_init_moves_toward_mean_optimum() -> None:
    """On a zero-mean optimum family, the MAML init converges toward 0 (the mean optimum)."""
    family = QuadraticTaskFamily(dim=3, max_log_condition=0.0)  # A = I; optima ~ N(0, I)
    init_params = jnp.array([4.0, -4.0, 4.0])
    meta_params, _ = maml_meta_train(
        family,
        init_params,
        jax.random.key(0),
        num_outer_steps=300,
        num_tasks=16,
        inner_steps=2,
        inner_lr=0.1,
        meta_lr=0.1,
    )
    # The meta-init must end up much closer to the mean optimum (0) than it started.
    assert jnp.linalg.norm(meta_params) < jnp.linalg.norm(init_params) / 2.0


def test_maml_meta_step_is_jit_safe() -> None:
    """The full MAML meta-training is jit-compiled internally and returns finite results."""
    family = QuadraticTaskFamily(dim=4, max_log_condition=1.0)
    meta_params, losses = maml_meta_train(
        family,
        jnp.zeros(4),
        jax.random.key(0),
        num_outer_steps=20,
        num_tasks=4,
        inner_steps=2,
        inner_lr=0.05,
        meta_lr=0.05,
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(meta_params))
    assert jnp.all(jnp.isfinite(losses))
