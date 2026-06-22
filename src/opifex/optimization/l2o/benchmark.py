"""Honest learned-optimiser benchmarking: learning curves and speedup-at-target.

The primary metric is the loss-vs-step learning curve; the secondary metric is
speedup-at-target-loss = (baseline steps to reach the target) / (learned-optimiser steps),
**censored** when a method never reaches the target.

The baseline is tuned with the standard L2O protocol (Andrychowicz et al. 2016): a *single*
learning rate is selected on a tuning batch from the task family (best mean final loss) and then
applied unchanged to every held-out task. This is the honest comparison — a learned optimiser's
value is precisely that it adapts per-task and per-coordinate without re-tuning, whereas a
per-task learning-rate sweep would be an undeployable oracle. Generalisation claims are scoped to
in-distribution held-out tasks (cf. the VeLO-scaling critique, arXiv:2310.18191).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from opifex.optimization.l2o.baselines import loss_curve
from opifex.optimization.l2o.optimizers import OptaxOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import PyTree

    from opifex.optimization.l2o.core import TaskFamily
    from opifex.optimization.l2o.learned import LearnedOptimizer

# Default learning-rate sweep for the tuned first-order baseline.
DEFAULT_LR_SWEEP: jax.Array = jnp.asarray([3e-3, 1e-2, 3e-2, 1e-1, 3e-1])


def steps_to_target(curve: jax.Array, target: jax.Array) -> jax.Array:
    """First step index at which ``curve`` reaches ``target`` (``inf`` if never)."""
    reached = curve <= target
    first = jnp.argmax(reached)
    return jnp.where(jnp.any(reached), first.astype(jnp.float32), jnp.inf)


def speedup_at_target(
    baseline_curve: jax.Array, candidate_curve: jax.Array, target: jax.Array
) -> jax.Array:
    """Speedup = baseline-steps / candidate-steps to reach ``target`` (censored).

    Returns ``0`` when the candidate never reaches the target, ``inf`` when only the baseline
    fails to — so the value is never fabricated when a method does not converge.
    """
    baseline_steps = steps_to_target(baseline_curve, target)
    candidate_steps = steps_to_target(candidate_curve, target)
    return baseline_steps / candidate_steps


def distribution_tuned_lr(
    task_family: TaskFamily,
    learning_rates: jax.Array,
    key: jax.Array,
    *,
    num_steps: int,
    num_tune_tasks: int = 16,
    transformation: Callable[[jax.Array], optax.GradientTransformation] = optax.adam,
) -> jax.Array:
    """Select one learning rate minimising the mean final loss over a tuning batch.

    The standard L2O baseline protocol: tune a *single* hyperparameter on the task
    distribution, then apply it unchanged to held-out tasks. Returns the chosen scalar rate.
    """
    tune_keys = jax.random.split(key, num_tune_tasks)

    def mean_final_loss(learning_rate: jax.Array) -> jax.Array:
        def per_task(task_key: jax.Array) -> jax.Array:
            task = task_family.sample(task_key)
            start = task.init(jax.random.fold_in(task_key, 1))
            optimizer = OptaxOptimizer(transformation(learning_rate))
            curve = loss_curve(
                optimizer, task, start, num_steps=num_steps, key=jax.random.fold_in(task_key, 2)
            )
            return curve[-1]

        finals = jax.vmap(per_task)(tune_keys)
        return jnp.mean(jnp.where(jnp.isfinite(finals), finals, jnp.inf))

    losses = jax.vmap(mean_final_loss)(learning_rates)
    return learning_rates[jnp.argmin(losses)]


def benchmark_on_held_out_tasks(  # noqa: PLR0913 - distinct benchmark knobs
    learned_optimizer: LearnedOptimizer,
    theta: PyTree,
    task_family: TaskFamily,
    key: jax.Array,
    *,
    num_tasks: int = 16,
    num_steps: int = 100,
    learning_rates: jax.Array = DEFAULT_LR_SWEEP,
    target_fraction: float = 0.1,
    transformation: Callable[[jax.Array], optax.GradientTransformation] = optax.adam,
) -> dict[str, jax.Array]:
    """Meta-test the learned optimiser against a distribution-tuned baseline on held-out tasks.

    A single baseline learning rate is tuned on a separate batch from ``task_family`` and applied
    to every held-out task. For each held-out task: run the learned optimiser and the fixed-rate
    baseline for ``num_steps``, then compute speedup at a per-task target loss (``target_fraction``
    of the baseline's initial loss). Returns mean learning curves, the per-task speedups, the
    median speedup (robust to censored ``inf``/``0`` entries), and the tuned baseline rate.
    """
    optimizer = learned_optimizer.opt_fn(theta)
    tune_key, eval_key = jax.random.split(key)
    baseline_lr = distribution_tuned_lr(
        task_family,
        learning_rates,
        tune_key,
        num_steps=num_steps,
        transformation=transformation,
    )
    task_keys = jax.random.split(eval_key, num_tasks)

    def per_task(task_key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        task = task_family.sample(task_key)
        start = task.init(jax.random.fold_in(task_key, 1))
        run_key = jax.random.fold_in(task_key, 2)
        learned_curve = loss_curve(optimizer, task, start, num_steps=num_steps, key=run_key)
        baseline_optimizer = OptaxOptimizer(transformation(baseline_lr))
        baseline_curve = loss_curve(
            baseline_optimizer, task, start, num_steps=num_steps, key=run_key
        )
        target = target_fraction * baseline_curve[0]
        speedup = speedup_at_target(baseline_curve, learned_curve, target)
        return learned_curve, baseline_curve, speedup

    learned_curves, baseline_curves, speedups = jax.vmap(per_task)(task_keys)
    finite_speedups = jnp.where(jnp.isfinite(speedups), speedups, jnp.nan)
    return {
        "learned_curve_mean": jnp.mean(learned_curves, axis=0),
        "baseline_curve_mean": jnp.mean(baseline_curves, axis=0),
        "baseline_learning_rate": baseline_lr,
        "speedups": speedups,
        "median_speedup": jnp.nanmedian(finite_speedups),
        "fraction_reached_target": jnp.mean((speedups > 0) & jnp.isfinite(speedups)),
    }


__all__ = [
    "DEFAULT_LR_SWEEP",
    "benchmark_on_held_out_tasks",
    "distribution_tuned_lr",
    "speedup_at_target",
    "steps_to_target",
]
