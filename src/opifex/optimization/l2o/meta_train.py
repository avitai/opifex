"""Meta-training of learned optimisers via Persistent Evolution Strategies (PES).

Meta-training searches for optimiser meta-parameters ``theta`` that minimise the inner task
loss accumulated over an unroll, across a distribution of tasks. Back-propagating through a long
inner unroll is biased (short truncations) or has exploding/chaotic gradients (long
truncations) — Metz et al. 2019 (``arXiv:1810.10180``). PES (Vicol, Metz & Sohl-Dickstein 2021,
``arXiv:2112.13835``) instead estimates the meta-gradient with antithetic Gaussian perturbations
of ``theta`` over short truncations, while keeping a **persistent accumulator** of the
perturbations across truncation boundaries so the estimate is unbiased w.r.t. the full-horizon
objective. Faithful to ``learned_optimization/outer_trainers/truncated_pes.py``
(``compute_pes_grad``): ``es_grad = (1 / (2 std**2)) * delta_loss * accumulator``.

This implementation runs ``num_tasks`` inner problems in parallel (``jax.vmap``). Each trajectory
is started at a random clock offset in ``[0, total_horizon)``
(``random_initial_iteration_offset`` in ``learned_optimization``'s ``lopt_truncated_step``) and is
reset **per inner step** when its clock reaches ``total_horizon``; the truncation's meta-gradient
is split at that reset (``has_finished = cumsum(is_done) > 0``) so the pre-reset losses attribute
to the full accumulator and the post-reset losses only to the new perturbation. Staggering the
truncations so the parallel tasks are not phase-aligned is load-bearing
(``learned_optimization/outer_trainers/truncation_schedule.py``): it removes the sawtooth that a
synchronous reset would imprint on the meta-loss and lowers the PES gradient variance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from flax import struct


if TYPE_CHECKING:
    from jaxtyping import PyTree

    from opifex.optimization.l2o.core import Task, TaskFamily
    from opifex.optimization.l2o.learned import LearnedOptimizer
    from opifex.optimization.l2o.optimizers import Optimizer


class PESState(struct.PyTreeNode):
    """Persistent PES state, batched over ``num_tasks`` inner problems.

    ``task_keys`` fixes which task each parallel trajectory optimises (so the persistent inner
    state corresponds to a consistent task); ``inner_state`` is the batched inner optimiser
    state; ``accumulator`` is the batched, theta-shaped sum of perturbations since the last
    horizon reset; ``inner_step`` is the per-task inner-step count in the current trajectory
    (shape ``(num_tasks,)``) — per-task so resets stagger across the parallel trajectories.
    """

    task_keys: jax.Array
    inner_state: PyTree
    accumulator: PyTree
    inner_step: jax.Array


def _sample_perturbation(theta: PyTree, key: jax.Array, std: float) -> PyTree:
    """Sample a Gaussian perturbation with std ``std`` matching ``theta``'s pytree."""
    leaves, treedef = jax.tree.flatten(theta)
    keys = jax.random.split(key, len(leaves))
    perturbed = [
        std * jax.random.normal(k, leaf.shape) for leaf, k in zip(leaves, keys, strict=True)
    ]
    return jax.tree.unflatten(treedef, perturbed)


def _tree_axpy(alpha: float, x: PyTree, y: PyTree) -> PyTree:
    """Return ``alpha * x + y`` over matching pytrees."""
    return jax.tree.map(lambda a, b: alpha * a + b, x, y)


def _truncated_unroll(
    optimizer: Optimizer,
    inner_state: PyTree,
    inner_step: jax.Array,
    task: Task,
    key: jax.Array,
    length: int,
    total_horizon: int,
) -> tuple[PyTree, jax.Array, jax.Array, jax.Array]:
    """Run ``length`` inner steps with a per-step horizon reset (one task).

    Faithful to ``learned_optimization`` truncated unrolls: ``inner_step`` is the per-task clock;
    at each step the (pre-update) normalised loss is recorded, the optimiser steps, and when the
    clock reaches ``total_horizon`` the trajectory re-initialises (and the clock zeroes) *mid
    truncation*. Returns ``(final_state, final_inner_step, per_step_losses, per_step_is_done)``.
    Resets depend only on the shared clock, so the antithetic +/- unrolls reset identically.
    """

    def step(
        carry: tuple[PyTree, jax.Array], step_key: jax.Array
    ) -> tuple[tuple[PyTree, jax.Array], tuple[jax.Array, jax.Array]]:
        state, clock = carry
        params = optimizer.get_params(state)
        loss, grad = task.loss_and_grad(params, step_key)
        updated = optimizer.update(state, grad, loss=loss)
        new_clock = clock + 1
        is_done = new_clock >= total_horizon
        # Re-initialise from a fresh inner problem when the horizon is reached. The reset state
        # is independent of ``theta`` (only the update rule is), so +/- particles stay coupled.
        reset_state = optimizer.init(
            task.init(jax.random.fold_in(step_key, 1)), num_steps=total_horizon
        )
        next_state = jax.tree.map(lambda r, u: jnp.where(is_done, r, u), reset_state, updated)
        next_clock = jnp.where(is_done, 0, new_clock)
        return (next_state, next_clock), (task.normalizer(loss), is_done)

    step_keys = jax.random.split(key, length)
    (final_state, final_clock), (losses, is_done) = jax.lax.scan(
        step, (inner_state, inner_step), step_keys
    )
    return final_state, final_clock, losses, is_done


def pes_gradient_step(
    learned_optimizer: LearnedOptimizer,
    task_family: TaskFamily,
    theta: PyTree,
    pes_state: PESState,
    key: jax.Array,
    *,
    std: float,
    trunc_length: int,
    total_horizon: int,
) -> tuple[jax.Array, PyTree, PESState]:
    """One PES truncation: return ``(mean_loss, meta_gradient, new_pes_state)``.

    Faithful to ``truncated_pes.compute_pes_grad``. Antithetic perturbations ``theta +/- pos`` are
    unrolled ``trunc_length`` steps from the persistent inner state; the per-step delta-losses are
    split at the (per-step) horizon reset by ``has_finished = cumsum(is_done) > 0``: losses *before*
    the reset attribute to the running ``accumulator`` (all perturbations since the last reset),
    losses *after* attribute only to the current perturbation ``pos``. The accumulator carries
    ``pos`` forward, or restarts at ``pos`` if the trajectory reset during this truncation.
    """
    num_tasks = pes_state.task_keys.shape[0]
    pert_keys = jax.random.split(jax.random.fold_in(key, 0), num_tasks)
    unroll_keys = jax.random.split(jax.random.fold_in(key, 1), num_tasks)
    factor = 1.0 / (2.0 * std**2)

    def per_task(
        task_key: jax.Array,
        pert_key: jax.Array,
        unroll_key: jax.Array,
        inner_state: PyTree,
        inner_step: jax.Array,
        accumulator: PyTree,
    ) -> tuple[PyTree, PyTree, jax.Array, PyTree, jax.Array]:
        task = task_family.sample(task_key)
        pos = _sample_perturbation(theta, pert_key, std)
        optimizer_pos = learned_optimizer.opt_fn(_tree_axpy(1.0, pos, theta))
        optimizer_neg = learned_optimizer.opt_fn(_tree_axpy(-1.0, pos, theta))

        state_pos, step_pos, losses_pos, is_done = _truncated_unroll(
            optimizer_pos, inner_state, inner_step, task, unroll_key, trunc_length, total_horizon
        )
        _, _, losses_neg, _ = _truncated_unroll(
            optimizer_neg, inner_state, inner_step, task, unroll_key, trunc_length, total_horizon
        )

        delta_losses = losses_pos - losses_neg  # (trunc_length,)
        has_finished = jnp.cumsum(is_done.astype(jnp.int32)) > 0
        before = jnp.sum(delta_losses * (1.0 - has_finished)) / trunc_length
        after = jnp.sum(delta_losses * has_finished) / trunc_length

        new_accumulator = _tree_axpy(1.0, accumulator, pos)  # pos + accumulator
        es_grad = jax.tree.map(
            lambda accum, perturb: factor * (before * accum + after * perturb),
            new_accumulator,
            pos,
        )
        # If the trajectory reset during this truncation, only the current perturbation survives.
        reset_happened = has_finished[-1]
        carried_accumulator = jax.tree.map(
            lambda perturb, accum: jnp.where(reset_happened, perturb, accum), pos, new_accumulator
        )
        mean_loss = (jnp.mean(losses_pos) + jnp.mean(losses_neg)) / 2.0
        return es_grad, state_pos, step_pos, carried_accumulator, mean_loss

    es_grads, states_pos, steps_pos, accumulators, losses = jax.vmap(per_task)(
        pes_state.task_keys,
        pert_keys,
        unroll_keys,
        pes_state.inner_state,
        pes_state.inner_step,
        pes_state.accumulator,
    )
    meta_gradient = jax.tree.map(lambda g: jnp.mean(g, axis=0), es_grads)
    new_pes_state = pes_state.replace(
        inner_state=states_pos, inner_step=steps_pos, accumulator=accumulators
    )
    return jnp.mean(losses), meta_gradient, new_pes_state


def _fresh_inner_state(
    learned_optimizer: LearnedOptimizer,
    task_family: TaskFamily,
    theta: PyTree,
    task_keys: jax.Array,
    key: jax.Array,
    total_horizon: int,
) -> PyTree:
    """Batched fresh inner optimiser state (one per task), all at inner step 0."""
    optimizer = learned_optimizer.opt_fn(theta)
    init_keys = jax.random.split(key, task_keys.shape[0])

    def init_one(task_key: jax.Array, init_key: jax.Array) -> PyTree:
        task = task_family.sample(task_key)
        return optimizer.init(task.init(init_key), num_steps=total_horizon)

    return jax.vmap(init_one)(task_keys, init_keys)


def init_pes_state(
    learned_optimizer: LearnedOptimizer,
    task_family: TaskFamily,
    theta: PyTree,
    key: jax.Array,
    *,
    num_tasks: int,
    total_horizon: int,
    trunc_length: int = 1,  # noqa: ARG001 - kept for interface symmetry; offsets span the horizon
) -> PESState:
    """Build the initial :class:`PESState` with per-task staggered truncation phases.

    Each trajectory starts from a fresh inner state but with a random clock offset in
    ``[0, total_horizon)`` (``random_initial_iteration_offset`` in the reference
    ``lopt_truncated_step``): because the per-step reset in :func:`pes_gradient_step` then fires at
    the staggered clock crossings, the parallel trajectories are never phase-aligned, which removes
    the sawtooth a synchronous reset would imprint on the meta-loss and lowers the PES variance.
    """
    task_key, init_key, offset_key = jax.random.split(key, 3)
    task_keys = jax.random.split(task_key, num_tasks)
    offsets = jax.random.randint(offset_key, (num_tasks,), 0, total_horizon)
    inner_state = _fresh_inner_state(
        learned_optimizer, task_family, theta, task_keys, init_key, total_horizon
    )
    accumulator = jax.vmap(lambda _: jax.tree.map(jnp.zeros_like, theta))(jnp.arange(num_tasks))
    return PESState(
        task_keys=task_keys,
        inner_state=inner_state,
        accumulator=accumulator,
        inner_step=offsets.astype(jnp.int32),
    )


def meta_train(  # noqa: PLR0913 - distinct meta-training hyperparameters
    learned_optimizer: LearnedOptimizer,
    task_family: TaskFamily,
    key: jax.Array,
    *,
    num_outer_steps: int = 1000,
    num_tasks: int = 16,
    trunc_length: int = 20,
    total_horizon: int = 100,
    perturbation_std: float = 0.01,
    meta_learning_rate: float = 3e-3,
) -> tuple[PyTree, jax.Array]:
    """Meta-train ``learned_optimizer`` on ``task_family`` with PES + outer Adam.

    Returns the trained ``theta`` and the per-outer-step mean loss curve. PES does not
    back-propagate through the unroll: ``theta`` is updated by the ES estimate fed to Adam.
    """
    init_key, train_key = jax.random.split(key)
    theta = learned_optimizer.init(init_key)
    meta_optimizer = optax.adam(meta_learning_rate)
    opt_state = meta_optimizer.init(theta)
    pes_state = init_pes_state(
        learned_optimizer,
        task_family,
        theta,
        init_key,
        num_tasks=num_tasks,
        total_horizon=total_horizon,
        trunc_length=trunc_length,
    )

    @jax.jit
    def outer_step(
        carry: tuple[PyTree, optax.OptState, PESState], step_key: jax.Array
    ) -> tuple[tuple[PyTree, optax.OptState, PESState], jax.Array]:
        theta, opt_state, pes_state = carry
        loss, grad, pes_state = pes_gradient_step(
            learned_optimizer,
            task_family,
            theta,
            pes_state,
            step_key,
            std=perturbation_std,
            trunc_length=trunc_length,
            total_horizon=total_horizon,
        )
        updates, opt_state = meta_optimizer.update(grad, opt_state)
        theta = optax.apply_updates(theta, updates)
        return (theta, opt_state, pes_state), loss

    step_keys = jax.random.split(train_key, num_outer_steps)
    (theta, _, _), losses = jax.lax.scan(outer_step, (theta, opt_state, pes_state), step_keys)
    return theta, losses


__all__ = ["PESState", "init_pes_state", "meta_train", "pes_gradient_step"]
