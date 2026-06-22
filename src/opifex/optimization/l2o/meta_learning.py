"""Gradient-based meta-learning: MAML and Reptile over a :class:`TaskFamily`.

Unlike the learned-optimiser L2O (which meta-learns an *update rule*; see ``meta_train.py``),
these methods meta-learn an *initialisation* ``theta`` such that a few ordinary inner gradient
steps adapt it to any task drawn from the family. This is the dominant meta-learning recipe in
SciML for amortising across PDE-parameter families — e.g. meta-learning a PINN initialisation so a
new PDE instance converges in a handful of steps.

- **MAML** (Finn, Abbeel & Levine 2017, ``arXiv:1703.03400``): the meta-objective is the
  post-adaptation loss, and the meta-gradient differentiates *through* the inner adaptation steps
  (second order). ``first_order=True`` selects FOMAML, which stops the gradient through the inner
  trajectory (Finn 2017 §5.2) — cheaper, usually competitive.
- **Reptile** (Nichol, Achiam & Schulman 2018, ``arXiv:1803.02999``): a first-order method whose
  meta-update moves ``theta`` towards each task's inner-adapted parameters,
  ``theta <- theta + meta_lr * mean_tasks(adapted - theta)`` — no differentiation through the
  unroll.

All tasks in the family must share the optimisee-parameter structure (same model / dimension);
``theta`` is a single pytree of that structure. Every stochastic boundary takes a caller-owned
``key``; the inner loop uses ``Task.loss`` (the real objective), never a placeholder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax


if TYPE_CHECKING:
    from jaxtyping import PyTree

    from opifex.optimization.l2o.core import Task, TaskFamily


def _sgd_adapt(
    task: Task, params: PyTree, key: jax.Array, inner_steps: int, inner_lr: float
) -> PyTree:
    """Run ``inner_steps`` plain-SGD steps of ``task.loss`` from ``params`` (differentiable)."""

    def step(current: PyTree, step_key: jax.Array) -> tuple[PyTree, None]:
        _loss, grad = task.loss_and_grad(current, step_key)
        return jax.tree.map(lambda p, g: p - inner_lr * g, current, grad), None

    adapted, _ = jax.lax.scan(step, params, jax.random.split(key, inner_steps))
    return adapted


def adapt(
    task: Task, init_params: PyTree, key: jax.Array, *, inner_steps: int, inner_lr: float
) -> PyTree:
    """Few-shot adaptation: ``inner_steps`` SGD steps of ``task`` from ``init_params``."""
    return _sgd_adapt(task, init_params, key, inner_steps, inner_lr)


def _maml_task_query_loss(
    learned_init: PyTree,
    task: Task,
    key: jax.Array,
    inner_steps: int,
    inner_lr: float,
    *,
    first_order: bool,
) -> jax.Array:
    """Post-adaptation (query) loss for one task; the MAML per-task meta-objective.

    Support and query draw separate keys so the meta-objective rewards generalisation rather than
    memorising the adaptation batch. ``first_order`` stops the gradient through the inner unroll.
    """
    support_key, query_key = jax.random.split(key)
    if not first_order:
        # Full MAML: differentiate the query loss through the inner adaptation.
        adapted = _sgd_adapt(task, learned_init, support_key, inner_steps, inner_lr)
        return task.loss(adapted, query_key)
    # FOMAML: adapt with no gradient path, then attach an identity dependence on ``learned_init``
    # (value unchanged, derivative = I) so the meta-gradient is the query gradient at the adapted
    # params (Finn 2017 §5.2) without backpropagating through the inner trajectory.
    adapted = jax.lax.stop_gradient(
        _sgd_adapt(task, jax.lax.stop_gradient(learned_init), support_key, inner_steps, inner_lr)
    )
    phi = jax.tree.map(lambda a, i: a + (i - jax.lax.stop_gradient(i)), adapted, learned_init)
    return task.loss(phi, query_key)


def maml_meta_train(  # noqa: PLR0913 - distinct meta-learning hyperparameters
    task_family: TaskFamily,
    init_params: PyTree,
    key: jax.Array,
    *,
    num_outer_steps: int = 1000,
    num_tasks: int = 16,
    inner_steps: int = 5,
    inner_lr: float = 0.01,
    meta_lr: float = 1e-3,
    first_order: bool = False,
) -> tuple[PyTree, jax.Array]:
    """Meta-train an initialisation with MAML; return ``(meta_params, meta_loss_curve)``.

    Each outer step samples ``num_tasks`` tasks, adapts ``init_params`` with ``inner_steps`` SGD
    steps per task, and takes an outer-Adam step on the mean post-adaptation (query) loss. The
    meta-gradient flows through the inner adaptation (second order) unless ``first_order=True``.
    """
    meta_optimizer = optax.adam(meta_lr)
    opt_state = meta_optimizer.init(init_params)

    def meta_loss(theta: PyTree, step_key: jax.Array) -> jax.Array:
        task_keys = jax.random.split(step_key, num_tasks)

        def per_task(task_key: jax.Array) -> jax.Array:
            task = task_family.sample(task_key)
            return _maml_task_query_loss(
                theta,
                task,
                jax.random.fold_in(task_key, 1),
                inner_steps,
                inner_lr,
                first_order=first_order,
            )

        return jnp.mean(jax.vmap(per_task)(task_keys))

    @jax.jit
    def outer_step(
        carry: tuple[PyTree, optax.OptState], step_key: jax.Array
    ) -> tuple[tuple[PyTree, optax.OptState], jax.Array]:
        theta, opt_state = carry
        loss, grad = jax.value_and_grad(meta_loss)(theta, step_key)
        updates, opt_state = meta_optimizer.update(grad, opt_state)
        return (optax.apply_updates(theta, updates), opt_state), loss

    step_keys = jax.random.split(key, num_outer_steps)
    (meta_params, _), losses = jax.lax.scan(outer_step, (init_params, opt_state), step_keys)
    return meta_params, losses


def reptile_meta_train(
    task_family: TaskFamily,
    init_params: PyTree,
    key: jax.Array,
    *,
    num_outer_steps: int = 1000,
    num_tasks: int = 16,
    inner_steps: int = 5,
    inner_lr: float = 0.01,
    meta_lr: float = 0.1,
) -> tuple[PyTree, jax.Array]:
    """Meta-train an initialisation with Reptile; return ``(meta_params, meta_loss_curve)``.

    Each outer step adapts ``init_params`` on ``num_tasks`` tasks and moves ``theta`` towards the
    mean adapted parameters: ``theta <- theta + meta_lr * mean(adapted - theta)`` (Nichol 2018).
    The reported per-step loss is the mean pre-adaptation task loss at ``theta`` (a comparable
    progress signal across outer steps).
    """

    @jax.jit
    def outer_step(theta: PyTree, step_key: jax.Array) -> tuple[PyTree, jax.Array]:
        task_keys = jax.random.split(step_key, num_tasks)

        def per_task(task_key: jax.Array) -> tuple[PyTree, jax.Array]:
            task = task_family.sample(task_key)
            adapt_key = jax.random.fold_in(task_key, 1)
            adapted = _sgd_adapt(task, theta, adapt_key, inner_steps, inner_lr)
            return adapted, task.loss(theta, jax.random.fold_in(task_key, 2))

        adapted_batch, losses = jax.vmap(per_task)(task_keys)
        mean_adapted = jax.tree.map(lambda a: jnp.mean(a, axis=0), adapted_batch)  # over tasks
        new_theta = jax.tree.map(lambda t, m: t + meta_lr * (m - t), theta, mean_adapted)
        return new_theta, jnp.mean(losses)

    step_keys = jax.random.split(key, num_outer_steps)
    meta_params, losses = jax.lax.scan(outer_step, init_params, step_keys)
    return meta_params, losses


__all__ = ["adapt", "maml_meta_train", "reptile_meta_train"]
