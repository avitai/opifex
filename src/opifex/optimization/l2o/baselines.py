"""Real classical optimisation baselines for honest L2O comparison.

A learned optimiser is only interesting if it beats a *properly tuned* classical optimiser, so
the baselines here are genuine: ``optimistix`` second-order/line-search minimisers
(``BFGS``/``GradientDescent``/``NonlinearCG``) run to convergence, and a tuned first-order
optax optimiser (learning-rate swept on the task) for a step-by-step learning-curve comparison.
No fabricated baselines or speedups (cf. the deleted ``_traditional_fallback``).

Honest benchmarking follows the L2O literature (Andrychowicz et al. 2016; the VeLO-scaling
critique arXiv:2310.18191): compare learning curves and report speedup against a *tuned*
baseline, never an arbitrary one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
import optimistix as optx

from opifex.optimization.l2o.optimizers import OptaxOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import PyTree

    from opifex.optimization.l2o.core import Task
    from opifex.optimization.l2o.optimizers import Optimizer


def optimistix_minimise(
    task: Task,
    solver: optx.AbstractMinimiser,
    start: PyTree,
    *,
    max_steps: int = 256,
    key: jax.Array | None = None,
) -> tuple[PyTree, jax.Array]:
    """Run a real ``optimistix`` minimiser on ``task`` from ``start``.

    Adapts ``Task.loss(params, key)`` to optimistix's ``fn(y, args) -> scalar`` (a fixed key,
    since classical solvers need a deterministic objective). Returns ``(minimiser, final_loss)``.
    """
    eval_key = key if key is not None else jax.random.key(0)

    def objective(y: PyTree, args: object) -> jax.Array:  # noqa: ARG001 - optimistix fn(y, args)
        return task.loss(y, eval_key)

    solution = optx.minimise(objective, solver, start, max_steps=max_steps, throw=False)
    return solution.value, task.loss(solution.value, eval_key)


def loss_curve(
    optimizer: Optimizer,
    task: Task,
    start: PyTree,
    *,
    num_steps: int,
    key: jax.Array,
) -> jax.Array:
    """Return the per-step loss curve of an :class:`Optimizer` on ``task``.

    Index 0 is the loss at ``start``; index ``t`` is the loss after ``t`` update steps.
    """
    state = optimizer.init(start, num_steps=num_steps)

    def step(state: object, step_key: jax.Array) -> tuple[object, jax.Array]:
        params = optimizer.get_params(state)
        loss, grad = task.loss_and_grad(params, step_key)
        return optimizer.update(state, grad, loss=loss), loss

    step_keys = jax.random.split(key, num_steps)
    _, losses = jax.lax.scan(step, state, step_keys)
    return jnp.concatenate([task.loss(start, key)[None], losses])


def tuned_optax_baseline(
    task: Task,
    start: PyTree,
    learning_rates: jax.Array,
    *,
    num_steps: int,
    key: jax.Array,
    transformation: Callable[[jax.Array], optax.GradientTransformation] = optax.adam,
) -> tuple[jax.Array, jax.Array]:
    """Sweep ``learning_rates`` and return the best optimiser's ``(loss_curve, learning_rate)``.

    "Tuned" means the learning rate is chosen by a real sweep (best final loss) — the honest
    classical first-order baseline for a learning-curve comparison against a learned optimiser.
    """

    def run(learning_rate: jax.Array) -> jax.Array:
        optimizer = OptaxOptimizer(transformation(learning_rate))
        return loss_curve(optimizer, task, start, num_steps=num_steps, key=key)

    curves = jax.vmap(run)(learning_rates)
    final_losses = jnp.where(jnp.isfinite(curves[:, -1]), curves[:, -1], jnp.inf)
    best = jnp.argmin(final_losses)
    return curves[best], learning_rates[best]


__all__ = ["loss_curve", "optimistix_minimise", "tuned_optax_baseline"]
