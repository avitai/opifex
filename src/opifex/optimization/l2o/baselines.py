"""Real classical optimisation baselines for honest L2O comparison.

A learned optimiser is only interesting if it beats a *properly tuned* classical optimiser, so
the baselines here are genuine: pure-JAX quasi-Newton minimisers (:class:`BFGSSolver` with zoom
line search) run to convergence, and a tuned first-order optax optimiser (learning-rate swept on
the task) for a step-by-step learning-curve comparison.

Honest benchmarking follows the L2O literature (Andrychowicz et al. 2016; the VeLO-scaling
critique ``arXiv:2310.18191``): compare learning curves and report speedup against a *tuned*
baseline, never an arbitrary one.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003

import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree  # noqa: TC002

from opifex.core.solver.status import is_successful
from opifex.optimization.l2o.core import Task  # noqa: TC001
from opifex.optimization.l2o.optimizers import OptaxOptimizer, Optimizer
from opifex.optimization.second_order.wrappers import BFGSSolver


def bfgs_minimise(
    task: Task,
    start: PyTree,
    *,
    solver: BFGSSolver | None = None,
    max_steps: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    key: jax.Array,
) -> tuple[PyTree, jax.Array]:
    """Run a pure-JAX :class:`BFGSSolver` minimiser on ``task`` from ``start``.

    Every evaluation uses the caller-supplied ``key``, since classical quasi-Newton
    solvers require a deterministic objective. When the solver does not reach a
    successful :class:`~opifex.core.solver.status.Status`, non-converged parameters
    and loss are guarded to ``NaN`` rather than silently treated as optimal.

    Args:
        task: Objective-carrying :class:`Task`.
        start: Initial parameter pytree.
        solver: Optional pre-configured :class:`BFGSSolver`. Mutually exclusive with
            ``max_steps``, ``rtol``, and ``atol``.
        max_steps: Maximum BFGS iterations when ``solver`` is ``None`` (default ``256``).
        rtol: Relative convergence tolerance when ``solver`` is ``None`` (default ``1e-8``).
        atol: Absolute convergence tolerance when ``solver`` is ``None`` (default ``1e-8``).
        key: PRNG key passed to ``task.loss`` on every objective evaluation.

    Returns:
        Tuple ``(minimiser_params, final_loss)`` (``NaN`` if ``~is_successful(status)``).

    Raises:
        ValueError: When both ``solver`` and any of ``(max_steps, rtol, atol)`` are given.
    """
    if solver is not None:
        if any(opt is not None for opt in (max_steps, rtol, atol)):
            raise ValueError("Pass either `solver` or (`max_steps`, `rtol`, `atol`), not both.")
        active_solver = solver
    else:
        active_solver = BFGSSolver(
            max_iterations=256 if max_steps is None else max_steps,
            rtol=1e-8 if rtol is None else rtol,
            atol=1e-8 if atol is None else atol,
        )

    def objective(y: PyTree) -> jax.Array:
        """Evaluate deterministic task loss with the fixed caller key."""
        return task.loss(y, key)

    solution = active_solver.minimise(objective, start)
    ok = is_successful(solution.status)
    raw_params = solution.fields["params"]
    best_params = jax.tree.map(
        lambda leaf: jnp.where(ok, leaf, jnp.full_like(leaf, jnp.nan)),
        raw_params,
    )
    raw_loss = task.loss(raw_params, key)
    final_loss = jnp.where(ok, raw_loss, jnp.asarray(jnp.nan, dtype=raw_loss.dtype))
    return best_params, final_loss


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
        """Advance one optimizer step and record pre-update loss."""
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
        """Compute the loss curve for one candidate learning rate."""
        optimizer = OptaxOptimizer(transformation(learning_rate))
        return loss_curve(optimizer, task, start, num_steps=num_steps, key=key)

    curves = jax.vmap(run)(learning_rates)
    final_losses = jnp.where(jnp.isfinite(curves[:, -1]), curves[:, -1], jnp.inf)
    best = jnp.argmin(final_losses)
    return curves[best], learning_rates[best]


__all__ = ["bfgs_minimise", "loss_curve", "tuned_optax_baseline"]
