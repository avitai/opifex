"""Contracts for the per-parameter MLP learned optimiser (P2b).

The learned optimiser must: produce ``theta`` as a clean pytree, run a coordinatewise inner
loop on a task, be jit-safe, and be vmap-safe over a batch of ``theta`` (the prerequisite for
PES meta-training). Reference: ``learned_optimization/learned_optimizers/mlp_lopt.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx  # noqa: TC002 — kept eager per project convention

from opifex.optimization.l2o.core import Task  # noqa: TC001 — eager type
from opifex.optimization.l2o.learned import MLPLearnedOptimizer
from opifex.optimization.l2o.optimizers import Optimizer  # noqa: TC001 — eager type
from opifex.optimization.l2o.tasks import QuadraticTaskFamily


def _inner_loop(optimizer: Optimizer, task: Task, steps: int = 25) -> jax.Array:
    """Run ``steps`` of an :class:`Optimizer` on ``task``; return the final loss."""
    state = optimizer.init(task.init(jax.random.key(0)), num_steps=steps)
    key = jax.random.key(1)
    for _ in range(steps):
        params = optimizer.get_params(state)
        loss, grad = task.loss_and_grad(params, key)
        state = optimizer.update(state, grad, loss=loss)
    return task.loss(optimizer.get_params(state), key)


def test_init_returns_pytree_theta() -> None:
    """theta is a non-empty pytree of float arrays (the MLP parameters)."""
    lopt = MLPLearnedOptimizer()
    theta = lopt.init(jax.random.key(0))
    leaves = jax.tree.leaves(theta)
    assert len(leaves) > 0
    assert all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in leaves)


def test_inner_loop_runs_and_is_finite() -> None:
    """An (untrained) learned optimiser runs a coordinatewise inner loop without NaNs."""
    lopt = MLPLearnedOptimizer()
    theta = lopt.init(jax.random.key(0))
    task = QuadraticTaskFamily(dim=8).sample(jax.random.key(1))
    final_loss = _inner_loop(lopt.opt_fn(theta), task)
    assert jnp.isfinite(final_loss)


def test_inner_step_is_jit_safe() -> None:
    """A single learned-optimiser update jits (theta traced, graphdef static)."""
    lopt = MLPLearnedOptimizer()
    theta = lopt.init(jax.random.key(0))
    task = QuadraticTaskFamily(dim=6).sample(jax.random.key(1))
    optimizer = lopt.opt_fn(theta)
    state = optimizer.init(task.init(jax.random.key(2)))

    @jax.jit
    def step(state: object) -> object:
        params = optimizer.get_params(state)
        grad = jax.grad(lambda p: task.loss(p, jax.random.key(0)))(params)
        return optimizer.update(state, grad)

    new_state = step(state)
    assert jnp.isfinite(task.loss(optimizer.get_params(new_state), jax.random.key(0)))


def test_vmap_over_theta_batch() -> None:
    """opt_fn is vmap-safe over a batch of theta (the PES antithetic-perturbation path)."""
    lopt = MLPLearnedOptimizer()
    base_theta = lopt.init(jax.random.key(0))
    task = QuadraticTaskFamily(dim=5).sample(jax.random.key(1))
    start = task.init(jax.random.key(2))

    # A batch of perturbed thetas (as PES would build).
    def perturb(seed: int) -> nnx.State:
        noise = jax.tree.map(
            lambda leaf, i=seed: 0.01 * jax.random.normal(jax.random.key(i), leaf.shape),
            base_theta,
        )
        return jax.tree.map(jnp.add, base_theta, noise)

    thetas = jax.tree.map(lambda *xs: jnp.stack(xs), *[perturb(i) for i in range(4)])

    def one_step_loss(theta: nnx.State) -> jax.Array:
        optimizer = lopt.opt_fn(theta)
        state = optimizer.init(start)
        grad = jax.grad(lambda p: task.loss(p, jax.random.key(0)))(start)
        state = optimizer.update(state, grad)
        return task.loss(optimizer.get_params(state), jax.random.key(0))

    losses = jax.vmap(one_step_loss)(thetas)
    assert losses.shape == (4,)
    assert jnp.all(jnp.isfinite(losses))


def test_theta_round_trips_through_merge() -> None:
    """theta split out and merged back reproduces the same update (serialisable state)."""
    lopt = MLPLearnedOptimizer()
    theta = lopt.init(jax.random.key(0))
    # Round-trip theta through a tree flatten/unflatten (what serialisation does).
    flat, treedef = jax.tree.flatten(theta)
    restored = jax.tree.unflatten(treedef, flat)

    task = QuadraticTaskFamily(dim=4).sample(jax.random.key(1))
    loss_a = _inner_loop(lopt.opt_fn(theta), task, steps=10)
    loss_b = _inner_loop(lopt.opt_fn(restored), task, steps=10)
    assert jnp.allclose(loss_a, loss_b)
