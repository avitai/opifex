"""TDD contracts for the rebuilt L2O core abstractions (tasks + optimiser interface).

References: ``learned_optimization/tasks/base.py`` (Task/TaskFamily),
``learned_optimization/optimizers/base.py`` (Optimizer ABC).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from opifex.optimization.l2o.core import single_task_to_family
from opifex.optimization.l2o.optimizers import OptaxOptimizer, OptaxOptState
from opifex.optimization.l2o.tasks import QuadraticTaskFamily


def test_quadratic_task_init_shape_and_loss_minimum() -> None:
    """init returns the right shape; loss is minimised (==0) at the optimum."""
    family = QuadraticTaskFamily(dim=6)
    task = family.sample(jax.random.key(0))

    params = task.init(jax.random.key(1))
    assert params.shape == (6,)

    loss_at_start = task.loss(params, jax.random.key(2))
    loss_at_optimum = task.loss(task.optimum, jax.random.key(2))
    assert jnp.isfinite(loss_at_start)
    assert float(loss_at_optimum) == 0.0
    assert loss_at_optimum <= loss_at_start


def test_quadratic_normalizer_is_order_one_at_initialization() -> None:
    """The loss-scale normalizer maps the expected initial loss to roughly O(1)."""
    family = QuadraticTaskFamily(dim=20, max_log_condition=3.0)
    task = family.sample(jax.random.key(0))
    # Average normalised initial loss over several starts should be ~1.
    losses = [
        task.normalizer(task.loss(task.init(jax.random.key(i)), jax.random.key(0)))
        for i in range(64)
    ]
    mean_normalised = float(jnp.mean(jnp.array(losses)))
    assert 0.3 < mean_normalised < 3.0


def test_task_family_sample_varies_and_is_spd() -> None:
    """Distinct draws give distinct tasks; the curvature matrix is SPD."""
    family = QuadraticTaskFamily(dim=8)
    task_a = family.sample(jax.random.key(0))
    task_b = family.sample(jax.random.key(1))
    assert not jnp.allclose(task_a.optimum, task_b.optimum)
    eigenvalues = jnp.linalg.eigvalsh(task_a.matrix)
    assert float(jnp.min(eigenvalues)) > 0.0  # strictly positive definite


def test_single_task_to_family_returns_same_task() -> None:
    """single_task_to_family yields a family whose draws are the wrapped task."""
    task = QuadraticTaskFamily(dim=4).sample(jax.random.key(0))
    family = single_task_to_family(task)
    assert family.sample(jax.random.key(1)) is task


def test_optax_optimizer_reduces_quadratic_loss() -> None:
    """An optax-wrapped optimiser drives the quadratic loss down through the interface."""
    task = QuadraticTaskFamily(dim=10).sample(jax.random.key(0))
    optimizer = OptaxOptimizer(optax.adam(0.1))
    state = optimizer.init(task.init(jax.random.key(1)), num_steps=300)

    eval_key = jax.random.key(2)
    initial_loss = task.loss(optimizer.get_params(state), eval_key)
    for _ in range(300):
        params = optimizer.get_params(state)
        loss, grad = task.loss_and_grad(params, eval_key)
        state = optimizer.update(state, grad, loss=loss)
    final_loss = task.loss(optimizer.get_params(state), eval_key)

    assert final_loss < initial_loss
    assert float(final_loss) < 1e-3  # adam converges the convex quadratic


def test_optax_optimizer_update_is_jit_safe() -> None:
    """One optimiser step is jit-compilable (transform compatibility)."""
    task = QuadraticTaskFamily(dim=10).sample(jax.random.key(0))
    optimizer = OptaxOptimizer(optax.sgd(0.05))
    state = optimizer.init(task.init(jax.random.key(1)))

    @jax.jit
    def step(state: OptaxOptState) -> OptaxOptState:
        params = optimizer.get_params(state)
        grad = jax.grad(lambda p: task.loss(p, jax.random.key(0)))(params)
        return optimizer.update(state, grad)

    new_state = step(state)
    assert jnp.isfinite(task.loss(optimizer.get_params(new_state), jax.random.key(0)))
