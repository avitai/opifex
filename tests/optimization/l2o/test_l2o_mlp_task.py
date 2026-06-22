"""TDD contracts for the small-MLP task family (the canonical L2O showcase task).

Mirrors the ``MLPTask`` used throughout Google's ``learned_optimization`` tutorials
(``docs/notebooks/no_dependency_learned_optimizer``): a small multilayer perceptron trained by
the inner optimiser. Unlike the convex ``QuadraticTask``, this is a genuinely non-convex
neural-network training objective — the setting where learned optimisers demonstrably beat
fixed-hyperparameter baselines (Metz et al. 2020, arXiv:2009.11243). The family is self-contained
(teacher-student regression on synthetic Gaussian data; no dataset dependency).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.optimization.l2o.optimizers import OptaxOptimizer
from opifex.optimization.l2o.tasks import MLPTaskFamily


def test_mlp_task_init_structure_and_finite_loss() -> None:
    """init returns a per-layer (weight, bias) pytree; the loss is finite and non-negative."""
    family = MLPTaskFamily(input_dim=4, hidden_dim=16, output_dim=3, num_data=32)
    task = family.sample(jax.random.key(0))
    params = task.init(jax.random.key(1))

    # Two layers: (input->hidden), (hidden->output).
    assert len(params) == 2
    (w0, b0), (w1, b1) = params
    assert w0.shape == (4, 16)
    assert b0.shape == (16,)
    assert w1.shape == (16, 3)
    assert b1.shape == (3,)

    loss = task.loss(params, jax.random.key(2))
    assert jnp.isfinite(loss)
    assert float(loss) >= 0.0


def test_mlp_task_is_realizable_low_loss_at_teacher() -> None:
    """The student matches the teacher exactly at the generating parameters (loss ~ 0)."""
    family = MLPTaskFamily(input_dim=4, hidden_dim=16, output_dim=2, num_data=64)
    task = family.sample(jax.random.key(0))
    loss_at_teacher = task.loss(task.teacher_params, jax.random.key(0))
    assert float(loss_at_teacher) < 1e-8


def test_mlp_task_loss_is_stochastic_over_minibatches() -> None:
    """Different keys draw different minibatches, so the loss varies (stochastic gradients)."""
    family = MLPTaskFamily(input_dim=4, hidden_dim=16, output_dim=2, num_data=128, batch_size=16)
    task = family.sample(jax.random.key(0))
    params = task.init(jax.random.key(1))
    loss_a = task.loss(params, jax.random.key(2))
    loss_b = task.loss(params, jax.random.key(3))
    assert not jnp.allclose(loss_a, loss_b)


def test_mlp_normalizer_is_order_one_at_initialization() -> None:
    """The normalizer maps the expected initial loss to roughly O(1) across seeds."""
    family = MLPTaskFamily(input_dim=6, hidden_dim=32, output_dim=4, num_data=64)
    task = family.sample(jax.random.key(0))
    losses = [
        task.normalizer(task.loss(task.init(jax.random.key(i)), jax.random.key(0)))
        for i in range(32)
    ]
    mean_normalised = float(jnp.mean(jnp.array(losses)))
    assert 0.3 < mean_normalised < 3.0


def test_mlp_task_family_sample_varies() -> None:
    """Distinct draws give distinct teachers and datasets."""
    family = MLPTaskFamily(input_dim=4, hidden_dim=8, output_dim=2, num_data=16)
    task_a = family.sample(jax.random.key(0))
    task_b = family.sample(jax.random.key(1))
    assert not jnp.allclose(task_a.inputs, task_b.inputs)


def test_mlp_optimizer_reduces_loss() -> None:
    """A tuned optax optimiser drives the non-convex MLP loss down through the interface."""
    task = MLPTaskFamily(input_dim=4, hidden_dim=16, output_dim=2, num_data=32).sample(
        jax.random.key(0)
    )
    import optax

    optimizer = OptaxOptimizer(optax.adam(0.05))
    state = optimizer.init(task.init(jax.random.key(1)), num_steps=200)
    eval_key = jax.random.key(2)
    initial_loss = task.loss(optimizer.get_params(state), eval_key)
    for _ in range(200):
        params = optimizer.get_params(state)
        loss, grad = task.loss_and_grad(params, eval_key)
        state = optimizer.update(state, grad, loss=loss)
    final_loss = task.loss(optimizer.get_params(state), eval_key)
    assert final_loss < initial_loss


def test_mlp_task_sample_and_loss_are_vmap_safe() -> None:
    """sample-from-traced-key + loss must be vmap-safe (the meta-training inner path)."""
    family = MLPTaskFamily(input_dim=4, hidden_dim=8, output_dim=2, num_data=16)

    def initial_loss(key: jax.Array) -> jax.Array:
        task_key, init_key = jax.random.split(key)
        task = family.sample(task_key)
        return task.loss(task.init(init_key), init_key)

    losses = jax.vmap(initial_loss)(jax.random.split(jax.random.key(0), 5))
    assert losses.shape == (5,)
    assert jnp.all(jnp.isfinite(losses))
