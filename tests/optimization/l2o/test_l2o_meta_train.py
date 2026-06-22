"""Contracts for PES meta-training (P3).

The decisive test is unbiasedness: the antithetic-ES meta-gradient (the core of PES over a full
unroll) must match the exact meta-gradient from autodiff through the unroll. Reference:
``learned_optimization/outer_trainers/truncated_pes.py``; Vicol et al. 2021 (``arXiv:2112.13835``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx  # noqa: TC002 — pyproject dep kept eager (project convention)

from opifex.optimization.l2o.learned import LearnableSGD, MLPLearnedOptimizer
from opifex.optimization.l2o.meta_train import (
    _sample_perturbation,
    _tree_axpy,
    _truncated_unroll,
    meta_train,
    pes_gradient_step,
    PESState,
)
from opifex.optimization.l2o.tasks import QuadraticTaskFamily


def test_antithetic_es_matches_autodiff_gradient() -> None:
    """The full-unroll antithetic-ES meta-gradient equals the autodiff meta-gradient.

    This is the unbiasedness property at the heart of PES (a single full-horizon truncation,
    accumulator == perturbation). Averaged over many perturbations with common random numbers,
    the ES estimate must match ``jax.grad`` through the unroll.
    """
    learned_optimizer = LearnableSGD(initial_learning_rate=0.05)
    theta = learned_optimizer.init(jax.random.key(0))
    task = QuadraticTaskFamily(dim=3).sample(jax.random.key(1))
    start = jnp.ones(3)  # fixed inner start so both estimators see the same trajectory
    horizon = 8
    unroll_key = jax.random.key(2)

    def full_unroll_loss(theta_value: nnx.State) -> jax.Array:
        optimizer = learned_optimizer.opt_fn(theta_value)
        state = optimizer.init(start, num_steps=horizon)
        # A single full-horizon truncation (total_horizon > horizon, so no mid-unroll reset);
        # the mean per-step loss is the differentiable full-unroll meta-loss.
        _, _, losses, _ = _truncated_unroll(
            optimizer, state, jnp.zeros((), jnp.int32), task, unroll_key, horizon, horizon + 1
        )
        return jnp.mean(losses)

    exact_grad = jax.grad(full_unroll_loss)(theta)

    std = 0.02

    def es_sample(perturbation_key: jax.Array) -> object:
        pos = _sample_perturbation(theta, perturbation_key, std)
        loss_pos = full_unroll_loss(_tree_axpy(1.0, pos, theta))
        loss_neg = full_unroll_loss(_tree_axpy(-1.0, pos, theta))
        factor = (loss_pos - loss_neg) / (2.0 * std**2)
        return jax.tree.map(lambda p: factor * p, pos)

    samples = jax.vmap(es_sample)(jax.random.split(jax.random.key(3), 8000))
    es_grad = jax.tree.map(lambda s: jnp.mean(s, axis=0), samples)

    exact_leaf = jax.tree.leaves(exact_grad)[0]
    es_leaf = jax.tree.leaves(es_grad)[0]
    assert jnp.allclose(es_leaf, exact_leaf, rtol=0.1, atol=1e-3)


def test_pes_gradient_step_is_finite_and_jit_safe() -> None:
    """One PES truncation produces a finite meta-gradient and is jit-compilable."""
    learned_optimizer = LearnableSGD(0.05)
    # Well-conditioned family so a fixed lr=0.05 is numerically stable for this smoke check
    # (high-condition tasks legitimately diverge under a fixed step size).
    family = QuadraticTaskFamily(dim=4, max_log_condition=1.0)
    theta = learned_optimizer.init(jax.random.key(0))
    from opifex.optimization.l2o.meta_train import init_pes_state

    pes_state = init_pes_state(
        learned_optimizer, family, theta, jax.random.key(1), num_tasks=8, total_horizon=40
    )

    @jax.jit
    def step(
        theta: nnx.State, pes_state: PESState, key: jax.Array
    ) -> tuple[jax.Array, nnx.State, PESState]:
        return pes_gradient_step(
            learned_optimizer,
            family,
            theta,
            pes_state,
            key,
            std=0.01,
            trunc_length=20,
            total_horizon=40,
        )

    loss, grad, _ = step(theta, pes_state, jax.random.key(2))
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grad))


def test_meta_train_reduces_meta_loss_learnable_sgd() -> None:
    """Meta-training a learnable learning rate lowers the meta-loss over outer steps."""
    learned_optimizer = LearnableSGD(initial_learning_rate=0.01)  # deliberately too small
    family = QuadraticTaskFamily(dim=5, max_log_condition=1.0)
    _, losses = meta_train(
        learned_optimizer,
        family,
        jax.random.key(0),
        num_outer_steps=150,
        num_tasks=8,
        trunc_length=10,
        total_horizon=30,
        perturbation_std=0.02,
        meta_learning_rate=0.05,
    )
    assert jnp.all(jnp.isfinite(losses))
    assert jnp.mean(losses[-20:]) < jnp.mean(losses[:20])


def test_meta_train_runs_for_mlp_learned_optimizer() -> None:
    """The full PES pipeline runs end-to-end for the MLP learned optimizer (smoke)."""
    learned_optimizer = MLPLearnedOptimizer(hidden_size=8, hidden_layers=1)
    family = QuadraticTaskFamily(dim=4, max_log_condition=1.0)
    theta, losses = meta_train(
        learned_optimizer,
        family,
        jax.random.key(0),
        num_outer_steps=20,
        num_tasks=4,
        trunc_length=10,
        total_horizon=20,
        perturbation_std=0.01,
        meta_learning_rate=1e-3,
    )
    assert jnp.all(jnp.isfinite(losses))
    assert len(jax.tree.leaves(theta)) > 0
