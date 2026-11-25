"""Contracts for the L2O baselines and honest benchmark."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from substrax.testing import TraceCounter

from opifex.optimization.l2o.baselines import (
    bfgs_minimise,
    loss_curve,
    tuned_optax_baseline,
)
from opifex.optimization.l2o.benchmark import (
    benchmark_on_held_out_tasks,
    speedup_at_target,
    steps_to_target,
)
from opifex.optimization.l2o.core import Task
from opifex.optimization.l2o.learned import MLPLearnedOptimizer
from opifex.optimization.l2o.optimizers import OptaxOptimizer
from opifex.optimization.l2o.tasks import QuadraticTaskFamily
from opifex.optimization.second_order.wrappers import BFGSSolver


def test_bfgs_minimise_solves_quadratic_to_optimum() -> None:
    """Pure-JAX BFGS solve reaches the quadratic optimum (loss < 1e-6) and traces once."""
    task = QuadraticTaskFamily(dim=6).sample(jax.random.key(0))
    start = task.init(jax.random.key(1))
    eval_key = jax.random.key(2)
    solver = BFGSSolver(max_iterations=200, rtol=1e-8, atol=1e-8)
    minimiser, final_loss = bfgs_minimise(task, start, solver=solver, key=eval_key)
    assert minimiser.shape == (6,)
    assert float(final_loss) < 1e-6

    counter = TraceCounter()
    jitted = jax.jit(counter.wrap(lambda y0, k: bfgs_minimise(task, y0, max_steps=200, key=k)))
    with counter.expect(new_traces=1):
        _, loss_a = jitted(start, eval_key)
    with counter.expect(new_traces=0):
        _, loss_b = jitted(start + 0.1, jax.random.key(3))
    assert float(loss_a) < 1e-6
    assert float(loss_b) < 1e-6


def test_loss_curve_decreases_for_tuned_optimizer() -> None:
    """The loss curve of a well-tuned optimiser is monotone-ish down and length num_steps+1."""
    task = QuadraticTaskFamily(dim=8, max_log_condition=1.0).sample(jax.random.key(0))
    optimizer = OptaxOptimizer(optax.adam(0.1))
    curve = loss_curve(
        optimizer, task, task.init(jax.random.key(1)), num_steps=50, key=jax.random.key(2)
    )
    assert curve.shape == (51,)
    assert curve[-1] < curve[0]


def test_tuned_optax_baseline_picks_a_working_rate() -> None:
    """The lr sweep returns a curve that converges (better than the worst rate)."""
    task = QuadraticTaskFamily(dim=6, max_log_condition=1.0).sample(jax.random.key(0))
    curve, lr = tuned_optax_baseline(
        task,
        task.init(jax.random.key(1)),
        jnp.asarray([1e-3, 1e-2, 1e-1]),
        num_steps=60,
        key=jax.random.key(2),
    )
    assert curve[-1] < curve[0]
    assert float(lr) > 0.0


def test_steps_to_target_and_speedup_censoring() -> None:
    """steps_to_target finds the crossing; speedup censors non-convergence."""
    fast = jnp.array([1.0, 0.5, 0.2, 0.05])  # reaches 0.1 at step 3
    slow = jnp.array([1.0, 0.9, 0.8, 0.7])  # never reaches 0.1
    target = jnp.asarray(0.1)
    assert float(steps_to_target(fast, target)) == 3.0
    assert jnp.isinf(steps_to_target(slow, target))
    # candidate fast vs baseline slow: baseline never reaches -> inf speedup (candidate better).
    assert jnp.isinf(speedup_at_target(slow, fast, target))
    # candidate slow vs baseline fast: candidate never reaches -> 0 speedup.
    assert float(speedup_at_target(fast, slow, target)) == 0.0


def test_benchmark_runs_and_returns_finite_curves() -> None:
    """benchmark_on_held_out_tasks runs end-to-end with finite learning curves."""
    lopt = MLPLearnedOptimizer(hidden_size=8, hidden_layers=1)
    theta = lopt.init(jax.random.key(0))
    family = QuadraticTaskFamily(dim=5, max_log_condition=1.0)
    result = benchmark_on_held_out_tasks(
        lopt, theta, family, jax.random.key(1), num_tasks=6, num_steps=40
    )
    assert result["learned_curve_mean"].shape == (41,)
    assert result["baseline_curve_mean"].shape == (41,)
    assert jnp.all(jnp.isfinite(result["baseline_curve_mean"]))
    assert 0.0 <= float(result["fraction_reached_target"]) <= 1.0


def test_bfgs_minimise_requires_keyword_key_and_forwards_it() -> None:
    """bfgs_minimise has no default key and forwards caller key to task.loss."""

    class _KeyAnchoredQuadratic(Task):
        def init(self, key: jax.Array) -> jax.Array:
            return jax.random.normal(key, (3,))

        def loss(self, params: jax.Array, key: jax.Array) -> jax.Array:
            target = jax.random.uniform(key, (3,), minval=-1.0, maxval=1.0)
            return 0.5 * jnp.sum((params - target) ** 2)

    task = _KeyAnchoredQuadratic()
    start = jnp.zeros((3,))
    with pytest.raises(TypeError, match="key"):
        bfgs_minimise(task, start)  # type: ignore[call-arg]

    sol_a, loss_a = bfgs_minimise(task, start, key=jax.random.key(11))
    sol_b, loss_b = bfgs_minimise(task, start, key=jax.random.key(29))
    assert float(loss_a) < 1e-10
    assert float(loss_b) < 1e-10
    assert not bool(jnp.allclose(sol_a, sol_b))


def test_bfgs_minimise_rejects_conflicting_kwargs_and_guards_unconverged_to_nan() -> None:
    """Simultaneous solver + tolerance/step kwargs fail fast; unconverged solves return NaN."""

    class _RosenbrockTask(Task):
        def init(self, key: jax.Array) -> jax.Array:
            del key
            return jnp.array([-1.2, 1.0], dtype=jnp.float32)

        def loss(self, params: jax.Array, key: jax.Array) -> jax.Array:
            del key
            return (1.0 - params[0]) ** 2 + 100.0 * (params[1] - params[0] ** 2) ** 2

    task = _RosenbrockTask()
    start = task.init(jax.random.key(0))
    custom_solver = BFGSSolver(max_iterations=50, rtol=1e-6, atol=1e-6)

    for conflict in ({"max_steps": 10}, {"rtol": 1e-4}, {"atol": 1e-4}):
        with pytest.raises(ValueError, match="not both"):
            bfgs_minimise(task, start, solver=custom_solver, key=jax.random.key(1), **conflict)  # type: ignore[arg-type]

    bad_params, bad_loss = bfgs_minimise(
        task,
        start,
        max_steps=1,
        rtol=1e-12,
        atol=1e-12,
        key=jax.random.key(2),
    )
    assert bool(jnp.all(jnp.isnan(bad_params)))
    assert bool(jnp.isnan(bad_loss))
