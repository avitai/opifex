"""Contracts for the L2O baselines and honest benchmark (P4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import optimistix as optx

from opifex.optimization.l2o.baselines import (
    loss_curve,
    optimistix_minimise,
    tuned_optax_baseline,
)
from opifex.optimization.l2o.benchmark import (
    benchmark_on_held_out_tasks,
    speedup_at_target,
    steps_to_target,
)
from opifex.optimization.l2o.learned import MLPLearnedOptimizer
from opifex.optimization.l2o.optimizers import OptaxOptimizer
from opifex.optimization.l2o.tasks import QuadraticTaskFamily


def test_optimistix_bfgs_solves_quadratic_to_optimum() -> None:
    """A real optimistix BFGS solve reaches the quadratic optimum (loss ~ 0)."""
    task = QuadraticTaskFamily(dim=6).sample(jax.random.key(0))
    minimiser, final_loss = optimistix_minimise(
        task, optx.BFGS(rtol=1e-8, atol=1e-8), task.init(jax.random.key(1)), max_steps=200
    )
    assert minimiser.shape == (6,)
    assert float(final_loss) < 1e-6


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
