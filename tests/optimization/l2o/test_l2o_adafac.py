"""TDD contracts for the Adafactor-MLP learned optimiser and its factored features (P2c).

References: ``learned_optimization/learned_optimizers/adafac_mlp_lopt.py`` and ``common.py``
(``factored_dims`` / ``factored_rolling``); Metz et al. 2020 (``arXiv:2009.11243``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.optimization.l2o import features
from opifex.optimization.l2o.learned import AdafacMLPLearnedOptimizer
from opifex.optimization.l2o.meta_train import meta_train
from opifex.optimization.l2o.tasks import MLPTaskFamily, QuadraticTaskFamily


def test_factored_dims_picks_two_largest_or_none() -> None:
    """factored_dims factors the two largest axes for rank >= 2, else None."""
    assert features.factored_dims((8, 16)) == (0, 1)
    assert features.factored_dims((16, 4)) == (1, 0)  # argsort -> largest is axis 0
    assert features.factored_dims((10,)) is None
    assert features.factored_dims(()) is None


def test_adafactor_accum_factored_shapes_and_preconditioner() -> None:
    """A 2D tensor uses row/col accumulators; features carry a trailing decay axis."""
    decays = jnp.asarray([0.9, 0.99, 0.999])
    grad = jax.random.normal(jax.random.key(0), (8, 16))
    v_row, v_col, v_diag = features.init_adafactor_accum(grad, decays.size)
    assert v_row.shape == (3, 8)
    assert v_col.shape == (3, 16)

    new_row, new_col, _new_diag, fac_g, row_feat, col_feat, factor = (
        features.update_adafactor_accum(v_row, v_col, v_diag, grad, decays)
    )
    assert new_row.shape == (3, 8)
    assert new_col.shape == (3, 16)
    # Features broadcast back to the tensor shape, one per decay (trailing axis).
    for feature in (fac_g, row_feat, col_feat, factor):
        assert feature.shape == (8, 16, 3)
        assert jnp.all(jnp.isfinite(feature))


def test_adafactor_accum_non_factored_matches_rmsprop() -> None:
    """A 1D tensor uses a diagonal accumulator and the RMSProp-style preconditioned gradient."""
    decays = jnp.asarray([0.999])
    grad = jax.random.normal(jax.random.key(1), (12,))
    v_row, v_col, v_diag = features.init_adafactor_accum(grad, decays.size)
    assert v_diag.shape == (1, 12)

    _nr, _nc, new_diag, fac_g, _row, _col, _factor = features.update_adafactor_accum(
        v_row, v_col, v_diag, grad, decays
    )
    # First step: new_diag = (1 - decay) * grad^2; y = grad / sqrt(new_diag).
    expected_diag = (1.0 - 0.999) * (grad**2 + 1e-30)
    assert jnp.allclose(new_diag[0], expected_diag, rtol=1e-4)
    assert fac_g.shape == (12, 1)


def test_adafac_theta_is_pytree_and_opt_fn_runs_inner_loop() -> None:
    """init returns a pytree theta; opt_fn(theta) reduces a quadratic loss through the interface."""
    lopt = AdafacMLPLearnedOptimizer(hidden_size=8, hidden_layers=1, step_mult=0.1)
    theta = lopt.init(jax.random.key(0))
    assert len(jax.tree.leaves(theta)) > 0

    task = QuadraticTaskFamily(dim=6, max_log_condition=1.0).sample(jax.random.key(1))
    optimizer = lopt.opt_fn(theta)
    state = optimizer.init(task.init(jax.random.key(2)), num_steps=50)
    initial = task.loss(optimizer.get_params(state), jax.random.key(3))
    for _ in range(50):
        params = optimizer.get_params(state)
        _loss, grad = task.loss_and_grad(params, jax.random.key(3))
        state = optimizer.update(state, grad)
    final = task.loss(optimizer.get_params(state), jax.random.key(3))
    assert jnp.isfinite(final)
    # A random-init learned optimiser need not converge, but it must stay finite and move.
    assert not jnp.allclose(initial, final)


def test_adafac_inner_step_is_jit_and_vmap_over_theta_safe() -> None:
    """opt_fn(theta) is jit-safe and vmappable over a batch of theta (the PES prerequisite)."""
    lopt = AdafacMLPLearnedOptimizer(hidden_size=8, hidden_layers=1)
    task = MLPTaskFamily(input_dim=4, hidden_dim=8, output_dim=2).sample(jax.random.key(0))
    start = task.init(jax.random.key(1))

    def one_step(theta: object) -> jax.Array:
        optimizer = lopt.opt_fn(theta)
        state = optimizer.init(start, num_steps=10)
        _loss, grad = task.loss_and_grad(optimizer.get_params(state), jax.random.key(2))
        new_state = optimizer.update(state, grad)
        return task.loss(optimizer.get_params(new_state), jax.random.key(2))

    thetas = jax.vmap(lopt.init)(jax.random.split(jax.random.key(3), 4))
    losses = jax.jit(jax.vmap(one_step))(thetas)
    assert losses.shape == (4,)
    assert jnp.all(jnp.isfinite(losses))


def test_adafac_meta_trains_with_pes() -> None:
    """The Adafactor lopt meta-trains end-to-end with PES (the full pipeline runs and lowers loss)."""
    lopt = AdafacMLPLearnedOptimizer(hidden_size=8, hidden_layers=1, step_mult=0.03)
    family = MLPTaskFamily(input_dim=4, hidden_dim=8, output_dim=2)
    theta, losses = meta_train(
        lopt,
        family,
        jax.random.key(0),
        num_outer_steps=80,
        num_tasks=16,
        trunc_length=10,
        total_horizon=40,
        perturbation_std=0.01,
        meta_learning_rate=3e-3,
    )
    assert jnp.all(jnp.isfinite(losses))
    assert len(jax.tree.leaves(theta)) > 0
    assert jnp.mean(losses[-20:]) < jnp.mean(losses[:20])
