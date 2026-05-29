r"""MUMBO multi-fidelity Bayesian-optimisation acquisition — slice 35.

Tests the Moss, Leslie, Rayson 2020 MUMBO acquisition built on top of
the linear multi-fidelity GP from slice 33. MUMBO extends MES
(Max-value Entropy Search) to multi-fidelity by adding fidelity as a
candidate dimension and weighting information gain by query cost.

The opifex implementation accepts a fitted multi-fidelity GP state and
a batch of ``(x, fidelity_level)`` candidates, and returns one
acquisition score per candidate. Cost weighting is applied by the
caller (the canonical recipe divides the acquisition by per-level
cost; see ``MUMBO/cost``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def test_mumbo_acquisition_returns_one_score_per_candidate() -> None:
    """``mumbo_acquisition`` returns ``(m,)`` for ``m`` candidate (x, level) pairs."""
    from opifex.uncertainty.multi_fidelity import (
        fit_linear_multi_fidelity_gp,
        mumbo_acquisition,
    )

    x_low = jnp.linspace(0.0, 1.0, 20).reshape(-1, 1)
    x_high = jnp.linspace(0.1, 0.9, 5).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten())
    state = fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.3),
        scaling_factors=(1.0,),
        noise_std=0.05,
    )
    x_candidates = jnp.linspace(0.05, 0.95, 8).reshape(-1, 1)
    candidate_levels = jnp.array([0, 0, 1, 1, 0, 0, 1, 1])
    scores = mumbo_acquisition(
        state=state,
        x_candidates=x_candidates,
        candidate_levels=candidate_levels,
        target_level=1,
        rng_key=jax.random.PRNGKey(0),
        grid_size=200,
        num_gumbel_samples=8,
    )
    assert scores.shape == (8,)
    assert jnp.all(jnp.isfinite(scores))


def test_mumbo_acquisition_is_non_negative() -> None:
    """MUMBO is an information-gain quantity — non-negative up to numerical noise."""
    from opifex.uncertainty.multi_fidelity import (
        fit_linear_multi_fidelity_gp,
        mumbo_acquisition,
    )

    x_low = jnp.linspace(0.0, 1.0, 15).reshape(-1, 1)
    x_high = jnp.linspace(0.2, 0.8, 4).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten())
    state = fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.3),
        scaling_factors=(1.0,),
        noise_std=0.05,
    )
    x_candidates = jnp.linspace(0.05, 0.95, 6).reshape(-1, 1)
    candidate_levels = jnp.zeros((6,), dtype=jnp.int32)
    scores = mumbo_acquisition(
        state=state,
        x_candidates=x_candidates,
        candidate_levels=candidate_levels,
        target_level=1,
        rng_key=jax.random.PRNGKey(1),
        grid_size=200,
        num_gumbel_samples=8,
    )
    assert jnp.all(scores > -1e-4)


def test_mumbo_acquisition_low_fidelity_candidate_carries_information() -> None:
    """A low-fidelity candidate carries non-trivial information about the high-fidelity max.

    With a strong AR(1) coupling (rho close to 1), observing the low
    fidelity at a novel ``x`` reduces uncertainty about the
    high-fidelity maximum: the MUMBO score is strictly positive (above
    a small numerical floor).
    """
    from opifex.uncertainty.multi_fidelity import (
        fit_linear_multi_fidelity_gp,
        mumbo_acquisition,
    )

    x_low = jnp.linspace(0.0, 1.0, 16).reshape(-1, 1)
    x_high = jnp.linspace(0.1, 0.9, 4).reshape(-1, 1)
    y_low = jnp.sin(2.0 * jnp.pi * x_low.flatten())
    y_high = jnp.sin(2.0 * jnp.pi * x_high.flatten())
    state = fit_linear_multi_fidelity_gp(
        x_train_per_level=(x_low, x_high),
        y_train_per_level=(y_low, y_high),
        lengthscales=(0.3, 0.3),
        output_scales=(1.0, 0.3),
        scaling_factors=(0.95,),
        noise_std=0.05,
    )
    x_test = jnp.array([[0.45]])
    score = mumbo_acquisition(
        state=state,
        x_candidates=x_test,
        candidate_levels=jnp.array([0]),
        target_level=1,
        rng_key=jax.random.PRNGKey(2),
        grid_size=200,
        num_gumbel_samples=12,
    )
    assert jnp.all(jnp.isfinite(score))
    assert float(score[0]) > 1e-6
