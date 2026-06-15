r"""D8 + D9 closure tests — Student-t / Beta wrappers on PEP and PL.

Closes the deferred items D8 (Student-t / Beta Power-EP wrappers) and
D9 (Student-t / Beta Posterior-Linearisation wrappers) by mirroring
the existing Markov-VI Student-t / Beta tests against the PEP and PL
inference paths.

References
----------
* Slice 26 / 27 — :func:`opifex.uncertainty.markov.fit_studentst_markov_vi_gp`,
  :func:`fit_beta_markov_vi_gp` (the templates being mirrored here).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import matern32_kernel as state_space_matern32_kernel


# -----------------------------------------------------------------------------
# Student-t robustness — Markov-PEP and Markov-PL
# -----------------------------------------------------------------------------


def test_fit_studentst_markov_pep_gp_is_robust_to_outliers() -> None:
    """Student-t Markov-PEP dampens heavy-tailed outliers."""
    from opifex.uncertainty.markov import fit_studentst_markov_pep_gp

    times = jnp.linspace(0.0, 6.0, 25)
    clean = jnp.sin(2.0 * times)
    observations = clean.at[5].set(3.0).at[15].set(-3.0)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_studentst_markov_pep_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        df=4.0,
        scale=0.3,
        power=0.5,
        num_iterations=30,
        learning_rate=0.3,
    )
    assert jnp.max(jnp.abs(state.smoothed_means)) < 2.5


def test_fit_studentst_markov_pl_gp_is_robust_to_outliers() -> None:
    """Student-t Markov-PL dampens heavy-tailed outliers."""
    from opifex.uncertainty.markov import fit_studentst_markov_pl_gp

    times = jnp.linspace(0.0, 6.0, 25)
    clean = jnp.sin(2.0 * times)
    observations = clean.at[5].set(3.0).at[15].set(-3.0)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_studentst_markov_pl_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        df=4.0,
        scale=0.3,
        num_iterations=20,
    )
    assert jnp.max(jnp.abs(state.smoothed_means)) < 2.5


# -----------------------------------------------------------------------------
# Beta unit-interval — Markov-PEP and Markov-PL
# -----------------------------------------------------------------------------


def test_fit_beta_markov_pep_gp_recovers_unit_interval_predictions() -> None:
    """Beta Markov-PEP predict yields means in [0, 1]."""
    from opifex.uncertainty.markov import (
        fit_beta_markov_pep_gp,
        predict_beta_markov_pep_gp,
    )

    times = jnp.linspace(0.0, 4.0, 18)
    mean = jax.nn.sigmoid(jnp.sin(2.0 * times))
    scale = 20.0
    alpha = mean * scale
    beta = scale * (1.0 - mean)
    observations = jax.random.beta(jax.random.PRNGKey(101), alpha, beta)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_beta_markov_pep_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        scale=scale,
        power=0.5,
        num_iterations=25,
        learning_rate=0.3,
    )
    predictive = predict_beta_markov_pep_gp(
        state=state, times_test=jnp.linspace(0.5, 3.5, 6), scale=scale
    )
    assert jnp.all(predictive.mean >= 0.0)
    assert jnp.all(predictive.mean <= 1.0)


def test_fit_beta_markov_pl_gp_recovers_unit_interval_predictions() -> None:
    """Beta Markov-PL predict yields means in [0, 1]."""
    from opifex.uncertainty.markov import (
        fit_beta_markov_pl_gp,
        predict_beta_markov_pl_gp,
    )

    times = jnp.linspace(0.0, 4.0, 18)
    mean = jax.nn.sigmoid(jnp.sin(2.0 * times))
    scale = 20.0
    alpha = mean * scale
    beta = scale * (1.0 - mean)
    observations = jax.random.beta(jax.random.PRNGKey(102), alpha, beta)
    kernel = state_space_matern32_kernel(variance=1.0, lengthscale=0.5)
    state = fit_beta_markov_pl_gp(
        times=times,
        observations=observations,
        state_space_kernel=kernel,
        scale=scale,
        num_iterations=20,
    )
    predictive = predict_beta_markov_pl_gp(
        state=state, times_test=jnp.linspace(0.5, 3.5, 6), scale=scale
    )
    assert jnp.all(predictive.mean >= 0.0)
    assert jnp.all(predictive.mean <= 1.0)
