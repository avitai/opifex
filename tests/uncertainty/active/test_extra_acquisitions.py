r"""Tests for extra acquisition kernels — Slice 23 (audit finding #4b).

Phase 8 Task 8.3 source map (``08-...:557-584``) lists three
acquisitions that ship in the **base** ``acquisition.py``:

* MES — Min-Value Entropy Search (Wang & Jegelka 2017).
* GIBBON — General-purpose Information-Based Bayesian OptimisatioN
  (Moss+ 2021).
* IntegratedVarianceReduction — active-learning acquisition for GP
  models (Cohn, Ghahramani & Jordan 1996).

Plus three additions to ``batch_active.py``:

* qHSRI — Batch Hypervolume Sharpe Ratio Indicator (Binois+ 2020).
* Fantasizer — sequential greedy batch via fantasised observations
  (Snoek+ 2012).
* LocalPenalization — Gonzalez+ 2016 batch acquisition.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate
from scipy.stats import norm


# MES scores are means of O(1) float32 terms; four float32 ULPs at magnitude 1 bound their
# rounding. Measured against the quadrature reference: 6.9e-8 on scores up to 0.875.
_MES_TOLERANCE = 4.0 * float(np.finfo(np.float32).eps)


# -----------------------------------------------------------------------------
# Min-Value Entropy Search (MES) — single-point
# -----------------------------------------------------------------------------


def test_mes_returns_finite_score_per_candidate() -> None:
    """MES returns one entropy-reduction score per candidate."""
    from opifex.uncertainty.active.acquisition import min_value_entropy_search

    means = jnp.array([-1.0, 0.0, 1.0])
    variances = jnp.array([0.1, 0.2, 0.15])
    sampled_min_values = jnp.array([-2.0, -1.5, -1.8])
    scores = min_value_entropy_search(
        means=means, variances=variances, sampled_min_values=sampled_min_values
    )
    assert scores.shape == (3,)
    assert jnp.all(jnp.isfinite(scores))
    # MES scores are non-negative (information gain).
    assert jnp.all(scores >= -1e-6)


def _minimum_entropy_reduction(mean: float, std: float, minima: np.ndarray) -> float:
    """Mean entropy reduction of ``N(mean, std^2)`` from learning ``f >= y*``, by quadrature."""
    reductions = []
    for minimum in minima:
        log_mass = float(norm.logsf(minimum, loc=mean, scale=std))

        def integrand(value: float, log_mass: float = log_mass) -> float:
            log_density = float(norm.logpdf(value, loc=mean, scale=std)) - log_mass
            return math.exp(log_density) * log_density

        upper = max(minimum, mean) + 40.0 * std
        breakpoints = [mean] if minimum < mean < upper else None
        truncated_entropy = -integrate.quad(
            integrand, minimum, upper, limit=400, points=breakpoints
        )[0]
        reductions.append(0.5 * math.log(2.0 * math.pi * math.e * std * std) - truncated_entropy)
    return float(np.mean(reductions))


def test_mes_matches_the_entropy_reduction_about_the_minimum() -> None:
    """MES is the expected entropy reduction of ``f(x)`` from learning that ``f >= y*``.

    Wang & Jegelka (2017), eq. 6, applied to ``-f`` for a minimum: ``gamma = (mu - y*) / sigma``.
    The reference integrates the truncated Gaussian entropy with scipy in float64.
    """
    from opifex.uncertainty.active.acquisition import min_value_entropy_search

    grid_means, grid_stds = (
        axis.ravel() for axis in np.meshgrid([-2.5, -1.0, 0.0, 1.0, 3.0, 8.0], [0.3, 1.0, 2.0])
    )
    minima = np.asarray([-3.0, -2.2, -1.9])
    scores = min_value_entropy_search(
        means=jnp.asarray(grid_means),
        variances=jnp.asarray(grid_stds**2),
        sampled_min_values=jnp.asarray(minima),
    )
    reference = np.asarray(
        [
            _minimum_entropy_reduction(m, s, minima)
            for m, s in zip(grid_means, grid_stds, strict=True)
        ]
    )
    error = np.abs(np.asarray(scores, dtype=np.float64) - reference)
    assert float(np.max(error)) <= _MES_TOLERANCE, float(np.max(error))


def test_mes_ranks_the_candidate_nearest_the_minimum_first() -> None:
    """A candidate whose mean is far above every sampled minimum carries almost no information."""
    from opifex.uncertainty.active.acquisition import min_value_entropy_search

    scores = min_value_entropy_search(
        means=jnp.asarray([0.0, 2.0, 5.0]),
        variances=jnp.ones(3),
        sampled_min_values=jnp.asarray([-3.0, -2.5]),
    )
    assert float(scores[0]) > float(scores[1]) > float(scores[2])


def test_mes_is_finite_and_differentiable_far_from_the_minimum() -> None:
    """Values and mean-gradients stay finite at ``|gamma| = 40``, and jit agrees with eager."""
    from opifex.uncertainty.active.acquisition import min_value_entropy_search

    def total(means: jax.Array) -> jax.Array:
        return jnp.sum(
            min_value_entropy_search(
                means=means, variances=jnp.ones(2), sampled_min_values=jnp.asarray([0.0])
            )
        )

    means = jnp.asarray([-40.0, 40.0])
    assert bool(jnp.isfinite(total(means)))
    assert bool(jnp.all(jnp.isfinite(jax.grad(total)(means))))
    assert bool(jnp.allclose(jax.jit(total)(means), total(means)))


# -----------------------------------------------------------------------------
# GIBBON — batch-information acquisition
# -----------------------------------------------------------------------------


def test_gibbon_reduces_to_mes_at_batch_size_one() -> None:
    """At batch size 1, GIBBON reduces to MES (Moss+ 2021 §3)."""
    from opifex.uncertainty.active.acquisition import (
        gibbon,
        min_value_entropy_search,
    )

    means = jnp.array([0.0, 0.5, 1.0])
    variances = jnp.array([0.1, 0.1, 0.1])
    sampled_min_values = jnp.array([-1.0, -0.5])
    mes_scores = min_value_entropy_search(
        means=means, variances=variances, sampled_min_values=sampled_min_values
    )
    gibbon_scores = gibbon(means=means, variances=variances, sampled_min_values=sampled_min_values)
    assert jnp.allclose(mes_scores, gibbon_scores, atol=1e-5)


# -----------------------------------------------------------------------------
# IntegratedVarianceReduction
# -----------------------------------------------------------------------------


def test_integrated_variance_reduction_returns_positive_score() -> None:
    """IVR ranks candidates by how much they reduce integrated posterior variance."""
    from opifex.uncertainty.active.acquisition import integrated_variance_reduction_score

    candidate_variances = jnp.array([0.5, 0.2, 0.8])
    cross_variances = jnp.array([[0.3, 0.2, 0.1], [0.2, 0.1, 0.0], [0.4, 0.3, 0.2]])
    scores = integrated_variance_reduction_score(
        candidate_variances=candidate_variances, cross_variances=cross_variances
    )
    assert scores.shape == (3,)
    assert jnp.all(scores >= 0.0)


# -----------------------------------------------------------------------------
# qHSRI — batch hypervolume Sharpe-ratio indicator
# -----------------------------------------------------------------------------


def test_qhsri_returns_a_subset_index_array_of_length_batch_size() -> None:
    """qHSRI selects ``batch_size`` indices via Sharpe-style ranking."""
    from opifex.uncertainty.active.batch_active import (
        batch_hypervolume_sharpe_ratio_indicator,
    )

    means = jnp.array([[1.0, 0.5], [0.5, 1.0], [0.8, 0.8], [0.2, 0.2]])
    stds = jnp.array([[0.1, 0.1], [0.1, 0.1], [0.2, 0.2], [0.1, 0.1]])
    selected = batch_hypervolume_sharpe_ratio_indicator(
        means=means, stds=stds, batch_size=2, reference_point=jnp.array([0.0, 0.0])
    )
    assert selected.shape == (2,)
    assert selected.dtype in (jnp.int32, jnp.int64)


# -----------------------------------------------------------------------------
# Fantasizer — sequential greedy batch
# -----------------------------------------------------------------------------


def test_fantasizer_returns_batch_size_indices_without_duplicates() -> None:
    """Fantasizer's greedy loop selects ``batch_size`` distinct candidates."""
    from opifex.uncertainty.active.batch_active import fantasizer

    scores = jnp.array([0.9, 0.7, 0.8, 0.6, 0.5])
    selected = fantasizer(initial_scores=scores, batch_size=3, key=jax.random.PRNGKey(0))
    assert selected.shape == (3,)
    # Each index appears at most once.
    unique = jnp.unique(selected)
    assert unique.shape[0] == 3


# -----------------------------------------------------------------------------
# Local Penalization
# -----------------------------------------------------------------------------


def test_local_penalization_dampens_candidates_close_to_a_pending_observation() -> None:
    """Local penalisation reduces acquisition near pending workers (Gonzalez+ 2016)."""
    from opifex.uncertainty.active.batch_active import local_penalization

    candidates = jnp.array([[0.0], [0.4], [1.0]])
    pending_points = jnp.array([[0.4]])
    base_scores = jnp.array([1.0, 1.0, 1.0])
    penalised = local_penalization(
        candidates=candidates,
        pending_points=pending_points,
        base_scores=base_scores,
        lipschitz_constant=1.0,
        max_value=2.0,
    )
    assert float(penalised[1]) < float(penalised[0])
    assert float(penalised[1]) < float(penalised[2])
