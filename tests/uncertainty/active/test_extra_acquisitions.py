r"""Tests for extra acquisition kernels — Slice 23 (audit finding #4b).

Phase 8 Task 8.3 trieste source map (``08-...:557-584``) lists three
acquisitions that ship in the **base** ``acquisition.py``:

* MES — Min-Value Entropy Search (Wang+ 2017; trieste
  ``function/entropy.py:50``).
* GIBBON — General-purpose Information-Based Bayesian OptimisatioN
  (Moss+ 2021; trieste ``function/entropy.py:236``).
* IntegratedVarianceReduction — active-learning acquisition for GP
  models (trieste ``function/active_learning.py:250``).

Plus three additions to ``batch_active.py``:

* qHSRI — Batch Hypervolume Sharpe Ratio Indicator (Binois+ 2020).
* Fantasizer — sequential greedy batch via fantasised observations
  (Snoek+ 2012).
* LocalPenalization — Gonzalez+ 2016 batch acquisition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


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
