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
import pytest
from scipy import integrate
from scipy.stats import norm


_EPS32 = float(np.finfo(np.float32).eps)


# MES scores are means of O(1) float32 terms; four float32 ULPs at magnitude 1 bound their
# rounding. Measured against the quadrature reference: 6.9e-8 on scores up to 0.875.
_MES_TOLERANCE = 4.0 * _EPS32


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


_GIBBON_MEANS = np.asarray([-2.5, -1.0, 0.0, 1.0, 3.0])
_GIBBON_VARIANCES = np.asarray([0.09, 1.0, 4.0])
_GIBBON_MINIMA = np.asarray([-3.0, -2.2, -1.9])


def _gibbon_definition_4(mean: float, variance: float, noise_variance: float) -> float:
    """Float64 GIBBON at batch size one (Moss et al. 2021, Definition 4) for samples of a minimum."""
    gamma = (mean - _GIBBON_MINIMA) / math.sqrt(variance)
    ratio = np.exp(norm.logpdf(gamma) - norm.logcdf(gamma))
    rho_squared = variance / (variance + noise_variance)
    return float(-0.5 * np.mean(np.log1p(-rho_squared * ratio * (gamma + ratio))))


def _gibbon_float32_error_bound(mean: float, variance: float, noise_variance: float) -> float:
    """First-order float32 rounding error of a GIBBON score, from the conditioning of Definition 4.

    Rounding moves ``gamma`` by ``eps |gamma|`` and ``log phi - log Phi`` by
    ``eps (|log phi| + |log Phi| + 1)``. Both move ``p = r (gamma + r)``, which moves each sample's
    term by ``rho^2 dp / (2 (1 - rho^2 p))``; four ULPs cover the sums. Without noise the argument of
    the logarithm cancels as ``gamma`` falls below zero, and the bound grows with it. Measured float32
    errors stay below 0.66 of the bound over ``gamma`` in [-12, 40], variances from 1e-4 to 100 and
    noise variances from 0 to 2, and below 0.13 of it on the grid used here.
    """
    gamma = (mean - _GIBBON_MINIMA) / math.sqrt(variance)
    log_pdf, log_cdf = norm.logpdf(gamma), norm.logcdf(gamma)
    ratio = np.exp(log_pdf - log_cdf)
    rho_squared = variance / (variance + noise_variance)
    slope = np.abs(gamma + 2.0 * ratio)
    product_error = (
        _EPS32
        * ratio
        * (slope * (np.abs(log_pdf) + np.abs(log_cdf) + 1.0) + np.abs(gamma) * (1.0 + slope**2))
    )
    log_argument = 1.0 - rho_squared * ratio * (gamma + ratio)
    return float(0.5 * np.mean(rho_squared * product_error / log_argument) + 4.0 * _EPS32)


def _gibbon_grid_bounds(noise_variance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Means, variances and float32 error bounds over the GIBBON test grid."""
    means, variances = (axis.ravel() for axis in np.meshgrid(_GIBBON_MEANS, _GIBBON_VARIANCES))
    bounds = np.asarray(
        [
            _gibbon_float32_error_bound(m, v, noise_variance)
            for m, v in zip(means, variances, strict=True)
        ]
    )
    return means, variances, bounds


@pytest.mark.parametrize("noise_variance", [0.0, 0.1, 2.0])
def test_gibbon_matches_definition_4_at_batch_size_one(noise_variance: float) -> None:
    """GIBBON's single-point score is Definition 4 with ``rho^2 = var / (var + noise)``.

    The log-determinant term of Definition 4 vanishes for one point; the remaining term is applied
    to ``-f`` for samples of the minimum, ``gamma = (mu - y*) / sigma``.
    """
    from opifex.uncertainty.active.acquisition import gibbon

    means, variances, bounds = _gibbon_grid_bounds(noise_variance)
    scores = gibbon(
        means=jnp.asarray(means),
        variances=jnp.asarray(variances),
        sampled_min_values=jnp.asarray(_GIBBON_MINIMA),
        noise_variance=noise_variance,
    )
    reference = np.asarray(
        [_gibbon_definition_4(m, v, noise_variance) for m, v in zip(means, variances, strict=True)]
    )
    error = np.abs(np.asarray(scores, dtype=np.float64) - reference)
    assert np.all(error <= bounds), float(np.max(error / bounds))


def test_gibbon_is_a_lower_bound_on_mes_without_noise() -> None:
    """Without noise GIBBON bounds MES from below.

    GIBBON replaces the entropy of the truncated predictive by that of a Gaussian with the same
    variance, which is at least as large, so its information gain cannot exceed MES's.
    """
    from opifex.uncertainty.active.acquisition import gibbon, min_value_entropy_search

    means, variances, bounds = _gibbon_grid_bounds(0.0)
    candidate_means = jnp.asarray(means)
    candidate_variances = jnp.asarray(variances)
    minima = jnp.asarray(_GIBBON_MINIMA)
    gibbon_scores = np.asarray(
        gibbon(means=candidate_means, variances=candidate_variances, sampled_min_values=minima),
        dtype=np.float64,
    )
    mes_scores = np.asarray(
        min_value_entropy_search(
            means=candidate_means, variances=candidate_variances, sampled_min_values=minima
        ),
        dtype=np.float64,
    )
    assert np.all(gibbon_scores <= mes_scores + bounds + _MES_TOLERANCE)
    assert np.all(gibbon_scores >= -bounds)


def test_gibbon_observation_noise_lowers_the_score() -> None:
    """Noisier observations say less about the minimum; very large noise says almost nothing."""
    from opifex.uncertainty.active.acquisition import gibbon

    def score(noise_variance: float) -> float:
        return float(
            gibbon(
                means=jnp.asarray([-1.0]),
                variances=jnp.asarray([1.0]),
                sampled_min_values=jnp.asarray(_GIBBON_MINIMA),
                noise_variance=noise_variance,
            )[0]
        )

    assert score(0.0) > score(0.5) > score(5.0) > 0.0
    assert score(1e6) < 1e-5


def test_gibbon_supports_jit_vmap_and_grad() -> None:
    """GIBBON compiles, vectorises over candidates, and differentiates in the posterior mean."""
    from opifex.uncertainty.active.acquisition import gibbon

    def single(mean: jax.Array) -> jax.Array:
        return gibbon(
            means=mean[None],
            variances=jnp.asarray([0.5]),
            sampled_min_values=jnp.asarray(_GIBBON_MINIMA),
            noise_variance=0.1,
        )[0]

    means = jnp.linspace(-3.0, 3.0, 7)
    batched = jax.jit(jax.vmap(single))(means)
    assert bool(jnp.allclose(batched, jax.vmap(single)(means)))
    assert bool(jnp.all(jnp.isfinite(jax.vmap(jax.grad(single))(means))))


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
