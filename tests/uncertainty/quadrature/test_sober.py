r"""SOBER kernel-recombination point-set Bayesian quadrature.

Tests for :mod:`opifex.uncertainty.quadrature.sober` — a JAX-native
port of Adachi et al.'s kernel-recombination algorithm
(arXiv:2206.04734 + arXiv:2301.11832), with the inner
Tchernychova-Lyons CAR algorithm following the SOBER PyTorch
reference at ``../SOBER/SOBER/_rchq.py``.

Algorithm invariants verified here:

* **Caratheodory reduction.** Given ``N`` candidate points carrying
  positive weights ``μ`` (summing to one) over ``n`` features, the
  recombination returns at most ``n + 1`` non-zero output weights
  while preserving the weighted-feature moments:
  ``Σ_i μ_i φ(x_i) == Σ_j w_j φ(x_{σ(j)})``. This is the published
  guarantee of Caratheodory's theorem applied to kernel features.

* **Probability simplex.** Output weights are non-negative and sum
  to one — they form a valid empirical measure over a sparser
  point set.

* **Kernel mean discrepancy reduction.** The kernel mean embedding
  of the SOBER-thinned subset matches the candidate distribution's
  empirical kernel mean: ``Σ_j w_j k(·, x_j) == Σ_i μ_i k(·, x_i)``
  in the Nyström feature space.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from opifex.uncertainty.quadrature import SOBERAdapterSpec
from opifex.uncertainty.quadrature.sober import (
    caratheodory_recombination,
    sober_kernel_recombination,
)
from opifex.uncertainty.registry import UQCapability


def _rbf_kernel(x_left: jax.Array, x_right: jax.Array, lengthscale: float = 1.0) -> jax.Array:
    """Standard RBF kernel matrix with amplitude 1."""
    squared_diff = jnp.sum(
        (x_left[:, None, :] - x_right[None, :, :]) ** 2 / (lengthscale**2), axis=-1
    )
    return jnp.exp(-0.5 * squared_diff)


# ---------------------------------------------------------------------------
# caratheodory_recombination — inner Tchernychova-Lyons CAR primitive
# ---------------------------------------------------------------------------


def test_caratheodory_recombination_preserves_feature_moments() -> None:
    """Recombination preserves ``Σ_i μ_i φ(x_i)`` in the feature space.

    Construction: 6 feature vectors in 2-D plus uniform initial weights;
    the recombination must yield at most 3 non-zero weights whose
    convex combination of features matches the initial mean.
    """
    features = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.5, 0.5],
            [-0.5, -0.5],
        ]
    )
    init_weights = jnp.ones(6) / 6.0
    initial_moment = init_weights @ features

    indices, weights = caratheodory_recombination(features=features, weights=init_weights)
    selected_features = features[indices]
    final_moment = weights @ selected_features

    assert jnp.allclose(final_moment, initial_moment, atol=1e-5)


def test_caratheodory_recombination_reduces_to_at_most_n_plus_one_points() -> None:
    """At most ``n + 1`` weights remain non-zero where ``n`` is the feature dim."""
    features = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.5, 0.5],
            [-0.5, -0.5],
        ]
    )
    init_weights = jnp.ones(6) / 6.0
    feature_dim = features.shape[1]

    indices, weights = caratheodory_recombination(features=features, weights=init_weights)
    assert weights.shape[0] <= feature_dim + 1
    assert indices.shape[0] == weights.shape[0]


def test_caratheodory_recombination_outputs_are_probability_simplex() -> None:
    """Returned weights are non-negative and sum to one."""
    features = jnp.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [-1.0, 0.0, 0.5],
            [0.0, -1.0, -0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    init_weights = jnp.ones(5) / 5.0

    _, weights = caratheodory_recombination(features=features, weights=init_weights)
    assert jnp.all(weights >= -1e-7)
    assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-6)


def test_caratheodory_recombination_passes_through_when_already_minimal() -> None:
    """If ``N <= n + 1`` the input is already minimal — pass through unchanged."""
    features = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    init_weights = jnp.array([0.5, 0.3, 0.2])
    initial_moment = init_weights @ features

    indices, weights = caratheodory_recombination(features=features, weights=init_weights)
    selected_features = features[indices]
    final_moment = weights @ selected_features

    assert jnp.allclose(final_moment, initial_moment, atol=1e-7)


# ---------------------------------------------------------------------------
# sober_kernel_recombination — full SOBER entry point
# ---------------------------------------------------------------------------


def test_sober_kernel_recombination_returns_probability_simplex_weights() -> None:
    """SOBER output weights form a valid empirical measure."""
    key = jax.random.PRNGKey(0)
    candidates = jax.random.normal(key, (32, 1))
    nystrom_points = candidates[:8]

    indices, weights = sober_kernel_recombination(
        candidate_points=candidates,
        nystrom_points=nystrom_points,
        num_selected=4,
        kernel_fn=_rbf_kernel,
    )
    assert weights.shape[0] == indices.shape[0]
    assert jnp.all(weights >= -1e-6)
    assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-5)


def test_sober_kernel_recombination_preserves_kernel_mean_at_full_rank() -> None:
    r"""SOBER recombination exactly preserves the kernel mean at full Nyström rank.

    When ``num_selected - 1 == num_nystrom_points`` the Nyström feature
    basis spans the full kernel column space, so the moment matching
    is exact (modulo FP roundoff). With ``num_selected < M`` the top
    eigenvectors are matched and the remaining components carry
    truncation error.
    """
    key = jax.random.PRNGKey(2026)
    candidates = jax.random.normal(key, (24, 1))
    nystrom_points = candidates[:4]

    indices, weights = sober_kernel_recombination(
        candidate_points=candidates,
        nystrom_points=nystrom_points,
        num_selected=5,
        kernel_fn=_rbf_kernel,
    )

    init_weights = jnp.ones(candidates.shape[0]) / candidates.shape[0]
    full_kernel_mean = init_weights @ _rbf_kernel(candidates, nystrom_points)
    thinned_kernel_mean = weights @ _rbf_kernel(candidates[indices], nystrom_points)

    # FP32 roundoff on SVD + matmul chain.
    assert jnp.allclose(full_kernel_mean, thinned_kernel_mean, atol=1e-4)


def test_sober_kernel_recombination_preserves_top_nystrom_eigenfeatures() -> None:
    r"""Truncated-rank SOBER preserves the projection onto the top eigenfeatures.

    The algorithmic guarantee is: ``Σ_j w_j U^T K(nys, x_{σ(j)}) ==
    Σ_i μ_i U^T K(nys, x_i)`` where ``U`` is the matrix of leading
    Nyström eigenvectors used as the recombination feature basis.
    """
    key = jax.random.PRNGKey(2026)
    candidates = jax.random.normal(key, (24, 1))
    nystrom_points = candidates[:6]
    num_selected = 4

    gram_matrix = _rbf_kernel(nystrom_points, nystrom_points)
    gram_symmetric = 0.5 * (gram_matrix + gram_matrix.T)
    left_vectors, _, _ = jnp.linalg.svd(gram_symmetric, full_matrices=False)
    feature_basis = left_vectors[:, : num_selected - 1].T

    indices, weights = sober_kernel_recombination(
        candidate_points=candidates,
        nystrom_points=nystrom_points,
        num_selected=num_selected,
        kernel_fn=_rbf_kernel,
    )

    init_weights = jnp.ones(candidates.shape[0]) / candidates.shape[0]
    full_features = init_weights @ (feature_basis @ _rbf_kernel(nystrom_points, candidates)).T
    thinned_features = weights @ (
        feature_basis @ _rbf_kernel(nystrom_points, candidates[indices])
    ).T

    assert jnp.allclose(full_features, thinned_features, atol=1e-4)


def test_sober_kernel_recombination_with_non_uniform_initial_weights() -> None:
    """Recombination respects a non-uniform initial weight distribution.

    Verified on the top Nyström eigenfeatures (the algorithmic guarantee).
    """
    key = jax.random.PRNGKey(7)
    key_a, key_b = jax.random.split(key)
    candidates = jax.random.normal(key_a, (20, 1))
    nystrom_points = candidates[:5]
    init_weights = jax.random.dirichlet(key_b, jnp.ones(20))
    num_selected = 4

    gram_symmetric = 0.5 * (
        _rbf_kernel(nystrom_points, nystrom_points)
        + _rbf_kernel(nystrom_points, nystrom_points).T
    )
    left_vectors, _, _ = jnp.linalg.svd(gram_symmetric, full_matrices=False)
    feature_basis = left_vectors[:, : num_selected - 1].T

    indices, weights = sober_kernel_recombination(
        candidate_points=candidates,
        nystrom_points=nystrom_points,
        num_selected=num_selected,
        kernel_fn=_rbf_kernel,
        weights_init=init_weights,
    )

    full_features = init_weights @ (feature_basis @ _rbf_kernel(nystrom_points, candidates)).T
    thinned_features = weights @ (
        feature_basis @ _rbf_kernel(nystrom_points, candidates[indices])
    ).T
    assert jnp.allclose(full_features, thinned_features, atol=1e-4)


# ---------------------------------------------------------------------------
# Adapter-spec wrap() concretization
# ---------------------------------------------------------------------------


def test_sober_adapter_spec_wrap_returns_kernel_recombination_callable() -> None:
    """``SOBERAdapterSpec.wrap`` returns the kernel-recombination primitive."""
    spec: Any = SOBERAdapterSpec()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)
    assert fn is sober_kernel_recombination
