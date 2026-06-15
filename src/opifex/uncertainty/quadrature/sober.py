r"""SOBER kernel-recombination point-set Bayesian quadrature.

A JAX-native port of the SOBER algorithm (Adachi et al,
arXiv:2206.04734 + arXiv:2301.11832) — kernel recombination via the
Tchernychova-Lyons CAR (Caratheodory recombination) algorithm with a
Nyström low-rank kernel approximation.

The algorithm reduces a large ``N``-point empirical measure to a
sparse ``n+1``-point empirical measure while exactly preserving the
weighted moments of the kernel-feature embedding. With the kernel
features given by the leading singular vectors of a Nyström subsample
``K(nys, nys)``, this produces a point set whose kernel mean
embedding matches the candidate distribution's — i.e. an optimal
point set for kernel-based quadrature on the Nyström subspace.

Sibling reference (READ-ONLY port — never imported at runtime):
``../SOBER/SOBER/_rchq.py`` — specifically ``recombination``
(line 5), ``rc_kernel_svd`` (line 42), ``ker_svd_sparsify`` (line
34), ``Mod_Tchernychova_Lyons`` (line 51), and ``Tchernychova_Lyons_CAR``
(line 224). The original lives at github.com/ma921/SOBER, derived
from the Caratheodory-Tchernychova-Lyons recombination of
github.com/FraCose/Recombination_Random_Algos.

References
----------
* Adachi, M. et al. 2022 — *Fast Bayesian Inference with Batch
  Bayesian Quadrature via Kernel Recombination*, NeurIPS.
  arXiv:2206.04734.
* Adachi, M. et al. 2023 — *SOBER: Highly Parallel Bayesian
  Optimization and Bayesian Quadrature over Discrete and Mixed
  Spaces*, TMLR. arXiv:2301.11832.
* Tchernychova, M. & Lyons, T. 2016 — *Caratheodory cubature
  measures*. PhD thesis (Oxford).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager per opifex convention

import jax
import jax.numpy as jnp


def caratheodory_recombination(
    *,
    features: jax.Array,
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Caratheodory recombination via null-space iteration.

    Given ``N`` feature vectors ``φ(x_i) ∈ R^n`` with positive
    convex-combination weights ``μ_i`` summing to one, returns a
    sparser point set of size at most ``n + 1`` together with new
    positive weights that exactly preserve the weighted moment
    ``Σ_i μ_i φ(x_i)`` (Caratheodory's theorem).

    The algorithm builds a null-space basis ``Φ`` of the augmented
    feature matrix ``[1, φ]^T`` via SVD, then performs ``N - n - 1``
    rank-one updates that zero out one weight per iteration while
    holding the moment invariant.

    Sibling reference (READ-ONLY port — no runtime import):
    ``../SOBER/SOBER/_rchq.py:Tchernychova_Lyons_CAR`` (line 224).

    Args:
        features: ``(N, n)`` feature matrix in the kernel-eigenfunction
            basis.
        weights: ``(N,)`` non-negative initial weights summing to one.

    Returns:
        ``(indices, recombination_weights)`` — the selected support
        indices into the input ``features`` and the positive recombined
        weights. ``len(indices) <= n + 1``.
    """
    num_points = features.shape[0]
    augmented = jnp.concatenate([jnp.ones((num_points, 1)), features], axis=1)  # (N, n + 1)
    augmented_dim = augmented.shape[1]

    if num_points <= augmented_dim:
        nonzero_mask = weights > 0.0
        indices = jnp.arange(num_points)[nonzero_mask]
        return indices, weights[nonzero_mask]

    _, _, right_vectors = jnp.linalg.svd(augmented.T, full_matrices=True)
    null_space = right_vectors[augmented_dim:, :].T  # (N, N - augmented_dim)

    mutable_weights = weights
    for column_index in range(num_points - augmented_dim):
        direction = null_space[:, column_index]
        positive_direction_mask = direction > 0.0
        safe_direction = jnp.where(direction > 0.0, direction, 1.0)
        ratio = jnp.where(positive_direction_mask, mutable_weights / safe_direction, jnp.inf)
        pivot_index = jnp.argmin(ratio)
        pivot_step = mutable_weights[pivot_index] / direction[pivot_index]
        mutable_weights = mutable_weights - pivot_step * direction
        mutable_weights = mutable_weights.at[pivot_index].set(0.0)

        if column_index + 1 < num_points - augmented_dim:
            pivot_column = direction
            remaining = null_space[:, column_index + 1 :]
            update = jnp.outer(
                pivot_column,
                remaining[pivot_index] / pivot_column[pivot_index],
            )
            null_space = null_space.at[:, column_index + 1 :].set(remaining - update)
            null_space = null_space.at[pivot_index, column_index + 1 :].set(0.0)

    support_mask = mutable_weights > 0.0
    indices = jnp.arange(num_points)[support_mask]
    return indices, mutable_weights[support_mask]


def _kernel_svd_sparsify(
    nystrom_points: jax.Array,
    num_features: int,
    kernel_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    r"""Top ``num_features`` left singular vectors of the Nyström gram matrix.

    Returns ``U^T`` (shape ``(num_features, M)``) — the leading
    eigenfunctions of ``K(nystrom_points, nystrom_points)`` used as
    the feature basis for SOBER's moment-matching.

    Sibling reference (READ-ONLY port — no runtime import):
    ``../SOBER/SOBER/_rchq.py:ker_svd_sparsify`` (line 34).
    """
    gram_matrix = kernel_fn(nystrom_points, nystrom_points)
    symmetrised = 0.5 * (gram_matrix + gram_matrix.T)
    left_vectors, _, _ = jnp.linalg.svd(symmetrised, full_matrices=False)
    return left_vectors[:, :num_features].T


def sober_kernel_recombination(
    *,
    candidate_points: jax.Array,
    nystrom_points: jax.Array,
    num_selected: int,
    kernel_fn: Callable[[jax.Array, jax.Array], jax.Array],
    weights_init: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""SOBER kernel recombination entry point.

    Selects ``num_selected`` representatives from ``candidate_points``
    together with positive weights that exactly preserve the weighted
    kernel mean embedding in the ``(num_selected - 1)``-dimensional
    Nyström subspace spanned by the leading singular vectors of
    ``K(nystrom_points, nystrom_points)``.

    Sibling reference (READ-ONLY port — no runtime import):
    ``../SOBER/SOBER/_rchq.py:recombination`` (line 5) and
    ``rc_kernel_svd`` (line 42).

    Args:
        candidate_points: ``(N, d)`` empirical-measure samples.
        nystrom_points: ``(M, d)`` subsample for low-rank approximation
            of the kernel.
        num_selected: Target number of output points ``n``. The
            Nyström rank is ``n - 1`` so the feature dimension is
            ``n - 1`` and Caratheodory yields at most ``n`` non-zero
            weights.
        kernel_fn: ``(X, Y) -> K`` kernel matrix function. Must produce
            a symmetric PSD matrix on ``X == Y``.
        weights_init: ``(N,)`` optional initial weights (must be
            non-negative and sum to one). Defaults to the uniform
            distribution.

    Returns:
        ``(selected_indices, selected_weights)`` — indices into
        ``candidate_points`` and the corresponding positive
        recombination weights.
    """
    num_candidates = candidate_points.shape[0]
    if weights_init is None:
        weights_init = jnp.ones(num_candidates) / num_candidates

    feature_basis = _kernel_svd_sparsify(
        nystrom_points, num_selected - 1, kernel_fn
    )  # (num_selected - 1, M)
    candidate_kernels = kernel_fn(nystrom_points, candidate_points)  # (M, N)
    features = (feature_basis @ candidate_kernels).T  # (N, num_selected - 1)
    return caratheodory_recombination(features=features, weights=weights_init)
