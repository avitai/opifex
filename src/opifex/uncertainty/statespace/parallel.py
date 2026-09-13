r"""Parallel-scan Kalman filter and smoother.

Both primitives produce numerically identical results to the sequential
forms in :mod:`opifex.uncertainty.statespace.kalman` but run in
:math:`O(\log N)` parallel depth instead of :math:`O(N)`. The associative
operator encodes the affine pencil ``(A, b, C, J, η)`` for filtering and
``(E, g, L)`` for smoothing — both of which compose associatively under
the standard Kalman fusion identities.

The filtering elements, their operator, the smoothing elements and their operator follow Lemmas 7
to 10 (eqs. 10 to 14) of Särkkä & García-Fernández (2021).

References:
----------
* Särkkä, S., García-Fernández, Á. F. 2021 — *Temporal Parallelization of
  Bayesian Smoothers*, IEEE Transactions on Automatic Control 66(1), 299-306,
  arXiv:1905.13002.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _make_filtering_element(
    transition: jax.Array,
    process_noise: jax.Array,
    observation_matrix: jax.Array,
    observation_cov: jax.Array,
    observation: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build a single filtering element ``(A, b, C, J, η)`` (Lemma 7, eqs. 10 and 12)."""
    h_q = observation_matrix @ process_noise
    h_a = observation_matrix @ transition
    innovation_cov = h_q @ observation_matrix.T + observation_cov
    inv_s_h = jnp.linalg.solve(innovation_cov, observation_matrix)
    gain = process_noise @ inv_s_h.T
    effective_transition = transition - gain @ h_a
    effective_observation = gain @ observation
    effective_noise = process_noise - gain @ h_q
    inv_s_h_a = (inv_s_h @ transition).T
    info_mean = inv_s_h_a @ observation
    info_precision = inv_s_h_a @ h_a
    return (
        effective_transition,
        effective_observation,
        effective_noise,
        info_precision,
        info_mean,
    )


def _filtering_operator(
    elem1: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    elem2: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Associative filtering operator (Lemma 8, eqs. 13 and 14)."""
    a1, b1, c1, j1, eta1 = elem1
    a2, b2, c2, j2, eta2 = elem2
    identity = jnp.eye(c1.shape[-1], dtype=c1.dtype)
    temp = jnp.linalg.solve(identity + c1 @ j2, identity)
    a2_temp = a2 @ temp
    new_transition = a2_temp @ a1
    new_b = a2_temp @ (b1 + c1 @ eta2) + b2
    new_c = a2_temp @ c1 @ a2.T + c2
    a1_temp = a1.T @ temp.T
    new_eta = a1_temp @ (eta2 - j2 @ b1) + eta1
    new_j = a1_temp @ j2 @ a1 + j1
    return new_transition, new_b, new_c, new_j, new_eta


def kalman_filter_parallel(
    *,
    transitions: jax.Array,
    process_noises: jax.Array,
    observations: jax.Array,
    observation_matrix: jax.Array,
    observation_covs: jax.Array,
    initial_mean: jax.Array,
    initial_cov: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Parallel-scan Kalman filter via associative composition.

    Identical output to :func:`kalman_filter` but runs in :math:`O(\log N)`
    parallel depth via :func:`jax.lax.associative_scan`.

    As in :func:`kalman_filter`, observation ``y_t`` follows the transition
    ``A_t`` out of step ``t - 1``. The first element conditions on the prior
    ``N(A_0 m_0, A_0 P_0 A_0^T + Q_0)``: its transition slot is the identity,
    its process noise is that prior covariance, and its offset is shifted by
    the prior mean.

    Args:
        transitions: Per-step transition matrices, shape ``(N, n, n)``.
        process_noises: Per-step process-noise covariances, ``(N, n, n)``.
        observations: Observation sequence, ``(N, k)``.
        observation_matrix: Time-invariant observation matrix, ``(k, n)``.
        observation_covs: Per-step observation noise covariances,
            ``(N, k, k)``.
        initial_mean: Prior mean ``m_0``, ``(n,)``.
        initial_cov: Prior covariance ``P_0``, ``(n, n)``.

    Returns:
        Filter means ``(N, n)`` and filter covariances ``(N, n, n)``.
    """
    state_dim = initial_mean.shape[0]
    first_transition = transitions[0]
    first_process_noise = process_noises[0]
    effective_mean = first_transition @ initial_mean
    effective_cov = first_transition @ initial_cov @ first_transition.T + first_process_noise

    identity = jnp.eye(state_dim, dtype=transitions.dtype)
    transitions_eff = transitions.at[0].set(identity)
    # The first element takes the effective prior covariance as its process noise.
    process_noises_eff = process_noises.at[0].set(effective_cov)

    elements = jax.vmap(_make_filtering_element, in_axes=(0, 0, None, 0, 0))(
        transitions_eff,
        process_noises_eff,
        observation_matrix,
        observation_covs,
        observations,
    )

    # Shift b[0] by the effective prior mean, which the element built above omits.
    init_innovation_cov = (
        observation_matrix @ effective_cov @ observation_matrix.T + observation_covs[0]
    )
    init_gain = jnp.linalg.solve(init_innovation_cov, observation_matrix @ effective_cov).T
    mean_correction = effective_mean - init_gain @ observation_matrix @ effective_mean
    adjusted_b = elements[1].at[0].add(mean_correction)
    elements = (elements[0], adjusted_b, elements[2], elements[3], elements[4])

    final_elements = jax.lax.associative_scan(jax.vmap(_filtering_operator), elements)
    filter_means = final_elements[1]
    filter_covs = final_elements[2]
    return filter_means, filter_covs


def _smoothing_element(
    transition: jax.Array,
    process_noise: jax.Array,
    filter_mean: jax.Array,
    filter_cov: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build a single smoothing element ``(E, g, L)`` (Lemma 9, eq. 11)."""
    predicted_cov = transition @ filter_cov @ transition.T + process_noise
    smoothing_gain = jnp.linalg.solve(predicted_cov, transition @ filter_cov).T
    g = filter_mean - smoothing_gain @ transition @ filter_mean
    l = filter_cov - smoothing_gain @ predicted_cov @ smoothing_gain.T
    return smoothing_gain, g, l


def _smoothing_operator(
    elem1: tuple[jax.Array, jax.Array, jax.Array],
    elem2: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Associative smoothing operator (Lemma 10)."""
    e1, g1, l1 = elem1
    e2, g2, l2 = elem2
    new_e = e2 @ e1
    new_g = e2 @ g1 + g2
    new_l = e2 @ l1 @ e2.T + l2
    return new_e, new_g, new_l


def kalman_smoother_parallel(
    *,
    filter_means: jax.Array,
    filter_covs: jax.Array,
    transitions: jax.Array,
    process_noises: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Parallel-scan Rauch-Tung-Striebel smoother.

    Identical output to :func:`kalman_smoother` but runs in
    :math:`O(\log N)` parallel depth via :func:`jax.lax.associative_scan`
    with ``reverse=True``.

    Args:
        filter_means: Sequential filter means, ``(N, n)``.
        filter_covs: Sequential filter covariances, ``(N, n, n)``.
        transitions: Per-step transition matrices, ``(N, n, n)``, as in
            :func:`kalman_filter`: the element at index ``t`` carries the state from step
            ``t - 1`` to step ``t``, so smoothing step ``t`` uses ``transitions[t + 1]``.
        process_noises: Per-step process noise covariances, ``(N, n, n)``, indexed like
            ``transitions``.

    Returns:
        Smoothed means ``(N, n)`` and smoothed covariances ``(N, n, n)``.
    """
    smoothing_elements = jax.vmap(_smoothing_element)(
        transitions[1:], process_noises[1:], filter_means[:-1], filter_covs[:-1]
    )
    state_dim = filter_means.shape[-1]
    last_gain = jnp.zeros((state_dim, state_dim), dtype=filter_means.dtype)
    last_g = filter_means[-1]
    last_l = filter_covs[-1]
    initial_elements = (
        jnp.concatenate([smoothing_elements[0], last_gain[None]], axis=0),
        jnp.concatenate([smoothing_elements[1], last_g[None]], axis=0),
        jnp.concatenate([smoothing_elements[2], last_l[None]], axis=0),
    )
    final_elements = jax.lax.associative_scan(
        jax.vmap(_smoothing_operator), initial_elements, reverse=True
    )
    smoothed_means = final_elements[1]
    smoothed_covs = final_elements[2]
    return smoothed_means, smoothed_covs
