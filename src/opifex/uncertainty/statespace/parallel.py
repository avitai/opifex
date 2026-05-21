r"""Parallel-scan Kalman filter and smoother.

Both primitives produce numerically identical results to the sequential
forms in :mod:`opifex.uncertainty.statespace.kalman` but run in
:math:`O(\log N)` parallel depth instead of :math:`O(N)`. The associative
operator encodes the affine pencil ``(A, b, C, J, η)`` for filtering and
``(E, g, L)`` for smoothing — both of which compose associatively under
the standard Kalman fusion identities.

Canonical reference (line-by-line port):
* ``../bayesnewton/bayesnewton/ops.py`` ``parallel_filtering_element_``
  (line 183), ``parallel_filtering_operator`` (line 204),
  ``make_associative_filtering_elements`` (line 222),
  ``parallel_smoothing_element`` (line 319),
  ``parallel_smoothing_operator`` (line 329),
  ``_parallel_rts`` (line 338).

References
----------
* Särkkä & García-Fernández 2021 — *Temporal parallelization of Bayesian
  smoothers*, IEEE TAC arXiv:1905.13002.
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
    """Build a single filtering element ``(A, b, C, J, η)``.

    Ports bayesnewton ``parallel_filtering_element_`` (line 183).
    """
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
    """Associative filtering operator. Ports bayesnewton line 204."""
    a1, b1, c1, j1, eta1 = elem1
    a2, b2, c2, j2, eta2 = elem2
    c1_inv = jnp.linalg.inv(c1)
    temp = jnp.linalg.solve(c1_inv + j2, c1_inv)
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

    The bayesnewton reference uses an "observe at step ``t`` then transition
    to step ``t+1``" indexing, whereas opifex uses "predict via ``A_t`` then
    update with ``y_t``" (i.e., observation ``y_t`` is observed at time
    ``t+1`` relative to the initial prior). To bridge the two, the first
    transition is folded into an effective prior
    ``N(A_0 m_0, A_0 P_0 A_0^T + Q_0)`` and the first transition slot in
    the parallel pipeline is replaced with the identity.

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
    # bayesnewton's substitution (line 223): set process_noises[0] = prior cov
    # — here the effective prior cov is what we just computed.
    process_noises_eff = process_noises.at[0].set(effective_cov)

    elements = jax.vmap(_make_filtering_element, in_axes=(0, 0, None, 0, 0))(
        transitions_eff,
        process_noises_eff,
        observation_matrix,
        observation_covs,
        observations,
    )

    # Adjust b[0] for non-zero effective_mean (ports line 228).
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
    """Build a single smoothing element ``(E, g, L)``. Ports bayesnewton line 319."""
    predicted_cov = transition @ filter_cov @ transition.T + process_noise
    smoothing_gain = jnp.linalg.solve(predicted_cov, transition @ filter_cov).T
    g = filter_mean - smoothing_gain @ transition @ filter_mean
    l = filter_cov - smoothing_gain @ predicted_cov @ smoothing_gain.T
    return smoothing_gain, g, l


def _smoothing_operator(
    elem1: tuple[jax.Array, jax.Array, jax.Array],
    elem2: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Associative smoothing operator. Ports bayesnewton line 329."""
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
        transitions: Per-step transition matrices, ``(N, n, n)``. The
            element at index ``t`` is the transition from ``t`` to ``t+1``.
        process_noises: Per-step process noise covariances, ``(N, n, n)``.

    Returns:
        Smoothed means ``(N, n)`` and smoothed covariances ``(N, n, n)``.
    """
    smoothing_elements = jax.vmap(_smoothing_element)(
        transitions, process_noises, filter_means, filter_covs
    )
    state_dim = filter_means.shape[-1]
    last_gain = jnp.zeros((state_dim, state_dim), dtype=filter_means.dtype)
    last_g = filter_means[-1]
    last_l = filter_covs[-1]
    initial_elements = (
        jnp.concatenate([smoothing_elements[0][:-1], last_gain[None]], axis=0),
        jnp.concatenate([smoothing_elements[1][:-1], last_g[None]], axis=0),
        jnp.concatenate([smoothing_elements[2][:-1], last_l[None]], axis=0),
    )
    final_elements = jax.lax.associative_scan(
        jax.vmap(_smoothing_operator), initial_elements, reverse=True
    )
    smoothed_means = final_elements[1]
    smoothed_covs = final_elements[2]
    return smoothed_means, smoothed_covs
