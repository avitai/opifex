r"""Assimilation update helpers (Task 6.7).

Thin orchestration layer over :mod:`opifex.uncertainty.statespace`:
this module supplies the digital-twin-shaped wrappers
(state-aware ``AssimilationState`` in/out, optional sensor mask via
``observation_matrix``, sequential ``jax.lax.scan`` driver) but the
actual Kalman math is imported from the canonical state-space module.

Per Task 6.7 design constraint: **no formula body for Kalman math
appears in this module** — the Phase 9 audit checks
``rg 'kalman_gain|innovation_cov|posterior_mean\s*=|posterior_cov\s*='
src/opifex/uncertainty/assimilation`` returns zero matches.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from opifex.uncertainty.assimilation.state import AssimilationState  # noqa: TC001
from opifex.uncertainty.statespace import (
    kalman_predict as _kalman_predict,
    kalman_update as _kalman_update,
)


def predict(
    state: AssimilationState,
    *,
    transition: jax.Array,
    process_noise: jax.Array,
    new_time: jax.Array | None = None,
) -> AssimilationState:
    """Advance ``state`` one step using the Kalman prediction primitive.

    Math is delegated to ``opifex.uncertainty.statespace.kalman_predict``.

    Args:
        state: Current ``AssimilationState``.
        transition: ``A`` matrix, shape ``(state_dim, state_dim)``.
        process_noise: ``Q`` matrix, shape ``(state_dim, state_dim)``.
        new_time: Optional new timestamp; defaults to ``state.time + 1``.

    Returns:
        Updated ``AssimilationState`` with carried-over metadata.
    """
    predicted_mean, predicted_cov = _kalman_predict(
        mean=state.mean,
        cov=state.covariance,
        transition=transition,
        process_noise=process_noise,
    )
    new_t = state.time + 1.0 if new_time is None else jnp.asarray(new_time)
    return dataclasses.replace(state, mean=predicted_mean, covariance=predicted_cov, time=new_t)


def update(
    state: AssimilationState,
    *,
    observation: jax.Array,
    observation_matrix: jax.Array,
    observation_cov: jax.Array,
) -> AssimilationState:
    """Condition ``state`` on a (possibly sparse) sensor observation.

    Math is delegated to ``opifex.uncertainty.statespace.kalman_update``.
    Sparse / partial observations are handled by passing a thinned
    ``observation_matrix`` that selects only the observed dimensions
    (or a boolean-mask-built matrix on the caller side).

    Args:
        state: Predicted ``AssimilationState``.
        observation: Sensor reading ``y`` of shape ``(obs_dim,)``.
        observation_matrix: ``H`` of shape ``(obs_dim, state_dim)``.
        observation_cov: ``R`` of shape ``(obs_dim, obs_dim)``.

    Returns:
        Updated ``AssimilationState`` with carried-over metadata.
    """
    updated_mean, updated_cov = _kalman_update(
        mean=state.mean,
        cov=state.covariance,
        observation=observation,
        observation_matrix=observation_matrix,
        observation_cov=observation_cov,
    )
    return dataclasses.replace(state, mean=updated_mean, covariance=updated_cov)


def observation_matrix_from_mask(mask: jax.Array, state_dim: int) -> jax.Array:
    """Build ``H`` that selects only observed dimensions of the state vector.

    Args:
        mask: 1-D boolean / 0-1 array, length ``state_dim``. ``True``
            entries are observed.
        state_dim: Full state dimension.

    Returns:
        ``H`` of shape ``(num_observed, state_dim)`` whose rows are the
        canonical basis vectors of the observed dimensions.
    """
    return jnp.eye(state_dim)[jnp.asarray(mask, dtype=bool)]


def sequential_update(
    state: AssimilationState,
    *,
    transitions: jax.Array,
    process_noises: jax.Array,
    observations: jax.Array,
    observation_matrices: jax.Array,
    observation_covs: jax.Array,
) -> tuple[AssimilationState, AssimilationState]:
    """Roll ``state`` forward through ``T`` predict/update steps via ``lax.scan``.

    All time-varying inputs have a leading ``T`` axis. The state and the
    per-step posterior history are returned so the caller can recover
    smoothed estimates downstream.

    Args:
        state: Initial ``AssimilationState``.
        transitions: ``(T, state_dim, state_dim)``.
        process_noises: ``(T, state_dim, state_dim)``.
        observations: ``(T, obs_dim)``.
        observation_matrices: ``(T, obs_dim, state_dim)``.
        observation_covs: ``(T, obs_dim, obs_dim)``.

    Returns:
        ``(final_state, history_state)``. ``history_state`` has each
        leaf prefixed with the time axis, so e.g.
        ``history_state.mean.shape == (T, state_dim)``.
    """

    def step(
        carry: AssimilationState,
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[AssimilationState, AssimilationState]:
        """Run one sequential assimilation step, updating the state with new data."""
        transition, process_noise, obs, obs_matrix, obs_cov = inputs
        predicted = predict(carry, transition=transition, process_noise=process_noise)
        updated = update(
            predicted,
            observation=obs,
            observation_matrix=obs_matrix,
            observation_cov=obs_cov,
        )
        return updated, updated

    final_state, history = jax.lax.scan(
        step,
        state,
        (
            transitions,
            process_noises,
            observations,
            observation_matrices,
            observation_covs,
        ),
    )
    return final_state, history


def annotate_metadata(state: AssimilationState, **extra: Any) -> AssimilationState:
    """Return a copy of ``state`` with ``extra`` merged into ``metadata``.

    Useful when each assimilation step records additional bookkeeping
    (e.g. sensor name, calibration status). Preserves the required
    canonical keys.
    """
    merged = dict(state.metadata)
    merged.update(extra)
    return dataclasses.replace(state, metadata=tuple(merged.items()))


__all__ = [
    "annotate_metadata",
    "observation_matrix_from_mask",
    "predict",
    "sequential_update",
    "update",
]
