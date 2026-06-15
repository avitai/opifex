"""Tests for assimilation predict / update / sequential helpers (Task 6.7)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.assimilation import (
    annotate_metadata,
    AssimilationState,
    build_default_metadata,
    observation_matrix_from_mask,
    predict,
    sequential_update,
    update,
)


def _state(mean: jax.Array, cov: jax.Array) -> AssimilationState:
    return AssimilationState(
        mean=mean,
        covariance=cov,
        time=jnp.array(0.0),
        metadata=build_default_metadata(
            physical_state="x",
            observation_uncertainty=0.0,
            model_discrepancy=0.0,
            numerical_uncertainty=0.0,
            calibration_uncertainty=0.0,
        ),
    )


def test_linear_gaussian_update_matches_textbook_innovation_and_gain() -> None:
    """Exit criterion: linear-Gaussian update reproduces the closed form.

    Scalar example with ``m_- = 0``, ``P_- = 1``, ``H = 1``, ``R = 1``, and
    observation ``y = 2``. Textbook posterior: ``m_+ = 1``, ``P_+ = 0.5``.
    """
    state = _state(jnp.array([0.0]), jnp.array([[1.0]]))
    posterior = update(
        state,
        observation=jnp.array([2.0]),
        observation_matrix=jnp.array([[1.0]]),
        observation_cov=jnp.array([[1.0]]),
    )
    assert jnp.allclose(posterior.mean, jnp.array([1.0]))
    assert jnp.allclose(posterior.covariance, jnp.array([[0.5]]))


def test_predict_then_update_carries_metadata_through() -> None:
    state = _state(jnp.array([1.0, 0.0]), jnp.eye(2))
    state = predict(
        state,
        transition=jnp.array([[1.0, 1.0], [0.0, 1.0]]),
        process_noise=0.1 * jnp.eye(2),
    )
    state = update(
        state,
        observation=jnp.array([2.0]),
        observation_matrix=jnp.array([[1.0, 0.0]]),
        observation_cov=jnp.array([[0.5]]),
    )
    assert state.metadata_dict()["physical_state"] == "x"


def test_observation_matrix_from_mask_selects_observed_dimensions() -> None:
    """Sparse/partial observations must be expressible via an observation operator."""
    mask = jnp.array([True, False, True])
    h = observation_matrix_from_mask(mask, state_dim=3)
    assert h.shape == (2, 3)
    # Row 0 should pick up dim 0, row 1 should pick up dim 2.
    assert jnp.allclose(h, jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


def test_sequential_update_scans_over_fixed_length_series() -> None:
    """Exit criterion: ``lax.scan``-style sequential update."""
    initial = _state(jnp.array([0.0]), jnp.array([[2.0]]))
    num_steps = 5
    transitions = jnp.broadcast_to(jnp.eye(1), (num_steps, 1, 1))
    process_noises = jnp.broadcast_to(0.1 * jnp.eye(1), (num_steps, 1, 1))
    observations = jnp.linspace(0.5, 2.5, num_steps).reshape(num_steps, 1)
    observation_matrices = jnp.broadcast_to(jnp.eye(1), (num_steps, 1, 1))
    observation_covs = jnp.broadcast_to(0.25 * jnp.eye(1), (num_steps, 1, 1))

    final_state, history = sequential_update(
        initial,
        transitions=transitions,
        process_noises=process_noises,
        observations=observations,
        observation_matrices=observation_matrices,
        observation_covs=observation_covs,
    )
    assert history.mean.shape == (num_steps, 1)
    assert history.covariance.shape == (num_steps, 1, 1)
    # Final estimate should track the observation series, which increases
    # monotonically — so the final mean is roughly the latest observation.
    assert final_state.mean[0] > history.mean[0, 0]


def test_annotate_metadata_records_sensor_calibration_status() -> None:
    """Exit criterion: update metadata records the sensor + calibration source."""
    state = _state(jnp.array([1.0]), jnp.array([[1.0]]))
    state = annotate_metadata(
        state,
        sensor_id="P-101",
        calibration_status="passed",
        observation_operator_name="pressure-sensor-linear",
    )
    meta = state.metadata_dict()
    assert meta["sensor_id"] == "P-101"
    assert meta["calibration_status"] == "passed"
    assert meta["observation_operator_name"] == "pressure-sensor-linear"


def test_update_and_predict_are_jit_compatible() -> None:
    """Exit criterion: JAX-transform compatibility."""

    @jax.jit
    def step(state: AssimilationState) -> AssimilationState:
        predicted = predict(
            state,
            transition=jnp.eye(2),
            process_noise=0.01 * jnp.eye(2),
        )
        return update(
            predicted,
            observation=jnp.array([1.0]),
            observation_matrix=jnp.array([[1.0, 0.0]]),
            observation_cov=jnp.array([[0.5]]),
        )

    state = _state(jnp.array([0.0, 0.0]), jnp.eye(2))
    result = step(state)
    assert result.mean.shape == (2,)
    assert result.covariance.shape == (2, 2)
