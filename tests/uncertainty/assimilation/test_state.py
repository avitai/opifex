"""Tests for ``AssimilationState`` (Task 6.7)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.assimilation import (
    AssimilationState,
    build_default_metadata,
)


def _well_formed_metadata() -> tuple[tuple[str, object], ...]:
    return build_default_metadata(
        physical_state="reservoir-pressure",
        observation_uncertainty=0.05,
        model_discrepancy=0.02,
        numerical_uncertainty=1e-6,
        calibration_uncertainty=0.01,
    )


def test_assimilation_state_validate_passes_on_well_formed_input() -> None:
    state = AssimilationState(
        mean=jnp.array([1.0, 2.0, 3.0]),
        covariance=jnp.eye(3),
        time=jnp.array(0.0),
        metadata=_well_formed_metadata(),
    )
    state.validate()  # must not raise


def test_assimilation_state_validate_rejects_non_1d_mean() -> None:
    state = AssimilationState(
        mean=jnp.ones((2, 3)),
        covariance=jnp.eye(2),
        time=jnp.array(0.0),
        metadata=_well_formed_metadata(),
    )
    with pytest.raises(ValueError, match="mean must be 1-D"):
        state.validate()


def test_assimilation_state_validate_rejects_mismatched_covariance() -> None:
    state = AssimilationState(
        mean=jnp.array([1.0, 2.0]),
        covariance=jnp.eye(3),
        time=jnp.array(0.0),
        metadata=_well_formed_metadata(),
    )
    with pytest.raises(ValueError, match="covariance shape"):
        state.validate()


def test_assimilation_state_validate_requires_canonical_metadata_keys() -> None:
    state = AssimilationState(
        mean=jnp.array([1.0, 2.0]),
        covariance=jnp.eye(2),
        time=jnp.array(0.0),
        metadata=(("physical_state", "x"),),  # missing 4 of 5 required keys
    )
    with pytest.raises(ValueError, match="missing required keys"):
        state.validate()


def test_assimilation_state_metadata_dict_round_trips() -> None:
    state = AssimilationState(
        mean=jnp.array([1.0]),
        covariance=jnp.array([[1.0]]),
        time=jnp.array(0.0),
        metadata=_well_formed_metadata(),
    )
    meta = state.metadata_dict()
    assert meta["physical_state"] == "reservoir-pressure"
    assert meta["calibration_uncertainty"] == 0.01


def test_assimilation_state_is_a_jax_pytree() -> None:
    """Pattern-B ``flax.struct.dataclass`` round-trips through ``jax.tree``."""
    state = AssimilationState(
        mean=jnp.array([1.0, 2.0]),
        covariance=jnp.eye(2),
        time=jnp.array(0.0),
        metadata=_well_formed_metadata(),
    )
    leaves, treedef = jax.tree.flatten(state)
    # Data arrays are leaves; metadata stays static aux data.
    assert len(leaves) == 3
    restored = jax.tree.unflatten(treedef, leaves)
    assert restored.metadata == state.metadata
    assert jnp.array_equal(restored.mean, state.mean)


def test_assimilation_state_is_immutable() -> None:
    """``replace`` returns a new state — original is untouched."""
    original = AssimilationState(
        mean=jnp.array([1.0, 2.0]),
        covariance=jnp.eye(2),
        time=jnp.array(0.0),
        metadata=_well_formed_metadata(),
    )
    updated = original.replace(mean=jnp.array([5.0, 5.0]))
    assert jnp.array_equal(original.mean, jnp.array([1.0, 2.0]))
    assert jnp.array_equal(updated.mean, jnp.array([5.0, 5.0]))
