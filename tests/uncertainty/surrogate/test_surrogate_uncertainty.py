"""Tests for the surrogate-uncertainty decomposition (Task 6.6)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from opifex.uncertainty.surrogate import (
    decompose_surrogate_uncertainty,
    SurrogateUncertaintyResult,
)


def test_decomposition_combines_all_three_components_in_quadrature() -> None:
    """Plan exit criterion: pred + resid + cal combine via sum-of-squares."""
    pred = jnp.array([0.5, 1.0])
    resid = jnp.array([0.5, 0.5])
    cal = jnp.array([0.5, 0.0])
    result = decompose_surrogate_uncertainty(
        prediction_std=pred,
        residual_std=resid,
        calibration_std=cal,
    )
    assert isinstance(result, SurrogateUncertaintyResult)
    expected_total = jnp.sqrt(pred**2 + resid**2 + cal**2)
    assert jnp.allclose(result.total_std, expected_total)


def test_decomposition_defaults_unsupplied_components_to_zero() -> None:
    """When only ``prediction_std`` is supplied, total == prediction."""
    pred = jnp.array([0.3, 0.7, 1.0])
    result = decompose_surrogate_uncertainty(prediction_std=pred)
    assert jnp.allclose(result.total_std, pred)
    assert jnp.allclose(result.residual_std, jnp.zeros_like(pred))
    assert jnp.allclose(result.calibration_std, jnp.zeros_like(pred))


def test_decomposition_rejects_negative_standard_deviations() -> None:
    """Plan exit criterion: invalid (negative) std raises ``ValueError``."""
    with pytest.raises(ValueError, match="non-negative"):
        decompose_surrogate_uncertainty(
            prediction_std=jnp.array([-1.0, 0.5]),
        )
