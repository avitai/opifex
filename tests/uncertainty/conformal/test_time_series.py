"""EnbPI and Adaptive Conformal Inference (ACI) contracts.

References:

* Xu, Xie 2021, "Conformal Prediction Interval for Dynamic Time-Series"
  (EnbPI, arXiv:2010.09107) — ensemble bootstrap residuals; sequential
  online update without retraining.
* Gibbs, Candès 2021, "Adaptive Conformal Inference Under Distribution
  Shift" (arXiv:2106.00170) — online alpha-update rule that adapts to
  drift; coverage holds in the long run regardless of exchangeability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def _import_ts():
    from opifex.uncertainty.conformal import time_series

    return time_series


# ---------------------------------------------------------------------------
# EnbPI
# ---------------------------------------------------------------------------


def test_enbpi_state_initialises_with_residual_window() -> None:
    ts = _import_ts()
    residuals = jnp.array([0.1, 0.2, 0.05, -0.1, 0.3])
    state = ts.EnbPIState(residual_window=residuals, alpha=0.1)
    assert state.residual_window.shape == (5,)
    assert state.alpha == 0.1


def test_enbpi_update_appends_residual_and_drops_oldest() -> None:
    """Sequential update slides the residual window without hidden state."""
    ts = _import_ts()
    initial = jnp.array([0.1, 0.2, 0.05, -0.1, 0.3])
    state = ts.EnbPIState(residual_window=initial, alpha=0.1)
    new_state = ts.enbpi_update(state=state, new_residual=jnp.asarray(0.5))
    expected = jnp.array([0.2, 0.05, -0.1, 0.3, 0.5])
    assert bool(jnp.allclose(new_state.residual_window, expected))
    # Original state must be unchanged (no in-place mutation).
    assert bool(jnp.allclose(state.residual_window, initial))


def test_enbpi_interval_width_grows_with_residual_magnitude() -> None:
    ts = _import_ts()
    small_residuals = jnp.array([0.01, 0.02, -0.01, 0.0, 0.01])
    large_residuals = jnp.array([1.0, -1.5, 0.8, -0.9, 1.2])
    state_small = ts.EnbPIState(residual_window=small_residuals, alpha=0.1)
    state_large = ts.EnbPIState(residual_window=large_residuals, alpha=0.1)
    predictions = jnp.array([0.0, 0.5, -0.3])
    interval_small = ts.enbpi_predict(state=state_small, predictions=predictions)
    interval_large = ts.enbpi_predict(state=state_large, predictions=predictions)
    width_small = float(jnp.mean(interval_small.upper - interval_small.lower))
    width_large = float(jnp.mean(interval_large.upper - interval_large.lower))
    assert width_large > width_small * 10


def test_enbpi_records_window_size_and_alpha_metadata() -> None:
    ts = _import_ts()
    residuals = jnp.zeros((32,))
    state = ts.EnbPIState(residual_window=residuals, alpha=0.1)
    predictions = jnp.zeros((4,))
    interval = ts.enbpi_predict(state=state, predictions=predictions)
    md = dict(interval.metadata)
    assert md["method"] == "enbpi"
    assert int(md["window_size"]) == 32
    assert md["alpha"] == pytest.approx(0.1)
    assert md["assumption_status"] == "approximate_marginal"


# ---------------------------------------------------------------------------
# Adaptive Conformal Inference (ACI)
# ---------------------------------------------------------------------------


def test_aci_state_carries_running_alpha() -> None:
    ts = _import_ts()
    state = ts.AdaptiveConformalState(
        current_alpha=jnp.asarray(0.1),
        target_alpha=0.1,
        learning_rate=0.05,
    )
    assert float(state.current_alpha) == pytest.approx(0.1)
    assert state.target_alpha == 0.1
    assert state.learning_rate == 0.05


def test_aci_update_increases_alpha_on_uncovered_observation() -> None:
    """When the observation falls outside the interval, the running alpha
    must increase (interval will widen next step). Per Gibbs & Candès 2021:
    alpha_{t+1} = alpha_t + lr * (target_alpha - 1_{y_t \\notin C_t})."""
    ts = _import_ts()
    state = ts.AdaptiveConformalState(
        current_alpha=jnp.asarray(0.1),
        target_alpha=0.1,
        learning_rate=0.05,
    )
    # Observation outside interval: indicator=1 → alpha decreases (yields wider next).
    new_state = ts.aci_update(state=state, was_covered=False)
    assert float(new_state.current_alpha) < float(state.current_alpha)


def test_aci_update_increases_alpha_on_covered_observation() -> None:
    """When observation falls inside, alpha drifts back upward."""
    ts = _import_ts()
    state = ts.AdaptiveConformalState(
        current_alpha=jnp.asarray(0.05),
        target_alpha=0.1,
        learning_rate=0.05,
    )
    new_state = ts.aci_update(state=state, was_covered=True)
    assert float(new_state.current_alpha) > float(state.current_alpha)


def test_aci_records_distribution_shift_assumption_status() -> None:
    ts = _import_ts()
    state = ts.AdaptiveConformalState(
        current_alpha=jnp.asarray(0.1),
        target_alpha=0.1,
        learning_rate=0.05,
    )
    info = ts.aci_metadata(state=state)
    md = dict(info)
    assert md["method"] == "adaptive_conformal_inference"
    assert md["assumption_status"] == "long_run_marginal"


# ---------------------------------------------------------------------------
# Transform compatibility
# ---------------------------------------------------------------------------


def test_enbpi_update_is_jit_compatible() -> None:
    ts = _import_ts()
    residuals = jnp.zeros((16,))
    state = ts.EnbPIState(residual_window=residuals, alpha=0.1)
    jitted = jax.jit(lambda s, r: ts.enbpi_update(state=s, new_residual=r))
    out = jitted(state, jnp.asarray(0.5))
    assert out.residual_window.shape == (16,)
    assert float(out.residual_window[-1]) == pytest.approx(0.5, abs=1e-6)


def test_aci_update_is_jit_compatible() -> None:
    ts = _import_ts()
    state = ts.AdaptiveConformalState(
        current_alpha=jnp.asarray(0.1),
        target_alpha=0.1,
        learning_rate=0.05,
    )
    jitted = jax.jit(lambda s, c: ts.aci_update(state=s, was_covered=c))
    out_covered = jitted(state, True)
    out_uncovered = jitted(state, False)
    assert float(out_covered.current_alpha) > float(state.current_alpha)
    assert float(out_uncovered.current_alpha) < float(state.current_alpha)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_conformal_surface_includes_time_series_components() -> None:
    from opifex.uncertainty import conformal

    expected = {
        "EnbPIState",
        "AdaptiveConformalState",
        "enbpi_update",
        "enbpi_predict",
        "aci_update",
        "aci_metadata",
    }
    missing = expected - set(dir(conformal))
    assert not missing, f"missing public time-series symbols: {sorted(missing)}"
