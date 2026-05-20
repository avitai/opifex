"""Field / function-space conformal contracts.

Score functions:

* ``L2``: ``sqrt(mean((y - ŷ)^2))`` over the spatial axes.
* ``Linf``: ``max(|y - ŷ|)`` over the spatial axes.
* ``H1``: ``L2`` of the field plus L2 of its finite-difference gradient.

Returned metadata MUST include: ``grid_axes``, ``time_axis``,
``spatial_axes``, ``norm``, ``alpha``, ``calibration_size``,
``assumption_status``. The metadata schema lives in
:class:`opifex.uncertainty.scientific.fields.FieldMetadata`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _import_fields():
    from opifex.uncertainty.conformal import fields

    return fields


# ---------------------------------------------------------------------------
# Field score functions
# ---------------------------------------------------------------------------


def test_l2_field_score_matches_definition() -> None:
    fields = _import_fields()
    predictions = jnp.zeros((2, 4, 4))
    targets = jnp.ones((2, 4, 4))
    # spatial axes (-2, -1): mean((1 - 0)^2) = 1.0 over each sample → sqrt = 1.0.
    scores = fields.field_l2_score(predictions=predictions, targets=targets, spatial_axes=(-2, -1))
    assert scores.shape == (2,)
    assert bool(jnp.allclose(scores, jnp.array([1.0, 1.0]), atol=1e-6))


def test_linf_field_score_matches_definition() -> None:
    fields = _import_fields()
    predictions = jnp.zeros((1, 3, 3))
    targets = jnp.asarray(np.arange(9).reshape((1, 3, 3)).astype(np.float32))
    scores = fields.field_linf_score(
        predictions=predictions, targets=targets, spatial_axes=(-2, -1)
    )
    assert float(scores[0]) == pytest.approx(8.0, abs=1e-6)


def test_h1_field_score_includes_gradient_component() -> None:
    """H1 norm = sqrt(L2(field)^2 + L2(grad)^2). Score should exceed L2 alone
    for a non-constant field."""
    fields = _import_fields()
    rng = np.random.default_rng(0)
    predictions = jnp.zeros((1, 8, 8))
    targets = jnp.asarray(rng.standard_normal(size=(1, 8, 8)).astype(np.float32))
    l2 = float(
        fields.field_l2_score(predictions=predictions, targets=targets, spatial_axes=(-2, -1))[0]
    )
    h1 = float(
        fields.field_h1_score(predictions=predictions, targets=targets, spatial_axes=(-2, -1))[0]
    )
    assert h1 > l2


# ---------------------------------------------------------------------------
# Field calibrator
# ---------------------------------------------------------------------------


def test_field_split_conformal_returns_per_sample_interval() -> None:
    fields = _import_fields()
    rng = np.random.default_rng(0)
    n_calib = 256
    spatial = (8, 8)
    cal_preds = jnp.zeros((n_calib, *spatial))
    cal_targets = jnp.asarray(
        rng.standard_normal(size=(n_calib, *spatial)).astype(np.float32) * 0.3
    )
    cp = fields.FieldSplitConformalRegressor(alpha=0.1, norm="L2", spatial_axes=(-2, -1))
    state = cp.fit(predictions=cal_preds, targets=cal_targets)
    test_preds = jnp.zeros((4, *spatial))
    interval = cp.with_state(state).predict(predictions=test_preds)
    # Interval bounds broadcast the scalar field threshold over the predictions.
    assert interval.lower.shape == (4, *spatial)
    assert interval.upper.shape == (4, *spatial)
    assert interval.method == "field_split_conformal"


def test_field_calibrator_metadata_records_norm_axes_and_assumption_status() -> None:
    fields = _import_fields()
    cal_preds = jnp.zeros((64, 8, 8))
    cal_targets = jnp.zeros((64, 8, 8))
    cp = fields.FieldSplitConformalRegressor(alpha=0.1, norm="L2", spatial_axes=(-2, -1))
    state = cp.fit(predictions=cal_preds, targets=cal_targets)
    md = dict(state.metadata)
    assert md["method"] == "field_split_conformal"
    assert md["norm"] == "L2"
    assert tuple(md["spatial_axes"]) == (-2, -1)
    assert md["alpha"] == pytest.approx(0.1)
    assert int(md["calibration_size"]) == 64
    assert md["assumption_status"] in {"exchangeable", "exchangeable_assumed"}


def test_field_calibrator_rejects_unknown_norm() -> None:
    fields = _import_fields()
    with pytest.raises(ValueError, match=r"(?i)norm"):
        fields.FieldSplitConformalRegressor(
            alpha=0.1,
            norm="invalid_norm",
            spatial_axes=(-2, -1),  # type: ignore[arg-type]
        )


def test_field_calibrator_predict_before_fit_raises() -> None:
    fields = _import_fields()
    cp = fields.FieldSplitConformalRegressor(alpha=0.1, norm="L2", spatial_axes=(-2, -1))
    with pytest.raises(RuntimeError, match=r"(?i)(fit|calibrate)"):
        cp.predict(predictions=jnp.zeros((1, 4, 4)))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Failed-exchangeability propagation
# ---------------------------------------------------------------------------


def test_field_conformal_propagates_failed_exchangeability_into_metadata() -> None:
    """When a caller supplies a failing ExchangeabilityReport, the field
    calibrator must NOT claim 'exchangeable' coverage."""
    fields = _import_fields()
    from opifex.uncertainty.conformal import ExchangeabilityReport

    cal_preds = jnp.zeros((64, 8, 8))
    cal_targets = jnp.zeros((64, 8, 8))
    failing_report = ExchangeabilityReport(p_value=jnp.asarray(0.001), passes=False)
    cp = fields.FieldSplitConformalRegressor(alpha=0.1, norm="L2", spatial_axes=(-2, -1))
    state = cp.fit(
        predictions=cal_preds, targets=cal_targets, exchangeability_report=failing_report
    )
    md = dict(state.metadata)
    assert md["assumption_status"] == "exchangeability_failed"


# ---------------------------------------------------------------------------
# Transform compatibility
# ---------------------------------------------------------------------------


def test_field_score_kernels_are_jit_compatible() -> None:
    fields = _import_fields()
    predictions = jnp.zeros((4, 8, 8))
    targets = jnp.ones((4, 8, 8))
    jitted_l2 = jax.jit(
        lambda p, t: fields.field_l2_score(predictions=p, targets=t, spatial_axes=(-2, -1))
    )
    out = jitted_l2(predictions, targets)
    assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_conformal_surface_includes_field_components() -> None:
    from opifex.uncertainty import conformal

    expected = {
        "field_l2_score",
        "field_linf_score",
        "field_h1_score",
        "FieldSplitConformalRegressor",
        "FieldSplitConformalState",
    }
    missing = expected - set(dir(conformal))
    assert not missing, f"missing public field-conformal symbols: {sorted(missing)}"
