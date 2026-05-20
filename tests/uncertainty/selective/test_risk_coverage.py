"""Selective prediction: risk-coverage curve, AURC, and abstention decisions.

Reference: Geifman & El-Yaniv 2017 ("Selective Classification for Deep
Neural Networks", NeurIPS, arXiv:1705.08500).

Conventions:

* ``confidences``: higher = more confident, more likely accepted.
* ``errors``: per-sample loss (e.g. 0/1 error, squared error, etc.); HIGHER
  is WORSE.
* Coverage = fraction accepted at a given confidence threshold.
* Selective risk = mean error on accepted samples.
* AURC = area under the risk-coverage curve from coverage 0 → 1.
"""

from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _import_selective():
    from opifex.uncertainty import selective

    return selective


# ---------------------------------------------------------------------------
# Risk-coverage curve
# ---------------------------------------------------------------------------


def test_risk_coverage_curve_returns_coverages_in_ascending_order() -> None:
    """Sweeping confidence thresholds from highest to lowest grows coverage
    monotonically from 0 to 1."""
    selective = _import_selective()
    confidences = jnp.array([0.1, 0.4, 0.6, 0.9])
    errors = jnp.array([1.0, 1.0, 0.0, 0.0])
    coverages, _ = selective.risk_coverage_curve(confidences=confidences, errors=errors)
    # n+1 points (from coverage 0/n to n/n) or n points; coverage monotone.
    assert bool(jnp.all(coverages[:-1] <= coverages[1:]))
    assert float(coverages[-1]) == pytest.approx(1.0, abs=1e-6)


def test_selective_risk_is_computed_on_accepted_samples_only() -> None:
    """At a confidence threshold that admits only the highest-confidence
    samples, selective risk equals the mean error on those samples."""
    selective = _import_selective()
    confidences = jnp.array([0.1, 0.4, 0.6, 0.9])
    errors = jnp.array([1.0, 1.0, 0.0, 0.0])
    coverages, risks = selective.risk_coverage_curve(confidences=confidences, errors=errors)
    # When the top-2 samples (confidences 0.6 and 0.9) are accepted,
    # coverage = 0.5, selective risk = mean([0, 0]) = 0.
    coverage_idx = int(jnp.argmin(jnp.abs(coverages - 0.5)))
    assert float(risks[coverage_idx]) == pytest.approx(0.0, abs=1e-6)


def test_selective_risk_at_full_coverage_equals_unconditional_mean_error() -> None:
    selective = _import_selective()
    confidences = jnp.array([0.1, 0.4, 0.6, 0.9])
    errors = jnp.array([1.0, 0.5, 0.25, 0.0])
    _, risks = selective.risk_coverage_curve(confidences=confidences, errors=errors)
    full = float(risks[-1])
    expected = float(jnp.mean(errors))
    assert full == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# AURC
# ---------------------------------------------------------------------------


def test_aurc_known_synthetic_example() -> None:
    """For confidences sorted descending matching error 0,0,1,1:

    Risk-coverage curve at thresholds covering the top-k samples:
        k=1 → cov=0.25, risk=0
        k=2 → cov=0.50, risk=0
        k=3 → cov=0.75, risk=1/3 ≈ 0.333
        k=4 → cov=1.00, risk=0.5

    AURC = mean over the curve of risk values = (0 + 0 + 1/3 + 0.5) / 4 = 5/24 ≈ 0.2083.
    """
    selective = _import_selective()
    confidences = jnp.array([0.9, 0.8, 0.4, 0.1])
    errors = jnp.array([0.0, 0.0, 1.0, 1.0])
    aurc = float(selective.area_under_risk_coverage(confidences=confidences, errors=errors))
    expected = (0.0 + 0.0 + 1.0 / 3.0 + 0.5) / 4.0
    assert aurc == pytest.approx(expected, abs=1e-6)


def test_aurc_lower_when_confidence_is_more_predictive_of_error() -> None:
    """A perfectly ordered (high conf → low error) calibration produces a
    lower AURC than a random ordering on the same errors."""
    selective = _import_selective()
    rng = np.random.default_rng(0)
    n = 64
    errors = jnp.asarray(rng.uniform(size=(n,)))
    # Confidence perfectly anti-correlated with error.
    aligned_conf = -errors
    misaligned_conf = jnp.asarray(rng.uniform(size=(n,)))
    aurc_aligned = float(
        selective.area_under_risk_coverage(confidences=aligned_conf, errors=errors)
    )
    aurc_misaligned = float(
        selective.area_under_risk_coverage(confidences=misaligned_conf, errors=errors)
    )
    assert aurc_aligned < aurc_misaligned


# ---------------------------------------------------------------------------
# Abstention decisions
# ---------------------------------------------------------------------------


def test_abstention_decision_returns_named_masks_and_metadata() -> None:
    selective = _import_selective()
    confidences = jnp.array([0.1, 0.4, 0.6, 0.9])
    decision = selective.abstention_decision(confidences=confidences, threshold=0.5)
    assert bool(jnp.array_equal(decision.accepted_mask, jnp.array([False, False, True, True])))
    assert bool(jnp.array_equal(decision.rejected_mask, jnp.array([True, True, False, False])))
    md = dict(decision.metadata)
    assert md["threshold"] == pytest.approx(0.5)
    assert int(md["num_accepted"]) == 2
    assert int(md["num_rejected"]) == 2


def test_abstention_decision_is_frozen_dataclass() -> None:
    selective = _import_selective()
    decision = selective.abstention_decision(confidences=jnp.array([0.6, 0.4]), threshold=0.5)
    with pytest.raises(dc.FrozenInstanceError):
        decision.threshold = 0.7  # type: ignore[misc]


def test_abstention_decision_threshold_stored_in_state() -> None:
    selective = _import_selective()
    decision = selective.abstention_decision(confidences=jnp.array([0.6, 0.4]), threshold=0.5)
    assert decision.threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Transform compatibility
# ---------------------------------------------------------------------------


def test_risk_coverage_curve_is_jit_compatible() -> None:
    selective = _import_selective()
    rng = np.random.default_rng(0)
    confidences = jnp.asarray(rng.uniform(size=(64,)))
    errors = jnp.asarray(rng.uniform(size=(64,)))
    jitted = jax.jit(lambda c, e: selective.risk_coverage_curve(confidences=c, errors=e))
    coverages, risks = jitted(confidences, errors)
    assert bool(jnp.all(coverages[:-1] <= coverages[1:]))
    assert bool(jnp.all(jnp.isfinite(risks)))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_selective_surface() -> None:
    selective = _import_selective()
    expected = {
        "risk_coverage_curve",
        "area_under_risk_coverage",
        "abstention_decision",
        "AbstentionDecision",
    }
    missing = expected - set(dir(selective))
    assert not missing, f"missing public selective symbols: {sorted(missing)}"
