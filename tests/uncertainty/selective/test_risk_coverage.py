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


def test_abstention_decision_is_jit_and_vmap_compatible() -> None:
    """``abstention_decision`` traces once under ``jit`` and composes under ``vmap``."""
    from substrax.testing import TraceCounter

    selective = _import_selective()
    counter = TraceCounter()
    jitted = jax.jit(
        counter.wrap(lambda c: selective.abstention_decision(confidences=c, threshold=0.5))
    )
    c1 = jnp.array([0.1, 0.5, 0.8, 0.3], dtype=jnp.float32)
    with counter.expect(new_traces=1):
        d1 = jitted(c1)
    with counter.expect(new_traces=0):
        d2 = jitted(c1 + 0.1)
    assert bool(jnp.array_equal(d1.accepted_mask, jnp.array([False, True, True, False])))
    assert bool(jnp.array_equal(d1.rejected_mask, jnp.array([True, False, False, True])))
    assert d2.accepted_mask.shape == (4,)

    batch_c = jnp.stack([c1, 1.0 - c1])
    batched = jax.vmap(lambda c: selective.abstention_decision(confidences=c, threshold=0.5))(
        batch_c
    )
    assert batched.accepted_mask.shape == (2, 4)
    assert batched.rejected_mask.shape == (2, 4)

    # vmap over traced threshold, eager == jit PyTree structure, zero downstream retraces, 2-D counts
    thresholds = jnp.array([0.2, 0.5, 0.85], dtype=jnp.float32)
    by_thresh = jax.vmap(lambda t: selective.abstention_decision(confidences=c1, threshold=t))(
        thresholds
    )
    assert by_thresh.accepted_mask.shape == (3, 4)
    assert jnp.allclose(by_thresh.threshold, thresholds)

    d_eager = selective.abstention_decision(confidences=c1, threshold=jnp.float32(0.5))
    d_jit = jax.jit(lambda c, t: selective.abstention_decision(confidences=c, threshold=t))(
        c1, jnp.float32(0.5)
    )
    assert jax.tree.structure(d_eager) == jax.tree.structure(d_jit)
    assert all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(d_eager), jax.tree.leaves(d_jit), strict=True)
    )

    consumer_counter = TraceCounter()
    consume = jax.jit(
        consumer_counter.wrap(lambda d: jnp.sum(d.accepted_mask.astype(jnp.float32)) + d.threshold)
    )
    d_all = selective.abstention_decision(confidences=jnp.ones(4, dtype=jnp.float32), threshold=0.5)
    with consumer_counter.expect(new_traces=1):
        _ = consume(d_eager)
    with consumer_counter.expect(new_traces=0):
        _ = consume(d_all)

    d_2d = selective.abstention_decision(
        confidences=jnp.ones((2, 2), dtype=jnp.float32), threshold=0.5
    )
    md_2d = dict(d_2d.metadata)
    assert int(md_2d["num_accepted"]) == 4
    assert int(md_2d["num_rejected"]) == 0


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
        "excess_aurc",
        "abstention_decision",
        "AbstentionDecision",
    }
    missing = expected - set(dir(selective))
    assert not missing, f"missing public selective symbols: {sorted(missing)}"


def test_risk_coverage_curve_tie_invariance_and_excess_aurc_contracts() -> None:
    """Tied confidences are permutation-invariant, and ``excess_aurc`` equals ``0.0``
    at oracle rankings and composes under ``jit``, ``vmap``, ``grad``, and ``float64``."""
    from substrax.testing import restored_jax_config, TraceCounter

    selective = _import_selective()
    # Constant confidences -> every point on RC curve equals exact mean(errors)
    errors = jnp.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    const_conf = jnp.full_like(errors, 0.5)
    _, const_risks = selective.risk_coverage_curve(confidences=const_conf, errors=errors)
    assert jnp.allclose(const_risks, jnp.full_like(const_risks, jnp.mean(errors)), atol=1e-6)

    # Partial ties: permutation invariance across arbitrary sample shuffles
    tied_conf = jnp.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.1], dtype=jnp.float32)
    ref_cov, ref_risks = selective.risk_coverage_curve(confidences=tied_conf, errors=errors)
    rng = np.random.default_rng(7)
    for _ in range(10):
        perm = rng.permutation(tied_conf.shape[0])
        p_cov, p_risks = selective.risk_coverage_curve(
            confidences=tied_conf[perm], errors=errors[perm]
        )
        assert jnp.allclose(p_cov, ref_cov, atol=1e-6)
        assert jnp.allclose(p_risks, ref_risks, atol=1e-6)

    # Oracle ranking on binary errors gives excess_aurc == 0.0; reversed ranking > 0.0
    oracle_conf = jnp.array([0.95, 0.2, 0.85, 0.1, 0.3, 0.8], dtype=jnp.float32)
    e_oracle = float(selective.excess_aurc(confidences=oracle_conf, errors=errors))
    e_bad = float(selective.excess_aurc(confidences=-oracle_conf, errors=errors))
    assert e_oracle == pytest.approx(0.0, abs=1e-7)
    assert e_bad > 0.2

    # Single-trace jit + vmap + grad + float64
    counter = TraceCounter()
    jitted_eaurc = jax.jit(
        counter.wrap(lambda c, e: selective.excess_aurc(confidences=c, errors=e))
    )
    with counter.expect(new_traces=1):
        val1 = jitted_eaurc(tied_conf, errors)
    with counter.expect(new_traces=0):
        _ = jitted_eaurc(oracle_conf, errors)
    assert float(val1) >= 0.0

    batched = jax.vmap(lambda c: selective.excess_aurc(confidences=c, errors=errors))(
        jnp.stack([oracle_conf, -oracle_conf])
    )
    assert batched.shape == (2,)
    assert float(batched[0]) == pytest.approx(0.0, abs=1e-7)
    assert float(batched[1]) > 0.2

    grad_err = jax.grad(lambda e: selective.excess_aurc(confidences=tied_conf, errors=e))(
        jnp.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.3], dtype=jnp.float32)
    )
    assert bool(jnp.all(jnp.isfinite(grad_err)))

    with restored_jax_config():
        jax.config.update("jax_enable_x64", True)
        e64 = selective.excess_aurc(
            confidences=oracle_conf.astype(jnp.float64),
            errors=errors.astype(jnp.float64),
        )
        assert e64.dtype == jnp.float64
        assert float(e64) == pytest.approx(0.0, abs=1e-15)

    with pytest.raises(ValueError, match=r"matching shape"):
        selective.risk_coverage_curve(confidences=jnp.zeros((3,)), errors=jnp.zeros((4,)))
