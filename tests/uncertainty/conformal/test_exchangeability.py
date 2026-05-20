"""Exchangeability diagnostic contract.

The diagnostic is a two-sample Kolmogorov–Smirnov test on calibration vs
evaluation residual scores; under exchangeability the p-value distribution
is uniform, so

* Shuffled i.i.d. residuals should pass (p > alpha).
* Deliberately distribution-shifted residuals should fail (p < alpha).

See Vovk et al. 2005 ("Algorithmic Learning in a Random World", §2.4) for
the formal exchangeability assumption that the diagnostic guards.
"""

from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _import_exchangeability():
    from opifex.uncertainty.conformal import exchangeability

    return exchangeability


def test_exchangeability_diagnostic_passes_on_shuffled_iid_scores() -> None:
    exch = _import_exchangeability()
    rng = np.random.default_rng(0)
    scores = jnp.asarray(rng.standard_normal(size=(1024,)))
    permuted_indices = rng.permutation(1024)
    half = 512
    calibration = scores[permuted_indices[:half]]
    evaluation = scores[permuted_indices[half:]]
    report = exch.check_exchangeability(
        calibration_scores=calibration, evaluation_scores=evaluation, alpha=0.05
    )
    assert report.passes
    assert float(report.p_value) > 0.05


def test_exchangeability_diagnostic_fails_on_shifted_distribution() -> None:
    exch = _import_exchangeability()
    rng = np.random.default_rng(1)
    calibration = jnp.asarray(rng.standard_normal(size=(512,)))
    evaluation = jnp.asarray(2.0 + rng.standard_normal(size=(512,)))  # location-shifted
    report = exch.check_exchangeability(
        calibration_scores=calibration, evaluation_scores=evaluation, alpha=0.05
    )
    assert not report.passes
    assert float(report.p_value) < 0.05


def test_exchangeability_report_is_frozen_dataclass() -> None:
    exch = _import_exchangeability()
    report = exch.ExchangeabilityReport(p_value=jnp.asarray(0.5), passes=True)
    assert float(report.p_value) == pytest.approx(0.5)
    assert report.passes is True
    with pytest.raises(dc.FrozenInstanceError):
        report.passes = False  # type: ignore[misc]


def test_exchangeability_records_alpha_and_method_metadata() -> None:
    exch = _import_exchangeability()
    rng = np.random.default_rng(2)
    cal = jnp.asarray(rng.standard_normal(size=(256,)))
    evl = jnp.asarray(rng.standard_normal(size=(256,)))
    report = exch.check_exchangeability(calibration_scores=cal, evaluation_scores=evl, alpha=0.05)
    md = dict(report.metadata)
    assert md["method"] == "ks_two_sample"
    assert md["alpha"] == pytest.approx(0.05)
    assert int(md["calibration_size"]) == 256
    assert int(md["evaluation_size"]) == 256


def test_exchangeability_check_is_vmap_compatible() -> None:
    """Diagnostic should trace under jax.vmap over an extra trial axis."""
    exch = _import_exchangeability()
    rng = np.random.default_rng(3)
    cal = jnp.asarray(rng.standard_normal(size=(4, 256)))
    evl = jnp.asarray(rng.standard_normal(size=(4, 256)))
    p_values = jax.vmap(
        lambda c, e: exch.ks_two_sample_pvalue(calibration_scores=c, evaluation_scores=e)
    )(cal, evl)
    assert p_values.shape == (4,)
    assert bool(jnp.all((p_values >= 0.0) & (p_values <= 1.0)))
