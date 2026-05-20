"""Marker test file — conformal prediction tests live under ``tests/uncertainty/conformal``.

The legacy ``ConformalPrediction`` / ``ConformalPredictor`` classes have
been removed in favour of the shared subsystem in
:mod:`opifex.uncertainty.conformal`. The audit-mandated coverage now
lives under ``tests/uncertainty/conformal/`` — specifically:

* ``test_regression.py`` — ``SplitConformalRegressor``,
  ``ConformalizedQuantileRegressor``, ``GroupedSplitConformalRegressor``.
* ``test_classification.py`` — LAC / APS / RAPS scores and
  ``LACConformalClassifier``.
* ``test_advanced.py`` — CV+ / jackknife+ / weighted variants.
* ``test_time_series.py`` — EnbPI / ACI online conformal.
* ``test_fields.py`` — function-space conformal regression.
* ``test_exchangeability.py`` — exchangeability diagnostic.

This file is intentionally minimal: it documents the migration so pytest
discovery does not silently lose the historical entry-point. Pin one
sanity assertion to keep the module non-empty.
"""

from __future__ import annotations


def test_conformal_prediction_moved_to_shared_subsystem() -> None:
    """Importing ``ConformalPrediction`` from ``opifex.neural.bayesian``
    must fail — the symbol was deliberately removed."""
    import opifex.neural.bayesian as bayesian_module

    assert not hasattr(bayesian_module, "ConformalPrediction")
    assert not hasattr(bayesian_module, "ConformalPredictor")
    assert not hasattr(bayesian_module, "ConformalConfig")


def test_shared_conformal_subsystem_is_importable() -> None:
    """The canonical replacement surface lives under
    :mod:`opifex.uncertainty.conformal`."""
    from opifex.uncertainty.conformal import (
        ConformalizedQuantileRegressor,
        SplitConformalRegressor,
    )

    assert callable(SplitConformalRegressor)
    assert callable(ConformalizedQuantileRegressor)
