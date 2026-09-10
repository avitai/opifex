"""Internal uncertainty code binds calibrax's metric functions directly.

The one-release wrappers 0.2.2 kept under ``opifex.uncertainty.forecasting_metrics``,
``opifex.uncertainty.metrics``, ``opifex.uncertainty.calibration`` and
``opifex.core.metrics`` were removed in 0.2.3; callers import the calibrax function.
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import pytest
from calibrax.metrics.functional import (
    calibration as cx_calibration,
    regression as cx_regression,
)

from opifex.uncertainty.aggregators import CalibrationAssessment
from opifex.uncertainty.aggregators.basic import UncertaintyQuantifier


_KEY = jax.random.key(7)
_BINARY = jnp.array([1.0, 0.0, 1.0, 0.0])
_CONFIDENCES = jnp.array([0.9, 0.6, 0.8, 0.3])


def test_removed_wrapper_modules_do_not_import() -> None:
    for module in (
        "opifex.uncertainty.forecasting_metrics",
        "opifex.uncertainty.metrics",
        "opifex.core.metrics",
        "opifex.uncertainty.calibration.base",
        "opifex.uncertainty.calibration.regression",
        "opifex._deprecated",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)


def test_internal_code_binds_calibrax_directly() -> None:
    trainer = importlib.import_module("opifex.core.training.trainer")

    assert trainer.relative_l2_error is cx_regression.relative_l2_error


def test_calibration_assessment_is_calibrax_ece_and_mce() -> None:
    assessment = CalibrationAssessment()
    ece = assessment.expected_calibration_error(_CONFIDENCES, _BINARY, n_bins=4)
    mce = assessment.maximum_calibration_error(_CONFIDENCES, _BINARY, n_bins=4)
    bins = assessment.reliability_diagram_data(_CONFIDENCES, _BINARY, n_bins=4)
    expected = cx_calibration.reliability_diagram_bins(_CONFIDENCES, _BINARY, num_bins=4)

    assert ece == pytest.approx(
        float(cx_calibration.expected_calibration_error(_CONFIDENCES, _BINARY, num_bins=4))
    )
    assert mce == pytest.approx(
        float(cx_calibration.maximum_calibration_error(_CONFIDENCES, _BINARY, num_bins=4))
    )
    assert jnp.allclose(bins["bin_accuracies"], expected["bin_accuracies"])
    assert jnp.allclose(bins["bin_confidences"], expected["bin_confidences"])
    assert jnp.allclose(bins["bin_counts"], expected["bin_counts"])
    assert jnp.allclose(
        bins["bin_centers"], (expected["bin_edges"][:-1] + expected["bin_edges"][1:]) / 2
    )
    assert not hasattr(
        importlib.import_module("opifex.uncertainty.aggregators.calibration"),
        "_bin_calibration_stats",
    )


def test_basic_quantifier_bins_are_calibrax_bins() -> None:
    ece, mce, bins = UncertaintyQuantifier._compute_calibration_bins(
        UncertaintyQuantifier.__new__(UncertaintyQuantifier), _CONFIDENCES, _BINARY, 4
    )
    expected = cx_calibration.reliability_diagram_bins(_CONFIDENCES, _BINARY, num_bins=4)

    assert ece == pytest.approx(
        float(cx_calibration.expected_calibration_error(_CONFIDENCES, _BINARY, num_bins=4))
    )
    assert mce == pytest.approx(
        float(cx_calibration.maximum_calibration_error(_CONFIDENCES, _BINARY, num_bins=4))
    )
    assert jnp.allclose(bins["bin_counts"], expected["bin_counts"])
    assert jnp.allclose(bins["bin_boundaries"], expected["bin_edges"])
