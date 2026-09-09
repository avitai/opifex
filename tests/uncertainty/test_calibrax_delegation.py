"""The uncertainty metrics are calibrax's; opifex's names are one-release deprecated wrappers.

Every public metric function under ``opifex.uncertainty.forecasting_metrics``,
``opifex.uncertainty.metrics``, ``opifex.uncertainty.calibration`` and
``opifex.core.metrics`` returns exactly what the calibrax function returns on
the same inputs, emits a ``DeprecationWarning`` naming its calibrax home, and is
removed in 0.2.3. Internal code binds calibrax directly.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from calibrax.metrics.functional import (
    calibration as cx_calibration,
    forecasting as cx_forecasting,
    regression as cx_regression,
    uncertainty as cx_uncertainty,
)

from opifex.core import metrics as core_metrics
from opifex.uncertainty import calibration, forecasting_metrics, metrics
from opifex.uncertainty.aggregators import CalibrationAssessment
from opifex.uncertainty.aggregators.basic import UncertaintyQuantifier


_KEY = jax.random.key(7)
_ENSEMBLE = jax.random.normal(_KEY, (6, 5))  # (samples, members)
_TARGETS = jnp.linspace(-0.5, 0.5, 6)
_MEANS = jnp.linspace(-1.0, 1.0, 6)
_VARIANCES = jnp.full((6,), 0.4)
_LOWER = _MEANS - 1.0
_UPPER = _MEANS + 1.0
_PROBS = jnp.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8], [0.4, 0.4, 0.2]])
_ENSEMBLE_PROBS = jnp.stack([_PROBS, _PROBS[::-1]])  # (members, samples, classes)
_LABELS = jnp.array([0, 1, 2, 0])
_BINARY = jnp.array([1.0, 0.0, 1.0, 0.0])
_CONFIDENCES = jnp.array([0.9, 0.6, 0.8, 0.3])
_COVS = jnp.stack([jnp.eye(2) * s for s in (0.5, 1.0, 2.0)])
_MEANS_2D = jnp.zeros((3, 2))
_REFS_2D = jnp.array([[0.1, -0.2], [0.3, 0.3], [-0.5, 0.4]])
_FIELDS = jax.random.normal(jax.random.fold_in(_KEY, 1), (4, 8, 8))
_FIELDS_HAT = _FIELDS + 0.1 * jax.random.normal(jax.random.fold_in(_KEY, 2), (4, 8, 8))

Case = tuple[Any, str, dict[str, Any], Callable[[], Any]]

CASES: list[Case] = [
    (
        forecasting_metrics,
        "crps",
        {"predictions": _ENSEMBLE, "targets": _TARGETS},
        lambda: cx_regression.crps(_ENSEMBLE, _TARGETS),
    ),
    (
        forecasting_metrics,
        "fair_crps",
        {"predictions": _ENSEMBLE, "targets": _TARGETS},
        lambda: cx_forecasting.fair_crps(_ENSEMBLE, _TARGETS),
    ),
    (
        forecasting_metrics,
        "energy_score",
        {"ensemble": _ENSEMBLE.T[None], "targets": _TARGETS[None]},
        lambda: cx_forecasting.energy_score(_ENSEMBLE.T[None], _TARGETS[None]),
    ),
    (
        forecasting_metrics,
        "rank_histogram",
        {"ensemble": _ENSEMBLE, "targets": _TARGETS},
        lambda: cx_forecasting.rank_histogram(_ENSEMBLE, _TARGETS),
    ),
    (
        forecasting_metrics,
        "spread_skill_ratio",
        {"ensemble": _ENSEMBLE, "targets": _TARGETS},
        lambda: cx_forecasting.spread_skill_ratio(_ENSEMBLE, _TARGETS),
    ),
    (
        forecasting_metrics,
        "pit_histogram",
        {"means": _MEANS, "variances": _VARIANCES, "targets": _TARGETS, "num_bins": 4},
        lambda: cx_forecasting.pit_histogram(_MEANS, _VARIANCES, _TARGETS, num_bins=4),
    ),
    (
        forecasting_metrics,
        "ranked_probability_score",
        {"probabilities": _PROBS, "targets": _LABELS},
        lambda: cx_forecasting.ranked_probability_score(_PROBS, _LABELS),
    ),
    (
        forecasting_metrics,
        "event_reliability",
        {"predicted_event_probabilities": _CONFIDENCES, "event_indicators": _BINARY, "num_bins": 2},
        lambda: cx_forecasting.event_reliability(_CONFIDENCES, _BINARY, num_bins=2),
    ),
    (
        forecasting_metrics,
        "ensemble_ranked_probability_score",
        {
            "samples": _ENSEMBLE,
            "targets": _TARGETS,
            "thresholds": jnp.array([-0.2, 0.2]),
            "fair": True,
        },
        lambda: cx_forecasting.ensemble_ranked_probability_score(
            _ENSEMBLE, _TARGETS, thresholds=jnp.array([-0.2, 0.2]), fair=True
        ),
    ),
    (
        forecasting_metrics,
        "ranked_probability_skill_score",
        {"rps": 0.2, "rps_reference": 0.5},
        lambda: cx_forecasting.ranked_probability_skill_score(0.2, 0.5),
    ),
    (
        metrics,
        "predictive_entropy",
        {"ensemble_probabilities": _ENSEMBLE_PROBS},
        lambda: cx_uncertainty.predictive_entropy(_ENSEMBLE_PROBS),
    ),
    (
        metrics,
        "mutual_information",
        {"ensemble_probabilities": _ENSEMBLE_PROBS},
        lambda: cx_uncertainty.ensemble_mutual_information(_ENSEMBLE_PROBS),
    ),
    (
        metrics,
        "interval_score",
        {"lower": _LOWER, "upper": _UPPER, "targets": _TARGETS, "alpha": 0.1},
        lambda: cx_uncertainty.interval_score(_LOWER, _UPPER, _TARGETS, alpha=0.1),
    ),
    (
        metrics,
        "winkler_score",
        {"lower": _LOWER, "upper": _UPPER, "targets": _TARGETS, "alpha": 0.1},
        lambda: cx_uncertainty.winkler_score(_LOWER, _UPPER, _TARGETS, alpha=0.1),
    ),
    (
        metrics,
        "anees",
        {"predicted_means": _MEANS_2D, "predicted_covariances": _COVS, "references": _REFS_2D},
        lambda: cx_uncertainty.anees(_MEANS_2D, _COVS, _REFS_2D),
    ),
    (
        metrics,
        "non_credibility_index",
        {
            "predicted_means": _MEANS_2D,
            "predicted_covariances": _COVS,
            "references": _REFS_2D,
            "reference_covariances": _COVS * 1.5,
        },
        lambda: cx_uncertainty.non_credibility_index(_MEANS_2D, _COVS, _REFS_2D, _COVS * 1.5),
    ),
    (
        metrics,
        "chi2_confidence_intervals",
        {"dim": 2, "percentile": 0.95},
        lambda: cx_uncertainty.chi2_confidence_interval(2, percentile=0.95),
    ),
    (
        calibration,
        "gaussian_nll",
        {"mean": _MEANS, "variance": _VARIANCES, "target": _TARGETS},
        lambda: cx_uncertainty.gaussian_nll(_MEANS, _VARIANCES, _TARGETS),
    ),
    (
        calibration,
        "brier_score",
        {"probabilities": _CONFIDENCES, "targets": _BINARY},
        lambda: cx_calibration.brier_score(_CONFIDENCES, _BINARY),
    ),
    (
        calibration,
        "expected_calibration_error",
        {"probabilities": _CONFIDENCES, "targets": _BINARY, "num_bins": 4},
        lambda: cx_calibration.expected_calibration_error(_CONFIDENCES, _BINARY, num_bins=4),
    ),
    (
        calibration,
        "pinball_loss",
        {"predictions": _MEANS, "targets": _TARGETS, "quantile": 0.3},
        lambda: cx_regression.quantile_loss(_MEANS, _TARGETS, quantile=0.3),
    ),
    (
        calibration,
        "picp",
        {"lower": _LOWER, "upper": _UPPER, "target": _TARGETS},
        lambda: cx_uncertainty.picp(_LOWER, _UPPER, _TARGETS),
    ),
    (
        calibration,
        "mpiw",
        {"lower": _LOWER, "upper": _UPPER},
        lambda: cx_uncertainty.mpiw(_LOWER, _UPPER),
    ),
    (
        calibration,
        "regression_calibration_error",
        {
            "mean": _MEANS,
            "variance": _VARIANCES,
            "target": _TARGETS,
            "quantile_levels": jnp.array([0.25, 0.5, 0.75]),
        },
        lambda: cx_uncertainty.regression_calibration_error(
            _MEANS, _VARIANCES, _TARGETS, quantile_levels=jnp.array([0.25, 0.5, 0.75])
        ),
    ),
    (
        core_metrics,
        "per_sample_relative_l2",
        {"prediction": _FIELDS_HAT, "target": _FIELDS},
        lambda: cx_regression.per_sample_relative_l2(_FIELDS_HAT, _FIELDS),
    ),
    (
        core_metrics,
        "relative_l2_error",
        {"prediction": _FIELDS_HAT, "target": _FIELDS},
        lambda: cx_regression.relative_l2_error(_FIELDS_HAT, _FIELDS),
    ),
]


@pytest.mark.parametrize(("module", "name", "kwargs", "expected"), CASES, ids=[c[1] for c in CASES])
def test_wrapper_returns_calibrax_result_and_warns(
    module: Any, name: str, kwargs: dict[str, Any], expected: Callable[[], Any]
) -> None:
    with pytest.warns(DeprecationWarning, match=rf"{name}.*calibrax"):
        result = getattr(module, name)(**kwargs)

    assert jnp.allclose(jnp.asarray(result), jnp.asarray(expected()), atol=1e-6, equal_nan=True)


def test_every_wrapped_name_has_a_case() -> None:
    """The public surfaces are exactly the wrapped names (they leave in 0.2.3)."""
    wrapped = {(m.__name__, n) for m, n, _, _ in CASES}

    assert {n for m, n in wrapped if m.endswith("forecasting_metrics")} == set(
        forecasting_metrics.__all__
    )
    assert {n for m, n in wrapped if m.endswith(".calibration")} == set(calibration.__all__) - {
        "TemperatureScaling",
        "TemperatureScalingState",
        "nll_loss_at_temperature",
    }
    assert {n for m, n in wrapped if m.endswith("uncertainty.metrics")} == set(metrics.__all__)
    assert {n for m, n in wrapped if m.endswith("core.metrics")} == set(core_metrics.__all__)


def test_validation_flags_still_guard_inputs() -> None:
    """The one-release wrappers keep the explicit validation their callers relied on."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="variance"):
            calibration.gaussian_nll(
                mean=_MEANS, variance=-_VARIANCES, target=_TARGETS, validate=True
            )
        with pytest.raises(ValueError, match="upper < lower"):
            calibration.picp(lower=_UPPER, upper=_LOWER, target=_TARGETS, validate=True)


def test_fluctifex_call_shapes_keep_working() -> None:
    """fluctifex imports fair_crps and spread_skill_ratio with these keywords."""
    with pytest.warns(DeprecationWarning, match="fair_crps"):
        crps_value = forecasting_metrics.fair_crps(predictions=_ENSEMBLE, targets=_TARGETS)
    with pytest.warns(DeprecationWarning, match="spread_skill_ratio"):
        ratio = forecasting_metrics.spread_skill_ratio(ensemble=_ENSEMBLE, targets=_TARGETS)

    assert jnp.isfinite(crps_value)
    assert jnp.isfinite(ratio)


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
