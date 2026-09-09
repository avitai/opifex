"""Probabilistic forecast metrics: one-release wrappers over calibrax.

Every function here is ``calibrax.metrics.functional.forecasting``'s (``crps`` is
``calibrax.metrics.functional.regression.crps``) under opifex's keyword names.
Each call emits a ``DeprecationWarning``; the names are removed in 0.2.3. New
code calls calibrax directly:

    from calibrax.metrics.functional.forecasting import fair_crps, spread_skill_ratio
"""

from __future__ import annotations

from typing import Any

from calibrax.metrics.functional import forecasting as _forecasting, regression as _regression

from opifex._deprecated import warn_deprecated


_HOME = "calibrax.metrics.functional.forecasting"
_HERE = "opifex.uncertainty.forecasting_metrics"


def crps(*, predictions: Any, targets: Any) -> Any:
    """Empirical CRPS of an ensemble: ``calibrax.metrics.functional.regression.crps``."""
    warn_deprecated(f"{_HERE}.crps", "calibrax.metrics.functional.regression.crps")
    return _regression.crps(predictions, targets)


def fair_crps(*, predictions: Any, targets: Any) -> Any:
    """Fair CRPS (Ferro 2014): ``calibrax.metrics.functional.forecasting.fair_crps``."""
    warn_deprecated(f"{_HERE}.fair_crps", f"{_HOME}.fair_crps")
    return _forecasting.fair_crps(predictions, targets)


def energy_score(*, ensemble: Any, targets: Any) -> Any:
    """Energy score: ``calibrax.metrics.functional.forecasting.energy_score``."""
    warn_deprecated(f"{_HERE}.energy_score", f"{_HOME}.energy_score")
    return _forecasting.energy_score(ensemble, targets)


def rank_histogram(*, ensemble: Any, targets: Any) -> Any:
    """Rank histogram: ``calibrax.metrics.functional.forecasting.rank_histogram``."""
    warn_deprecated(f"{_HERE}.rank_histogram", f"{_HOME}.rank_histogram")
    return _forecasting.rank_histogram(ensemble, targets)


def spread_skill_ratio(*, ensemble: Any, targets: Any) -> Any:
    """Spread-skill ratio: ``calibrax.metrics.functional.forecasting.spread_skill_ratio``."""
    warn_deprecated(f"{_HERE}.spread_skill_ratio", f"{_HOME}.spread_skill_ratio")
    return _forecasting.spread_skill_ratio(ensemble, targets)


def pit_histogram(*, means: Any, variances: Any, targets: Any, num_bins: int = 10) -> Any:
    """PIT histogram: ``calibrax.metrics.functional.forecasting.pit_histogram``."""
    warn_deprecated(f"{_HERE}.pit_histogram", f"{_HOME}.pit_histogram")
    return _forecasting.pit_histogram(means, variances, targets, num_bins=num_bins)


def ranked_probability_score(*, probabilities: Any, targets: Any) -> Any:
    """RPS: ``calibrax.metrics.functional.forecasting.ranked_probability_score``."""
    warn_deprecated(f"{_HERE}.ranked_probability_score", f"{_HOME}.ranked_probability_score")
    return _forecasting.ranked_probability_score(probabilities, targets)


def event_reliability(
    *, predicted_event_probabilities: Any, event_indicators: Any, num_bins: int = 10
) -> Any:
    """Event reliability: ``calibrax.metrics.functional.forecasting.event_reliability``."""
    warn_deprecated(f"{_HERE}.event_reliability", f"{_HOME}.event_reliability")
    return _forecasting.event_reliability(
        predicted_event_probabilities, event_indicators, num_bins=num_bins
    )


def ensemble_ranked_probability_score(
    *, samples: Any, targets: Any, thresholds: Any, fair: bool = False
) -> Any:
    """Ensemble RPS: ``calibrax...forecasting.ensemble_ranked_probability_score``."""
    warn_deprecated(
        f"{_HERE}.ensemble_ranked_probability_score", f"{_HOME}.ensemble_ranked_probability_score"
    )
    return _forecasting.ensemble_ranked_probability_score(
        samples, targets, thresholds=thresholds, fair=fair
    )


def ranked_probability_skill_score(*, rps: Any, rps_reference: Any) -> Any:
    """RPSS: ``calibrax.metrics.functional.forecasting.ranked_probability_skill_score``."""
    warn_deprecated(
        f"{_HERE}.ranked_probability_skill_score", f"{_HOME}.ranked_probability_skill_score"
    )
    return _forecasting.ranked_probability_skill_score(rps, rps_reference)


__all__ = [
    "crps",
    "energy_score",
    "ensemble_ranked_probability_score",
    "event_reliability",
    "fair_crps",
    "pit_histogram",
    "rank_histogram",
    "ranked_probability_score",
    "ranked_probability_skill_score",
    "spread_skill_ratio",
]
