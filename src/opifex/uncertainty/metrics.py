"""Ensemble and interval scoring metrics: one-release wrappers over calibrax.

Every function here is ``calibrax.metrics.functional.uncertainty``'s under
opifex's keyword names (``mutual_information`` is calibrax's
``ensemble_mutual_information``, ``chi2_confidence_intervals`` its
``chi2_confidence_interval``). Each call emits a ``DeprecationWarning``; the
names are removed in 0.2.3.
"""

from __future__ import annotations

from typing import Any

from calibrax.metrics.functional import uncertainty as _uncertainty

from opifex.uncertainty._deprecated_metric import warn_deprecated_metric


_HOME = "calibrax.metrics.functional.uncertainty"
_HERE = "opifex.uncertainty.metrics"


def predictive_entropy(*, ensemble_probabilities: Any) -> Any:
    """Entropy of the mean member probabilities: calibrax's ``predictive_entropy``."""
    warn_deprecated_metric(f"{_HERE}.predictive_entropy", f"{_HOME}.predictive_entropy")
    return _uncertainty.predictive_entropy(ensemble_probabilities)


def mutual_information(*, ensemble_probabilities: Any) -> Any:
    """BALD mutual information: calibrax's ``ensemble_mutual_information``."""
    warn_deprecated_metric(f"{_HERE}.mutual_information", f"{_HOME}.ensemble_mutual_information")
    return _uncertainty.ensemble_mutual_information(ensemble_probabilities)


def interval_score(*, lower: Any, upper: Any, targets: Any, alpha: float) -> Any:
    """Interval score: calibrax's ``interval_score``."""
    warn_deprecated_metric(f"{_HERE}.interval_score", f"{_HOME}.interval_score")
    return _uncertainty.interval_score(lower, upper, targets, alpha=alpha)


def winkler_score(*, lower: Any, upper: Any, targets: Any, alpha: float) -> Any:
    """Winkler score: calibrax's ``winkler_score``."""
    warn_deprecated_metric(f"{_HERE}.winkler_score", f"{_HOME}.winkler_score")
    return _uncertainty.winkler_score(lower, upper, targets, alpha=alpha)


def anees(*, predicted_means: Any, predicted_covariances: Any, references: Any) -> Any:
    """ANEES: calibrax's ``anees``."""
    warn_deprecated_metric(f"{_HERE}.anees", f"{_HOME}.anees")
    return _uncertainty.anees(predicted_means, predicted_covariances, references)


def non_credibility_index(
    *,
    predicted_means: Any,
    predicted_covariances: Any,
    references: Any,
    reference_covariances: Any,
) -> Any:
    """Non-credibility index: calibrax's ``non_credibility_index``."""
    warn_deprecated_metric(f"{_HERE}.non_credibility_index", f"{_HOME}.non_credibility_index")
    return _uncertainty.non_credibility_index(
        predicted_means, predicted_covariances, references, reference_covariances
    )


def chi2_confidence_intervals(*, dim: int, percentile: float) -> Any:
    """Chi-square interval: calibrax's ``chi2_confidence_interval``."""
    warn_deprecated_metric(
        f"{_HERE}.chi2_confidence_intervals", f"{_HOME}.chi2_confidence_interval"
    )
    return _uncertainty.chi2_confidence_interval(dim, percentile=percentile)


__all__ = [
    "anees",
    "chi2_confidence_intervals",
    "interval_score",
    "mutual_information",
    "non_credibility_index",
    "predictive_entropy",
    "winkler_score",
]
