"""Uncertainty quantification utilities for Bayesian neural networks.

Submodules:

* :mod:`opifex.uncertainty.aggregators.types` — value-object containers
  (``UncertaintyComponents``, ``CalibrationMetrics``,
  ``UncertaintyIntegrationResults``, ``EnhancedUncertaintyComponents``).
* :mod:`opifex.uncertainty.aggregators.basic` — basic epistemic / aleatoric
  estimators and the :class:`UncertaintyQuantifier` integration interface.
* :mod:`opifex.uncertainty.aggregators.calibration` — reliability-binning
  helper and :class:`CalibrationAssessment` tooling.
* :mod:`opifex.uncertainty.aggregators.enhanced` — ensemble / distributional /
  multi-source quantifiers (:class:`EnhancedUncertaintyQuantifier`).
"""

from opifex.uncertainty.aggregators.basic import (
    AleatoricUncertainty,
    EpistemicUncertainty,
    UncertaintyQuantifier,
)

# Re-export the private bin-stats helper so callers importing it from the
# original ``opifex.uncertainty.aggregators`` path keep working.
from opifex.uncertainty.aggregators.calibration import (
    _bin_calibration_stats,
    CalibrationAssessment,
)
from opifex.uncertainty.aggregators.enhanced import (
    DistributionalAleatoricUncertainty,
    EnhancedUncertaintyQuantifier,
    EnsembleEpistemicUncertainty,
    MultiSourceUncertaintyAggregator,
)
from opifex.uncertainty.aggregators.types import (
    CalibrationMetrics,
    EnhancedUncertaintyComponents,
    UncertaintyComponents,
    UncertaintyIntegrationResults,
)


__all__ = [
    "AleatoricUncertainty",
    "CalibrationAssessment",
    "CalibrationMetrics",
    "DistributionalAleatoricUncertainty",
    "EnhancedUncertaintyComponents",
    "EnhancedUncertaintyQuantifier",
    "EnsembleEpistemicUncertainty",
    "EpistemicUncertainty",
    "MultiSourceUncertaintyAggregator",
    "UncertaintyComponents",
    "UncertaintyIntegrationResults",
    "UncertaintyQuantifier",
    "_bin_calibration_stats",
]
