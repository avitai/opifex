"""Probabilistic / ensemble / event reliability forecasting metrics.

Module surface:

* :mod:`opifex.uncertainty.forecasting_metrics.crps` — empirical and fair CRPS.
* :mod:`opifex.uncertainty.forecasting_metrics.ensemble` — energy score.
* :mod:`opifex.uncertainty.forecasting_metrics.rank_histogram` — rank histogram.
* :mod:`opifex.uncertainty.forecasting_metrics.spread_skill` — spread / skill ratio.
* :mod:`opifex.uncertainty.forecasting_metrics.reliability` — PIT histogram,
  ranked probability score, event reliability.
"""

from __future__ import annotations

from opifex.uncertainty.forecasting_metrics.crps import crps, fair_crps
from opifex.uncertainty.forecasting_metrics.ensemble import energy_score
from opifex.uncertainty.forecasting_metrics.rank_histogram import rank_histogram
from opifex.uncertainty.forecasting_metrics.reliability import (
    ensemble_ranked_probability_score,
    event_reliability,
    pit_histogram,
    ranked_probability_score,
    ranked_probability_skill_score,
)
from opifex.uncertainty.forecasting_metrics.spread_skill import spread_skill_ratio


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
