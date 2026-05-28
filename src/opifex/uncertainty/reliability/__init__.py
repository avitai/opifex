"""Reliability-engineering utilities (Task 6.5).

Two primitives that operate on arrays — never on NNX modules — so they
compose cleanly with sampling utilities and pre-tabulated indicator
sequences:

* :func:`failure_probability` — empirical ``p_f`` + Wilson binomial
  confidence interval (Wilson 1927).
* :func:`reliability_index` — Cornell / Hasofer-Lind index
  ``beta = -Phi^{-1}(p_f)`` (Hasofer-Lind 1974).

Advanced FORM / SORM / subset-simulation adapters are out of scope for
this slice per the Task 6.5 implementation requirements.
"""

from opifex.uncertainty.reliability.failure_probability import (
    failure_probability,
    ReliabilityResult,
)
from opifex.uncertainty.reliability.reliability_index import reliability_index


__all__ = [
    "ReliabilityResult",
    "failure_probability",
    "reliability_index",
]
