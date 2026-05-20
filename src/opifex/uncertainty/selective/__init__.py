"""Selective prediction utilities: risk-coverage curve, AURC, abstention."""

from __future__ import annotations

from opifex.uncertainty.selective.abstention import abstention_decision, AbstentionDecision
from opifex.uncertainty.selective.risk_coverage import (
    area_under_risk_coverage,
    risk_coverage_curve,
)


__all__ = [
    "AbstentionDecision",
    "abstention_decision",
    "area_under_risk_coverage",
    "risk_coverage_curve",
]
