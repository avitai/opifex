"""Scientific-domain UQ utilities: field metadata, field metrics, domain metrics."""

from __future__ import annotations

from opifex.uncertainty.scientific.domain_metrics import (
    boundary_condition_coverage,
    chemical_accuracy_coverage,
    DomainMetricSummary,
    feasibility_coverage,
    parameter_credible_interval_coverage,
    physics_residual_coverage,
    regret_interval_summary,
    sensor_reliability_summary,
    spectral_coverage,
    UNSUPPORTED_ACTIVE_LEARNING,
    UNSUPPORTED_LIKELIHOOD_FREE,
    UNSUPPORTED_PAC_BAYES,
)
from opifex.uncertainty.scientific.fields import (
    conservation_law_residual_summary,
    FieldMetadata,
    function_space_l2_coverage,
    residual_uncertainty_alignment,
    spatial_calibration_error,
)
from opifex.uncertainty.scientific.solutions import SolutionDistribution


__all__ = [
    "UNSUPPORTED_ACTIVE_LEARNING",
    "UNSUPPORTED_LIKELIHOOD_FREE",
    "UNSUPPORTED_PAC_BAYES",
    "DomainMetricSummary",
    "FieldMetadata",
    "SolutionDistribution",
    "boundary_condition_coverage",
    "chemical_accuracy_coverage",
    "conservation_law_residual_summary",
    "feasibility_coverage",
    "function_space_l2_coverage",
    "parameter_credible_interval_coverage",
    "physics_residual_coverage",
    "regret_interval_summary",
    "residual_uncertainty_alignment",
    "sensor_reliability_summary",
    "spatial_calibration_error",
    "spectral_coverage",
]
