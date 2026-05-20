"""Scientific-domain UQ utilities: field metadata + field-level metrics."""

from __future__ import annotations

from opifex.uncertainty.scientific.fields import (
    conservation_law_residual_summary,
    FieldMetadata,
    function_space_l2_coverage,
    residual_uncertainty_alignment,
    spatial_calibration_error,
)


__all__ = [
    "FieldMetadata",
    "conservation_law_residual_summary",
    "function_space_l2_coverage",
    "residual_uncertainty_alignment",
    "spatial_calibration_error",
]
