"""Uncertainty-quantification value-object containers.

Dataclass result types returned by the basic and enhanced uncertainty
quantifiers, plus calibration-assessment metrics.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from jaxtyping import Array, Float  # noqa: TC002


if TYPE_CHECKING:
    batch = None  # type-var placeholder for jaxtyping array dimensions


@dataclasses.dataclass
class UncertaintyComponents:
    """Decomposed uncertainty components."""

    epistemic: Float[Array, ...]  # Model uncertainty
    aleatoric: Float[Array, ...]  # Data uncertainty
    total: Float[Array, ...]  # Combined uncertainty


@dataclasses.dataclass
class CalibrationMetrics:
    """Uncertainty calibration assessment metrics."""

    expected_calibration_error: float
    maximum_calibration_error: float
    reliability_diagram: dict[str, Array]
    confidence_histogram: Array
    accuracy_histogram: Array


@dataclasses.dataclass
class UncertaintyIntegrationResults:
    """Results from uncertainty propagation through model pipeline."""

    predictions: Float[Array, "batch output"]
    uncertainty_components: UncertaintyComponents
    calibration_metrics: CalibrationMetrics
    confidence_intervals: tuple[Float[Array, "batch output"], Float[Array, "batch output"]]
    prediction_intervals: tuple[Float[Array, "batch output"], Float[Array, "batch output"]]


@dataclasses.dataclass
class EnhancedUncertaintyComponents:
    """Enhanced uncertainty components with multiple sources."""

    epistemic_ensemble: Float[Array, "batch output"]  # Ensemble-based epistemic uncertainty
    aleatoric_distributional: Float[Array, "batch output"]  # Distributional aleatoric uncertainty
    total_uncertainty: Float[Array, "batch output"]  # Combined uncertainty
    uncertainty_breakdown: dict[str, Float[Array, "batch output"]]  # Detailed breakdown
    epistemic_dropout: Float[Array, "batch output"] | None = (
        None  # Dropout-based epistemic uncertainty
    )
