"""Calibration assessment over calibrax's reliability binning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from calibrax.metrics.functional.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram_bins,
)

from opifex.uncertainty.aggregators.types import CalibrationMetrics


if TYPE_CHECKING:
    import jax


class CalibrationAssessment:
    """Enhanced uncertainty calibration assessment tools.

    Every score is calibrax's: ``expected_calibration_error``,
    ``maximum_calibration_error`` and ``reliability_diagram_bins`` over the same
    equal-width confidence bins (the last bin closed on the right, empty bins
    reported with zero statistics).
    """

    @staticmethod
    def expected_calibration_error(
        confidences: jax.Array, accuracies: jax.Array, n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        return float(expected_calibration_error(confidences, accuracies, num_bins=n_bins))

    @staticmethod
    def maximum_calibration_error(
        confidences: jax.Array, accuracies: jax.Array, n_bins: int = 10
    ) -> float:
        """Compute Maximum Calibration Error (MCE) over the non-empty bins."""
        return float(maximum_calibration_error(confidences, accuracies, num_bins=n_bins))

    @staticmethod
    def reliability_diagram_data(
        confidences: jax.Array, accuracies: jax.Array, n_bins: int = 10
    ) -> dict[str, jax.Array]:
        """Compute reliability diagram data for visualization."""
        bins = reliability_diagram_bins(confidences, accuracies, num_bins=n_bins)
        edges = bins["bin_edges"]
        return {
            "bin_centers": (edges[:-1] + edges[1:]) / 2,
            "bin_accuracies": bins["bin_accuracies"],
            "bin_confidences": bins["bin_confidences"],
            "bin_counts": bins["bin_counts"],
        }

    def assess_calibration(
        self, confidences: jax.Array, accuracies: jax.Array, n_bins: int = 10
    ) -> CalibrationMetrics:
        """Assess overall calibration with multiple metrics."""
        ece = self.expected_calibration_error(confidences, accuracies, n_bins)
        mce = self.maximum_calibration_error(confidences, accuracies, n_bins)
        rel_data = self.reliability_diagram_data(confidences, accuracies, n_bins)

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            reliability_diagram=rel_data,
            confidence_histogram=rel_data["bin_confidences"],
            accuracy_histogram=rel_data["bin_accuracies"],
        )


__all__ = ["CalibrationAssessment"]
